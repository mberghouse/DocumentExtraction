import pytest
import pytest_asyncio
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from db.entity_matcher import RAGEntityMatcher
import os
import logging
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import json
from db.entity_matcher import EntityMatcher
from db.db_handler import DatabaseHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Create test_results directory
os.makedirs("test_results", exist_ok=True)

# Test cases with edge cases
TEST_CASES = [
    # Original cases
    {
        "current_name": "Meta",
        "old_name": "Facebook",
        "abbreviations": ["FB", "META"],
        "info": {
            "name": "Facebook",
            "address": "1 Hacker Way, Menlo Park, CA"
        }
    },
    {
        "current_name": "Alphabet",
        "old_name": "Google",
        "abbreviations": ["GOOG", "GOOGL"],
        "info": {
            "name": "Google",
            "address": "1600 Amphitheatre Parkway, Mountain View, CA"
        }
    },
    {
        "current_name": "Block",
        "old_name": "Square",
        "abbreviations": ["SQ", "BLK"],
        "info": {
            "name": "Square",
            "address": "1455 Market St #600, San Francisco, CA"
        }
    },
    # Edge cases
    {
        "current_name": "International Business Machines",
        "old_name": "Computing-Tabulating-Recording Company",
        "abbreviations": ["IBM", "CTR", "Big Blue"],
        "info": {
            "name": "IBM",
            "address": "1 New Orchard Road, Armonk, NY"
        },
        "subsidiaries": ["Red Hat", "Kyndryl"]
    },
    {
        "current_name": "Microsoft Corporation",
        "old_name": "Micro-Soft",
        "abbreviations": ["MSFT", "MS"],
        "info": {
            "name": "Microsoft Corp.",  # Different formatting
            "address": "One Microsoft Way, Redmond, WA"
        },
        "subsidiaries": ["LinkedIn", "GitHub"]
    },
    {
        "current_name": "FedEx Corporation",
        "old_name": "Federal Express",
        "abbreviations": ["FDX", "FedEx"],
        "info": {
            "name": "FEDEX CORP",  # All caps variation
            "address": "942 South Shady Grove Road, Memphis, TN"
        }
    }
]

class TestMetrics:
    """Class to track test metrics and generate visualizations"""
    def __init__(self):
        self.matches = []
        self.similarity_scores = []
        self.match_types = []
        self.entity_connections = nx.Graph()
        self.test_timings = []

    def add_match(self, test_name: str, query: str, matched_entity: str, similarity: float, match_type: str, duration: float):
        self.matches.append({
            "test_name": test_name,
            "query": query,
            "matched_entity": matched_entity,
            "similarity": similarity,
            "match_type": match_type,
            "duration": duration
        })
        self.similarity_scores.append(similarity)
        self.match_types.append(match_type)
        self.test_timings.append(duration)

    def add_connection(self, entity1: str, entity2: str, connection_type: str):
        self.entity_connections.add_edge(entity1, entity2, type=connection_type)

    def generate_visualizations(self):
        """Generate all visualizations"""
        # Create directory
        os.makedirs("test_results", exist_ok=True)
        
        # Simple plots with immediate saves
        plt.figure()
        sns.histplot(self.similarity_scores, bins=20)
        plt.title("Distribution of Similarity Scores")
        plt.savefig("test_results/similarity_distribution.png")
        plt.close()

        plt.figure()
        match_type_counts = pd.Series(self.match_types).value_counts()
        sns.barplot(x=match_type_counts.index, y=match_type_counts.values)
        plt.title("Distribution of Match Types")
        plt.xticks(rotation=45)
        plt.savefig("test_results/match_types.png", bbox_inches='tight')
        plt.close()

        plt.figure()
        pos = nx.spring_layout(self.entity_connections)
        nx.draw(self.entity_connections, pos, with_labels=True, 
               node_color='lightblue', node_size=2000, 
               font_size=8, font_weight='bold')
        plt.savefig("test_results/entity_network.png", bbox_inches='tight')
        plt.close()

        plt.figure()
        df = pd.DataFrame(self.matches)
        sns.boxplot(x='test_name', y='duration', data=df)
        plt.title("Test Execution Times")
        plt.xticks(rotation=45)
        plt.savefig("test_results/test_performance.png", bbox_inches='tight')
        plt.close()

        # Save metrics
        with open("test_results/test_results.json", "w") as f:
            json.dump(self.matches, f, indent=2)

@pytest.fixture(scope="session")
def session_metrics(request):
    """Create test metrics instance for the entire session"""
    metrics = TestMetrics()
    request.session.metrics = metrics
    return metrics

@pytest_asyncio.fixture
async def metrics(session_metrics):
    """Get metrics instance for each test"""
    return session_metrics

@pytest_asyncio.fixture
async def db():
    """Create a test database and return it"""
    logger.info("[test_entity_matcher.py:db] Setting up test database")
    db = DatabaseHandler(storage_type="json", base_path="test_data")
    
    # Insert test entities with subsidiary information
    for case in TEST_CASES:
        logger.info(f"[test_entity_matcher.py:db] Inserting test entity: {case['current_name']}")
        entity = {
            "name": case["current_name"],
            "entity_type": "company",
            "aliases": [case["old_name"]] + case.get("subsidiaries", []),  # Include subsidiaries as aliases
            "abbreviations": case.get("abbreviations", []),
            "address": case["info"]["address"],
            "subsidiaries": case.get("subsidiaries", []),
            "relationships": [
                {
                    "type": "subsidiary",
                    "entity": subsidiary
                } for subsidiary in case.get("subsidiaries", [])
            ]
        }
        db.insert_one("entities", entity)
    
    yield db
    
    # Cleanup
    logger.info("[test_entity_matcher.py:db] Cleaning up test database")
    if os.path.exists("test_data"):
        for file in os.listdir("test_data"):
            os.remove(os.path.join("test_data", file))
        os.rmdir("test_data")

@pytest_asyncio.fixture
async def openai_client():
    """Create an OpenAI client"""
    logger.info("[test_entity_matcher.py:openai_client] Creating OpenAI client")
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@pytest_asyncio.fixture
async def matcher(db, openai_client):
    """Create an entity matcher instance"""
    logger.info("[test_entity_matcher.py:matcher] Creating entity matcher")
    return RAGEntityMatcher(db, openai_client)

@pytest.mark.asyncio
async def test_exact_match(matcher):
    logger.info("\n[test_entity_matcher.py:test_exact_match] Starting exact match test")
    logger.info("[test_entity_matcher.py:test_exact_match] Testing with: Meta")
    
    # Test matching current name
    match = await matcher.match_entity(
        {"name": "Meta", "address": "1 Hacker Way, Menlo Park, CA"},
        "company"
    )
    
    logger.info(f"[test_entity_matcher.py:test_exact_match] Match result: {match['name'] if match else None}")
    assert match is not None
    assert match["name"] == "Meta"

@pytest.mark.asyncio
async def test_historical_name_match(matcher):
    logger.info("\n[test_entity_matcher.py:test_historical_name_match] Starting historical name match test")
    logger.info("[test_entity_matcher.py:test_historical_name_match] Testing with: Facebook")
    
    # Test matching old name
    match = await matcher.match_entity(
        {"name": "Facebook", "address": "1 Hacker Way, Menlo Park, CA"},
        "company"
    )
    
    logger.info(f"[test_entity_matcher.py:test_historical_name_match] Match result: {match['name'] if match else None}")
    logger.info(f"[test_entity_matcher.py:test_historical_name_match] Aliases: {match['aliases'] if match else []}")
    assert match is not None
    assert match["name"] == "Meta"
    assert "Facebook" in match["aliases"]

@pytest.mark.asyncio
async def test_rag_similarity_match(matcher):
    logger.info("\n[test_entity_matcher.py:test_rag_similarity_match] Starting RAG similarity match test")
    logger.info("[test_entity_matcher.py:test_rag_similarity_match] Testing with: FB")
    
    # Test matching similar name
    match = await matcher.match_entity(
        {"name": "FB", "address": "1 Hacker Way, Menlo Park, CA"},
        "company"
    )
    
    logger.info(f"[test_entity_matcher.py:test_rag_similarity_match] Match result: {match['name'] if match else None}")
    logger.info(f"[test_entity_matcher.py:test_rag_similarity_match] Abbreviations: {match.get('abbreviations', []) if match else []}")
    assert match is not None
    assert match["name"] == "Meta"

@pytest.mark.asyncio
async def test_web_search_match(matcher):
    logger.info("\n[test_entity_matcher.py:test_web_search_match] Starting web search match test")
    logger.info("[test_entity_matcher.py:test_web_search_match] Testing with: TheFacebook")
    
    # Test matching through web search
    match = await matcher.match_entity(
        {"name": "TheFacebook", "address": "1 Hacker Way, Menlo Park, CA"},
        "company"
    )
    
    logger.info(f"[test_entity_matcher.py:test_web_search_match] Match result: {match['name'] if match else None}")
    logger.info(f"[test_entity_matcher.py:test_web_search_match] Aliases found: {match.get('aliases', []) if match else []}")
    assert match is not None
    assert match["name"] == "Meta"

@pytest.mark.asyncio
async def test_no_match(matcher):
    logger.info("\n[test_entity_matcher.py:test_no_match] Starting no match test")
    logger.info("[test_entity_matcher.py:test_no_match] Testing with: NonexistentCompany")
    
    # Test no match case
    match = await matcher.match_entity(
        {"name": "NonexistentCompany", "address": "123 Fake St"},
        "company"
    )
    
    logger.info(f"[test_entity_matcher.py:test_no_match] Match result: {match['name'] if match else None}")
    assert match is None

@pytest.mark.asyncio
async def test_abbreviation_variations(matcher, metrics):
    """Test matching with different abbreviation formats"""
    start_time = datetime.now()
    
    test_cases = [
        ("IBM", "International Business Machines"),
        ("I.B.M.", "International Business Machines"),
        ("I B M", "International Business Machines"),
        ("ibm", "International Business Machines")
    ]
    
    for query, expected in test_cases:
        match = await matcher.match_entity(
            {"name": query, "address": "1 New Orchard Road, Armonk, NY"},
            "company"
        )
        duration = (datetime.now() - start_time).total_seconds()
        
        metrics.add_match(
            "abbreviation_variations",
            query,
            match["name"] if match else None,
            match.get("similarity", 0) if match else 0,
            "abbreviation",
            duration
        )
        
        assert match is not None
        assert match["name"] == expected

@pytest.mark.asyncio
async def test_company_suffixes(matcher, metrics):
    """Test matching with different company suffixes"""
    start_time = datetime.now()
    
    test_cases = [
        ("Microsoft Corporation", "Microsoft Corporation"),
        ("Microsoft Corp", "Microsoft Corporation"),
        ("Microsoft Corp.", "Microsoft Corporation"),
        ("Microsoft Inc", "Microsoft Corporation"),
        ("Microsoft LLC", "Microsoft Corporation")
    ]
    
    for query, expected in test_cases:
        match = await matcher.match_entity(
            {"name": query, "address": "One Microsoft Way, Redmond, WA"},
            "company"
        )
        duration = (datetime.now() - start_time).total_seconds()
        
        metrics.add_match(
            "company_suffixes",
            query,
            match["name"] if match else None,
            match.get("similarity", 0) if match else 0,
            "suffix_variation",
            duration
        )
        
        assert match is not None
        assert match["name"] == expected

@pytest.mark.asyncio
async def test_case_sensitivity(matcher, metrics):
    """Test matching with different case variations"""
    start_time = datetime.now()
    
    test_cases = [
        ("FEDEX", "FedEx Corporation"),
        ("fedex", "FedEx Corporation"),
        ("FeDex", "FedEx Corporation"),
        ("FedEx", "FedEx Corporation")
    ]
    
    for query, expected in test_cases:
        match = await matcher.match_entity(
            {"name": query, "address": "942 South Shady Grove Road, Memphis, TN"},
            "company"
        )
        duration = (datetime.now() - start_time).total_seconds()
        
        metrics.add_match(
            "case_sensitivity",
            query,
            match["name"] if match else None,
            match.get("similarity", 0) if match else 0,
            "case_variation",
            duration
        )
        
        assert match is not None
        assert match["name"] == expected

@pytest.mark.asyncio
async def test_subsidiary_matching(matcher, metrics):
    """Test matching with subsidiary companies"""
    start_time = datetime.now()
    
    test_cases = [
        ("Red Hat", "International Business Machines"),
        ("GitHub", "Microsoft Corporation"),
        ("LinkedIn Corporation", "Microsoft Corporation")
    ]
    
    for query, expected in test_cases:
        match = await matcher.match_entity(
            {"name": query, "address": ""},
            "company"
        )
        duration = (datetime.now() - start_time).total_seconds()
        
        metrics.add_match(
            "subsidiary_matching",
            query,
            match["name"] if match else None,
            match.get("similarity", 0) if match else 0,
            "subsidiary",
            duration
        )
        
        metrics.add_connection(query, expected, "subsidiary")
        
        assert match is not None
        assert match["name"] == expected

@pytest.mark.asyncio
async def test_fuzzy_matching(matcher, metrics):
    """Test matching with typos and misspellings"""
    start_time = datetime.now()
    
    test_cases = [
        ("Microsft", "Microsoft Corporation"),
        ("Facebok", "Meta"),
        ("Googel", "Alphabet"),
        ("Fed Ex", "FedEx Corporation")
    ]
    
    for query, expected in test_cases:
        match = await matcher.match_entity(
            {"name": query, "address": ""},
            "company"
        )
        duration = (datetime.now() - start_time).total_seconds()
        
        metrics.add_match(
            "fuzzy_matching",
            query,
            match["name"] if match else None,
            match.get("similarity", 0) if match else 0,
            "fuzzy",
            duration
        )
        
        assert match is not None
        assert match["name"] == expected

@pytest.mark.asyncio
async def test_entity_matching(db):
    """Test entity matching functionality"""
    matcher = EntityMatcher(storage_type="json", base_path="test_data")
    
    # Test exact name match
    company_info = {
        "name": "ABC Logistics",
        "address": {
            "street": "123 Main St",
            "city": "Chicago",
            "state": "IL"
        }
    }
    
    matches = await matcher.find_potential_matches(company_info, "company")
    assert len(matches) > 0
    assert matches[0]["name"] == "ABC Logistics"
    
    # Test alias match
    company_info = {
        "name": "ABC Transport",
        "address": {
            "street": "123 Main St",
            "city": "Chicago",
            "state": "IL"
        }
    }
    
    matches = await matcher.find_potential_matches(company_info, "company")
    assert len(matches) > 0
    assert matches[0]["name"] == "ABC Logistics"
    
    # Test subsidiary match
    company_info = {
        "name": "ABC Express",
        "address": {
            "street": "123 Main St",
            "city": "Chicago",
            "state": "IL"
        }
    }
    
    matches = await matcher.find_potential_matches(company_info, "company")
    assert len(matches) > 0
    assert matches[0]["name"] == "ABC Logistics"
    
    # Test abbreviation match
    company_info = {
        "name": "ABC",
        "address": {
            "street": "123 Main St",
            "city": "Chicago",
            "state": "IL"
        }
    }
    
    matches = await matcher.find_potential_matches(company_info, "company")
    assert len(matches) > 0
    assert matches[0]["name"] == "ABC Logistics"

def pytest_sessionfinish(session, exitstatus):
    """Generate visualizations after all tests complete"""
    metrics = getattr(session, 'metrics', None)
    if metrics:
        # Generate visualizations
        metrics.generate_visualizations()
        
        # Save test results as JSON
        with open("test_results/test_results.json", "w") as f:
            json.dump(metrics.matches, f, indent=2)

if __name__ == "__main__":
    pytest.main([__file__]) 