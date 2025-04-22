from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import json
from datetime import datetime
import re
from difflib import SequenceMatcher
import os
import asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from .db_handler import DatabaseHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection configuration
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://localhost:27017/?directConnection=true&serverSelectionTimeoutMS=2000"
)

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

def json_dumps(obj):
    return json.dumps(obj, cls=JSONEncoder, indent=2)

class EntityMatch(BaseModel):
    entity_id: str
    confidence_score: float
    matching_factors: List[str]
    suggested_updates: Dict

class EntityMatcher:
    def __init__(self, storage_type: str = "json", base_path: str = "data"):
        try:
            self.db = DatabaseHandler(storage_type=storage_type, base_path=base_path)
            self.openai_client = OpenAI()
        except Exception as e:
            print(f"Failed to initialize database: {e}")
            raise

    def normalize_name(self, name: str) -> str:
        """Normalize company name for better matching"""
        if not name:
            return ""
        # Remove common business entities and punctuation
        normalized = name.lower()
        normalized = re.sub(r'\b(inc|llc|ltd|corp|corporation|company|co)\b\.?', '', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return " ".join(normalized.split())

    def normalize_address(self, address_dict: Dict) -> Dict:
        """Normalize address fields"""
        if isinstance(address_dict, str):
            return {"raw_address": address_dict}
        
        normalized = {}
        field_mappings = {
            'address': 'street',
            'street': 'street',
            'city': 'city',
            'state': 'state',
            'province': 'state',
            'country': 'country',
            'postal_code': 'postal_code',
            'zip': 'postal_code'
        }
        
        for orig_field, norm_field in field_mappings.items():
            if orig_field in address_dict:
                normalized[norm_field] = address_dict[orig_field]
        
        # Create raw_address from components
        raw_parts = []
        for field in ['street', 'city', 'state', 'country', 'postal_code']:
            if field in normalized:
                raw_parts.append(str(normalized[field]))
        normalized['raw_address'] = ", ".join(raw_parts)
        
        return normalized

    async def find_potential_matches(self, entity_data: Dict, entity_type: str) -> List[Dict]:
        """Find potential matches using traditional DB queries"""
        normalized_name = self.normalize_name(entity_data.get('name', ''))
        
        # Build query
        query = {
            "entity_type": entity_type,
            "name": entity_data.get('name', '')  # Exact match first
        }
        
        matches = self.db.find_many("entities", query)
        
        # If no exact matches, try normalized name
        if not matches:
            query = {
                "entity_type": entity_type,
                "normalized_name": normalized_name
            }
            matches = self.db.find_many("entities", query)
        
        # Add identifier matches if available
        identifiers = entity_data.get('identifiers', {})
        for field in ['usdot', 'mc_number', 'scac']:
            if identifiers.get(field):
                query = {
                    f"identifiers.{field}": identifiers[field]
                }
                id_matches = self.db.find_many("entities", query)
                matches.extend(id_matches)
        
        return matches[:5]  # Return top 5 matches

    async def rank_matches_with_llm(self, extracted_entity: Dict, potential_matches: List[Dict]) -> List[EntityMatch]:
        """Use LLM to rank and validate potential matches"""
        prompt = f"""
        Compare the following extracted entity with potential matches from our database.
        Consider the following matching criteria:
        1. Name similarity (accounting for typos, abbreviations, trade names)
        2. Address matching (partial matches, formatting differences)
        3. Contact information correlation
        4. Business identifiers (USDOT, MC numbers, etc.)
        5. Operational context (routes, locations, business relationships)

        Extracted Entity:
        {json_dumps(extracted_entity)}

        Potential Matches:
        {json_dumps(potential_matches)}

        For each potential match, provide:
        1. Confidence score (0-1)
        2. Matching factors that contributed to the score
        3. Any suggested updates to the database record

        You must respond with ONLY a valid JSON object in this exact format:
        {{
            "matches": [
                {{
                    "entity_id": "id",
                    "confidence_score": 0.95,
                    "matching_factors": ["exact_name", "partial_address", "matching_usdot"],
                    "suggested_updates": {{"contacts": [], "addresses": []}}
                }}
            ]
        }}
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert entity matching assistant. You must ONLY respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return [EntityMatch(**match) for match in result.get("matches", [])]

    async def process_document(self, document: Dict) -> Dict[str, List[EntityMatch]]:
        """Process a document and match all entities"""
        entity_types = {
            "shipper": "shipper_company_info",
            "carrier": "carrier_info",
            "consignee": "consignee_company_info",
            "broker": "broker_info"
        }
        
        matches = {}
        for entity_type, field_name in entity_types.items():
            entity_data = document["ocr_contents"].get(field_name)
            if entity_data:
                potential_matches = await self.find_potential_matches(entity_data, entity_type)
                if potential_matches:
                    matches[entity_type] = await self.rank_matches_with_llm(entity_data, potential_matches)
                else:
                    # Create new entity if no matches found
                    new_entity = await self.create_new_entity(entity_data, entity_type, document["file_name"])
                    matches[entity_type] = [EntityMatch(
                        entity_id=str(new_entity.inserted_id),
                        confidence_score=1.0,
                        matching_factors=["new_entity"],
                        suggested_updates={}
                    )]
        
        # Update analytics
        await self.update_analytics(matches)
        return matches

    async def create_new_entity(self, entity_data: Dict, entity_type: str, source_doc: str) -> Dict:
        """Create a new entity in the database"""
        normalized_name = self.normalize_name(entity_data.get('name', ''))
        
        entity = {
            "name": entity_data.get('name', ''),
            "normalized_name": normalized_name,
            "entity_type": entity_type,
            "addresses": [self.normalize_address(entity_data)],
            "identifiers": {
                "usdot": entity_data.get('usdot', ''),
                "mc_number": entity_data.get('mc_number', ''),
                "scac": entity_data.get('scac', ''),
                "tax_id": entity_data.get('tax_id', '')
            },
            "contacts": [
                {
                    "name": entity_data.get('contact_name', ''),
                    "phone": entity_data.get('phone', ''),
                    "email": entity_data.get('email', ''),
                    "role": "primary"
                }
            ],
            "metadata": {
                "confidence_score": 1.0,
                "source_documents": [source_doc]
            }
        }
        
        result = await self.db.insert_one("entities", entity)
        return result

    async def update_entity(self, entity_id: str, updates: Dict):
        """Update an entity with new information"""
        self.db.update_one(
            "entities",
            {"_id": entity_id},
            {"$set": updates}
        )

    async def update_analytics(self, matches: Dict[str, List[EntityMatch]]):
        """Update match analytics"""
        analytics = {
            "date": datetime.utcnow().isoformat(),
            "total_documents": 1,
            "total_entities_matched": sum(len(m) for m in matches.values()),
            "match_scores": {
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0
            },
            "entity_types": {
                "carrier": 0,
                "shipper": 0,
                "consignee": 0,
                "broker": 0
            },
            "new_entities_created": 0,
            "updated_entities": 0,
            "error_rate": 0.0
        }

        for entity_type, entity_matches in matches.items():
            analytics["entity_types"][entity_type] = len(entity_matches)
            for match in entity_matches:
                confidence_score = match.confidence_score
                if confidence_score > 0.9:
                    analytics["match_scores"]["high_confidence"] += 1
                elif confidence_score > 0.7:
                    analytics["match_scores"]["medium_confidence"] += 1
                else:
                    analytics["match_scores"]["low_confidence"] += 1

                if "new_entity" in match.matching_factors:
                    analytics["new_entities_created"] += 1
                elif match.suggested_updates:
                    analytics["updated_entities"] += 1

        await self.db.insert_one("matching_history", analytics)

class RAGEntityMatcher:
    def __init__(self, db, openai_client: AsyncOpenAI):
        """Initialize with database instance and OpenAI client"""
        logger.info("[entity_matcher.py:init] Initializing RAGEntityMatcher")
        self.db = db
        self.openai = openai_client
        self.embedding_cache = {}

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using OpenAI API"""
        logger.info(f"[entity_matcher.py:get_embedding] Getting embedding for text: {text[:50]}...")
        
        if text in self.embedding_cache:
            logger.info("[entity_matcher.py:get_embedding] Using cached embedding")
            return self.embedding_cache[text]
        
        # Add context for abbreviations
        if len(text) <= 3:
            logger.info(f"[entity_matcher.py:get_embedding] Adding context for abbreviation: {text}")
            text = f"company abbreviation: {text}"
        
        logger.info("[entity_matcher.py:get_embedding] Calling OpenAI API for embedding")
        response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = np.array(response.data[0].embedding)
        self.embedding_cache[text] = embedding
        return embedding

    async def search_similar_entities(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar entities using embeddings"""
        logger.info(f"[entity_matcher.py:search_similar_entities] Searching for similar entities to: {query}")
        
        # For short queries, try to match as abbreviation first
        if len(query) <= 3:
            logger.info(f"[entity_matcher.py:search_similar_entities] Trying abbreviation match for: {query}")
            abbrev_matches = await self.db.entities.find({
                "$or": [
                    {"abbreviations": query.upper()},
                    {"abbreviations": query.lower()},
                    {"name": {"$regex": f"^{re.escape(query)}$", "$options": "i"}}
                ]
            }).to_list(length=1)
            
            if abbrev_matches:
                logger.info(f"[entity_matcher.py:search_similar_entities] Found exact abbreviation match: {abbrev_matches[0]['name']}")
                return [(abbrev_matches[0], 1.0)]

        logger.info("[entity_matcher.py:search_similar_entities] Getting query embedding")
        query_embedding = await self.get_embedding(query)
        
        # Get all entities from database
        logger.info("[entity_matcher.py:search_similar_entities] Fetching all entities from database")
        entities = await self.db.entities.find({}).to_list(length=None)
        logger.info(f"[entity_matcher.py:search_similar_entities] Found {len(entities)} entities to compare")
        
        # Get embeddings for all entity names and calculate similarities
        similarities = []
        for entity in entities:
            # Get embedding for entity name
            logger.info(f"[entity_matcher.py:search_similar_entities] Processing entity: {entity['name']}")
            name_embedding = await self.get_embedding(entity['name'])
            similarity = cosine_similarity([query_embedding], [name_embedding])[0][0]
            logger.info(f"[entity_matcher.py:search_similar_entities] Name similarity score: {similarity:.3f}")
            similarities.append((entity, similarity))
            
            # Also check aliases and abbreviations
            for alias in entity.get('aliases', []):
                logger.info(f"[entity_matcher.py:search_similar_entities] Checking alias: {alias}")
                alias_embedding = await self.get_embedding(alias)
                alias_similarity = cosine_similarity([query_embedding], [alias_embedding])[0][0]
                logger.info(f"[entity_matcher.py:search_similar_entities] Alias similarity score: {alias_similarity:.3f}")
                # Keep the highest similarity score
                if alias_similarity > similarity:
                    logger.info(f"[entity_matcher.py:search_similar_entities] Found better match with alias: {alias}")
                    similarities[-1] = (entity, alias_similarity)
            
            # Check abbreviations
            if len(query) <= 3:
                for abbrev in entity.get('abbreviations', []):
                    logger.info(f"[entity_matcher.py:search_similar_entities] Checking abbreviation: {abbrev}")
                    if query.lower() == abbrev.lower():
                        logger.info(f"[entity_matcher.py:search_similar_entities] Found exact abbreviation match: {abbrev}")
                        similarities[-1] = (entity, 1.0)
                        break
        
        # Sort by similarity and return top_k entities with their scores
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:top_k]
        logger.info("[entity_matcher.py:search_similar_entities] Top matches:")
        for entity, score in top_matches:
            logger.info(f"  - {entity['name']}: {score:.3f}")
        return top_matches

    async def match_entity(self, 
                         entity_info: Dict, 
                         entity_type: str, 
                         threshold: float = 0.6) -> Optional[Dict]:
        """Match entity using RAG and web search fallback"""
        name = entity_info['name']
        logger.info(f"[entity_matcher.py:match_entity] Matching entity: {name} (type: {entity_type})")
        
        # First try exact match
        logger.info("[entity_matcher.py:match_entity] Trying exact match")
        exact_match = await self.db.entities.find_one({
            "entity_type": entity_type,
            "$or": [
                {"name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}},
                {"aliases": {"$regex": f"^{re.escape(name)}$", "$options": "i"}},
                {"abbreviations": name.upper()},
                {"abbreviations": name.lower()}
            ]
        })
        if exact_match:
            logger.info(f"[entity_matcher.py:match_entity] Found exact match: {exact_match['name']}")
            return exact_match

        # Try RAG-based matching
        logger.info("[entity_matcher.py:match_entity] Trying RAG-based matching")
        similar_entities = await self.search_similar_entities(
            name, 
            top_k=3
        )
        
        if similar_entities:
            best_match = similar_entities[0]
            logger.info(f"[entity_matcher.py:match_entity] Best RAG match: {best_match[0]['name']} (score: {best_match[1]:.3f})")
            if best_match[1] > threshold:
                # Update the matched entity with the new alias/abbreviation
                updates = {}
                if len(name) <= 3:
                    logger.info(f"[entity_matcher.py:match_entity] Adding as abbreviation: {name}")
                    updates["$addToSet"] = {"abbreviations": name}
                else:
                    logger.info(f"[entity_matcher.py:match_entity] Adding as alias: {name}")
                    updates["$addToSet"] = {"aliases": name}
                updates["$set"] = {"last_updated": datetime.utcnow()}
                
                await self.db.entities.update_one(
                    {"_id": best_match[0]["_id"]},
                    updates
                )
                return best_match[0]

        # If no good match, try web search for aliases
        logger.info("[entity_matcher.py:match_entity] No good matches found, trying web search")
        aliases = await self.search_web_aliases(entity_info)
        if aliases:
            logger.info(f"[entity_matcher.py:match_entity] Found aliases from web search: {aliases}")
            # Search again with aliases
            alias_matches = await self.db.entities.find({
                "entity_type": entity_type,
                "$or": [
                    {"name": {"$in": aliases}},
                    {"aliases": {"$in": aliases}},
                    {"abbreviations": {"$in": [a.upper() for a in aliases if len(a) <= 3]}},
                    {"name": {"$regex": f".*{re.escape(name)}.*", "$options": "i"}},
                    {"aliases": {"$regex": f".*{re.escape(name)}.*", "$options": "i"}}
                ]
            }).to_list(length=1)
            
            if alias_matches:
                logger.info(f"[entity_matcher.py:match_entity] Found match through web search: {alias_matches[0]['name']}")
                # Update the matched entity with the new alias/abbreviation
                updates = {}
                if len(name) <= 3:
                    updates["$addToSet"] = {"abbreviations": name}
                else:
                    updates["$addToSet"] = {"aliases": name}
                updates["$set"] = {"last_updated": datetime.utcnow()}
                
                await self.db.entities.update_one(
                    {"_id": alias_matches[0]["_id"]},
                    updates
                )
                return alias_matches[0]

        logger.info("[entity_matcher.py:match_entity] No matches found")
        return None

    async def search_web_aliases(self, entity_info: Dict) -> List[str]:
        """Search web for company aliases and historical names"""
        logger.info(f"[entity_matcher.py:search_web_aliases] Searching web for aliases of: {entity_info.get('name', '')}")
        
        prompt = f"""Find information about the company:
        Name: {entity_info.get('name', '')}
        Address: {entity_info.get('address', '')}
        
        Please identify:
        1. Current legal name
        2. Previous company names
        3. Parent company and subsidiaries
        4. Trade names and DBAs
        5. Common abbreviations
        6. Any similar names or misspellings
        
        Focus on verifiable information from official sources.
        Include common variations and abbreviations of the company name."""

        logger.info("[entity_matcher.py:search_web_aliases] Calling OpenAI API for web search")
        response = await self.openai.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={
                "search_context_size": "low",
            },
            messages=[{
                "role": "user",
                "content": prompt,
            }],
        )
        print(response.choices[0].message.content)
        logger.info("Response: " + response.choices[0].message.content)
        logger.info("[entity_matcher.py:search_web_aliases] Processing web search results")
        # Parse response to extract aliases
        alias_prompt = f"""Extract all company names from this text as a JSON array:
        {response.choices[0].message.content}
        Format: ["Name 1", "Name 2", "Name 3"]
        Include ONLY company names, no other information.
        Include common abbreviations and variations."""

        logger.info("[entity_matcher.py:search_web_aliases] Calling OpenAI API to extract aliases")
        alias_response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON formatter. Extract company names into a JSON array format."
                },
                {
                    "role": "user",
                    "content": alias_prompt
                }
            ]
        )

        try:
            content = alias_response.choices[0].message.content
            # Clean up the response to ensure it's valid JSON
            content = re.sub(r'[^\[\]\{\}"\':\w\s,.-]', '', content)
            aliases = json.loads(content)
            logger.info(f"[entity_matcher.py:search_web_aliases] Found aliases: {aliases}")
            return aliases if isinstance(aliases, list) else []
        except Exception as e:
            logger.error(f"[entity_matcher.py:search_web_aliases] Error parsing aliases: {str(e)}")
            return []

# Example usage:
"""
matcher = EntityMatcher()

# Process a document
document_matches = await matcher.process_document(document)

# Print matches
for entity_type, matches in document_matches.items():
    print(f"\nMatches for {entity_type}:")
    for match in matches:
        print(f"- Match ID: {match.entity_id}")
        print(f"  Confidence: {match.confidence_score}")
        print(f"  Factors: {', '.join(match.matching_factors)}")
        if match.suggested_updates:
            print(f"  Updates: {json.dumps(match.suggested_updates, indent=2)}")
""" 