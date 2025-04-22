import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from datetime import datetime
import pandas as pd
import networkx as nx

def load_matching_results(extracted_dir):
    """Load all matching results from the extracted JSON files"""
    all_results = []
    for filename in os.listdir(extracted_dir):
        if filename.endswith('.json'):
            with open(os.path.join(extracted_dir, filename), 'r') as f:
                data = json.load(f)
                if "matching_results" in data:
                    all_results.extend(data["matching_results"])
    return all_results

def create_visualizations(results, output_dir):
    """Create and save various visualizations of the matching results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data
    entity_counts = defaultdict(int)
    match_status = defaultdict(int)
    entity_relationships = defaultdict(list)
    
    for result in results:
        entity_type = result["entity_type"]
        entity_counts[entity_type] += 1
        match_status["Matched" if result["match_found"] else "New Entity"] += 1
        
        if result["match_found"] and result.get("match_details"):
            company_name = result["company_name"]
            matched_name = result["match_details"]["matched_name"]
            entity_relationships[entity_type].append((company_name, matched_name))

    # 1. Entity Type Distribution (Pie Chart)
    plt.figure(figsize=(10, 6))
    plt.pie(entity_counts.values(), labels=entity_counts.keys(), autopct='%1.1f%%')
    plt.title('Distribution of Entity Types')
    plt.savefig(os.path.join(output_dir, f'entity_distribution_{timestamp}.png'))
    plt.close()

    # 2. Match Status (Bar Chart)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(match_status.keys()), y=list(match_status.values()))
    plt.title('Match Status Distribution')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, f'match_status_{timestamp}.png'))
    plt.close()

    # 3. Entity Matching Network Graph
    for entity_type, relationships in entity_relationships.items():
        if relationships:
            G = nx.Graph()
            for source, target in relationships:
                G.add_edge(source, target)
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                   node_size=1500, font_size=8, font_weight='bold')
            plt.title(f'Entity Matching Network - {entity_type}')
            plt.savefig(os.path.join(output_dir, f'network_{entity_type}_{timestamp}.png'))
            plt.close()

    # 4. Match Success Rate by Entity Type
    match_success = defaultdict(lambda: {"matched": 0, "total": 0})
    for result in results:
        entity_type = result["entity_type"]
        match_success[entity_type]["total"] += 1
        if result["match_found"]:
            match_success[entity_type]["matched"] += 1

    success_rates = {k: v["matched"]/v["total"]*100 for k, v in match_success.items()}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(success_rates.keys()), y=list(success_rates.values()))
    plt.title('Match Success Rate by Entity Type')
    plt.ylabel('Success Rate (%)')
    plt.savefig(os.path.join(output_dir, f'success_rate_{timestamp}.png'))
    plt.close()

    # 5. Alias Distribution (if available)
    alias_counts = defaultdict(int)
    for result in results:
        if result["match_found"] and result["match_details"].get("aliases"):
            num_aliases = len(result["match_details"]["aliases"])
            alias_counts[num_aliases] += 1

    if alias_counts:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(alias_counts.keys()), y=list(alias_counts.values()))
        plt.title('Distribution of Number of Aliases per Entity')
        plt.xlabel('Number of Aliases')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, f'alias_distribution_{timestamp}.png'))
        plt.close()

    # Generate summary statistics
    summary = {
        "total_entities": len(results),
        "match_rate": (match_status["Matched"] / len(results)) * 100,
        "entity_type_distribution": dict(entity_counts),
        "match_status": dict(match_status),
        "success_rates": success_rates
    }

    # Save summary to JSON
    with open(os.path.join(output_dir, f'matching_summary_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    return summary

if __name__ == "__main__":
    # Load results
    extracted_dir = "/Users/marcberghouse/Desktop/boon_hackathon/data/extracted"
    output_dir = "/Users/marcberghouse/Desktop/boon_hackathon/data/visualizations"
    
    results = load_matching_results(extracted_dir)
    summary = create_visualizations(results, output_dir)
    
    # Print summary
    print("\nMatching Analysis Summary:")
    print(f"Total Entities Processed: {summary['total_entities']}")
    print(f"Overall Match Rate: {summary['match_rate']:.1f}%")
    print("\nEntity Type Distribution:")
    for entity_type, count in summary['entity_type_distribution'].items():
        print(f"  {entity_type}: {count}")
    print("\nMatch Success Rates:")
    for entity_type, rate in summary['success_rates'].items():
        print(f"  {entity_type}: {rate:.1f}%") 