import asyncio
import json
import os
from entity_matcher import EntityMatcher, JSONEncoder
from setup_db import setup_database

async def main():
    # Setup the database first
    print("Initializing database...")
    await setup_database()
    print("Database initialized with sample data")
    
    # Initialize the matcher
    matcher = EntityMatcher()
    
    # Load sample documents from the extracted folder
    extracted_dir = "/Users/marcberghouse/Desktop/boon_hackathon/data/extracted"
    results = []
    
    for filename in os.listdir(extracted_dir):
        if filename.endswith('.json'):
            with open(os.path.join(extracted_dir, filename), 'r') as f:
                document = json.load(f)
                
                print(f"\nProcessing document: {filename}")
                matches = await matcher.process_document(document)
                
                # Print matches
                for entity_type, entity_matches in matches.items():
                    print(f"\nMatches for {entity_type}:")
                    for match in entity_matches:
                        print(f"- Match ID: {match.entity_id}")
                        print(f"  Confidence: {match.confidence_score}")
                        print(f"  Factors: {', '.join(match.matching_factors)}")
                        if match.suggested_updates:
                            print(f"  Updates: {json.dumps(match.suggested_updates, indent=2)}")
                
                results.append({
                    "document": filename,
                    "matches": matches
                })
    
    # Save results
    output_dir = "/Users/marcberghouse/Desktop/boon_hackathon/data/matching_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "matching_results.json"), 'w') as f:
        json.dump(results, f, cls=JSONEncoder, indent=2, default=str)
    
    print("\nMatching results have been saved to matching_results.json")

if __name__ == "__main__":
    asyncio.run(main()) 