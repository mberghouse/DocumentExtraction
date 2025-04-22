import streamlit as st
import os
from pathlib import Path
import asyncio
from mistralai import Mistral
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import json
from datetime import datetime
import base64
import io
from PIL import Image
import pandas as pd
from pdf2image import convert_from_path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

# Initialize API clients
client_oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Keep all the existing model definitions...
class Classifier(BaseModel):
    entity_type: Literal["freight bill","delivery instruction sheet","customs declaration",
                         "import/export license", "vehicle inspection report", "hazmat documentation",
                         "maintenance records", "lease agreement", "accident report", "carrier contract",
                         "driver employment agreement", "insurance certificate", "permit/license",
                         "driver logbook/ELD", "trip manifest", "load tender", "proof of delivery",
                         "bill of lading", "invoice", "purchase order", "receipt","other"] = Field(description="The type of document")

class Signature(BaseModel):
    driver_signature: bool = Field(description="Whether the driver signature is present and valid")
    shipper_signature: bool = Field(description="Whether the shipper signature is present and valid")
    carrier_signature: bool = Field(description="Whether the carrier signature is present and valid")
    consignee_signature: bool = Field(description="Whether the consignee signature is present and valid")

class OCR(BaseModel):
    primary_entity_company_name: str = Field(description="The name of the company that has produced the document")
    additional_entities: list[str] = Field(description="Any additional entities involved in the transaction that are relevant to the primary entity")
    shipper_company_info: dict = Field(description="The shipper company information")
    shipment_info: dict = Field(description="The shipment information")
    shipment_product_info: dict = Field(description="The shipment product information")
    driver_info: dict = Field(description="The driver information")
    vehicle_info: dict = Field(description="The vehicle information")
    cost_info: dict = Field(description="The cost information")
    consignee_company_info: dict = Field(description="The consignee company information")
    delivery_info: dict = Field(description="The delivery information")
    carrier_info: dict = Field(description="The carrier information")
    broker_info: dict = Field(description="The broker information")
    tms_info: dict = Field(description="The TMS information")
    eld_provider_info: dict = Field(description="The ELD provider information")
    signatures: dict = Field(description="The signatures of the document")
    signature_validation: Signature = Field(description="The signature validation of the document")
    date: str = Field(description="The date of the document")
    entity_type: Literal["proof of delivery", "bill of lading"] = Field(description="The type of document")
    special_instructions: str = Field(description="Any special instructions in the document")
    other_info: dict = Field(description="Any other information in the document not covered by the other fields")

class Transaction(BaseModel):
    primary_entity_company_name: str = Field(description="The company name of the primary entity")
    entity_address: str = Field(description="The address of the primary entity")
    entity_id: str = Field(description="The id of the primary entity")
    additional_entities: list[str] = Field(description="Any additional entities involved in the transaction that are relevant to the primary entity")
    additional_entity_ids: list[str] = Field(description="The ids of the additional entities")
    product_description: list[str] = Field(description="The product description of items being shipped/carried/received by the entity")
    product_price: int = Field(description="The price of the products being shipped/carried/received")
    transportation_cost: int = Field(description="All costs associated with the transportation of the product by the entity")
    delivery_location: list[str] = Field(description="The delivery location of the entity")
    delivery_date: list[str] = Field(description="The delivery date of the entity")
    transaction_id: str = Field(description="The transaction id of the entity")
    shipment_id: str = Field(description="The shipment id of the entity")
    primary_entity_present_in_database: bool = Field(description="Whether the primary entity is present in the database")
    additional_entities_present_in_database: list[bool] = Field(description="Whether the additional entities are present in the database")

def get_nested_value(obj, path):
    """Get value from nested dictionary using dot notation path"""
    try:
        for key in path.split('.'):
            obj = obj[key]
        return obj
    except (KeyError, TypeError):
        return None

async def process_single_file(dbase_file_name: str, ocr_file_name: str) -> tuple[float, dict]:
    """Process a single file pair asynchronously"""
    print(f"\nProcessing file pair: {dbase_file_name} / {ocr_file_name}")
    
    try:
        dbase_file = json.load(open("/Users/marcberghouse/Downloads/tms/" + dbase_file_name))
        ocr_file = json.load(open("/Users/marcberghouse/Downloads/extraction/" + ocr_file_name))
        
        prompt = f"""
        Given the following OCR output and database of entities, please determine what fields in the OCR json output correspond to the fields in the TMS json, and determine the probability of the match.
        You must respond with a valid JSON object that maps OCR fields to TMS fields.
        
        The response should be a JSON object with this exact structure:
        {{
            "field_mappings": {{
                "tms_field_name": ["corresponding_ocr_field_name", probability of match]
                ...
            }}
        }}
        
        Include all fields that have even a minor correspondence. Do not include any explanatory text, only the JSON object.
        
        OCR output:
        {ocr_file}

        Database:
        {dbase_file}
        """

        response = await client_oai.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            response_format={ "type": "json_object" },
            timeout=120
        )
        
        response_text = response.choices[0].message.content
        response_data = json.loads(response_text)
        print("\n[DEBUG] Parsed response data:", json.dumps(response_data, indent=2))

        # Extract matching values
        print("\n[DEBUG] Extracting matching values...")
        value_matches = {}

        for tms_field, [ocr_field, probability] in response_data["field_mappings"].items():
            tms_value = get_nested_value(dbase_file, tms_field)
            ocr_value = get_nested_value(ocr_file, ocr_field)
            
            if tms_value is not None and ocr_value is not None:
                value_matches[tms_field] = {
                    "tms_value": tms_value,
                    "ocr_value": ocr_value,
                    "probability": probability,
                    "exact_match": str(tms_value).lower() == str(ocr_value).lower()
                }

        print("\n[DEBUG] Value matches:", json.dumps(value_matches, indent=2))

        # Analyze matches with GPT-4.1
        analysis_prompt = f"""
        Analyze these field matches between TMS and OCR data and determine the best possible matches.
        For each TMS field, determine if there are better matching OCR fields than currently mapped.
        Assign a probability score (0-1) to each match based on:
        1. Exact text matches (case-insensitive)
        2. Semantic similarity
        3. Data type and format consistency
        4. Context and field meaning

        Current matches:
        {json.dumps(value_matches, indent=2)}

        Original TMS data:
        {json.dumps(dbase_file, indent=2)}

        Original OCR data:
        {json.dumps(ocr_file, indent=2)}

        Respond with a JSON object in this format:
        {{
            "matches": [
                {{
                    "tms_key": "field_name",
                    "ocr_key": "matching_field",
                    "tms_value": "value from TMS",
                    "ocr_value": "value from OCR",
                    "probability": 0.95
                }}
            ]
        }}
        """

        analysis_response = await client_oai.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": analysis_prompt,
            }],
            response_format={ "type": "json_object" },
            timeout=120
        )

        analysis_result = json.loads(analysis_response.choices[0].message.content)

        # Split into high and low confidence matches
        high_confidence_matches = [match for match in analysis_result["matches"] if match["probability"] >= 0.8]
        low_confidence_matches = [match for match in analysis_result["matches"] if match["probability"] < 0.8]

        # Create filtered results dictionary
        filtered_results = {
            "high_confidence_matches": high_confidence_matches,
            "low_confidence_matches": low_confidence_matches
        }

        print("\n[DEBUG] Original Low Confidence Matches (p < 0.8):")
        print(json.dumps(filtered_results["low_confidence_matches"], indent=2))

        # Process low confidence matches with web search concurrently
        async def process_low_confidence_match(match):
            web_search_prompt = f"""
            Given this TMS field and value:
            TMS Field: {match['tms_key']}
            TMS Value: {match['tms_value']}
            
            And this OCR context:
            {json.dumps(ocr_file, indent=2)}
            
            Search for information that could help verify if there's a better match between the TMS field and any OCR data.
            Focus on company names, addresses, dates, and other factual information that could be verified.
            
            Return a JSON object with either:
            1. A confirmed better match with higher probability
            2. The original match if no better match is found

            Format:
            {{
                "tms_key": "field_name",
                "ocr_key": "matching_field",
                "tms_value": "value from TMS",
                "ocr_value": "value from OCR",
                "probability": 0.95,
                "verification_source": "description of what was found in web search"
            }}
            """
            
            try:
                web_analysis = await client_oai.chat.completions.create(
                    model="gpt-4o-search-preview",
                    web_search_options={
                        "search_context_size": "low",
                    },
                    messages=[{
                        "role": "user",
                        "content": web_search_prompt,
                    }],
                    timeout=120
                )
                
                response_text = web_analysis.choices[0].message.content
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    print(f"\n[WARNING] Failed to parse web search response for {match['tms_key']}")
                    return match
            except Exception as e:
                print(f"\n[WARNING] Web search failed for {match['tms_key']}: {str(e)}")
                return match

        # Process all low confidence matches concurrently
        enhanced_low_confidence = await asyncio.gather(
            *[process_low_confidence_match(match) for match in low_confidence_matches]
        )

        # Update low confidence matches with enhanced results
        filtered_results["low_confidence_matches"] = enhanced_low_confidence

        print("\n[DEBUG] High Confidence Matches (p ≥ 0.8):")
        print(json.dumps(filtered_results["high_confidence_matches"], indent=2))

        print("\n[DEBUG] Enhanced Low Confidence Matches (p < 0.8):")
        print(json.dumps(filtered_results["low_confidence_matches"], indent=2))

        # Print summary statistics
        high_conf_count = len(filtered_results["high_confidence_matches"])
        low_conf_count = len(filtered_results["low_confidence_matches"])
        avg_high_prob = sum(m["probability"] for m in filtered_results["high_confidence_matches"]) / high_conf_count if high_conf_count > 0 else 0

        # Calculate how many low confidence matches were improved
        improved_matches = sum(1 for m in filtered_results["low_confidence_matches"] if m["probability"] > 0.7)

        print(f"\n[DEBUG] Match Analysis Summary:")
        print(f"High confidence matches (≥0.8): {high_conf_count}")
        print(f"Low confidence matches (<0.8): {low_conf_count}")
        print(f"Low confidence matches improved through web search: {improved_matches}")
        print(f"Average probability for high confidence matches: {avg_high_prob:.2f}")

        # Calculate accuracy
        total_matches = high_conf_count + low_conf_count
        accuracy = high_conf_count / total_matches if total_matches > 0 else 0
        print(f"\nFile accuracy: {accuracy:.2f}")

        return accuracy, filtered_results["low_confidence_matches"] if low_conf_count > 0 else None

    except Exception as e:
        print(f"\n[ERROR] Failed to process file pair: {str(e)}")
        return 0.0, None

async def main():
    dbase_files = sorted(os.listdir("/Users/marcberghouse/Downloads/tms/"))
    ocr_files = sorted(os.listdir("/Users/marcberghouse/Downloads/extraction/"))
    print(f"Found {len(dbase_files)} TMS files and {len(ocr_files)} OCR files")

    # Initialize tracking
    accuracies = []
    all_low_confidence_matches = {}
    max_files = 50

    # Process all files concurrently
    tasks = []
    for i in range(min(max_files, len(dbase_files))):
        task = asyncio.create_task(process_single_file(dbase_files[i], ocr_files[i]))
        tasks.append((dbase_files[i], task))

    # Wait for all tasks to complete and collect results
    for dbase_file_name, task in tasks:
        accuracy, low_confidence_matches = await task
        accuracies.append(accuracy)
        if low_confidence_matches:
            all_low_confidence_matches[dbase_file_name] = low_confidence_matches

    # Print final summary
    print("\n=== Final Results ===")
    print(f"Files processed: {len(accuracies)}")
    print(f"Average accuracy: {sum(accuracies) / len(accuracies):.2f}")
    print(f"Files with low confidence matches: {len(all_low_confidence_matches)}")

    # Save results
    results = {
        "accuracies": accuracies,
        "average_accuracy": sum(accuracies) / len(accuracies),
        "low_confidence_matches": all_low_confidence_matches
    }

    with open("matching_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to matching_results.json")

if __name__ == "__main__":
    asyncio.run(main()) 