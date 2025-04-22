import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Union, Literal
from pdf2image import convert_from_path
import io
import anthropic
import os
import base64
import requests
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from anthropic import Anthropic
from datetime import datetime


client_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class Signature(BaseModel):
    driver_signature: bool = Field(description="Whether the driver signature is present and valid")
    shipper_signature: bool = Field(description="Whether the shipper signature is present and valid")
    carrier_signature: bool = Field(description="Whether the carrier signature is present and valid")
    consignee_signature: bool = Field(description="Whether the consignee signature is present and valid")

class OCR(BaseModel):
    entity_company_name: str = Field(description="The name of the company")
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

# class StructuredOCR(BaseModel):
#     file_name: str = Field(description="The name of the file")
#     topics: list[str] = Field(description="The topics of the document")
#     languages: str = Field(description="The languages of the document")
#     ocr_contents: OCR = Field(description="The OCR contents of the document")



MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



def analyze_text_with_claude(text: str, image_data: str, image_media_type: str) -> OCR:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
 
    ocr_schema = OCR.model_json_schema()
 
    tools = [
        {
            "name": "build_ocr_result",
            "description": "build the ocr result object",
            "input_schema": ocr_schema
        }
    ]
    # image_media_type = get_media_type(file, encoded)
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=3000,
        temperature=0.0,
        system="You are analyzing a document and extracting structured information from it.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract the information from the image and return it in the specified structured JSON format."
                    }
                ],
            }
        ],
        tools=tools,
        tool_choice={"type": "tool", "name": "build_ocr_result"}
    )
 
    function_call = message.content[0].input
    return OCR(**function_call)


def get_media_type(file_path: str, base64_data: str = None) -> str:
    """Determine the media type from file extension and optionally base64 data"""
    # Check file extension first
    # ext = file_path.lower().split('.')[-1]
    # ext_to_media_type = {
    #     'png': 'image/png',
    #     'jpg': 'image/jpeg',
    #     'jpeg': 'image/jpeg',
    #     'gif': 'image/gif',
    #     'bmp': 'image/bmp',
    #     'tiff': 'image/tiff',
    #     'webp': 'image/webp'
    # }
    
    # media_type = ext_to_media_type.get(ext)
    # if media_type:
    #     print(f"[DEBUG] Media type from extension: {media_type}")
    #     return media_type
        
    # If no extension match or base64 data provided, check base64 header
    if base64_data:
        # Common base64 image headers
        headers = {
            '/9j/': 'image/jpeg',
            'iVBORw0KGgo': 'image/png',
            'R0lGOD': 'image/gif',
            'Qk0': 'image/bmp',
            'SUkqAA': 'image/tiff',
            'UklGR': 'image/webp'
        }
        
        for header, mime_type in headers.items():
            if base64_data.startswith(header):
                print(f"[DEBUG] Media type from base64 header: {mime_type}")
                return mime_type
    
    # Default to JPEG if we can't determine the type
    print("[DEBUG] Could not determine media type, defaulting to image/jpeg")
    return 'image/jpeg'

extract_dir = "/Users/marcberghouse/Desktop/boon_hackathon_csv/data/extract"
files = os.listdir(extract_dir)
file = files[0]
print(file)
extraction_file = os.path.join(extract_dir, file)
extraction_file = Path(extraction_file)
encoded = base64.b64encode(extraction_file.read_bytes()).decode()

# Determine media type
image_media_type = get_media_type(file, encoded)
print(f"[DEBUG] Using media type: {image_media_type}")

ocr_result = analyze_text_with_claude(file, encoded, image_media_type)
print(json.dumps(ocr_result.model_dump(), indent=4))
ocr_json = ocr_result.model_dump()
database_path = "/Users/marcberghouse/Desktop/boon_hackathon_csv/data/entities.json"
database = json.load(open(database_path))
print(database)
#RAG solution

#straight shot LLM solution
print("\n[DEBUG] Looking for exact matches in database...")
print(f"[DEBUG] Entity name from OCR: {ocr_json['entity_company_name']}")

# Check for exact matches (case-insensitive)
exact_matches = []
for entity in database:
    if entity.get("name", "").upper() == ocr_json["entity_company_name"].upper():
        exact_matches.append(entity)

if exact_matches:
    print("\n[DEBUG] Found exact matches:")
    for match in exact_matches:
        print(f"- Name: {match['name']}")
        print(f"  ID: {match.get('_id', 'No ID')}")
else:
    print("[DEBUG] No exact matches found in database")

class Transaction(BaseModel):
    entity_company_name: str = Field(description="The company name of the primary entity")
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

gemini_prompt = f"""
Given the following OCR output and database of entities, please determine if the entities in the OCR output are present in the database.

OCR output:
{ocr_json}

Database:
{database}

"""

from google import genai
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client2 = genai.Client(api_key="")
response = client2.models.generate_content(
    model="gemini-2.0-flash", contents=gemini_prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': Transaction,
    },
)
print(response.text)

# Parse the JSON response
response_data = json.loads(response.text)
print("\n[DEBUG] Parsed response data:", response_data)

# Get all entities (primary and additional)
entities_to_search = []
if response_data.get("entity_company_name"):
    entities_to_search.append({
        "name": response_data["entity_company_name"],
        "address": response_data.get("entity_address", "")
    })

additional_entities = response_data.get("additional_entities", [])
additional_addresses = response_data.get("delivery_location", [])
for i, entity in enumerate(additional_entities):
    address = additional_addresses[i] if i < len(additional_addresses) else ""
    entities_to_search.append({
        "name": entity,
        "address": address
    })

# Remove duplicates based on company name
seen_names = set()
unique_entities = []
for entity in entities_to_search:
    if entity["name"].upper() not in seen_names:
        seen_names.add(entity["name"].upper())
        unique_entities.append(entity)

print("\n[DEBUG] Unique entities to search:", unique_entities)

# Create prompt for all entities
prompt = "Please find information about the following companies:\n\n"
for entity in unique_entities:
    prompt += f"""Company: {entity['name']}
Address: {entity['address']}

"""

prompt += """For each company, please identify any and all potential aliases, abbreviations, alternate spellings, and alternative names for the company.

Focus on verifiable information and official sources. Only give common aliases, nothing obscure.
You must respond with a valid JSON object in the following format:
{
    "entities": [
        {
            "name": "extracted company name",
            "aliases": ["list", "of", "verified", "aliases","abbreviations","alternate spellings","and","alternative names"]
        }
    ]
}

Only respond with the JSON object, nothing else. If you cannot find any aliases for a company, use an empty list for aliases."""

completion = client_oai.chat.completions.create(
    model="gpt-4o-search-preview",
    web_search_options={
        "search_context_size": "medium",
    },
    messages=[{
        "role": "user",
        "content": prompt,
    }],
)

print("\n[DEBUG] OpenAI response:")
print(completion.choices[0].message.content)

# Clean up the response and parse JSON
response_text = completion.choices[0].message.content
response_text = response_text.replace("```json", "").replace("```", "").strip()
print("\n[DEBUG] Cleaned response:", response_text)

# Parse the JSON response
aliases_data = json.loads(response_text)
print("\n[DEBUG] Parsed aliases data:", json.dumps(aliases_data, indent=2))












