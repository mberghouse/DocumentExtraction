import json
import logging
import time
import re
from typing import Dict, List, Any, Optional, Union, Literal
from pdf2image import convert_from_path
import io
from mistralai import Mistral
import anthropic
import os
import base64
import requests
from pathlib import Path
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
from pydantic import BaseModel
from openai import AsyncOpenAI
from datetime import datetime
import asyncio
import aiohttp
from functools import partial
import argparse
from db.db_handler import DatabaseHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process documents and match entities')
    parser.add_argument('--storage-type', 
                       choices=['json', 'csv'],
                       default='json',
                       help='Storage type for the database (json or csv)')
    parser.add_argument('--data-dir',
                       default='data',
                       help='Directory for storing database files')
    parser.add_argument('--extract-dir',
                       default='/Users/marcberghouse/Desktop/boon_hackathon/data/extract',
                       help='Directory containing files to process')
    parser.add_argument('--output-dir',
                       default='/Users/marcberghouse/Desktop/boon_hackathon/data/extracted',
                       help='Directory for output files')
    parser.add_argument('--limit',
                       type=int,
                       default=5,
                       help='Maximum number of files to process')
    return parser.parse_args()

# Initialize database handler
db = DatabaseHandler(storage_type="json", base_path="data")  # Can be changed to "csv" if needed

async def search_company_aliases(client_oai: AsyncOpenAI, company_name: str, company_info: dict) -> List[str]:
    """Search for company aliases using OpenAI web search"""
    address_str = company_info.get('address', '')
    if isinstance(address_str, dict):
        address_parts = []
        for key in ['street', 'city', 'state', 'country']:
            if address_str.get(key):
                address_parts.append(address_str[key])
        address_str = ', '.join(address_parts)

    prompt = f"""Find information about the company:
    Name: {company_name}
    Address: {address_str}
    
    Please identify:
    1. Any alternative names, trade names, or aliases
    2. Parent company or subsidiaries if relevant
    3. Previous company names if any
    4. Common abbreviations used
    
    Focus on verifiable information and official sources."""

    completion = await client_oai.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options={
            "search_context_size": "medium",
        },
        messages=[{
            "role": "user",
            "content": prompt,
        }],
    )
    
    # Extract aliases from the response
    response_text = completion.choices[0].message.content
    aliases = []
    for line in response_text.split('\n'):
        if any(keyword in line.lower() for keyword in ['alias', 'trade name', 'also known as', 'formerly', 'abbreviation']):
            # Extract potential company names
            parts = line.split(':')
            if len(parts) > 1:
                names = parts[1].strip().split(',')
                aliases.extend([name.strip() for name in names if name.strip()])
    
    return aliases

async def find_entity_match(entity_type: str, company_info: dict, client_oai: AsyncOpenAI) -> Optional[Dict]:
    """Find matching entity in database, including alias search if needed"""
    company_name = company_info.get('name', '')
    if not company_name:
        return None

    # First, try direct name match
    matches = db.find_many("entities", {
        "entity_type": entity_type,
        "name": company_name
    })

    if matches:
        return matches[0]

    # If no match found, search for aliases
    aliases = await search_company_aliases(client_oai, company_name, company_info)
    if aliases:
        # Search for matches with the aliases
        for alias in aliases:
            match = db.find_one("entities", {
                "entity_type": entity_type,
                "aliases": alias
            })
            if match:
                # Update the matched entity with the new alias
                db.update_one(
                    "entities",
                    {"_id": match["_id"]},
                    {
                        "$addToSet": {"aliases": company_name},
                        "$set": {"last_updated": datetime.utcnow().isoformat()}
                    }
                )
                return match

    return None

def normalize_address(address_dict: Dict) -> Dict:
    """Normalize address data"""
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

def normalize_entity_data(company_info: Dict) -> Dict:
    """Normalize company data to match schema"""
    normalized = {
        "name": company_info.get("name", ""),
        "addresses": [],
        "phones": [],
        "emails": [],
        "identifiers": {
            "usdot": "",
            "mc_number": "",
            "scac": "",
            "tax_id": ""
        }
    }
    
    # Normalize addresses
    if "address" in company_info:
        if isinstance(company_info["address"], list):
            normalized["addresses"] = [normalize_address(addr) for addr in company_info["address"]]
        else:
            normalized["addresses"] = [normalize_address(company_info["address"])]
    
    # Normalize phone numbers
    if "phone" in company_info:
        if isinstance(company_info["phone"], list):
            normalized["phones"] = [str(phone) for phone in company_info["phone"]]
        else:
            normalized["phones"] = [str(company_info["phone"])]
    
    # Normalize emails
    if "email" in company_info:
        if isinstance(company_info["email"], list):
            normalized["emails"] = [str(email) for email in company_info["email"]]
        else:
            normalized["emails"] = [str(company_info["email"])]
    
    # Normalize identifiers
    for id_type in ["usdot", "mc_number", "scac", "tax_id"]:
        if id_type in company_info:
            normalized["identifiers"][id_type] = str(company_info[id_type])
    
    return normalized

async def create_or_update_entity(entity_type: str, company_info: dict, document_info: dict, client_oai: AsyncOpenAI):
    """Create new entity or update existing one with transaction information"""
    match = await find_entity_match(entity_type, company_info, client_oai)
    match_results = []
    
    if match:
        # Check if this transaction already exists
        transaction_exists = db.find_one(
            "transactions",
            {
                "entity_id": match["_id"],
                "document_info.file_name": document_info["file_name"]
            }
        )
        
        if not transaction_exists:
            # Add new transaction
            transaction = {
                "entity_id": match["_id"],
                "document_info": document_info,
                "transaction_date": datetime.utcnow().isoformat(),
                "shipment_info": document_info.get("shipment_info", {}),
                "cost_info": document_info.get("cost_info", {})
            }
            db.insert_one("transactions", transaction)
            
            # Update entity with any new information
            normalized_info = normalize_entity_data(company_info)
            update_data = {}
            
            # Add new addresses if they don't exist
            for address in normalized_info["addresses"]:
                if address not in match.get("addresses", []):
                    if "$addToSet" not in update_data:
                        update_data["$addToSet"] = {}
                    if "addresses" not in update_data["$addToSet"]:
                        update_data["$addToSet"]["addresses"] = {"$each": []}
                    update_data["$addToSet"]["addresses"]["$each"].append(address)
            
            # Add new phones if they don't exist
            for phone in normalized_info["phones"]:
                if phone not in match.get("phones", []):
                    if "$addToSet" not in update_data:
                        update_data["$addToSet"] = {}
                    if "phones" not in update_data["$addToSet"]:
                        update_data["$addToSet"]["phones"] = {"$each": []}
                    update_data["$addToSet"]["phones"]["$each"].append(phone)
            
            if update_data:
                update_data["$set"] = {"last_updated": datetime.utcnow().isoformat()}
                db.update_one("entities", {"_id": match["_id"]}, update_data)

        match_results.append({
            "entity_type": entity_type,
            "matched_entity": match,
            "is_new": False,
            "confidence": 1.0
        })
    else:
        # Create new entity
        normalized_info = normalize_entity_data(company_info)
        new_entity = {
            "name": normalized_info["name"],
            "entity_type": entity_type,
            "addresses": normalized_info["addresses"],
            "phones": normalized_info["phones"],
            "emails": normalized_info["emails"],
            "identifiers": normalized_info["identifiers"],
            "aliases": await search_company_aliases(client_oai, company_info["name"], company_info),
            "created_date": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        entity_id = db.insert_one("entities", new_entity)
        
        # Create first transaction
        transaction = {
            "entity_id": entity_id,
            "document_info": document_info,
            "transaction_date": datetime.utcnow().isoformat(),
            "shipment_info": document_info.get("shipment_info", {}),
            "cost_info": document_info.get("cost_info", {})
        }
        db.insert_one("transactions", transaction)

        # Get the created entity for result display
        created_entity = db.find_one("entities", {"_id": entity_id})
        match_results.append({
            "entity_type": entity_type,
            "matched_entity": created_entity,
            "is_new": True,
            "confidence": 1.0
        })

    return match_results

class Signature(BaseModel):
    driver_signature: bool
    shipper_signature: bool
    carrier_signature: bool
    consignee_signature: bool

class OCR(BaseModel):
    entity_company_name: str
    shipper_company_info: dict
    shipment_info: dict
    shipment_product_info: dict
    driver_info: dict
    vehicle_info: dict
    cost_info: dict
    consignee_company_info: dict
    delivery_info: dict
    carrier_info: dict
    broker_info: dict
    tms_info: dict
    eld_provider_info: dict
    signatures: dict
    signature_validation: Signature
    date: str
    entity_type: Literal["proof of delivery", "bill of lading"]
    special_instructions: str

class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: str
    ocr_contents: OCR

class BaseDocument(BaseModel):
    """Base document model that all specific document types inherit from"""
    document_type: str
    raw_text: str
    confidence: float
    metadata: Dict[str, Any]

class Entity(BaseModel):
    """Base entity model"""
    name: str
    type: str
    confidence: float
    metadata: Dict[str, Any]
    aliases: List[str] = []
    identifiers: Dict[str, str] = {}
    relationships: List[Dict[str, Any]] = []

class DocumentExtraction(BaseModel):
    """Extraction result from any document type"""
    document: BaseDocument
    entities: List[Entity]
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class BillOfLading(BaseDocument):
    """Specific model for Bill of Lading documents"""
    shipper: Entity
    carrier: Entity
    consignee: Entity
    broker: Optional[Entity]
    shipment_info: Dict[str, Any]

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def process_image(file: str, image_client: Mistral, client_oai: AsyncOpenAI, db: DatabaseHandler, extract_dir: str, output_dir: str):
    """Process a single image file"""
    extraction_file = os.path.join(extract_dir, file)
    extraction_file = Path(extraction_file)
    encoded = base64.b64encode(extraction_file.read_bytes()).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded}"

    # Process image with OCR
    image_response = await asyncio.to_thread(
        image_client.ocr.process,
        document=ImageURLChunk(image_url=base64_data_url),
        model="mistral-ocr-latest"
    )

    # Convert response to JSON
    response_dict = json.loads(image_response.model_dump_json())
    json_string = json.dumps(response_dict, indent=4)
    print(json_string)
    image_ocr_markdown = image_response.pages[0].markdown

    completion = await asyncio.to_thread(
        image_client.chat.parse,
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=base64_data_url),
                    TextChunk(text=(
                        f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n.\n"
                        "Convert this into a structured JSON response according to the requested response format."
                        "For each dictionary field in the given schema, create as many unique fields within that dictionary as possible."
                        "Try to extract all information that can be logically organized in the structured output."
                        "Don't create any extraneous or illogical fields."
                        )
                    )
                ]
            }
        ],
        response_format=StructuredOCR,
        temperature=0
    )

    structured_response = completion.choices[0].message.parsed
    response_dict = json.loads(structured_response.model_dump_json())
    print(json.dumps(response_dict, indent=4))

    # Validate signatures using OpenAI's vision model
    signature_validation = await client_oai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a signature validation assistant. You must ONLY respond with a valid JSON object containing boolean values for each signature type."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Examine this document and determine if it contains valid signatures. Focus specifically on:
                        1. Driver signature
                        2. Shipper signature
                        3. Carrier signature
                        4. Consignee signature

                        For each signature, determine if it is present and appears to be a valid signature (must be a typed or written name).
                        
                        IMPORTANT: You must respond with ONLY a valid JSON object. No other text.
                        The JSON must be in this exact format:
                        {{"driver_signature": true, "shipper_signature": true, "carrier_signature": true, "consignee_signature": true}}

                        Use true if the signature is present and valid, false otherwise."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                        },
                    },
                ],
            }
        ],
        temperature=0
    )

    try:
        # Try to parse the response as JSON
        content = signature_validation.choices[0].message.content
        if content is None:
            raise ValueError("No content in signature validation response")
            
        content = content.strip()
        # Remove any potential markdown code block markers
        content = content.replace("```json", "").replace("```", "").strip()
        signature_validation_dict = json.loads(content)
    except (json.JSONDecodeError, ValueError, AttributeError, IndexError) as e:
        # If parsing fails, use a default value indicating failure to validate
        print(f"Failed to parse signature validation response: {str(e)}")
        signature_validation_dict = {
            "driver_signature": False,
            "shipper_signature": False,
            "carrier_signature": False,
            "consignee_signature": False
        }
    
    response_dict["ocr_contents"]["signature_validation"] = signature_validation_dict

    # Process entities
    entity_types = {
        "shipper": "shipper_company_info",
        "carrier": "carrier_info",
        "consignee": "consignee_company_info",
        "broker": "broker_info"
    }

    all_match_results = []
    for entity_type, info_field in entity_types.items():
        if company_info := response_dict["ocr_contents"].get(info_field):
            match_results = await create_or_update_entity(
                entity_type,
                company_info,
                {
                    "file_name": file,
                    "document_type": response_dict["ocr_contents"]["entity_type"],
                    "date": response_dict["ocr_contents"]["date"],
                    "shipment_info": response_dict["ocr_contents"]["shipment_info"],
                    "cost_info": response_dict["ocr_contents"]["cost_info"]
                },
                client_oai
            )
            all_match_results.extend(match_results)

    # Save matching results along with the OCR results
    response_dict["matching_results"] = all_match_results
    
    with open(os.path.join(output_dir, f"{file}.json"), "w") as f:
        json.dump(response_dict, f, indent=4)

async def process_pdf_page(image: Any, file: str, page_num: int, image_client: Mistral, client_oai: AsyncOpenAI, db: DatabaseHandler):
    """Process a single PDF page"""
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Convert to base64
    encoded = base64.b64encode(img_byte_arr).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded}"
    
    # Process image with OCR
    image_response = await asyncio.to_thread(
        image_client.ocr.process,
        document=ImageURLChunk(image_url=base64_data_url),
        model="mistral-ocr-latest"
    )
    
    image_ocr_markdown = image_response.pages[0].markdown
    
    # Process with Pixtral
    completion = await asyncio.to_thread(
        image_client.chat.parse,
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=base64_data_url),
                    TextChunk(text=(
                        f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n.\n"
                        "Convert this into a structured JSON response according to the requested response format."
                        "For each dictionary field in the given schema, create as many unique fields within that dictionary as possible."
                        "Try to extract all information that can be logically organized in the structured output."
                        "Don't create any extraneous or illogical fields."
                        )
                    )
                ]
            }
        ],
        response_format=StructuredOCR,
        temperature=0
    )
    
    structured_response = completion.choices[0].message.parsed
    response_dict = json.loads(structured_response.model_dump_json())

    # Validate signatures
    signature_validation = await client_oai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a signature validation assistant. You must ONLY respond with a valid JSON object containing boolean values for each signature type."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Examine this document and determine if it contains valid signatures. Focus specifically on:
                        1. Driver signature
                        2. Shipper signature
                        3. Carrier signature
                        4. Consignee signature

                        For each signature, determine if it is present and appears to be a valid signature (not just a typed name or blank space).
                        
                        IMPORTANT: You must respond with ONLY a valid JSON object. No other text.
                        The JSON must be in this exact format:
                        {{"driver_signature": true, "shipper_signature": true, "carrier_signature": true, "consignee_signature": true}}

                        Use true if the signature is present and valid, false otherwise."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                        },
                    },
                ],
            }
        ]
    )

    try:
        # Try to parse the response as JSON
        content = signature_validation.choices[0].message.content
        if content is None:
            raise ValueError("No content in signature validation response")
            
        content = content.strip()
        # Remove any potential markdown code block markers
        content = content.replace("```json", "").replace("```", "").strip()
        signature_validation_dict = json.loads(content)
    except (json.JSONDecodeError, ValueError, AttributeError, IndexError) as e:
        # If parsing fails, use a default value indicating failure to validate
        print(f"Failed to parse signature validation response: {str(e)}")
        signature_validation_dict = {
            "driver_signature": False,
            "shipper_signature": False,
            "carrier_signature": False,
            "consignee_signature": False
        }
    
    response_dict["ocr_contents"]["signature_validation"] = signature_validation_dict
    
    # Process entities
    entity_types = {
        "shipper": "shipper_company_info",
        "carrier": "carrier_info",
        "consignee": "consignee_company_info",
        "broker": "broker_info"
    }

    all_match_results = []
    for entity_type, info_field in entity_types.items():
        if company_info := response_dict["ocr_contents"].get(info_field):
            match_results = await create_or_update_entity(
                entity_type,
                company_info,
                {
                    "file_name": f"{file}_page_{page_num}",
                    "document_type": response_dict["ocr_contents"]["entity_type"],
                    "date": response_dict["ocr_contents"]["date"],
                    "shipment_info": response_dict["ocr_contents"]["shipment_info"],
                    "cost_info": response_dict["ocr_contents"]["cost_info"]
                },
                client_oai
            )
            all_match_results.extend(match_results)

    response_dict["matching_results"] = all_match_results
    return response_dict

async def process_pdf(file: str, image_client: Mistral, client_oai: AsyncOpenAI, db: DatabaseHandler, extract_dir: str, output_dir: str):
    """Process a PDF file"""
    extraction_file = os.path.join(extract_dir, file)
    
    # Convert PDF to images
    images = convert_from_path(extraction_file)
    
    # Process all pages concurrently
    tasks = []
    for i, image in enumerate(images):
        task = process_pdf_page(image, file, i + 1, image_client, client_oai, db)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Save results
    with open(os.path.join(output_dir, f"{file}.json"), "w") as f:
        json.dump(results, f, indent=4)

async def main():
    """Main async function to process all files"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize database handler with command line arguments
    db = DatabaseHandler(storage_type=args.storage_type, base_path=args.data_dir)
    
    # Initialize API clients
    image_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    client_oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Get list of files to process
    files = os.listdir(args.extract_dir)
    
    # Process files concurrently
    tasks = []
    for file in files[:5]:
        if file.endswith((".jpg", ".png")):
            task = process_image(file, image_client, client_oai, db, args.extract_dir, args.output_dir)
            tasks.append(task)
        elif file.endswith(".pdf"):
            task = process_pdf(file, image_client, client_oai, db, args.extract_dir, args.output_dir)
            tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    logger.info(f"Processed {len(tasks)} files. Results saved in {args.output_dir}")
    logger.info(f"Database files stored in {args.data_dir} using {args.storage_type} format")

if __name__ == "__main__":
    asyncio.run(main())