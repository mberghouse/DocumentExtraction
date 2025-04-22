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
# Model definitions from simple_main_multidoc_async
# with Classifier, we can classify the document into one of many types and get more 
# precise structured outputs by making a different format class for each type.
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

async def classify_document(text: str, image_data: str, image_media_type: str) -> Classifier:
    
    prompt = f"""
    You are a helpful assistant that can classify documents into one of many types.
    Please only respond with the most likely type of document. Don't include any other text.

    """
    response = await client_oai.beta.chat.completions.parse(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": [
                { "type": "text", "text": prompt },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_media_type};base64,{image_data}",
                    },
                },
            ],
        }],
        response_format=Classifier
    )
    response=response.choices[0].message.content
    return response
# Functions from simple_main_multidoc_async
async def analyze_text_with_claude(text: str, image_data: str, image_media_type: str) -> OCR:
    client = client_anthropic
    ocr_schema = OCR.model_json_schema()
    tools = [
        {
            "name": "build_ocr_result",
            "description": "build the ocr result object",
            "input_schema": ocr_schema
        }
    ]
    if image_media_type == 'application/pdf':
        doc_type="document"
    else:
        doc_type="image"

    message = await client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=8000,
        temperature=0.0,
        system="You are analyzing a document and extracting structured information from it.",
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": doc_type,
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
    if base64_data:
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
    
    ext = file_path.lower().split('.')[-1]
    ext_to_media_type = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'tiff': 'image/tiff',
        'webp': 'image/webp',
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'xls': 'application/vnd.ms-excel',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'ppt': 'application/vnd.ms-powerpoint',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    }
    
    media_type = ext_to_media_type.get(ext)
    if media_type:
        print(f"[DEBUG] Media type from extension: {media_type}")
        return media_type
    
    print("[DEBUG] Could not determine media type, defaulting to image/jpeg")
    return 'image/jpeg'

async def process_file(file: str, extract_dir: str, database: List[dict]) -> Dict:
    print(f"\n[DEBUG] Processing file: {file}")
    extraction_file = os.path.join(extract_dir, file)
    extraction_file = Path(extraction_file)
    
    if file.endswith('.pdf'):
        with open(extraction_file, "rb") as f:
            encoded = base64.standard_b64encode(f.read()).decode("utf-8")
    else:
        encoded = base64.b64encode(extraction_file.read_bytes()).decode()

    image_media_type = get_media_type(file, encoded)
    print(f"[DEBUG] Using media type: {image_media_type}")

    # classifier = await classify_document(file, encoded, image_media_type)
    # print(f"[DEBUG] Classified document as: {classifier}")

    ocr_result = await analyze_text_with_claude(file, encoded, image_media_type)
    print(json.dumps(ocr_result.model_dump(), indent=4))
    ocr_json = ocr_result.model_dump()


    print("\n[DEBUG] Looking for exact matches in database...")
    print(f"[DEBUG] Entity name from OCR: {ocr_json['primary_entity_company_name']}")
    print(f"[DEBUG] Database: {database}")

    exact_matches = []
    for entity in database:
        if entity.get("name", "").upper() == ocr_json["primary_entity_company_name"].upper():
            exact_matches.append(entity)
    
    if len(exact_matches)==len(ocr_json["additional_entities"])+1:
        print("\n[DEBUG] All entities found in database, skipping search")
        return {"ocr_result": ocr_json, "matches": exact_matches}

    gemini_prompt = f"""
    Given the following OCR output and database of entities, please extract the details of the transaction and determine if the entities in the OCR output are present in the database.

    OCR output:
    {ocr_json}

    Database:
    {database}
    """

    response = await client_oai.beta.chat.completions.parse(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": gemini_prompt,
        }],
        response_format=Transaction
    )
    response=response.choices[0].message.content

    response_data = json.loads(response)
    print("\n[DEBUG] Parsed response data:", response_data)

    all_present = response_data["primary_entity_present_in_database"] and \
                 all(present for present in response_data.get("additional_entities_present_in_database", []))

    if all_present:
        print("\n[DEBUG] All entities found in database, skipping search")
        return {"ocr_result": ocr_json, "matches": exact_matches}

    print("\n[DEBUG] Some entities not found, proceeding with search...")
    
    entities_to_search = []
    if not response_data["primary_entity_present_in_database"]:
        entities_to_search.append({
            "name": response_data["primary_entity_company_name"],
            "address": response_data.get("entity_address", "")
        })

    additional_entities = response_data.get("additional_entities", [])
    additional_present = response_data.get("additional_entities_present_in_database", [])
    for i, (entity, present) in enumerate(zip(additional_entities, additional_present)):
        if not present:
            address = response_data.get("delivery_location", [""])[i] if i < len(response_data.get("delivery_location", [])) else ""
            entities_to_search.append({
                "name": entity,
                "address": address
            })

    seen_names = set()
    unique_entities = []
    for entity in entities_to_search:
        if entity["name"].upper() not in seen_names:
            seen_names.add(entity["name"].upper())
            unique_entities.append(entity)

    print("\n[DEBUG] Unique entities to search:", unique_entities)

    if not unique_entities:
        return {"ocr_result": ocr_json, "matches": exact_matches}

    prompt = "Please find information about the following company names that have been extracted via OCR:\n\n"
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

    Only respond with the JSON object, nothing else. If you cannot find any aliases for a company, use an empty list for aliases.
    Don't include any aliases that you don't think have a significant correspondance to the given company name.
    The name field in your JSON response must be the original company/entity name that was extracted via OCR."""

    print(prompt)
    completion = await client_oai.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options={
            "search_context_size": "low",
        },
        messages=[{
            "role": "user",
            "content": prompt,
        }],
    )

    response_text = completion.choices[0].message.content
    response_text = response_text.replace("```json", "").replace("```", "").strip()
    aliases_data = json.loads(response_text)
    print("\n[DEBUG] Parsed aliases data:", json.dumps(aliases_data, indent=2))
    
    # Deduplication logic
    if "entities" in aliases_data:
        # Create a map of entity name/alias to the full entity data
        entity_map = {}
        for entity in aliases_data["entities"]:
            # Add primary name
            if entity["name"].upper() in entity_map:
                # Combine aliases and remove duplicates
                combined_aliases = list(set(entity_map[entity["name"].upper()]["aliases"] + entity["aliases"]))
                entity_map[entity["name"].upper()]["aliases"] = combined_aliases
            else:
                entity_map[entity["name"].upper()] = entity
            
            # Add all aliases
            for alias in entity["aliases"]:
                if alias.upper() in entity_map:
                    # If alias matches another entity's name/alias, combine them
                    combined_aliases = list(set(entity_map[alias.upper()]["aliases"] + entity["aliases"]))
                    entity_map[alias.upper()]["aliases"] = combined_aliases
                else:
                    entity_map[alias.upper()] = entity
        
        # Convert back to list, taking only unique entities
        seen = set()
        unique_entities = []
        for entity in entity_map.values():
            if entity["name"].upper() not in seen:
                seen.add(entity["name"].upper())
                unique_entities.append(entity)
        
        aliases_data["entities"] = unique_entities
        print("\n[DEBUG] Deduplicated entities:", json.dumps(aliases_data, indent=2))

        # Check for matches against deduplicated aliases
        for entity in database:
            if entity.get("name", "").upper() == ocr_json["primary_entity_company_name"].upper():
                if entity not in exact_matches:
                    exact_matches.append(entity)
            # Check against all aliases from deduplicated entities
            for dedup_entity in aliases_data["entities"]:
                if dedup_entity["name"].upper() == entity.get("name", "").upper():
                    for alias in dedup_entity["aliases"]:
                        if alias.upper() == ocr_json["primary_entity_company_name"].upper():
                            if entity not in exact_matches:
                                exact_matches.append(entity)
                                break
    
    return {
        "ocr_result": ocr_json,
        "matches": exact_matches,
        "aliases": aliases_data
    }

async def chat_with_document(question: str, ocr_result: Dict, chat_history: List[Dict]) -> str:
    prompt = f"""You are a helpful assistant that can answer questions about a document. 
    Here is the extracted information from the document:
    {json.dumps(ocr_result, indent=2)}
    
    Chat History:
    {chat_history}
    
    Current Question: {question}
    
    Please provide a clear and concise answer based on the document information."""

    response = await client_oai.chat.completions.create(
        model="gpt-4.1-mini",
        max_tokens=1000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# Streamlit app setup
st.set_page_config(
    page_title="Document Processing Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Home", "Process Documents", "Chat with Documents", "View Results", "View Database", "Analytics", "Settings"]
)

# Home page
if page == "Home":
    st.title("📄 Document Processing Dashboard")
    st.markdown("""
    ### Welcome to the Document Processing System
    
    This application helps you process and analyze documents using advanced AI techniques.
    
    #### Features:
    - 🔍 Process PDF and image documents
    - 🤖 AI-powered OCR and entity extraction
    - 💬 Chat with your processed documents
    - 📊 View detailed results
    
    Get started by selecting a function from the sidebar.
    """)

# Process Documents page
elif page == "Process Documents":
    st.title("Process Documents")
    
    process_type = st.radio(
        "Select processing mode",
        ["Single File", "Batch Processing"]
    )
    
    if process_type == "Single File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "jpg", "jpeg", "png"]
        )
        print(f"Uploaded file: {uploaded_file}")
        
        if uploaded_file:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    database_path = "data/entities.json"
                    if not os.path.exists(database_path):
                        with open(database_path, 'w') as f:
                            json.dump([], f)
                    
                    with open(database_path) as f:
                        database = json.load(f)
                    
                    
                    result = asyncio.run(process_file(uploaded_file.name, str(temp_dir), database))
                    
                    # Store result in session state
                    st.session_state.processed_docs[uploaded_file.name] = result
                    
                    st.success("Processing complete!")
                    st.json(result)
                    st.session_state.chat_history = []

                    # Check for unmatched entities
                    if len(result.get("matches", [])) < len(result["ocr_result"]["additional_entities"]) + 1:
                        st.warning("Some entities remain unmatched!")
                        
                        # Get list of unmatched entities
                        matched_names = {entity.get("name", "").upper() for entity in result.get("matches", [])}
                        unmatched_entities = [result["ocr_result"]["primary_entity_company_name"]]
                        unmatched_entities.extend(result["ocr_result"]["additional_entities"])
                        unmatched_entities = [e for e in unmatched_entities if e.upper() not in matched_names]
                        
                        st.write("Unmatched entities:", ", ".join(unmatched_entities))
                        
                        # action = st.radio(
                        #     "What would you like to do?",
                        #     ["Add to database", "Try fuzzy matching with aliases"]
                        # )
                        
                        # if action == "Add to database":
                        for entity in unmatched_entities:
                            # if st.button(f"Add {entity} to database"):
                                # Add entity to database
                            new_entity = {
                                "name": entity,
                                "aliases": result.get("aliases", {}).get("entities", [])[0].get("aliases", [])
                                    if result.get("aliases") and entity == result["ocr_result"]["primary_entity_company_name"]
                                    else []
                            }
                            database.append(new_entity)
                            with open("data/entities.json", "w") as f:
                                json.dump(database, f, indent=2)
                            st.success(f"Added {entity} to database!")
                        
                        # else:  # Try fuzzy matching
                        #     if st.button("Try fuzzy matching"):
                        #         with st.spinner("Performing fuzzy matching..."):
                        #             async def perform_fuzzy_matching():
                        #                 # Create context with all available information
                        #                 context = {
                        #                     "database": database,
                        #                     "ocr_result": result["ocr_result"],
                        #                     "aliases": result.get("aliases", {}),
                        #                     "unmatched_entities": unmatched_entities
                        #                 }
                                        
                        #                 fuzzy_prompt = f"""
                        #                 Given the following context, please identify any potential matches between the unmatched entities and the database entries, considering all known aliases.
                        #                 Use fuzzy matching to account for minor differences in spelling, formatting, or word order.
                        #                 Only suggest matches that are highly likely to be the same entity.

                        #                 Context:
                        #                 {json.dumps(context, indent=2)}

                        #                 Return a JSON object with suggested matches in this format:
                        #                 {{
                        #                     "matches": [
                        #                         {{
                        #                             "unmatched_entity": "entity name",
                        #                             "suggested_match": "database entity name",
                        #                             "confidence": "high/medium/low",
                        #                             "explanation": "brief explanation of why this is a match"
                        #                         }}
                        #                     ]
                        #                 }}
                        #                 """
                                        
                        #                 response = await client_oai.chat.completions.create(
                        #                     model="gpt-4.1-mini",
                        #                     messages=[{"role": "user", "content": fuzzy_prompt}]
                        #                 )
                        #                 return json.loads(response.choices[0].message.content)
                                    
                        #             fuzzy_matches = asyncio.run(perform_fuzzy_matching())
                                    
                        #             if fuzzy_matches.get("matches"):
                        #                 st.subheader("Suggested Matches:")
                        #                 for match in fuzzy_matches["matches"]:
                        #                     st.write(f"- {match['unmatched_entity']} → {match['suggested_match']}")
                        #                     st.write(f"  Confidence: {match['confidence']}")
                        #                     st.write(f"  Explanation: {match['explanation']}")
                        #                     if st.button(f"Accept match: {match['unmatched_entity']} = {match['suggested_match']}"):
                        #                         # Update database entry with new alias
                        #                         for entity in database:
                        #                             if entity["name"] == match["suggested_match"]:
                        #                                 if "aliases" not in entity:
                        #                                     entity["aliases"] = []
                        #                                 if match["unmatched_entity"] not in entity["aliases"]:
                        #                                     entity["aliases"].append(match["unmatched_entity"])
                        #                         with open("data/entities.json", "w") as f:
                        #                             json.dump(database, f, indent=2)
                        #                         st.success("Database updated!")
                        #             else:
                        #                 st.info("No likely matches found.")
    
    else:  # Batch Processing
        folder_path = st.text_input("Enter folder path containing documents")
        
        if folder_path and os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png'))]
            
            st.write(f"Found {len(files)} files")
            
            if st.button("Process All Files"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                database_path = "data/entities.json"
                if not os.path.exists(database_path):
                    with open(database_path, 'w') as f:
                        json.dump([], f)
                
                with open(database_path) as f:
                    database = json.load(f)
                
                for i, file in enumerate(files):
                    status_text.text(f"Processing {file}...")
                    result = asyncio.run(process_file(file, folder_path, database))
                    st.session_state.processed_docs[file] = result
                    progress_bar.progress((i + 1) / len(files))
                
                status_text.text("Processing complete!")
                st.success(f"Successfully processed {len(files)} files!")

# Chat with Documents page
elif page == "Chat with Documents":
    st.title("Chat with Documents")
    
    if not st.session_state.processed_docs:
        st.warning("No processed documents available. Please process a document first.")
    else:
        doc_name = st.selectbox(
            "Select a document to chat with",
            list(st.session_state.processed_docs.keys())
        )
        
        if doc_name:
            st.write("Ask a question about the document:")
            question = st.text_input("Your question", value="Give me a breakdown of the information in this document")
            
            if question and st.button("Ask"):
                with st.spinner("Getting answer..."):
                    answer = asyncio.run(chat_with_document(
                        question,
                        st.session_state.processed_docs[doc_name]["ocr_result"],
                        st.session_state.chat_history
                    ))
                    st.write("Answer:", answer)
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    print(st.session_state.chat_history)

# View Results page
elif page == "View Results":
    st.title("View Results")
    
    if not st.session_state.processed_docs:
        st.warning("No processed documents available. Please process a document first.")
    else:
        doc_name = st.selectbox(
            "Select a document to view",
            list(st.session_state.processed_docs.keys())
        )
        
        if doc_name:
            st.json(st.session_state.processed_docs[doc_name])

# View Database page
elif page == "View Database":
    st.title("View Database")
    
    # Load database
    try:
        with open("data/entities.json", "r") as f:
            database = json.load(f)
    except FileNotFoundError:
        database = []
    
    if not database:
        st.warning("Database is empty. Process some documents to add entities.")
    else:
        # Convert to DataFrame for better display
        df = pd.DataFrame(database)
        
        # Display options
        view_option = st.radio(
            "Select view option",
            ["Table View", "JSON View"]
        )
        
        if view_option == "Table View":
            # Add search/filter functionality
            search_term = st.text_input("Search entities (by name or alias)")
            if search_term:
                # Search in both names and aliases
                filtered_data = []
                for entity in database:
                    name_match = search_term.lower() in entity["name"].lower()
                    alias_match = any(search_term.lower() in alias.lower() 
                                    for alias in entity.get("aliases", []))
                    if name_match or alias_match:
                        filtered_data.append(entity)
                df = pd.DataFrame(filtered_data)
            
            # Display as table
            st.dataframe(df)
        else:
            # Display as formatted JSON
            st.json(database)
        
        # Export options
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "entities_database.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("Export as JSON"):
                json_str = json.dumps(database, indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    "entities_database.json",
                    "application/json"
                )
        
        # Database stats
        st.subheader("Database Statistics")
        total_entities = len(database)
        total_aliases = sum(len(entity.get("aliases", [])) for entity in database)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Entities", total_entities)
        with col2:
            st.metric("Total Aliases", total_aliases)

# Analytics page
elif page == "Analytics":
    st.title("Analytics Dashboard")
    
    if not st.session_state.processed_docs:
        st.warning("No processed documents available. Please process documents first.")
    else:
        # Entity type distribution
        st.subheader("Document Type Distribution")
        doc_types = [doc["ocr_result"]["entity_type"] for doc in st.session_state.processed_docs.values()]
        doc_type_counts = pd.Series(doc_types).value_counts()
        fig = px.pie(values=doc_type_counts.values, names=doc_type_counts.index)
        st.plotly_chart(fig)
        
        # Entity timeline
        st.subheader("Document Processing Timeline")
        dates = [doc["ocr_result"]["date"] for doc in st.session_state.processed_docs.values()]
        doc_types = [doc["ocr_result"]["entity_type"] for doc in st.session_state.processed_docs.values()]
        timeline_df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "document_type": doc_types
        })
        timeline_df = timeline_df.groupby([timeline_df["date"].dt.date, "document_type"]).size().reset_index()
        timeline_df.columns = ["date", "document_type", "count"]
        fig = px.line(timeline_df, x="date", y="count", color="document_type")
        st.plotly_chart(fig)
        
        # Entity matches analysis
        st.subheader("Entity Matching Analysis")
        matches_data = []
        for doc in st.session_state.processed_docs.values():
            matches_data.append({
                "document": doc["ocr_result"]["primary_entity_company_name"],
                "matches_found": len(doc.get("matches", [])),
                "has_aliases": "aliases" in doc
            })
        matches_df = pd.DataFrame(matches_data)
        fig = px.bar(matches_df, x="document", y="matches_found",
                    color="has_aliases",
                    title="Entity Matches per Document")
        st.plotly_chart(fig)

# Settings page
elif page == "Settings":
    st.title("Settings")
    
    st.subheader("API Configuration")
    
    # API Keys
    mistral_key = st.text_input("Mistral API Key", value=os.getenv("MISTRAL_API_KEY", ""), type="password")
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    anthropic_key = st.text_input("Anthropic API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password")
    
    if st.button("Save API Keys"):
        # Save to .env file
        with open(".env", "w") as f:
            f.write(f"MISTRAL_API_KEY={mistral_key}\n")
            f.write(f"OPENAI_API_KEY={openai_key}\n")
            f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")
        st.success("API keys saved!")
    
    st.subheader("Data Management")
    
    if st.button("Clear Processed Documents"):
        st.session_state.processed_docs = {}
        st.success("Processed documents cleared!")
    
    if st.button("Clear Database"):
        database_path = "data/entities.json"
        with open(database_path, 'w') as f:
            json.dump([], f)
        st.success("Database cleared successfully!")

    st.subheader("Debug Information")
    if st.checkbox("Show Debug Info"):
        st.write("Current working directory:", os.getcwd())
        st.write("Temp directory exists:", os.path.exists("temp"))
        st.write("Results directory exists:", os.path.exists("results"))
        st.write("Data directory exists:", os.path.exists("data"))
        st.write("Number of processed documents:", len(st.session_state.processed_docs))

if __name__ == "__main__":
    st.write("Ready to process documents!") 