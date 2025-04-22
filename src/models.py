from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import streamlit as st
import os
from pathlib import Path
import asyncio
from mistralai import Mistral
from openai import OpenAI
from anthropic import Anthropic
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
client_oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State code")
    zip_code: str = Field(description="ZIP/Postal code")
    country: str = Field(default="USA", description="Country name")

class Contact(BaseModel):
    name: str = Field(description="Contact name")
    phone: Optional[str] = Field(description="Phone number")
    email: Optional[str] = Field(description="Email address")

class StopInfo(BaseModel):
    location_name: str = Field(description="Name of the stop location")
    address: Address = Field(description="Address of the stop")
    contact: Contact = Field(description="Contact information")
    stop_type: Literal["PU", "SO", "SP", "SD"] = Field(description="Type of stop (PU=Pickup, SO=Delivery, SP=Split pickup, SD=Split drop)")
    appointment_required: bool = Field(description="Whether appointment is required")
    scheduled_arrival: Optional[str] = Field(description="Scheduled arrival time")
    scheduled_departure: Optional[str] = Field(description="Scheduled departure time")
    special_instructions: Optional[str] = Field(description="Special instructions for the stop")
    reference_numbers: Optional[Dict[str, str]] = Field(description="Reference numbers associated with the stop")

class ShipmentDetails(BaseModel):
    commodity_type: str = Field(description="Type of commodity being shipped")
    weight: float = Field(description="Weight of shipment")
    weight_unit: str = Field(default="LB", description="Unit of weight measurement")
    pieces: int = Field(description="Number of pieces")
    pallets_required: bool = Field(description="Whether pallets are required")
    hazmat: bool = Field(description="Whether shipment contains hazardous materials")
    temperature_requirements: Optional[Dict[str, float]] = Field(description="Temperature requirements if applicable")

class ChargeInfo(BaseModel):
    freight_charge: float = Field(description="Base freight charge")
    fuel_surcharge: Optional[float] = Field(description="Fuel surcharge")
    other_charges: Optional[Dict[str, float]] = Field(description="Additional charges")
    total_charge: float = Field(description="Total charges")
    currency: str = Field(default="USD", description="Currency type")

class OCRExtraction(BaseModel):
    """Main OCR extraction model that matches TMS database structure"""
    
    # Order/Shipment Identification
    order_id: str = Field(description="Unique order identifier (e.g. bill number)")
    customer_id: str = Field(description="Customer identifier")
    
    # Basic Information
    order_date: str = Field(description="Date order was placed")
    status: str = Field(description="Current status of the order")
    order_type: str = Field(description="Type of order")
    
    # Equipment Details
    equipment_type: str = Field(description="Type of equipment required")
    trailer_type: Optional[str] = Field(description="Type of trailer needed")
    
    # Stops Information
    stops: List[StopInfo] = Field(description="List of all stops in sequence")
    
    # Shipment Details
    shipment_details: ShipmentDetails = Field(description="Details about the shipment")
    
    # Financial Information
    charges: ChargeInfo = Field(description="Charge and payment information")
    
    # Additional Fields
    special_requirements: Optional[Dict[str, bool]] = Field(description="Special requirements like team drivers, high value, etc.")
    reference_numbers: Dict[str, str] = Field(description="Reference numbers (BOL, PO, etc.)")
    notes: Optional[str] = Field(description="General notes or comments")
    
    # Extracted Raw Text
    raw_text: str = Field(description="Raw extracted text from the document")
    
    # Confidence Scores
    confidence_scores: Dict[str, float] = Field(description="Confidence scores for extracted fields")

    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "10271",
                "customer_id": "LOGIPICA",
                "order_date": "2022-11-02",
                "status": "D",
                "order_type": "T",
                "equipment_type": "53VR",
                "stops": [
                    {
                        "location_name": "WESTERN RECYCLING",
                        "address": {
                            "street": "1020 DENVER STREET",
                            "city": "IDAHO FALLS",
                            "state": "ID",
                            "zip_code": "83402"
                        },
                        "contact": {
                            "name": "MAIN",
                            "phone": "208-529-9908"
                        },
                        "stop_type": "PU",
                        "appointment_required": False,
                        "reference_numbers": {
                            "POL": "71088"
                        }
                    }
                ],
                "shipment_details": {
                    "commodity_type": "DRY",
                    "weight": 0,
                    "weight_unit": "LB",
                    "pieces": 0,
                    "pallets_required": False,
                    "hazmat": False
                },
                "charges": {
                    "freight_charge": 400.0,
                    "total_charge": 400.0,
                    "currency": "USD"
                }
            }
        } 

client = client_anthropic
ocr_schema = OCRExtraction.model_json_schema()
tools = [
    {
        "name": "build_ocr_result",
        "description": "build the ocr result object",
        "input_schema": ocr_schema
    }
]
# if image_media_type == 'application/pdf':
#     doc_type="document"
# else:
#     doc_type="image"
doc_type='document'
extract_dir = "/Users/marcberghouse/Downloads/pdf/"
files = os.listdir(extract_dir)
file = files[0]
extraction_file = os.path.join(extract_dir, file)
extraction_file = Path(extraction_file)

if file.endswith('.pdf'):
    with open(extraction_file, "rb") as f:
        encoded = base64.standard_b64encode(f.read()).decode("utf-8")
else:
    encoded = base64.b64encode(extraction_file.read_bytes()).decode()

message =  client.messages.create(
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
                    "media_type": "application/pdf",
                    "data": encoded,
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
output= OCRExtraction(**function_call)
print(output)