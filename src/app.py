import streamlit as st
import os
from pathlib import Path
import asyncio
from mistralai import Mistral
from openai import AsyncOpenAI
from db.db_handler import DatabaseHandler
import json
from datetime import datetime
import base64
import io
from PIL import Image
import pandas as pd
from pdf2image import convert_from_path
import plotly.express as px
import plotly.graph_objects as go
from main import process_image, process_pdf

# Page configuration
st.set_page_config(
    page_title="Document Processing Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create necessary directories
print("[DEBUG] Creating necessary directories")
os.makedirs("temp", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)
print(f"[DEBUG] Current working directory: {os.getcwd()}")
print(f"[DEBUG] Directories created/verified: temp, results, data")

# Initialize session state
if 'db' not in st.session_state:
    print("[DEBUG] Initializing database handler")
    st.session_state.db = DatabaseHandler(storage_type="json", base_path="data")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Home", "Process Documents", "View Database", "Analytics", "Settings"]
)

# Database settings in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Database Settings")
storage_type = st.sidebar.selectbox(
    "Storage Type",
    ["json", "csv"],
    index=0
)
data_dir = st.sidebar.text_input(
    "Data Directory",
    value="data"
)

if st.sidebar.button("Update Database Settings"):
    st.session_state.db = DatabaseHandler(storage_type=storage_type, base_path=data_dir)
    st.sidebar.success("Database settings updated!")

# Home page
if page == "Home":
    st.title("📄 Document Processing Dashboard")
    
    # Welcome message
    st.markdown("""
    ### Welcome to the Document Processing System
    
    This application helps you process and analyze documents using advanced AI techniques.
    
    #### Features:
    - 🔍 Process PDF and image documents
    - 🤖 AI-powered OCR and entity extraction
    - 📊 Data visualization and analytics
    - 💾 Flexible storage options (JSON/CSV)
    
    Get started by selecting a function from the sidebar.
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Documents",
            str(st.session_state.db.count_documents("transactions"))
        )
    
    with col2:
        st.metric(
            "Total Entities",
            str(st.session_state.db.count_documents("entities"))
        )
    
    with col3:
        st.metric(
            "Storage Type",
            storage_type.upper()
        )

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
            print(f"\n[DEBUG] File uploaded: {uploaded_file.name}")
            
            # Create temporary directory for processing
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            print(f"[DEBUG] Created temp directory: {temp_dir}")
            
            # Save uploaded file
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                print(f"[DEBUG] Writing file content, size: {len(uploaded_file.getvalue())} bytes")
                f.write(uploaded_file.getvalue())
            print(f"[DEBUG] Saved file to: {file_path}")
            
            if st.button("Process File"):
                with st.spinner("Processing..."):
                    # Initialize API clients
                    print("[DEBUG] Starting API client initialization")
                    print(f"[DEBUG] Mistral API key present: {bool(os.getenv('MISTRAL_API_KEY'))}")
                    print(f"[DEBUG] OpenAI API key present: {bool(os.getenv('OPENAI_API_KEY'))}")
                    
                    image_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
                    client_oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    print("[DEBUG] API clients initialized")
                    
                    # Process file
                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        result = asyncio.run(process_image(
                            str(file_path.name),
                            image_client,
                            client_oai,
                            st.session_state.db,
                            str(temp_dir),
                            "results"
                        ))
                    else:  # PDF
                        result = asyncio.run(process_pdf(
                            str(file_path.name),
                            image_client,
                            client_oai,
                            st.session_state.db,
                            str(temp_dir),
                            "results"
                        ))
                    
                    print(f"[DEBUG] Processing complete! Result type: {type(result)}")
                    
                    # Try to load the results file
                    try:
                        result_file = os.path.join("results", f"{file_path.name}.json")
                        print(f"[DEBUG] Loading results from {result_file}")
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                            print("[DEBUG] Successfully loaded results file")
                            st.success("Processing complete!")
                            st.json(result_data)
                    except Exception as e:
                        print(f"[ERROR] Failed to load results: {str(e)}")
                        st.error("Failed to display results. Check the results directory for the output file.")
    
    else:  # Batch Processing
        folder_path = st.text_input("Enter folder path containing documents")
        
        if folder_path and os.path.exists(folder_path):
            print(f"\n[DEBUG] Scanning folder: {folder_path}")
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png'))]
            print(f"[DEBUG] Found {len(files)} files")
            
            st.write(f"Found {len(files)} files")
            
            if st.button("Process All Files"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize API clients
                print("[DEBUG] Initializing API clients for batch processing")
                print(f"[DEBUG] Mistral API key present: {bool(os.getenv('MISTRAL_API_KEY'))}")
                print(f"[DEBUG] OpenAI API key present: {bool(os.getenv('OPENAI_API_KEY'))}")
                
                image_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
                client_oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                print("[DEBUG] API clients initialized")
                
                for i, file in enumerate(files):
                    print(f"\n[DEBUG] Processing file {i+1}/{len(files)}: {file}")
                    status_text.text(f"Processing {file}...")
                    
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        asyncio.run(process_image(
                            file,
                            image_client,
                            client_oai,
                            st.session_state.db,
                            folder_path,
                            "results"
                        ))
                    else:  # PDF
                        asyncio.run(process_pdf(
                            file,
                            image_client,
                            client_oai,
                            st.session_state.db,
                            folder_path,
                            "results"
                        ))
                    
                    progress_bar.progress((i + 1) / len(files))
                
                print("[DEBUG] Batch processing completed")
                status_text.text("Processing complete!")
                st.success(f"Successfully processed {len(files)} files!")

# View Database page
elif page == "View Database":
    st.title("View Database")
    
    collection = st.selectbox(
        "Select collection to view",
        ["entities", "transactions", "matching_history"]
    )
    
    # Get data
    if st.session_state.db.storage_type == "json":
        data = st.session_state.db._read_json(collection)
        df = pd.DataFrame(data)
    else:
        df = st.session_state.db._read_csv(collection)
    
    # Display data
    st.dataframe(df)
    
    # Export options
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"{collection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("Export to JSON"):
                json_str = df.to_json(orient="records", indent=2)
                st.download_button(
                    "Download JSON",
                    json_str,
                    f"{collection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )

# Analytics page
elif page == "Analytics":
    st.title("Analytics Dashboard")
    
    # Entity type distribution
    entities_df = pd.DataFrame(st.session_state.db._read_json("entities"))
    if not entities_df.empty:
        st.subheader("Entity Type Distribution")
        entity_counts = entities_df["entity_type"].value_counts()
        fig = px.pie(values=entity_counts.values, names=entity_counts.index)
        st.plotly_chart(fig)
        
        # Entity creation timeline
        st.subheader("Entity Creation Timeline")
        entities_df["created_date"] = pd.to_datetime(entities_df["created_date"])
        timeline_df = entities_df.groupby([entities_df["created_date"].dt.date, "entity_type"]).size().reset_index()
        timeline_df.columns = ["date", "entity_type", "count"]
        fig = px.line(timeline_df, x="date", y="count", color="entity_type")
        st.plotly_chart(fig)
    
    # Transaction analysis
    transactions_df = pd.DataFrame(st.session_state.db._read_json("transactions"))
    if not transactions_df.empty:
        st.subheader("Transaction Analysis")
        transactions_df["transaction_date"] = pd.to_datetime(transactions_df["transaction_date"])
        daily_transactions = transactions_df.groupby(transactions_df["transaction_date"].dt.date).size()
        fig = px.bar(x=daily_transactions.index, y=daily_transactions.values)
        fig.update_layout(title="Daily Transactions", xaxis_title="Date", yaxis_title="Number of Transactions")
        st.plotly_chart(fig)

# Settings page
else:
    st.title("Settings")
    
    st.subheader("API Configuration")
    
    # API Keys
    mistral_key = st.text_input("Mistral API Key", value=os.getenv("MISTRAL_API_KEY", ""), type="password")
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    
    if st.button("Save API Keys"):
        # Save to .env file
        with open(".env", "w") as f:
            f.write(f"MISTRAL_API_KEY={mistral_key}\n")
            f.write(f"OPENAI_API_KEY={openai_key}\n")
        st.success("API keys saved!")
    
    st.subheader("Database Maintenance")
    
    if st.button("Clear Database"):
        if st.session_state.db.storage_type == "json":
            for collection in ["entities", "transactions", "matching_history"]:
                st.session_state.db._write_json(collection, [])
        else:
            for collection in ["entities", "transactions", "matching_history"]:
                st.session_state.db._write_csv(collection, pd.DataFrame())
        st.success("Database cleared successfully!") 