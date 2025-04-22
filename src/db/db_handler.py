import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import uuid

class DatabaseHandler:
    def __init__(self, storage_type: str = "json", base_path: str = "data"):
        """Initialize database handler with specified storage type ('json' or 'csv')"""
        self.storage_type = storage_type.lower()
        if self.storage_type not in ["json", "csv"]:
            raise ValueError("Storage type must be either 'json' or 'csv'")
        
        self.base_path = base_path
        self.collections = {
            "entities": os.path.join(base_path, f"entities.{storage_type}"),
            "transactions": os.path.join(base_path, f"transactions.{storage_type}"),
            "matching_history": os.path.join(base_path, f"matching_history.{storage_type}")
        }
        
        # Create base directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Initialize empty collections if they don't exist
        self._initialize_collections()

    def _initialize_collections(self):
        """Initialize empty collection files if they don't exist"""
        for collection_path in self.collections.values():
            if not os.path.exists(collection_path):
                if self.storage_type == "json":
                    with open(collection_path, 'w') as f:
                        json.dump([], f)
                else:  # csv
                    df = pd.DataFrame()
                    df.to_csv(collection_path, index=False)

    def _read_json(self, collection: str) -> List[Dict]:
        """Read data from JSON file"""
        try:
            with open(self.collections[collection], 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _write_json(self, collection: str, data: List[Dict]):
        """Write data to JSON file"""
        with open(self.collections[collection], 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _read_csv(self, collection: str) -> pd.DataFrame:
        """Read data from CSV file"""
        try:
            return pd.read_csv(self.collections[collection])
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return pd.DataFrame()

    def _write_csv(self, collection: str, data: Union[List[Dict], pd.DataFrame]):
        """Write data to CSV file"""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        df.to_csv(self.collections[collection], index=False)

    def find_one(self, collection: str, query: Dict) -> Optional[Dict]:
        """Find a single document matching the query"""
        if self.storage_type == "json":
            data = self._read_json(collection)
            for item in data:
                if all(item.get(k) == v for k, v in query.items()):
                    return item
        else:  # csv
            df = self._read_csv(collection)
            mask = pd.Series([True] * len(df))
            for k, v in query.items():
                mask &= df[k] == v
            matching = df[mask]
            if not matching.empty:
                return matching.iloc[0].to_dict()
        return None

    def find_many(self, collection: str, query: Dict = None) -> List[Dict]:
        """Find all documents matching the query"""
        if query is None:
            query = {}
            
        if self.storage_type == "json":
            data = self._read_json(collection)
            return [item for item in data if all(item.get(k) == v for k, v in query.items())]
        else:  # csv
            df = self._read_csv(collection)
            if not query:
                return df.to_dict('records')
            mask = pd.Series([True] * len(df))
            for k, v in query.items():
                mask &= df[k] == v
            return df[mask].to_dict('records')

    def insert_one(self, collection: str, document: Dict) -> str:
        """Insert a single document and return its ID"""
        # Generate a unique ID using UUID
        document['_id'] = str(uuid.uuid4())
        document['created_date'] = datetime.utcnow().isoformat()
        document['last_updated'] = document['created_date']
        
        if self.storage_type == "json":
            data = self._read_json(collection)
            data.append(document)
            self._write_json(collection, data)
        else:  # csv
            df = self._read_csv(collection)
            # Convert lists and dicts to strings for CSV storage
            doc_for_csv = {}
            for k, v in document.items():
                if isinstance(v, (list, dict)):
                    doc_for_csv[k] = json.dumps(v)
                else:
                    doc_for_csv[k] = v
            df = pd.concat([df, pd.DataFrame([doc_for_csv])], ignore_index=True)
            self._write_csv(collection, df)
        
        return document['_id']

    def update_one(self, collection: str, query: Dict, update: Dict) -> bool:
        """Update a single document matching the query"""
        if self.storage_type == "json":
            data = self._read_json(collection)
            for item in data:
                if all(item.get(k) == v for k, v in query.items()):
                    if "$set" in update:
                        item.update(update["$set"])
                    if "$addToSet" in update:
                        for field, value in update["$addToSet"].items():
                            if field not in item:
                                item[field] = []
                            if isinstance(value, dict) and "$each" in value:
                                for v in value["$each"]:
                                    if v not in item[field]:
                                        item[field].append(v)
                            elif value not in item[field]:
                                item[field].append(value)
                    item['last_updated'] = datetime.utcnow().isoformat()
                    self._write_json(collection, data)
                    return True
        else:  # csv
            df = self._read_csv(collection)
            mask = pd.Series([True] * len(df))
            for k, v in query.items():
                mask &= df[k] == v
            if mask.any():
                if "$set" in update:
                    for k, v in update["$set"].items():
                        if isinstance(v, (list, dict)):
                            df.loc[mask, k] = json.dumps(v)
                        else:
                            df.loc[mask, k] = v
                df.loc[mask, 'last_updated'] = datetime.utcnow().isoformat()
                self._write_csv(collection, df)
                return True
        return False

    def delete_one(self, collection: str, query: Dict) -> bool:
        """Delete a single document matching the query"""
        if self.storage_type == "json":
            data = self._read_json(collection)
            for i, item in enumerate(data):
                if all(item.get(k) == v for k, v in query.items()):
                    data.pop(i)
                    self._write_json(collection, data)
                    return True
        else:  # csv
            df = self._read_csv(collection)
            mask = pd.Series([True] * len(df))
            for k, v in query.items():
                mask &= df[k] == v
            if mask.any():
                df = df[~mask]
                self._write_csv(collection, df)
                return True
        return False

    def count_documents(self, collection: str, query: Dict = None) -> int:
        """Count documents matching the query"""
        if query is None:
            query = {}
            
        if self.storage_type == "json":
            data = self._read_json(collection)
            return len([item for item in data if all(item.get(k) == v for k, v in query.items())])
        else:  # csv
            df = self._read_csv(collection)
            if not query:
                return len(df)
            mask = pd.Series([True] * len(df))
            for k, v in query.items():
                mask &= df[k] == v
            return mask.sum() 