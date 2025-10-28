import pandas as pd
from typing import Dict, Optional, List
import threading

class DataStore:
    """Thread-safe in-memory data store for analyzed datasets"""
    
    def __init__(self):
        self._data = {}
        self._metadata = {}
        self._lock = threading.Lock()
    
    def store_data(self, run_id: str, df: pd.DataFrame, analysis_result: dict):
        """Store analyzed data and metadata"""
        with self._lock:
            self._data[run_id] = df
            self._metadata[run_id] = analysis_result
    
    def get_data(self, run_id: str) -> Optional[pd.DataFrame]:
        """Retrieve dataframe by run_id"""
        with self._lock:
            return self._data.get(run_id)
    
    def get_metadata(self, run_id: str) -> Optional[dict]:
        """Retrieve analysis metadata by run_id"""
        with self._lock:
            return self._metadata.get(run_id)
    
    def get_latest_run_id(self) -> Optional[str]:
        """Get the most recent run_id"""
        with self._lock:
            if not self._data:
                return None
            return max(self._data.keys())
    
    def list_runs(self) -> List[str]:
        """List all available run_ids"""
        with self._lock:
            return list(self._data.keys())

# Global instance
data_store = DataStore()
