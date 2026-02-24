"""
data/handlers.py
----------------
Unified data loading and saving abstraction layers.

This module provides a common interface (`FileHandler`) for reading and writing
different file formats while exposing a consistent, format agnostic API to the 
rest of the project.

The goal is to abstract away file-format specific logic, enable chunked 
processing for large datasets. Also standardize conversions between `NumPy`
arrays and `pandas` DataFrames. This relying primarily on `PyArrow` and `pandas`
for performance and stability.

Supports:
    - CSV

The design allows easy extension to additional formats such as CSV, Parquet,
JSON and so on.
"""
from __future__ import annotations

import orjson
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional, Type, Union
from abc import ABC, abstractmethod



class FileHandler(ABC):
    """ Abstract base class of shared implementations """

    @abstractmethod
    def load_chunks(self, path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """ Yield `pd.DataFrame` chunks from a file """
        raise NotImplementedError
    

    def save(self, data: Dict[str,np.ndarray], path: Path):
        """ Save processed data to file """
        if not data: 
            raise ValueError("Cannot save empty data directory")

        df = self._prepare_dataframe(data)
        if df.empty: 
            raise ValueError("Prepared dataframe is empty")
        
        self._write_dataframe(df, path)
        print(f"Saved data to `{path}`")

    
    @abstractmethod
    def _write_dataframe(self, df: pd.DataFrame, path: Path):
        """ Write `pd.DataFrame` to specific format """
        raise NotImplementedError
    

    def _prepare_dataframe(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Convert processed array back to `pd.DataFrame`"""
        if not data: 
            return pd.DataFrame()
        
        df: Dict[str, List[Any]] = {}
        for key, array in data.items():
            array = np.asarray(array)
            
            if array.size == 0:
                df[key] = []
                continue
            
            if array.ndim == 1 and array.dtype != object:
                df[key] = array.tolist()
                continue
            
            if array.ndim > 1:
                df[key] = array.tolist()
                continue
            
            # Handle object arrays (the expensive case)
            # Pre-allocate list and use simple checks
            if array.dtype == object:
                column = [None] * len(array)
                for i, value in enumerate(array):
                    
                    if isinstance(value, (list, np.ndarray, dict)):
                        column[i] = orjson.dumps(value).decode("utf-8")
                    else:
                        column[i] = str(value)
                
                df[key] = column
            
            else:
                df[key] = array.tolist()
        
        return pd.DataFrame(df)


class CsvHandler(FileHandler):
    """ Chunked CSV handler using PyArrow """

    # Average row size in bytes (estimate)
    ROW_SIZE        : Final[int] = 1024
    MIN_BLOCK_SIZE  : Final[int] = 1 << 20    # 1 Megabyte

    def load_chunks(self, path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """ Read CSV file in chunks using PyArrow """
        if not path.exists(): 
            raise FileNotFoundError(f"CSV file `{path}` not found")
        
        try:
            ro = pv.ReadOptions(
                block_size=max(self.MIN_BLOCK_SIZE, chunk_size * self.ROW_SIZE),
                use_threads=True
            )
            co = pv.ConvertOptions(auto_dict_encode=False, strings_can_be_null=True)
            reader = pv.open_csv(path, read_options=ro, convert_options=co)
            
            for batch in reader:
                df = batch.to_pandas()
                if not df.empty: 
                    yield df
            
            print(f"Loaded CSV from `{path}`")
        
        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            print(f"Arrow error reading `{path}`: {e}")
            raise 

        except Exception as e:
            print(f"Error reading `{path}`: {e}")
            raise
    

    def _write_dataframe(self, df: pd.DataFrame, path: Path):
        """ Write `pd.DataFrame` to CSV """
        df.to_csv(path)




class HandlerFactory:
    """ Factory for creating appropriate file-handler """
    HANDLERS: Dict[str,Type[FileHandler]] = {
        ".csv": CsvHandler
    }

    @classmethod
    def get_handler(cls, path: Union[str,Path]) -> FileHandler:
        """ Get the appropriate file extension """
        path = Path(path)
        suffix = path.suffix.lower()
        if not suffix:
            raise ValueError(f"Cannot infer file type from file `{path}`")

        handler = cls.HANDLERS.get(suffix)
        if not handler:
            raise ValueError(
                f"!`{suffix}`, supported: `{', '.join(cls.HANDLERS.keys())}`"
            )
        return handler()
