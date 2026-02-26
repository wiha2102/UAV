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
    """
    Abstract base class defining the unified file I/O interface. Concrete 
    subclasses implement format-specific logic while adhering to a common
    set of methods,

    - `load_chunks` yields pandas DataFrames in a streaming fashion
    - `save` persists processed NumPy-based datasets
    - `_write_dataframe` handles backend-specific writing logic

    This abstraction ensures that higher-level components remain fully
    decoupled from file format details.
    """

    @abstractmethod
    def load_chunks(self, path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Yield DataFrame chunks from the specified file. Implementations must 
        support memory-efficient streaming to allow processing of large 
        datasets without loading the entire file into memory.

        Args:
        -----
        path : File location.
        chunk_size : Approximate number of rows per chunk.

        Yields:
        -------
        Sequential chunks of the dataset.
        """
        raise NotImplementedError
    

    def save(self, data: Dict[str,np.ndarray], path: Path):
        """
        Persist processed NumPy arrays to disk. The provided dictionary of 
        arrays is first converted into a pandas DataFrame using 
        `_prepare_dataframe`, then written using the format-specific 
        `_write_dataframe` implementation.

        Raises:
        -------
        ValueError: If the input data dictionary is empty or results in an \
            empty DataFrame after preparation.
        """
        if not data: 
            raise ValueError("Cannot save empty data directory")

        df = self._prepare_dataframe(data)
        if df.empty: 
            raise ValueError("Prepared dataframe is empty")
        
        self._write_dataframe(df, path)
        print(f"Saved data to `{path}`")

    
    @abstractmethod
    def _write_dataframe(self, df: pd.DataFrame, path: Path):
        """
        Write a pandas DataFrame to disk in a format-specific manner.

        Subclasses must implement this method to define how DataFrames are
        serialized for their respective storage format.
        """
        raise NotImplementedError
    

    def _prepare_dataframe(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Convert a dictionary of NumPy arrays into a pandas DataFrame
        suitable for persistence.

        Behavior
        --------
        - 1D numeric arrays are converted directly to Python lists
        - Multi-dimensional arrays are serialized as nested lists
        - Object arrays containing structured data (e.g., lists, dicts)
        are JSON-encoded for safe storage
        - Empty arrays are preserved as empty columns

        This method ensures consistent and loss-minimized round-trip
        conversion between processed NumPy outputs and tabular storage
        formats.
        """
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
    """
    CSV file handler with streaming read support via PyArrow. Designed for 
    efficient loading of large CSV files using block-based reading and 
    threaded parsing. Writing is delegated to pandas for simplicity and 
    reliability.
    """
    # Average row size in bytes (estimate)
    ROW_SIZE        : Final[int] = 1024
    MIN_BLOCK_SIZE  : Final[int] = 1 << 20    # 1 Megabyte

    def load_chunks(self, path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Stream a CSV file in chunked DataFrame batches. PyArrow is used for 
        high-performance parsing and block-based reading. The effective block 
        size is dynamically derived from the requested chunk size and an 
        estimated average row size.

        Args:
        -----
        path : Location of the CSV file.
        chunk_size : Target number of rows per chunk (approximate).

        Yields:
        -------
        Non-empty DataFrame chunks.

        Raises:
        -------
        FileNotFoundError
            If the file does not exist.
        pyarrow.ArrowInvalid, pyarrow.ArrowTypeError
            If Arrow encounters parsing or type issues.
        Exception
            For unexpected read failures.
        """
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
        """
        Write a DataFrame to CSV format. This implementation delegates to 
        pandas' CSV writer for stable and widely compatible serialization.
        """
        df.to_csv(path)




class HandlerFactory:
    """
    Factory responsible for resolving file extensions to their corresponding 
    `FileHandler` implementations. This enables dynamic handler selection 
    based solely on file suffix, allowing new formats to be added by registering 
    them in the `HANDLERS` mapping.
    """
    HANDLERS: Dict[str,Type[FileHandler]] = {
        ".csv": CsvHandler
    }

    @classmethod
    def get_handler(cls, path: Union[str,Path]) -> FileHandler:
        """
        Return the appropriate file handler for a given file path. The handler 
        is selected based on the lowercase file suffix.

        Args:
        -----
        path : Target file path.

        Returns:
        --------
        FileHandler:
            Instantiated handler matching the file extension.

        Raises:
        -------
        ValueError:
            If the file has no extension or the extension is unsupported.
        """
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
