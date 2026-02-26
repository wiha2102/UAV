"""
data/processors.py
------------------
Data validation and transformation layer in loading the datasets.

This module defines the `DataProcessor`, responsible for transforming raw 
`pd.DataFrame` chunks into structured and type-consistent `NumPy` arrays
according to a predefined schema.

- Validate expected column in the data
- Enforce data types (dtype) consistency
- Decode nested / stacked structures (including JSON-encoded arrays)
- Handle malformed or ragged data softly
- Provide safe concatenated across multiple processed chunks

The processors ensures that downstream components receive clean, 
well-structured `NumPy` arrays  independent oof the original file format.
"""
from __future__ import annotations

import orjson
import numpy as np
import pandas as pd

from typing import Any, cast, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from numpy.typing import DTypeLike, NDArray


class DataProcessor:
    """
    Schema-driven transformation layer for tabular dataset ingestion. The
    `DataProcessor` converts the raw `pandas.DataFrame` chunks into structured,
    dtype-consistent `NumPy` arrays according to a predefined schema.

        - Validate presence of expected columns
        - Enforce strict dtype casting where possible
        - Decode nested / stacked structured (including JSON-encoded arrays)
        - Gracefully handle of malformed, missing or ragged data.
        - Prove safe concatenated across multiple processed chunks.

    The processor guarantees that downstream components receive clean, shape
    consistent `NumPy`arrays independent of the original file format or
    serialization details,  
    """
    SCHEMA = {
        "dvec"      : {"dtype": np.float32, "stacked": True, "dim": 3},
        "rx_type"   : {"dtype": np.uint8,   "stacked": False},
        "link_state": {"dtype": np.uint8,   "stacked": False},
        "los_pl"    : {"dtype": np.float32, "stacked": False},
        "los_ang"   : {"dtype": np.float32, "stacked": True, "dim": 4},
        "los_dly"   : {"dtype": np.float32, "stacked": False},
        "nlos_pl"   : {"dtype": np.float32, "stacked": True, "dim": 20},
        "nlos_ang"  : {"dtype": np.float32, "stacked": True, "dim": (20, 4)},
        "nlos_dly"  : {"dtype": np.float32, "stacked": True, "dim": 20},
    }

    def __init__(self):
        """
            Initialize Data-Processors Instance
        """
        self._dtype_cache: Dict[str,np.dtype] = {}
    
    def process_chunk(self, chunk: pd.DataFrame) -> Dict[str, NDArray[Any]]:
        """
        Transform a raw DataFrame chunk into schema-compliant NumPy arrays.
        Each column defined in `SCHEMA` is validated and converted according to
        following features

        - Expected dtype
        - Whether the column contains stacked (nested) structures
        - Required decoding (e.g., JSON arrays)

        Malformed or incompatible columns are safely coerced to `object` dtype 
        instead of raising, ensuring that ingestion pipelines remain fault
        tolerant.

        Args:
        -----
        chunk: A raw batch of tabular data.

        Returns:
        --------
        Dictionary mapping column names to processed NumPy arrays.
        """
        pc: Dict[str,NDArray[Any]] = {}
        for column, spec in self.SCHEMA.items():
            
            if column not in chunk.columns: 
                continue

            dtype: DTypeLike = spec["dtype"]
            stacked: bool = spec.get("stacked", False)
            values: NDArray[Any] = chunk[column].to_numpy(copy=False)
            
            if values.size == 0:
                pc[column] = np.empty((0,), dtype=dtype)
                continue
            
            try:
                if not stacked:
                    pc[column] = self._convert_simple_column(values, dtype, column)
                else:
                    pc[column] = self._convert_stacked_column(values, dtype, column)
            
            except Exception as e:
                print(f"Warning: Failed to process column: '{column}', dtype=object")
                pc[column] = np.asarray(values, dtype=object)
        
        return pc


    def _convert_simple_column(self,
        values: NDArray[Any], dtype: DTypeLike, column: str
    ) -> NDArray[Any]:
        """
        Convert a non-stacked (scalar) column to the target dtype. If casting 
        fails due to invalid or non-convertible values, the column is coerced 
        to `object` dtype and processing continues without raising. This method 
        ensures robustness while preserving as much type integrity as possible.
        """
        try:  
            return values.astype(dtype, copy=False)
        
        except (ValueError, TypeError):
            print(f"Column `{column}` contains non-castable values; dtype=object")
            return np.asarray(values, dtype=object)


    def _convert_stacked_column(self,
        values: NDArray[Any], dtype: DTypeLike, column: str
    ) -> NDArray[Any]:
        """
        Convert a stacked (nested) column into a structured NumPy array. 
        Stacked columns may contain following features

        - Python lists / arrays
        - JSON-encoded arrays (strings)
        - Potentially ragged or malformed entries

        The method detects the structure from the first valid sample and
        applies the appropriate decoding or casting strategy. If shape or
        dtype consistency cannot be guaranteed, the column safely falls back
        to `object` dtype.
        """
        sample = self._first_valid(values)
        if sample is None: 
            return np.asarray(values, dtype=object)
        
        if isinstance(sample, str):
            return self._decode_json_array(values, dtype, column)
        
        try: 
            return np.asarray(values.tolist(), dtype=dtype)
        
        except (ValueError, TypeError):
            print(f"Column `{column}` contains ragged arrays; fallback to object")
            return np.asarray(values, dtype=object)
    

    # def _decode_json_array(self,
    #         values: NDArray[Any], dtype: DTypeLike, column: str
    # ) -> NDArray[Any]:
    #     """ Decode JSON arrays from string values """
    #     try: 
    #         return np.asarray([
    #             orjson.loads(v) if v is not None else None for v in values
    #     ], dtype=dtype)
        
    #     except Exception:
    #         print(f"Column `{column}` contains malformed JSON; dtype=object")
    #         return np.array([self._safe_parse(v) for v in values], dtype=object)

    def _decode_json_array(self,
            values: NDArray[Any], dtype: DTypeLike, column: str
    ) -> NDArray[Any]:
        """
        Decode JSON-encoded array elements within a stacked column. Each string 
        entry is parsed using `orjson` and converted into a NumPy array of the 
        target dtype. Non-string or missing values are preserved as `None`. If 
        any decoding error occurs (e.g., malformed JSON), the entire column is 
        safely coerced to `object` dtype to prevent pipeline interruption.
        """
        try: 
            dv: List[Any] = []
            for v in values:

                if v is not None and isinstance(v, str):
                    dv.append(orjson.loads(v))
                
                else:
                    dv.append(None)
            
            return np.asarray(dv, dtype=dtype)
        
        except Exception:
            print(f"Column `{column}` contains malformed JSON; coercing to dtype=object")
            return np.array([self._safe_parse(v) for v in values], dtype=object)


    @staticmethod
    def _first_valid(values: NDArray[Any]) -> Optional[Any]:
        """
        Return the first non-null, non-NaN element from an array. Used to infer 
        the structural type of a stacked column (e.g., list vs. JSON string) 
        before selecting the appropriate conversion strategy.

        Returns:
        --------
        The first valid value, or None if no such value exists.
        """
        if values.dtype != object:
            return values[0] if values.size > 0 else None

        for value in values:

            if value is not None and not pd.isna(value):
                return value

        return None


    def concatenate(self,results: List[Dict[str,NDArray[Any]]]) -> Dict[str,NDArray[Any]]:
        """
        Safely concatenate multiple processed chunk results. Each key present 
        in the first result dictionary is concatenated across all chunks along 
        axis 0. If dtype or shape incompatibilities prevent standard 
        concatenation, the method falls back to concatenation with `object` 
        dtype to preserve data while avoiding failure.

        Args:
        -----
        results: List of processed chunk outputs.

        Returns
        -------
        Dictionary of concatenated arrays per column.
        """
        if not results: 
            return {
                key: np.empty((0,), dtype=cast(DTypeLike, spec["dtype"])) 
                for key, spec in self.SCHEMA.items()
            }

        out: Dict[str, NDArray[Any]] = {}
        fr = results[0]

        for key in fr.keys():
            
            arr: List[NDArray[Any]] = []
            for result in results:
                
                if key in result:
                    arr.append(result[key])
            
            if not arr: 
                continue
            
            try:
                out[key] = np.concatenate(arr, axis=0)
            
            except (ValueError, TypeError):
                print(f"Concatenate failed for `{key}`; using object dtype")
                out[key] = np.concatenate([np.asarray(a, dtype=object) for a in arr], axis=0)
        
        return out
    

    def _safe_parse(self, value: Any) -> Any:
        """
        Attempt to safely parse a value that may contain JSON-encoded data.

        Behavior
        --------
        - Returns None for null or NaN values
        - Attempts JSON decoding for string inputs
        - Returns the original value if decoding fails
        - Returns non-string values unchanged

        This utility is used as a defensive fallback during malformed JSON
        handling.
        """
        if value is None or pd.isna(value):
            return None
        
        if isinstance(value, str):
            try:
                return orjson.loads(value)
            except:
                return value
        
        return value
