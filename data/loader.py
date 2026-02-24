"""
data/loader.py
--------------
High-level dataset loading layer, this module defines the `DataLoader`,
responsible for:
    
    - Locating dataset files
    - Selecting the appropriate file handler
    - Loading data in chunks
    - Processing these chunks in parallel
    - Concatenating structured NumPy outputs
    - Validating dataset integrity
    - Providing reproducible train/validation/test splits

The loader abstracts file format handling and data transformation, exposing
a Numpy-based interface ready for modeling 
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from data.processors import DataProcessor
from data.handlers import HandlerFactory, FileHandler


class DataLoader:
    """
    High-level loader for the datasets used for training the UAV model.
    This model includes manages the supported file formats to load the
    data into the program applying the appropriate file-handler. 
    Thereafter, data-processing is managing all the transformations 
    from the loaded data from pd.DataFrame into dictionary of NumPy 
    arrays for simplistic manipulation.
    """
    REQUIRED_COLUMNS = [
        'dvec', 'rx_type', 'link_state', 'los_pl',
        'los_ang', 'los_dly', 'nlos_pl', 'nlos_ang', 'nlos_dly'
    ]

    def __init__(self,
        n_workers: Optional[int]=None,chunk_size: int=10_000, 
        prefer_processes: bool=False
    ):
        """
            Initialize Data-Loader instance
        """
        self.dir = Path(__file__).parent / "datasets"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.n_workers = n_workers or mp.cpu_count()
        self.chunk_size = max(100, chunk_size)
        self.prefer_processes = prefer_processes

        self.proc = DataProcessor()
    

    def load(self, paths: Union[str,List[str]]) -> Dict[str, np.ndarray]:
        if isinstance(paths,(str,Path)): 
            paths = [paths]
        
        print(f"Loading `{len(paths)}` file(s) with `{self.n_workers}` worker(s)...")

        # Choose executor based on configuration
        exe = ProcessPoolExecutor if self.prefer_processes else ThreadPoolExecutor
        chunks: List[Dict[str, NDArray[Any]]] = []

        for path in paths:
            path = Path(path) if Path(path).is_absolute() else self.dir / path

            if not path.exists():
                print(f"Error, `{path}` not found!")
                continue

            try: 
                handler: FileHandler = HandlerFactory.get_handler(path)

                with exe(max_workers=self.n_workers) as executor:

                    futures: List[Future[Dict[str, NDArray[Any]]]] = []
                    for chunk in handler.load_chunks(path, self.chunk_size):
                        missing = [
                            c for c in self.REQUIRED_COLUMNS if c not in chunk.columns
                        ]

                        if missing:
                            print(f"Warning: Missing column in `{path}`: {missing}")
                            continue

                        future = executor.submit(self.proc.process_chunk, chunk)
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=60)
                            if result: chunks.append(result)
                        
                        except Exception as e:
                            print(f"Chunk processing failed: {e}")
            
            except Exception as e:
                print(f"Failed to process `{path}`: {e}")
                continue
        
        if not chunks:
            raise RuntimeError("No data successfully were processed")
        
        # Concatenate all chunks
        processed = self.proc.concatenate(chunks)
        
        # Compute the number of rows
        rows = len(next(iter(processed.values()))) if processed else 0
        print(f"Successfully loaded `{rows}` rows from {len(paths)} file(s)")

        return processed
    
    def save(self,
        data: Dict[str,np.ndarray], path: Union[str,Path], fmt: Optional[str]=None
    ):
        """
        """
        path = Path(path)
        if fmt is None:
            handler = HandlerFactory.get_handler(path)
        
        else:

            if fmt not in HandlerFactory.HANDLERS:
                raise ValueError(f"Unsupported format: `{fmt}`")
            
            handler_class = HandlerFactory.HANDLERS[f".{fmt}"]
            handler = handler_class()
        
        handler.save(data, path)
    

    def validate_data(self, data: Dict[str, np.ndarray]) -> bool:
        """
        """
        if not data: 
            return False
        
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data]
        if missing:
            print(f"Missing required columns: {missing}")
            return False
        
        lengths = {key: len(value) for key, value in data.items()}
        unique_lengths = set(lengths.values())
        
        if len(unique_lengths) > 1:
            print(f"Inconsistent array lengths: {lengths}")
            return False
        
        return True


def shuffle_and_split(
    data: Dict[str, np.ndarray], val_ratio: float = 0.2, 
    test_ratio: float = 0.0, seed: int = 42
) -> Tuple[Dict[str, np.ndarray], ...]:
    """
    """
    if not 0 <= val_ratio <= 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if not 0 <= test_ratio <= 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    if val_ratio + test_ratio >= 1:
        raise ValueError(f"Sum of val_ratio and test_ratio must be < 1")
    
    # Get consistent length
    lengths = {len(value) for value in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent array lengths: {lengths}")
    
    n = next(iter(lengths))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    
    # Calculate split indices
    val_split = int(n * (1 - val_ratio - test_ratio))
    test_split = int(n * (1 - test_ratio)) if test_ratio > 0 else n
    
    train_idx = indices[:val_split]
    val_idx = indices[val_split:test_split]
    
    # Create splits
    train_data = {key: value[train_idx] for key, value in data.items()}
    val_data = {key: value[val_idx] for key, value in data.items()}
    
    if test_ratio > 0:
        test_idx = indices[test_split:]
        test_data = {key: value[test_idx] for key, value in data.items()}
        return train_data, val_data, test_data
    
    return train_data, val_data
