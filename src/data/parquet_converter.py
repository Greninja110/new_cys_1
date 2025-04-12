"""
Convert Parquet files to CSV format for analysis.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import fastparquet as fp
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import multiprocessing
import threading
import queue

# Local imports
from ..utils.logger import setup_logger, log_function_call
from ..utils.config import get_config, get_nested_config

# Set up logger
logger = setup_logger(__name__)


class ParquetConverter:
    """Class to convert Parquet files to CSV with preprocessing."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Parquet converter.
        
        Args:
            config_path (str, optional): Path to the data configuration file.
        """
        if config_path is None:
            self.config = get_config('data_config')
        else:
            from ..utils.config import load_yaml_config
            self.config = load_yaml_config(config_path)
        
        self.chunk_size = get_nested_config(self.config, 'data.parquet.chunk_size', 100000)
        
        # Create directories if they don't exist
        os.makedirs(get_nested_config(self.config, 'data.processed_path.training'), exist_ok=True)
        os.makedirs(get_nested_config(self.config, 'data.processed_path.testing'), exist_ok=True)
    
    @log_function_call(logger)
    def convert_file(self, parquet_file: str, output_file: str, use_fastparquet: bool = False) -> str:
        """
        Convert a single Parquet file to CSV.
        
        Args:
            parquet_file (str): Path to the Parquet file.
            output_file (str): Path to the output CSV file.
            use_fastparquet (bool): Whether to use fastparquet instead of pyarrow.
            
        Returns:
            str: Path to the created CSV file.
        """
        logger.info(f"Converting Parquet file: {parquet_file} to CSV: {output_file}")
        
        try:
            if use_fastparquet:
                return self._convert_with_fastparquet(parquet_file, output_file)
            else:
                return self._convert_with_pyarrow(parquet_file, output_file)
        except Exception as e:
            logger.error(f"Error converting file {parquet_file}: {str(e)}", exc_info=True)
            raise
    
    def _convert_with_pyarrow(self, parquet_file: str, output_file: str) -> str:
        """
        Convert Parquet to CSV using PyArrow.
        
        Args:
            parquet_file (str): Path to the Parquet file.
            output_file (str): Path to the output CSV file.
            
        Returns:
            str: Path to the created CSV file.
        """
        try:
            # Open the parquet file
            parquet_table = pq.read_table(parquet_file)
            
            # Get the number of rows for progress tracking
            num_rows = parquet_table.num_rows
            logger.info(f"Parquet file contains {num_rows} rows")
            
            # Convert to pandas and save as CSV
            if num_rows > self.chunk_size:
                # Process in chunks
                with open(output_file, 'w') as f:
                    # Write header first
                    headers = ','.join(parquet_table.column_names) + '\n'
                    f.write(headers)
                
                # Process and append chunks
                for i in tqdm(range(0, num_rows, self.chunk_size), desc="Processing chunks"):
                    end = min(i + self.chunk_size, num_rows)
                    chunk = parquet_table.slice(i, end - i).to_pandas()
                    
                    # Append to CSV without header
                    chunk.to_csv(output_file, mode='a', header=False, index=False)
                    
                    logger.info(f"Processed chunk {i//self.chunk_size + 1} ({i}-{end})")
            else:
                # Small enough to process at once
                df = parquet_table.to_pandas()
                df.to_csv(output_file, index=False)
            
            logger.info(f"Successfully converted {parquet_file} to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error converting file with PyArrow: {str(e)}", exc_info=True)
            raise
    
    def _convert_with_fastparquet(self, parquet_file: str, output_file: str) -> str:
        """
        Convert Parquet to CSV using fastparquet.
        
        Args:
            parquet_file (str): Path to the Parquet file.
            output_file (str): Path to the output CSV file.
            
        Returns:
            str: Path to the created CSV file.
        """
        try:
            # Open the parquet file
            pf = fp.ParquetFile(parquet_file)
            
            # Get the number of rows for progress tracking
            # Note: fastparquet doesn't have a direct way to get row count
            # We'll use the number of row groups as an approximation
            num_row_groups = len(pf.row_groups)
            logger.info(f"Parquet file contains {num_row_groups} row groups")
            
            # Get column names
            df_temp = pf.to_pandas(rows=1)
            column_names = df_temp.columns.tolist()
            
            # Process row groups
            header_written = False
            for i in tqdm(range(num_row_groups), desc="Processing row groups"):
                # Read row group
                df = pf.read_row_group(i)
                
                # Save to CSV
                if not header_written:
                    df.to_csv(output_file, index=False)
                    header_written = True
                else:
                    df.to_csv(output_file, mode='a', header=False, index=False)
                
                logger.info(f"Processed row group {i+1}/{num_row_groups}")
            
            logger.info(f"Successfully converted {parquet_file} to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error converting file with fastparquet: {str(e)}", exc_info=True)
            raise
    
    @log_function_call(logger)
    def convert_directory(self, parquet_dir: str, output_dir: str, use_fastparquet: bool = False, parallel: bool = True) -> List[str]:
        """
        Convert all Parquet files in a directory to CSV.
        
        Args:
            parquet_dir (str): Directory containing Parquet files.
            output_dir (str): Directory to save CSV files.
            use_fastparquet (bool): Whether to use fastparquet instead of pyarrow.
            parallel (bool): Whether to use parallel processing.
            
        Returns:
            List[str]: List of created CSV file paths.
        """
        if not os.path.exists(parquet_dir):
            raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all Parquet files in the directory
        parquet_files = [
            os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir)
            if f.endswith('.parquet')
        ]
        
        if not parquet_files:
            logger.warning(f"No Parquet files found in {parquet_dir}")
            return []
        
        logger.info(f"Found {len(parquet_files)} Parquet files in {parquet_dir}")
        
        # Process files
        output_files = []
        if parallel and len(parquet_files) > 1:
            output_files = self._parallel_convert(parquet_files, output_dir, use_fastparquet)
        else:
            for parquet_file in tqdm(parquet_files, desc="Converting Parquet files"):
                output_file = os.path.join(
                    output_dir, 
                    os.path.splitext(os.path.basename(parquet_file))[0] + '.csv'
                )
                try:
                    result = self.convert_file(parquet_file, output_file, use_fastparquet)
                    output_files.append(result)
                except Exception as e:
                    logger.error(f"Failed to convert {parquet_file}: {str(e)}")
        
        return output_files
    
    def _parallel_convert(self, parquet_files: List[str], output_dir: str, use_fastparquet: bool) -> List[str]:
        """
        Convert Parquet files in parallel.
        
        Args:
            parquet_files (List[str]): List of Parquet file paths.
            output_dir (str): Directory to save CSV files.
            use_fastparquet (bool): Whether to use fastparquet instead of pyarrow.
            
        Returns:
            List[str]: List of created CSV file paths.
        """
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {num_workers} workers for parallel conversion")
        
        # Create output queue and list
        result_queue = queue.Queue()
        output_files = []
        
        # Create conversion tasks
        tasks = []
        for parquet_file in parquet_files:
            output_file = os.path.join(
                output_dir, 
                os.path.splitext(os.path.basename(parquet_file))[0] + '.csv'
            )
            tasks.append((parquet_file, output_file))
        
        # Worker function
        def worker():
            while True:
                try:
                    # Get a task from the queue
                    parquet_file, output_file = task_queue.get(block=False)
                    
                    # Process the task
                    try:
                        result = self.convert_file(parquet_file, output_file, use_fastparquet)
                        result_queue.put(result)
                    except Exception as e:
                        logger.error(f"Error converting {parquet_file}: {str(e)}")
                    
                    # Mark the task as done
                    task_queue.task_done()
                    
                except queue.Empty:
                    break
        
        # Create task queue and add tasks
        task_queue = queue.Queue()
        for task in tasks:
            task_queue.put(task)
        
        # Create and start worker threads
        threads = []
        for _ in range(num_workers):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for all tasks to complete
        task_queue.join()
        
        # Get results
        while not result_queue.empty():
            output_files.append(result_queue.get())
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        
        return output_files
    
    @log_function_call(logger)
    def extract_csv_from_excel(self, excel_file: str, output_file: str) -> str:
        """
        Extract data from Excel file (XLSX, CSV) to CSV.
        
        Args:
            excel_file (str): Path to the Excel file.
            output_file (str): Path to the output CSV file.
            
        Returns:
            str: Path to the created CSV file.
        """
        logger.info(f"Extracting data from Excel file: {excel_file} to CSV: {output_file}")
        
        try:
            # Read excel file
            if excel_file.endswith('.xlsx') or excel_file.endswith('.xls'):
                df = pd.read_excel(excel_file)
            elif excel_file.endswith('.csv'):
                df = pd.read_csv(excel_file)
            else:
                raise ValueError(f"Unsupported file format: {excel_file}")
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            
            logger.info(f"Successfully extracted data from {excel_file} to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error extracting data from Excel file: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    # Example usage
    converter = ParquetConverter()
    
    # Convert a single file
    # converter.convert_file("data/raw/example.parquet", "data/processed/example.csv")
    
    # Convert a directory
    # csv_files = converter.convert_directory("data/raw", "data/processed")