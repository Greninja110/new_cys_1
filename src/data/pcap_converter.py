"""
Convert PCAP files to CSV format for analysis.
"""

import os
import csv
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import pyshark
from scapy.all import rdpcap, PcapReader
import multiprocessing
from tqdm import tqdm
import threading
import queue

# Local imports
from ..utils.logger import setup_logger, log_function_call
from ..utils.config import get_config, get_nested_config

# Set up logger
logger = setup_logger(__name__)

class PCAPConverter:
    """Class to convert PCAP files to CSV with extracted features."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the PCAP converter.
        
        Args:
            config_path (str, optional): Path to the data configuration file.
        """
        if config_path is None:
            self.config = get_config('data_config')
        else:
            from ..utils.config import load_yaml_config
            self.config = load_yaml_config(config_path)
        
        self.batch_size = get_nested_config(self.config, 'data.pcap.batch_size', 1000)
        self.timeout = get_nested_config(self.config, 'data.pcap.timeout', 120)
        
        # Create directories if they don't exist
        os.makedirs(get_nested_config(self.config, 'data.processed_path.training'), exist_ok=True)
        os.makedirs(get_nested_config(self.config, 'data.processed_path.testing'), exist_ok=True)
    
    @log_function_call(logger)
    def convert_file(self, pcap_file: str, output_file: str, use_scapy: bool = False) -> str:
        """
        Convert a single PCAP file to CSV.
        
        Args:
            pcap_file (str): Path to the PCAP file.
            output_file (str): Path to the output CSV file.
            use_scapy (bool): Whether to use Scapy instead of PyShark.
            
        Returns:
            str: Path to the created CSV file.
        """
        logger.info(f"Converting PCAP file: {pcap_file} to CSV: {output_file}")
        
        if use_scapy:
            return self._convert_with_scapy(pcap_file, output_file)
        else:
            return self._convert_with_pyshark(pcap_file, output_file)
    
    def _convert_with_pyshark(self, pcap_file: str, output_file: str) -> str:
        """
        Convert PCAP to CSV using PyShark.
        
        Args:
            pcap_file (str): Path to the PCAP file.
            output_file (str): Path to the output CSV file.
            
        Returns:
            str: Path to the created CSV file.
        """
        try:
            # Open the pcap file
            capture = pyshark.FileCapture(pcap_file)
            
            # Define the CSV fieldnames
            fieldnames = [
                'timestamp', 'ip_src', 'ip_dst', 'src_port', 'dst_port',
                'protocol', 'packet_length', 'ttl', 'tcp_flags', 'tcp_window_size',
                'udp_length', 'icmp_type', 'http_method', 'dns_query'
            ]
            
            # Open the CSV file for writing
            with open(output_file, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                
                # Process each packet
                for packet_num, packet in enumerate(capture):
                    row = self._extract_packet_data_pyshark(packet)
                    if row:
                        writer.writerow(row)
                    
                    # Log progress
                    if packet_num > 0 and packet_num % self.batch_size == 0:
                        logger.info(f"Processed {packet_num} packets")
            
            logger.info(f"Successfully converted {pcap_file} to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error converting file {pcap_file}: {str(e)}", exc_info=True)
            raise
    
    def _convert_with_scapy(self, pcap_file: str, output_file: str) -> str:
        """
        Convert PCAP to CSV using Scapy.
        
        Args:
            pcap_file (str): Path to the PCAP file.
            output_file (str): Path to the output CSV file.
            
        Returns:
            str: Path to the created CSV file.
        """
        try:
            # Define the CSV fieldnames
            fieldnames = [
                'timestamp', 'ip_src', 'ip_dst', 'src_port', 'dst_port',
                'protocol', 'packet_length', 'ttl', 'tcp_flags', 'tcp_window_size',
                'udp_length', 'icmp_type'
            ]
            
            # Open the CSV file for writing
            with open(output_file, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                
                # Use PcapReader for more efficient streaming
                with PcapReader(pcap_file) as packets:
                    packet_count = 0
                    for packet in packets:
                        row = self._extract_packet_data_scapy(packet)
                        if row:
                            writer.writerow(row)
                        
                        packet_count += 1
                        if packet_count % self.batch_size == 0:
                            logger.info(f"Processed {packet_count} packets")
            
            logger.info(f"Successfully converted {pcap_file} to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error converting file {pcap_file}: {str(e)}", exc_info=True)
            raise
    
    def _extract_packet_data_pyshark(self, packet) -> Dict[str, Any]:
        """
        Extract relevant data from a packet using PyShark.
        
        Args:
            packet: PyShark packet object.
            
        Returns:
            Dict[str, Any]: Extracted packet data.
        """
        row = {field: '' for field in [
            'timestamp', 'ip_src', 'ip_dst', 'src_port', 'dst_port',
            'protocol', 'packet_length', 'ttl', 'tcp_flags', 'tcp_window_size',
            'udp_length', 'icmp_type', 'http_method', 'dns_query'
        ]}
        
        try:
            # Basic packet info
            row['timestamp'] = packet.sniff_time.timestamp()
            row['packet_length'] = packet.length
            
            # IP layer
            if hasattr(packet, 'ip'):
                row['ip_src'] = packet.ip.src
                row['ip_dst'] = packet.ip.dst
                row['ttl'] = packet.ip.ttl
                row['protocol'] = packet.ip.proto
            
            # Transport layer
            if hasattr(packet, 'tcp'):
                row['src_port'] = packet.tcp.srcport
                row['dst_port'] = packet.tcp.dstport
                row['tcp_flags'] = packet.tcp.flags
                row['tcp_window_size'] = packet.tcp.window_size
            elif hasattr(packet, 'udp'):
                row['src_port'] = packet.udp.srcport
                row['dst_port'] = packet.udp.dstport
                row['udp_length'] = packet.udp.length
            
            # ICMP
            if hasattr(packet, 'icmp'):
                row['icmp_type'] = packet.icmp.type
            
            # Application layer
            if hasattr(packet, 'http'):
                if hasattr(packet.http, 'request_method'):
                    row['http_method'] = packet.http.request_method
            
            # DNS
            if hasattr(packet, 'dns'):
                if hasattr(packet.dns, 'qry_name'):
                    row['dns_query'] = packet.dns.qry_name
            
            return row
            
        except Exception as e:
            logger.debug(f"Error extracting data from packet: {str(e)}")
            return row
    
    def _extract_packet_data_scapy(self, packet) -> Dict[str, Any]:
        """
        Extract relevant data from a packet using Scapy.
        
        Args:
            packet: Scapy packet object.
            
        Returns:
            Dict[str, Any]: Extracted packet data.
        """
        row = {field: '' for field in [
            'timestamp', 'ip_src', 'ip_dst', 'src_port', 'dst_port',
            'protocol', 'packet_length', 'ttl', 'tcp_flags', 'tcp_window_size',
            'udp_length', 'icmp_type'
        ]}
        
        try:
            # Basic packet info
            if hasattr(packet, 'time'):
                row['timestamp'] = packet.time
            row['packet_length'] = len(packet)
            
            # IP layer
            if packet.haslayer('IP'):
                ip = packet['IP']
                row['ip_src'] = ip.src
                row['ip_dst'] = ip.dst
                row['ttl'] = ip.ttl
                row['protocol'] = ip.proto
            
            # Transport layer
            if packet.haslayer('TCP'):
                tcp = packet['TCP']
                row['src_port'] = tcp.sport
                row['dst_port'] = tcp.dport
                row['tcp_flags'] = tcp.flags
                row['tcp_window_size'] = tcp.window
            elif packet.haslayer('UDP'):
                udp = packet['UDP']
                row['src_port'] = udp.sport
                row['dst_port'] = udp.dport
                row['udp_length'] = len(udp)
            
            # ICMP
            if packet.haslayer('ICMP'):
                icmp = packet['ICMP']
                row['icmp_type'] = icmp.type
            
            return row
            
        except Exception as e:
            logger.debug(f"Error extracting data from packet: {str(e)}")
            return row
    
    @log_function_call(logger)
    def convert_directory(self, pcap_dir: str, output_dir: str, use_scapy: bool = False, parallel: bool = True) -> List[str]:
        """
        Convert all PCAP files in a directory to CSV.
        
        Args:
            pcap_dir (str): Directory containing PCAP files.
            output_dir (str): Directory to save CSV files.
            use_scapy (bool): Whether to use Scapy instead of PyShark.
            parallel (bool): Whether to use parallel processing.
            
        Returns:
            List[str]: List of created CSV file paths.
        """
        if not os.path.exists(pcap_dir):
            raise FileNotFoundError(f"PCAP directory not found: {pcap_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all PCAP files in the directory
        pcap_files = [
            os.path.join(pcap_dir, f) for f in os.listdir(pcap_dir)
            if f.endswith(('.pcap', '.pcapng'))
        ]
        
        if not pcap_files:
            logger.warning(f"No PCAP files found in {pcap_dir}")
            return []
        
        logger.info(f"Found {len(pcap_files)} PCAP files in {pcap_dir}")
        
        # Process files
        output_files = []
        if parallel and len(pcap_files) > 1:
            output_files = self._parallel_convert(pcap_files, output_dir, use_scapy)
        else:
            for pcap_file in tqdm(pcap_files, desc="Converting PCAP files"):
                output_file = os.path.join(
                    output_dir, 
                    os.path.splitext(os.path.basename(pcap_file))[0] + '.csv'
                )
                try:
                    result = self.convert_file(pcap_file, output_file, use_scapy)
                    output_files.append(result)
                except Exception as e:
                    logger.error(f"Failed to convert {pcap_file}: {str(e)}")
        
        return output_files
    
    def _parallel_convert(self, pcap_files: List[str], output_dir: str, use_scapy: bool) -> List[str]:
        """
        Convert PCAP files in parallel.
        
        Args:
            pcap_files (List[str]): List of PCAP file paths.
            output_dir (str): Directory to save CSV files.
            use_scapy (bool): Whether to use Scapy instead of PyShark.
            
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
        for pcap_file in pcap_files:
            output_file = os.path.join(
                output_dir, 
                os.path.splitext(os.path.basename(pcap_file))[0] + '.csv'
            )
            tasks.append((pcap_file, output_file))
        
        # Worker function
        def worker():
            while True:
                try:
                    # Get a task from the queue
                    pcap_file, output_file = task_queue.get(block=False)
                    
                    # Process the task
                    try:
                        result = self.convert_file(pcap_file, output_file, use_scapy)
                        result_queue.put(result)
                    except Exception as e:
                        logger.error(f"Error converting {pcap_file}: {str(e)}")
                    
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
    def merge_csv_files(self, csv_files: List[str], output_file: str) -> str:
        """
        Merge multiple CSV files into one.
        
        Args:
            csv_files (List[str]): List of CSV file paths.
            output_file (str): Output CSV file path.
            
        Returns:
            str: Path to the merged CSV file.
        """
        if not csv_files:
            logger.warning("No CSV files to merge")
            return ""
        
        try:
            # Read the first file to get the header
            first_df = pd.read_csv(csv_files[0])
            first_df.to_csv(output_file, index=False)
            
            # Append other files
            for csv_file in tqdm(csv_files[1:], desc="Merging CSV files"):
                df = pd.read_csv(csv_file)
                df.to_csv(output_file, mode='a', header=False, index=False)
            
            logger.info(f"Successfully merged {len(csv_files)} CSV files into {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error merging CSV files: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    # Example usage
    converter = PCAPConverter()
    
    # Convert a single file
    # converter.convert_file("data/raw/example.pcap", "data/processed/example.csv")
    
    # Convert a directory
    # csv_files = converter.convert_directory("data/raw", "data/processed")
    
    # Merge CSV files
    # converter.merge_csv_files(csv_files, "data/processed/merged.csv")