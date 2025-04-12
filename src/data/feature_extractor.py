"""
Extract features from processed CSV network traffic data.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import math
import ipaddress
from collections import defaultdict
import time

# Local imports
from ..utils.logger import setup_logger, log_function_call
from ..utils.config import get_config, get_nested_config

# Set up logger
logger = setup_logger(__name__)

class FeatureExtractor:
    """Extract features from network traffic data for model training."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config_path (str, optional): Path to the feature configuration file.
        """
        # Load configurations
        if config_path is None:
            self.config = get_config('feature_config')
        else:
            from ..utils.config import load_yaml_config
            self.config = load_yaml_config(config_path)
        
        # Get data config for file paths
        self.data_config = get_config('data_config')
        
        # Feature extraction settings
        self.window_size = get_nested_config(self.config, 'features.flow_features.window_size', 100)
        self.use_statistics = get_nested_config(self.config, 'features.flow_features.use_statistics', True)
        self.use_entropy = get_nested_config(self.config, 'features.flow_features.use_entropy', True)
        
        # Feature groups to extract
        self.feature_groups = get_nested_config(self.config, 'features.groups', 
            ['basic', 'time', 'flow', 'statistical', 'payload'])
        
        # Specific features to extract
        self.features_to_extract = get_nested_config(self.config, 'features.extract', [])
    
    @log_function_call(logger)
    def extract_features_from_file(self, csv_file: str, output_file: Optional[str] = None, 
                                  label: Optional[str] = None) -> pd.DataFrame:
        """
        Extract features from a CSV file.
        
        Args:
            csv_file (str): Path to the CSV file.
            output_file (str, optional): Path to save the feature CSV file.
            label (str, optional): Label for the data (normal, ddos, etc).
            
        Returns:
            pd.DataFrame: DataFrame with extracted features.
        """
        logger.info(f"Extracting features from CSV file: {csv_file}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Extract features
            features_df = self._extract_features(df, label)
            
            # Save to file if specified
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                features_df.to_csv(output_file, index=False)
                logger.info(f"Features saved to {output_file}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting features from {csv_file}: {str(e)}", exc_info=True)
            raise
    
    def _extract_features(self, df: pd.DataFrame, label: Optional[str] = None) -> pd.DataFrame:
        """
        Extract network traffic features from a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw network data.
            label (str, optional): Label for the data (normal, ddos, etc).
            
        Returns:
            pd.DataFrame: DataFrame with extracted features.
        """
        # Convert timestamp to datetime if it exists and isn't already
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                except Exception:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except Exception as e:
                        logger.warning(f"Could not convert timestamp column: {str(e)}")
        
        # Group by flows (IP src/dst and ports)
        flows = []
        feature_dicts = []
        
        # Determine if we have enough data for flow-based features
        if len(df) < 10:
            logger.warning(f"Not enough data for flow-based features, only {len(df)} rows")
            return pd.DataFrame()
        
        # Extract basic features that apply to each packet
        basic_features = self._extract_basic_features(df)
        
        # Extract flow-based features if requested
        if 'flow' in self.feature_groups:
            # Define what constitutes a flow
            if 'ip_src' in df.columns and 'ip_dst' in df.columns:
                # Group by source/destination IP and ports if available
                group_cols = ['ip_src', 'ip_dst']
                if 'src_port' in df.columns:
                    group_cols.append('src_port')
                if 'dst_port' in df.columns:
                    group_cols.append('dst_port')
                if 'protocol' in df.columns:
                    group_cols.append('protocol')
                
                # Extract flow features
                flow_features = self._extract_flow_features(df, group_cols)
                basic_features.update(flow_features)
        
        # Extract time-based features if requested
        if 'time' in self.feature_groups and 'timestamp' in df.columns:
            time_features = self._extract_time_features(df)
            basic_features.update(time_features)
        
        # Extract statistical features if requested
        if 'statistical' in self.feature_groups and self.use_statistics:
            statistical_features = self._extract_statistical_features(df)
            basic_features.update(statistical_features)
        
        # Add label if provided
        if label is not None:
            basic_features['label'] = label
        
        # Create a DataFrame with all features
        features_df = pd.DataFrame([basic_features])
        
        return features_df
    
    def _extract_basic_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract basic features from network traffic.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw network data.
            
        Returns:
            Dict[str, Any]: Dictionary of basic features.
        """
        features = {}
        
        # Packet counts
        features['total_packets'] = len(df)
        
        # Protocol distribution
        if 'protocol' in df.columns:
            protocol_counts = df['protocol'].value_counts(normalize=True).to_dict()
            for proto, count in protocol_counts.items():
                features[f'protocol_{proto}_ratio'] = count
        
        # TCP flags distribution if available
        if 'tcp_flags' in df.columns:
            # Handle different TCP flag formats
            tcp_flags = df['tcp_flags'].dropna()
            if len(tcp_flags) > 0:
                if tcp_flags.dtype == 'object':
                    # Parse string representations of flags
                    syn_count = sum(1 for f in tcp_flags if 'S' in str(f))
                    fin_count = sum(1 for f in tcp_flags if 'F' in str(f))
                    rst_count = sum(1 for f in tcp_flags if 'R' in str(f))
                    psh_count = sum(1 for f in tcp_flags if 'P' in str(f))
                    ack_count = sum(1 for f in tcp_flags if 'A' in str(f))
                else:
                    # Numeric flags representation
                    # SYN (0x02), FIN (0x01), RST (0x04), PSH (0x08), ACK (0x10)
                    syn_count = sum(1 for f in tcp_flags if f & 0x02)
                    fin_count = sum(1 for f in tcp_flags if f & 0x01)
                    rst_count = sum(1 for f in tcp_flags if f & 0x04)
                    psh_count = sum(1 for f in tcp_flags if f & 0x08)
                    ack_count = sum(1 for f in tcp_flags if f & 0x10)
                
                # Calculate ratios
                total_tcp = len(tcp_flags)
                if total_tcp > 0:
                    features['syn_ratio'] = syn_count / total_tcp
                    features['fin_ratio'] = fin_count / total_tcp
                    features['rst_ratio'] = rst_count / total_tcp
                    features['psh_ratio'] = psh_count / total_tcp
                    features['ack_ratio'] = ack_count / total_tcp
        
        # Packet length statistics if available
        if 'packet_length' in df.columns:
            packet_lengths = df['packet_length'].dropna()
            if len(packet_lengths) > 0:
                features['packet_length_mean'] = packet_lengths.mean()
                features['packet_length_std'] = packet_lengths.std()
                features['packet_length_min'] = packet_lengths.min()
                features['packet_length_max'] = packet_lengths.max()
                features['packet_length_median'] = packet_lengths.median()
                
                # Calculate percentiles
                percentiles = [25, 75, 90, 95]
                for p in percentiles:
                    features[f'packet_length_percentile_{p}'] = np.percentile(packet_lengths, p)
        
        # TTL statistics if available
        if 'ttl' in df.columns:
            ttl_values = df['ttl'].dropna()
            if len(ttl_values) > 0:
                features['ttl_mean'] = ttl_values.mean()
                features['ttl_std'] = ttl_values.std()
                features['ttl_min'] = ttl_values.min()
                features['ttl_max'] = ttl_values.max()
        
        # Port statistics if available
        if 'src_port' in df.columns:
            src_ports = df['src_port'].dropna()
            if len(src_ports) > 0:
                features['unique_src_ports'] = src_ports.nunique()
                
                # Well-known ports (0-1023)
                well_known_src = sum(1 for p in src_ports if 0 <= p <= 1023)
                features['well_known_src_port_ratio'] = well_known_src / len(src_ports) if len(src_ports) > 0 else 0
        
        if 'dst_port' in df.columns:
            dst_ports = df['dst_port'].dropna()
            if len(dst_ports) > 0:
                features['unique_dst_ports'] = dst_ports.nunique()
                
                # Well-known ports (0-1023)
                well_known_dst = sum(1 for p in dst_ports if 0 <= p <= 1023)
                features['well_known_dst_port_ratio'] = well_known_dst / len(dst_ports) if len(dst_ports) > 0 else 0
        
        # IP statistics if available
        if 'ip_src' in df.columns and 'ip_dst' in df.columns:
            src_ips = df['ip_src'].dropna()
            dst_ips = df['ip_dst'].dropna()
            
            if len(src_ips) > 0 and len(dst_ips) > 0:
                features['unique_src_ips'] = src_ips.nunique()
                features['unique_dst_ips'] = dst_ips.nunique()
                
                # IP entropy if requested
                if self.use_entropy:
                    features['entropy_src_ip'] = self._calculate_entropy(src_ips)
                    features['entropy_dst_ip'] = self._calculate_entropy(dst_ips)
        
        return features
    
    def _extract_flow_features(self, df: pd.DataFrame, group_cols: List[str]) -> Dict[str, Any]:
        """
        Extract flow-based features from network traffic.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw network data.
            group_cols (List[str]): Columns to group by for flow identification.
            
        Returns:
            Dict[str, Any]: Dictionary of flow features.
        """
        features = {}
        
        try:
            # Group by flow
            flows = df.groupby(group_cols)
            
            # Number of flows
            features['flow_count'] = len(flows)
            
            # Packets per flow
            packets_per_flow = flows.size().values
            if len(packets_per_flow) > 0:
                features['packets_per_flow_mean'] = np.mean(packets_per_flow)
                features['packets_per_flow_std'] = np.std(packets_per_flow)
                features['packets_per_flow_max'] = np.max(packets_per_flow)
                features['packets_per_flow_min'] = np.min(packets_per_flow)
            
            # Flow duration if timestamp available
            if 'timestamp' in df.columns:
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    except Exception:
                        try:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        except Exception as e:
                            logger.warning(f"Could not convert timestamp column for flow duration: {str(e)}")
                            return features
                
                # Calculate flow durations
                flow_durations = []
                flows_with_duration = 0
                
                for name, group in flows:
                    if len(group) > 1:
                        min_time = group['timestamp'].min()
                        max_time = group['timestamp'].max()
                        duration = (max_time - min_time).total_seconds()
                        
                        if duration > 0:
                            flow_durations.append(duration)
                            flows_with_duration += 1
                
                if flows_with_duration > 0:
                    features['flow_duration_mean'] = np.mean(flow_durations)
                    features['flow_duration_std'] = np.std(flow_durations) if len(flow_durations) > 1 else 0
                    features['flow_duration_max'] = np.max(flow_durations)
                    features['flow_duration_min'] = np.min(flow_durations)
                    
                    # Flow rate (packets per second)
                    packets_per_second = []
                    for name, group in flows:
                        if len(group) > 1:
                            min_time = group['timestamp'].min()
                            max_time = group['timestamp'].max()
                            duration = (max_time - min_time).total_seconds()
                            
                            if duration > 0:
                                pps = len(group) / duration
                                packets_per_second.append(pps)
                    
                    if len(packets_per_second) > 0:
                        features['packets_per_second_mean'] = np.mean(packets_per_second)
                        features['packets_per_second_std'] = np.std(packets_per_second) if len(packets_per_second) > 1 else 0
                        features['packets_per_second_max'] = np.max(packets_per_second)
                        features['packets_per_second_min'] = np.min(packets_per_second)
            
            # Bytes per flow if packet length available
            if 'packet_length' in df.columns:
                bytes_per_flow = flows['packet_length'].sum().values
                if len(bytes_per_flow) > 0:
                    features['bytes_per_flow_mean'] = np.mean(bytes_per_flow)
                    features['bytes_per_flow_std'] = np.std(bytes_per_flow)
                    features['bytes_per_flow_max'] = np.max(bytes_per_flow)
                    features['bytes_per_flow_min'] = np.min(bytes_per_flow)
                    
                    # Bytes per second if timestamp available
                    if 'timestamp' in df.columns and flows_with_duration > 0:
                        bytes_per_second = []
                        for name, group in flows:
                            if len(group) > 1:
                                min_time = group['timestamp'].min()
                                max_time = group['timestamp'].max()
                                duration = (max_time - min_time).total_seconds()
                                
                                if duration > 0:
                                    bps = group['packet_length'].sum() / duration
                                    bytes_per_second.append(bps)
                        
                        if len(bytes_per_second) > 0:
                            features['bytes_per_second_mean'] = np.mean(bytes_per_second)
                            features['bytes_per_second_std'] = np.std(bytes_per_second) if len(bytes_per_second) > 1 else 0
                            features['bytes_per_second_max'] = np.max(bytes_per_second)
                            features['bytes_per_second_min'] = np.min(bytes_per_second)
        
        except Exception as e:
            logger.warning(f"Error extracting flow features: {str(e)}")
        
        return features
    
    def _extract_time_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract time-based features from network traffic.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw network data.
            
        Returns:
            Dict[str, Any]: Dictionary of time features.
        """
        features = {}
        
        if 'timestamp' not in df.columns:
            return features
        
        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                except Exception:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except Exception as e:
                        logger.warning(f"Could not convert timestamp column for time features: {str(e)}")
                        return features
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Total time span
            time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            features['time_span'] = time_span
            
            if time_span > 0:
                # Packets per second overall
                features['packets_per_second'] = len(df) / time_span
                
                # Bytes per second overall if packet length available
                if 'packet_length' in df.columns:
                    features['bytes_per_second'] = df['packet_length'].sum() / time_span
                
                # Inter-arrival times
                inter_arrival_times = np.diff(df['timestamp'].view(np.int64)) / 1e9  # Convert to seconds
                if len(inter_arrival_times) > 0:
                    features['inter_arrival_time_mean'] = np.mean(inter_arrival_times)
                    features['inter_arrival_time_std'] = np.std(inter_arrival_times)
                    features['inter_arrival_time_max'] = np.max(inter_arrival_times)
                    features['inter_arrival_time_min'] = np.min(inter_arrival_times)
                    
                    # Variance or coefficient of variation of inter-arrival times
                    features['inter_arrival_time_cv'] = (features['inter_arrival_time_std'] / 
                                                         features['inter_arrival_time_mean'] 
                                                         if features['inter_arrival_time_mean'] > 0 else 0)
                
                # Time-window based features
                if len(df) > self.window_size:
                    # Packet count in windows
                    windows = [df.iloc[i:i+self.window_size] for i in range(0, len(df), self.window_size)]
                    
                    if windows:
                        # Time duration of windows
                        window_durations = [(w['timestamp'].max() - w['timestamp'].min()).total_seconds() 
                                           for w in windows if len(w) > 1]
                        
                        if window_durations:
                            # Packets per second in windows
                            packets_per_second_windows = [len(w) / d if d > 0 else 0 
                                                         for w, d in zip(windows, window_durations)]
                            
                            features['window_packets_per_second_mean'] = np.mean(packets_per_second_windows)
                            features['window_packets_per_second_std'] = np.std(packets_per_second_windows)
                            features['window_packets_per_second_max'] = np.max(packets_per_second_windows)
                            
                            # Bytes per second in windows if packet length available
                            if 'packet_length' in df.columns:
                                bytes_per_second_windows = [w['packet_length'].sum() / d if d > 0 else 0 
                                                           for w, d in zip(windows, window_durations)]
                                
                                features['window_bytes_per_second_mean'] = np.mean(bytes_per_second_windows)
                                features['window_bytes_per_second_std'] = np.std(bytes_per_second_windows)
                                features['window_bytes_per_second_max'] = np.max(bytes_per_second_windows)
        
        except Exception as e:
            logger.warning(f"Error extracting time features: {str(e)}")
        
        return features
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract statistical features from network traffic.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw network data.
            
        Returns:
            Dict[str, Any]: Dictionary of statistical features.
        """
        features = {}
        
        try:
            # Entropy of ports, protocols, IP addresses
            for col in ['protocol', 'src_port', 'dst_port', 'ip_src', 'ip_dst', 'ttl']:
                if col in df.columns:
                    features[f'entropy_{col}'] = self._calculate_entropy(df[col])
            
            # Entropy of packet lengths if available
            if 'packet_length' in df.columns:
                # Discretize packet lengths into bins for entropy calculation
                bins = 10
                df['packet_length_bin'] = pd.cut(df['packet_length'], bins)
                features['entropy_packet_length'] = self._calculate_entropy(df['packet_length_bin'])
            
            # Distribution of packet directions (src->dst vs dst->src)
            if 'ip_src' in df.columns and 'ip_dst' in df.columns:
                # Create a set of all IPs
                all_ips = set(df['ip_src'].unique()) | set(df['ip_dst'].unique())
                
                # Count packets in each direction for each IP pair
                ip_pairs = defaultdict(lambda: {'forward': 0, 'backward': 0})
                
                for _, row in df.iterrows():
                    src = row['ip_src']
                    dst = row['ip_dst']
                    
                    # Determine the canonical direction by lexicographic ordering
                    if src < dst:
                        forward = (src, dst)
                        ip_pairs[forward]['forward'] += 1
                    else:
                        backward = (dst, src)
                        ip_pairs[backward]['backward'] += 1
                
                # Calculate ratios of packets in each direction
                direction_ratios = []
                for pair, counts in ip_pairs.items():
                    total = counts['forward'] + counts['backward']
                    if total > 0:
                        ratio = max(counts['forward'], counts['backward']) / total
                        direction_ratios.append(ratio)
                
                if direction_ratios:
                    features['direction_ratio_mean'] = np.mean(direction_ratios)
                    features['direction_ratio_min'] = np.min(direction_ratios)
                    features['direction_ratio_max'] = np.max(direction_ratios)
            
            # Distributions of flags or other categorical fields
            for col in ['tcp_flags', 'icmp_type', 'http_method']:
                if col in df.columns:
                    col_values = df[col].dropna()
                    if len(col_values) > 0:
                        # Calculate distribution
                        value_counts = col_values.value_counts(normalize=True)
                        
                        # Get top values
                        top_values = value_counts.head(5)
                        for val, count in top_values.items():
                            features[f'{col}_{val}_ratio'] = count
                
        except Exception as e:
            logger.warning(f"Error extracting statistical features: {str(e)}")
        
        return features
    
    def _calculate_entropy(self, data: pd.Series) -> float:
        """
        Calculate Shannon entropy of a series.
        
        Args:
            data (pd.Series): Input data series.
            
        Returns:
            float: Shannon entropy value.
        """
        if len(data) == 0:
            return 0
        
        # Count frequency of values
        value_counts = data.value_counts(normalize=True)
        
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in value_counts if p > 0)
        
        return entropy
    
    @log_function_call(logger)
    def extract_features_batch(self, csv_files: List[Tuple[str, str]], output_dir: str) -> List[str]:
        """
        Extract features from multiple CSV files.
        
        Args:
            csv_files (List[Tuple[str, str]]): List of (file_path, label) tuples.
            output_dir (str): Directory to save feature CSV files.
            
        Returns:
            List[str]: List of output feature file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        for csv_file, label in tqdm(csv_files, desc="Extracting features"):
            output_file = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(csv_file))[0]}_features.csv"
            )
            
            try:
                self.extract_features_from_file(csv_file, output_file, label)
                output_files.append(output_file)
            except Exception as e:
                logger.error(f"Failed to extract features from {csv_file}: {str(e)}")
        
        return output_files
    
    @log_function_call(logger)
    def combine_feature_files(self, feature_files: List[str], output_file: str) -> str:
        """
        Combine multiple feature files into one dataset.
        
        Args:
            feature_files (List[str]): List of feature file paths.
            output_file (str): Path to save the combined dataset.
            
        Returns:
            str: Path to the combined feature file.
        """
        if not feature_files:
            logger.warning("No feature files to combine")
            return ""
        
        try:
            # Read first file to get columns
            combined_df = pd.read_csv(feature_files[0])
            
            # Append other files
            for file in tqdm(feature_files[1:], desc="Combining feature files"):
                df = pd.read_csv(file)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            
            # Save combined file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"Combined {len(feature_files)} feature files into {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error combining feature files: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Extract features from a single file
    # features_df = extractor.extract_features_from_file(
    #     "data/processed/example.csv", 
    #     "data/processed/example_features.csv",
    #     "normal"
    # )
    
    # Extract features from multiple files
    # csv_files = [
    #     ("data/processed/normal.csv", "normal"),
    #     ("data/processed/ddos.csv", "ddos")
    # ]
    # feature_files = extractor.extract_features_batch(csv_files, "data/processed/features")
    
    # Combine feature files
    # combined_file = extractor.combine_feature_files(
    #     feature_files, 
    #     "data/processed/combined_features.csv"
    # )