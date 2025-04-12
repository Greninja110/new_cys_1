"""
End-to-end inference pipeline for network traffic classifier.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Local imports
from ..utils.logger import setup_logger, log_function_call
from ..utils.config import get_config, get_nested_config
from ..data.pcap_converter import PCAPConverter
from ..data.feature_extractor import FeatureExtractor
from ..models.model_loader import ModelLoader

# Set up logger
logger = setup_logger(__name__)


class InferencePipeline:
    """End-to-end pipeline for network traffic classification inference."""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path (str, optional): Path to the saved model.
            scaler_path (str, optional): Path to the saved scaler.
        """
        # Load configurations
        self.data_config = get_config('data_config')
        
        # Initialize components
        self.pcap_converter = PCAPConverter()
        self.feature_extractor = FeatureExtractor()
        self.model_loader = ModelLoader(model_path, scaler_path)
        
        # Create directory for uploads if it doesn't exist
        self.upload_dir = get_nested_config(self.data_config, 'data.upload_path', 'data/uploads')
        os.makedirs(self.upload_dir, exist_ok=True)
    
    @log_function_call(logger)
    def process_pcap_file(self, pcap_file: str) -> Dict[str, Any]:
        """
        Process a PCAP file and classify the traffic.
        
        Args:
            pcap_file (str): Path to the PCAP file.
            
        Returns:
            Dict[str, Any]: Classification results.
        """
        logger.info(f"Processing PCAP file: {pcap_file}")
        
        start_time = time.time()
        
        # Create temporary output directory
        temp_dir = os.path.join(self.upload_dir, 'temp', os.path.basename(pcap_file).split('.')[0])
        os.makedirs(temp_dir, exist_ok=True)
        
        # Convert PCAP to CSV
        csv_file = os.path.join(temp_dir, f"{os.path.basename(pcap_file).split('.')[0]}.csv")
        self.pcap_converter.convert_file(pcap_file, csv_file)
        
        # Extract features
        feature_file = os.path.join(temp_dir, f"{os.path.basename(pcap_file).split('.')[0]}_features.csv")
        features_df = self.feature_extractor.extract_features_from_file(csv_file, feature_file)
        
        # Make prediction
        prediction = self.model_loader.predict(features_df)[0]
        
        # Get prediction probability if available
        probability = None
        if hasattr(self.model_loader.model, 'predict_proba'):
            probabilities = self.model_loader.predict_proba(features_df)[0]
            probability = probabilities[1] if prediction == 1 else probabilities[0]
        
        # Get processing time
        processing_time = time.time() - start_time
        
        # Create result dictionary
        result = {
            'file_name': os.path.basename(pcap_file),
            'prediction': 'ddos' if prediction == 1 or prediction == 'ddos' else 'normal',
            'probability': float(probability) if probability is not None else None,
            'processing_time': processing_time,
            'features': features_df.to_dict(orient='records')[0]
        }
        
        logger.info(f"Classification result: {result['prediction']} " +
                   f"(probability: {result['probability']:.4f})" if result['probability'] is not None else "")
        
        return result
    
    @log_function_call(logger)
    def batch_process_pcap_files(self, pcap_files: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple PCAP files and classify the traffic.
        
        Args:
            pcap_files (List[str]): List of paths to PCAP files.
            
        Returns:
            List[Dict[str, Any]]: Classification results for each file.
        """
        results = []
        
        for pcap_file in pcap_files:
            try:
                result = self.process_pcap_file(pcap_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pcap_file}: {str(e)}")
                results.append({
                    'file_name': os.path.basename(pcap_file),
                    'error': str(e)
                })
        
        return results
    
    @log_function_call(logger)
    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> str:
        """
        Save classification results to a file.
        
        Args:
            results (List[Dict[str, Any]]): Classification results.
            output_file (str): Path to the output file.
            
        Returns:
            str: Path to the output file.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        return output_file
    
    @log_function_call(logger)
    def generate_report(self, results: List[Dict[str, Any]], output_file: str) -> str:
        """
        Generate a report from classification results.
        
        Args:
            results (List[Dict[str, Any]]): Classification results.
            output_file (str): Path to the output file.
            
        Returns:
            str: Path to the output file.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Count predictions
        total = len(results)
        normal_count = sum(1 for r in results if 'prediction' in r and r['prediction'] == 'normal')
        ddos_count = sum(1 for r in results if 'prediction' in r and r['prediction'] == 'ddos')
        error_count = sum(1 for r in results if 'error' in r)
        
        # Create report
        report = {
            'total_files': total,
            'normal_count': normal_count,
            'ddos_count': ddos_count,
            'error_count': error_count,
            'normal_percentage': normal_count / total * 100 if total > 0 else 0,
            'ddos_percentage': ddos_count / total * 100 if total > 0 else 0,
            'error_percentage': error_count / total * 100 if total > 0 else 0,
            'results': results
        }
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report generated at {output_file}")
        
        return output_file
    
    @log_function_call(logger)
    def visualize_results(self, results: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Visualize classification results.
        
        Args:
            results (List[Dict[str, Any]]): Classification results.
            output_dir (str): Directory to save visualizations.
            
        Returns:
            List[str]: Paths to the generated visualizations.
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        visualization_files = []
        
        # Count predictions
        predictions = [r['prediction'] for r in results if 'prediction' in r]
        prediction_counts = pd.Series(predictions).value_counts()
        
        # Create pie chart
        plt.figure(figsize=(8, 6))
        plt.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', shadow=True)
        plt.title('Traffic Classification Results')
        plt.axis('equal')
        
        pie_chart_file = os.path.join(output_dir, 'prediction_pie_chart.png')
        plt.savefig(pie_chart_file)
        plt.close()
        
        visualization_files.append(pie_chart_file)
        
        # Create bar chart
        plt.figure(figsize=(8, 6))
        sns.barplot(x=prediction_counts.index, y=prediction_counts.values)
        plt.title('Traffic Classification Results')
        plt.xlabel('Classification')
        plt.ylabel('Count')
        
        bar_chart_file = os.path.join(output_dir, 'prediction_bar_chart.png')
        plt.savefig(bar_chart_file)
        plt.close()
        
        visualization_files.append(bar_chart_file)
        
        # Create probability histogram if available
        probabilities = [r['probability'] for r in results if 'probability' in r and r['probability'] is not None]
        if probabilities:
            plt.figure(figsize=(8, 6))
            sns.histplot(probabilities, bins=10, kde=True)
            plt.title('Prediction Probabilities')
            plt.xlabel('Probability')
            plt.ylabel('Count')
            
            prob_hist_file = os.path.join(output_dir, 'probability_histogram.png')
            plt.savefig(prob_hist_file)
            plt.close()
            
            visualization_files.append(prob_hist_file)
        
        logger.info(f"Created {len(visualization_files)} visualizations")
        
        return visualization_files
    
    @log_function_call(logger)
    def analyze_live_traffic(self, pcap_file: str, window_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze live traffic from a PCAP file with sliding window.
        
        Args:
            pcap_file (str): Path to the PCAP file.
            window_size (int): Size of the sliding window in packets.
            
        Returns:
            Dict[str, Any]: Analysis results.
        """
        # This method simulates live traffic analysis
        # In a real system, this would process a continuous stream
        
        logger.info(f"Analyzing live traffic from {pcap_file}")
        
        # Convert PCAP to CSV
        temp_dir = os.path.join(self.upload_dir, 'temp', os.path.basename(pcap_file).split('.')[0])
        os.makedirs(temp_dir, exist_ok=True)
        
        csv_file = os.path.join(temp_dir, f"{os.path.basename(pcap_file).split('.')[0]}.csv")
        self.pcap_converter.convert_file(pcap_file, csv_file)
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Process in windows
        window_results = []
        
        for i in range(0, len(df), window_size):
            window_df = df.iloc[i:i+window_size]
            
            # Extract features for this window
            window_features_file = os.path.join(temp_dir, f"window_{i}.csv")
            window_features = self.feature_extractor.extract_features_from_file(
                window_df, window_features_file
            )
            
            # Make prediction
            prediction = self.model_loader.predict(window_features)[0]
            
            # Get prediction probability if available
            probability = None
            if hasattr(self.model_loader.model, 'predict_proba'):
                probabilities = self.model_loader.predict_proba(window_features)[0]
                probability = probabilities[1] if prediction == 1 else probabilities[0]
            
            # Record result
            window_results.append({
                'window_start': i,
                'window_end': min(i + window_size, len(df)),
                'packet_count': len(window_df),
                'prediction': 'ddos' if prediction == 1 or prediction == 'ddos' else 'normal',
                'probability': float(probability) if probability is not None else None
            })
        
        # Calculate overall statistics
        normal_windows = sum(1 for r in window_results if r['prediction'] == 'normal')
        ddos_windows = sum(1 for r in window_results if r['prediction'] == 'ddos')
        
        result = {
            'file_name': os.path.basename(pcap_file),
            'total_packets': len(df),
            'window_size': window_size,
            'window_count': len(window_results),
            'normal_windows': normal_windows,
            'ddos_windows': ddos_windows,
            'normal_percentage': normal_windows / len(window_results) * 100 if window_results else 0,
            'ddos_percentage': ddos_windows / len(window_results) * 100 if window_results else 0,
            'windows': window_results
        }
        
        return result

if __name__ == "__main__":
    # Example usage
    pipeline = InferencePipeline()
    
    # Process a single PCAP file
    # result = pipeline.process_pcap_file("data/uploads/example.pcap")
    
    # Batch process PCAP files
    # pcap_files = ["data/uploads/example1.pcap", "data/uploads/example2.pcap"]
    # results = pipeline.batch_process_pcap_files(pcap_files)
    
    # Save results
    # pipeline.save_results(results, "data/uploads/results.json")
    
    # Generate report
    # pipeline.generate_report(results, "data/uploads/report.json")
    
    # Visualize results
    # pipeline.visualize_results(results, "data/uploads/visualizations")