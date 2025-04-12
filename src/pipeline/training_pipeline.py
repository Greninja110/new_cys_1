"""
End-to-end training pipeline for network traffic classifier.
"""

import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from ..utils.logger import setup_logger, log_function_call
from ..utils.config import get_config, get_nested_config
from ..data.pcap_converter import PCAPConverter
from ..data.parquet_converter import ParquetConverter
from ..data.feature_extractor import FeatureExtractor
from ..models.classifier import NetworkTrafficClassifier

# Set up logger
logger = setup_logger(__name__)


class TrainingPipeline:
    """End-to-end pipeline for training network traffic classifiers."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the training pipeline.
        
        Args:
            config_path (str, optional): Path to configuration files directory.
        """
        # Load configurations
        self.data_config = get_config('data_config')
        self.feature_config = get_config('feature_config')
        self.model_config = get_config('model_config')
        
        # Initialize components
        self.pcap_converter = PCAPConverter()
        self.parquet_converter = ParquetConverter()
        self.feature_extractor = FeatureExtractor()
        self.classifier = NetworkTrafficClassifier()
        
        # Create directories if they don't exist
        os.makedirs(get_nested_config(self.data_config, 'data.processed_path.training'), exist_ok=True)
        os.makedirs(get_nested_config(self.data_config, 'data.processed_path.testing'), exist_ok=True)
        os.makedirs(get_nested_config(self.data_config, 'data.models_path', 'data/models'), exist_ok=True)
        
        # Results storage
        self.results = {}
    
    @log_function_call(logger)
    def process_parquet_files(self, source_dir: str, target_dir: str, label: str) -> List[Tuple[str, str]]:
        """
        Process parquet files to CSV and extract features.
        
        Args:
            source_dir (str): Directory with parquet files.
            target_dir (str): Directory to save processed CSV files.
            label (str): Label for the data (normal, ddos, etc.).
            
        Returns:
            List[Tuple[str, str]]: List of (feature_file_path, label) tuples.
        """
        logger.info(f"Processing parquet files from {source_dir}")
        
        # Convert parquet files to CSV
        csv_files = self.parquet_converter.convert_directory(
            source_dir, target_dir, parallel=True
        )
        
        logger.info(f"Converted {len(csv_files)} parquet files to CSV")
        
        # Extract features from CSV files
        feature_files = []
        for csv_file in tqdm(csv_files, desc="Extracting features"):
            feature_file = os.path.join(
                target_dir,
                f"{os.path.splitext(os.path.basename(csv_file))[0]}_features.csv"
            )
            
            try:
                self.feature_extractor.extract_features_from_file(
                    csv_file, feature_file, label
                )
                feature_files.append((feature_file, label))
            except Exception as e:
                logger.error(f"Failed to extract features from {csv_file}: {str(e)}")
        
        logger.info(f"Extracted features for {len(feature_files)} files")
        
        return feature_files
    
    @log_function_call(logger)
    def process_pcap_files(self, source_dir: str, target_dir: str, label: str) -> List[Tuple[str, str]]:
        """
        Process pcap files to CSV and extract features.
        
        Args:
            source_dir (str): Directory with pcap files.
            target_dir (str): Directory to save processed CSV files.
            label (str): Label for the data (normal, ddos, etc.).
            
        Returns:
            List[Tuple[str, str]]: List of (feature_file_path, label) tuples.
        """
        logger.info(f"Processing pcap files from {source_dir}")
        
        # Convert pcap files to CSV
        csv_files = self.pcap_converter.convert_directory(
            source_dir, target_dir, parallel=True
        )
        
        logger.info(f"Converted {len(csv_files)} pcap files to CSV")
        
        # Extract features from CSV files
        feature_files = []
        for csv_file in tqdm(csv_files, desc="Extracting features"):
            feature_file = os.path.join(
                target_dir,
                f"{os.path.splitext(os.path.basename(csv_file))[0]}_features.csv"
            )
            
            try:
                self.feature_extractor.extract_features_from_file(
                    csv_file, feature_file, label
                )
                feature_files.append((feature_file, label))
            except Exception as e:
                logger.error(f"Failed to extract features from {csv_file}: {str(e)}")
        
        logger.info(f"Extracted features for {len(feature_files)} files")
        
        return feature_files
    
    @log_function_call(logger)
    def process_excel_files(self, source_dir: str, target_dir: str, label: str) -> List[Tuple[str, str]]:
        """
        Process Excel files to CSV and extract features.
        
        Args:
            source_dir (str): Directory with Excel files.
            target_dir (str): Directory to save processed CSV files.
            label (str): Label for the data (normal, ddos, etc.).
            
        Returns:
            List[Tuple[str, str]]: List of (feature_file_path, label) tuples.
        """
        logger.info(f"Processing Excel files from {source_dir}")
        
        # Get Excel files
        excel_files = [
            os.path.join(source_dir, f) for f in os.listdir(source_dir)
            if f.endswith(('.xlsx', '.xls', '.csv', 'pcap_ISCX'))
        ]
        
        # Convert Excel files to CSV
        csv_files = []
        for excel_file in tqdm(excel_files, desc="Converting Excel files"):
            csv_file = os.path.join(
                target_dir, 
                f"{os.path.splitext(os.path.basename(excel_file))[0]}.csv"
            )
            
            try:
                self.parquet_converter.extract_csv_from_excel(
                    excel_file, csv_file
                )
                csv_files.append(csv_file)
            except Exception as e:
                logger.error(f"Failed to convert Excel file {excel_file}: {str(e)}")
        
        logger.info(f"Converted {len(csv_files)} Excel files to CSV")
        
        # Extract features from CSV files
        feature_files = []
        for csv_file in tqdm(csv_files, desc="Extracting features"):
            feature_file = os.path.join(
                target_dir,
                f"{os.path.splitext(os.path.basename(csv_file))[0]}_features.csv"
            )
            
            try:
                self.feature_extractor.extract_features_from_file(
                    csv_file, feature_file, label
                )
                feature_files.append((feature_file, label))
            except Exception as e:
                logger.error(f"Failed to extract features from {csv_file}: {str(e)}")
        
        logger.info(f"Extracted features for {len(feature_files)} files")
        
        return feature_files
    
    @log_function_call(logger)
    def train_model(self, feature_files: List[Tuple[str, str]], eval_split: float = 0.2) -> Dict[str, Any]:
        """
        Train a model on the extracted features.
        
        Args:
            feature_files (List[Tuple[str, str]]): List of (feature_file_path, label) tuples.
            eval_split (float): Fraction of data to use for evaluation.
            
        Returns:
            Dict[str, Any]: Training results.
        """
        logger.info(f"Training model with {len(feature_files)} feature files")
        
        # Train the model
        start_time = time.time()
        results = self.classifier.train_from_files(feature_files, eval_split)
        training_time = time.time() - start_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Save the model
        model_path, scaler_path = self.classifier.save_model()
        
        # Plot feature importance
        if hasattr(self.classifier.model, 'feature_importances_'):
            plot_path = os.path.join(
                os.path.dirname(model_path), 
                'feature_importance.png'
            )
            self.classifier.plot_feature_importance(
                top_n=20, save_path=plot_path
            )
        
        # Store results
        self.results = {
            'metrics': results,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'training_time': training_time,
            'feature_files': feature_files
        }
        
        return self.results
    
    @log_function_call(logger)
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns:
            Dict[str, Any]: Training results.
        """
        logger.info("Starting training pipeline")
        
        # Get data paths
        cic_ids2017_path = get_nested_config(
            self.data_config, 'data.raw_path.cic_ids2017', 'data/raw/CIC-IDS2017'
        )
        cic_ddos2019_path = get_nested_config(
            self.data_config, 'data.raw_path.cic_ddos2019', 'data/raw/CIC-DDoS2019'
        )
        processed_training_path = get_nested_config(
            self.data_config, 'data.processed_path.training', 'data/processed/training'
        )
        
        # Process normal traffic data from CIC-IDS2017
        logger.info("Processing normal traffic data from CIC-IDS2017")
        normal_feature_files = []
        benign_parquet_files = os.path.join(cic_ids2017_path, 'Benign-*.parquet')
        if os.path.exists(os.path.dirname(benign_parquet_files)):
            normal_feature_files.extend(
                self.process_parquet_files(
                    cic_ids2017_path, 
                    os.path.join(processed_training_path, 'normal'),
                    'normal'
                )
            )
        
        # Process DDoS attack data from CIC-IDS2017
        logger.info("Processing DDoS attack data from CIC-IDS2017")
        ddos_feature_files = []
        ddos_parquet_files = os.path.join(cic_ids2017_path, 'DDoS-*.parquet')
        if os.path.exists(os.path.dirname(ddos_parquet_files)):
            ddos_feature_files.extend(
                self.process_parquet_files(
                    cic_ids2017_path, 
                    os.path.join(processed_training_path, 'ddos'),
                    'ddos'
                )
            )
        
        # Process DDoS attack data from CIC-DDoS2019
        logger.info("Processing DDoS attack data from CIC-DDoS2019")
        excel_files = os.path.join(cic_ddos2019_path, '*.pcap_ISCX')
        if os.path.exists(os.path.dirname(excel_files)):
            ddos_feature_files.extend(
                self.process_excel_files(
                    cic_ddos2019_path, 
                    os.path.join(processed_training_path, 'ddos'),
                    'ddos'
                )
            )
        
        # Combine all feature files
        all_feature_files = normal_feature_files + ddos_feature_files
        
        if not all_feature_files:
            logger.error("No feature files were generated!")
            return {}
        
        logger.info(f"Total feature files: {len(all_feature_files)}")
        logger.info(f"  Normal: {len(normal_feature_files)}")
        logger.info(f"  DDoS: {len(ddos_feature_files)}")
        
        # Train the model
        results = self.train_model(all_feature_files)
        
        logger.info("Training pipeline completed successfully")
        
        return results
    
    @log_function_call(logger)
    def cross_validate_model(self, feature_files: List[Tuple[str, str]], cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            feature_files (List[Tuple[str, str]]): List of (feature_file_path, label) tuples.
            cv (int): Number of cross-validation folds.
            
        Returns:
            Dict[str, Any]: Cross-validation results.
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Load and combine feature files
        X_list = []
        y_list = []
        
        for file_path, label in tqdm(feature_files, desc="Loading feature files"):
            try:
                # Load file
                df = pd.read_csv(file_path)
                
                # Check if label column exists
                if 'label' in df.columns:
                    # File already has label column
                    X = df.drop('label', axis=1)
                    y = df['label']
                else:
                    # Add label
                    X = df
                    y = pd.Series([label] * len(df))
                
                X_list.append(X)
                y_list.append(y)
                
            except Exception as e:
                logger.error(f"Error loading feature file {file_path}: {str(e)}")
        
        if not X_list:
            raise ValueError("No feature files could be loaded")
        
        # Combine data
        X = pd.concat(X_list, ignore_index=True)
        y = pd.concat(y_list, ignore_index=True)
        
        logger.info(f"Combined data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Perform cross-validation
        cv_results = self.classifier.cross_validate(X, y, cv=cv)
        
        return cv_results

if __name__ == "__main__":
    # Example usage
    pipeline = TrainingPipeline()
    
    # Run the pipeline
    results = pipeline.run_pipeline()
    
    # Alternatively, run components separately
    # feature_files = pipeline.process_parquet_files('data/raw', 'data/processed', 'normal')
    # results = pipeline.train_model(feature_files)