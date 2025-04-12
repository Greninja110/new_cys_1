"""
Utility for loading trained models for inference.
"""

import os
import pickle
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

# Local imports
from ..utils.logger import setup_logger, log_function_call
from ..utils.config import get_config, get_nested_config

# Set up logger
logger = setup_logger(__name__)

class ModelLoader:
    """Class for loading and managing trained models for inference."""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            model_path (str, optional): Path to the saved model.
            scaler_path (str, optional): Path to the saved scaler.
        """
        # Load configurations
        self.config = get_config('model_config')
        self.data_config = get_config('data_config')
        
        # Set model and scaler paths
        self.model_path = model_path or get_nested_config(self.config, 'model.save_path', 'data/models/model.pkl')
        self.scaler_path = scaler_path or get_nested_config(self.config, 'model.scaler_path', 'data/models/feature_scaler.pkl')
        
        # Initialize model and scaler
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load model and scaler
        self.load()
    
    @log_function_call(logger)
    def load(self) -> None:
        """Load the trained model and scaler."""
        try:
            # Check if model and scaler files exist
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if not os.path.exists(self.scaler_path):
                logger.error(f"Scaler file not found: {self.scaler_path}")
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
                self.scaler = scaler_data['scaler']
                self.feature_names = scaler_data['feature_names']
            
            logger.info(f"Model loaded from {self.model_path}")
            logger.info(f"Scaler loaded from {self.scaler_path}")
            logger.info(f"Loaded model type: {type(self.model).__name__}")
            logger.info(f"Number of features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @log_function_call(logger)
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
            
        Returns:
            np.ndarray: Predicted labels.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded yet")
        
        # Ensure feature order matches training if dataframe
        if isinstance(X, pd.DataFrame) and self.feature_names is not None:
            # Get common features
            common_features = list(set(X.columns) & set(self.feature_names))
            
            if len(common_features) < len(self.feature_names):
                missing_features = set(self.feature_names) - set(common_features)
                logger.warning(f"Missing {len(missing_features)} features: {missing_features}")
                
                # Add missing features with zeros
                for feature in missing_features:
                    X[feature] = 0
            
            # Reorder columns to match training
            X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    @log_function_call(logger)
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
            
        Returns:
            np.ndarray: Prediction probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded yet")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {type(self.model).__name__} does not support probability predictions")
        
        # Ensure feature order matches training if dataframe
        if isinstance(X, pd.DataFrame) and self.feature_names is not None:
            # Get common features
            common_features = list(set(X.columns) & set(self.feature_names))
            
            if len(common_features) < len(self.feature_names):
                missing_features = set(self.feature_names) - set(common_features)
                logger.warning(f"Missing {len(missing_features)} features: {missing_features}")
                
                # Add missing features with zeros
                for feature in missing_features:
                    X[feature] = 0
            
            # Reorder columns to match training
            X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    @log_function_call(logger)
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importances for tree-based models.
        
        Args:
            top_n (int): Number of top features to return.
            
        Returns:
            Dict[str, float]: Dictionary of feature importances.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded yet")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"Model {type(self.model).__name__} does not support feature importances")
            return {}
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create a dictionary of feature importances
        if self.feature_names is not None:
            feature_imp = {self.feature_names[i]: importances[i] for i in range(len(importances))}
        else:
            feature_imp = {f'Feature_{i}': importances[i] for i in range(len(importances))}
        
        # Sort by importance
        feature_imp = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        return feature_imp
    
    @log_function_call(logger)
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Dictionary of model information.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded yet")
        
        model_info = {
            'model_type': type(self.model).__name__,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'features': self.feature_names
        }
        
        # Add model-specific information
        if hasattr(self.model, 'n_estimators'):
            model_info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            model_info['max_depth'] = self.model.max_depth
        
        if hasattr(self.model, 'feature_importances_'):
            top_features = self.get_feature_importance(5)
            model_info['top_features'] = top_features
        
        return model_info

if __name__ == "__main__":
    # Example usage
    loader = ModelLoader()
    
    # Get model info
    model_info = loader.get_model_info()
    print(model_info)