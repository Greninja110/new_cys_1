"""
Classification models for network traffic analysis.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Local imports
from ..utils.logger import setup_logger, log_function_call
from ..utils.config import get_config, get_nested_config

# Set up logger
logger = setup_logger(__name__)

class NetworkTrafficClassifier:
    """Classification model for network traffic analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the classifier.
        
        Args:
            config_path (str, optional): Path to the model configuration file.
        """
        # Load configurations
        if config_path is None:
            self.config = get_config('model_config')
        else:
            from ..utils.config import load_yaml_config
            self.config = load_yaml_config(config_path)
        
        # Get data config for file paths
        self.data_config = get_config('data_config')
        
        # Model settings
        self.model_type = get_nested_config(self.config, 'model.type', 'random_forest')
        self.save_path = get_nested_config(self.config, 'model.save_path', 'data/models/model.pkl')
        self.scaler_path = get_nested_config(self.config, 'model.scaler_path', 'data/models/feature_scaler.pkl')
        
        # Initialize model
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def _create_model(self) -> Any:
        """
        Create a new model based on configuration.
        
        Returns:
            Any: Initialized model.
        """
        if self.model_type == 'random_forest':
            # Get Random Forest parameters
            n_estimators = get_nested_config(self.config, 'model.random_forest.n_estimators', 100)
            max_depth = get_nested_config(self.config, 'model.random_forest.max_depth', 20)
            min_samples_split = get_nested_config(self.config, 'model.random_forest.min_samples_split', 2)
            min_samples_leaf = get_nested_config(self.config, 'model.random_forest.min_samples_leaf', 1)
            bootstrap = get_nested_config(self.config, 'model.random_forest.bootstrap', True)
            class_weight = get_nested_config(self.config, 'model.random_forest.class_weight', 'balanced')
            random_state = get_nested_config(self.config, 'model.random_forest.random_state', 42)
            
            # Create model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1  # Use all available cores
            )
            
        elif self.model_type == 'xgboost':
            # Get XGBoost parameters
            n_estimators = get_nested_config(self.config, 'model.xgboost.n_estimators', 100)
            max_depth = get_nested_config(self.config, 'model.xgboost.max_depth', 10)
            learning_rate = get_nested_config(self.config, 'model.xgboost.learning_rate', 0.1)
            subsample = get_nested_config(self.config, 'model.xgboost.subsample', 0.8)
            colsample_bytree = get_nested_config(self.config, 'model.xgboost.colsample_bytree', 0.8)
            objective = get_nested_config(self.config, 'model.xgboost.objective', 'binary:logistic')
            random_state = get_nested_config(self.config, 'model.xgboost.random_state', 42)
            
            # Create model
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective=objective,
                random_state=random_state,
                n_jobs=-1  # Use all available cores
            )
            
        elif self.model_type == 'neural_network':
            # Get Neural Network parameters
            hidden_layers = get_nested_config(self.config, 'model.neural_network.hidden_layers', [128, 64])
            activation = get_nested_config(self.config, 'model.neural_network.activation', 'relu')
            solver = get_nested_config(self.config, 'model.neural_network.solver', 'adam')
            alpha = get_nested_config(self.config, 'model.neural_network.alpha', 0.0001)
            batch_size = get_nested_config(self.config, 'model.neural_network.batch_size', 'auto')
            learning_rate = get_nested_config(self.config, 'model.neural_network.learning_rate', 'adaptive')
            max_iter = get_nested_config(self.config, 'model.neural_network.max_iter', 200)
            random_state = get_nested_config(self.config, 'model.neural_network.random_state', 42)
            
            # Create model
            model = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_layers),
                activation=activation,
                solver=solver,
                alpha=alpha,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=random_state
            )
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def _create_scaler(self) -> Any:
        """
        Create a new feature scaler.
        
        Returns:
            Any: Initialized scaler.
        """
        # Get scaling method
        scaling_method = get_nested_config(self.config, 'features.scaling.method', 'standard_scaler')
        
        if scaling_method == 'standard_scaler':
            return StandardScaler()
        elif scaling_method == 'minmax_scaler':
            return MinMaxScaler()
        else:
            return StandardScaler()  # Default to StandardScaler
    
    @log_function_call(logger)
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
             eval_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
            y (Union[pd.Series, np.ndarray]): Target vector.
            eval_split (float): Fraction of data to use for evaluation.
            
        Returns:
            Dict[str, Any]: Training results including evaluation metrics.
        """
        logger.info(f"Training {self.model_type} model with {X.shape[0]} samples")
        
        # Split data for training and evaluation
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=eval_split, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples, Evaluation set: {X_eval.shape[0]} samples")
        
        # Save feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Initialize scaler
        self.scaler = self._create_scaler()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_eval_scaled = self.scaler.transform(X_eval)
        
        # Initialize model
        self.model = self._create_model()
        
        # Train the model
        logger.info(f"Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_preds = self.model.predict(X_train_scaled)
        eval_preds = self.model.predict(X_eval_scaled)
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics(y_train, train_preds, y_eval, eval_preds, X_train_scaled, X_eval_scaled)
        
        # Log metrics
        logger.info(f"Training metrics:")
        for metric, value in metrics['train'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"Evaluation metrics:")
        for metric, value in metrics['eval'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_train: np.ndarray, train_preds: np.ndarray,y_eval: np.ndarray, eval_preds: np.ndarray, X_train_scaled=None, X_eval_scaled=None) -> Dict[str, Dict[str, float]]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_train (np.ndarray): Training set ground truth.
            train_preds (np.ndarray): Training set predictions.
            y_eval (np.ndarray): Evaluation set ground truth.
            eval_preds (np.ndarray): Evaluation set predictions.
            X_train_scaled: Scaled training features (for ROC AUC)
            X_eval_scaled: Scaled evaluation features (for ROC AUC)
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of metrics.
        """
        metrics = {
            'train': {},
            'eval': {}
        }
        
        # Training metrics
        metrics['train']['accuracy'] = accuracy_score(y_train, train_preds)
        metrics['train']['precision'] = precision_score(y_train, train_preds, average='weighted')
        metrics['train']['recall'] = recall_score(y_train, train_preds, average='weighted')
        metrics['train']['f1'] = f1_score(y_train, train_preds, average='weighted')
        
        # Check if binary classification for ROC AUC
        if len(np.unique(y_train)) == 2 and hasattr(self.model, 'predict_proba') and X_train_scaled is not None:
            train_probs = self.model.predict_proba(X_train_scaled)[:, 1]
            metrics['train']['roc_auc'] = roc_auc_score(y_train, train_probs)
        
        # Evaluation metrics
        metrics['eval']['accuracy'] = accuracy_score(y_eval, eval_preds)
        metrics['eval']['precision'] = precision_score(y_eval, eval_preds, average='weighted')
        metrics['eval']['recall'] = recall_score(y_eval, eval_preds, average='weighted')
        metrics['eval']['f1'] = f1_score(y_eval, eval_preds, average='weighted')
        
        # Check if binary classification for ROC AUC
        if len(np.unique(y_eval)) == 2 and hasattr(self.model, 'predict_proba') and X_eval_scaled is not None:
            eval_probs = self.model.predict_proba(X_eval_scaled)[:, 1]
            metrics['eval']['roc_auc'] = roc_auc_score(y_eval, eval_probs)
        
        return metrics
    
    @log_function_call(logger)
    def save_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Save the trained model and scaler.
        
        Args:
            model_path (str, optional): Path to save the model.
            scaler_path (str, optional): Path to save the scaler.
            
        Returns:
            Tuple[str, str]: Paths to the saved model and scaler.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if model_path is None:
            model_path = self.save_path
        
        if scaler_path is None:
            scaler_path = self.scaler_path
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        
        return model_path, scaler_path
    
    @log_function_call(logger)
    def load_model(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None) -> None:
        """
        Load a trained model and scaler.
        
        Args:
            model_path (str, optional): Path to the saved model.
            scaler_path (str, optional): Path to the saved scaler.
        """
        if model_path is None:
            model_path = self.save_path
        
        if scaler_path is None:
            scaler_path = self.scaler_path
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            self.scaler = scaler_data['scaler']
            self.feature_names = scaler_data['feature_names']
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")
    
    @log_function_call(logger)
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
            
        Returns:
            np.ndarray: Predicted labels.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        # Ensure feature order matches training if dataframe
        if isinstance(X, pd.DataFrame) and self.feature_names is not None:
            # Get common features
            common_features = list(set(X.columns) & set(self.feature_names))
            
            if len(common_features) < len(self.feature_names):
                logger.warning(f"Missing features: {set(self.feature_names) - set(common_features)}")
            
            X = X[common_features]
        
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
            raise ValueError("Model has not been trained or loaded yet")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {type(self.model).__name__} does not support probability predictions")
        
        # Ensure feature order matches training if dataframe
        if isinstance(X, pd.DataFrame) and self.feature_names is not None:
            # Get common features
            common_features = list(set(X.columns) & set(self.feature_names))
            
            if len(common_features) < len(self.feature_names):
                logger.warning(f"Missing features: {set(self.feature_names) - set(common_features)}")
            
            X = X[common_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    @log_function_call(logger)
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
            y (Union[pd.Series, np.ndarray]): Ground truth labels.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted'),
            'recall': recall_score(y, predictions, average='weighted'),
            'f1': f1_score(y, predictions, average='weighted')
        }
        
        # Add ROC AUC if binary classification
        if len(np.unique(y)) == 2 and hasattr(self.model, 'predict_proba'):
            probabilities = self.predict_proba(X)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y, probabilities)
        
        # Log metrics
        logger.info(f"Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Print classification report
        logger.info("Classification Report:")
        report = classification_report(y, predictions)
        logger.info(f"\n{report}")
        
        # Print confusion matrix
        logger.info("Confusion Matrix:")
        cm = confusion_matrix(y, predictions)
        logger.info(f"\n{cm}")
        
        return metrics
    
    @log_function_call(logger)
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """
        Plot feature importances for tree-based models.
        
        Args:
            top_n (int): Number of top features to show.
            save_path (str, optional): Path to save the plot.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"Model {type(self.model).__name__} does not support feature importances")
            return
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create a DataFrame for easier plotting
        if self.feature_names is not None:
            feature_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        else:
            feature_imp = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(importances))],
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        
        # Select top N features
        feature_imp = feature_imp.head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_imp)
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    @log_function_call(logger)
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
                      cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the model.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
            y (Union[pd.Series, np.ndarray]): Target vector.
            cv (int): Number of cross-validation folds.
            
        Returns:
            Dict[str, List[float]]: Cross-validation results.
        """
        if self.model is None:
            self.model = self._create_model()
        
        if self.scaler is None:
            self.scaler = self._create_scaler()
        
        # Save feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Get scoring metrics
        scoring = get_nested_config(self.config, 'model.evaluation.scoring', 
                                  ['accuracy', 'precision', 'recall', 'f1'])
        
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validate
        cv_results = {}
        for metric in scoring:
            scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring=metric)
            cv_results[metric] = scores
            logger.info(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return cv_results
    
    @log_function_call(logger)
    def hyperparameter_tuning(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                             param_grid: Dict[str, List[Any]], cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature matrix.
            y (Union[pd.Series, np.ndarray]): Target vector.
            param_grid (Dict[str, List[Any]]): Parameter grid for search.
            cv (int): Number of cross-validation folds.
            
        Returns:
            Dict[str, Any]: Hyperparameter tuning results.
        """
        if self.scaler is None:
            self.scaler = self._create_scaler()
        
        # Save feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Create base model
        base_model = self._create_model()
        
        logger.info(f"Performing hyperparameter tuning with {cv}-fold cross-validation")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Get best parameters and results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_
        }
    
    @log_function_call(logger)
    def train_from_files(self, feature_files: List[Tuple[str, str]], 
                        eval_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model from feature files.
        
        Args:
            feature_files (List[Tuple[str, str]]): List of (file_path, label) tuples.
            eval_split (float): Fraction of data to use for evaluation.
            
        Returns:
            Dict[str, Any]: Training results.
        """
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
        
        # Train model
        return self.train(X, y, eval_split=eval_split)

if __name__ == "__main__":
    # Example usage
    classifier = NetworkTrafficClassifier()
    
    # Train from feature files
    # feature_files = [
    #     ("data/processed/features/normal_features.csv", "normal"),
    #     ("data/processed/features/ddos_features.csv", "ddos")
    # ]
    # results = classifier.train_from_files(feature_files)
    
    # Save model
    # classifier.save_model()
    
    # Load model
    # classifier.load_model()
    
    # Make predictions
    # predictions = classifier.predict(X_test)