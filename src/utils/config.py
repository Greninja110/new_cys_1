"""
Configuration utilities for loading and validating config files.
"""

import os
import yaml
from typing import Dict, Any, Optional

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"Error parsing YAML configuration file: {exc}")

def get_config(config_name: str) -> Dict[str, Any]:
    """
    Get configuration by name.
    
    Args:
        config_name (str): Name of the configuration file (without extension).
        
    Returns:
        Dict[str, Any]: Dictionary containing the configuration.
    """
    config_path = os.path.join('configs', f"{config_name}.yaml")
    return load_yaml_config(config_path)

def get_nested_config(config: Dict[str, Any], keys: str, default: Optional[Any] = None) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary.
        keys (str): Dot-separated keys to access nested values (e.g., "model.random_forest.n_estimators").
        default (Any, optional): Default value if the key is not found.
        
    Returns:
        Any: The configuration value, or the default if not found.
    """
    parts = keys.split('.')
    current = config
    
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    
    return current

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Variable number of configuration dictionaries.
        
    Returns:
        Dict[str, Any]: Merged configuration dictionary.
    """
    result = {}
    
    for config in configs:
        _recursive_merge(result, config)
    
    return result

def _recursive_merge(d1: Dict[str, Any], d2: Dict[str, Any]) -> None:
    """
    Recursively merge d2 into d1.
    
    Args:
        d1 (Dict[str, Any]): First dictionary (target).
        d2 (Dict[str, Any]): Second dictionary (source).
    """
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            _recursive_merge(d1[key], value)
        else:
            d1[key] = value