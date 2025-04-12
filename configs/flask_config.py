"""
Flask configuration file
"""

import os

class Config:
    """Base Flask configuration class"""
    # General config
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'my-secret-key-for-development'
    SESSION_TYPE = 'filesystem'
    
    # Upload settings
    UPLOAD_FOLDER = 'data/uploads'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max upload
    ALLOWED_EXTENSIONS = {'pcap', 'pcapng'}
    
    # Model settings
    MODEL_PATH = 'data/models/model.pkl'
    SCALER_PATH = 'data/models/feature_scaler.pkl'
    
    # App settings
    DEBUG = False
    TESTING = False
    
    # Flask session settings
    PERMANENT_SESSION_LIFETIME = 1800  # 30 minutes


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    DEVELOPMENT = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # In production, use a proper secret key
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Use in-memory database for testing
    UPLOAD_FOLDER = 'data/test_uploads'


# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

# Default configuration
default_config = 'development'