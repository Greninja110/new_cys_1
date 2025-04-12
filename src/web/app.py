"""
Flask application for network traffic analyzer web interface.
"""

import os
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

# Local imports
from ..utils.logger import setup_logger
from ..utils.config import get_config

# Set up logger
logger = setup_logger(__name__)

def create_app(test_config=None):
    """
    Create and configure the Flask application.
    
    Args:
        test_config: Test configuration to override default configs.
        
    Returns:
        Flask: Configured Flask application.
    """
    # Create and configure the app
    app = Flask(__name__, 
                instance_relative_config=True,
                template_folder='../../templates',
                static_folder='../../static')
    
    # Apply ProxyFix to handle reverse proxy headers
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Load config
    if test_config is None:
        # Load the default config
        from configs.flask_config import config_by_name, default_config
        app.config.from_object(config_by_name[os.getenv('FLASK_ENV', default_config)])
    else:
        # Load the test config
        app.config.from_mapping(test_config)
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints
    from . import routes
    app.register_blueprint(routes.main_bp)
    
    # Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        from flask import render_template
        return render_template('error.html', error=error, message='Page not found'), 404
    
    @app.errorhandler(500)
    def server_error(error):
        from flask import render_template
        logger.error(f"Server error: {str(error)}")
        return render_template('error.html', error=error, message='Server error'), 500
    
    # Log application startup
    logger.info(f"Flask application created with config: {app.config['ENV']}")
    
    return app

# Initialize model loader at application startup
def init_model():
    """Initialize the model loader."""
    from ..models.model_loader import ModelLoader
    
    try:
        model_loader = ModelLoader()
        model_info = model_loader.get_model_info()
        logger.info(f"Model loaded successfully: {model_info['model_type']}")
        return model_loader
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None