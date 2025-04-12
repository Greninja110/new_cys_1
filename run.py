"""
Entry point for running the network traffic analyzer web application.
"""

import os
import sys
import argparse

# Add the src directory to the path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Local imports
from src.utils.logger import setup_logger
from src.web.app import create_app, init_model

# Set up logger
logger = setup_logger('run')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run network traffic analyzer web application')
    
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to run the Flask server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the Flask server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--env', type=str, choices=['development', 'production', 'testing'],
                        default='development', help='Environment to run in')
    
    return parser.parse_args()

def main():
    """Main function to run the Flask application."""
    args = parse_arguments()
    
    # Set environment variables
    os.environ['FLASK_ENV'] = args.env
    
    logger.info(f"Starting network traffic analyzer web application in {args.env} mode")
    
    # Check if model exists
    model_loader = init_model()
    if model_loader:
        logger.info("Model loaded successfully")
        model_info = model_loader.get_model_info()
        logger.info(f"Model type: {model_info['model_type']}")
    else:
        logger.warning("Model not loaded. Some features may not work!")
    
    # Create and run the Flask application
    app = create_app()
    
    logger.info(f"Running on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error running application: {str(e)}")
        sys.exit(1)