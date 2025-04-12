"""
Entry point for training the network traffic classifier model.
"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime

# Add the src directory to the path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Local imports
from src.utils.logger import setup_logger
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.config import get_config

# Set up logger
logger = setup_logger('main')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train network traffic classifier model')
    
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing the data files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the trained model')
    parser.add_argument('--cv', type=int, default=0,
                        help='Number of cross-validation folds (0 to disable)')
    parser.add_argument('--eval-split', type=float, default=0.2,
                        help='Fraction of data to use for evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main function to train the model."""
    args = parse_arguments()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting network traffic classifier training")
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Create and run the training pipeline
    pipeline = TrainingPipeline()
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Calculate training duration
    duration = time.time() - start_time
    
    # Print results summary
    if results:
        logger.info("Training completed successfully!")
        logger.info(f"Total training time: {duration:.2f} seconds")
        
        # Print evaluation metrics
        if 'metrics' in results and 'eval' in results['metrics']:
            metrics = results['metrics']['eval']
            logger.info("Evaluation metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"Model saved to: {results.get('model_path', 'Unknown')}")
    else:
        logger.error("Training failed. Check logs for errors.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        sys.exit(1)