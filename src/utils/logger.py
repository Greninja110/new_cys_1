"""
Custom logging configuration for the network traffic analyzer.
"""

import os
import logging
import sys
from logging.handlers import RotatingFileHandler
import datetime

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

def setup_logger(name, log_file='app.log', level=logging.INFO):
    """
    Set up a logger with both console and file output.
    
    Args:
        name (str): Name of the logger.
        log_file (str): File name for the log.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create file handler which logs even debug messages
    file_handler = RotatingFileHandler(
        os.path.join('logs', log_file),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

def log_function_call(logger):
    """
    Decorator to log function calls with parameters and return values.
    
    Args:
        logger (logging.Logger): Logger instance.
        
    Returns:
        function: Decorator function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Function '{func_name}' called with args: {args}, kwargs: {kwargs}")
            
            start_time = datetime.datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                logger.debug(f"Function '{func_name}' executed in {execution_time:.4f} seconds")
                return result
            except Exception as e:
                logger.error(f"Exception in function '{func_name}': {str(e)}", exc_info=True)
                raise
        
        return wrapper
    
    return decorator