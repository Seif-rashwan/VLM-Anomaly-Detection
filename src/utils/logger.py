"""
Logging configuration for the anomaly detection system
"""

import logging
import sys
from pathlib import Path
from src.config import Config

def setup_logging():
    """Setup logging configuration from config file"""
    config = Config()
    
    log_level = getattr(logging, config.get('log_level', 'INFO').upper(), logging.INFO)
    log_file = config.get('log_file', 'logs/anomaly_detection.log')
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('open_clip').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={logging.getLevelName(log_level)}, file={log_file}")
    
    return logger

