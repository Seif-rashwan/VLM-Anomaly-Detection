"""
Configuration management for the anomaly detection system
Loads settings from config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager that loads settings from config.yaml"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from config.yaml"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            # Use default values if config file doesn't exist
            self._config = {
                'frame_interval_ms': 500,
                'default_model_name': 'ViT-B-32',
                'default_pretrained': 'openai',
                'model_dir': 'assets/models',
                'baseline_dir': 'assets/baselines',
                'cache_dir': 'assets/cache',
                'uploads_dir': 'assets/uploads',
                'default_similarity_threshold': 0.6,
                'default_nu': 0.1,
                'temporal_window_size': 5,
                'batch_size': 16,
                'log_level': 'INFO',
                'log_file': 'logs/anomaly_detection.log'
            }
            return
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading config.yaml: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config"""
        return key in self._config
    
    @property
    def config_dict(self) -> Dict[str, Any]:
        """Get the full configuration dictionary"""
        return self._config.copy()

