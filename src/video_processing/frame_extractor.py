"""
Frame extraction module with caching support
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional
import hashlib
import pickle
import os


class FrameExtractor:
    """
    Extracts frames from video files with caching support.
    """
    
    def __init__(self, cache_dir: str = "assets/cache"):
        """
        Initialize frame extractor with cache directory.
        
        Args:
            cache_dir: Directory to store cached frames
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_video_hash(self, video_path: str, frame_interval_ms: int) -> str:
        """
        Generate a hash for the video file and frame interval.
        
        Args:
            video_path: Path to video file
            frame_interval_ms: Interval in milliseconds between frames
            
        Returns:
            Hash string for caching
        """
        file_stat = os.stat(video_path)
        file_size = file_stat.st_size
        file_mtime = file_stat.st_mtime
        
        hash_input = f"{video_path}_{file_size}_{file_mtime}_{frame_interval_ms}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[Image.Image]]:
        """
        Load frames from cache if available.
        
        Args:
            cache_key: Cache key for the video
            
        Returns:
            List of PIL Images or None if not cached
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, frames: List[Image.Image]):
        """
        Save frames to cache.
        
        Args:
            cache_key: Cache key for the video
            frames: List of PIL Images to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(frames, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def extract_frames(
        self, 
        video_path: str, 
        frame_interval_ms: int = 500,
        use_cache: bool = True
    ) -> List[Image.Image]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to video file
            frame_interval_ms: Interval in milliseconds between frames (default: 500ms)
            use_cache: Whether to use caching (default: True)
            
        Returns:
            List of PIL Images extracted from video
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check cache
        if use_cache:
            cache_key = self._get_video_hash(str(video_path), frame_interval_ms)
            cached_frames = self._load_from_cache(cache_key)
            if cached_frames is not None:
                return cached_frames
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default FPS if not available
        
        frame_interval = max(1, int(fps * (frame_interval_ms / 1000.0)))
        
        frames = []
        frame_count = 0
        
        # Always extract the first frame
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            if pil_image.size[0] > 0 and pil_image.size[1] > 0:
                frames.append(pil_image)
            frame_count += 1
        
        # Extract remaining frames at specified interval
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                # Verify image is valid
                if pil_image.size[0] > 0 and pil_image.size[1] > 0:
                    frames.append(pil_image)
            
            frame_count += 1
        
        cap.release()
        
        # Save to cache
        if use_cache and frames:
            cache_key = self._get_video_hash(str(video_path), frame_interval_ms)
            self._save_to_cache(cache_key, frames)
        
        return frames
    
    def extract_frame_at_time(
        self, 
        video_path: str, 
        time_seconds: float
    ) -> Optional[Image.Image]:
        """
        Extract a single frame at a specific time.
        
        Args:
            video_path: Path to video file
            time_seconds: Time in seconds to extract frame
            
        Returns:
            PIL Image or None if extraction fails
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * time_seconds)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        
        return None

