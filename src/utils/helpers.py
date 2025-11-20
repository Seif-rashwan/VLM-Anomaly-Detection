"""
Helper utility functions
"""

import cv2
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple


def save_uploaded_file(uploaded_file, save_dir: str = "assets/uploads") -> Optional[str]:
    """
    Save uploaded file to disk.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        save_dir: Directory to save file
        
    Returns:
        Path to saved file or None if error
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    file_path = save_path / uploaded_file.name
    
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def get_video_info(video_path: str) -> Optional[Tuple[float, int, float]]:
    """
    Get video information (fps, frame_count, duration).
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (fps, frame_count, duration_seconds) or None if error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        return (fps, frame_count, duration)
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def format_time(seconds: float) -> str:
    """
    Format time in seconds to readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (MM:SS)
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

