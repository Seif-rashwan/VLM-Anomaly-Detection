"""
Streamlit Dashboard for Anomaly Detection System
Supports both video upload and live camera modes
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import cv2
import time
from PIL import Image
import pandas as pd
from collections import Counter

# Setup logging first
from src.utils.logger import setup_logging
setup_logging()

# Import project modules
from src.video_processing import FrameExtractor
from src.vlm import VLMEncoder
from src.anomaly_detection import AnomalyDetector
from src.utils import save_uploaded_file, get_video_info, format_time
from src.config import Config

# Load configuration
config = Config()

# Page configuration
st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'vlm_encoder' not in st.session_state:
    st.session_state.vlm_encoder = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_name' not in st.session_state:
    st.session_state.model_name = "anomaly_detector"  # Default model name
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'expected_objects' not in st.session_state:
    st.session_state.expected_objects = None
if 'use_temporal_averaging' not in st.session_state:
    st.session_state.use_temporal_averaging = True
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None  # Store results for threshold adjustment

# Title and description
st.title("🎓 Anomaly Detection System using VLM")
st.markdown("""
    This system uses Vision-Language Models (OpenCLIP) to detect anomalies in videos.
    **Phase 1**: Upload videos for offline analysis | **Phase 2**: Real-time camera detection
""")

# Sidebar for mode selection
st.sidebar.title("⚙️ Configuration")
mode = st.sidebar.radio(
    "Select Mode",
    ["📹 Upload Video Mode", "📷 Live Camera Mode"],
    index=0
)

# Model configuration
st.sidebar.subheader("Model Settings")
model_name = st.sidebar.selectbox(
    "VLM Model",
    ["ViT-B-32", "ViT-L-14", "ViT-B-16"],
    index=0,
    help="OpenCLIP model variant"
)

pretrained = st.sidebar.selectbox(
    "Pretrained Dataset",
    ["openai", "laion400m_e32", "laion2b_s34b_b79k"],
    index=0,
    help="Pretrained dataset for the model"
)

# Dynamic Prompts Configuration
st.sidebar.subheader("Expected Objects")
expected_objects_input = st.sidebar.text_input(
    "Expected Objects (comma-separated)",
    value="",
    help="Enter expected objects like 'worker, machine, forklift'. Leave empty for default prompts."
)

if expected_objects_input.strip():
    expected_objects = [obj.strip() for obj in expected_objects_input.split(",") if obj.strip()]
    st.session_state.expected_objects = expected_objects if expected_objects else None
else:
    st.session_state.expected_objects = None

# Temporal Averaging
st.sidebar.subheader("Processing Options")
use_temporal_averaging = st.sidebar.checkbox(
    "Use Temporal Averaging",
    value=config.get('temporal_window_size', 5) > 1,
    help="Average embeddings of last N frames to reduce noise (flickering anomalies)"
)
st.session_state.use_temporal_averaging = use_temporal_averaging

temporal_window_size = st.sidebar.slider(
    "Temporal Window Size",
    min_value=1,
    max_value=10,
    value=config.get('temporal_window_size', 5),
    step=1,
    help="Number of frames to average for temporal smoothing",
    disabled=not use_temporal_averaging
)

similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=config.get('default_similarity_threshold', 0.6),
    step=0.05,
    help="Below this threshold, frames are flagged as anomalies. Lower values = more sensitive to small changes."
)

frame_interval_ms = st.sidebar.slider(
    "Frame Interval (ms)",
    min_value=100,
    max_value=2000,
    value=config.get('frame_interval_ms', 500),
    step=100,
    help="Interval between extracted frames"
)

# Initialize VLM encoder
@st.cache_resource
def load_vlm_encoder(model_name: str, pretrained: str):
    """Load VLM encoder (cached)"""
    return VLMEncoder(model_name=model_name, pretrained=pretrained)

if st.session_state.vlm_encoder is None:
    with st.spinner("Loading VLM model... This may take a minute on first run."):
        st.session_state.vlm_encoder = load_vlm_encoder(model_name, pretrained)
        st.success("Model loaded successfully!")

# Initialize components
frame_extractor = FrameExtractor(cache_dir=config.get('cache_dir', 'assets/cache'))

# Initialize anomaly detector - use session state to persist across reruns
if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector(
        model_dir=config.get('model_dir', 'assets/models'),
        baseline_dir=config.get('baseline_dir', 'assets/baselines')
    )

anomaly_detector = st.session_state.anomaly_detector
vlm_encoder = st.session_state.vlm_encoder

# Auto-load model if trained but not loaded
if st.session_state.model_trained and not anomaly_detector.is_trained:
    model_name_to_load = st.session_state.model_name or "anomaly_detector"
    if anomaly_detector.load_model(model_name_to_load):
        st.session_state.model_trained = True
    else:
        # Model file might not exist, reset flag
        st.session_state.model_trained = False

# ============================================================================
# UPLOAD VIDEO MODE
# ============================================================================
if mode == "📹 Upload Video Mode":
    st.header("📹 Video Upload Mode")
    
    # Tabs for different operations
    tab1, tab2 = st.tabs(["📌 Create Baseline", "🔍 Test Video"])
    
    # ========== TAB 1: CREATE BASELINE ==========
    with tab1:
        st.subheader("Upload Normal Reference Videos (Phase 1)")
        st.markdown("""
        **Academic Approach:** Upload MANY normal videos to train a One-Class model.
        - Upload multiple videos showing normal behavior
        - System will extract embeddings from all videos
        - Train One-Class SVM model on all normal embeddings
        """)
        
        # Support multiple video uploads
        baseline_videos = st.file_uploader(
            "Choose one or more normal video files",
            type=['mp4', 'avi', 'mov', 'mkv'],
            key="baseline_upload",
            accept_multiple_files=True
        )
        
        # Initialize session state for collected embeddings and descriptions
        if 'collected_embeddings' not in st.session_state:
            st.session_state.collected_embeddings = []
        if 'collected_video_names' not in st.session_state:
            st.session_state.collected_video_names = []
        if 'collected_descriptions' not in st.session_state:
            st.session_state.collected_descriptions = []  # Store VLM text descriptions for validation
        
        # Display collected videos status
        if len(st.session_state.collected_embeddings) > 0:
            total_embeddings = sum(len(emb) for emb in st.session_state.collected_embeddings)
            st.info(f"📊 **Collected:** {len(st.session_state.collected_video_names)} videos, {total_embeddings} total embeddings")
            st.write("**Videos:**", ", ".join(st.session_state.collected_video_names))
        
        if baseline_videos and len(baseline_videos) > 0:
            st.write(f"**{len(baseline_videos)} video(s) selected**")
            
            # Process each video
            if st.button("📥 Process & Add Videos to Training Set", type="primary"):
                try:
                    # Process all uploaded videos
                    all_embeddings = []
                    all_descriptions = []  # Store descriptions for validation
                    processed_videos = []
                    video_frame_mapping = []  # Track which frames belong to which video
                    
                    for video_idx, baseline_video in enumerate(baseline_videos):
                        with st.spinner(f"Processing video {video_idx + 1}/{len(baseline_videos)}: {baseline_video.name}..."):
                            # Save uploaded file
                            baseline_path = save_uploaded_file(baseline_video)
                            
                            if baseline_path:
                                # Extract frames
                                frames = frame_extractor.extract_frames(
                                    baseline_path,
                                    frame_interval_ms=frame_interval_ms
                                )
                                
                                if frames and len(frames) > 0:
                                    # Generate embeddings and descriptions
                                    video_embeddings = []
                                    video_descriptions = []
                                    for frame_idx, frame in enumerate(frames):
                                        try:
                                            text, embedding = vlm_encoder.get_frame_embedding(
                                                frame, 
                                                expected_objects=st.session_state.expected_objects
                                            )
                                            video_embeddings.append(embedding)
                                            video_descriptions.append({
                                                'video_name': baseline_video.name,
                                                'frame_number': frame_idx + 1,
                                                'description': text
                                            })
                                        except Exception as e:
                                            continue
                                    
                                    if len(video_embeddings) > 0:
                                        all_embeddings.extend(video_embeddings)
                                        all_descriptions.extend(video_descriptions)
                                        processed_videos.append(baseline_video.name)
                                        video_frame_mapping.append({
                                            'video_name': baseline_video.name,
                                            'frame_count': len(video_embeddings)
                                        })
                                        st.success(f"✅ {baseline_video.name}: {len(video_embeddings)} embeddings")
                    
                    # Add to collected embeddings and descriptions
                    if len(all_embeddings) > 0:
                        st.session_state.collected_embeddings.append(all_embeddings)
                        st.session_state.collected_descriptions.append(all_descriptions)
                        st.session_state.collected_video_names.extend(processed_videos)
                        st.success(f"📊 Added {len(all_embeddings)} embeddings from {len(processed_videos)} video(s)")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing videos: {e}")
                    st.exception(e)
            
            # Display VLM Descriptions for Validation
            if st.session_state.collected_descriptions and len(st.session_state.collected_descriptions) > 0:
                st.markdown("---")
                with st.expander("📝 Extracted VLM Descriptions (For Validation)", expanded=False):
                    st.markdown("""
                    **Purpose:** Review the VLM-generated text descriptions before training the model.
                    This helps identify hallucination issues or incorrect descriptions.
                    """)
                    
                    # Flatten all descriptions
                    all_descriptions_flat = []
                    for video_descriptions in st.session_state.collected_descriptions:
                        all_descriptions_flat.extend(video_descriptions)
                    
                    if len(all_descriptions_flat) > 0:
                        st.info(f"**Total Descriptions:** {len(all_descriptions_flat)}")
                        
                        # Group by video for better organization
                        videos_with_descriptions = {}
                        for desc in all_descriptions_flat:
                            video_name = desc['video_name']
                            if video_name not in videos_with_descriptions:
                                videos_with_descriptions[video_name] = []
                            videos_with_descriptions[video_name].append(desc)
                        
                        # Display descriptions grouped by video
                        for video_name, descriptions in videos_with_descriptions.items():
                            st.markdown(f"### 📹 {video_name} ({len(descriptions)} frames)")
                            
                            # Display in a table format
                            df_data = []
                            for desc in descriptions:
                                df_data.append({
                                    'Frame #': desc['frame_number'],
                                    'VLM Description': desc['description']
                                })
                            
                            df = pd.DataFrame(df_data)
                            # Use scrollable dataframe with max height
                            st.dataframe(
                                df,
                                use_container_width=True,
                                hide_index=True,
                                height=min(400, max(200, len(descriptions) * 40))
                            )
                            
                            st.markdown("---")
                        
                        # Summary statistics
                        st.markdown("### 📊 Description Statistics")
                        unique_descriptions = set(desc['description'] for desc in all_descriptions_flat)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Frames", len(all_descriptions_flat))
                        with col2:
                            st.metric("Unique Descriptions", len(unique_descriptions))
                        with col3:
                            avg_length = sum(len(desc['description']) for desc in all_descriptions_flat) / len(all_descriptions_flat)
                            st.metric("Avg Description Length", f"{avg_length:.1f} chars")
                        
                        # Show most common descriptions
                        description_counts = Counter(desc['description'] for desc in all_descriptions_flat)
                        most_common = description_counts.most_common(5)
                        
                        if len(most_common) > 0:
                            st.markdown("### 🔝 Most Common Descriptions")
                            for desc_text, count in most_common:
                                percentage = (count / len(all_descriptions_flat)) * 100
                                st.write(f"- **{desc_text}** ({count} times, {percentage:.1f}%)")
            
            # Train One-Class Model button
            total_embeddings = sum(len(emb) for emb in st.session_state.collected_embeddings) if st.session_state.collected_embeddings else 0
            if total_embeddings > 0:
                st.markdown("---")
                st.subheader("🚀 Train One-Class Model (Phase 1)")
                st.markdown(f"**Ready to train:** {total_embeddings} embeddings from {len(st.session_state.collected_video_names)} videos")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Widget automatically syncs with st.session_state.model_name
                    # Use value parameter to set initial value if session state is None
                    model_name = st.text_input(
                        "Model Name", 
                        value=st.session_state.model_name or "anomaly_detector", 
                        key="model_name"
                    )
                with col2:
                    nu_value = st.slider("Nu (outlier fraction)", 0.01, 0.5, 0.1, 0.01, 
                                        help="Expected fraction of outliers (0.1 = 10%)")
                
                if st.button("🎓 Train One-Class SVM Model", type="primary"):
                    try:
                        # Flatten all collected embeddings
                        all_normal_embeddings = []
                        for video_embeddings in st.session_state.collected_embeddings:
                            all_normal_embeddings.extend(video_embeddings)
                        
                        with st.spinner(f"Training One-Class SVM on {len(all_normal_embeddings)} embeddings..."):
                            # Train the model
                            # Note: model_name is already synced with st.session_state.model_name via the widget
                            model_path = anomaly_detector.train_one_class_model(
                                all_normal_embeddings,
                                model_name=model_name,  # Use widget value directly
                                nu=nu_value
                            )
                            
                            # Mark model as trained (don't reassign model_name - widget handles it)
                            st.session_state.model_trained = True
                            
                            st.success(f"✅ Model trained successfully!")
                            st.info(f"Model saved: {model_path}")
                            st.info(f"Trained on {len(all_normal_embeddings)} embeddings from {len(st.session_state.collected_video_names)} videos")
                            
                            # Verify model and scaler are loaded (they should already be in memory after training)
                            if anomaly_detector.is_trained and anomaly_detector.model is not None and anomaly_detector.scaler is not None:
                                st.session_state.model_trained = True
                                st.info("✅ Model and scaler are ready for use")
                            else:
                                # Try to reload from disk
                                if anomaly_detector.load_model(model_name):
                                    if anomaly_detector.is_trained and anomaly_detector.scaler is not None:
                                        st.session_state.model_trained = True
                                        st.info("✅ Model and scaler reloaded successfully")
                                    else:
                                        st.warning("⚠️ Model loaded but scaler is missing. Please retrain.")
                                else:
                                    st.warning("⚠️ Model trained but failed to reload. Please check the model directory.")
                    except Exception as e:
                        st.error(f"❌ Error training model: {e}")
                        st.exception(e)
            
            # Clear collected data button
            if len(st.session_state.collected_embeddings) > 0:
                if st.button("🗑️ Clear Collected Videos", type="secondary"):
                    st.session_state.collected_embeddings = []
                    st.session_state.collected_descriptions = []
                    st.session_state.collected_video_names = []
                    st.rerun()
    
    # ========== TAB 2: TEST VIDEO ==========
    with tab2:
        st.subheader("Upload Test Video (Phase 2)")
        st.markdown("Upload a video to test against the trained One-Class model.")
        
        # Check if model is trained and loaded
        model_name_to_check = st.session_state.model_name or "anomaly_detector"
        
        # Ensure model is actually loaded in the detector object
        if st.session_state.model_trained:
            if not anomaly_detector.is_trained:
                # Model flag is set but detector doesn't have it loaded - try to load
                if anomaly_detector.load_model(model_name_to_check):
                    st.info(f"🔄 Reloaded model: {model_name_to_check}")
                else:
                    st.error(f"❌ Model '{model_name_to_check}' not found. Please retrain the model.")
                    st.session_state.model_trained = False
            else:
                # Model is loaded and ready
                st.success(f"✅ Model ready: {model_name_to_check}")
        else:
            # Try to load a default model from disk
            if anomaly_detector.load_model("anomaly_detector"):
                st.session_state.model_trained = True
                st.session_state.model_name = "anomaly_detector"
                st.success("✅ Loaded trained model: anomaly_detector")
            else:
                st.warning("⚠️ Please train a One-Class model first in the 'Create Baseline' tab.")
        
        # Only proceed if model is actually loaded
        if anomaly_detector.is_trained:
            
            test_video = st.file_uploader(
                "Choose a test video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key="test_upload"
            )
            
            if test_video is not None:
                # Save uploaded file
                test_path = save_uploaded_file(test_video)
                
                if test_path:
                    st.success(f"Video uploaded: {test_video.name}")
                    
                    # Process test video
                    if st.button("🔍 Analyze Test Video", type="primary"):
                        try:
                            with st.spinner("Processing test video..."):
                                # Extract frames
                                frames = frame_extractor.extract_frames(
                                    test_path,
                                    frame_interval_ms=frame_interval_ms
                                )
                                
                                if not frames or len(frames) == 0:
                                    st.error("❌ No frames extracted from video. Please check the video file.")
                                else:
                                    st.info(f"Extracted {len(frames)} frames")
                                    
                                    # Verify frames are valid PIL Images
                                    valid_frames = []
                                    for frame in frames:
                                        if isinstance(frame, Image.Image) and frame.size[0] > 0 and frame.size[1] > 0:
                                            valid_frames.append(frame)
                                    
                                    if len(valid_frames) != len(frames):
                                        st.warning(f"⚠️ Some frames were invalid. Using {len(valid_frames)} valid frames.")
                                    
                                    frames = valid_frames
                                    
                                    if len(frames) == 0:
                                        st.error("❌ No valid frames found. Please try a different video.")
                                    else:
                                        # Reset temporal window if using temporal averaging
                                        if st.session_state.use_temporal_averaging:
                                            anomaly_detector.reset_temporal_window(temporal_window_size)
                                        
                                        # Generate embeddings using batch processing
                                        batch_size = config.get('batch_size', 16)
                                        progress_bar = st.progress(0)
                                        st.info(f"Processing {len(frames)} frames with batch size {batch_size}...")
                                        
                                        # Use batch processing for better GPU utilization
                                        batch_results = vlm_encoder.get_frame_embeddings_batch(
                                            frames,
                                            expected_objects=st.session_state.expected_objects,
                                            batch_size=batch_size
                                        )
                                        
                                        test_embeddings = []
                                        test_texts = []
                                        for text, embedding in batch_results:
                                            test_embeddings.append(embedding)
                                            test_texts.append(text)
                                        
                                        progress_bar.progress(1.0)
                                        
                                        if len(test_embeddings) == 0:
                                            st.error("❌ Failed to generate embeddings. Please try again.")
                                        else:
                                            # Detect anomalies using trained One-Class model
                                            if st.session_state.use_temporal_averaging:
                                                # Reset temporal window for fresh start
                                                anomaly_detector.reset_temporal_window(temporal_window_size)
                                                
                                                # Use temporal averaging
                                                results = []
                                                for embedding in test_embeddings:
                                                    is_anomaly, score = anomaly_detector.predict_with_temporal_averaging(embedding)
                                                    results.append((is_anomaly, score))
                                            else:
                                                results = anomaly_detector.detect_anomalies(test_embeddings)
                                            
                                            # Store results for threshold adjustment
                                            st.session_state.analysis_results = {
                                                'frames': frames,
                                                'results': results,
                                                'test_texts': test_texts,
                                                'test_embeddings': test_embeddings
                                            }
                                            
                                            # Calculate statistics with current threshold (decision boundary is 0 for One-Class SVM)
                                            anomaly_count = sum(1 for is_anomaly, _ in results if is_anomaly)
                                            total_frames = len(results)
                                            anomaly_percentage = (anomaly_count / total_frames) * 100
                                            
                                            # Display results
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Total Frames", total_frames)
                                            with col2:
                                                st.metric("Anomalies Detected", anomaly_count)
                                            with col3:
                                                st.metric("Anomaly Rate", f"{anomaly_percentage:.1f}%")
                                            
                                            # Final decision
                                            if anomaly_count > 0:
                                                st.error(f"🚨 ANOMALY DETECTED: {anomaly_count} anomalous frames found!")
                                            else:
                                                st.success("✅ NO ANOMALIES: Video appears normal")
                                            
                                            # Decision score graph (from One-Class SVM)
                                            st.subheader("📊 Decision Scores Over Time")
                                            decision_scores = [score for _, score in results]
                                            frame_indices = list(range(len(decision_scores)))
                                            
                                            fig = go.Figure()
                                            fig.add_trace(go.Scatter(
                                                x=frame_indices,
                                                y=decision_scores,
                                                mode='lines+markers',
                                                name='Decision Score',
                                                line=dict(color='blue', width=2)
                                            ))
                                            fig.add_hline(
                                                y=0,
                                                line_dash="dash",
                                                line_color="red",
                                                annotation_text="Decision Boundary (0 = normal, <0 = anomaly)"
                                            )
                                            fig.update_layout(
                                                title="Frame-by-Frame Decision Scores (One-Class SVM)",
                                                xaxis_title="Frame Index",
                                                yaxis_title="Decision Score",
                                                height=400
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Threshold Adjustment Section
                                            st.subheader("🎚️ Adjust Decision Boundary")
                                            st.markdown("""
                                            **Note:** The One-Class SVM uses a decision boundary at 0. 
                                            You can adjust the visualization threshold to see different classification results.
                                            Negative scores indicate anomalies, positive scores indicate normal behavior.
                                            """)
                                            
                                            # Store original results and allow threshold adjustment
                                            adjusted_threshold = st.slider(
                                                "Visualization Threshold",
                                                min_value=-2.0,
                                                max_value=2.0,
                                                value=0.0,
                                                step=0.1,
                                                help="Adjust this to see how many frames would be flagged at different thresholds"
                                            )
                                            
                                            # Recalculate with adjusted threshold
                                            adjusted_anomaly_count = sum(1 for _, score in results if score < adjusted_threshold)
                                            adjusted_percentage = (adjusted_anomaly_count / total_frames) * 100
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric("Anomalies at Threshold", adjusted_anomaly_count, delta=adjusted_anomaly_count - anomaly_count)
                                            with col2:
                                                st.metric("Anomaly Rate", f"{adjusted_percentage:.1f}%", delta=f"{adjusted_percentage - anomaly_percentage:.1f}%")
                                            
                                            # Update graph with adjusted threshold line
                                            fig_adjusted = go.Figure()
                                            fig_adjusted.add_trace(go.Scatter(
                                                x=frame_indices,
                                                y=decision_scores,
                                                mode='lines+markers',
                                                name='Decision Score',
                                                line=dict(color='blue', width=2)
                                            ))
                                            fig_adjusted.add_hline(
                                                y=0,
                                                line_dash="dash",
                                                line_color="red",
                                                annotation_text="Decision Boundary (0)"
                                            )
                                            fig_adjusted.add_hline(
                                                y=adjusted_threshold,
                                                line_dash="dot",
                                                line_color="orange",
                                                annotation_text=f"Visualization Threshold ({adjusted_threshold:.1f})"
                                            )
                                            fig_adjusted.update_layout(
                                                title="Decision Scores with Adjustable Threshold",
                                                xaxis_title="Frame Index",
                                                yaxis_title="Decision Score",
                                                height=400
                                            )
                                            st.plotly_chart(fig_adjusted, use_container_width=True)
                                            
                                            # Display frames with annotations (using adjusted threshold)
                                            st.subheader("🎬 Frames with Anomaly Detection")
                                            cols_per_row = 4
                                            for i in range(0, len(frames), cols_per_row):
                                                cols = st.columns(cols_per_row)
                                                for j, col in enumerate(cols):
                                                    idx = i + j
                                                    if idx < len(frames) and idx < len(results):
                                                        is_anomaly, decision_score = results[idx]
                                                        # Use adjusted threshold for display
                                                        display_anomaly = decision_score < adjusted_threshold
                                                        with col:
                                                            try:
                                                                # Ensure frame is a valid PIL Image
                                                                frame_to_display = frames[idx]
                                                                if isinstance(frame_to_display, Image.Image):
                                                                    st.image(
                                                                        frame_to_display,
                                                                        caption=f"Frame {idx+1}",
                                                                        use_container_width=True
                                                                    )
                                                                    if display_anomaly:
                                                                        st.error(f"🚨 Anomaly (Score: {decision_score:.3f})")
                                                                    else:
                                                                        st.success(f"✅ Normal (Score: {decision_score:.3f})")
                                                                    st.caption(f"**Description:** {test_texts[idx] if idx < len(test_texts) else 'N/A'}")
                                                                else:
                                                                    st.error(f"Invalid frame {idx+1}")
                                                            except Exception as e:
                                                                st.error(f"Error displaying frame {idx+1}: {e}")
                                            
                        except Exception as e:
                            st.error(f"❌ Error processing test video: {e}")
                            st.exception(e)
        else:
            # Model not loaded - show message
            st.info("⏳ Waiting for model to be loaded...")

# ============================================================================
# LIVE CAMERA MODE
# ============================================================================
elif mode == "📷 Live Camera Mode":
    st.header("📷 Live Camera Mode")
    
    # Check if model is trained and loaded
    model_name_to_check = st.session_state.model_name or "anomaly_detector"
    
    # Ensure model is actually loaded in the detector object
    if st.session_state.model_trained:
        if not anomaly_detector.is_trained:
            # Model flag is set but detector doesn't have it loaded - try to load
            if anomaly_detector.load_model(model_name_to_check):
                st.info(f"🔄 Reloaded model: {model_name_to_check}")
            else:
                st.error(f"❌ Model '{model_name_to_check}' not found. Please retrain the model.")
                st.session_state.model_trained = False
        else:
            # Model is loaded and ready
            st.success(f"✅ Model ready: {model_name_to_check}")
    else:
        # Try to load a default model from disk
        if anomaly_detector.load_model("anomaly_detector"):
            st.session_state.model_trained = True
            st.session_state.model_name = "anomaly_detector"
            st.success("✅ Loaded trained model: anomaly_detector")
        else:
            st.warning("⚠️ Please train a One-Class model first in 'Upload Video Mode' before using live camera.")
    
    # Only proceed if model is actually loaded
    if anomaly_detector.is_trained:
        
        # Initialize similarity history in session state
        if 'similarity_history' not in st.session_state:
            st.session_state.similarity_history = []
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
        
        # Use Streamlit's camera input
        camera_image = st.camera_input("Take a picture for anomaly detection")
        
        if camera_image is not None:
            # Convert to PIL Image
            pil_image = Image.open(camera_image)
            
            # Process frame
            with st.spinner("Processing frame..."):
                text, embedding = vlm_encoder.get_frame_embedding(
                    pil_image, 
                    expected_objects=st.session_state.expected_objects
                )
                
                # Detect anomaly using trained model
                if st.session_state.use_temporal_averaging:
                    is_anomaly, decision_score = anomaly_detector.predict_with_temporal_averaging(embedding)
                else:
                    is_anomaly, decision_score = anomaly_detector.predict(embedding)
                
                # Convert decision score to similarity-like value for display
                similarity = 1.0 / (1.0 + np.exp(-decision_score))  # Sigmoid normalization
                
                # Update history (keep last 50 frames)
                st.session_state.similarity_history.append(similarity)
                if len(st.session_state.similarity_history) > 50:
                    st.session_state.similarity_history.pop(0)
                
                st.session_state.frame_count += 1
            
            # Display status
            col1, col2 = st.columns([2, 1])
            with col1:
                if is_anomaly:
                    st.error(f"🚨 **ANOMALY DETECTED!**")
                    st.write(f"**Similarity Score:** {similarity:.3f} (below threshold {similarity_threshold})")
                else:
                    st.success(f"✅ **Normal**")
                    st.write(f"**Similarity Score:** {similarity:.3f}")
                st.write(f"**Detected Text:** {text}")
                st.write(f"**Frame Count:** {st.session_state.frame_count}")
            
            with col2:
                st.metric("Similarity", f"{similarity:.3f}", delta=f"{similarity - similarity_threshold:.3f}")
            
            # Update graph
            if len(st.session_state.similarity_history) > 1:
                st.subheader("📊 Real-time Similarity History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.similarity_history,
                    mode='lines+markers',
                    name='Similarity',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                fig.add_hline(
                    y=similarity_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({similarity_threshold})"
                )
                fig.update_layout(
                    title="Similarity Scores Over Time",
                    xaxis_title="Frame",
                    yaxis_title="Cosine Similarity",
                    height=300,
                    yaxis_range=[0, 1],
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Auto-refresh option
            st.info("💡 **Tip:** The camera will automatically capture a new frame when you interact with the page. For continuous monitoring, use the refresh button or enable auto-refresh.")
            
            # Auto-refresh checkbox
            auto_refresh = st.checkbox("🔄 Enable Auto-refresh (refreshes every 2 seconds)", value=False)
            if auto_refresh:
                time.sleep(2)
                st.rerun()
    else:
        # Model not loaded - show message
        st.info("⏳ Waiting for model to be loaded...")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Anomaly Detection System using Vision-Language Models | Built with Streamlit & OpenCLIP</p>
    </div>
""", unsafe_allow_html=True)

