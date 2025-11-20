# 🎓 VLM-Based Anomaly Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

**A production-ready anomaly detection system using Vision-Language Models (VLM) and One-Class SVM for video surveillance and monitoring applications.**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-system-architecture)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements an **academic-grade anomaly detection system** that leverages Vision-Language Models (VLM) to identify anomalous behaviors in video sequences. The system follows a **one-class learning approach**, training exclusively on normal video data and detecting deviations from learned normal patterns.

### Key Highlights

- ✅ **One-Class Learning**: Trains on normal data only (no labeled anomalies required)
- ✅ **VLM-Powered**: Uses OpenCLIP for semantic understanding of video frames
- ✅ **Production-Ready**: Full Streamlit dashboard with real-time capabilities
- ✅ **Scalable**: Supports multiple normal videos for robust baseline creation
- ✅ **Feature Scaling**: Proper StandardScaler implementation for consistent embeddings

### Use Cases

- 🏠 **Home Security**: Detect unusual activities in surveillance footage
- 🏢 **Office Monitoring**: Identify unauthorized access or suspicious behavior
- 🏥 **Healthcare**: Monitor patient rooms for falls or emergencies
- 🏭 **Industrial**: Detect equipment malfunctions or safety violations
- 🎓 **Research**: Academic research on video anomaly detection

---

## ✨ Features

### Phase 1: Training (Offline Video Analysis)

- **Multi-Video Training**: Upload multiple normal videos to build a comprehensive baseline
- **Intelligent Frame Extraction**: Configurable frame intervals with automatic caching
- **VLM Text Extraction**: Action-focused descriptions for each frame (e.g., "a cat grooming itself")
- **Embedding Generation**: High-dimensional semantic embeddings using OpenCLIP
- **One-Class SVM Training**: Train robust anomaly detection model on normal embeddings
- **Feature Standardization**: Automatic StandardScaler for consistent feature scaling
- **Model Persistence**: Save/load trained models for reuse
- **Description Validation**: Review VLM-generated descriptions before training

### Phase 2: Testing (Anomaly Detection)

- **Video Upload**: Test new videos against trained model
- **Real-Time Analysis**: Frame-by-frame anomaly detection
- **Decision Scores**: One-Class SVM decision function scores
- **Visual Analytics**: Interactive similarity graphs and frame annotations
- **Statistical Reports**: Anomaly count, percentage, and frame-level details

### Phase 3: Real-Time Detection (Live Camera)

- **Webcam Integration**: Real-time anomaly detection from live camera feed
- **Live Visualization**: Real-time decision scores and alerts
- **Auto-Refresh**: Continuous monitoring with configurable refresh rates
- **Instant Alerts**: Visual and textual anomaly notifications

### Additional Features

- **Model Management**: Save, load, and manage multiple trained models
- **Caching System**: Intelligent frame caching for faster repeated processing
- **Error Handling**: Robust error handling with user-friendly messages
- **Progress Tracking**: Real-time progress bars for long operations
- **Responsive UI**: Clean, modern Streamlit interface

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
│  (Upload Videos | Train Model | Test Videos | Live Camera)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌───────────────┐ ┌──────────────┐ ┌──────────────┐
│ Video         │ │ VLM          │ │ Anomaly      │
│ Processing    │ │ Encoder      │ │ Detection    │
│               │ │              │ │              │
│ • Frame       │ │ • OpenCLIP   │ │ • One-Class  │
│   Extraction  │ │ • Text       │ │   SVM        │
│ • Caching     │ │   Extraction │ │ • Standard  │
│ • Validation  │ │ • Embeddings │ │   Scaler     │
└───────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   Assets Storage     │
            │ • Models (.pkl)      │
            │ • Baselines (.npy)   │
            │ • Cache (.pkl)       │
            │ • Uploads (videos)   │
            └──────────────────────┘
```

### Data Flow

1. **Training Phase**:
   ```
   Normal Videos → Frame Extraction → VLM Encoding → Embeddings → 
   StandardScaler → One-Class SVM → Trained Model + Scaler
   ```

2. **Testing Phase**:
   ```
   Test Video → Frame Extraction → VLM Encoding → Embeddings → 
   StandardScaler.transform() → One-Class SVM.predict() → Anomaly Decision
   ```

3. **Real-Time Phase**:
   ```
   Camera Frame → VLM Encoding → Embedding → Scaler → Model → 
   Real-Time Decision → Alert (if anomaly)
   ```

---

## 🛠️ Tech Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **Streamlit** | ≥1.28.0 | Web dashboard framework |
| **PyTorch** | ≥2.0.0 | Deep learning framework |
| **OpenCLIP** | ≥2.20.0 | Vision-Language Model |
| **scikit-learn** | ≥1.3.0 | One-Class SVM, StandardScaler |
| **OpenCV** | ≥4.8.0 | Video processing |
| **Plotly** | ≥5.17.0 | Interactive visualizations |

### Key Libraries

- **open-clip-torch**: OpenCLIP implementation for VLM
- **Pillow**: Image processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation (for description tables)

---

## 📦 Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager
- **Git** (for cloning)
- **Webcam** (optional, for live camera mode)
- **CUDA-capable GPU** (optional, for faster processing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Seif-rashwan/VLM-Anomaly-Detection.git
cd VLM-Anomaly-Detection
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first installation will download OpenCLIP model weights (~500MB-2GB depending on model), which may take 5-10 minutes.

### Step 4: Verify Installation

```bash
python -c "import streamlit; import torch; import open_clip; print('✅ All dependencies installed!')"
```

---

## 🚀 Quick Start

### 1. Launch the Application

```bash
streamlit run streamlit_app.py
```

The application will automatically open in your browser at `http://localhost:8501`.

### 2. Train a Model (Phase 1)

1. **Upload Normal Videos**:
   - Go to "📹 Upload Video Mode" → "📌 Create Baseline" tab
   - Upload **multiple** normal videos (MP4, AVI, MOV, MKV)
   - Click "📥 Process & Add Videos to Training Set"
   - Repeat until you have collected enough normal data

2. **Review Descriptions** (Optional):
   - Expand "📝 Extracted VLM Descriptions (For Validation)"
   - Review frame descriptions to ensure accuracy

3. **Train One-Class Model**:
   - Click "🎓 Train One-Class SVM Model"
   - Model will be saved automatically

### 3. Test a Video (Phase 2)

1. Go to "🔍 Test Video" tab
2. Upload a test video
3. Click "🔍 Analyze Test Video"
4. View results: anomaly statistics, graphs, and frame annotations

### 4. Real-Time Detection (Phase 3)

1. Switch to "📷 Live Camera Mode"
2. Click camera to capture frames
3. System will detect anomalies in real-time

---

## 📖 Usage Guide

### Detailed Workflow

#### Phase 1: Training on Normal Videos

**Objective**: Build a robust baseline of normal behavior.

1. **Collect Normal Videos**:
   - Gather 5-20+ videos showing normal behavior
   - Ensure videos are representative of expected normal scenarios
   - Videos can be of different lengths and scenes

2. **Process Videos**:
   - Upload videos one batch at a time
   - System extracts frames at configured intervals (default: 500ms)
   - Each frame is processed through VLM to generate embeddings
   - All embeddings are collected in a training set

3. **Validate Descriptions**:
   - Review VLM-generated descriptions
   - Check for hallucination or incorrect descriptions
   - Ensure descriptions match actual frame content

4. **Train Model**:
   - Configure model name and `nu` parameter (outlier fraction)
   - Click "Train One-Class SVM Model"
   - Model and scaler are saved to `assets/models/`

**Best Practices**:
- Use **10+ normal videos** for robust training
- Ensure videos cover **diverse normal scenarios**
- Review descriptions before training
- Use consistent VLM model for training and testing

#### Phase 2: Testing New Videos

**Objective**: Detect anomalies in test videos.

1. **Load Trained Model**:
   - System automatically loads the trained model
   - Verify "✅ Model ready" message appears

2. **Upload Test Video**:
   - Upload video to test (normal or anomalous)
   - System processes frame-by-frame

3. **Analyze Results**:
   - **Anomaly Statistics**: Total frames, anomalies detected, anomaly rate
   - **Decision Scores Graph**: Visual representation of frame-by-frame scores
   - **Frame Annotations**: Each frame labeled as normal/anomaly with score

**Interpreting Results**:
- **Decision Score > 0**: Normal (closer to training data)
- **Decision Score < 0**: Anomaly (deviates from training data)
- **Lower scores** = More anomalous
- **Higher scores** = More normal

#### Phase 3: Real-Time Camera Detection

**Objective**: Monitor live camera feed for anomalies.

1. **Prerequisites**:
   - Trained model must exist
   - Camera must be connected and accessible

2. **Start Detection**:
   - Switch to "📷 Live Camera Mode"
   - Camera input will appear
   - Capture frames for analysis

3. **Monitor Results**:
   - Real-time decision scores displayed
   - Anomaly alerts appear immediately
   - Historical graph shows score trends

---

## 🧠 Model Architecture

### Vision-Language Model (VLM)

**Model**: OpenCLIP (CLIP implementation)

**Architecture**:
- **Vision Encoder**: Vision Transformer (ViT)
  - Variants: ViT-B-32, ViT-L-14, ViT-B-16
  - Input: 224×224 RGB images
  - Output: 512-768 dimensional embeddings (normalized)

**Text Extraction**:
- **Method**: Zero-shot classification with action-focused prompts
- **Prompt Set**: 20+ action-specific prompts (e.g., "a cat grooming itself")
- **Output**: Single best-matching action description

**Embedding Generation**:
```python
Image → Preprocessing → ViT Encoder → Normalized Embedding (512-768 dims)
```

### One-Class SVM

**Algorithm**: One-Class Support Vector Machine (RBF kernel)

**Hyperparameters**:
- **`nu`**: Upper bound on fraction of outliers (default: 0.1 = 10%)
- **`gamma`**: RBF kernel coefficient (default: "scale")
- **`kernel`**: Radial Basis Function (RBF)

**Feature Scaling**:
- **StandardScaler**: Normalizes features to mean=0, std=1
- **Critical**: Same scaler used for training and testing
- **Persistence**: Scaler saved alongside model

**Decision Function**:
- **Output**: Distance from decision boundary
- **Positive values**: Normal (inside the learned region)
- **Negative values**: Anomaly (outside the learned region)

### Training Pipeline

```
1. Collect Normal Videos
   ↓
2. Extract Frames (configurable interval)
   ↓
3. VLM Processing:
   - Extract text description (action-focused)
   - Generate image embedding (512-768 dims)
   ↓
4. Feature Standardization:
   - Fit StandardScaler on all training embeddings
   - Transform embeddings: (X - mean) / std
   ↓
5. One-Class SVM Training:
   - Train on scaled embeddings
   - Learn decision boundary for normal data
   ↓
6. Model Persistence:
   - Save OneClassSVM model (.pkl)
   - Save StandardScaler (.pkl)
```

### Inference Pipeline

```
1. Test Video / Camera Frame
   ↓
2. Extract Frame
   ↓
3. VLM Processing:
   - Generate embedding (same model as training)
   ↓
4. Feature Scaling:
   - Transform using SAVED StandardScaler
   - Critical: Use transform(), not fit_transform()
   ↓
5. One-Class SVM Prediction:
   - decision_function() → score
   - predict() → normal (-1) or anomaly (+1)
   ↓
6. Anomaly Decision:
   - score < 0 → Anomaly
   - score > 0 → Normal
```

---

## 📊 Evaluation & Metrics

### Metrics Provided

1. **Anomaly Count**: Number of frames flagged as anomalous
2. **Anomaly Rate**: Percentage of anomalous frames
3. **Decision Scores**: Frame-by-frame distance from decision boundary
4. **Description Statistics**: VLM description quality metrics

### Interpreting Results

**Decision Score Interpretation**:
- **Score > 1.0**: Very normal (close to training data center)
- **Score 0.0 to 1.0**: Normal (within learned region)
- **Score -1.0 to 0.0**: Borderline (near decision boundary)
- **Score < -1.0**: Strong anomaly (far from normal region)

**Anomaly Rate Guidelines**:
- **0-5%**: Likely normal video
- **5-20%**: Some anomalous frames (investigate)
- **20-50%**: Significant anomalies detected
- **50%+**: Highly anomalous video

### Model Performance Factors

**Training Data Quality**:
- More diverse normal videos → Better generalization
- Representative scenarios → Lower false positives
- Sufficient quantity (10+ videos) → Robust baseline

**VLM Model Selection**:
- **ViT-B-32**: Fast, good for real-time (default)
- **ViT-L-14**: More accurate, slower processing
- **ViT-B-16**: Balanced alternative

**Hyperparameter Tuning**:
- **`nu` (0.01-0.5)**: Lower = stricter (fewer false positives, more false negatives)
- **`gamma`**: "scale" (default) or "auto" for automatic tuning

---

## 📁 Project Structure

```
VLM-Anomaly-Detection/
│
├── src/                          # Source code
│   ├── __init__.py
│   │
│   ├── video_processing/         # Video frame extraction
│   │   ├── __init__.py
│   │   └── frame_extractor.py   # Frame extraction with caching
│   │
│   ├── vlm/                      # Vision-Language Model
│   │   ├── __init__.py
│   │   └── vlm_encoder.py        # OpenCLIP integration, embeddings
│   │
│   ├── anomaly_detection/        # Anomaly detection logic
│   │   ├── __init__.py
│   │   └── detector.py           # One-Class SVM, StandardScaler
│   │
│   └── utils/                    # Helper functions
│       ├── __init__.py
│       └── helpers.py            # File I/O, video info
│
├── assets/                       # Data storage
│   ├── baselines/                # Baseline embeddings (.npy)
│   │   └── .gitkeep
│   ├── cache/                    # Cached frames (.pkl)
│   │   └── .gitkeep
│   ├── models/                   # Trained models (.pkl)
│   │   ├── {model_name}_model.pkl
│   │   ├── {model_name}_scaler.pkl
│   │   └── .gitkeep
│   └── uploads/                  # Uploaded videos
│       └── .gitkeep
│
├── streamlit_app.py              # Main Streamlit dashboard
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

### Key Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main application entry point, UI logic |
| `src/vlm/vlm_encoder.py` | VLM model loading, text extraction, embeddings |
| `src/anomaly_detection/detector.py` | One-Class SVM training, prediction, scaling |
| `src/video_processing/frame_extractor.py` | Video frame extraction with caching |
| `src/utils/helpers.py` | Utility functions for file handling |

---

## 📚 API Documentation

### VLMEncoder

**Class**: `src.vlm.VLMEncoder`

**Methods**:

#### `__init__(model_name: str = "ViT-B-32", pretrained: str = "openai")`
Initialize VLM encoder (singleton pattern).

**Parameters**:
- `model_name`: OpenCLIP model variant ("ViT-B-32", "ViT-L-14", "ViT-B-16")
- `pretrained`: Pretrained dataset ("openai", "laion400m_e32", "laion2b_s34b_b79k")

#### `extract_text_from_image(image: Image.Image) -> str`
Extract action-focused text description from image.

**Returns**: Action description string (e.g., "a cat grooming itself")

#### `get_image_embedding(image: Image.Image) -> np.ndarray`
Generate normalized image embedding.

**Returns**: NumPy array (512-768 dimensions, normalized)

#### `get_frame_embedding(image: Image.Image) -> Tuple[str, np.ndarray]`
Extract both text and embedding from frame.

**Returns**: Tuple of (description, embedding)

---

### AnomalyDetector

**Class**: `src.anomaly_detection.AnomalyDetector`

**Methods**:

#### `train_one_class_model(normal_embeddings: List[np.ndarray], model_name: str = "default", nu: float = 0.1, gamma: str = "scale") -> str`
Train One-Class SVM on normal embeddings.

**Parameters**:
- `normal_embeddings`: List of embedding arrays from normal videos
- `model_name`: Name for saving the model
- `nu`: Outlier fraction (0.1 = 10% expected outliers)
- `gamma`: RBF kernel coefficient

**Returns**: Path to saved model file

**Process**:
1. Standardizes embeddings using StandardScaler
2. Trains One-Class SVM on scaled data
3. Saves both model and scaler to disk

#### `load_model(model_name: str = "default") -> bool`
Load trained model and scaler from disk.

**Returns**: True if successful, False otherwise

#### `predict(test_embedding: np.ndarray) -> Tuple[bool, float]`
Predict if embedding is normal or anomalous.

**Parameters**:
- `test_embedding`: Single embedding array

**Returns**: Tuple of (is_anomaly: bool, decision_score: float)

**Process**:
1. Scales embedding using saved StandardScaler
2. Computes decision function score
3. Returns anomaly flag and score

#### `detect_anomalies(test_embeddings: List[np.ndarray]) -> List[Tuple[bool, float]]`
Detect anomalies in multiple embeddings.

**Returns**: List of (is_anomaly, decision_score) tuples

---

### FrameExtractor

**Class**: `src.video_processing.FrameExtractor`

**Methods**:

#### `extract_frames(video_path: str, frame_interval_ms: int = 500, use_cache: bool = True) -> List[Image.Image]`
Extract frames from video at specified intervals.

**Parameters**:
- `video_path`: Path to video file
- `frame_interval_ms`: Milliseconds between frames (default: 500ms)
- `use_cache`: Whether to use cached frames

**Returns**: List of PIL Images

**Features**:
- Automatic caching based on video hash
- Always extracts first frame
- Validates frame dimensions

---

## ⚙️ Configuration

### Model Settings

**VLM Model Selection**:
- **ViT-B-32** (default): Balanced speed/accuracy, ~512 dims
- **ViT-L-14**: Higher accuracy, slower, ~768 dims
- **ViT-B-16**: Alternative variant, ~512 dims

**Pretrained Datasets**:
- **openai** (default): OpenAI CLIP weights (general-purpose)
- **laion400m_e32**: LAION-400M dataset
- **laion2b_s34b_b79k**: LAION-2B dataset

### Detection Parameters

**Similarity Threshold** (Legacy, for display only):
- Range: 0.0 - 1.0
- Default: 0.6
- Note: One-Class SVM uses decision scores, not similarity

**Frame Interval**:
- Range: 100 - 2000 ms
- Default: 500 ms
- Lower = more frames, slower processing
- Higher = fewer frames, faster processing

**One-Class SVM Parameters**:
- **nu**: 0.01 - 0.5 (default: 0.1)
  - Lower = stricter (fewer false positives)
  - Higher = more lenient (fewer false negatives)

### Environment Variables

No environment variables required. All configuration is done through the Streamlit UI.

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Symptoms**: Error loading OpenCLIP model

**Solutions**:
- Check internet connection
- First download may take 5-10 minutes
- Model weights are cached after first download
- Try different pretrained dataset if one fails

#### 2. "Model not trained" Error

**Symptoms**: ValueError when testing videos

**Solutions**:
- Ensure model is trained before testing
- Check that model files exist in `assets/models/`
- Verify both `{model_name}_model.pkl` and `{model_name}_scaler.pkl` exist
- Retrain model if files are missing

#### 3. 100% Anomaly Detection

**Symptoms**: All frames flagged as anomalous

**Solutions**:
- **Most Common**: Ensure test embeddings use same VLM model as training
- Verify StandardScaler is properly loaded
- Check embedding dimensions match (should be same as training)
- Retrain model with more diverse normal videos
- Lower `nu` parameter if too strict

#### 4. Camera Not Working

**Symptoms**: Camera input fails or not detected

**Solutions**:
- Ensure camera is connected and not used by another app
- Check camera permissions in browser/OS
- Try restarting Streamlit app
- Use "Upload Video Mode" as alternative

#### 5. Slow Processing

**Symptoms**: Long processing times

**Solutions**:
- Increase frame interval (fewer frames to process)
- Use smaller VLM model (ViT-B-32 instead of ViT-L-14)
- Enable GPU if available (automatic if CUDA installed)
- Process shorter videos
- Clear cache if it becomes too large

#### 6. Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
- Process videos in smaller batches
- Increase frame interval
- Close other applications
- Use CPU instead of GPU (if GPU memory limited)
- Clear cached frames: `assets/cache/`

#### 7. Embedding Dimension Mismatch

**Symptoms**: "Embedding dimension mismatch" error

**Solutions**:
- Ensure same VLM model used for training and testing
- Don't change model after training
- Retrain model if you changed VLM model

#### 8. VLM Hallucination

**Symptoms**: Incorrect frame descriptions

**Solutions**:
- Review descriptions in validation section
- Descriptions are action-focused (may not describe background)
- This is expected behavior for action-focused prompts
- Descriptions are for validation only; embeddings are what matter

---

## 🚀 Future Improvements

### Performance Enhancements

- [ ] **GPU Optimization**: Better CUDA utilization for batch processing
- [ ] **Batch Processing**: Process multiple videos in parallel
- [ ] **Model Quantization**: Reduce model size for faster inference
- [ ] **Streaming Processing**: Process videos without full loading

### Feature Additions

- [ ] **Temporal Analysis**: Sequence-based anomaly detection (LSTM/Transformer)
- [ ] **Multiple Baselines**: Support for different normal scenarios
- [ ] **Custom Thresholds**: Per-video-type sensitivity settings
- [ ] **Export Functionality**: CSV/JSON export of results
- [ ] **Video Annotation**: Export annotated videos with anomaly markers
- [ ] **Model Comparison**: Compare multiple trained models
- [ ] **A/B Testing**: Test different model configurations

### Model Improvements

- [ ] **Fine-Tuning**: Domain-specific VLM fine-tuning
- [ ] **Ensemble Methods**: Combine multiple models for better accuracy
- [ ] **Alternative Architectures**: Support for BLIP, LLaVA, etc.
- [ ] **Two-Class Classifier**: Supervised learning option (Phase 3)
- [ ] **Active Learning**: Interactive model improvement

### UI/UX Enhancements

- [ ] **Video Playback**: Built-in video player with frame navigation
- [ ] **Comparison View**: Side-by-side baseline vs test comparison
- [ ] **Advanced Filtering**: Filter results by score, time, etc.
- [ ] **Dashboard Analytics**: Historical anomaly trends
- [ ] **Mobile Support**: Responsive design for mobile devices

### Deployment

- [ ] **Docker Support**: Containerized deployment
- [ ] **Cloud Deployment**: Streamlit Cloud, AWS, GCP options
- [ ] **API Endpoint**: REST API for programmatic access
- [ ] **Database Integration**: Store results in database
- [ ] **Authentication**: User management and access control

---

## 🤝 Contributing

This is a graduation project. Contributions are welcome!

### Contribution Guidelines

1. **Fork the Repository**: Create your own fork
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Follow Code Style**: Match existing code style and comments
4. **Test Thoroughly**: Ensure all functionality works
5. **Update Documentation**: Update README if adding features
6. **Submit Pull Request**: Describe changes clearly

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where appropriate
- Add docstrings to all functions/classes
- Keep functions focused and modular

### Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Error messages (full traceback)
- Steps to reproduce
- Expected vs actual behavior

---

## 📄 License

This project is developed for **educational purposes** as part of a graduation project.

**Note**: This project uses:
- **OpenCLIP**: Licensed under Apache 2.0
- **Streamlit**: Licensed under Apache 2.0
- **scikit-learn**: Licensed under BSD 3-Clause

Please refer to individual library licenses for commercial use.

---

## 🙏 Acknowledgments

### Libraries & Frameworks

- **[OpenCLIP](https://github.com/mlfoundations/open_clip)**: Vision-Language Model implementation
- **[Streamlit](https://streamlit.io/)**: Interactive web application framework
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning utilities
- **[OpenCV](https://opencv.org/)**: Computer vision and video processing
- **[Plotly](https://plotly.com/)**: Interactive visualizations

### Research & Inspiration

- CLIP: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)
- One-Class SVM for Anomaly Detection (Schölkopf et al., 2001)
- OpenCLIP: An Open Source Implementation of CLIP

### Special Thanks

- Open source community for excellent tools and libraries
- Academic advisors for project guidance

---

## 📞 Contact & Support

- **GitHub**: [@Seif-rashwan](https://github.com/Seif-rashwan)
- **Repository**: [VLM-Anomaly-Detection](https://github.com/Seif-rashwan/VLM-Anomaly-Detection)

For questions, issues, or contributions, please open an issue on GitHub.

---

<div align="center">

**Built with ❤️ for Graduation Project**

⭐ **Star this repo if you find it useful!** ⭐

</div>
