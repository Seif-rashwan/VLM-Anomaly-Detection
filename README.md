# 🎓 Anomaly Detection System using Vision-Language Models

A comprehensive anomaly detection system that uses Vision-Language Models (VLM) to detect anomalies in videos by comparing them against a normal baseline. The system supports both offline video analysis and real-time camera detection.

## ✨ Features

### Phase 1: Video-Based Anomaly Detection (Offline Videos)
- **Baseline Creation**: Upload a normal reference video to create an embedding baseline
- **Frame Extraction**: Automatic frame extraction at configurable intervals
- **VLM Integration**: Uses OpenCLIP for text extraction and embedding generation
- **Anomaly Detection**: Compares test video embeddings against baseline using cosine similarity
- **Visual Dashboard**: 
  - Preview of video frames
  - Extracted text descriptions per frame
  - Embedding similarity graph over time
  - Final anomaly decision with statistics

### Phase 2: Real-Time Detection (Live Camera)
- **Live Camera Support**: Real-time anomaly detection from webcam
- **Mode Toggle**: Switch between upload video mode and live camera mode
- **Real-time Visualization**: Live similarity scores and anomaly alerts
- **Interactive Dashboard**: Real-time frame display with status indicators

## 🏗️ Architecture

The project follows clean architecture principles with modular design:

```
Grad 2/
├── src/
│   ├── video_processing/      # Frame extraction and caching
│   │   ├── __init__.py
│   │   └── frame_extractor.py
│   ├── vlm/                    # Vision-Language Model integration
│   │   ├── __init__.py
│   │   └── vlm_encoder.py
│   ├── anomaly_detection/      # Anomaly detection logic
│   │   ├── __init__.py
│   │   └── detector.py
│   └── utils/                  # Helper functions
│       ├── __init__.py
│       └── helpers.py
├── assets/
│   ├── baselines/              # Saved baseline embeddings
│   ├── cache/                  # Cached frame extractions
│   └── uploads/                # Uploaded video files
├── streamlit_app.py            # Main Streamlit dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Webcam (for Phase 2 - Live Camera Mode)

### Step 1: Clone or Download the Project
```bash
cd "Grad 2"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The first run will download the OpenCLIP model weights, which may take a few minutes depending on your internet connection.

## 📖 How to Run

### Start the Streamlit Application
```bash
streamlit run streamlit_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

### Usage Guide

#### Phase 1: Create Baseline and Test Videos

1. **Create Baseline**:
   - Select "📹 Upload Video Mode" (default)
   - Go to "📌 Create Baseline" tab
   - Upload a normal reference video (MP4, AVI, MOV, MKV)
   - Click "🚀 Process Baseline Video"
   - Wait for processing to complete
   - Baseline embeddings will be saved automatically

2. **Test Video**:
   - Go to "🔍 Test Video" tab
   - Upload a test video (normal or anomalous)
   - Click "🔍 Analyze Test Video"
   - View results:
     - Anomaly statistics
     - Similarity graph
     - Frame-by-frame analysis with annotations

#### Phase 2: Live Camera Detection

1. **Setup**:
   - First, create a baseline in "Upload Video Mode"
   - Switch to "📷 Live Camera Mode" in the sidebar

2. **Start Detection**:
   - Click "🎥 Start Camera"
   - Camera feed will start processing
   - Real-time anomaly alerts will appear
   - Click "⏹️ Stop Camera" to stop

### Configuration Options

In the sidebar, you can adjust:

- **VLM Model**: Choose between different OpenCLIP variants
  - `ViT-B-32`: Balanced speed/accuracy (default)
  - `ViT-L-14`: Higher accuracy, slower
  - `ViT-B-16`: Alternative variant

- **Pretrained Dataset**: Select pretrained weights
  - `openai`: OpenAI CLIP weights (default)
  - `laion400m_e32`: LAION-400M dataset
  - `laion2b_s34b_b79k`: LAION-2B dataset

- **Similarity Threshold**: Adjust sensitivity (0.0 - 1.0)
  - Lower values = more sensitive (more anomalies detected)
  - Higher values = less sensitive (fewer anomalies detected)
  - Default: 0.7

- **Frame Interval**: Time between extracted frames (100-2000 ms)
  - Lower = more frames, slower processing
  - Higher = fewer frames, faster processing
  - Default: 500ms

## 🧠 Model Explanation

### How It Works

1. **Baseline Creation**:
   - Normal video is processed frame-by-frame
   - Each frame is passed through OpenCLIP VLM
   - VLM generates:
     - Text description (e.g., "a photo of a person")
     - Image embedding (high-dimensional vector)
   - All embeddings are saved as the "normal baseline"

2. **Anomaly Detection**:
   - Test video frames are processed similarly
   - Each frame embedding is compared to baseline embeddings
   - Cosine similarity is computed (range: 0-1)
   - If similarity < threshold → flagged as anomaly

3. **Why It Works**:
   - VLM embeddings capture semantic content of images
   - Normal frames should have similar embeddings
   - Anomalous frames will have different embeddings
   - Cosine similarity measures embedding similarity

### Technical Details

- **Embedding Dimension**: Depends on model (typically 512-768 dimensions)
- **Similarity Metric**: Cosine similarity (normalized dot product)
- **Comparison Method**: Maximum similarity across all baseline embeddings
- **Caching**: Frame extractions are cached to speed up repeated runs

## 🔧 Troubleshooting

### Common Issues

1. **Model Download Fails**:
   - Check internet connection
   - First download may take several minutes
   - Model weights are cached after first download

2. **Camera Not Working**:
   - Ensure camera is connected and not used by another application
   - Try restarting the application
   - Check camera permissions

3. **Slow Processing**:
   - Reduce frame interval (extract fewer frames)
   - Use smaller model (ViT-B-32 instead of ViT-L-14)
   - Enable GPU if available (CUDA)

4. **Memory Issues**:
   - Process shorter videos
   - Increase frame interval
   - Close other applications

## 🚀 Future Improvements

### Potential Enhancements

1. **Performance**:
   - GPU acceleration optimization
   - Batch processing for multiple videos
   - Parallel frame processing

2. **Features**:
   - Multiple baseline support
   - Temporal anomaly detection (sequence-based)
   - Custom threshold per video type
   - Export results to CSV/JSON
   - Video annotation export

3. **Models**:
   - Support for other VLM architectures
   - Fine-tuning on domain-specific data
   - Ensemble methods for better accuracy

4. **UI/UX**:
   - Video playback controls
   - Frame-by-frame navigation
   - Comparison view (baseline vs test)
   - Advanced filtering options

5. **Deployment**:
   - Docker containerization
   - Cloud deployment options
   - API endpoint for programmatic access

## 📝 Notes

- **First Run**: Model download may take 5-10 minutes
- **Caching**: Frame extractions are cached in `assets/cache/`
- **Baselines**: Saved in `assets/baselines/` as `.npy` files
- **Uploads**: Videos are saved in `assets/uploads/`

## 🤝 Contributing

This is a graduation project. For improvements or bug fixes:
1. Test thoroughly
2. Document changes
3. Ensure code follows existing style

## 📄 License

This project is for educational purposes as part of a graduation project.

## 🙏 Acknowledgments

- **OpenCLIP**: For the Vision-Language Model implementation
- **Streamlit**: For the dashboard framework
- **OpenCV**: For video processing capabilities

---

**Built with ❤️ for Graduation Project**

