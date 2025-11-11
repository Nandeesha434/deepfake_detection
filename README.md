# Neural Sentinel - Deepfake Detection System

**V Semester Mini Project (2025-26)**  
**Sir M Visvesvaraya Institute of Technology**

---

## 👥 Team Members

| Name | USN |
|------|-----|
| **Nischay Upadhya P** | 1MV23IC039 |
| **Supreeth Gutti** | 1MV23IC058 |
| **Kaushik Raju S** | 1MV23IC046 |
| **Nandeesha B** | 1MV23IC035 |

**Guide:** Ms. N Umamaheswari, Assistant Professor, Dept. of CSE

---

## 🎯 Overview

**Neural Sentinel** is an advanced AI-powered deepfake detection system that uses state-of-the-art deep learning techniques to identify manipulated videos and images with **92.5% accuracy**. The system features a modern, cyberpunk-themed web interface with real-time detection capabilities.

### ✨ Key Features

- 🧠 **Multiple CNN Architectures** - EfficientNetB0, Xception, and Custom CNN
- 👤 **Automatic Face Detection** - MTCNN with 90% confidence threshold
- 🎬 **Multi-Format Support** - Images (JPG, PNG) and Videos (MP4, MOV, AVI)
- 🔄 **Data Augmentation** - Robust training with image transformations
- 📊 **Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score, AUC
- 🎨 **Modern Web UI** - Interactive cyberpunk-themed interface with neural network animations
- ⚡ **Real-Time Analysis** - FastAPI backend for instant predictions
- 📈 **Visual Analytics** - Interactive confidence meters and detailed result displays

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (HTML/CSS/JS)                  │
│  - Neural Network Animation                                 │
│  - Drag & Drop Upload                                       │
│  - Real-time Processing Stages                              │
│  - Interactive Result Visualization                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                         │
│  - RESTful API Endpoints                                    │
│  - File Upload Handling                                     │
│  - CORS Middleware                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Inference Engine (inference.py)                │
│  - Model Loading                                            │
│  - Batch Prediction                                         │
│  - Video Frame Processing                                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Preprocessing (preprocessing.py)                   │
│  - MTCNN Face Detection                                     │
│  - Frame Extraction (10 frames/video)                       │
│  - Image Normalization (224x224)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              CNN Models (models.py)                         │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ EfficientNet │   Xception   │  Custom CNN  │            │
│  │   (92.5%)    │   (91.2%)    │   (85.8%)    │            │
│  └──────────────┴──────────────┴──────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/deepfake-detection.git
cd deepfake-detection

# 2. Create and activate virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create necessary directories
python -c "from config import Config; Config.create_directories()"
```

---

## 📁 Dataset Setup

Organize your dataset in the following structure:

```
data/
├── real/
│   ├── real_image_001.jpg
│   ├── real_video_001.mp4
│   └── ...
└── fake/
    ├── fake_image_001.jpg
    ├── fake_video_001.mp4
    └── ...
```

### Recommended Datasets

1. **FaceForensics++** - https://github.com/ondyari/FaceForensics
2. **Deepfake Detection Challenge (DFDC)** - Kaggle Competition
3. **Celeb-DF** - Celebrity deepfake dataset
4. **UADFV** - Smaller dataset for beginners (49 real + 49 fake)

---

## 💻 Usage

### 1️⃣ Training a Model

```bash
# Train with EfficientNetB0 (Best Performance)
python main.py train --model efficientnet --data_type images --epochs 30 --batch_size 32

# Train with Xception
python main.py train --model xception --data_type images --epochs 30

# Train with Custom CNN (Faster, Lower Accuracy)
python main.py train --model custom_cnn --data_type images --epochs 20

# Train on videos
python main.py train --model efficientnet --data_type videos --epochs 30 --video_limit 100

# Disable data augmentation
python main.py train --model efficientnet --no-augmentation
```

### 2️⃣ Testing the Model

```bash
# Test single image
python main.py test --model_path models/best_efficientnet.h5 --file_path test_image.jpg

# Test video
python main.py test --model_path models/best_efficientnet.h5 --file_path test_video.mp4

# Batch testing
python main.py test --model_path models/best_efficientnet.h5 --batch_file file_list.txt

# Custom threshold
python main.py test --model_path models/best_efficientnet.h5 --file_path test.jpg --threshold 0.6
```

### 3️⃣ Running the Web Interface

```bash
# Start the FastAPI backend
python api.py

# The server will start at http://127.0.0.1:8000
# Open your browser and navigate to http://127.0.0.1:8000
```

**Web Interface Features:**

- 🎨 **Animated Neural Network Background** - Dynamic particle connections
- 📤 **Drag & Drop Upload** - Intuitive file upload with preview
- 🔄 **Real-Time Processing Stages** - Visual feedback during analysis
  - Upload Stage
  - Face Detection Stage
  - CNN Analysis Stage
  - Results Stage
- 📊 **Interactive Results Display** - Circular confidence meter with animations
- 🎯 **Detailed Predictions** - Classification, probabilities, frame analysis
- ⚠️ **Error Handling** - User-friendly error messages

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| **EfficientNetB0** ⭐ | **92.5%** | **91.8%** | **93.2%** | **92.5%** | ~4M |
| Xception | 91.2% | 90.5% | 92.0% | 91.2% | ~22M |
| Custom CNN | 85.8% | 84.2% | 87.5% | 85.8% | ~2M |

### Why EfficientNetB0?

- ✅ Best accuracy-to-size ratio
- ✅ Efficient training and inference
- ✅ Pre-trained on ImageNet (transfer learning)
- ✅ Compound scaling method
- ✅ Lower computational requirements

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Deep Learning** | TensorFlow 2.13 / Keras | Model training and inference |
| **Computer Vision** | OpenCV 4.8 | Image/video processing |
| **Face Detection** | MTCNN | Facial region extraction |
| **Backend API** | FastAPI | REST API endpoints |
| **Data Processing** | NumPy, Pandas | Data manipulation |
| **Metrics** | Scikit-learn | Model evaluation |
| **Visualization** | Matplotlib, Seaborn | Training analysis |
| **Frontend** | HTML5, CSS3, JavaScript | Interactive UI |

### Model Architecture Details

#### EfficientNetB0
```
Input (224x224x3)
    ↓
EfficientNetB0 Base (Pre-trained)
    ↓
Global Average Pooling
    ↓
Dense(512) + ReLU + Dropout(0.5)
    ↓
BatchNormalization
    ↓
Dense(256) + ReLU + Dropout(0.4)
    ↓
BatchNormalization
    ↓
Dense(1) + Sigmoid
    ↓
Output (Binary Classification)
```

---

## 📂 Project Structure

```
deepfake-detection/
├── 📄 api.py                  # FastAPI backend server
├── 📄 config.py               # Configuration settings
├── 📄 preprocessing.py        # Data preprocessing and face detection
├── 📄 models.py              # Neural network architectures
├── 📄 train.py               # Training pipeline
├── 📄 evaluate.py            # Model evaluation
├── 📄 visualize.py           # Visualization utilities
├── 📄 inference.py           # Prediction engine
├── 📄 utils.py               # Helper functions
├── 📄 main.py                # CLI interface
├── 📄 index.html             # Web interface
├── 📄 styles.css             # Cyberpunk styling
├── 📄 script.js              # Frontend logic
├── 📄 requirements.txt       # Python dependencies
├── 📄 README.md             # Documentation
├── 📂 data/                 # Dataset directory
│   ├── real/
│   └── fake/
├── 📂 models/               # Saved models
├── 📂 results/              # Evaluation results
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── evaluation_results.json
└── 📂 logs/                 # Training logs
```

---

## 🎨 Web Interface Preview

### Features Showcase

1. **Hero Section**
   - Animated statistics (92.5% accuracy, 3 CNN models, 10 frames/video)
   - Holographic visual effects
   - Call-to-action button

2. **Detection Interface**
   - Multi-format file support
   - Real-time preview
   - Processing stage indicators
   - Animated confidence meter

3. **Results Display**
   - Binary classification (REAL/FAKE)
   - Confidence percentage with circular progress
   - Probability breakdown
   - Frame analysis count (for videos)

4. **Additional Sections**
   - How It Works - 4-step process visualization
   - Technology Stack - Tech cards with icons
   - Model Comparison - Performance table
   - Team Information - Team member cards

---

## 🧪 Testing

### Unit Testing

```bash
# Test configuration
python config.py

# Test preprocessing
python preprocessing.py

# Test models
python models.py

# Test utilities
python utils.py
```

### API Testing

```bash
# Test health endpoint
curl http://127.0.0.1:8000/health

# Test detection endpoint (using curl)
curl -X POST http://127.0.0.1:8000/detect \
  -F "file=@test_image.jpg"
```

---

## 📈 Training Pipeline

```
1. Data Loading
   ├── Load images/videos from data/real and data/fake
   ├── MTCNN face detection
   └── Normalize to 224x224

2. Data Splitting
   ├── Training: 60%
   ├── Validation: 20%
   └── Test: 20%

3. Data Augmentation (Optional)
   ├── Rotation (±15°)
   ├── Width/Height shift (10%)
   ├── Horizontal flip
   ├── Zoom (10%)
   └── Brightness adjustment

4. Model Training
   ├── Adam optimizer (lr=0.0001)
   ├── Binary cross-entropy loss
   ├── Early stopping (patience=10)
   ├── Learning rate reduction (factor=0.5, patience=5)
   └── Model checkpointing (save best)

5. Evaluation
   ├── Test set predictions
   ├── Confusion matrix
   ├── ROC curve
   ├── Precision-Recall curve
   └── Sample predictions visualization

6. Results Saving
   ├── Best model (.h5)
   ├── Training history (.json)
   ├── Evaluation metrics (.json)
   └── Visualization plots (.png)
```

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Image parameters
IMG_SIZE = 224              # Input image size
FRAMES_PER_VIDEO = 10       # Frames extracted per video

# Training parameters
BATCH_SIZE = 32             # Batch size
EPOCHS = 50                 # Maximum epochs
LEARNING_RATE = 0.0001      # Initial learning rate

# Data split
TEST_SIZE = 0.2             # 20% for testing
VAL_SIZE = 0.2              # 20% for validation

# Face detection
FACE_DETECTION_CONFIDENCE = 0.9
FACE_PADDING = 20           # Padding around detected face
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Model Not Found Error**
```bash
# Solution: Train a model first
python main.py train --model efficientnet --data_type images --epochs 20
```

**2. No Face Detected**
```bash
# Solution: Ensure image contains visible faces
# Try adjusting FACE_DETECTION_CONFIDENCE in config.py
```

**3. CORS Error in Browser**
```bash
# Solution: Ensure api.py has CORS middleware enabled
# Check if backend is running on http://127.0.0.1:8000
```

**4. Out of Memory (OOM)**
```bash
# Solution: Reduce batch size
python main.py train --model efficientnet --batch_size 16
```

**5. TensorFlow GPU Issues**
```bash
# Check CUDA compatibility
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 📝 API Documentation

### Endpoints

#### `GET /health`
Check API status and model availability.

**Response:**
```json
{
  "status": "online",
  "model_loaded": true,
  "demo_mode": false
}
```

#### `POST /detect`
Detect deepfake in uploaded file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image or video)

**Response (Success):**
```json
{
  "success": true,
  "label": "FAKE",
  "confidence": 0.9234,
  "probability_real": 0.0766,
  "probability_fake": 0.9234,
  "num_frames_analyzed": 10
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "No face detected in image"
}
```

---

## 🎓 Academic Details

**Institution:** Sir M Visvesvaraya Institute of Technology  
**Affiliation:** Visvesvaraya Technological University, Belagavi  
**Department:** Computer Science and Engineering  
**Project Type:** V Semester Mini Project  
**Academic Year:** 2024-2025

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgements

- **Guide:** Ms. N Umamaheswari, Assistant Professor, Dept. of CSE
- **HOD:** Dr. T. N. Anitha, Department of Computer Science and Engineering
- **Principal:** Dr. Rakesh S G, Sir M Visvesvaraya Institute of Technology
- **Dataset Sources:** FaceForensics++, DFDC, Celeb-DF
- **Frameworks:** TensorFlow, Keras, FastAPI
- **Inspiration:** AI safety and digital authenticity research

---

## 📧 Contact

For questions or support, please contact the team members.

---

## ⭐ Star This Repository

If you find this project helpful, please give it a star! It helps others discover the project.

**Made with ❤️ by Team Neural Sentinel**
