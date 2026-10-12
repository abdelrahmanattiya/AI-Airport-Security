# AI Airport Security ✈️

### Real-Time Face Detection, Face Recognition, and Occupancy Logging System Using Deep Learning

## 📌 Project Overview

AI Airport Security is a computer vision project designed to support airport security and monitoring through face detection, face recognition, and database-based logging.

The system uses deep learning models to detect and recognize faces in images or video streams, with the goal of supporting automated identification and occupancy monitoring.

## 🎯 Project Objectives

- Detect faces using YOLOv8.
- Recognize faces using InsightFace.
- Store and manage face-related records using SQLite.
- Support database creation from images.
- Explore GPU acceleration for faster inference.
- Provide a foundation for real-time monitoring and logging.

## 🛠️ Technologies Used

- **Python**: Main programming language.
- **YOLOv8 (Ultralytics)**: Face detection.
- **InsightFace**: Face analysis and recognition.
- **OpenCV**: Image and video processing.
- **PyTorch**: Deep learning model execution.
- **ONNX Runtime**: Model inference with GPU/CPU execution providers.
- **SQLite**: Local database management.
- **NumPy**: Numerical operations.
- **OpenPyXL**: Excel file handling.

## 📂 Project Structure

```text
AI-Airport-Security/
│
├── Recognition.py
├── build_database.py
├── build_db_from_single_image.py
├── yolov8n-face.pt
├── requirements.txt
└── README.md
```

## ⚙️ Requirements

- Python installed on your system.
- A compatible PyTorch installation.
- The required Python packages.
- An NVIDIA GPU and compatible CUDA/cuDNN dependencies if GPU acceleration is needed.

CPU execution may be possible depending on the model and installed dependencies.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/abdelrahmanattiya/AI-Airport-Security.git
cd AI-Airport-Security
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** PyTorch and ONNX Runtime GPU require compatible versions of their runtime dependencies. Refer to their official installation instructions if GPU execution is required.

## 🧩 Main Files

- **`Recognition.py`**: Face recognition and related processing.
- **`build_database.py`**: Database-building functionality.
- **`build_db_from_single_image.py`**: Database creation from a single image.
- **`yolov8n-face.pt`**: YOLOv8 face detection model weights.

## ▶️ Running the Project

The exact command depends on the script configuration and required input files.

For example, to run the recognition script:

```bash
python Recognition.py
```

Check the script configuration before running it to confirm that the required model weights, database files, and input sources are available.

## 🔐 Privacy and Security

- Use face images only when you have appropriate permission.
- Do not upload private face databases, personal information, passwords, API keys, or Firebase service-account credentials
