# Face-Mask-Detection
📌 Overview

This project implements a real-time Face Mask Detection System using Transfer Learning on MobileNetV2, a lightweight deep neural network architecture optimized for edge and mobile devices.
The model classifies whether a person is wearing a mask or no mask in both images and real-time video streams.

The entire pipeline covers data preprocessing, model training, evaluation, and deployment (via OpenCV webcam inference).

⸻

🚀 Key Highlights
	•	Model Architecture: MobileNetV2 (pretrained on ImageNet) fine-tuned for binary classification.
	•	Frameworks Used: TensorFlow/Keras + OpenCV.
	•	Inference Speed: ~25 FPS on CPU (depending on webcam and system specs).
	•	Use Cases:
	•	COVID-19 safety monitoring
	•	Smart surveillance systems
	•	Workplace or public space compliance

⸻

🧩 Project Structure

face-mask-detection/
│
├── dataset/
│   ├── train/
│   │   ├── mask/
│   │   └── no_mask/
│   └── val/
│       ├── mask/
│       └── no_mask/
│
├── models/
│   ├── mask_detector.h5
│   └── training_plot.png
│
├── train_mask_detector.py
├── real_time_mask_detection.py
├── requirements.txt
├── README.md
└── utils.py  (optional helper functions)


⸻

⚙ Installation

1️⃣ Create a virtual environment

python -m venv venv
source venv/bin/activate     # on macOS/Linux
venv\Scripts\activate        # on Windows

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Verify TensorFlow installation

python -c "import tensorflow as tf; print(tf._version_)"


⸻

📂 Dataset

You can use any public face mask dataset, e.g.:
	•	Kaggle Face Mask Detection Dataset
	•	RMFD (Real-World Masked Face Dataset)

Ensure the folder structure matches this pattern:

dataset/
  train/
    mask/
    no_mask/
  val/
    mask/
    no_mask/

Each image should be a clear face image labeled accordingly.

⸻

🧪 Model Training

Run the training script:

python train_mask_detector.py

Training features:
	•	Transfer learning from MobileNetV2 pretrained on ImageNet
	•	Data augmentation (rotation, zoom, shift, shear, flip)
	•	Early stopping and learning rate scheduling
	•	Model checkpointing (best validation accuracy saved)

After training, the model will be saved as:

models/mask_detector.h5

and a training visualization as

models/training_plot.png


⸻

📊 Evaluation Metrics

Metric	Description
Accuracy	Overall correctness of the model
Precision	% of predicted “mask” that were correct
Recall	% of actual “mask” detected correctly
F1-Score	Harmonic mean of precision and recall

You can extend the script with sklearn.metrics.classification_report() to compute these values.

⸻

📸 Real-Time Detection

Run real-time inference from your webcam:

python real_time_mask_detection.py

Controls:
	•	Press Q to quit.
	•	Green box = Mask detected
	•	Red box = No Mask detected

⸻

🧮 Model Architecture

Base: MobileNetV2 (frozen convolutional base)
Head:

GlobalAveragePooling2D
Dropout(0.3)
Dense(128, activation='relu')
Dropout(0.3)
Dense(1, activation='sigmoid')

	•	Optimizer: Adam (lr = 1e-4)
	•	Loss: Binary Cross-Entropy
	•	Batch Size: 32
	•	Image Size: 224 × 224 × 3

⸻

🔍 Results

Metric	Training	Validation
Accuracy	~98%	~96%
Loss	~0.08	~0.11

(Values depend on dataset and training duration)

⸻

🧠 Technical Insights
	•	Transfer Learning: Using pretrained MobileNetV2 significantly reduces training time while maintaining high accuracy.
	•	Fine-Tuning Strategy: Initially froze all convolutional layers, then optionally unfroze top N layers for domain adaptation.
	•	Data Augmentation: Increases generalization and reduces overfitting on limited datasets.
	•	Lightweight Deployment: Model optimized for real-time CPU inference with OpenCV (no GPU required).

⸻

☁ Deployment Ideas
	•	Web App: Use Streamlit or Flask for web-based image upload and detection.
	•	Edge Devices: Convert .h5 model to .tflite using TensorFlow Lite for deployment on Raspberry Pi or Android.
	•	Dockerization: Create a Dockerfile with Python + dependencies for production-ready containerization.

Example TFLite conversion snippet:

import tensorflow as tf
model = tf.keras.models.load_model("models/mask_detector.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("models/mask_detector.tflite", "wb").write(tflite_model)


⸻

🔧 Future Enhancements
	•	Integrate MTCNN / DNN for more accurate face detection
	•	Add multi-class support (mask, incorrect mask, no mask)
	•	Implement Grad-CAM for model explainability
	•	Add MLOps pipeline for retraining + CI/CD integration
	•	Deploy via Streamlit Cloud or AWS Lambda

⸻

📚 Tech Stack

Category	Tools
Language	Python 3.9+
Deep Learning	TensorFlow / Keras
Computer Vision	OpenCV
Data Manipulation	NumPy, Matplotlib
Model Deployment	Flask / Streamlit (optional)

====================================================================
