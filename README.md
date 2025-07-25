Early Dyslexia Detection in Children

A web-based application to detect early signs of dyslexia using Machine Learning and Deep Learning.

 Overview

This system uses:
- A Support Vector Machine (SVM) for text-based assessments (like confidence, reading fluency, etc.)
- A Convolutional Neural Network (CNN) for analyzing handwriting samples

Features

- Easy-to-use web interface
- Upload handwriting images or fill a cognitive form
- Real-time dyslexia detection with confidence scores
- Downloadable/printable result reports

Tech Stack

| Component          | Technology            |
|--------------------|------------------------|
| Language           | Python, HTML, CSS      |
| ML/DL              | scikit-learn, TensorFlow, Keras |
| Web Framework      | Flask / Django         |
| Database           | MySQL / MongoDB        |
| Image Processing   | OpenCV                 |

Project Structure

├── backend/
│ ├── app.py
│ ├── model_svm.pkl
│ ├── model_cnn.h5
│ └── utils.py
├── frontend/
│ ├── templates/
│ └── static/
├── dataset/
│ ├── text_dataset.csv
│ └── handwriting_samples/
├── README.md
└── requirements.txt



