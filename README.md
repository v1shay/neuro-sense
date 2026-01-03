# NeuroSense

NeuroSense is an end-to-end machine learning system for **non-invasive neurological risk estimation** using **speech biomarkers**. The project focuses on detecting neurological conditions (e.g., Parkinson’s disease) from voice recordings by extracting acoustic features and training supervised learning models, with inference exposed through a REST API.

---

## Project Overview

NeuroSense explores how acoustic properties of human speech can be leveraged for neurological screening. The system processes raw voice recordings, extracts clinically relevant features, trains supervised classifiers on public datasets, and serves predictions via an API for downstream applications.

The project is designed as a **research-grade ML pipeline** with clean separation between data processing, modeling, and deployment.

---

## Core Capabilities

- Speech signal preprocessing
- Acoustic feature extraction
- Supervised model training and evaluation
- Parkinson’s disease risk classification
- REST API for model inference
- Modular, extensible architecture

---

## Repository Structure

```text
neuro-sense/
│
├── api/                # REST API for inference
│   └── main.py         # API entrypoint
│
├── src/                # Core ML pipeline
│   ├── preprocessing/  # Audio preprocessing & cleaning
│   ├── features/       # Feature extraction (MFCCs, spectral features)
│   ├── models/         # Training and evaluation scripts
│   └── utils/          # Shared utilities
│
├── data/               # Voice datasets (raw / processed)
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
