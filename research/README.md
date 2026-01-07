# NeuroSense  
**Non-Invasive Neurological Risk Estimation Using Speech Biomarkers**

---

## Abstract

**NeuroSense** is a research-oriented, end-to-end machine learning system designed to explore the feasibility of **non-invasive neurological risk screening** through speech analysis. The project investigates whether clinically relevant acoustic features extracted from voice recordings can be used to detect neurological conditions—most notably **Parkinson’s disease (PD)**—using supervised learning models.

By integrating signal processing, statistical learning, and modern ML deployment practices, NeuroSense serves as both a **proof-of-concept research pipeline** and a **production-ready inference system** exposed via a REST API.

---

## Research Motivation

Neurological disorders such as Parkinson’s disease often manifest early through subtle motor and speech impairments. Traditional diagnostic methods are costly, invasive, and dependent on specialist evaluation. Recent research suggests that **speech biomarkers**—including pitch variability, articulation stability, and spectral irregularities—may offer a scalable and non-invasive alternative for early screening.

**NeuroSense addresses the following research question:**

> *Can acoustic features derived from short voice recordings reliably distinguish between neurologically healthy individuals and those exhibiting early signs of Parkinson’s disease?*

---

## Methodology Overview

NeuroSense is structured as a **modular ML research pipeline**, reflecting best practices used in academic and industry research environments.

### 1. Speech Signal Preprocessing
- Noise reduction and normalization
- Silence trimming
- Resampling and signal consistency enforcement
- Designed to minimize dataset-specific artifacts

### 2. Acoustic Feature Extraction
Extracted features are grounded in speech pathology and clinical literature:
- **MFCCs (Mel-Frequency Cepstral Coefficients)**
- **Spectral centroid, bandwidth, rolloff**
- **Pitch and jitter-related measures**
- **Energy and temporal features**

These features capture both **phonatory instability** and **motor control degradation**, key indicators in Parkinsonian speech.

### 3. Supervised Model Training
- Binary classification (PD vs. control)
- Traditional ML models (e.g., SVMs, Random Forests, Logistic Regression)
- Cross-validation and performance benchmarking
- Emphasis on interpretability and robustness over black-box optimization

### 4. Model Evaluation
- Accuracy, precision, recall, F1-score
- ROC-AUC analysis
- Bias and variance considerations
- Dataset generalization analysis

### 5. Deployment & Inference
- Trained models are served via a **REST API**
- Enables real-time inference on unseen voice samples
- Designed for downstream integration (e.g., mobile apps, screening tools)

---

## System Architecture

```text
Raw Audio
   ↓
Preprocessing
   ↓
Feature Extraction
   ↓
Supervised Model
   ↓
Risk Prediction
   ↓
REST API Output

