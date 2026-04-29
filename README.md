<div align="center">

<h1>Neuro-Sense</h1>

<p><strong>An end-to-end system for non-invasive neurological risk estimation using speech-based biomarkers.</strong></p>

</div>

---


## Results

- **Accuracy:** 91%  
- **Dataset:** 195 voice recordings (UCI Machine Learning Repository)  
- **Publication:** Published in the *International Journal for Research* 
- **Indexes:** Neuro-Sense has been indexed/is available on Google Scholar, ResearchGate, and OiPub
- **Paper:** https://scholar.google.com/citations?view_op=view_citation&hl=en&user=qfZMYcMAAAAJ&citation_for_view=qfZMYcMAAAAJ:u5HHmVD_uO8C

---

## Overview

NeuroSense is a speech-based machine learning system for estimating neurological risk from non-invasive audio signals. The system treats voice recordings as measurable proxies for underlying motor and cognitive function, extracting acoustic structure that correlates with neurological degradation.

The pipeline converts raw speech into structured features and applies supervised learning to infer risk scores. Outputs are exposed through a programmatic interface for downstream integration.

---

## Method / Approach

<p align="center">
  <img width="631" height="309" alt="Pipeline Overview" src="https://github.com/user-attachments/assets/5a54741f-95a8-4328-9133-eaa9165f30d6" />
</p>


- **Input Standardization**  
  Voice recordings are normalized for sampling rate, duration, and noise characteristics.

- **Feature Extraction**  
  Acoustic features are computed from raw audio, including:
  - frequency-domain characteristics  
  - temporal stability measures  
  - perturbation metrics (e.g., jitter, shimmer)

- **Supervised Inference**  
  Models map feature space → neurological risk:
  - classification objective (disease vs control)  
  - probabilistic outputs  

- **Evaluation + Serving**  
  Models are validated and deployed through a REST API.
---

## Data

- **Source:** UCI Machine Learning Repository (Parkinson’s voice dataset)  
- **Type:** structured acoustic feature dataset derived from voice recordings  
- **Size:** 195 samples  

<p align="center">
  <img width="635" height="421" alt="Training Data Scaling" src="https://github.com/user-attachments/assets/b5cf439f-9208-4bb9-8d40-48f3a74ea84e" />
</p>

Preprocessing:
- normalization  
- feature scaling  
- partitioning  

<p align="center"> <img width="540" height="449" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/4da1d86e-e5a6-48ff-8ef2-d05ab1ddc9ee" /> </p>


---

## Experiments / Reproduction


```bash
python training/train.py
python evaluation/evaluate.py
```

## Run inference:

```bash
python api/infer.py --input sample.wav
```

## Train model:

```bash
python training/train.py --config configs/default.yaml
```

Input: .wav file
Output: risk score + classification

Dependencies
```bash
Python 3.x
NumPy
SciPy
scikit-learn
librosa
FastAPI / Flask
```

## Repository Structure

```bash
neurosense/
├── data/
│   ├── raw/
│   └── processed/
├── features/
├── models/
├── training/
├── api/
├── evaluation/
└── README.md
```

## Installation
```bash
git clone https://github.com/your-username/neurosense.git
cd neurosense
pip install -r requirements.txt
```

## Optional:

```bash
conda env create -f environment.yml
conda activate neurosense
```
