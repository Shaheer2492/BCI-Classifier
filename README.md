# Pre-Screening Motor Imagery BCI Literacy Using Low-Sample EEG-Derived Features

> DSC 180B Capstone · UCSD · 2026

A pre-screening pipeline that predicts a person's motor imagery BCI literacy from only 15 early trials and a brief resting-state baseline — no lengthy calibration required.

🌐 **Website:** [shaheer2492.github.io/BCI-Classifier](https://shaheer2492.github.io/BCI-Classifier)

---

## Overview

Brain-computer interfaces (BCIs) suffer from high dropout: ~20% of users fail to achieve reliable control even after weeks of calibration. This project asks: *can we identify likely low-performers early, before wasting hours of lab time?*

We extract 38 interpretable EEG features from a short resting baseline + the first 15 motor imagery trials, then train regression and classification models to predict a subject's final CSP-LDA decoding accuracy.

**Key results:**
| Model | Metric | Value |
|---|---|---|
| Random Forest Regression | Pearson *r* | 0.580 |
| Random Forest Regression | R² | 0.302 |
| Random Forest Classifier (0.65 threshold) | LOOCV Accuracy | 78.8% |
| RF Classifier — LOW class | Recall | 89.0% |
| RF Classifier — HIGH class | Recall | 50.0% |

---

## Website Pages

| Page | URL | Description |
|---|---|---|
| Home | `index.html` | Scrollytelling overview of the project |
| Interface | `ml-demo.html` | Live Neuralink-style BCI demo with real-time ML predictions |
| Methodology | `methodology.html` | Full pipeline: features, selection, models |
| Results | `results.html` | Tables, charts, and model evaluation |
| Report | `report.html` | Full research report with inline citations |
| Docs | `documentation.html` | Setup guide and API reference |

---

## Quick Start

### Prerequisites
- Python 3.9+
- ~3 GB disk space (PhysioNet dataset)

### Install Dependencies

```bash
git clone https://github.com/Shaheer2492/BCI-Classifer.git
cd BCI-Classifer
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Step 1 — Generate ground truth labels (optional, pre-computed)
python src/generate_ground_truth_labels.py       # ~1-2 hours

# Step 2 — Extract early-trial features (optional, pre-computed)
python src/extract_early_trial_features.py       # ~5-10 minutes

# Step 3 — Merge resting + task features
python src/merge_features.py

# Step 4 — Train models (optional, pre-trained models included)
python src/train_performance_predictor.py        # ~10-30 seconds

# Step 5 — Start prediction server
python src/prediction_server.py                  # Runs on http://localhost:5001

# Step 6 — Serve website
cd docs && python -m http.server 8000            # Visit http://localhost:8000
```

---

## Project Structure

```
BCI-Classifer/
├── src/
│   ├── generate_ground_truth_labels.py   # Phase 1: CSP-LDA ground truth
│   ├── extract_early_trial_features.py   # Phase 2: Early-trial EEG features
│   ├── merge_features.py                 # Phase 3: Merge resting + task features
│   ├── train_performance_predictor.py    # Phase 4: Regression + classifier training
│   ├── prediction_server.py              # Flask API (port 5001)
│   └── results/
│       ├── ground_truth_labels.json      # Per-subject CSP-LDA accuracies
│       ├── early_trial_features.json     # 38-feature vectors (109 subjects)
│       ├── early_trial_features_merged.json
│       ├── model_evaluation.json         # CV metrics
│       └── models/                       # Trained .pkl files
├── docs/
│   ├── index.html                        # Scrollytelling homepage
│   ├── ml-demo.html                      # Live BCI interface
│   ├── methodology.html                  # Methods deep-dive
│   ├── results.html                      # Results + dynamic subject table
│   ├── report.html                       # Full research report
│   ├── documentation.html                # Setup & API docs
│   ├── css/neuralink.css                 # Design system (mobile-responsive)
│   ├── js/
│   │   ├── neuralink-viz.js              # Canvas EEG visualizations
│   │   ├── neuralink-controller.js       # Demo state machine
│   │   └── bci-demo-ml.js               # ML prediction integration
│   └── images/                           # Figures from the report
├── PHASE1_README.md
├── PHASE2_README.md
├── PHASE3_README.md
├── METHODS.md
├── MODELS_AND_RESULTS.md
├── QUICKSTART.md
└── requirements.txt
```

---

## Pipeline Details

### Phase 1 — Ground Truth (CSP-LDA Decoder)
- PhysioNet EEG Motor Movement/Imagery Dataset (109 subjects, 160 Hz, 64 channels)
- Recorded with BCI2000; left-vs-right hand motor imagery (runs 4, 8, 12)
- CSP-LDA decoder with 5-fold LOOCV per subject
- Output: per-subject accuracy ∈ [0, 1]

### Phase 2 — Feature Extraction (38 features)
Features extracted from a 3-electrode ROI (C3, Cz, C4) using only the first *n*=15 trials:

| Group | Features |
|---|---|
| Resting alpha/mu power | Band power, IAF, aperiodic exponent, SMR strength |
| Resting beta power | Lower beta (13–20 Hz), upper beta (20–30 Hz) |
| ERD/ERS | Mu and beta event-related de/synchronization |
| CSP separability | Class separability from spatial filters |
| Trial variability | SNR, trial-to-trial consistency |
| Power Spectral Entropy (PSE) | Shannon entropy of PSD (alpha + beta) |
| Lempel-Ziv Complexity (LZC) | Temporal pattern diversity, real vs imagined gap |
| Theta/Alpha Ratio (TAR) | Cognitive load proxy |
| Resting predictors | Resting TAR, resting PSE, resting RPLα |

### Phase 3 — Feature Selection
- Spearman correlation with permutation *p*-values (5000 permutations)
- Benjamini–Hochberg FDR correction (α = 0.05)
- Retained **12 predictors** from 38

### Phase 4 — Model Training
- **Regression:** Random Forest, Gradient Boosting, RBF-SVR, Ridge (5-fold CV, *N*=99)
- **Classifier:** Random Forest binary (HIGH/LOW at 0.65 threshold, LOOCV)

---

## API Reference

```
GET  /api/health             Health check
GET  /api/subjects           All subjects with features + predictions
GET  /api/simulate_subject   Random subject simulation
POST /api/predict            Predict from feature vector
```

Example:
```bash
curl http://localhost:5001/api/health
curl http://localhost:5001/api/simulate_subject
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

---

## Technologies

| Layer | Stack |
|---|---|
| EEG processing | MNE-Python, SciPy |
| BCI decoding | CSP-LDA (scikit-learn) |
| ML models | scikit-learn (Random Forest, GBM, SVR, Ridge) |
| Backend API | Flask + flask-cors |
| Frontend | Vanilla HTML/CSS/JS, Canvas API |
| Fonts | Google Fonts (Inter, Roboto Mono) |

---

## Limitations

- Single dataset (PhysioNet only); no external validation
- Class imbalance: 73 LOW / 26 HIGH subjects (73.7% base rate)
- No cross-session or live EEG testing
- Feature selection applied globally, not strictly within each CV fold
- CSP-LDA decoder uses mu band (8–13 Hz) only

---

## Contributors

| Name | Email |
|---|---|
| Shaheer Khan | shk021@ucsd.edu |
| Andrew Li | anl082@ucsd.edu |
| Daniel Mansperger | dmansperger@ucsd.edu |
| Gabriel Riegner | gariegner@ucsd.edu |
| Armin Schwartzman (Advisor) | armins@ucsd.edu |

---

## License

MIT License
