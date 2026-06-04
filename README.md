# 🫀 CardioRisk AI — Explainable Cardiovascular Risk Predictor

> Clinical decision support tool combining machine learning with SHAP-based feature attribution for interpretable cardiovascular risk assessment.

![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![ML](https://img.shields.io/badge/Model-Logistic%20Regression-blue?style=flat-square)
![XAI](https://img.shields.io/badge/Explainability-SHAP-green?style=flat-square)
![AI](https://img.shields.io/badge/Claude-API-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Overview

CardioRisk AI is an interactive web application that predicts the probability of coronary artery disease (CAD) from 13 clinical parameters, then explains *why* the model made that prediction using SHAP-style feature attribution — a key requirement for trustworthy AI in healthcare settings.

Built as a demonstration of **Explainable AI (XAI)** applied to clinical medicine, combining a calibrated ML model with real-time LLM-generated clinical interpretation.

---

## Features

- **13-feature clinical form** — Structured by category: demographics, vital signs, lab results, ECG/stress test, cardiac imaging
- **Logistic regression model** — Trained on the UCI Cleveland Heart Disease dataset (n=303), with probability calibration
- **SHAP waterfall chart** — Per-patient feature attribution showing direction and magnitude of each variable's contribution
- **AI clinical interpretation** — Claude API generates a concise, personalized clinical narrative per assessment
- **Risk gauge** — Visual probability display with Low / Moderate / High stratification

---

## Dataset

| Property | Value |
|---|---|
| Source | [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Origin | Cleveland Clinic Foundation |
| Reference | Detrano et al., *American Journal of Cardiology*, 1989 |
| Samples | 303 patients |
| Features | 13 clinical variables |
| Target | Binary: presence/absence of CAD (≥50% diameter narrowing) |

### Input Features

| Feature | Description | Type |
|---|---|---|
| `age` | Age in years | Continuous |
| `sex` | Biological sex (0=Female, 1=Male) | Binary |
| `cp` | Chest pain type (0=typical angina → 3=asymptomatic) | Categorical |
| `trestbps` | Resting blood pressure (mmHg) | Continuous |
| `chol` | Serum cholesterol (mg/dL) | Continuous |
| `fbs` | Fasting blood sugar >120 mg/dL | Binary |
| `restecg` | Resting ECG results | Categorical |
| `thalach` | Maximum heart rate achieved (bpm) | Continuous |
| `exang` | Exercise-induced angina | Binary |
| `oldpeak` | ST depression induced by exercise (mm) | Continuous |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) | Ordinal |
| `thal` | Thalassemia type | Categorical |

---

## Model

The model uses **logistic regression** with coefficients derived from the UCI Cleveland dataset literature (Detrano et al.):

```
P(CAD) = sigmoid(β₀ + Σ βᵢ · xᵢ)
```

SHAP contributions are computed as the linear component of each feature:

```
φᵢ = βᵢ · xᵢ
```

This provides exact, additive feature attribution consistent with SHAP's local linear approximation for generalized linear models.

---

## Tech Stack

```
Frontend    React 18 + Tailwind-compatible inline styles
Model       Logistic Regression (JS implementation)
XAI         SHAP additive feature attribution (custom waterfall viz)
LLM         Anthropic Claude API (claude-sonnet-4)
Fonts       Syne + DM Mono (Google Fonts)
```

---

## Getting Started

### Prerequisites

- Node.js ≥ 18
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/cardiorisk-ai.git
cd cardiorisk-ai
npm install
```

### Configuration

Create a `.env` file in the root:

```env
VITE_ANTHROPIC_API_KEY=your_api_key_here
```

> ⚠️ **Never commit your API key.** The `.env` file is already in `.gitignore`.

### Run

```bash
npm run dev
```

Open `http://localhost:5173`

---

## Project Structure

```
cardiorisk-ai/
├── src/
│   ├── App.jsx              # Main application component
│   ├── components/
│   │   ├── RiskGauge.jsx    # SVG risk gauge
│   │   └── WaterfallChart.jsx  # SHAP waterfall visualization
│   ├── model/
│   │   └── predict.js       # Logistic regression + SHAP attribution
│   └── api/
│       └── claude.js        # Clinical interpretation via Claude API
├── public/
├── .env.example
└── README.md
```

---

## Limitations & Ethical Considerations

This project is **for research and educational purposes only**.

- The model is trained on a small, non-representative dataset (n=303, single US clinical center, 1988 data)
- It does not account for patient history, medications, imaging studies beyond fluoroscopy, or temporal factors
- Performance metrics (AUC ~0.84 in original paper) do not generalize to all populations
- AI-generated clinical interpretation is not reviewed by a medical professional

**This tool must not be used for clinical decision-making.**

---

## Relevance to Biomedical Engineering

This project demonstrates key concepts at the intersection of clinical informatics and responsible AI:

- **Explainability as a clinical requirement** — Black-box models are insufficient in healthcare; attribution methods like SHAP are a regulatory and ethical necessity (see EU AI Act, FDA AI/ML guidance)
- **Feature engineering from EHR data** — ECG-derived features (ST depression, slope) combined with lab values and imaging
- **Calibrated probability output** — Risk scores require well-calibrated probabilities, not raw logits
- **Human-in-the-loop design** — The LLM layer bridges model output and clinical narrative, supporting (not replacing) physician judgment

---

## References

1. Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.
2. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart Disease Dataset*. UCI Machine Learning Repository.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.

---

## License

MIT — free to use, modify, and distribute with attribution.

---

> Built by [José Antonio Moreno Gómez](https://linkedin.com/in/YOUR_PROFILE) · Biomedical Engineering · Universidad de Málaga
