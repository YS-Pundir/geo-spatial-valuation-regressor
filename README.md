# 🏠 Geo-Spatial Housing Valuation Engine

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Live-Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/yuvrajpundir/Geo-Spatial-Housing-Valuation-Engine)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![IIT Roorkee](https://img.shields.io/badge/IIT%20Roorkee-Module%202-blue?style=for-the-badge)](https://www.iitr.ac.in/)
[![UE Potsdam](https://img.shields.io/badge/UE%20Potsdam-Software%20Engineering-red?style=for-the-badge)](https://www.ue-germany.com/)

---

# 📌 Project Overview

This project is a **Geo-Spatial Non-Linear Regression Engine** developed during **Module 2: Machine Learning Foundations** under the **IIT Roorkee: Agentic AI Systems & Design Program**, while pursuing a **B.Sc. in Software Engineering** at the **University of Europe for Applied Sciences (UE), Potsdam**.

The application predicts **California housing market valuations** by analyzing a combination of:

- Geographic Coordinates
- Socio-economic Indicators
- Population & Income Metrics

Unlike traditional linear models, this system captures complex **spatial patterns**, regional housing hotspots, and non-linear relationships using ensemble machine learning techniques.

---

# 🚀 Live Deployment

The model is deployed as a fully interactive web dashboard on Hugging Face:

👉 **[Interactive Housing Valuation Interface](https://huggingface.co/spaces/yuvrajpundir/Geo-Spatial-Housing-Valuation-Engine)**

---

# 🛠️ ML Engineering Workflow

## 1️⃣ Model Selection & Comparison

Two regression architectures were benchmarked to evaluate performance on non-linear housing data.

### 📉 Linear Regression

- Captured only simple linear relationships
- Struggled with geographic clustering
- Achieved approximately:

\[
R^2 \approx 0.59
\]

### 🌲 Random Forest Regressor *(Selected Model)*

- Captured complex spatial dependencies
- Learned regional housing valuation patterns
- Significantly improved prediction accuracy

\[
R^2 \approx 0.80
\]

### ✅ Why Random Forest?

The Random Forest model was selected because it:

- Handles non-linear relationships effectively
- Captures geo-spatial patterns
- Reduces variance through ensemble learning
- Produces strong real-world generalization

---

## 2️⃣ Hyperparameter Tuning & Optimization

To create a deployable production-grade model, extensive optimization was performed.

### 🔍 GridSearchCV

Used exhaustive hyperparameter tuning to identify the optimal:

- Number of Estimators
- Tree Depth
- Model Complexity

### ✂️ Model Pruning

Applied `max_depth` constraints to build a lightweight inference engine.

### 📦 Optimization Results

| Metric | Before Optimization | After Optimization |
|---|---|---|
| Model Size | 276 MB | ~20 MB |
| Deployment Efficiency | Poor | Excellent |
| Predictive Performance | High | High |

This optimization enabled deployment within GitHub and Hugging Face storage limitations without sacrificing significant predictive accuracy.

---

## 3️⃣ Production Pipeline

To ensure consistent inference and eliminate preprocessing mismatches, the scaler and model were bundled into a single Scikit-Learn pipeline.

```python
# Spatial Valuation Inference Pipeline

best_rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(
        n_estimators=50,
        max_depth=10
    ))
])
```

This guarantees that every input from the Gradio dashboard undergoes the exact same preprocessing workflow used during training.

---

# 📂 Repository Structure

```text
geo-spatial-valuation-regressor/
│
├── Export/
│   └── housing_regressor_pipeline.joblib
│
├── notebook/
│   └── Regression_models.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

## 📖 Folder Explanation

| Folder/File | Purpose |
|---|---|
| `Export/` | Stores optimized trained regression pipeline |
| `notebook/` | Research, EDA & experimentation |
| `app.py` | Gradio-based interactive dashboard |
| `requirements.txt` | Deployment dependency configuration |
| `README.md` | Project documentation |

---

# 🔧 Technical Stack

## 💻 Core Technologies

- Python 3.13
- Scikit-Learn

## 📊 Data Science Libraries

- Pandas
- NumPy

## ☁️ Deployment & UI

- Gradio
- Hugging Face Spaces

## 🗂️ Version Control & Storage

- Git
- Git LFS (Large File Storage)

---

# 🎓 IIT Roorkee Journey

This project marks the successful completion of:

## **Module 2 — Machine Learning Foundations**

under the **IIT Roorkee Agentic AI Systems & Design Program**.

The project demonstrates a progression from:

- Classification Systems
➡️ to
- Advanced Regression Architectures

These foundations will later be integrated into:

- RAG Pipelines
- Autonomous AI Agents
- Tool-Augmented LLM Systems

during **Module 3: GenAI & Agents**.

---

# 👨‍💻 Developed By

## **Yuvraj Singh Pundir**

- 🎓 Software Engineering Student  
  **University of Europe for Applied Sciences (UE), Potsdam**

- 🤖 IIT Roorkee Agentic AI Fellow

---

# 🌐 Connect With Me

- [LinkedIn](#)
- [GitHub](#)
- [Hugging Face](https://huggingface.co/yuvrajpundir)

---
