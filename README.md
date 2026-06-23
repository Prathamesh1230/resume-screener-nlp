# AI-Powered Resume Screener & Job Category Classifier

An end-to-end NLP + Machine Learning pipeline that classifies resumes into job categories and serves predictions via a REST API.

---

## Project Overview

Recruiters screen hundreds of resumes manually. This project automates that process — paste a resume, get a predicted job category and confidence score in milliseconds.

**Pipeline:** Raw Resume Text → Text Cleaning → TF-IDF Vectorization → ML Classification → REST API

---

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 64.19%   |
| Linear SVC          | 70.62%   |
| **Random Forest**   | **71.03%** ✅ |

- **Dataset:** 2,484 resumes across 24 job categories
- **Features:** TF-IDF (5,000 features, unigrams + bigrams)
- **Best Model:** Random Forest Classifier

---

## Tech Stack

- **Language:** Python 3.10
- **ML:** Scikit-learn (Pipeline API, TF-IDF, Random Forest, Logistic Regression, LinearSVC)
- **NLP:** Regex-based text cleaning, TF-IDF vectorization
- **API:** FastAPI + Uvicorn
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Serving:** Joblib

---

## Project Structure

```
resume-screener-nlp/
│
├── resume_screener.ipynb       # Full EDA + model training notebook
├── app.py                      # FastAPI inference API
├── resume_screener_model.pkl   # Trained pipeline (generated after running notebook)
├── label_encoder.pkl           # Label encoder (generated after running notebook)
├── Resume.csv                  # Dataset
├── requirements.txt            # Dependencies
└── README.md
```

---

## How to Run

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/resume-screener-nlp.git
cd resume-screener-nlp
pip install -r requirements.txt
```

### 2. Run the Notebook
Open `resume_screener.ipynb` in Jupyter and run all cells.  
This trains the model and saves `resume_screener_model.pkl` and `label_encoder.pkl`.

### 3. Start the API
```bash
uvicorn app:app --reload
```
API will be live at: `http://localhost:8000`

---

## API Endpoints

### `GET /health`
```json
{
  "status": "healthy",
  "model": "Random Forest + TF-IDF",
  "version": "1.0.0"
}
```

### `POST /predict`
**Request:**
```json
{
  "resume_text": "Experienced Data Scientist with 3 years in Python and ML..."
}
```
**Response:**
```json
{
  "predicted_category": "INFORMATION-TECHNOLOGY",
  "confidence_percent": 71.5,
  "top_3_categories": [
    { "category": "INFORMATION-TECHNOLOGY", "confidence_percent": 71.5 },
    { "category": "ENGINEERING", "confidence_percent": 15.2 },
    { "category": "BUSINESS-DEVELOPMENT", "confidence_percent": 8.1 }
  ]
}
```

---

## Key Features

- **Multi-model benchmarking** — trained and compared 3 models, selected best by accuracy
- **Scikit-learn Pipeline API** — clean → vectorize → classify in one object
- **Top-3 predictions** — returns confidence scores for top 3 matching categories
- **Input validation** — rejects empty or too-short resume text
- **Production-ready API** — FastAPI with Pydantic schemas and proper error handling

---

## Dataset

- Source: [Resume Classification Dataset for NLP — Kaggle](https://www.kaggle.com/datasets/hassnainzaidi/resume-classification-dataset-for-nlp)
- 2,484 resumes across 24 job categories including IT, Finance, Healthcare, Engineering, and more

---

## Requirements

```
fastapi
uvicorn
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
pydantic
```

---

## Author

**Prathamesh Jadhav**  
[LinkedIn](https://linkedin.com/in/prathameshjadhav) | [GitHub](https://github.com/Prathamesh1230)
