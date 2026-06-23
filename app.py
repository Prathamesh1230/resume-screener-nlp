from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import numpy as np

# ── Load model & encoder ──────────────────────────────────────────
pipeline = joblib.load("resume_screener_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI(
    title="AI Resume Screener API",
    description="Classifies resumes into job categories using NLP + Machine Learning.",
    version="1.0.0"
)

# ── Request / Response Schemas ────────────────────────────────────
class ResumeInput(BaseModel):
    resume_text: str

class PredictionOutput(BaseModel):
    predicted_category: str
    confidence_percent: float
    top_3_categories: list

# ── Text Cleaning (same as notebook) ─────────────────────────────
def clean_resume(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'\S+@\S+', '', text)               # remove emails
    text = re.sub(r'[^a-zA-Z\s]', '', text)           # remove special chars
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "AI Resume Screener API is running.",
        "endpoints": {
            "POST /predict": "Classify a resume into a job category",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "Random Forest + TF-IDF", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: ResumeInput):
    if not data.resume_text or len(data.resume_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Resume text is too short. Please provide at least 50 characters."
        )

    cleaned = clean_resume(data.resume_text)

    # Predict
    prediction = pipeline.predict([cleaned])[0]
    predicted_category = label_encoder.inverse_transform([prediction])[0]

    # Top 3 categories with probabilities
    try:
        probas = pipeline.predict_proba([cleaned])[0]
        top3_indices = np.argsort(probas)[::-1][:3]
        top3 = [
            {
                "category": label_encoder.inverse_transform([i])[0],
                "confidence_percent": round(probas[i] * 100, 2)
            }
            for i in top3_indices
        ]
        confidence = round(probas[prediction] * 100, 2)
    except Exception:
        top3 = [{"category": predicted_category, "confidence_percent": 100.0}]
        confidence = 100.0

    return PredictionOutput(
        predicted_category=predicted_category,
        confidence_percent=confidence,
        top_3_categories=top3
    )
