import pickle
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK resources are available
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

app = FastAPI(title="SMS Spam Classification API", description="Distributed microservice for NLP classification")

# Load the serialized ML models safely
try:
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Model files not found. Ensure model.pkl and vectorizer.pkl exist in the backend folder.")


class SMSInput(BaseModel):
    text: str


def transform_text(text: str) -> str:
    """Preprocess text exactly how the model was trained."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Simple tokenization/cleaning
    words = text.split()
    # Filter stopwords and stem words
    y = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return " ".join(y)


@app.get("/health")
def health_check():
    """Liveness probe requirement for distributed services."""
    return {"status": "healthy"}


@app.post("/predict")
def predict_spam(sms: SMSInput):
    if not sms.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    # 1. Preprocess the raw incoming text
    transformed_sms = transform_text(sms.text)

    # 2. Vectorize the processed string tokens
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict classification probability array
    result = model.predict(vector_input)

    # Extract the scalar integer index [0] to safely avoid NumPy dimensionality errors
    return {
        "input_text": sms.text,
        "is_spam": int(result[0]) == 1
    }