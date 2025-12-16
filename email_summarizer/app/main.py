import os
import torch
from fastapi import FastAPI, Query
from .schemas import EmailRequest, PredictionResponse
from .model_loader import classifier, load_bert_model
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("WARNING: OPENAI_API_KEY not set. Summaries will be simple heuristics.")

app = FastAPI(title="Smart Email Triage & Summarization API")

def simple_summary(subject: str, body: str) -> str:
    short_body = " ".join(body.split()[:40])
    return f"Email about: {subject[:60]}. Content preview: {short_body}..."


def llm_summary(subject: str, body: str) -> str:
    if client is None:
        return simple_summary(subject, body)

    prompt = (
        "You are an assistant that summarizes business emails.\n"
        "Summarize the following email in 2–3 sentences, focusing on:\n"
        "1) what is being requested or communicated\n"
        "2) any key deadlines or actions.\n\n"
        f"Subject: {subject}\n\n"
        f"Body:\n{body}\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You summarize emails for busy professionals."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("LLM summary error:", e)
        return simple_summary(subject, body)
    
bert_model, bert_tokenizer, device = load_bert_model()

LABEL_MAP = {
    0: "HR",
    1: "Finance",
    2: "Support",
    3: "Sales",
}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_email(req: EmailRequest):

    text = f"SUBJECT: {req.subject or ''} BODY: {req.body or ''}"
    label, confidence = classifier.predict_with_proba(text)
    summary = llm_summary(req.subject, req.body)

    return PredictionResponse(label=label, confidence=confidence, summary=summary)

@app.post("/predict_bert", response_model=PredictionResponse)
def predict_bert(req: EmailRequest):
    
    text = f"SUBJECT: {req.subject or ''} BODY: {req.body or ''}".strip()
    summary = llm_summary(req.subject, req.body)
    inputs = bert_tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()

    return PredictionResponse(label=LABEL_MAP[pred_id], confidence=round(confidence, 4), summary=summary)