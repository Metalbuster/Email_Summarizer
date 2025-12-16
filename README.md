Smart Email Triage & Summarization System

A production-style NLP + MLOps project using FastAPI, Airflow, Docker, scikit-learn, and LLMs

🚀 Overview

This project is an end-to-end machine learning system designed to classify corporate emails (HR, Finance, Support, Sales) and generate concise summaries using both traditional NLP and modern LLMs.

It demonstrates full-stack AI engineering skills:
Data ingestion & preprocessing
Machine learning model training
FastAPI-based inference service
Automated MLOps pipeline with Apache Airflow
Docker containerization
Model evaluation & monitoring
Clean modular project architecture

This project is structured to resemble a real production AI system and is suitable for AI Engineer / Data Engineer roles.

🧱 Project Architecture

1.Incoming emails (raw CSV)

2.prepare_data.py  → clean + split data

3.train_model.py  → train TF-IDF + Logistic Regression classifier

4.evaluate_model.py  → compute precision/recall/F1

5.Saved model (joblib)

6.FastAPI API
/predict → return label + confidence + summary (LLM or fallback)

7.Airflow DAG automates entire pipeline daily

✨ Features

✔ Email Classification
TF-IDF vectorization + Logistic Regression
Predicts HR, Finance, Support, Sales
Returns label + confidence score

✔ Email Summaries
Uses OpenAI LLMs (gpt-4o-mini) when API key available
Fallback heuristic summary for offline use

✔ REST API with FastAPI
Endpoints:
GET /health – health check
POST /predict – classify and summarize emails

✔ Automated MLOps Pipeline with Airflow
DAG steps:
prepare_data – clean dataset & split into train/val/test
train_model – retrain model using updated data
evaluate_model – compute classification metrics
Can be scheduled daily using Airflow.

✔ Dockerized Deployment
Airflow webserver
Airflow scheduler
Postgres (metadata DB)
FastAPI app can also be containerized and deployed separately

🛠️ Installation

Clone the repository
git clone https://github.com/Metalbuster/Email_Summarizer.git
cd Email_Summarizer
Then, you can modify email.csv to have any data you want for training.

🚀 Running the project for FAST API

1.Run ML pipeline locally
Install requirements:
pip install -r requirements.txt

2.Run:
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/evaluate_model.py

3.Run FastAPI API
uvicorn app.main:app --reload
Open Swagger UI:
http://127.0.0.1:8000/docs

🚀 Running Airflow

After you have run 
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/evaluate_model.py

Start Airflow services:
docker compose up -d

Initialize Airflow DB (first time only):
docker compose run airflow-webserver airflow db init

Create admin user:
docker compose run airflow-webserver airflow users create \
  --username admin --password admin \
  --firstname Air --lastname Flow \
  --role Admin --email admin@example.com

Access Airflow UI:
http://localhost:8080
DAG: email_ml_pipeline
Trigger and monitor your ML pipeline.

✨ Example API request

POST /predict
{
  "subject": "Request for document review",
  "body": "Hi team, could someone review the attached report before tomorrow?"
}

Response:
{
  "label": "HR",
  "confidence": 0.87,
  "summary": "Employee is requesting a document review with a short deadline."
}

📊 Model Performance

Metrics from evaluate_model.py:
Precision:	How accurate predictions are
Recall:	How many real positives correctly predicted
F1:	Harmonic mean of precision & recall

Improves automatically as email.csv grows.

📦 Technologies Used

Language:	Python
ML:	scikit-learn, pandas, numpy
API:	FastAPI, Uvicorn
MLOps:	Apache Airflow
Containerization:	Docker, Docker Compose
LLM:	OpenAI (GPT-4o-mini)
Serialization: joblib
Storage:	CSV + Postgres for Airflow metadata
