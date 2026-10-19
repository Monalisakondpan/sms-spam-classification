# 📬 Distributed SMS Spam Classification Platform

A production-ready, microservice-based Natural Language Processing (NLP) system decoupled into independent containerized environments. The platform separates the web user application interface from a high-performance machine learning inference engine.

## 🚀 Live Production Web Dashboard
👉 **Click here to interact with the live application:** [https://ai-spam-detector-v2.streamlit.app/](https://ai-spam-detector-v2.streamlit.app/)

---

## 🏗️ Microservice System Architecture

The project has been refactored from a monolithic script into a distributed system to ensure high availability, fault isolation, and independent scaling:

* **Presentation Layer (`frontend/`):** A lightweight web client built with **Streamlit** that serves the user interface. It contains zero machine learning logic and delegates classification calculations entirely via network payloads.
* **Inference Engine (`backend/`):** A high-performance **FastAPI** server that houses a Multinomial Naive Bayes model trained with custom alpha smoothing (`alpha=0.05`) and character-level N-gram processing (`ngram_range=(3,5)`).
* **Virtual Private Network:** Both services run in completely isolated sandboxed Linux environments (`python:3.14-rc-slim`), securely networked together via a private **Docker Compose** network bridge mapping localized ports.

---

## 🧠 Distributed Technology Stack & Frameworks
* **Core Language Paradigms:** Python 3.14
* **API Framework & Schema Validation:** FastAPI, Pydantic, Uvicorn
* **Container Orchestration Engine:** Docker, Docker Compose
* **Machine Learning & NLP Pipelines:** Scikit-Learn, NLTK, Pandas, NumPy

---

## ⚙️ Local Installation & Container Deployment

### Prerequisites
Ensure **Docker Desktop** is running locally on your computer environment.

### 1. Spin Up the Multi-Container Network
Open your terminal panel within the root project workspace folder and execute this single orchestration script:
```bash
docker compose up --build
```

### 2. Access the Localized Microservices Interface
Once Docker completes the system layer assembly, the isolated instances will be live across your network loops:
* 🖥️ **Web User Application Interface:** `http://localhost:8501`
* 📑 **Interactive OpenAPI Swagger Documentation:** `http://localhost:8000/docs`
* 🩺 **Backend Microservice Health Check Endpoint:** `http://localhost:8000/health`

---

## 🛡️ Enterprise Systems Features Implemented
1. **Decoupled System Fault Isolation:** If the Streamlit frontend crashes due to massive client traffic, the core inference engine layer remains fully functional and accessible by other microservices.
2. **Strict Schema Type Validation:** Built custom **Pydantic data models** on the POST endpoints, serving as a protective payload gatekeeper before strings hit the Scikit-learn vectors.
3. **Automated Orchestration Health Checks:** Implemented an explicit GET `/health` endpoint providing instant application liveness metrics designed for cluster orchestrators like Kubernetes.
