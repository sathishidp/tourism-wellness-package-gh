# Wellness Tourism Package – Model Deployment (FastAPI)

This Space deploys a **FastAPI** application that predicts whether a customer is likely to purchase the **Wellness Tourism Package**.  
The service loads a trained machine learning pipeline directly from the Hugging Face Model Hub.

---

## Features
- Loads the best trained model from:  
  **`sathishaiuse/wellness-tourism-model-best`**
- Provides REST API endpoints for real-time inference
- Predicts:
  - Whether a customer will purchase the package (`0` or `1`)
  - Probability score (if supported by the model)
- Fully containerized through a custom `Dockerfile`

---

## API Endpoints

### **GET /**
Health check endpoint.

**Response Example:**
```json
{
  "status": "running",
  "model_loaded": true
}
