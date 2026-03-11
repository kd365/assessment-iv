from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import json
import os
import time
import random
from botocore.exceptions import ClientError


app = FastAPI(title="SageMaker Credit Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# A/B routing configuration — read from environment (set via ConfigMap)
ENDPOINT_NAME_V1 = os.environ.get('ENDPOINT_NAME_V1', 'credit-xgboost-endpoint')
ENDPOINT_NAME_V2 = os.environ.get('ENDPOINT_NAME_V2', '')
TRAFFIC_WEIGHT_V1 = int(os.environ.get('TRAFFIC_WEIGHT_V1', '100'))
MODEL_VERSION_V1 = os.environ.get('MODEL_VERSION_V1', 'v1')
MODEL_VERSION_V2 = os.environ.get('MODEL_VERSION_V2', 'v2')
REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
runtime = boto3.client('sagemaker-runtime', region_name=REGION)


class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: str
    model_version: str
    endpoint_used: str


def select_endpoint() -> tuple:
    """Select a SageMaker endpoint based on A/B traffic weights."""
    if not ENDPOINT_NAME_V2:
        return ENDPOINT_NAME_V1, MODEL_VERSION_V1
    if random.randint(1, 100) <= TRAFFIC_WEIGHT_V1:
        return ENDPOINT_NAME_V1, MODEL_VERSION_V1
    return ENDPOINT_NAME_V2, MODEL_VERSION_V2


@app.get("/health")
def health():
    """Liveness check - is the process alive and responsive?"""
    return {
        "status": "healthy",
        "service": "credit-risk-api",
        "ab_routing": {
            "v1_endpoint": ENDPOINT_NAME_V1,
            "v2_endpoint": ENDPOINT_NAME_V2 or "not configured",
            "v1_traffic_pct": TRAFFIC_WEIGHT_V1,
            "v2_traffic_pct": 100 - TRAFFIC_WEIGHT_V1 if ENDPOINT_NAME_V2 else 0,
        }
    }

@app.get("/ready")
def ready():
    """Readiness check - can this pod serve traffic?"""
    try:
        if runtime is not None and ENDPOINT_NAME_V1:
            result = {"status": "ready", "v1_endpoint": ENDPOINT_NAME_V1}
            if ENDPOINT_NAME_V2:
                result["v2_endpoint"] = ENDPOINT_NAME_V2
                result["v1_traffic_pct"] = TRAFFIC_WEIGHT_V1
            return result
        raise HTTPException(status_code=503, detail={"status": "not ready"})
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "not ready", "error": str(e)})

@app.get("/routing")
def routing():
    """Return the current A/B routing configuration."""
    return {
        "v1": {
            "endpoint": ENDPOINT_NAME_V1,
            "version": MODEL_VERSION_V1,
            "traffic_pct": TRAFFIC_WEIGHT_V1,
        },
        "v2": {
            "endpoint": ENDPOINT_NAME_V2 or "not configured",
            "version": MODEL_VERSION_V2,
            "traffic_pct": 100 - TRAFFIC_WEIGHT_V1 if ENDPOINT_NAME_V2 else 0,
        },
        "ab_enabled": bool(ENDPOINT_NAME_V2),
    }

def call_endpoint_with_retry(features, endpoint_name, max_retries=3):
    """Call SageMaker endpoint with retry logic for cold starts."""
    csv_data = ",".join(str(f) for f in features)

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='text/csv',
                Body=csv_data
            )
            elapsed = time.time() - start_time

            if elapsed > 5.0 and attempt < max_retries - 1:
                time.sleep(1)
                continue

            return response

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ModelError' and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise HTTPException(
                status_code=503,
                detail=f"Endpoint error: {error_code}"
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error: {str(e)}"
                )
            time.sleep(2 ** attempt)

    raise HTTPException(status_code=503, detail="Endpoint unavailable after retries")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != 43:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 43 features, got {len(request.features)}"
        )

    # A/B routing: select endpoint based on traffic weight
    endpoint_name, model_version = select_endpoint()

    response = call_endpoint_with_retry(request.features, endpoint_name)

    raw_response = response['Body'].read().decode().strip()
    prediction_value = float(raw_response)
    confidence = "high" if prediction_value > 0.7 else "medium" if prediction_value > 0.3 else "low"

    return PredictionResponse(
        prediction=prediction_value,
        confidence=confidence,
        model_version=model_version,
        endpoint_used=endpoint_name,
    )