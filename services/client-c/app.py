from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import json
import os
import time
from botocore.exceptions import ClientError


app = FastAPI(title="SageMaker Legal NLP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'legal-nlp-ner-endpoint')
REGION = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
runtime = boto3.client('sagemaker-runtime', region_name=REGION)


class NERRequest(BaseModel):
    text: str


class Entity(BaseModel):
    word: str
    entity: str
    score: float
    start: int
    end: int


class NERResponse(BaseModel):
    entities: list[Entity]
    text: str


@app.get("/health")
def health():
    """Liveness check - is the process alive and responsive?"""
    return {"status": "healthy", "service": "legal-nlp-api"}


@app.get("/ready")
def ready():
    """Readiness check - can this pod serve traffic?"""
    try:
        if runtime is not None and ENDPOINT_NAME:
            return {"status": "ready", "endpoint": ENDPOINT_NAME}
        raise HTTPException(status_code=503, detail={"status": "not ready"})
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "not ready", "error": str(e)})


def call_endpoint_with_retry(text, max_retries=3):
    """Call SageMaker endpoint with retry logic for cold starts."""
    payload = json.dumps({"inputs": text})

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType='application/json',
                Body=payload
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


@app.post("/predict", response_model=NERResponse)
def predict(request: NERRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    response = call_endpoint_with_retry(request.text)

    result = json.loads(response['Body'].read().decode())

    entities = [
        Entity(
            word=e['word'],
            entity=e['entity'],
            score=e['score'],
            start=e['start'],
            end=e['end']
        )
        for e in result
    ]

    return NERResponse(entities=entities, text=request.text)
