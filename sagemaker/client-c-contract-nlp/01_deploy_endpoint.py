import boto3
import sagemaker
import json
import time
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

region = 'us-west-2'
session = boto3.Session(profile_name='class', region_name=region)
sess = sagemaker.Session(boto_session=session)
role_arn = 'arn:aws:iam::388691194728:role/SageMakerExecutionRole'

hub_config = {
    'HF_MODEL_ID': 'dslim/bert-base-NER',
    'HF_TASK': 'token-classification'
}
model = HuggingFaceModel(
    env=hub_config,
    role=role_arn,
    transformers_version='4.26.0',
    pytorch_version='1.13.1',
    py_version='py39',
    sagemaker_session=sess)

# Section 3: Deploy as serverless endpoint
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,
    max_concurrency=5
)

endpoint_name = 'legal-nlp-ner-endpoint'
predictor = model.deploy(
    endpoint_name=endpoint_name,
    serverless_inference_config=serverless_config
)
print(f"Endpoint deployed: {endpoint_name}")

# Section 4: Test the endpoint
runtime = session.client('sagemaker-runtime')
test_payload = json.dumps({
    "inputs": "Acme Corp shall pay $50,000 to John Smith by December 31, 2025."
})

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=test_payload
)

result = json.loads(response['Body'].read().decode())
print(f"\nNER results: {json.dumps(result, indent=2)}")

# Section 5: Save endpoint name
with open('endpoint_name.txt', 'w') as f:
    f.write(endpoint_name)
print(f"Saved: endpoint_name.txt")
