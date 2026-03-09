import boto3
import time
import sagemaker
import json
from sagemaker import image_uris

region = 'us-west-2'
session = boto3.Session(profile_name='class', region_name=region)
sess = session.client('sagemaker')

bucket = 'kathleen-sagemaker-batch-lab'
role_arn = 'arn:aws:iam::388691194728:role/SageMakerExecutionRole'
image_uri = image_uris.retrieve('kmeans', session.region_name)
model_uri = f's3://kathleen-sagemaker-batch-lab/park-clustering/models/park-clustering-kmeans-1773031056/output/model.tar.gz'

# Step 1: Create model
model_name = f'park-clustering-kmeans-{int(time.time())}'
sess.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': image_uri,
        'ModelDataUrl': model_uri
    },
    ExecutionRoleArn=role_arn
)
print(f"Model created: {model_name}")

 # Step 2: Create endpoint config (serverless)
config_name = f'{model_name}-config'
sess.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        'VariantName': 'default',
        'ModelName': model_name,
        'ServerlessConfig': {
            'MemorySizeInMB': 2048,
            'MaxConcurrency': 5
        }
    }]
)
print(f"Endpoint config created: {config_name}")

 # Step 3: Create endpoint
endpoint_name = 'park-clustering-kmeans-endpoint'
sess.create_endpoint(
     EndpointName=endpoint_name,
     EndpointConfigName=config_name
)
print(f"Endpoint creating: {endpoint_name}")
print("This takes 2-5 minutes...")

 # Step 4: Wait for it
while True:
    resp = sess.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print(f"  Status: {status}")
    if status == 'InService':
        print(f"\nEndpoint ready: {endpoint_name}")
        break
    elif status == 'Failed':
        print(f"Failed: {resp.get('FailureReason')}")
        break
    time.sleep(30)

# Step 5: Test the endpoint
runtime = session.client('sagemaker-runtime')

# Test row: dummy label (0) + 12 normalized features
test_payload = "0.5,0.5,1,1,0,1,0,1,0,0,0.4,0.6"


response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='text/csv',
    Body=test_payload
)

result = json.loads(response['Body'].read().decode())
print(f"\nTest prediction: {json.dumps(result, indent=2)}")

# Save endpoint name for FastAPI service
with open('endpoint_name.txt', 'w') as f:
    f.write(endpoint_name)
print(f"Saved: endpoint_name.txt")

ServerlessInferenceConfig(memory_size_in_mb=4096, max_concurrency=5)