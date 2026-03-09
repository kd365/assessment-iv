import boto3
import time

region = 'us-west-2'
session = boto3.Session(profile_name='class', region_name=region)
sagemaker = session.client('sagemaker')

bucket = 'kathleen-sagemaker-batch-lab'
role_arn = 'arn:aws:iam::388691194728:role/SageMakerExecutionRole'
image_uri = f'246618743249.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1'
model_uri = f's3://kathleen-sagemaker-batch-lab/model_output/credit-risk-xgboost-1772300569/output/model.tar.gz'

# Step 1: Create model
model_name = f'credit-risk-serverless-{int(time.time())}'
sagemaker.create_model(
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
sagemaker.create_endpoint_config(
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
endpoint_name = 'credit-xgboost-endpoint'
sagemaker.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=config_name
)
print(f"Endpoint creating: {endpoint_name}")
print("This takes 2-5 minutes...")

# Step 4: Wait for it
while True:
    resp = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print(f"  Status: {status}")
    if status == 'InService':
        print(f"\nEndpoint ready: {endpoint_name}")
        break
    elif status == 'Failed':
        print(f"Failed: {resp.get('FailureReason')}")
        break
    time.sleep(30)
