import boto3
import time
import datetime
import sys
import sagemaker
from sagemaker import image_uris


# Session Setup
session = boto3.Session(profile_name="class",
                        region_name="us-west-2")
sm_client = session.client("sagemaker") 
bucket="kathleen-sagemaker-batch-lab"
role_arn="arn:aws:iam::388691194728:role/SageMakerExecutionRole"
image_uri = image_uris.retrieve('kmeans', session.region_name)

# Uploading Prpared Data to S3
s3 = session.client("s3")
uploaded_train_path = s3.upload_file("prepared_data/train.csv", bucket, "park-clustering/train/train.csv")

with open('prepared_data/feature_count.txt', 'r') as f:
    feature_dim = int(f.read().strip())



job_name = f"park-clustering-kmeans-{int(time.time())}"
park_job = sm_client.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        "TrainingInputMode": "File",
        "TrainingImage": image_uri
    },
    RoleArn=role_arn,
    HyperParameters={
        "k": "5",
        "feature_dim": str(feature_dim),
        "mini_batch_size": "50",
        "epochs": "10", 
        "init_method": "kmeans++"},
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/park-clustering/train/train.csv",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "text/csv"   
            },
    ],
    OutputDataConfig={
        "S3OutputPath": f"s3://{bucket}/park-clustering/models/"
    },
    ResourceConfig={
        "InstanceType": "ml.m5.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 10},
    StoppingCondition={
        "MaxRuntimeInSeconds": 3600}
)

# Monitoring the Training Job
print(f"Started SageMaker training job: {job_name}")
while True:
    try:
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        print(f"Training job status: {status}")
        if status in ['Completed', 'Failed', 'Stopped']:
            break
        time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Job will continue running.")
        print(f"Check status: aws sagemaker describe-training-job "
              f"--training-job-name {job_name} --region {session.region_name} --profile class")
        sys.exit(0)


# Printing Results




if status == 'Completed':
    print(f"Training job completed with status: {status}")
    output_location = response['ModelArtifacts']['S3ModelArtifacts']    
    print(f"Model artifacts saved to: {output_location}")
elif status == 'Failed':
    print(f"Training job failed: {response.get('FailureReason', 'Unknown')}")
