import boto3
import time
import datetime
import sys

# Session Setup
session = boto3.Session(profile_name="class",
                        region_name="us-west-2")
sagemaker = session.client("sagemaker") 
bucket="kathleen-sagemaker-batch-lab"
role_arn="arn:aws:iam::388691194728:role/SageMakerExecutionRole"
image_uri="246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1"

# Uploading Prpared Data to S3
s3 = session.client("s3")
uploaded_train_path = s3.upload_file("prepared_data/train.csv", bucket, "prepared_data/train.csv")
uploaded_val_path = s3.upload_file("prepared_data/validation.csv", bucket, "prepared_data/validation.csv")

# Training Job Configuration
with open("prepared_data/scale_pos_weight.txt", 'r') as f:
    scale_pos_weight = f.read().strip()

job_name = f"credit-risk-xgboost-{int(time.time())}"
credit_job = sagemaker.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        "TrainingInputMode": "File",
        "TrainingImage": image_uri
    },
    RoleArn=role_arn,
    HyperParameters={
        "objective": "binary:logistic",
        "num_round": "100",
        "max_depth": "5",
        "eta": "0.2", # learning rate
        "eval_metric": "auc", # Area Under the Curve for evaluation
        "scale_pos_weight": scale_pos_weight}, # Address class imbalance
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/prepared_data/train.csv",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
        "ContentType": "text/csv"   
            },
        
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/prepared_data/validation.csv",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
        "ContentType": "text/csv"
        }
    ],
    OutputDataConfig={
        "S3OutputPath": f"s3://{bucket}/model_output/"
    },
    ResourceConfig={
        "InstanceType": "ml.m5.large",
        "InstanceCount": 1,
        "VolumeSizeInGB": 10},
    StoppingCondition={
        "MaxRuntimeInSeconds": 3600}
)

# Monitoring the Training Job
print(f"Started SageMaker training job: {job_name}")
while True:
    try:
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
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
