variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "k8s-training-cluster"
}

variable "s3_bucket" {
  description = "SageMaker S3 bucket"
  type        = string
  default     = "kathleen-sagemaker-batch-lab"
}
