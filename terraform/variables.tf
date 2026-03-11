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

variable "client_a_endpoint_v1" {
  description = "SageMaker endpoint name for Client A model v1"
  type        = string
  default     = "credit-xgboost-endpoint"
}

variable "client_a_endpoint_v2" {
  description = "SageMaker endpoint name for Client A model v2 (empty = disabled)"
  type        = string
  default     = ""
}

variable "client_a_traffic_weight_v1" {
  description = "Percentage of traffic routed to v1 (0-100)"
  type        = number
  default     = 100
}
