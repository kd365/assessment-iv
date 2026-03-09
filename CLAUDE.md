# Assessment IV — Internal ML Platform Delivery with Kubernetes, SageMaker & CI/CD

## Overview
This assessment evaluates ability to deliver internal platform tooling that orchestrates multiple ML endpoints using Kubernetes. The system combines SageMaker models, FastAPI services, K8s orchestration, Terraform IaC, GitHub Actions CI/CD, and a React dashboard.

## Scenario: 2 — Data Products Contracting Firm
A devops team at a contracting firm builds ML-powered data pipelines for external clients. Each client has a different dataset and use case. The team trains, deploys, and maintains model endpoints.

### Three Client Contracts

| Client | Use Case | Model Type | SageMaker Endpoint | Dataset |
|--------|----------|------------|-------------------|---------|
| **Client A** (Financial Services) | Credit risk scoring | XGBoost (binary classification) | `credit-xgboost-endpoint` | UCI Credit Card Default (30K records) |
| **Client B** (Outdoor Recreation) | Park accessibility/feasibility ranking | K-Means (clustering) | `park-clustering-kmeans-endpoint` | Fairfax County Parks JSON (~90 parks) |
| **Client C** (Legal Tech) | NLP entity extraction from contracts | HuggingFace BERT NER | `legal-nlp-ner-endpoint` | Sample contract text |

---

## Full Assessment Requirements

### 1. Terraform Infrastructure as Code (20%)
- [x] Provision cloud resources through Terraform
- [x] Variables, outputs, proper state management — no hardcoded credentials
- [ ] Document lifecycle: `terraform init`, `plan`, `apply`, `destroy`
- [x] **Bonus**: Remote state with S3/DynamoDB locking
- [x] **Bonus**: Terraform-managed K8s resources (namespaces, RBAC, ConfigMaps) via Kubernetes provider

### 2. Kubernetes Orchestration Quality (25%)
- [x] Deploy to EKS with namespace separation per team (client-a, client-b, client-c)
- [x] ConfigMaps for non-sensitive config (endpoint names, regions, log levels)
- [x] Secrets for credentials (AWS keys, registry tokens)
- [x] Readiness, liveness, and startup probes with appropriate thresholds
- [x] ResourceQuota and LimitRange per namespace
- [ ] **Bonus**: Controlled failure scenario (probe restart, quota rejection, readiness-gated traffic)

### 3. Multi-Endpoint SageMaker Integration (20%)
- [x] Three SageMaker endpoints — Client A (XGBoost), Client B (K-Means), Client C (HuggingFace NER)
- [x] Each wrapped in FastAPI with /health, /ready, /predict
- [x] Routing is explicit — each client has its own service and namespace
- [x] Failure path handled — retry logic with exponential backoff for cold starts
- [ ] **Bonus**: Gateway service as single entry point proxying to each team's service
- [ ] **Bonus**: Model versioning and A/B routing

### 4. GitHub Actions CI/CD Quality (15%)
- [x] Workflow: build 3 Docker images, push to GHCR, deploy to EKS
- [x] GitHub Actions secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, GITHUB_TOKEN)
- [x] Verification steps (kubectl rollout status, get pods, get svc)
- [ ] **Bonus**: Rollback steps, branch-based targeting
- [ ] **Bonus**: Multiple workflows chained together
- [ ] **Bonus**: Additional workflows (destroy infra, run tests, lint manifests)

### 5. Internal Operations UI — React Dashboard (10%)
- [x] Dashboard showing deployed services, health status, team ownership
- [x] Internal ops tool, not consumer app
- [x] Live polling (15s interval) + test-request interface
- [x] **Bonus**: React framework
- [x] **Bonus**: Tailwind CSS styling

### 6. Documentation & Presentation Quality (10%)
- [ ] Setup/config steps reproducible by a colleague
- [ ] At least one architecture diagram (cluster, endpoints, CI/CD, dashboard)
- [ ] Present and answer questions about design decisions
- [x] Clean GitHub repo structure
- [ ] **Bonus**: Helper scripts (secrets config, env scaffolding, local dev bootstrap)

---

## Current Progress

### Client A — Credit Risk (COMPLETE)

**SageMaker Pipeline:**
- `sagemaker/client-a-credit-risk/01_explore_data.py` — UCI dataset download, EDA, 4 plots
- `sagemaker/client-a-credit-risk/02_prepare_data.py` — Feature engineering (43 features), SageMaker CSV formatting
- `sagemaker/client-a-credit-risk/03_train_model.py` — S3 upload, XGBoost training job with scale_pos_weight
- `sagemaker/client-a-credit-risk/04_deploy_endpoint.py` — Serverless endpoint deployment

**FastAPI Service:**
- `services/client-a/app.py` — /health, /ready, /predict with retry logic, 43-feature validation
- `services/client-a/Dockerfile` — Python 3.11-slim, uvicorn
- `services/client-a/requirements.txt` — fastapi, uvicorn, boto3, pydantic

**Kubernetes Manifests (k8s/client-a/):**
- `namespace.yml` — client-a namespace with team label
- `configmap.yml` — ENDPOINT_NAME, AWS_DEFAULT_REGION, LOG_LEVEL, MODEL_TYPE, TEAM_NAME
- `secret.yml` — AWS credentials template (created via kubectl CLI, not committed)
- `deployment.yml` — Container spec with ConfigMap/Secret env refs, all 3 probes
- `service.yml` — LoadBalancer, port 80 → 8000
- `resourcequota.yml` — 1 CPU request / 2 CPU limit / 2Gi memory / 5 pods max
- `limitrange.yml` — Per-container defaults and max

### Client B — Park Clustering (COMPLETE)

**SageMaker Pipeline:**
- `sagemaker/client-b-park-clustering/01_explore_data.py` — Parks JSON exploration, amenity analysis, geo analysis, 4 plots
- `sagemaker/client-b-park-clustering/02_prepare_data.py` — JSON flattening, 12 features, MinMaxScaler, dummy label column for K-Means
- `sagemaker/client-b-park-clustering/03_train_model.py` — S3 upload, K-Means training (k=5, feature_dim=12)
- `sagemaker/client-b-park-clustering/04_deploy_endpoint.py` — Serverless endpoint deployment, test returns cluster + distance

**FastAPI Service:**
- `services/client-b/app.py` — /health, /ready, /predict with retry logic, 12-feature validation, returns cluster + distance
- `services/client-b/Dockerfile` — Python 3.11-slim, uvicorn
- `services/client-b/requirements.txt` — fastapi, uvicorn, boto3, pydantic

**Kubernetes Manifests (k8s/client-b/):** Same pattern as Client A, namespace client-b

### Client C — Legal NLP (COMPLETE)

**SageMaker Pipeline:**
- `sagemaker/client-c-contract-nlp/01_deploy_endpoint.py` — HuggingFace BERT NER (dslim/bert-base-NER), serverless deploy, no training needed

**FastAPI Service:**
- `services/client-c/app.py` — /health, /ready, /predict accepting text, returns NER entities
- `services/client-c/Dockerfile` — Python 3.11-slim, uvicorn
- `services/client-c/requirements.txt` — fastapi, uvicorn, boto3, pydantic

**Kubernetes Manifests (k8s/client-c/):** Same pattern as Client A, namespace client-c

### CI/CD (COMPLETE)
- `.github/workflows/cicd.yml` — Build 3 Docker images → push to GHCR → deploy to EKS → verify rollouts

### Terraform (COMPLETE)
- `terraform/provider.tf` — AWS + Kubernetes providers, S3 remote state backend
- `terraform/variables.tf` — region (us-east-1), cluster_name, s3_bucket
- `terraform/main.tf` — 3 namespaces, 3 configmaps, 3 resource quotas via Kubernetes provider
- `terraform/outputs.tf` — cluster_name, cluster_endpoint, namespaces

### Dashboard (COMPLETE)
- `dashboard/` — React + Vite + Tailwind CSS
- Live polling (15s), health status badges, team ownership, test-request interface
- Run: `cd dashboard && npm install && npm run dev` → http://localhost:3000

### Remaining Work
- [ ] Documentation: README.md with setup steps, architecture diagram

---

## AWS Config

| Setting | Value |
|---------|-------|
| SageMaker Region | us-west-2 |
| EKS Region | us-east-1 |
| Profile | class |
| S3 Bucket | kathleen-sagemaker-batch-lab |
| Role ARN | arn:aws:iam::388691194728:role/SageMakerExecutionRole |
| XGBoost Image | 246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1 |
| K-Means Image | 382416733822.dkr.ecr.us-west-2.amazonaws.com/kmeans:1 |
| HuggingFace Model | dslim/bert-base-NER (token-classification) |
| EKS Cluster | k8s-training-cluster |
| GHCR Registry | ghcr.io/kd365/ |
| Terraform State | s3://ai-ops-tf-remote-state-0/kathleenh/assessment-iv/ |

## SageMaker Endpoints

| Endpoint | Status | Model |
|----------|--------|-------|
| `credit-xgboost-endpoint` | InService | XGBoost binary classification |
| `park-clustering-kmeans-endpoint` | InService | K-Means (k=5, 12 features) |
| `legal-nlp-ner-endpoint` | InService | HuggingFace BERT NER |

## Common Issues Log
- Client A `02_prepare_data.py` had wrong import (`test_train_split` → `train_test_split`), wrong module (`sklearn.preprocessing` → `sklearn.model_selection`), wrong CSV filename, missing PAY_1 column
- YAML metadata must use nested indentation, never dot notation (`metadata.name:` is invalid)
- Endpoint names must match between deploy script and app.py
- SageMaker ECR account IDs differ by region (us-west-2 = 246618743249 for XGBoost)
- Namespace must exist before creating resources in it
- Secrets created via `kubectl create secret generic` CLI, not from YAML with placeholders
- K-Means CSV requires dummy label column (first column, all zeros) + features = feature_dim + 1 columns
- K-Means inference expects only features (no label column), unlike training
- EKS cluster is in us-east-1, SageMaker endpoints in us-west-2 — ConfigMaps pass us-west-2 to pods
- Terraform import needed for pre-existing K8s resources: `terraform import kubernetes_namespace.client_a client-a`
- `ServerlessInferenceConfig` (high-level SDK) vs `ServerlessConfig` (boto3 low-level API) — different parameter names
