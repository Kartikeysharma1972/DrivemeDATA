# 🚗 Autonomous Driving Data Platform

A cloud-native, scalable data platform designed for Autonomous Driving R&D. Built with containerized microservices, distributed data processing, and AI/ML pipelines on AWS.

![Architecture](docs/architecture.png)
![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?logo=amazon-aws)
![Terraform](https://img.shields.io/badge/IaC-Terraform-purple?logo=terraform)
![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)
![Kubernetes](https://img.shields.io/badge/K8s-Orchestration-blue?logo=kubernetes)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-green?logo=github-actions)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                  │
├──────────┬──────────┬──────────┬────────────────────────┤
│ Ingestion│Processing│ ML       │ Observability           │
│ Service  │ Service  │ Pipeline │ (CloudWatch + Prometheus│
├──────────┴──────────┴──────────┴────────────────────────┤
│              Message Queue (SQS / Kafka)                 │
├──────────┬──────────┬───────────────────────────────────┤
│ S3 Data  │ DynamoDB │ RDS (PostgreSQL)                   │
│ Lake     │ Metadata │ Structured Data                    │
└──────────┴──────────┴───────────────────────────────────┘
```

## 🔍 Key Components

### 1. Data Ingestion Service
- Streams sensor data (LiDAR, camera, radar) from autonomous vehicles
- Supports batch and real-time ingestion via SQS/Kafka
- Auto-scales based on data volume using ECS/EKS
- Validates, deduplicates, and partitions raw data into S3 Data Lake

### 2. Data Processing Service
- Distributed processing using Python multiprocessing + AWS Lambda
- ETL pipelines: raw sensor data → cleaned → feature-engineered datasets
- Supports Parquet/Arrow columnar formats for efficient analytics
- Step Functions orchestration for complex multi-stage pipelines

### 3. ML Pipeline
- Model training pipeline with SageMaker integration
- Feature store backed by DynamoDB
- Model registry and versioning
- A/B testing infrastructure for model evaluation

### 4. REST API (FastAPI)
- Query processed datasets, trigger pipelines, monitor jobs
- JWT authentication + role-based access control
- OpenAPI docs auto-generated
- Rate limiting and request validation

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11+ |
| **API Framework** | FastAPI, Uvicorn |
| **Containerization** | Docker, Docker Compose |
| **Orchestration** | Kubernetes (EKS), Helm |
| **IaC** | Terraform (modular) |
| **CI/CD** | GitHub Actions |
| **AWS Services** | S3, Lambda, Step Functions, ECS/EKS, DynamoDB, RDS, SQS, CloudWatch, SageMaker |
| **Monitoring** | Prometheus, Grafana, CloudWatch |
| **Testing** | pytest, moto (AWS mocking) |

## 📁 Project Structure

```
├── src/
│   ├── ingestion/        # Data ingestion microservice
│   ├── processing/       # Data processing & ETL
│   ├── api/              # FastAPI REST service
│   └── ml_pipeline/      # ML training & inference
├── terraform/
│   ├── modules/          # Reusable Terraform modules
│   │   ├── ecs/          # ECS cluster + services
│   │   ├── s3/           # S3 buckets + lifecycle
│   │   ├── lambda/       # Lambda functions
│   │   └── dynamodb/     # DynamoDB tables
│   └── environments/     # Dev / Prod configs
├── k8s/
│   ├── base/             # Base K8s manifests
│   └── overlays/         # Kustomize overlays (dev/prod)
├── docker/               # Dockerfiles
├── .github/workflows/    # CI/CD pipelines
├── tests/                # Unit + integration tests
├── scripts/              # Utility scripts
└── docs/                 # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Terraform >= 1.5
- AWS CLI configured
- kubectl (for K8s deployment)

### Local Development
```bash
# Clone
git clone https://github.com/kartikey1972/autonomous-driving-data-platform.git
cd autonomous-driving-data-platform

# Setup virtual environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start all services locally
docker-compose up -d

# Run API
uvicorn src.api.main:app --reload --port 8000

# Run tests
pytest tests/ -v --cov=src
```

### Deploy to AWS
```bash
# Initialize Terraform
cd terraform/environments/dev
terraform init
terraform plan
terraform apply

# Deploy to EKS
kubectl apply -k k8s/overlays/dev/
```

## 🧪 Testing

```bash
# Unit tests with AWS mocking
pytest tests/unit/ -v

# Integration tests (requires local Docker services)
pytest tests/integration/ -v

# Coverage report
pytest --cov=src --cov-report=html
```

## 📊 Monitoring & Observability

- **Metrics**: Prometheus + Grafana dashboards
- **Logs**: CloudWatch Logs with structured JSON logging
- **Traces**: AWS X-Ray for distributed tracing
- **Alerts**: CloudWatch Alarms + SNS notifications

## 🔒 Security

- All data encrypted at rest (S3 SSE, DynamoDB encryption)
- TLS for all inter-service communication
- IAM roles with least-privilege access
- Secrets managed via AWS Secrets Manager
- VPC with private subnets for data services


- LinkedIn: [linkedin.com/in/kartikey1972](https://linkedin.com/in/kartikey1972)
