"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # AWS
    aws_region: str = "ap-south-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # S3
    s3_raw_bucket: str = "ad-platform-raw-data"
    s3_processed_bucket: str = "ad-platform-processed-data"
    s3_models_bucket: str = "ad-platform-ml-models"

    # DynamoDB
    dynamodb_datasets_table: str = "ad-platform-datasets"
    dynamodb_pipelines_table: str = "ad-platform-pipelines"
    dynamodb_vehicles_table: str = "ad-platform-vehicles"

    # SQS
    sqs_ingestion_queue: str = "ad-platform-ingestion"
    sqs_processing_queue: str = "ad-platform-processing"

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    # App
    app_name: str = "AD Data Platform"
    debug: bool = False
    log_level: str = "INFO"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # SageMaker
    sagemaker_role_arn: Optional[str] = None
    sagemaker_instance_type: str = "ml.m5.xlarge"

    model_config = {"env_prefix": "AD_PLATFORM_", "env_file": ".env"}
