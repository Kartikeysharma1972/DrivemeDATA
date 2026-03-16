"""
ML Training Pipeline for Autonomous Driving models.
Integrates with AWS SageMaker for distributed training.
"""

import boto3
import json
import structlog
from datetime import datetime
from typing import Dict, Any, Optional

logger = structlog.get_logger(__name__)


class ModelTrainer:
    """Manages ML model training lifecycle."""

    def __init__(self, settings):
        self.settings = settings
        self.sagemaker = boto3.client("sagemaker", region_name=settings.aws_region)
        self.s3 = boto3.client("s3", region_name=settings.aws_region)

    def create_training_job(
        self,
        job_name: str,
        dataset_s3_uri: str,
        hyperparameters: Dict[str, str],
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
    ) -> Dict[str, Any]:
        """Launch a SageMaker training job."""
        training_config = {
            "TrainingJobName": job_name,
            "RoleArn": self.settings.sagemaker_role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": self._get_training_image(),
                "TrainingInputMode": "File",
            },
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": dataset_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "application/x-parquet",
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": f"s3://{self.settings.s3_models_bucket}/output/"
            },
            "ResourceConfig": {
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": 50,
            },
            "HyperParameters": hyperparameters,
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        }

        response = self.sagemaker.create_training_job(**training_config)
        logger.info("training_job_created", job_name=job_name)
        return response

    def get_training_status(self, job_name: str) -> Dict[str, Any]:
        """Check training job status."""
        response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
        return {
            "job_name": job_name,
            "status": response["TrainingJobStatus"],
            "secondary_status": response.get("SecondaryStatus", ""),
            "creation_time": str(response.get("CreationTime", "")),
            "training_time_seconds": response.get("TrainingTimeInSeconds", 0),
            "billable_time_seconds": response.get("BillableTimeInSeconds", 0),
        }

    def register_model(
        self,
        model_name: str,
        model_artifact_s3: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Register a trained model in the model registry."""
        response = self.sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": self._get_inference_image(),
                "ModelDataUrl": model_artifact_s3,
            },
            ExecutionRoleArn=self.settings.sagemaker_role_arn,
        )
        logger.info("model_registered", model_name=model_name)
        return response

    def _get_training_image(self) -> str:
        region = self.settings.aws_region
        return f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1-gpu-py310"

    def _get_inference_image(self) -> str:
        region = self.settings.aws_region
        return f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.1-gpu-py310"
