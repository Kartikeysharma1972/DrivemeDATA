"""
Data Ingestion Service for Autonomous Driving sensor data.
Handles streaming and batch ingestion from vehicle sensors (LiDAR, camera, radar, GPS).
"""

import boto3
import json
import uuid
import hashlib
import structlog
from datetime import datetime
from typing import Optional, List, Dict, Any
from botocore.exceptions import ClientError

logger = structlog.get_logger(__name__)


class IngestionService:
    """Manages sensor data ingestion into the platform."""

    VALID_SENSOR_TYPES = {"lidar", "camera", "radar", "gps", "imu", "ultrasonic"}

    def __init__(self, settings):
        self.settings = settings
        self.s3 = boto3.client("s3", region_name=settings.aws_region)
        self.dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)
        self.sqs = boto3.client("sqs", region_name=settings.aws_region)
        self.datasets_table = self.dynamodb.Table(settings.dynamodb_datasets_table)
        self.vehicles_table = self.dynamodb.Table(settings.dynamodb_vehicles_table)

    async def process_upload(self, job_id: str, data: Dict[str, Any]) -> None:
        """
        Process a sensor data upload:
        1. Validate sensor type
        2. Generate S3 partition key
        3. Upload to S3 raw bucket
        4. Create metadata entry in DynamoDB
        5. Queue for processing
        """
        try:
            sensor_type = data["sensor_type"].lower()
            vehicle_id = data["vehicle_id"]
            timestamp = data.get("timestamp", datetime.utcnow().isoformat())

            # Validate
            if sensor_type not in self.VALID_SENSOR_TYPES:
                raise ValueError(f"Invalid sensor type: {sensor_type}")

            # Generate partitioned S3 key
            date_prefix = datetime.fromisoformat(str(timestamp)).strftime("%Y/%m/%d/%H")
            s3_key = f"raw/{vehicle_id}/{sensor_type}/{date_prefix}/{job_id}.json"

            # Deduplication check
            content_hash = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
            if await self._is_duplicate(content_hash):
                logger.info("duplicate_detected", job_id=job_id, hash=content_hash)
                return

            # Upload to S3
            self.s3.put_object(
                Bucket=self.settings.s3_raw_bucket,
                Key=s3_key,
                Body=json.dumps(data, default=str),
                ContentType="application/json",
                Metadata={
                    "vehicle_id": vehicle_id,
                    "sensor_type": sensor_type,
                    "content_hash": content_hash,
                },
            )

            # Record in DynamoDB
            dataset_id = str(uuid.uuid4())
            self.datasets_table.put_item(
                Item={
                    "dataset_id": dataset_id,
                    "vehicle_id": vehicle_id,
                    "sensor_type": sensor_type,
                    "s3_path": f"s3://{self.settings.s3_raw_bucket}/{s3_key}",
                    "content_hash": content_hash,
                    "record_count": 1,
                    "size_bytes": len(json.dumps(data, default=str)),
                    "status": "raw",
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": data.get("metadata", {}),
                }
            )

            # Queue for processing
            self.sqs.send_message(
                QueueUrl=self.settings.sqs_ingestion_queue,
                MessageBody=json.dumps(
                    {
                        "job_id": job_id,
                        "dataset_id": dataset_id,
                        "s3_key": s3_key,
                        "sensor_type": sensor_type,
                        "vehicle_id": vehicle_id,
                    }
                ),
                MessageGroupId=vehicle_id,
            )

            logger.info(
                "ingestion_complete",
                job_id=job_id,
                dataset_id=dataset_id,
                s3_key=s3_key,
            )

        except Exception as e:
            logger.error("ingestion_failed", job_id=job_id, error=str(e))
            raise

    async def _is_duplicate(self, content_hash: str) -> bool:
        """Check if data with same hash already exists."""
        try:
            response = self.datasets_table.scan(
                FilterExpression="content_hash = :hash",
                ExpressionAttributeValues={":hash": content_hash},
                Limit=1,
            )
            return len(response.get("Items", [])) > 0
        except ClientError:
            return False

    async def list_datasets(
        self,
        vehicle_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """List datasets with optional filtering."""
        scan_kwargs = {"Limit": limit}

        filter_parts = []
        expr_values = {}

        if vehicle_id:
            filter_parts.append("vehicle_id = :vid")
            expr_values[":vid"] = vehicle_id
        if sensor_type:
            filter_parts.append("sensor_type = :st")
            expr_values[":st"] = sensor_type

        if filter_parts:
            scan_kwargs["FilterExpression"] = " AND ".join(filter_parts)
            scan_kwargs["ExpressionAttributeValues"] = expr_values

        response = self.datasets_table.scan(**scan_kwargs)
        return response.get("Items", [])

    async def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset details by ID."""
        try:
            response = self.datasets_table.get_item(Key={"dataset_id": dataset_id})
            return response.get("Item")
        except ClientError:
            return None

    async def list_vehicles(self) -> List[Dict]:
        """List all registered vehicles."""
        response = self.vehicles_table.scan(Limit=100)
        return response.get("Items", [])

    async def get_vehicle_stats(self, vehicle_id: str) -> Dict:
        """Get data collection statistics for a vehicle."""
        response = self.datasets_table.scan(
            FilterExpression="vehicle_id = :vid",
            ExpressionAttributeValues={":vid": vehicle_id},
        )
        items = response.get("Items", [])
        total_size = sum(item.get("size_bytes", 0) for item in items)
        sensor_counts = {}
        for item in items:
            st = item.get("sensor_type", "unknown")
            sensor_counts[st] = sensor_counts.get(st, 0) + 1

        return {
            "vehicle_id": vehicle_id,
            "total_datasets": len(items),
            "total_size_bytes": total_size,
            "sensor_breakdown": sensor_counts,
        }

    async def check_s3_health(self) -> str:
        try:
            self.s3.head_bucket(Bucket=self.settings.s3_raw_bucket)
            return "healthy"
        except Exception:
            return "unhealthy"

    async def check_dynamodb_health(self) -> str:
        try:
            self.datasets_table.table_status
            return "healthy"
        except Exception:
            return "unhealthy"

    async def check_sqs_health(self) -> str:
        try:
            self.sqs.get_queue_url(QueueName=self.settings.sqs_ingestion_queue)
            return "healthy"
        except Exception:
            return "unhealthy"
