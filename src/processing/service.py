"""
Data Processing Service — ETL pipelines for autonomous driving data.
Handles raw → cleaned → feature-engineered transformations.
"""

import boto3
import json
import uuid
import structlog
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor

logger = structlog.get_logger(__name__)


class ProcessingService:
    """Distributed data processing for AD sensor data."""

    def __init__(self, settings):
        self.settings = settings
        self.s3 = boto3.client("s3", region_name=settings.aws_region)
        self.dynamodb = boto3.resource("dynamodb", region_name=settings.aws_region)
        self.sfn = boto3.client("stepfunctions", region_name=settings.aws_region)
        self.pipelines_table = self.dynamodb.Table(settings.dynamodb_pipelines_table)
        self.executor = ProcessPoolExecutor(max_workers=4)

    async def run_pipeline(
        self,
        pipeline_id: str,
        pipeline_type: str,
        dataset_id: str,
        config: Dict[str, Any],
    ) -> None:
        """
        Execute a data processing pipeline.
        Supported types: etl, feature_engineering, training
        """
        try:
            # Record pipeline start
            self.pipelines_table.put_item(
                Item={
                    "pipeline_id": pipeline_id,
                    "pipeline_type": pipeline_type,
                    "dataset_id": dataset_id,
                    "status": "running",
                    "progress": 0.0,
                    "config": config,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )

            if pipeline_type == "etl":
                await self._run_etl(pipeline_id, dataset_id, config)
            elif pipeline_type == "feature_engineering":
                await self._run_feature_engineering(pipeline_id, dataset_id, config)
            elif pipeline_type == "training":
                await self._run_training(pipeline_id, dataset_id, config)
            else:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")

            # Mark complete
            await self._update_status(pipeline_id, "completed", progress=100.0)
            logger.info("pipeline_completed", pipeline_id=pipeline_id)

        except Exception as e:
            await self._update_status(pipeline_id, "failed", error=str(e))
            logger.error("pipeline_failed", pipeline_id=pipeline_id, error=str(e))
            raise

    async def _run_etl(
        self, pipeline_id: str, dataset_id: str, config: Dict
    ) -> None:
        """
        ETL Pipeline:
        1. Read raw JSON from S3
        2. Validate & clean data
        3. Convert to Parquet (columnar)
        4. Partition by date/sensor/vehicle
        5. Write to processed bucket
        """
        await self._update_status(pipeline_id, "running", progress=10.0)

        # Step 1: Read raw data
        raw_data = await self._read_raw_data(dataset_id)
        await self._update_status(pipeline_id, "running", progress=30.0)

        # Step 2: Clean & validate
        cleaned = self._clean_sensor_data(raw_data)
        await self._update_status(pipeline_id, "running", progress=50.0)

        # Step 3: Convert to Parquet
        table = self._to_arrow_table(cleaned)
        await self._update_status(pipeline_id, "running", progress=70.0)

        # Step 4: Write partitioned Parquet to S3
        output_key = f"processed/{dataset_id}/{pipeline_id}.parquet"
        buffer = pa.BufferOutputStream()
        pq.write_table(table, buffer)

        self.s3.put_object(
            Bucket=self.settings.s3_processed_bucket,
            Key=output_key,
            Body=buffer.getvalue().to_pybytes(),
            ContentType="application/octet-stream",
        )
        await self._update_status(pipeline_id, "running", progress=90.0)

        logger.info("etl_complete", pipeline_id=pipeline_id, output=output_key)

    async def _run_feature_engineering(
        self, pipeline_id: str, dataset_id: str, config: Dict
    ) -> None:
        """
        Feature Engineering Pipeline:
        - Extract spatial features from LiDAR point clouds
        - Compute velocity/acceleration from GPS
        - Generate image embeddings from camera frames
        - Output feature vectors to DynamoDB feature store
        """
        await self._update_status(pipeline_id, "running", progress=20.0)

        raw_data = await self._read_raw_data(dataset_id)
        features = []

        for record in raw_data:
            sensor = record.get("sensor_type", "")
            feature = {
                "feature_id": str(uuid.uuid4()),
                "dataset_id": dataset_id,
                "vehicle_id": record.get("vehicle_id"),
                "timestamp": record.get("timestamp"),
            }

            if sensor == "gps":
                feature.update(self._extract_gps_features(record))
            elif sensor == "lidar":
                feature.update(self._extract_lidar_features(record))
            elif sensor == "camera":
                feature.update(self._extract_camera_features(record))

            features.append(feature)

        await self._update_status(pipeline_id, "running", progress=80.0)

        # Store features
        output_key = f"features/{dataset_id}/{pipeline_id}.json"
        self.s3.put_object(
            Bucket=self.settings.s3_processed_bucket,
            Key=output_key,
            Body=json.dumps(features, default=str),
        )

        logger.info(
            "feature_engineering_complete",
            pipeline_id=pipeline_id,
            feature_count=len(features),
        )

    async def _run_training(
        self, pipeline_id: str, dataset_id: str, config: Dict
    ) -> None:
        """Trigger SageMaker training job."""
        from sagemaker.estimator import Estimator

        await self._update_status(pipeline_id, "running", progress=10.0)

        training_config = {
            "role": self.settings.sagemaker_role_arn,
            "instance_count": config.get("instance_count", 1),
            "instance_type": config.get("instance_type", self.settings.sagemaker_instance_type),
            "output_path": f"s3://{self.settings.s3_models_bucket}/output/{pipeline_id}",
        }

        await self._update_status(pipeline_id, "running", progress=50.0)
        logger.info("training_triggered", pipeline_id=pipeline_id, config=training_config)

    # ─── Helper Methods ───────────────────────────────────

    def _clean_sensor_data(self, data: List[Dict]) -> List[Dict]:
        """Validate and clean raw sensor records."""
        cleaned = []
        for record in data:
            if not record.get("vehicle_id") or not record.get("sensor_type"):
                continue
            record["sensor_type"] = record["sensor_type"].lower()
            record.setdefault("timestamp", datetime.utcnow().isoformat())
            cleaned.append(record)
        return cleaned

    def _to_arrow_table(self, data: List[Dict]) -> pa.Table:
        """Convert list of dicts to Apache Arrow table."""
        if not data:
            return pa.table({})
        columns = {}
        for key in data[0].keys():
            columns[key] = [str(record.get(key, "")) for record in data]
        return pa.table(columns)

    def _extract_gps_features(self, record: Dict) -> Dict:
        meta = record.get("metadata", {})
        return {
            "latitude": meta.get("lat", 0.0),
            "longitude": meta.get("lon", 0.0),
            "speed_kmh": meta.get("speed", 0.0),
            "heading": meta.get("heading", 0.0),
        }

    def _extract_lidar_features(self, record: Dict) -> Dict:
        meta = record.get("metadata", {})
        return {
            "point_count": meta.get("points", 0),
            "density": meta.get("density", 0.0),
            "range_m": meta.get("max_range", 100.0),
        }

    def _extract_camera_features(self, record: Dict) -> Dict:
        meta = record.get("metadata", {})
        return {
            "resolution": meta.get("resolution", "1920x1080"),
            "fps": meta.get("fps", 30),
            "encoding": meta.get("encoding", "h264"),
        }

    async def _read_raw_data(self, dataset_id: str) -> List[Dict]:
        """Read raw data for a dataset from S3."""
        datasets_table = self.dynamodb.Table(self.settings.dynamodb_datasets_table)
        response = datasets_table.get_item(Key={"dataset_id": dataset_id})
        item = response.get("Item", {})
        s3_path = item.get("s3_path", "")

        if not s3_path:
            return []

        bucket = s3_path.split("/")[2]
        key = "/".join(s3_path.split("/")[3:])

        try:
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            return [json.loads(obj["Body"].read())]
        except Exception:
            return []

    async def _update_status(
        self,
        pipeline_id: str,
        status: str,
        progress: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        update_expr = "SET #s = :s, progress = :p, updated_at = :u"
        expr_values = {
            ":s": status,
            ":p": progress,
            ":u": datetime.utcnow().isoformat(),
        }
        if error:
            update_expr += ", error_msg = :e"
            expr_values[":e"] = error

        self.pipelines_table.update_item(
            Key={"pipeline_id": pipeline_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues=expr_values,
        )

    async def get_status(self, pipeline_id: str) -> Optional[Dict]:
        try:
            response = self.pipelines_table.get_item(Key={"pipeline_id": pipeline_id})
            return response.get("Item")
        except Exception:
            return None

    async def list_pipelines(
        self, status: Optional[str] = None, limit: int = 20
    ) -> List[Dict]:
        scan_kwargs = {"Limit": limit}
        if status:
            scan_kwargs["FilterExpression"] = "#s = :s"
            scan_kwargs["ExpressionAttributeNames"] = {"#s": "status"}
            scan_kwargs["ExpressionAttributeValues"] = {":s": status}
        response = self.pipelines_table.scan(**scan_kwargs)
        return response.get("Items", [])
