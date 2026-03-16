"""Unit tests for the API service using moto for AWS mocking."""

import pytest
import json
import boto3
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from moto import mock_aws

from src.api.main import app
from src.api.auth import create_access_token


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_token():
    return create_access_token({"sub": "test-user", "role": "admin"})


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


class TestHealthEndpoint:
    def test_health_check(self, client):
        """Health endpoint should return 200."""
        with patch.object(
            app.state if hasattr(app, "state") else app,
            "__class__",
            create=True,
        ):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "version" in data
            assert "services" in data


class TestIngestionEndpoints:
    def test_ingest_requires_auth(self, client):
        """Ingestion endpoint should require authentication."""
        response = client.post(
            "/api/v1/ingest",
            json={
                "vehicle_id": "VH-001",
                "sensor_type": "lidar",
            },
        )
        assert response.status_code == 403

    def test_ingest_sensor_data(self, client, auth_headers):
        """Should accept valid sensor data for ingestion."""
        with patch(
            "src.api.main.ingestion_svc.process_upload", new_callable=AsyncMock
        ):
            response = client.post(
                "/api/v1/ingest",
                headers=auth_headers,
                json={
                    "vehicle_id": "VH-001",
                    "sensor_type": "lidar",
                    "metadata": {"points": 150000, "density": 0.85},
                },
            )
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "queued"

    def test_list_datasets(self, client, auth_headers):
        """Should list datasets."""
        with patch(
            "src.api.main.ingestion_svc.list_datasets",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = client.get("/api/v1/datasets", headers=auth_headers)
            assert response.status_code == 200


class TestPipelineEndpoints:
    def test_trigger_pipeline(self, client, auth_headers):
        """Should trigger a processing pipeline."""
        with patch(
            "src.api.main.processing_svc.run_pipeline", new_callable=AsyncMock
        ):
            response = client.post(
                "/api/v1/pipelines",
                headers=auth_headers,
                json={
                    "pipeline_type": "etl",
                    "dataset_id": "ds-001",
                    "config": {"output_format": "parquet"},
                },
            )
            assert response.status_code == 202
            data = response.json()
            assert "pipeline_id" in data
            assert data["status"] == "running"

    def test_list_pipelines(self, client, auth_headers):
        """Should list pipelines."""
        with patch(
            "src.api.main.processing_svc.list_pipelines",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = client.get("/api/v1/pipelines", headers=auth_headers)
            assert response.status_code == 200


@mock_aws
class TestIngestionService:
    def setup_method(self):
        """Set up mock AWS resources."""
        self.s3 = boto3.client("s3", region_name="ap-south-1")
        self.s3.create_bucket(
            Bucket="ad-platform-raw-data",
            CreateBucketConfiguration={"LocationConstraint": "ap-south-1"},
        )
        self.dynamodb = boto3.resource("dynamodb", region_name="ap-south-1")
        self.dynamodb.create_table(
            TableName="ad-platform-datasets",
            KeySchema=[{"AttributeName": "dataset_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "dataset_id", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST",
        )

    def test_s3_bucket_exists(self):
        """Verify S3 bucket was created."""
        buckets = self.s3.list_buckets()["Buckets"]
        names = [b["Name"] for b in buckets]
        assert "ad-platform-raw-data" in names

    def test_dynamodb_table_exists(self):
        """Verify DynamoDB table was created."""
        table = self.dynamodb.Table("ad-platform-datasets")
        assert table.table_status == "ACTIVE"
