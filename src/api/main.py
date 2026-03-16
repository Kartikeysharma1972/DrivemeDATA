"""
FastAPI REST service for the Autonomous Driving Data Platform.
Provides endpoints for data ingestion, pipeline management, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import structlog
import uuid

from src.api.auth import get_current_user
from src.api.config import Settings
from src.ingestion.service import IngestionService
from src.processing.service import ProcessingService

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Autonomous Driving Data Platform",
    description="Cloud-native data platform for AD R&D",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = Settings()
ingestion_svc = IngestionService(settings)
processing_svc = ProcessingService(settings)


# ─── Models ───────────────────────────────────────────────

class SensorDataUpload(BaseModel):
    vehicle_id: str = Field(..., description="Unique vehicle identifier")
    sensor_type: str = Field(..., description="lidar | camera | radar | gps")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    s3_key: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class PipelineRequest(BaseModel):
    pipeline_type: str = Field(..., description="etl | feature_engineering | training")
    dataset_id: str
    config: dict = Field(default_factory=dict)


class PipelineStatus(BaseModel):
    pipeline_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    services: dict


class DatasetInfo(BaseModel):
    dataset_id: str
    vehicle_id: str
    sensor_type: str
    record_count: int
    size_bytes: int
    created_at: datetime
    s3_path: str


# ─── Health & Monitoring ──────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Platform health check with dependency status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
        services={
            "s3": await ingestion_svc.check_s3_health(),
            "dynamodb": await ingestion_svc.check_dynamodb_health(),
            "sqs": await ingestion_svc.check_sqs_health(),
        },
    )


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ─── Data Ingestion ──────────────────────────────────────

@app.post("/api/v1/ingest", status_code=202)
async def ingest_sensor_data(
    data: SensorDataUpload,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """
    Ingest sensor data from autonomous vehicles.
    Validates, deduplicates, and queues for processing.
    """
    job_id = str(uuid.uuid4())
    logger.info(
        "ingestion_request",
        job_id=job_id,
        vehicle_id=data.vehicle_id,
        sensor_type=data.sensor_type,
    )

    background_tasks.add_task(
        ingestion_svc.process_upload,
        job_id=job_id,
        data=data.model_dump(),
    )

    return {"job_id": job_id, "status": "queued", "message": "Data queued for ingestion"}


@app.get("/api/v1/datasets", response_model=List[DatasetInfo])
async def list_datasets(
    vehicle_id: Optional[str] = None,
    sensor_type: Optional[str] = None,
    limit: int = 50,
    user: dict = Depends(get_current_user),
):
    """List available datasets with optional filtering."""
    return await ingestion_svc.list_datasets(
        vehicle_id=vehicle_id,
        sensor_type=sensor_type,
        limit=limit,
    )


@app.get("/api/v1/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, user: dict = Depends(get_current_user)):
    """Get detailed info about a specific dataset."""
    dataset = await ingestion_svc.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


# ─── Pipeline Management ─────────────────────────────────

@app.post("/api/v1/pipelines", response_model=PipelineStatus, status_code=202)
async def trigger_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """Trigger a data processing or ML pipeline."""
    pipeline_id = str(uuid.uuid4())
    logger.info(
        "pipeline_triggered",
        pipeline_id=pipeline_id,
        pipeline_type=request.pipeline_type,
        dataset_id=request.dataset_id,
    )

    background_tasks.add_task(
        processing_svc.run_pipeline,
        pipeline_id=pipeline_id,
        pipeline_type=request.pipeline_type,
        dataset_id=request.dataset_id,
        config=request.config,
    )

    return PipelineStatus(
        pipeline_id=pipeline_id,
        status="running",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@app.get("/api/v1/pipelines/{pipeline_id}", response_model=PipelineStatus)
async def get_pipeline_status(
    pipeline_id: str, user: dict = Depends(get_current_user)
):
    """Check pipeline execution status."""
    status = await processing_svc.get_status(pipeline_id)
    if not status:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return status


@app.get("/api/v1/pipelines", response_model=List[PipelineStatus])
async def list_pipelines(
    status: Optional[str] = None,
    limit: int = 20,
    user: dict = Depends(get_current_user),
):
    """List all pipelines with optional status filter."""
    return await processing_svc.list_pipelines(status=status, limit=limit)


# ─── Vehicle Management ──────────────────────────────────

@app.get("/api/v1/vehicles")
async def list_vehicles(user: dict = Depends(get_current_user)):
    """List all registered autonomous vehicles."""
    return await ingestion_svc.list_vehicles()


@app.get("/api/v1/vehicles/{vehicle_id}/stats")
async def vehicle_stats(vehicle_id: str, user: dict = Depends(get_current_user)):
    """Get data collection stats for a specific vehicle."""
    return await ingestion_svc.get_vehicle_stats(vehicle_id)
