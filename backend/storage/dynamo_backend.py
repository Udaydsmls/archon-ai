import json
from datetime import datetime

import boto3
from boto3.dynamodb.conditions import Key

from backend.config import settings
from backend.storage.base import StorageBackend


class DynamoBackend(StorageBackend):
    """DynamoDB-backed storage for multi-tenant production deployments with RBAC."""

    def __init__(self) -> None:
        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            region_name=settings.aws_region,
        )
        dynamo = session.resource("dynamodb")
        self._runs = dynamo.Table(settings.dynamo_runs_table)
        self._tenants = dynamo.Table(settings.dynamo_tenants_table)

    def save_run(self, tenant_id: str, run_id: str, data: dict) -> None:
        """Write a run record with tenant partition key and timestamp."""
        self._runs.put_item(
            Item={
                "tenant_id": tenant_id,
                "run_id": run_id,
                "data": json.dumps(data),
                "created_at": datetime.utcnow().isoformat(),
            }
        )

    def get_run(self, tenant_id: str, run_id: str) -> dict | None:
        """Fetch a run by composite key (tenant_id + run_id)."""
        response = self._runs.get_item(Key={"tenant_id": tenant_id, "run_id": run_id})
        item = response.get("Item")
        return json.loads(item["data"]) if item else None

    def list_runs(self, tenant_id: str) -> list[dict]:
        """Query all runs for a tenant, sorted newest first."""
        response = self._runs.query(
            KeyConditionExpression=Key("tenant_id").eq(tenant_id),
            ScanIndexForward=False,
        )
        return [json.loads(item["data"]) for item in response.get("Items", [])]

    def save_tenant(self, tenant_id: str, data: dict) -> None:
        """Write or overwrite a tenant record including RBAC fields."""
        self._tenants.put_item(
            Item={
                "tenant_id": tenant_id,
                "data": json.dumps(data),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )

    def get_tenant(self, tenant_id: str) -> dict | None:
        """Fetch a tenant record including allowed_agents and rate_limit_per_hour."""
        response = self._tenants.get_item(Key={"tenant_id": tenant_id})
        item = response.get("Item")
        return json.loads(item["data"]) if item else None
