from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract interface for persistent storage of runs and tenant records."""

    @abstractmethod
    def save_run(self, tenant_id: str, run_id: str, data: dict) -> None:
        """Persist a completed run record."""
        ...

    @abstractmethod
    def get_run(self, tenant_id: str, run_id: str) -> dict | None:
        """Retrieve a run record by tenant and run ID."""
        ...

    @abstractmethod
    def list_runs(self, tenant_id: str) -> list[dict]:
        """Return all run records for a tenant, sorted newest first."""
        ...

    @abstractmethod
    def save_tenant(self, tenant_id: str, data: dict) -> None:
        """Persist or update a tenant record."""
        ...

    @abstractmethod
    def get_tenant(self, tenant_id: str) -> dict | None:
        """Retrieve a tenant record by ID."""
        ...
