from backend.config import settings
from backend.storage.base import StorageBackend


def get_storage_backend() -> StorageBackend:
    """Return the configured storage backend based on STORAGE_BACKEND env var."""
    if settings.storage_backend.lower() == "dynamo":
        from backend.storage.dynamo_backend import DynamoBackend

        return DynamoBackend()

    from backend.storage.sqlite_backend import SQLiteBackend

    return SQLiteBackend()
