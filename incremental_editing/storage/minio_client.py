"""Object storage abstraction.

LocalStorage mirrors the exact bucket/key layout a real MinIO bucket would
use, on local disk. Point MINIO_ENDPOINT at a running MinIO instance to swap
to MinioStorage with zero caller changes.
"""

import io
import json
import os
from abc import ABC, abstractmethod

from ..config import get_settings


class Storage(ABC):
    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str) -> list: ...

    def put_json(self, key: str, obj) -> None:
        self.put_bytes(key, json.dumps(obj, indent=2).encode("utf-8"))

    def get_json(self, key: str):
        return json.loads(self.get_bytes(key).decode("utf-8"))

    def put_text(self, key: str, text: str) -> None:
        self.put_bytes(key, text.encode("utf-8"))

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")


class LocalStorage(Storage):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root_dir, key)

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def get_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def list(self, prefix: str) -> list:
        results = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, self.root_dir).replace(os.sep, "/")
                if rel.startswith(prefix):
                    results.append(rel)
        return sorted(results)


class MinioStorage(Storage):
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = False):
        from minio import Minio

        self.bucket = bucket
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def put_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(self.bucket, key, io.BytesIO(data), length=len(data))

    def get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except Exception:
            return False

    def list(self, prefix: str) -> list:
        return [o.object_name for o in self.client.list_objects(self.bucket, prefix=prefix, recursive=True)]


def get_storage() -> Storage:
    settings = get_settings()
    endpoint = settings.minio_endpoint
    if endpoint:
        stripped = endpoint.replace("http://", "").replace("https://", "")
        return MinioStorage(
            endpoint=stripped,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=endpoint.startswith("https://"),
        )
    return LocalStorage(root_dir=settings.local_storage_dir)
