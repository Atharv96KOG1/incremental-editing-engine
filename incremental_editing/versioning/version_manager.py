"""Tracks version chain (v1 -> delta -> v2 -> ...) and writes checkpoints to storage."""

import time
from typing import Optional

from ..storage.minio_client import Storage


class VersionManager:
    def __init__(self, storage: Storage, project_id: str):
        self.storage = storage
        self.project_id = project_id
        self.head_key = f"projects/{project_id}/HEAD.json"

    def get_head(self) -> Optional[str]:
        if self.storage.exists(self.head_key):
            return self.storage.get_json(self.head_key)["version_id"]
        return None

    def _next_version_id(self) -> str:
        head = self.get_head()
        if head is None:
            return "v1"
        return f"v{int(head.lstrip('v')) + 1}"

    def create_version(
        self,
        files: dict,
        change_request: str,
        strategy: str,
        delta_id: str,
        validation_status: str,
        deleted_files: Optional[list] = None,
    ) -> str:
        version_id = self._next_version_id()
        parent_version = self.get_head()

        for filename, content in files.items():
            key = f"projects/{self.project_id}/checkpoints/{version_id}/files/{filename}"
            self.storage.put_text(key, content)

        manifest = {"version_id": version_id, "files": list(files.keys()), "deleted": deleted_files or []}
        self.storage.put_json(f"projects/{self.project_id}/checkpoints/{version_id}/manifest.json", manifest)

        version_meta = {
            "version_id": version_id,
            "parent_version": parent_version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "change_request": change_request,
            "strategy": strategy,
            "delta_id": delta_id,
            "validation_status": validation_status,
        }
        self.storage.put_json(f"projects/{self.project_id}/versions/{version_id}.json", version_meta)
        self.storage.put_json(self.head_key, {"version_id": version_id})
        return version_id
