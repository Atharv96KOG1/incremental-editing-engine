"""Local web frontend: a React, ChatGPT-styled chat page (source in
../frontend, built to ../frontend/dist) that submits a change request and
streams the same tagged pipeline checklist as the `iee` CLI (LOCALIZE ->
GENERATE -> VALIDATE -> APPLY -> TEST -> COMMIT), via Server-Sent Events,
then renders the same summary the CLI prints as a card.

`run_edit`/`run_create` are synchronous, so each request runs in a
background thread that pushes (tag, msg) step events onto a queue; the
async SSE generator just drains that queue without blocking the event loop.
"""

import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .api import pending_confirmations
from .api.create_pipeline import run_create, run_create_files
from .api.run_pipeline import run_edit
from .retrieval.locate_repo import locate, locate_best_file
from .retrieval.repo_index import iter_source_files
from .storage.minio_client import get_storage
from .strategies.create_target_classifier import classify_create_intent

app = FastAPI()

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

_TREE_IGNORE = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", "dist",
    "venv", ".venv", "egg-info", ".mypy_cache", ".ruff_cache",
}


def _is_ignored(name: str) -> bool:
    return name in _TREE_IGNORE or name.endswith(".egg-info")


def _build_tree(root: Path, current: Path) -> dict:
    entries = []
    try:
        children = sorted(
            current.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        )
    except OSError:
        children = []
    for child in children:
        if _is_ignored(child.name):
            continue
        rel = str(child.relative_to(root))
        if child.is_dir():
            entries.append({"name": child.name, "path": rel, "type": "dir", "children": _build_tree(root, child)["children"]})
        else:
            entries.append({"name": child.name, "path": rel, "type": "file"})
    return {"children": entries}


def _resolve_project_path(project_dir: str, rel_path: str) -> Path:
    root = Path(project_dir).resolve()
    target = (root / rel_path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="path escapes project_dir")
    return target


@app.get("/api/tree")
def api_tree(project_dir: str) -> dict:
    root = Path(project_dir).resolve()
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"'{project_dir}' is not a directory")
    tree = _build_tree(root, root)
    return {"name": root.name, "path": "", "type": "dir", "children": tree["children"]}


@app.get("/api/file")
def api_file(project_dir: str, path: str) -> dict:
    target = _resolve_project_path(project_dir, path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"'{path}' is not a file")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="binary file, cannot display")
    return {"path": path, "content": content}


class RunRequest(BaseModel):
    mode: str
    project_dir: str
    file: Optional[str] = None
    request: str
    test_target: Optional[str] = None
    project_id: Optional[str] = None
    confirm_symbol: Optional[str] = None
    confirm_symbol_type: Optional[str] = None
    confirm_symbol_line: Optional[int] = None
    use_joern: str = "off"
    require_confirmation: bool = True
    use_hybrid_retrieval: bool = True
    use_joern: bool = False


class ConfirmRequest(BaseModel):
    run_id: str
    accept: bool


def _project_id_for(project_dir: str, project_id: Optional[str]) -> str:
    return project_id or Path(project_dir).resolve().name.lower()


class ChatHistoryRequest(BaseModel):
    project_dir: str
    project_id: Optional[str] = None
    messages: list


@app.get("/api/chat_history")
def get_chat_history(project_dir: str, project_id: Optional[str] = None) -> dict:
    """Simple resume, not a multi-session browser: one continuous log per
    project, so reopening the app (or refreshing) picks the conversation
    back up where it left off instead of starting blank every time."""
    storage = get_storage()
    key = f"projects/{_project_id_for(project_dir, project_id)}/chat_history.json"
    if storage.exists(key):
        return storage.get_json(key)
    return {"messages": []}


@app.post("/api/chat_history")
def save_chat_history(req: ChatHistoryRequest) -> dict:
    storage = get_storage()
    key = f"projects/{_project_id_for(req.project_dir, req.project_id)}/chat_history.json"
    storage.put_json(key, {"messages": req.messages})
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.exists():
        return (
            "<pre>frontend not built yet. Run:\n\n"
            "  cd frontend && npm install && npm run build\n\n"
            "then restart `iee serve`.</pre>"
        )
    return index_file.read_text()


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/run")
async def run(req: RunRequest) -> StreamingResponse:
    events: "queue.Queue" = queue.Queue()

    def on_step(tag: str, msg: str) -> None:
        events.put({"type": "step", "tag": tag, "msg": msg})

    def on_preview(data: dict) -> None:
        events.put({"type": "preview", **data})

    def worker() -> None:
        try:
            if req.mode == "find":
                project_dir = str(Path(req.project_dir).resolve())
                on_step("INDEX", "walking the repository and indexing every symbol...")
                on_step("RETRIEVE", "symbol + BM25 + vector retrieval, then fusing rankings...")
                if req.use_joern == "on":
                    on_step("JOERN", "resolving call graph via a real Joern CPG (real JVM cost, can take a while)...")
                elif req.use_joern == "auto":
                    on_step("JOERN", "confidence/need gate armed -- will build a Joern CPG only if confidence is low")
                use_joern = {"off": False, "auto": "auto", "on": True}.get(req.use_joern, False)
                result = locate(project_dir, req.request, use_joern=use_joern)
                if req.use_joern != "off":
                    on_step(
                        "JOERN",
                        "resolved via real Joern CPG" if result["used_joern"] else "native graph used (Joern skipped)",
                    )
                metadata = {
                    "strategy": "HYBRID_RETRIEVAL",
                    "result": {"status": "success"},
                    "generation": {
                        "model": "n/a",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "latency_ms": 0,
                        "estimated_cost_usd": 0.0,
                    },
                    "confidence": result["confidence"],
                    "symbols_indexed": result["symbols_indexed"],
                    "candidates": result["candidates"],
                    "evidence": result["evidence"],
                    "used_joern": result["used_joern"],
                }
                on_step("RANK", f"{len(result['candidates'])} candidate(s), confidence={result['confidence']}")
                events.put({"type": "done", "metadata": metadata})
                return

            if req.mode == "create":
                metadata = run_create(
                    project_dir=Path(req.project_dir),
                    file=req.file,
                    request=req.request,
                    test_target=req.test_target or None,
                    project_id=req.project_id,
                    require_confirmation=req.require_confirmation,
                    on_step=on_step,
                    on_preview=on_preview,
                )
            else:
                file = req.file
                if file and file.strip() in (".", "./"):
                    file = None
                if not file:
                    on_step("LOCATE", "no file given -- searching the repository...")
                    project_dir = str(Path(req.project_dir).resolve())
                    file = locate_best_file(project_dir, req.request)
                    if not file:
                        on_step("LOCATE", "no existing file matches -- checking whether this describes a new file...")
                        listing = "\n".join(
                            sorted(str(Path(p).relative_to(project_dir)) for p in iter_source_files(project_dir))
                        )[:4000]
                        classification = classify_create_intent(req.request, listing)
                        paths = classification["paths"]
                        if classification["wants_new_file"] and paths:
                            if len(paths) == 1 and not paths[0].endswith("/"):
                                on_step("LOCATE", f"describes a new file -- creating '{paths[0]}'...")
                                metadata = run_create(
                                    project_dir=Path(req.project_dir),
                                    file=paths[0],
                                    request=req.request,
                                    test_target=req.test_target or None,
                                    project_id=req.project_id,
                                    require_confirmation=req.require_confirmation,
                                    on_step=on_step,
                                    on_preview=on_preview,
                                    classification_gen=classification,
                                )
                            else:
                                on_step("LOCATE", f"describes {len(paths)} new path(s) -- creating {', '.join(paths)}...")
                                metadata = run_create_files(
                                    project_dir=Path(req.project_dir),
                                    paths=paths,
                                    request=req.request,
                                    test_target=req.test_target,
                                    project_id=req.project_id,
                                    require_confirmation=req.require_confirmation,
                                    on_step=on_step,
                                    on_preview=on_preview,
                                    classification_gen=classification,
                                )
                            events.put({"type": "done", "metadata": metadata})
                            return
                        events.put({"type": "error", "message": "no file given and hybrid retrieval found no candidate"})
                        return
                    on_step("LOCATE", f"auto-located {file}")

                metadata = run_edit(
                    project_dir=Path(req.project_dir),
                    file=file,
                    request=req.request,
                    test_target=req.test_target or ".",
                    project_id=req.project_id,
                    confirm_symbol=req.confirm_symbol,
                    confirm_symbol_type=req.confirm_symbol_type,
                    confirm_symbol_line=req.confirm_symbol_line,
                    require_confirmation=req.require_confirmation,
                    use_hybrid_retrieval=req.use_hybrid_retrieval,
                    use_joern=req.use_joern,
                    on_step=on_step,
                    on_preview=on_preview,
                )
                if metadata["result"]["status"] == "needs_selection":
                    events.put({"type": "needs_selection", "metadata": metadata})
                    return
            events.put({"type": "done", "metadata": metadata})
        except (FileNotFoundError, FileExistsError) as e:
            events.put({"type": "error", "message": str(e)})
        except Exception as e:
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, events.get)
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/confirm")
def confirm(req: ConfirmRequest) -> dict:
    """Resolves a run paused by RunRequest.require_confirmation -- accept
    commits (writes the file, creates the version), reject discards it.
    Non-streaming: by this point tests already passed, so all that's left
    is a version write and a file write, both fast."""
    try:
        return pending_confirmations.resolve(req.run_id, req.accept)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


def main(host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)
