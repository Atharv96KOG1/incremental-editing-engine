"""The actual CLI: `iee create ...` / `iee edit ...`.

Loads .env automatically (OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL,
MINIO_*) so there's no `source .env` dance, prints each pipeline step as a
permanent checklist line as it happens (not a spinner that erases itself),
and finishes with a colored summary table.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .api.create_pipeline import run_create
from .api.run_pipeline import run_edit

console = Console()


class StepTracker:
    """Renders pipeline steps as a tagged checklist that grows in place: each
    line is stamped with which pipeline phase it's from (LOCALIZE, GENERATE,
    VALIDATE, APPLY, TEST, SYNTAX, REPAIR, COMMIT), the in-progress step shows a
    spinner glyph, every step before it shows a checkmark, and the whole
    list stays in the terminal's scrollback -- nothing gets overwritten or
    disappears once printed. The tags let someone unfamiliar with the code
    map each line to a phase of the pipeline without reading the source."""

    _SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _TAG_COLORS = {
        "LOCALIZE": "blue",
        "GENERATE": "magenta",
        "VALIDATE": "yellow",
        "APPLY": "cyan",
        "SYNTAX": "yellow",
        "TEST": "green",
        "REPAIR": "dark_orange",
        "COMMIT": "bright_green",
        "WARNING": "bold red",
    }
    _TAG_WIDTH = 8

    def __init__(self, console: Console):
        self._console = console
        self._steps: list[tuple[str, str]] = []
        self._frame = 0
        self._live = Live(console=console, refresh_per_second=12, transient=False)

    def __enter__(self) -> "StepTracker":
        self._live.__enter__()
        self._live.update(self._render())
        return self

    def __exit__(self, *exc) -> None:
        self._live.__exit__(*exc)

    def step(self, tag: str, msg: str) -> None:
        self._steps.append((tag, msg))
        self._live.update(self._render())

    def finish(self, ok: bool) -> None:
        icon = "[bold green]✓[/]" if ok else "[bold red]✗[/]"
        self._live.update(self._render(final_icon=icon))

    def _render(self, final_icon: str = None) -> Text:
        lines = []
        last = len(self._steps) - 1
        for i, (tag, msg) in enumerate(self._steps):
            if i < last:
                icon = "[bold green]✓[/]"
            elif final_icon is not None:
                icon = final_icon
            else:
                icon = f"[bold cyan]{self._SPINNER_FRAMES[self._frame % len(self._SPINNER_FRAMES)]}[/]"
                self._frame += 1
            color = self._TAG_COLORS.get(tag, "white")
            tag_label = f"[bold {color}]{tag:<{self._TAG_WIDTH}}[/]"
            lines.append(f"{icon} {tag_label} {msg}")
        if not lines:
            lines = ["[bold cyan]…[/] starting..."]
        return Text.from_markup("\n".join(lines))


def _op_line(op: dict) -> str:
    t = op["target"]
    if op["operation"] == "DELETE_FILE":
        return f"{op['operation']:<11} {t['file']}"
    where = t.get("symbol_name", "?")
    if t.get("anchor"):
        where += f" -> after {t['anchor']}"
    lr = op.get("line_range") or {}
    if "after_line" in lr:
        lines = f"after line {lr['after_line']}"
    elif lr:
        lines = f"line {lr['start']}" if lr["start"] == lr["end"] else f"lines {lr['start']}-{lr['end']}"
    else:
        lines = "?"
    return f"{op['operation']:<7} {where} ({lines})"


def _summary_table(metadata: dict) -> Table:
    status = metadata["result"]["status"]
    gen = metadata["generation"]

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("status", "[bold green]success[/]" if status == "success" else "[bold red]failed[/]")
    table.add_row("strategy", metadata["strategy"])
    if metadata.get("note"):
        table.add_row("note", f"[yellow]{metadata['note']}[/]")
    if "new_version" in metadata:
        table.add_row("version", f"{metadata['base_version']} -> {metadata['new_version']}")
    if "change_ratio" in metadata:
        table.add_row("change ratio", f"{metadata['change_ratio']:.1%}")
    if "context" in metadata:
        c = metadata["context"]
        table.add_row(
            "context",
            f"{c['context_lines']}/{c['total_lines']} lines (symbols={c['affected_symbols'] or 'whole file'})",
        )
    if metadata.get("operations"):
        table.add_row("operations", "\n".join(_op_line(op) for op in metadata["operations"]))
    if metadata.get("cross_file_impact_warning"):
        callers = "\n".join(metadata["cross_file_impact_warning"])
        table.add_row("cross-file impact", f"[bold yellow]other files still call the old name:[/]\n{callers}")
    table.add_row("model", gen["model"])
    table.add_row("tokens", f"in={gen['input_tokens']} out={gen['output_tokens']} total={gen['total_tokens']}")
    table.add_row("cost", f"${gen['estimated_cost_usd']:.6f}")
    table.add_row("latency", f"{gen['latency_ms']} ms")
    retry_count = metadata.get("result", {}).get("retry_count", 0)
    if retry_count:
        table.add_row("repaired", f"[dark_orange]{retry_count} attempt(s)[/]")
    if status != "success":
        if metadata.get("result", {}).get("failure_class"):
            table.add_row("failure class", metadata["result"]["failure_class"])
        table.add_row("error", str(metadata.get("error", "unknown"))[:400])
    return table


def cmd_create(args: argparse.Namespace) -> None:
    console.rule(f"[bold]create[/] {args.file}")
    console.print(f"[dim]{Path(args.project_dir).resolve()}[/dim]")
    try:
        with StepTracker(console) as tracker:
            metadata = run_create(
                project_dir=Path(args.project_dir),
                file=args.file,
                request=args.request,
                test_target=args.test_target,
                project_id=args.project_id,
                on_step=tracker.step,
            )
            tracker.finish(ok=metadata["result"]["status"] == "success")
    except FileExistsError as e:
        console.print(f"[bold red]error:[/] {e}")
        sys.exit(1)

    console.print()
    console.print(_summary_table(metadata))
    sys.exit(0 if metadata["result"]["status"] == "success" else 1)


def _resolve_file(args: argparse.Namespace) -> str:
    """If --file was omitted, use hybrid retrieval across the whole
    project to find it (PHOENIX doc sections 1/4) instead of requiring
    the caller to already know which file the request is about.

    Uses the fast path (symbol+BM25 only, no semgrep, no dependency
    graph) -- `iee find` is where the full evidence-gathering pipeline
    belongs; picking the edit target just needs the file, fast."""
    if args.file:
        return args.file

    from .retrieval.locate_repo import locate_best_file

    project_dir = str(Path(args.project_dir).resolve())
    with console.status("[bold cyan]no --file given, searching the repository...", spinner="dots"):
        file = locate_best_file(project_dir, args.request)

    if not file:
        console.print("[bold red]error:[/] --file omitted and hybrid retrieval found no candidate in this project")
        sys.exit(1)

    console.print(f"[dim]auto-located:[/] {file}")
    return file


def cmd_edit(args: argparse.Namespace) -> None:
    file = _resolve_file(args)
    console.rule(f"[bold]edit[/] {file}")
    console.print(f"[dim]{Path(args.project_dir).resolve()}[/dim]")

    # A "remove/delete X" request whose target word matches more than one
    # real symbol comes back as needs_selection instead of a guess -- ask,
    # then re-run with the chosen symbol pinned (skips the LLM entirely,
    # since the delta's shape is fully determined once confirmed).
    confirm_symbol = None
    confirm_symbol_type = None
    confirm_symbol_line = None
    try:
        while True:
            with StepTracker(console) as tracker:
                metadata = run_edit(
                    project_dir=Path(args.project_dir),
                    file=file,
                    request=args.request,
                    test_target=args.test_target,
                    project_id=args.project_id,
                    confirm_symbol=confirm_symbol,
                    confirm_symbol_type=confirm_symbol_type,
                    confirm_symbol_line=confirm_symbol_line,
                    use_hybrid_retrieval=args.hybrid,
                    use_joern=args.joern,
                    on_step=tracker.step,
                )
                status = metadata["result"]["status"]
                tracker.finish(ok=status == "success")

            if status != "needs_selection":
                break

            console.print()
            console.print("[bold yellow]multiple matches -- which one do you mean?[/bold yellow]")
            candidates = metadata["candidates"]
            for i, c in enumerate(candidates, start=1):
                console.print(f"  [{i}] {c['name']} ({c['symbol_type']}, lines {c['start_line']}-{c['end_line']})")
            console.print("  [0] cancel")
            try:
                choice = console.input("[bold cyan]select: [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                # No interactive terminal to answer from (piped input,
                # non-interactive script) -- exit cleanly instead of a raw
                # traceback; nothing was written, so there's nothing to
                # undo, just nothing further to do without an answer.
                console.print("\n[dim]no input available -- cancelled.[/dim]")
                sys.exit(1)
            if not choice.isdigit() or not (1 <= int(choice) <= len(candidates)):
                console.print("[dim]cancelled.[/dim]")
                sys.exit(0)
            picked = candidates[int(choice) - 1]
            confirm_symbol = picked["name"]
            confirm_symbol_type = picked["symbol_type"]
            confirm_symbol_line = picked["start_line"]
    except FileNotFoundError as e:
        console.print(f"[bold red]error:[/] {e}")
        sys.exit(1)

    console.print()
    console.print(_summary_table(metadata))
    sys.exit(0 if metadata["result"]["status"] == "success" else 1)


def cmd_find(args: argparse.Namespace) -> None:
    from .retrieval.locate_repo import locate

    console.rule(f"[bold]find[/] {args.request}")
    project_dir = str(Path(args.project_dir).resolve())
    console.print(f"[dim]{project_dir}[/dim]")

    status_msg = "indexing repository + hybrid retrieval..."
    if args.joern == "on":
        status_msg += " (building a Joern CPG -- real JVM cost, can take a while)"
    elif args.joern == "auto":
        status_msg += " (will build a Joern CPG only if confidence turns out low)"
    with console.status(f"[bold cyan]{status_msg}", spinner="dots"):
        result = locate(
            project_dir,
            args.request,
            use_vector=not args.no_vector,
            use_semgrep=not args.no_semgrep,
            use_joern={"off": False, "auto": "auto", "on": True}[args.joern],
        )

    console.print(f"symbols indexed: {result['symbols_indexed']}   confidence: {result['confidence']}")
    if args.joern != "off":
        joern_note = "resolved via real Joern CPG" if result["used_joern"] else "native graph used (Joern skipped)"
        console.print(f"[dim]call graph: {joern_note}[/dim]")
    console.print()

    if not result["candidates"]:
        console.print("[yellow]no candidates found[/]")
        return

    table = Table(show_header=True, header_style="bold")
    for col in ("file", "symbol", "lines", "fused score", "risk", "callers"):
        table.add_column(col)
    risk_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
    for c in result["candidates"]:
        caller_count = len(c["semgrep_call_sites"]) if not args.no_semgrep else c["called_by_count"]
        table.add_row(
            c["file"],
            c["symbol"],
            f"{c['start_line']}-{c['end_line']}",
            str(c["fused_score"]),
            f"[{risk_color.get(c['risk'], 'white')}]{c['risk']}[/]",
            str(caller_count),
        )
    console.print(table)


def cmd_metadata(args: argparse.Namespace) -> None:
    from .analyzer.metadata_builder import build_project_metadata, metadata_cache_path

    project_dir = str(Path(args.project_dir).resolve())
    console.rule(f"[bold]metadata[/] {project_dir}")

    with console.status("[bold cyan]parsing every file -- no LLM call...", spinner="dots"):
        doc = build_project_metadata(project_dir, use_cache=not args.no_cache)

    symbol_count = sum(len(f["symbols"]) for f in doc["files"].values())
    console.print(f"files indexed:   {len(doc['files'])}")
    console.print(f"symbols indexed: {symbol_count}")
    if args.no_cache:
        # build_project_metadata(use_cache=False) deliberately skips both
        # reading AND writing the cache file -- claiming a path here when
        # nothing was actually written to it was a real, misleading bug.
        console.print("written to:      [dim](--no-cache: not written to disk this run)[/dim]")
    else:
        console.print(f"written to:      [dim]{metadata_cache_path(project_dir)}[/dim]")

    if args.file:
        info = doc["files"].get(args.file)
        if not info:
            console.print(f"[bold red]error:[/] '{args.file}' not in this project's metadata")
            sys.exit(1)
        table = Table(show_header=True, header_style="bold")
        for col in ("name", "type", "lines", "params", "decorators", "calls", "called by"):
            table.add_column(col)
        for s in info["symbols"]:
            table.add_row(
                s["name"],
                s["symbol_type"],
                f"{s['start_line']}-{s['end_line']}",
                ", ".join(s["parameters"]),
                " ".join(s["decorators"]),
                ", ".join(s["calls"]),
                str(s["called_by_count"]),
            )
        console.print()
        console.print(table)


def cmd_serve(args: argparse.Namespace) -> None:
    from .webapp import main as serve_main

    console.print(f"[bold]serving[/] http://{args.host}:{args.port}")
    serve_main(host=args.host, port=args.port)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="iee", description="Adaptive Incremental Editing Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="write a brand-new file from a prompt (FULL_REGENERATION)")
    p_create.add_argument("--project-dir", default=".", help="project directory (created if missing), default '.'")
    p_create.add_argument("--file", required=True, help="new file path, relative to --project-dir")
    p_create.add_argument("--request", required=True, help="what the file should do")
    p_create.add_argument("--test-target", default=None, help="optional pytest path to validate against")
    p_create.add_argument("--project-id", default=None, help="storage project id, default = project dir name")
    p_create.set_defaults(func=cmd_create)

    p_edit = sub.add_parser("edit", help="incrementally edit an existing file (STRUCTURED_EDIT)")
    p_edit.add_argument("--project-dir", default=".", help="project directory, default '.'")
    p_edit.add_argument(
        "--file", default=None, help="target file, relative to --project-dir (omit to auto-locate via hybrid retrieval)"
    )
    p_edit.add_argument("--request", required=True, help="the change to make")
    p_edit.add_argument("--test-target", default=".", help="pytest path to validate against, default whole dir")
    p_edit.add_argument("--project-id", default=None, help="storage project id, default = project dir name")
    p_edit.add_argument(
        "--hybrid",
        action="store_true",
        help="also fuse BM25 + vector/semantic retrieval into localization (like `iee find`, scoped to "
        "this file) -- off by default: vector retrieval is a real embeddings-API call on every request",
    )
    p_edit.add_argument(
        "--joern",
        action="store_true",
        help="before a mechanical rename only, check for real cross-file callers via Joern's CPG -- the one "
        "case this pipeline can otherwise miss (the rename only rewrites the file it's given). Never blocks "
        "the rename, just warns. Off by default: a real ~12-45s+ JVM cost, and requires Joern installed.",
    )
    p_edit.set_defaults(func=cmd_edit)

    p_find = sub.add_parser("find", help="hybrid retrieval only: which file/symbol matches a request, no editing")
    p_find.add_argument("--project-dir", default=".", help="project directory to search, default '.'")
    p_find.add_argument("--request", required=True, help="what you're looking for")
    p_find.add_argument("--no-vector", action="store_true", help="skip the embeddings-based retriever")
    p_find.add_argument("--no-semgrep", action="store_true", help="skip semgrep call-site verification")
    p_find.add_argument(
        "--joern",
        choices=["off", "auto", "on"],
        default="off",
        help="off (default): never resolve the call graph via Joern. auto: only when this call's own "
        "confidence is low (top candidate barely beat the runner-up) -- the confidence/need gate. "
        "on: always. Joern gives a real CPG instead of the name-only native graph, but requires "
        "joern/joern-parse installed separately and costs real build time (~12s+ even for a tiny "
        "project, more for a real one, cached after the first run).",
    )
    p_find.set_defaults(func=cmd_find)

    p_metadata = sub.add_parser(
        "metadata", help="generate/inspect the static, zero-LLM-call metadata.json for a project"
    )
    p_metadata.add_argument("--project-dir", default=".", help="project directory to index, default '.'")
    p_metadata.add_argument("--file", default=None, help="show the per-symbol table for one file (relative path)")
    p_metadata.add_argument("--no-cache", action="store_true", help="force a full rebuild, ignoring any cached copy")
    p_metadata.set_defaults(func=cmd_metadata)

    p_serve = sub.add_parser("serve", help="launch the local web frontend (chat-style, streams pipeline steps)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
