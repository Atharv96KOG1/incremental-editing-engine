# Incremental Editing Engine (IEE) — Complete Technical Documentation

## 1. What this project is and why it exists

IEE takes a natural-language edit request ("add validation to `login`", "rename `col` to `column`", "add a subtract function") against an existing codebase and produces the **smallest possible, targeted change** — instead of doing what most LLM coding tools do by default: send the whole file in, get the whole file back out.

**Core thesis:** an LLM edit should cost roughly proportional to *how much actually changed*, not to *how big the file is*. A one-line addition to a 2,000-line file should not cost 2,000 lines of input and output tokens.

Everything in this codebase — the locator, the Delta IR, the mechanical fast paths, the hybrid retrieval, the escalate mechanism — exists in service of that one thesis. Where the codebase deviates from it (a genuine whole-file rewrite), it's because the request itself, or the language/format involved, structurally requires it — never because the system defaulted to the expensive path out of convenience.

### The optimization being made

```
minimize:  tokens_in + tokens_out + latency + risk_of_wrong_edit
subject to:  correctness (syntax valid, tests pass, semantically right)
```

Every mechanism below is a lever on one side of that trade.

---

## 2. High-level request flow

```
User request + file
        │
        ▼
┌───────────────────────┐
│  Zero-LLM fast paths   │  rename? delete? whole-file-delete-by-name?
│  (mechanical, free)    │  → if matched, dispatch directly, no LLM call at all
└───────────┬────────────┘
            │ no match
            ▼
┌───────────────────────┐
│  Jev pre-classification│  (optional) fast, cheap, typed model call:
│  (opt-in)              │  "question" or "whole_file"? → dispatch directly
└───────────┬────────────┘
            │ no confident answer
            ▼
┌───────────────────────┐
│  Does this language/   │  no →  TEXT_BLOCK_EDIT (locate the one relevant
│  format have a         │        section — heading, [ini] block, YAML key,
│  function/class        │        HTML element, paragraph — regenerate only
│  concept at all?       │        that, splice back)
└───────────┬────────────┘
            │ yes
            ▼
┌───────────────────────┐
│  Localize: which       │  name/docstring match, optionally fused with
│  symbol(s) does this   │  BM25 + vector retrieval (hybrid mode)
│  request target?       │
└───────────┬────────────┘
            ▼
┌───────────────────────┐
│  Build minimum context │  imports + candidate symbol(s) in full +
│                        │  one compact line naming every other symbol
└───────────┬────────────┘
            ▼
┌───────────────────────┐
│  STRUCTURED_EDIT:      │  model returns Delta IR JSON (REPLACE/INSERT/
│  ask for a Delta IR    │  DELETE on named symbols) OR escalates
└───────────┬────────────┘
            │
     ┌──────┴───────────────────────────────────────────┐
     │ escalate: whole_file (edit → REFUSED by policy,    │
     │ see §6) / language_conversion / create_files /     │  → dedicated
     │ delete_file / rename_identifier / question         │    strategy
     └──────┬───────────────────────────────────────────┘
            │ normal case: real Delta IR ops
            ▼
┌───────────────────────┐
│  Validate: schema +    │  every target must exist exactly once;
│  target existence      │  REPLACE must restate decorators; a target whose
│                        │  body was never shown can't be REPLACEd blind
└───────────┬────────────┘
            ▼
┌───────────────────────┐
│  Apply: splice into    │  bottom-to-top by line, so earlier edits never
│  the real source text  │  shift a later target's line numbers
└───────────┬────────────┘
            ▼
┌───────────────────────┐
│  Validate: syntax →    │  a genuine subprocess `import`, not just
│  imports → tests       │  ast.parse — catches "parses fine but doesn't
│  (in a temp copy)      │  actually resolve" bugs
└───────────┬────────────┘
            │ pass                              │ fail
            ▼                                    ▼
┌───────────────────────┐          ┌───────────────────────────┐
│  Commit: new version,  │          │  Bounded repair (structured │
│  write metadata, diff  │          │  edits) or fall back a tier  │
└───────────────────────┘          │  (narrow import fix → whole  │
                                    │  file; never silently corrupt)│
                                    └───────────────────────────┘
```

---

## 3. Delta IR — the structured edit representation

Nothing in this project lets an LLM blindly overwrite a file. Every structured edit is expressed as a small, validated JSON object first.

```json
{
  "schema_version": "1.0",
  "base_version": "v4",
  "operations": [
    {
      "operation": "REPLACE",
      "target": {"symbol_type": "function", "symbol_name": "login", "anchor": null},
      "content": "def login(user, password):\n    ..."
    }
  ]
}
```

- **Operations:** `REPLACE`, `INSERT`, `DELETE` — always scoped to a named `function`/`class` symbol.
- **`content`** for REPLACE/INSERT must be the symbol's *complete* new body — never a partial diff. This is deliberate: a partial diff needs its own conflict-resolution logic; a complete replacement just needs "does this symbol still exist, does it still parse."
- **`schema_version`/`base_version`/`target.file`** are never asked of the model — the caller already knows all three before the call is made, and nothing downstream re-reads the model's own copies of them. Asking for them was pure wasted output tokens (billed above input's rate) for values the model was never the source of truth for. They're backfilled onto the response afterward (`_fill_known_fields`).
- **`escalate`** (optional field, same JSON object) is how the model tells the pipeline "this genuinely can't be expressed as REPLACE/INSERT/DELETE" — see §6.

### Structured-output enforcement (strict JSON schema)

The API call itself enforces the operations/escalate *shape* via OpenAI-compatible strict `json_schema` structured output (`STRICT_DELTA_RESPONSE_SCHEMA` in `structured_edit.py`) — the system prompt no longer needs to restate the JSON shape in English/JSON-literal form at all. This cut the system prompt from 622 → 464 tokens (~25%) by removing pure schema restatement, leaving only the parts a schema can't express: *when* to use which field, not *what* fields exist. A later pass (fixing the `whole_file`/local-import edge cases in §6, plus wording-only tightening) landed at 494 tokens — a further ~6% cut even after adding two genuinely new correctness rules, by removing a quoted literal comment marker that no longer needed to match verbatim. One more real correctness fix (telling the model a shown class method, constructor included, is itself a directly REPLACEable function-type target — see §4.1's class-span bug) brought it to 517 tokens; still a net ~17% cut from the original 622 despite three genuinely new behavioral rules added along the way.

Strict mode requires every property to be listed as required (nullable where truly optional) — a real correctness trap this project already hit and fixed: `_strip_strict_nulls()` immediately normalizes a parsed response back to the lenient "key simply absent when unused" shape right after parsing, so every downstream consumer (`validate_schema`, `_fill_known_fields`, `Operation.from_dict`, run_pipeline's own `.get("escalate") or {}`) never has to know strict mode exists.

### Validation (two stages, `delta/validator.py`)

1. **Schema validation** — `jsonschema` against `DELTA_JSON_SCHEMA` (the internal, lenient shape; separate from the strict API-facing schema above).
2. **Target validation** (`validate_targets`) — confirms every REPLACE/DELETE target and INSERT anchor exists *exactly once* in the real source. Real bugs this stage exists to catch:
   - **Decorator loss**: a REPLACE for a Flask route handler once dropped its own `@app.route(...)` decorator — the file still parsed, nothing else failed, and the endpoint silently 404'd until someone noticed. `_check_decorators_preserved` now checks this structurally against the *original* symbol's real lines.
   - **Blind REPLACE on unseen content** (`UnseenReplaceTargetError`): when a request falls back to compact context (imports + bare names, no bodies) and the model still emits a REPLACE for one of those bare names, it has to *fabricate* the symbol's "new" body since it never saw the real one — a real, observed case silently dropped an extra table and half the original columns. This is a distinct exception subclass specifically so `run_pipeline.py` can force a direct escalation to whole-file regeneration instead of retrying with the same insufficient context (mechanical-over-prompt-trust: force the correct recovery path in code, don't hope the model self-corrects).
   - **Duplicate definitions**: an INSERT's *actual* defined name (via `defined_symbol_name`, which looks at what the content really defines, never the model's claimed `symbol_name`) is checked against existing symbols — a real corruption case once left a class-method version of a function alongside an un-removed old top-level one, both silently coexisting until a later edit hit an unresolvable "defined N times" error.

---

## 4. Localization — how the system decides *where* to edit

### 4.1 Single-file locator (`analyzer/locator.py`)

The oldest, most heuristic-dense part of the system — cheap, local, zero-LLM-call scoring of which symbol(s) in *one already-known file* a request likely targets. Falls back to "no confident match" rather than guessing (in which case context building degrades to imports + a compact name line, never silently to the whole file for the wrong reason).

Real production bugs this file's scoring rules exist to fix (each one a genuine regression caught live, not a hypothetical):

- **Stopword filtering**: "add the association types like aglomarative and divisive" shared nothing with a class's docstring except the word "and" — that alone scored high enough to select a class containing nearly every method in the file.
- **Generic leading-verb suppression**: "add sin inverse function" pulled in `add()`'s entire body purely because "add" is the request's own first word.
- **Corroboration requirement**: "add the cot inverse function" pulled in `atan_inverse` purely because both names share "inverse."
- **Trivial-delegate resolution**: after a "wrap this in a class" refactor leaves old flat functions as one-line forwarding shims, editing a name should mean the real implementation, not the shim — detected structurally (tiny body, a call expression referencing the impl's class + shared name), never by hardcoded name/file.
- **Semantic tiebreak**: when free heuristics can't uniquely resolve a tie (e.g. `Trig.tan` vs `Hyperbolic.tan`), a real embeddings call breaks it — with the enclosing class name prefixed onto the body text first, since near-duplicate method bodies otherwise carry almost no differentiating signal.
- **Delete-candidate fuzzy fallback**: "remove substract" (typo) against a real `subtract` used to escalate all the way to a full-file regeneration (~$0.03, 10+ seconds) just to find a name a spell-check-grade `difflib` match already could, for free.
- **Multi-delete parsing**: "delete tan, sec and cos" against a trig file matched 10 symbols via broad substring search (tan/tanh/atan_inverse/cos/cosin/...) — `find_multi_delete_targets` is a strict all-or-nothing parser that only fires when every comma/and-separated phrase resolves to exactly one unambiguous exact match.
- **Rename beyond ASCII word boundaries**: "use anthropic instead of openai" against a Go file using `apiKeyOpenAI`/`buildOpenAIClient` silently renamed nothing — plain `\bold_name\b` regex has no concept of a camelCase transition as a boundary. `rename_with_subword_fallback` splits on case transitions (with a second split rule specifically for acronym tails like "AI" in "OpenAI", so "AIClient" splits as "AI"+"Client" not "A"+"I"+"Client") and tries sub-word spans.
- **A class's own span swallows its methods' body matches**: `locate_candidates_by_body` scores a symbol by searching its own line range for request words — but a class's line range *structurally contains* every one of its methods, so a word matching only inside one nested method (e.g. "temperature" only inside a constructor) scored an independent "hit" for the *enclosing class* too, purely because the match happened to fall inside its line range. Real, live-observed case: "make the default temperature 0.5" against a TypeScript class whose constructor set it caused the whole class — three unrelated methods (`ingestDocuments`, `ingestText`, `ingestDirectory`) included — to get selected and handed to the model as the candidate, which then had to restate (or, worse, silently drop) all three just to change one value. Confirmed via the real user's own Bifrost logs, reproduced directly, and fixed: whenever one scored symbol's line span fully contains another's, the containing (outer) one is now dropped, keeping only the more specific match. Verified live end-to-end: the same request went from a 3009-token whole-class REPLACE to a 993-token constructor-only one.
- **A compound/multi-line request silently lost its second target** (`context/context_builder.py`'s `build_context`): `locate_candidates_by_body` (the body-content fallback, §4.1 above) used to run *only* when name/docstring matching found nothing at all. A request bundling two clauses against two *different* symbols ("change the temperature to 0.5" + "also rename ask_all to ask_everyone") named one target literally (`ask_all`, matched by name) while the other (`temperature`, inside a completely different method's body) never got a chance to surface at all once the first candidate existed — that method's content was never shown to the model, so it structurally couldn't be touched. Not escalated, not refused — reported as a plain `"success"` with half the request silently undone. Fixed: the body-content locator now always runs and merges in any additional symbol it finds, whole-word, whenever it doesn't overlap a symbol already selected — real cost stays proportional (only the additional *matched* symbol's body is added, never the whole file). Guarded against a real regression this surfaced: a body match sharing the same *name* as an already-selected symbol (the exact "`Trig.tan` vs `Hyperbolic.tan`" duplicate-name case above) is never added as a second entry — `candidate_lines` assumes at most one occurrence per name, and a second entry silently pointed a later REPLACE at the wrong occurrence.

**Architectural note directly from the module's own comment:** file-wide structural/hygiene detection and language-conversion detection *used to* be hand-written keyword/regex lists here — an "unwinnable, ever-growing whack-a-mole." That responsibility now belongs entirely to the model's own `escalate` field (§6). Nothing in this module hardcodes a language name or a structural keyword anymore. This is a deliberate, repeatedly-enforced project rule: **classification of open-ended natural language is the model's job, never a keyword/regex classifier** — the only "classifiers" that exist mechanically in this codebase (rename/delete fast paths) work by matching against *real, already-verified structure* (an exact existing symbol name), never by guessing intent from wording, and they always safely fall through to the model when they don't fire.

### 4.2 Multi-language symbol extraction (`retrieval/multilang_symbols.py`)

Python gets its own `ast`-based path (docstrings, decorator-aware spans, class scope). Every other language goes through Tree-sitter, generically: a node is function-like if its type name contains "function"/"method", class-like if it contains "class"/"struct"/"interface" — no per-language query needed, verified across the actual grammars rather than assumed.

**Two real, structural bugs found and fixed this session** (via live grammar introspection, the same technique used throughout):

1. **The declaration-word gate excluded valid languages.** The original gate required `"declaration"/"definition"/"specifier"` in the node type name. Rust's `function_item`/`struct_item` contain none of those (fixed by adding `"item"`); Ruby's function/class definitions are bare `method`/`class` node kinds (fixed by allowing those specific bare types too) — but bare words are dangerous: several grammars (Python, JS, C++...) separately expose the literal keyword *token* (`class`, `def`) under that exact same type name. The fix that makes allowing bare words safe: require the actual parsed node to be `is_named` and have `child_count > 0` — verified empirically that the keyword-literal token is always unnamed with zero children, while a real definition container never is.
2. **Name extraction failed for C/C++.** `child_by_field_name("name")` returned `None` for every C/C++ function — their grammar nests the identifier two levels down inside a `function_declarator`, with no top-level "name" field at all. Fixed with a generic DFS fallback (`_find_name_node`) that finds the first `*identifier`-suffixed descendant — safe because a definition's own name always appears in source order before its parameters/body.

Before this fix: **C returned zero functions, C++ missed every free function and every class method, Rust and Ruby returned nothing at all.** Verified live after the fix (real CLI runs): REPLACE/INSERT now work correctly on real `.c`/`.cpp`/`.rs`/`.rb` files.

**Currently confirmed working with full symbol-level precision** (tested directly, ~50 of ~61 mainstream languages checked): Python, JavaScript, TypeScript, Java, C, C++, C#, Go, Rust, Kotlin, Swift, PHP, Ruby, Dart, Scala, Lua, R, Julia, Haskell, Objective-C, Perl, Crystal, Fortran, COBOL, VB, F#, OCaml, MATLAB, Solidity, Bash, PowerShell, Fish, SQL, Pascal, Elm, D, Ada, Verilog, VHDL, GraphQL, and more.

**Known, verified gap** (falls back to the less-precise text-block path, not full symbol precision): Elixir, Erlang, Clojure, Groovy, Zig, Nim, T-SQL, Common Lisp, Scheme, Racket, Prolog, Tcl — their grammars name function-definition nodes in a shape the current heuristic doesn't yet recognize. Same class of gap as the C/C++/Rust/Ruby bug above, just not yet checked/fixed.

`language_has_symbol_concept(language)` answers, via real grammar introspection (not a hardcoded per-language list), whether a language's grammar has *any* function/class-shaped node kind at all — this is what lets the pipeline skip STRUCTURED_EDIT's classification call entirely for markdown/JSON/YAML/CSV/HTML/CSS/INI/dotenv (which could only ever answer "escalate," every single time, wasting the whole classification prompt).

### 4.3 Text-block locator (`analyzer/text_blocks.py`) — for languages/formats with no function/class concept

Treats a file's own natural section structure as pseudo-symbols instead of giving up and sending the whole file:

- Markdown → headings
- TOML/INI → `[section]` headers
- YAML → top-level keys
- HTML → real tag-nesting-aware blocks, each direct child of `<body>`, via the actual Tree-sitter parse tree (a blank-line heuristic alone finds nothing in typical, densely-formatted HTML)
- Anything else (Dockerfile, `.env`, plain text) → the universal blank-line-paragraph fallback

Real waste this closes: "add retrieval types in the notes part" against a 30-line README previously paid ~1045 tokens for STRUCTURED_EDIT's classification-call system prompt alone (a markdown file could only ever answer "escalate"), then regenerated the entire file to add three words to one section.

Supports the same hybrid retrieval fusion as code symbols (`locate_text_block(..., use_hybrid=True)`): name/word-overlap match, fused with BM25 and vector retrieval over every block in the file — closing the gap where a request sharing only a rare *body* word (not the section heading) with the right section would otherwise miss it entirely.

**Chunk overlap** (`block_context_window`): a few lines immediately outside the target block's own range are shown to the model as read-only context (never spliced back) — closing a real gap where a boundary-sensitive edit ("match the tone of the section above") had zero visibility into neighboring sections. The block's own splice boundaries stay exact; only the *generation-time context* overlaps neighbors, standard RAG chunk-overlap technique.

**`<script>`/`<style>` were invisible to the HTML locator entirely** (real bug, fixed): tree-sitter's HTML grammar gives `<script>`/`<style>` their own distinct node types (`script_element`/`style_element`), never plain `"element"` — the block extractor's children filter only matched `"element"`, so any request touching inline JS/CSS had zero narrow target, no matter how the rest of the file localized. Fixed by including both node types; verified they now index as their own blocks (`"script"`, `"style"`).

**Multi-block editing** (`locate_text_blocks`, `_run_multi_text_block_edit` in `run_pipeline.py`, `MULTI_TEXT_BLOCK_EDIT` strategy): `locate_text_block`'s single-best design commits to exactly one block, refusing on a tie — correct when the ambiguity is genuinely "which one," but wrong when a request legitimately needs *several* blocks touched together. Real, motivating case: "change the helloBtn to btn" against an HTML file needs both the button's own `id` attribute *and* the inline `<script>` referencing that id — neither block alone is a complete, correct change, so the single-best locator saw an equally-scoring tie and refused outright (`WHOLE_FILE_BLOCKED`) even though both real targets were known.

`locate_text_blocks` (plural) returns *every* block whose name or body content confidently relates to the request (merging `_name_match_ranking` with a new `_body_match_ranking` — the same name/body-content split `analyzer/locator.py` already uses for functions, mirrored here since a block's own "name" is just its tag/heading, never whatever identifier the request actually names) instead of committing to one. `_run_whole_file_edit`'s own dispatch checks this *before* the single-best locator: more than one match → one combined generation call regenerates every matched block at once (a controlled, delimited multi-block prompt/response format — `===BLOCK: <name>===` ... `===END BLOCK===` — parsed back per block, spliced bottom-to-top by line so an earlier splice's line-count change never shifts a later block's still-pending range); exactly one match → the existing single-block path, unchanged; zero matches → the existing whole-file refusal, unchanged. A block the request doesn't actually need is instructed to come back byte-identical, so a block that matched only by sharing a word costs nothing beyond that one block's own tokens — never silently kept unaddressed: a block missing from the model's response is a hard `GENERATION_INCOMPLETE` failure, not a silently-reused stale copy. Verified live: the exact `helloBtn`/`<script>` case above, real request, `MULTI_TEXT_BLOCK_EDIT`, 338 tokens, both blocks correctly and consistently updated.

### 4.4 Repo-wide hybrid retrieval (`retrieval/` — `iee find` and, more narrowly, per-file `use_hybrid_retrieval`)

For "which file/symbol across the whole repo" (`iee find`) and, at file scope, for localization within one already-known file:

- **`SymbolRetriever`** — literal-name/keyword-overlap scoring (same technique as the single-file locator, repo-wide).
- **`BM25Retriever`** — keyword retrieval over name + docstring + full source text, tokenized with camelCase/snake_case splitting so `build_knowledge_base` and `BuildKnowledgeBase` tokenize identically. Strong for exact identifiers, rare terms, error messages.
- **`VectorRetriever`** — real OpenAI embeddings (via the same Bifrost gateway used for generation), cosine similarity over an in-memory numpy array, with a content-hash cache so a long-running process doesn't re-embed unchanged symbols. Strong for semantic intent, paraphrases.
- **`graph_centrality_ranking` (`retrieval/graph_rank.py`)** — a *structural* signal, not a textual one: a personalized PageRank over the repo's own native call graph (`dependency_graph.py`), so a heavily-called symbol (e.g. `authenticate()`, called from several places) outranks an equally-uncalled one even when neither's name textually matches the request at all. Adapted from Aider's own repo-map ranking (`aider/repomap.py`'s `get_ranked_tags`) — same idea (a referencer → definer graph, personalized toward request-mentioned identifiers, ranked via PageRank) reimplemented at symbol granularity over this project's own already-computed call graph, and with a plain NumPy power iteration instead of adding `networkx` as a new dependency (NumPy is already one). Personalization: any symbol whose own name is a real content word in the request gets boosted before ranking — the same word-overlap technique `locate_candidates` already uses, not a new heuristic. Verified live: "fix the login bug" against a real `authenticate`/`check_password`/`hash_it` call chain plus two disconnected functions ranked the entire connected chain above both disconnected ones, despite "login bug" textually matching none of their names.
- **`fuse()`** — Reciprocal Rank Fusion across all signals: a symbol's fused score is the sum of `1/(60 + rank)` across every retriever that surfaced it, so a symbol every signal agrees on outranks one only a single retriever liked.
- **`retrieval_confidence()`** — not an absolute score (RRF scores are hard to interpret on their own); measures the *margin* between the top result and the runner-up. A clear winner scores high; a near-tie scores low regardless of the raw fused number.
- **`classify_risk()`** — heuristic LOW/MEDIUM/HIGH per symbol (keyword-matched against auth/crypto/session/sql/etc., or "public class" always HIGH) — used to decide how much scrutiny an edit needs, not a learned model, by design.
- **Semgrep as a structural filter, never a retrieval engine** (`semgrep_refs.py`): `find_call_sites` verifies a *real* call expression exists (not a name that happens to appear in a comment/string); `structural_match_score` checks whether a candidate's code actually *does* what the request describes (an `isinstance`/`raise`/loop/decorator/regex pattern really present in the body) — grounded in real code structure, not just word overlap. Only ever narrows/reranks a shortlist another signal already found; can never invent a candidate from nothing.
- **Joern (optional, tri-state `false`/`true`/`"auto"`)**: a real Code Property Graph call graph, resolving callers/callees by actual per-file identity instead of the native graph's name-only matching (where two unrelated `foo` functions in different files look identical to it). Real cost: ~12–45s+ (JVM/Scala startup dominates, not the analysis itself) — never on the fast edit-auto-locate path, only reachable from `iee find` when explicitly requested, and `"auto"` only pays that cost when confidence is already low. **Deliberately never wired into ranking/fusion itself** — it can only make the evidence attached to an already-chosen candidate more trustworthy, never change which candidate wins.

`use_hybrid_retrieval` defaults `False` everywhere in the core `run_edit()` function — vector retrieval is a real embeddings-API call, and the entire offline test suite relies on this defaulting off to stay deterministic and network-free. The CLI (`--hybrid`) and web UI (on by default there) opt in explicitly per surface.

### 4.5 Jev pre-classification (`retrieval/jev_router.py`) — optional, fast structured-decision routing

[TypeSafe AI's Jev](https://docs.typesafe.ai) is a "System One" model: not a chat model, a fast (70–500ms), type-safe structured-decision model — "unstructured state in, typed probabilistic decisions out," with a calibrated confidence score on every answer.

STRUCTURED_EDIT's own classification already decides "what kind of request is this" correctly — but only as a side effect of a real generation call that also produces Delta IR operations, priced for that whole job. Two of the seven escalate kinds (`question`, `whole_file`) need nothing beyond the request's own wording to dispatch directly — so a confident Jev call *ahead of* context-building can skip straight to the right handler and skip STRUCTURED_EDIT's classification generation entirely.

This is deliberately **not** a hardcoded keyword/regex classifier — the decision is still made by a model, with a real confidence score, on the request text itself; it's the same "escalate is model-driven" contract STRUCTURED_EDIT already holds itself to, just backed by a purpose-built, faster/cheaper model for exactly this kind of decision. A low-confidence or unavailable answer (no API key configured, package not installed, any API failure) always degrades to "don't shortcut" — the existing pipeline runs completely unchanged. Confidence threshold lives in `Settings.jev_confidence_threshold` (default 0.80 — higher than the 0.5 a customer-service routing example might use, since a wrong shortcut here means silently answering a real edit as a question, or vice versa, not just "a human re-reads the ticket").

Opt-in via `use_aider`-style flag (`use_jev`-equivalent wiring in `run_pipeline.py`); off by default (`TYPESAFE_API_KEY` unset = feature simply doesn't exist for that run).

### 4.6 Persisted metadata (`analyzer/metadata_builder.py`) — one JSON per file, covering every language and format uniformly

Zero-LLM-call, one JSON artifact per file (`<project>/iee_metadata/`, written at every commit point) or per project (`~/iee-metadata/`, fingerprint-cached): name, type, exact line span, and a keyword set for every indexed region — parameters/decorators/return-type/async-ness/docstring/parent-class/calls/called-by too, for Python specifically (its own real `ast` parse gives these cheaply; a generic Tree-sitter node-type heuristic can't).

**Covers every language and format the same way localization already does, not just "languages with functions"**: `extract_symbol_metadata` tries real function/class symbols first (`ast` for Python, Tree-sitter's generic node-type match for everything else), and whenever that finds *nothing* — either because the format has no such concept at all (markdown, TOML, YAML, HTML, Dockerfile, plain text, an unrecognized extension) or this particular file just happens to define none — falls back to `analyzer/text_blocks.py`'s own generic block locator (the exact same one `_run_text_block_edit`/`_run_multi_text_block_edit` use to localize an edit) instead of returning empty. A markdown heading, a TOML/INI section, a YAML top-level key, an HTML tag, or a blank-line paragraph all become a real, named, line-ranged `SymbolMetadata` entry (`symbol_type="block"`) — one locator technique, reused for metadata the same way it's already reused for hybrid retrieval, not a second per-format pass.

Real gap this closed: `build_project_metadata`'s own per-file loop used to `continue` (skip entirely, no metadata at all) for any file whose language couldn't be detected, and `write_file_metadata` returned `None` (wrote nothing) under the same condition — both assumed "no recognized language" meant "no symbols," which stopped being true the moment the block-level fallback existed. Fixed by routing both through `extract_symbol_metadata` itself (removing a second, duplicated symbol-extraction branch in the process) and only skipping when a file is genuinely empty. Verified live across five real files in one project — Python, Go, Markdown, TOML, YAML — each producing correct, real metadata through the identical code path.

---

## 5. Context building (`context/context_builder.py`)

Builds the *minimum* context sent to the model: module imports + the located candidate symbol(s) in full + one compact line naming every *other* symbol in the file (never bodies, never repeated if already shown in full).

When localization finds no confident candidate (the common case for "add a new function" — nothing existing to anchor against by name), context used to be the entire raw file. It's now imports + the same compact name line — still enough to avoid hallucinating an anchor or duplicating an existing symbol, without paying for every function's body.

The "other symbols" line includes real parameters and decorators per entry below a measured threshold (`_MAX_SYMBOLS_FOR_PARAM_DETAIL = 20`) — showing every parameter on every entry added 90 tokens (+71%) to that one line alone on a real 39-symbol file, genuinely useful per-entry but compounding linearly with symbol count; above the threshold, entries fall back to bare names (decorators still shown, since they're often the only way to identify a symbol without a body at all).

`use_hybrid=True` fuses this same name/docstring match with BM25 + vector retrieval scoped to just this one file — closing the gap where a request shares only a rare *body* word with a symbol, not its name or docstring.

---

## 6. Escalate kinds — when Delta IR genuinely can't express the change

STRUCTURED_EDIT's prompt tells the model to leave `operations` empty and set `escalate` (never guess symbol-level operations) whenever REPLACE/INSERT/DELETE genuinely can't do it:

| kind | meaning | dispatches to |
|---|---|---|
| `whole_file` | content genuinely outside *every* symbol's own span (a module docstring, a bare module-level statement, standalone comments, blank-line formatting between symbols) — **not** merely "touches many symbols" (that's still several REPLACE ops, each already restating its own complete body) | `_run_whole_file_edit` → **refused by policy** (see below) |
| `language_conversion` | a different language entirely; model supplies `target_language`/`target_extension` | full-file generation into a new file with the swapped extension |
| `create_files` | new file(s) unrelated to this file's own content; one path per real responsibility, never merged; a bare `"dir/"` path only when nothing at all belongs inside it | `_run_create_files` |
| `delete_file` | deletes this whole file, not one symbol in it | `_run_whole_file_delete` (with a real pytest safety check first) |
| `rename_identifier` | the whole change reduces to swapping exact text everywhere it occurs (identifier, env var, literal value); one entry per distinct swap | `_run_rename_identifier` (mechanical, zero-LLM-cost) |
| `question` | asks about the code rather than instructing a change | `_run_question` (no Delta IR, no apply/test/commit at all) |

### Whole-file regeneration is disabled for existing-file edits, by explicit policy

`_run_whole_file_edit` is the single choke point every `whole_file`-shaped escalation flows through — the model's own direct escalate, a confident Jev `"whole_file"` classification, the no-text-block-located fallback for a symbol-less language, the `UnseenReplaceTargetError` forced-escalation, and the mechanical rename's own "the narrow import fix didn't resolve it either" last resort. Rather than generating a full-file rewrite, it now **refuses outright**: `result.status = "failed"`, `failure_class = "WHOLE_FILE_BLOCKED"`, a clear explanation, and zero additional generation cost (only whatever classification call already ran — the one that discovered the escalation — is billed, folded in for honest accounting). Nothing is written to disk.

This is a deliberate, hard behavioral choice: **this pipeline only ever changes the specific code a request identifies, never the whole file** — even for content that genuinely has no smaller Delta-IR-expressible unit (a module docstring, a bare module-level statement, standalone formatting between symbols). The tradeoff is explicit: some requests that used to succeed via a full rewrite now fail cleanly instead. Narrowing the request (one function/class at a time) or splitting it into separate requests is the only way forward for that class of change.

**The prompt itself was tightened to genuinely reduce how often this triggers at all**, not just to refuse more cleanly when it does: `whole_file` is now explicitly scoped to content *outside every symbol's own span* — a request touching many or even all symbols is still expressible as several REPLACE operations (each already restates its own complete body regardless of how many others also change), so the model is steered to prefer that over escalating. A symbol needing an import not already present at module level is told to add it as a local import inside that symbol's own body instead of escalating — closing what was probably the single most common real trigger for an unnecessary `whole_file` escalate.

**CREATE and `language_conversion` are unaffected** and never call `_run_whole_file_edit` — both structurally have no existing symbol to target in the first place (a brand-new file, or a rewrite into a different language), so whole-file generation there isn't a fallback being chosen over something smaller, it's the only mechanism that could ever apply.

**Aider** was unwired from this path entirely (its only integration point) — see §8's own note; there is no `use_aider` parameter or flag anywhere in this codebase anymore. Jev's `"whole_file"` classification still fires and still skips STRUCTURED_EDIT's own classification call — it just leads to the same refusal afterward instead of a generation call.

### Module-level block editing (`analyzer/module_blocks.py`, `strategies/module_block_edit.py`) — a narrower escape hatch tried before the refusal above

Real gap: content genuinely outside every symbol's own span (a module-level constant, a config list, an `if __name__ == "__main__":` guard) is very often still just ONE small, narrowly-locatable region — not a reason to touch the rest of the file. Confirmed live case: "remove gpt" against a 138-line `chats.py` whose only `gpt`-related text lived inside a module-level `MODEL_SLOTS = [...]` list (never inside any function/class body) found zero symbol candidates and correctly reached `_run_whole_file_edit` — but the actual fix needed was a 7-line region.

Before refusing, `_run_whole_file_edit` now tries `locate_module_level_block(original_source, request, language)` — same "locate one narrow region, edit only that, splice back" technique `analyzer/text_blocks.py` already uses for symbol-less languages, applied here to content that sits *outside* every function/class's span in a language that otherwise has one:

- `index_module_level_blocks` walks every gap between/around real function/class spans (via the same `index_symbols` localization already uses), then chunks each gap into blank-line-separated blocks — a single gap can hold several unrelated statements (an import block, a constant, a `__main__` guard) that must stay independently targetable.
- Each block is scored by request-word overlap against its own body content (mirrors `locate_candidates_by_body`'s technique, §4.1) — a tie or zero matches returns `None` rather than guessing, same "refuse over guess" contract every other locator in this project holds itself to.
- On a confident, unique match: dispatches to `_run_module_block_edit` — mirrors `_run_text_block_edit` exactly (splice, syntax check, diff, preview, real pytest validation via `copy_project_for_validation`, confirmation, commit) — with `strategy = "MODULE_BLOCK_EDIT"` in the resulting metadata.
- On no match/ambiguous: falls through unchanged to the `WHOLE_FILE_BLOCKED` refusal.

**Guarded to only ever fire when the file has at least one real function/class symbol elsewhere** (`index_symbols(original_source, language)` non-empty). Without that guard, a file with zero symbols at all degenerates to a single gap spanning the *entire* file — "locating" that one block would just be a whole-file rewrite wearing a narrower name, exactly what the refusal above exists to prevent. This guard is what keeps e.g. a bare `import json` + one statement, or a single-paragraph markdown file, correctly refused rather than silently rewritten.

Verified live end-to-end: real `chats.py` copy, CLI `edit --request "remove gpt"` → `MODULE_BLOCK_EDIT`, only the 5-line `MODEL_SLOTS` list changed (`gpt-4o-mini` → `4o-mini` etc.), rest of the 138-line file byte-identical, syntax-valid, tests passed, 1465 tokens total.

---

## 7. Mechanical, zero-LLM-cost fast paths

These run *before* any LLM call, matched against real, already-verified structure (an exact existing symbol name) — never a guess at open-ended natural language, and always safely falling through to the model when they don't fire:

- **`find_rename_target`** — only fires when the request's `old_name` matches exactly one real symbol. Skips localization, context building, and the classification call entirely.
- **`is_delete_intent` / `find_delete_candidates` / `find_multi_delete_targets`** — an unambiguous single or multi-target delete constructs the Delta IR directly (empty-cost).
- **`is_whole_file_delete_target`** — matched against the file's own real basename/stem; runs a real pytest-in-temp-copy safety check before allowing the delete.

### Narrow import repair (`strategies/import_repair.py`) — closing a real gap in the rename fast path

`rename_with_subword_fallback` is mechanical text substitution — it correctly renames every occurrence of an identifier *inside the file it's given*, but a rename can leave the import statement's own module path inconsistent (real case: renaming a class reference `ChatOpenAI → ChatAnthropic` correctly updates every body call site, but the import's module path — `from langchain_openai import ...` — still points at the *wrong package*, a real `ImportError` the instant anything runs it).

Before this existed, **any** broken import after a mechanical rename escalated straight to a full-file rewrite — discarding the mechanical rename's own (otherwise entirely correct) output and asking the model to reconstruct the *entire* file from scratch to fix a handful of broken lines.

The fix: locate the exact line span of every top-level import statement (`_locate_import_span`, real AST node positions), send *only* that block plus the exact error text to a small, cheap, dedicated repair call, splice the fix back by line range — the same "locate the one region actually wrong, regenerate just that" technique as text-block editing, applied to imports specifically.

**A real correctness bug found and fixed while verifying this feature live**: the first version could silently commit broken code — reverting the import line without checking whether the function body still referenced a name the fix just removed (e.g., import reverted to `OpenAI`, but a function body still called `XAI(...)`, invisible to `check_python_imports` since that only executes module-level code, not function bodies, and no test suite necessarily covers the call path either). Fixed with a structural consistency check (`_imported_names` orphan-detection): if the fix would orphan a name the body still uses, the narrow fix is rejected and the pipeline correctly falls back to the full-file rewrite instead of committing something broken. Verified live both ways: a genuinely-fixable case (module path wrong, class name already consistent) now stays a zero-content-generation mechanical rename; a genuinely-unfixable case (renaming to a package that doesn't exist at all) correctly falls back, safely, instead of silently corrupting the file.

---

## 8. Generation strategies

| strategy | when | input/output shape |
|---|---|---|
| **STRUCTURED_EDIT** | default path for a symbol-scoped change | localized context in, Delta IR JSON out |
| **TEXT_BLOCK_EDIT** | no function/class concept in this language/format | one located block in, that block's new content out |
| **MECHANICAL_RENAME** | rename escalate/fast-path | zero LLM cost (regex substitution) unless a broken import needs the narrow repair call |
| **FULL_REGENERATION** | CREATE (nothing to target), or `language_conversion` | whole file out (nothing in, for CREATE); genuinely unavoidable for this class of change. **For an existing file's edit, `whole_file` escalate is refused instead of dispatched here** — see §6. |
| **FILE_DELETE** | `delete_file` escalate or whole-file-delete fast path | zero LLM cost; real pytest safety check first |
| **CREATE_FILES** | `create_files` escalate | one generation call per new file; optionally a follow-up call linking the current file to use them |
| **QUESTION_ANSWERING** | `question` escalate | whole file + question in, plain-text answer out; no Delta IR, no write, nothing to confirm |
| **binary artifacts** | target is `.xlsx`/`.db`/`.docx`/`.pptx`/`.pdf` | model outputs a *runnable script* (using openpyxl/python-docx/reportlab/sqlite3), executed once in a sandboxed temp dir to produce real bytes — an LLM's text output cannot literally *be* a binary file |
| **language_conversion** | model recognizes a full rewrite into a different language | whole file in, whole *new* file (different extension) out |

### Aider as an alternate generator — removed entirely

> **Deleted, by explicit request.** Aider was originally wired in as an alternate generator for the edit-time `whole_file` escalate. Once that path was refused rather than dispatched (§6), the integration became unreachable, and `use_aider`/CLI `--aider`/the webapp field were removed everywhere across `run_pipeline.py`, `cli.py`, and `webapp.py`. `strategies/aider_backend.py` (and its own test file) was left in place afterward, unwired but importable, in case whole-file generation for edits was ever reinstated — it never was, so both were deleted outright rather than kept as permanent dead code. Nothing in this codebase references Aider or `.aider-venv` anymore except `validation/tests.py`'s own copy-skip list (a defensive entry for a directory that might still exist on disk from before, not a live dependency).

The design is worth remembering even with the code gone: Aider (the external coding-agent CLI) could optionally generate the FULL_REGENERATION strategy's output instead of this project's own direct LLM call, run as a subprocess against its own isolated virtualenv (never imported in-process — real incident: installing it into the shared environment once silently downgraded this project's own `tree-sitter-language-pack` pin and broke multi-language detection, live, mid-session). Measured before removal: its own protocol overhead (~2.4k input tokens) was larger than this project's own ~40-token prompt for a typical small/medium file — a net regression, only paying off on a large enough file that output-token savings dominate. That measurement is why it was off by default even while it still existed.

---

## 9. Validation pipeline

1. **Syntax** (`check_syntax`) — real `ast.parse`/Tree-sitter parse, not a heuristic.
2. **Import resolution** (`check_python_imports`, Python only) — a genuine `python -c "import module"` subprocess against a temp copy of the project, catching "parses fine but doesn't actually resolve" bugs `ast.parse` alone can't see (real bug this caught: a mechanical rename left `from langchain_openai import ChatAnthropic` — syntactically perfect, a real `ImportError` the instant anything ran it).
3. **Tests** (`run_tests`) — real `pytest` subprocess against a temp copy of the project, parsed for pass/fail counts. `no_tests_collected` (pytest's own exit code 5) is treated as a pass, not a failure — nothing to regress isn't a regression.

**A REPLACE that renames a symbol as a side effect of a broader edit used to leave stale references elsewhere in the file** (`api/run_pipeline.py`'s `run_edit`, right before APPLY): a REPLACE only ever touches its own target's span — never the rest of the file — so a rename bundled inside a compound request (e.g. "change the temperature to 0.5, also rename `ask_all` to `ask_everyone`") could rewrite the *definition* while a call site elsewhere in the same file (`self.ask_all(...)`) stayed exactly as it was: syntactically valid, semantically broken, and committed as a plain `"success"` because the test suite never happened to exercise that call path. This is distinct from — and has no dedicated safety net the way — the mechanical `rename_identifier` fast path already has for a *pure* rename request. Fixed mechanically, not by trusting generation to have done it: for every REPLACE whose content defines a different name than its target (`defined_symbol_name` on the new content vs. the op's own `symbol_name`), check whether the old name is still referenced anywhere else in the *original* file outside that symbol's own span (guaranteed to still say the old name unchanged, since REPLACE never touches anything else) — and if so, mechanically sweep every remaining whole-word occurrence to the new name (`rename_with_subword_fallback`, the exact same regex the dedicated mechanical rename path already uses) before the syntax/test checks even run. Verified live across three languages (Python, JavaScript, Go) with a real call site outside the renamed method's span in each: all three now sweep correctly, zero stale references, valid syntax.

Closing this exposed a second, real gap in `defined_symbol_name` itself (`analyzer/locator.py`): a bare class-method snippet (JS/TS/Java/C#-style method-shorthand — valid only as a class member, never standalone) parsed alone to *zero* symbols, not because the content was ambiguous but because it has no class context to parse inside — silently defeating the new rename-sweep check for every language but Python. Fixed the same way analyzer/module_blocks.py's own techniques already work around missing context: when a direct parse finds nothing, retry once wrapped in a minimal synthetic `class __IEEWrapper__ { ... }` body and take the one real symbol found inside it (excluding the wrapper itself) — never reached for a language whose method syntax already parses standalone (Python, Go, Ruby, ...).

**Real crash bug found and fixed live this session**: `run_tests` had a hardcoded 120s subprocess timeout with **no exception handling at all** — a slow or genuinely hung test suite crashed the *entire* run with an uncaught `subprocess.TimeoutExpired`, a raw Python traceback instead of a controlled "failed" result. Every other external call in this project (Jev, vector retrieval, Joern) already degrades to a controlled failure on any error — this was the one place that didn't. Fixed to catch the timeout and return the same dict shape every caller already expects, with a new `"timed_out": True` field and a distinct `failure_class = "TEST_TIMEOUT"`. Verified live: the exact scenario that used to crash the whole CLI process now reports a clean `status: failed, failure_class: TEST_TIMEOUT` with an honest explanation instead.

**Real, measured speedup**: every syntax/import/test validation step copies the project into an isolated temp directory first (`shutil.copytree`) — every one of the ~8 call sites did this with a bare, unfiltered copy, which on a real repo containing this engine's own `.aider-venv` (a full separate Python virtualenv, ~735MB) or `.git` history copies gigabytes byte-for-byte before ever running a single check. `copy_project_for_validation` (`validation/tests.py`) replaces all of them with one shared helper that skips version control history, tool caches, and this engine's own operational artifacts (`.aider-venv`, `minio_local_data`, `iee_metadata`) — deliberately **not** `node_modules`/`venv`/`vendor`/`dist`/`build`, since a real project's own test suite might genuinely need any of those to run at all; excluding those would risk a false test failure, worse than the copy time saved. Measured live, isolated from pytest's own execution time: **84.96s → 9.06s** for the copy step alone against this engine's own real repo.

**Operational pitfall this same investigation surfaced** (not a code bug — a usage/configuration trap worth documenting): if a project's own `PROJECT DIRECTORY` is pointed at a directory that *also* contains this engine's own source and test suite (e.g., using this repo itself as scratch space for unrelated work), every single edit's `test_target` validation runs this engine's own, unrelated 222+-test suite — genuinely slow (pytest's own *execution* time for that many tests is ~150s, a cost no copy-time optimization can reduce) and reports failures/regressions that have nothing to do with the actual change being validated. **Always point `PROJECT DIRECTORY` at a project's own, separate directory.** Verified live, side by side: the identical request against a directory containing *only* the target file completed in ~33 seconds and succeeded; the same request with the project directory pointed at this engine's own repo took ~176 seconds and failed on an unrelated pre-existing test issue.

**`test_target` left at its default (`"."`) is auto-narrowed, mechanically, not just documented as a pitfall** (`_effective_test_target`/`_discover_relevant_test_target`, `api/run_pipeline.py`): real, observed case — editing `cores/newchatbot.py` with `PROJECT DIRECTORY` pointed at a large, cluttered directory made validation `pytest '.'` the *entire* thing, timing out after 120s on other, unrelated content that was never going to pass or fail because of this change either way. An explicit, narrower `test_target` the caller already chose is never touched; only the broad default gets a chance to narrow itself, and only ever down to something real, never invented: a discovered test file actually about the edited file (`test_<name>.<ext>` / `<name>_test.<ext>`, co-located or under a sibling/ancestor `tests/` directory), or failing that, the edited file's own containing directory — narrower than the whole project root, with `"no tests collected"` there still a legitimate pass, not a guess. A file that already lives at the project root has no narrower real target to fall back to and correctly stays unnarrowed. Verified live against a real, previously-timing-out case: `TEST test target '.' narrowed to 'cores'` → `pytest (cores)` completed in ~3s instead of hitting the 120s timeout.

---

## 10. Apply engine (`apply/edit_applier.py`)

Resolves every Delta IR operation's target against the *original* symbol index, then applies bottom-to-top by line (so an earlier edit's line-count change never shifts a later target's position).

- **Forced re-indentation**: generated content is re-indented to the target's original column regardless of what the model returned — trusting the model to preserve indentation produced a silently-wrong (but still Python-valid) de-indented method that Python treats as a different, un-nested definition.
- **DELETE never auto-resolves a delegate pair**; only REPLACE does — removing just the implementation half of a wrapper/impl pair would leave the wrapper calling a now-nonexistent method.
- **DELETE absorbs the contiguous blank-line run immediately following** the deleted symbol (not a hardcoded count) — self-adjusts to whatever blank-line convention the file already used.

---

## 11. Versioning, storage, and confirmation

- **`versioning/version_manager.py`** — a simple sequential checkpoint chain (`v1 → v2 → ...`) per project, tracked via `HEAD.json`. Each version's manifest records the change request, strategy, delta id, and validation status alongside the actual file snapshots.
- **`storage/minio_client.py`** — a `Storage` abstraction with two interchangeable backends: `LocalStorage` (disk, mirrors the exact bucket/key layout a real MinIO bucket would use) or `MinioStorage` (real MinIO, config-driven via `MINIO_ENDPOINT`). Switching backends is a zero-caller-change operation by design. This project's own application-level versioning is entirely separate from MinIO's *native* bucket versioning feature (which is never enabled by this project — confirmed directly against a live bucket: `status: None`).
- **`api/pending_confirmations.py`** — the human-in-the-loop gate for the web UI (`require_confirmation=True`; the CLI always auto-commits). A run pauses right before COMMIT; nothing is written to disk until explicitly accepted. An in-memory dict by default (lost on server restart, invisible to a second worker) — set `REDIS_URL` to back this with Redis instead, same "endpoint set → use it" switch MinIO already uses (§11 above), zero caller-visible change either way: `stash`/`resolve` are the only two operations, one write and one read-and-delete, with a TTL (`PENDING_CONFIRMATION_TTL_SECONDS`, default 24h) so an abandoned run doesn't linger forever. Verified live against a real local Redis instance: stash writes with the configured TTL, resolve reads-and-deletes exactly once, a second resolve on the same `run_id` correctly raises.

---

## 12. Surfaces: CLI and web app

- **`cli.py`** (`iee edit`/`iee create`/`iee find`) — always auto-commits (no confirmation gate); explicit opt-in flags for hybrid retrieval (`--hybrid`) and Joern cross-file rename checks (`--joern`).
- **`webapp.py`** (FastAPI, served alongside a built React frontend from `frontend/dist`) — streams step-by-step progress over SSE, supports the human-in-the-loop confirm/reject flow, defaults hybrid retrieval **on** (a real product surface, not a test suite that needs determinism) but Joern **off** (real added cost/dependency that shouldn't fire unasked).
- **Joern-backed cross-file rename safety** (opt-in, `use_joern`): before a mechanical rename, optionally checks Joern's real call graph for callers of the renamed symbol in *other files* — the one blind spot `rename_with_subword_fallback` has, since it only ever rewrites the file it's given. Never blocks the rename; surfaces `metadata["cross_file_impact_warning"]` for a human to act on. Verified live against a real 2-file project: correctly caught an external caller a same-file-only rename would have silently broken.

### Mode-agnostic file creation from a blank File field (`webapp.py`, `strategies/create_target_classifier.py`)

Edit mode with the **File** field left blank first tries auto-locate (hybrid retrieval over *existing* files, `locate_best_file`) — the right move for "fix the bug in the login handler." But that structurally can't help a request that describes something that doesn't exist yet at all ("make a new file called new.py", "write the RAG system, frontend and backend in separate folders") — auto-locate correctly finds nothing, and before this existed the request just refused outright, forcing a manual switch to Create mode (and typing the exact path by hand) for something the request itself already said plainly.

`classify_create_intent` closes that gap with one more real, structured-output model call — never a keyword/regex guess at intent, same "escalate is model-driven" contract every other classification in this project holds itself to: given the request and a bare listing of the project's existing files, it decides whether this genuinely describes creating brand-new file(s)/folder(s), and if so, proposes the real relative path(s) needed. Handles the multi-file/folder case the same way an escalated edit-mode `create_files` request already does — one real file per responsibility, never merged (`frontend/index.html` + `backend/server.py`, never one file mixing both) — dispatching to `run_create_files` (a new top-level entry point in `create_pipeline.py`, delegating to the existing `_run_create_files` machinery with no anchor file) when more than one path comes back, or the simpler single-file `run_create` when exactly one does. A genuinely ambiguous or edit-shaped request still refuses, same as before.

Real, live-verified case: "write the code for the RAG system and attach frontend and backend 2 different folder" — an early version of the prompt proposed bare, empty `frontend/`/`backend/` folders with nothing inside them (the same known model tendency toward folder-only output documented in §16's known gaps); strengthened to explicitly require real file paths inside each folder, it now proposes 5 real files (`frontend/index.html`, `frontend/app.js`, `frontend/styles.css`, `backend/server.py`, `backend/requirements.txt`) and generates each one individually.

---

## 13. Cost accounting (`benchmark/pricing.py`)

Every run's metadata includes real, measured `input_tokens`/`output_tokens`/`cached_tokens`/`estimated_cost_usd`/`latency_ms` — not estimates from guesswork. Cached-input pricing is tracked specifically because a bounded self-repair retry resends an identical system+context prefix, which real gateway behavior serves largely from cache (measured ~85% of prompt tokens cached on one real test) at a 90% discount — charging every repair attempt as if freshly computed would overstate real cost, sometimes substantially.

---

## 14. Prompt compression (`optimization/prompt_compression.py`)

Rule-based compression of this project's *own instructional text* only — never applied to code, docstrings, identifiers, or the user's own request. Drops filler (articles, hedging adverbs) but enforces a hard invariant: a fixed set of protected words (`not`, `no`, `never`, `only`, `except`, `must`, `always`, `none`, `unless`, `cannot`, and any `...n't` contraction) must appear the *same number of times* before and after compression — `compress()` raises rather than silently shipping a prompt that quietly means something different. A real regression this guards against: an earlier compression pass cut a sentence ("mentioning a folder name is never by itself a reason to treat the whole request as folder-only") that looked redundant but wasn't — cutting it reintroduced a real bug (an "empty folder created instead of real files" regression), caught by live re-testing, restored, and re-verified.

---

## 15. Testing philosophy

Every offline test in this project's own suite is network-free and deterministic by construction — any code path that could make a real LLM/embeddings/subprocess call is mocked at the exact boundary, and every optional real-cost feature (`use_hybrid_retrieval`, `use_joern`, Jev) defaults `False`/off specifically so the suite never depends on external state. Real, live verification (actual CLI runs against actual files, actual API calls) is treated as a *separate*, required step on top of the offline suite, not a substitute for it — the project's own history includes at least one case where the offline suite passed cleanly while the live behavior was subtly wrong (the folder-only `create_files` regression above), which is why "verified live" is called out explicitly throughout this document rather than assumed from test-suite success alone.

---

## 16. Known gaps and honest limitations

- **12 languages** (Elixir, Erlang, Clojure, Groovy, Zig, Nim, T-SQL, Common Lisp, Scheme, Racket, Prolog, Tcl) don't yet get full symbol-level precision — same fixable class of gap as the C/C++/Rust/Ruby fix, just not yet done.
- **`create_files` can produce folder-only output** for a request describing real functionality inside a named folder (e.g. "chatbot code in the backend folder" sometimes yields `files: ["backend/"]` with no actual code file) — confirmed via a direct A/B test to be a **pre-existing** model-behavior issue, not something introduced by any recent change; one targeted prompt-strengthening attempt did not fix it; likely needs either a mechanical repair-loop check or deeper prompt/example work.
- **Dependency/call-graph accuracy** (both the native graph and, to a lesser extent, Joern's CPG) is fundamentally limited by dynamic dispatch and duck typing in every language — treated as a real, honestly-labeled uncertainty (`RESOLVED`/`PARTIAL`/`DYNAMIC`/`UNRESOLVED`-style framing), never presented as ground truth.
- **Whole-file regeneration for edits is deliberately, permanently refused**, not merely deprioritized — by explicit policy decision, not an oversight. A request whose change genuinely can't be expressed as REPLACE/INSERT/DELETE on existing symbols (module-level statements, formatting spanning many symbols) now fails cleanly (`WHOLE_FILE_BLOCKED`) rather than falling back to a full rewrite that used to succeed. This is a real capability tradeoff, accepted in exchange for the guarantee that an edit run never touches code outside what the request identifies.
- **Project-directory scoping is the caller's responsibility** — pointing a project's directory at a much larger, unrelated directory tree (this engine's own repo included) means test validation runs whatever test suite happens to exist there, which may have nothing to do with the change being made.
