"""Offline tests for the phase-4 locator + context builder -- no LLM call.

Uses a synthetic 12-function file (small file to keep the test fast, but
enough unrelated symbols to prove the locator actually excludes them) since
the doc's own target scenario is a large multi-function file where sending
the whole thing wastes input tokens.
"""

import pytest

from incremental_editing.analyzer.locator import (
    AmbiguousSymbolError,
    defined_symbol_name,
    find_delete_candidates,
    find_multi_delete_targets,
    find_rename_target,
    find_symbol,
    index_symbols,
    is_whole_file_delete_target,
    locate_candidates,
    locate_candidates_by_body,
    looks_like_python,
    rename_with_subword_fallback,
    target_file_for_conversion,
)
from incremental_editing.context.context_builder import build_context

TWO_CLASS_SAME_METHOD_SOURCE = (
    "class Trig:\n"
    "    def tan(self, x):\n"
    "        import math\n"
    "        return math.tan(x)\n"
    "\n\n"
    "class Hyperbolic:\n"
    "    def tan(self, x):\n"
    "        import math\n"
    "        return math.tanh(x)\n"
)

MANY_FUNCTIONS_SOURCE = "\n\n".join(
    f"def helper_{i}(x):\n    return x + {i}" for i in range(20)
) + "\n\n" + (
    "def register_user(email, password):\n"
    "    \"\"\"Create a new user account.\"\"\"\n"
    "    return {\"email\": email, \"password\": password}\n"
)


def test_locator_finds_literal_name_mention():
    candidates = locate_candidates(MANY_FUNCTIONS_SOURCE, "add phone validation to register_user")
    assert [c.name for c in candidates] == ["register_user"]


def test_locator_ignores_stopword_only_docstring_overlap():
    """Real Bifrost log: a request sharing nothing with a class's docstring
    except the word "and" scored high enough to select that class as a
    "candidate" -- and since the class contained nearly every method in
    the file, "localized" context degenerated into ~92% of the whole
    file. A shared stopword must never count as relevance on its own."""
    source = (
        "class MLAlgoAgent:\n"
        '    """Provides markdown explanations, comparisons, and a roadmap for ML algorithms."""\n\n'
        "    def explain(self, query):\n"
        "        return query\n"
    )
    candidates = locate_candidates(source, "add the association types like aglomarative and divisive")
    assert candidates == []


def test_locator_ignores_lone_shared_content_word_between_unrelated_symbols():
    """Real Bifrost log: "add the cot inverse function" (a brand-new
    arccotangent function) pulled in an unrelated atan_inverse()'s entire
    body -- the two names share nothing but the word "inverse" (real
    content, not a stopword this time), which a math-heavy file full of
    similarly-named "X_inverse" symbols makes coincidental, not relevant.
    A single shared word with no literal name mention and nothing else
    corroborating it must not seat a symbol as a candidate, the same way a
    single shared stopword already can't."""
    source = (
        "def cot(x):\n    return 1 / tan(x)\n\n\n"
        "def atan_inverse(x):\n    import math\n    return math.atan(x)\n"
    )
    candidates = locate_candidates(source, "add the cot inverse function")
    assert [c.name for c in candidates] == ["cot"]


def test_target_file_for_conversion_swaps_only_the_extension():
    """target_extension comes straight from the model's own escalate
    response (see structured_edit.py's prompt / run_pipeline.py's
    _run_language_conversion) -- this function no longer looks up a
    language name in any hardcoded catalog, it just swaps the suffix."""
    assert target_file_for_conversion("SHA.go", "py") == "SHA.py"
    assert target_file_for_conversion("src/utils.js", "py") == "src/utils.py"
    assert target_file_for_conversion("Main.java", ".kt") == "Main.kt"  # leading dot tolerated


def test_find_delete_candidates_surfaces_each_duplicate_occurrence():
    """Real failure: a name defined 3 times used to collapse to one
    picker chip (deduped by name) -- picking it and confirming still
    failed with "defined N times", since the bare name alone can't say
    which occurrence, no matter how many times it's re-confirmed. Each
    occurrence must now be its own candidate, with its own real
    start_line, so the picker can offer (and confirm_symbol_line can
    resolve) an actual specific occurrence."""
    source = "def tan(x):\n    return 1\n\n\ndef tan(x):\n    return 2\n"
    candidates = find_delete_candidates(source, "remove the tan function")
    assert [c.name for c in candidates] == ["tan", "tan"]
    assert [c.start_line for c in candidates] == [1, 5]


def test_find_delete_candidates_ignores_short_target_words():
    """A target word under 3 characters (e.g. bare "x") would substring-
    match almost any real name in a typical file (max_value, box_area,
    example, ...) and turn the delete picker into a meaningless wall of
    unrelated chips instead of a real disambiguation -- must find nothing
    rather than everything."""
    source = (
        "def max_value(a, b):\n    return a\n\n\n"
        "def box_area(w, h):\n    return w * h\n\n\n"
        "def example(x):\n    return x\n"
    )
    assert find_delete_candidates(source, "remove the x function") == []


def test_find_delete_candidates_corrects_a_typo_via_fuzzy_match():
    """Real waste this closes: "remove substract" (typo for "subtract")
    found no substring match, fell through to a full LLM call, which
    escalated all the way to a ~10s, ~$0.03 full-file regeneration just
    to find a name a spell-check-grade match already could, for free.
    Fuzzy matching only kicks in when the exact substring search finds
    nothing at all -- never overrides a real substring match."""
    source = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n"
    candidates = find_delete_candidates(source, "remove substract")
    assert [c.name for c in candidates] == ["subtract"]


def test_find_delete_candidates_fuzzy_match_still_asks_when_ambiguous():
    """A typo close to more than one real name must still surface every
    close match as its own picker candidate, not silently guess one."""
    source = "def some(a, b):\n    return a\n\n\ndef same(a, b):\n    return b\n"
    candidates = find_delete_candidates(source, "remove sme")
    assert {c.name for c in candidates} == {"some", "same"}


def test_find_delete_candidates_fuzzy_match_finds_nothing_for_an_unrelated_word():
    """Must not fuzzy-match something that isn't actually close to any
    real name -- returning [] (not a wrong guess) so the caller can fall
    through to whatever escalation path fits (or genuinely find nothing)."""
    source = "def add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n"
    assert find_delete_candidates(source, "remove xyzabc") == []


MULTI_DELETE_SOURCE = (
    "def tan(x):\n    return 1\n\n\n"
    "def tanh(x):\n    return 2\n\n\n"
    "def atan_inverse(x):\n    return 3\n\n\n"
    "def cos(x):\n    return 4\n\n\n"
    "def cosin(x):\n    return 5\n\n\n"
    "def sec(x):\n    return 6\n\n\n"
    "def divide(a, b):\n    return a / b\n\n\n"
    "def multiply(a, b):\n    return a * b\n"
)


def test_find_multi_delete_targets_resolves_distinct_exact_names():
    """Real failure: "delete tan, sec and cos" against a file also
    containing tanh/atan_inverse/cosin matched all 10 via
    find_delete_candidates' broad substring search -- a "which one did
    you mean?" picker for a request that named three deliberate,
    distinct targets, not one ambiguous one. Each phrase must resolve to
    its own exact symbol, and only those three -- not the substring
    lookalikes."""
    result = find_multi_delete_targets(MULTI_DELETE_SOURCE, "delete tan, sec and cos")
    assert result is not None
    assert {s.name for s in result} == {"tan", "sec", "cos"}


def test_find_multi_delete_targets_returns_none_for_a_single_target():
    """A single-target request must fall back to the existing
    find_delete_candidates path unchanged -- this is purely additive."""
    assert find_multi_delete_targets(MULTI_DELETE_SOURCE, "remove the divide function") is None


def test_find_multi_delete_targets_bails_when_any_phrase_is_unresolved():
    """All-or-nothing: if any named target doesn't resolve to exactly one
    exact match (a typo, a name that doesn't exist), the whole multi-
    delete fast path must bail rather than partially resolving -- silently
    deleting 2 of 3 named things while confusingly failing on the third
    would be worse than falling back to the normal path for all of them."""
    assert find_multi_delete_targets(MULTI_DELETE_SOURCE, "delete tan and cosxyz") is None


def test_find_multi_delete_targets_bails_on_a_duplicated_target_name():
    """A named target that's itself defined more than once can't be
    safely auto-resolved here -- must bail to the existing disambiguation
    path instead of guessing which occurrence."""
    source = MULTI_DELETE_SOURCE + "\n\ndef divide(a, b):\n    return 0\n"
    assert find_multi_delete_targets(source, "delete divide and multiply") is None


def test_find_rename_target_matches_plain_rename_phrasing():
    source = "def oauthAuthorizationCodeFlow():\n    return 1\n"
    assert find_rename_target(source, "rename oauthAuthorizationCodeFlow to oauthAuthorizationFlow") == (
        "oauthAuthorizationCodeFlow",
        "oauthAuthorizationFlow",
    )


def test_find_rename_target_matches_the_real_reported_phrasing():
    """Real case: "replace name of function of X to Y" -- the redundant
    trailing "of" tolerated since that's how it was actually phrased,
    not a cleaned-up example."""
    source = "def oauthAuthorizationCodeFlow():\n    return 1\n"
    result = find_rename_target(
        source, "replace name of function of oauthAuthorizationCodeFlow to oauthAuthorizationFlow"
    )
    assert result == ("oauthAuthorizationCodeFlow", "oauthAuthorizationFlow")


def test_find_rename_target_matches_replace_with_and_replace_by():
    source = "def helper():\n    return 1\n"
    assert find_rename_target(source, "replace helper with worker") == ("helper", "worker")
    assert find_rename_target(source, "replace helper by worker") == ("helper", "worker")


def test_find_rename_target_returns_none_when_old_name_is_not_a_real_symbol():
    """Real safety requirement: this is a fully mechanical, zero-LLM
    path with no human review of *which* target was picked until after
    the fact -- must never fire on a guessed name that was never
    actually verified against the file's real symbols."""
    source = "def helper():\n    return 1\n"
    assert find_rename_target(source, "rename does_not_exist to worker") is None


def test_find_rename_target_returns_none_when_old_name_is_ambiguous():
    """A name defined more than once can't be safely auto-renamed here
    either -- same reasoning find_multi_delete_targets already applies
    to a duplicated delete target."""
    source = "def helper():\n    return 1\n\nclass C:\n    def helper(self):\n        return 2\n"
    assert find_rename_target(source, "rename helper to worker") is None


def test_find_rename_target_returns_none_for_unrelated_phrasing():
    """Must not misfire on a request that merely contains "replace" or
    "to" without actually being a rename -- falls through to the normal
    (LLM-driven) path unchanged, same degrading behavior is_delete_intent
    already has."""
    source = "def helper():\n    return 1\n"
    assert find_rename_target(source, "replace the hardcoded value with an env var lookup") is None


def test_rename_with_subword_fallback_handles_snake_case_unchanged():
    """The plain \\bold_name\\b path (unchanged) still wins when a real
    whole-word occurrence exists -- the sub-word fallback never
    triggers when it isn't needed."""
    source = "openai_client = build()\n"
    assert rename_with_subword_fallback(source, "openai", "anthropic") == "anthropic_client = build()\n"


def test_rename_with_subword_fallback_finds_a_camelcase_subword():
    """Real bug this closes: "use anthropic instead of openai" against
    a Go file using apiKeyOpenAI/buildOpenAIClient silently renamed
    nothing -- a plain \\bold_name\\b regex has no concept of a
    lowercase-to-uppercase transition as a word boundary, so it never
    found "OpenAI" packed inside those camelCase identifiers at all."""
    source = "var apiKeyOpenAI = \"x\"\n\nfunc buildOpenAIClient() string {\n\treturn apiKeyOpenAI\n}\n"
    result = rename_with_subword_fallback(source, "openai", "anthropic")
    assert "apiKeyAnthropic" in result
    assert "buildAnthropicClient" in result
    assert "OpenAI" not in result


def test_rename_with_subword_fallback_finds_a_pascalcase_subword():
    source = "class ChatOpenAIClient {}\n"
    result = rename_with_subword_fallback(source, "OpenAI", "Anthropic")
    assert result == "class ChatAnthropicClient {}\n"


def test_rename_with_subword_fallback_preserves_unrelated_identifiers():
    """Must not touch an identifier that doesn't actually contain
    old_name as one of its own sub-words -- "openaid" or "myopenaikey"
    are different words, not "openai" plus noise."""
    source = "openaidClient = 1\nunrelatedVar = 2\n"
    result = rename_with_subword_fallback(source, "openai", "anthropic")
    assert result == source  # "openaid" is not "openai" -- left untouched


def test_rename_with_subword_fallback_is_case_insensitive_but_preserves_style():
    """A lowercase old_name still matches (and correctly re-capitalizes)
    an uppercase sub-word -- the model naming the concept lowercase
    ("openai") must still match the real identifier's own casing."""
    source = "def buildOpenaiClient():\n    pass\n"
    result = rename_with_subword_fallback(source, "openai", "anthropic")
    assert "buildAnthropicClient" in result


def test_index_symbols_tracks_enclosing_class():
    """Two methods legitimately sharing a bare name in different classes
    is completely normal Python, not a bug -- index_symbols must be able
    to tell them apart by their enclosing class, or nothing downstream
    (disambiguation, find_symbol) has any way to distinguish them."""
    symbols = index_symbols(TWO_CLASS_SAME_METHOD_SOURCE)
    tans = [s for s in symbols if s.name == "tan"]
    assert len(tans) == 2
    assert {s.parent_class for s in tans} == {"Trig", "Hyperbolic"}
    classes = {s.name: s for s in symbols if s.symbol_type == "class"}
    assert classes["Trig"].parent_class is None  # a top-level class has no enclosing class itself
    assert classes["Hyperbolic"].parent_class is None


def test_locate_candidates_prefers_the_literally_named_class():
    """Real scenario: a 5000+ line file with the same method name (tan)
    repeated across many classes. A request naming the class disambiguates
    exactly like naming the method itself does -- must pick the method
    belonging to the class actually named, not whichever the file happens
    to define first."""
    candidates = locate_candidates(TWO_CLASS_SAME_METHOD_SOURCE, "fix Hyperbolic's tan method to validate input")
    tan_candidates = [c for c in candidates if c.name == "tan"]
    assert len(tan_candidates) == 1
    assert tan_candidates[0].parent_class == "Hyperbolic"


def test_find_symbol_resolves_duplicate_name_via_prefer_line():
    """Once locate_candidates has already picked a specific occurrence
    (by class-mention or semantic tiebreak), later lookups for the same
    bare name must resolve to that exact line instead of re-raising the
    same ambiguity a second time -- the whole point of resolving it once
    up front is that apply-time doesn't need to guess again."""
    symbols = index_symbols(TWO_CLASS_SAME_METHOD_SOURCE)
    hyperbolic_tan = next(s for s in symbols if s.name == "tan" and s.parent_class == "Hyperbolic")

    # No hint at all: still refuses, exactly as before -- this must not
    # have gotten silently permissive.
    import pytest

    with pytest.raises(AmbiguousSymbolError):
        find_symbol(symbols, "function", "tan")

    resolved = find_symbol(symbols, "function", "tan", prefer_line=hyperbolic_tan.start_line)
    assert resolved is hyperbolic_tan

    # A prefer_line that doesn't match any of the actual duplicates is not
    # a resolution -- still refuses rather than silently picking one.
    with pytest.raises(AmbiguousSymbolError):
        find_symbol(symbols, "function", "tan", prefer_line=9999)


DELEGATE_WRAPPER_SOURCE = (
    "class Calculator:\n"
    "    @staticmethod\n"
    "    def divide(a, b):\n"
    "        if b == 0:\n"
    "            return 0\n"
    "        return a / b\n"
    "\n\n"
    "def divide(a, b):\n"
    "    return Calculator.divide(a, b)\n"
)


def test_find_symbol_resolves_a_trivial_delegate_wrapper_to_the_real_impl():
    """Real scenario: "use OOP concepts" introduced a Calculator class
    while keeping every old flat function as a one-line backward-
    compatible wrapper (so existing `from calculator import divide`
    imports keep working) -- e.g. `def divide(a, b): return
    Calculator.divide(a, b)`. Editing "divide" should mean the real
    implementation, not its forwarding shim, without a human (or an LLM
    call) needing to disambiguate every single time."""
    symbols = index_symbols(DELEGATE_WRAPPER_SOURCE)
    resolved = find_symbol(symbols, "function", "divide", source=DELEGATE_WRAPPER_SOURCE)
    assert resolved.parent_class == "Calculator"


def test_find_symbol_delete_never_auto_resolves_a_delegate_pair():
    """Deleting just the implementation of a delegate pair would leave
    the wrapper calling a method that no longer exists -- a real
    regression the auto-resolution that's safe for REPLACE/INSERT would
    introduce here. DELETE must keep asking (allow_delegate_resolution=False)."""
    symbols = index_symbols(DELEGATE_WRAPPER_SOURCE)
    with pytest.raises(AmbiguousSymbolError):
        find_symbol(symbols, "function", "divide", source=DELEGATE_WRAPPER_SOURCE, allow_delegate_resolution=False)


def test_find_symbol_does_not_treat_two_real_implementations_as_a_delegate_pair():
    """Two genuinely independent class methods sharing a bare name (not
    one delegating to the other) must still raise -- the delegate check
    is a narrow, structural exception, not a general permissive fallback."""
    symbols = index_symbols(TWO_CLASS_SAME_METHOD_SOURCE)
    with pytest.raises(AmbiguousSymbolError):
        find_symbol(symbols, "function", "tan", source=TWO_CLASS_SAME_METHOD_SOURCE)


def test_locate_candidates_semantic_tiebreak_falls_back_safely_offline(monkeypatch):
    """The semantic tiebreak (real embeddings call) is best-effort by
    design -- if the embeddings gateway is offline, slow, or errors, this
    must degrade to a deterministic pick (the first tied candidate)
    rather than raising and failing the whole request over what was only
    ever meant to be a tie-breaking enhancement."""
    def _boom(texts):
        raise RuntimeError("embeddings gateway unreachable")

    monkeypatch.setattr("incremental_editing.retrieval.vector_retriever._embed", _boom)

    candidates = locate_candidates(TWO_CLASS_SAME_METHOD_SOURCE, "add input validation to tan")
    tan_candidates = [c for c in candidates if c.name == "tan"]
    assert len(tan_candidates) == 1  # a deterministic pick was still made, not a crash and not both


def test_locate_candidates_free_tiebreak_skips_the_paid_one_when_it_resolves_the_tie(monkeypatch):
    """The whole point of computing call relationships at all: a real
    signal this module already extracts (who calls whom, within this
    file) should resolve a same-name-across-classes tie for free before
    ever reaching for the embeddings-based tiebreak. Embeddings is
    monkeypatched to explode if called at all -- this only passes if the
    free signal alone won."""
    def _boom(texts):
        raise AssertionError("embeddings must not be reached -- the free tiebreak should have resolved this")

    monkeypatch.setattr("incremental_editing.retrieval.vector_retriever._embed", _boom)

    source = (
        "class Trig:\n"
        "    def tan(self, x):\n"
        "        return x\n"
        "\n\n"
        "class Hyperbolic:\n"
        "    def tan(self, x):\n"
        "        return x\n"
        "\n\n"
        "def uses_trig(x):\n"
        "    t = Trig()\n"
        "    return t.tan(x)\n"
    )
    candidates = locate_candidates(source, "add validation to tan")
    tan_candidates = [c for c in candidates if c.name == "tan"]
    assert len(tan_candidates) == 1
    assert tan_candidates[0].parent_class == "Trig"  # the one real code in this file actually calls


def test_locator_suppresses_uncorroborated_leading_generic_verb():
    """Real Bifrost log: request "add sin inverse function" pulled in add()'s
    entire body purely because "add" is both the request's own instruction
    verb and a real function name -- wasted input tokens on an irrelevant
    symbol. Leading "add" with nothing else corroborating it must be
    recognized as the verb, not a reference, and excluded entirely."""
    source = "def add(a, b):\n    return a + b\n\n\ndef sin(x):\n    import math\n    return math.sin(x)\n"
    candidates = locate_candidates(source, "add sin inverse function")
    assert [c.name for c in candidates] == ["sin"]


def test_locator_still_finds_generic_verb_name_when_not_leading():
    """The suppression must not overreach: "add" mid-sentence, referencing
    the real function, still has to match -- only a *leading* instruction
    verb with zero corroboration gets skipped."""
    source = "def add(a, b):\n    return a + b\n\n\ndef sin(x):\n    import math\n    return math.sin(x)\n"
    candidates = locate_candidates(source, "fix a bug in add so it validates argument types")
    assert [c.name for c in candidates] == ["add"]


def test_locator_returns_empty_when_nothing_matches():
    candidates = locate_candidates(MANY_FUNCTIONS_SOURCE, "completely unrelated request about nothing here")
    assert candidates == []


def test_context_builder_shrinks_context_for_targeted_request():
    ctx = build_context(MANY_FUNCTIONS_SOURCE, "add phone validation to register_user")
    assert ctx["used_localization"] is True
    assert ctx["candidate_symbols"] == ["register_user"]
    assert ctx["context_lines"] < ctx["total_lines"]
    assert "register_user" in ctx["context"]
    # unrelated symbols' bodies are excluded (that's the token saving)...
    assert "return x + 0" not in ctx["context"]
    # ...but their names are still listed, so the model can pick a real
    # INSERT anchor instead of hallucinating a plausible-sounding one.
    assert "helper_0" in ctx["context"]
    # a candidate shown in full must not ALSO appear in the name list --
    # that was pure duplication for no benefit.
    assert ctx["context"].count("register_user") == 1


def test_context_builder_dedupes_ambiguous_duplicate_names_in_name_list():
    """A name defined twice (the exact ambiguous-symbol case find_symbol
    already refuses to edit) must appear once in the name list, not twice
    -- listing it twice costs tokens without telling the model anything
    it didn't already know."""
    source = (
        "def add(a, b):\n    return a + b\n\n\n"
        "def tan(x):\n    import math\n    return math.tan(x)\n\n\n"
        "def tan(x):\n    return 0\n"  # duplicate, same name
    )
    ctx = build_context(source, "add a cos function")
    names_line = next(line for line in ctx["context"].splitlines() if line.startswith("# other symbols"))
    assert names_line.count("tan") == 1


def test_context_builder_includes_symbol_index_for_anchor_visibility():
    """Regression test: a real run asked to insert a trig function anchored
    after 'sin', and the model invented that name (only 'cosin' actually
    existed) because the trimmed context only showed it add()'s body, not
    the rest of the file's symbol names. The context must always carry a
    full name index so valid anchors are visible even when bodies aren't."""
    source = (
        "def add(a, b):\n    return a + b\n\n\ndef cosin(x):\n    import math\n    return math.cos(x)\n"
    )
    # "please add ..." (not "add ..." leading) so this exercises anchor-name
    # visibility, not the leading-generic-verb suppression covered below.
    ctx = build_context(source, "please add a modulo function")
    assert ctx["candidate_symbols"] == ["add"]  # only add()'s body is in context...
    assert "cosin" in ctx["context"]  # ...but cosin is still visible as a real anchor option


def test_context_builder_uses_name_list_not_full_source_when_unmatched():
    """When nothing localizes (typically: 'add a new thing', nothing existing
    to name), context used to be the *entire raw file* -- correct but
    wasteful. It's now imports + a compact name list (no bodies): still
    enough to avoid hallucinating a name or duplicating an existing one,
    without paying for bodies the request doesn't need. This fixture has 21
    symbols -- over _MAX_SYMBOLS_FOR_PARAM_DETAIL, so entries fall back to
    bare names here; see the dedicated cap tests below for why."""
    ctx = build_context(MANY_FUNCTIONS_SOURCE, "completely unrelated request about nothing here")
    assert ctx["used_localization"] is False
    assert ctx["candidate_symbols"] == []
    assert "return x + 0" not in ctx["context"]  # no bodies leak through
    assert "helper_0" in ctx["context"]  # but every real name is still visible
    assert "register_user" in ctx["context"]
    assert ctx["context_lines"] < ctx["total_lines"]


def test_context_builder_shows_parameters_on_a_small_file():
    """Real motivation: a request naming a parameter or a framework
    decorator ("the route handler for /login") has nothing to match
    against when the name list shows bare names only. On a file small
    enough to keep the cost trivial, each entry carries its real
    parameters (and decorators, when present) -- still no bodies."""
    source = (
        "@app.route('/login')\n"
        "def handler(request, timeout=30):\n"
        "    return request\n\n\n"
        "def helper():\n"
        "    return 1\n"
    )
    # Deliberately shares zero vocabulary with the source (not even
    # "request" -- coincidentally also a real parameter name here, which
    # would otherwise make locate_candidates_by_body's body-text search
    # correctly match handler() for a genuinely unrelated reason).
    ctx = build_context(source, "completely unconnected topic xyz123")
    assert "handler(request, timeout)" in ctx["context"]
    assert "@app.route('/login')" in ctx["context"]
    assert "helper()" in ctx["context"]


def test_context_builder_caps_parameter_detail_on_a_large_file():
    """The other half of the tradeoff, measured for real: showing every
    parameter on every entry of a real 39-symbol file added 90 tokens
    (127 -> 217, +71%) to the name line alone -- a cost that compounds
    with symbol count and starts fighting the token-cost goal exactly
    where the compact line matters most. Past the cap, entries fall back
    to bare names -- confirmed here via a 25-symbol fixture (over the
    20-symbol cap) rather than relying on the unrelated fixture above
    incidentally being large enough."""
    source = "\n\n".join(f"def helper_{i}(x, y, z):\n    return x + y + z" for i in range(25))
    ctx = build_context(source, "completely unrelated request naming nothing real")
    assert "helper_0(x, y, z)" not in ctx["context"]
    assert "helper_0" in ctx["context"]  # bare name still present, just not the signature


def test_build_context_hybrid_is_off_by_default():
    """The offline default matters for real: vector retrieval is a real
    embeddings-API call, and every test in this file relies on
    build_context staying network-free unless use_hybrid is explicitly
    requested."""
    source = "def alpha():\n    return 1\n"
    ctx = build_context(source, "completely unrelated request naming nothing real")
    assert ctx["used_localization"] is False  # would be True if hybrid somehow ran unrequested


def test_build_context_hybrid_finds_a_rare_body_word_via_bm25(monkeypatch):
    """Real motivation: a request sharing only a rare, specific word
    with a symbol's *body* -- not its name or docstring -- is exactly
    what BM25 exists to catch (PHOENIX doc section 4), on top of what
    plain name/docstring matching (locate_candidates) already covers.
    VectorRetriever mocked to return nothing so this isolates BM25's own
    contribution and stays network-free."""
    from incremental_editing.context import context_builder

    monkeypatch.setattr(context_builder, "VectorRetriever", lambda symbols: type("V", (), {"rank": lambda self, q, top_k: []})())

    source = (
        "def alpha():\n"
        "    return 1\n\n\n"
        "def beta():\n"
        "    zephyranthes_marker = True\n"
        "    return zephyranthes_marker\n"
    )
    request = "check the zephyranthes flag"

    # Confirms this genuinely isn't findable via plain name/docstring
    # matching alone -- proving hybrid adds real recall, not redundancy.
    assert locate_candidates(source, request) == []

    ctx = build_context(source, request, use_hybrid=True)
    assert ctx["candidate_symbols"] == ["beta"]


def test_build_context_hybrid_uses_the_vector_signal_too(monkeypatch):
    """Isolates the vector-retrieval contribution specifically -- mocked
    (a real embeddings call would make this test network-dependent and
    non-deterministic) to return a canned ranking, confirming its result
    actually participates in the fusion rather than being silently
    ignored."""
    from incremental_editing.context import context_builder

    source = "def alpha():\n    return 1\n\n\ndef gamma():\n    return 3\n"

    def _fake_vector_rank(self, query, top_k):
        gamma = next(s for s in self.symbols if s.name == "gamma")
        return [(gamma, 0.9)]

    monkeypatch.setattr(
        context_builder, "VectorRetriever", lambda symbols: type("V", (), {"symbols": symbols, "rank": _fake_vector_rank})()
    )

    ctx = build_context(source, "completely unrelated wording naming nothing real", use_hybrid=True)
    assert "gamma" in ctx["candidate_symbols"]


def test_build_context_hybrid_still_falls_back_when_no_signal_matches_anything(monkeypatch):
    """Every signal (name/docstring, BM25, mocked-empty vector) finding
    nothing must still degrade cleanly to the existing compact fallback
    -- not crash, not silently invent a candidate."""
    from incremental_editing.context import context_builder

    monkeypatch.setattr(context_builder, "VectorRetriever", lambda symbols: type("V", (), {"rank": lambda self, q, top_k: []})())

    source = "def alpha():\n    return 1\n"
    ctx = build_context(source, "totally unconnected xyzabc123 topic", use_hybrid=True)
    assert ctx["candidate_symbols"] == []
    assert ctx["used_localization"] is False


def test_looks_like_python_detects_real_python_in_a_non_py_extension():
    """Real case: a file literally named "new.db" containing real Python
    (a sqlite3 script), extension-based detect_language() returns None
    for it. Must still be recognized as Python so it gets real AST
    indexing/localization instead of falling through to raw-text
    handling that sends the whole file regardless of size."""
    source = (
        "import sqlite3\n\n"
        "def create_database():\n"
        "    conn = sqlite3.connect('new.db')\n"
        "    return conn\n"
    )
    assert looks_like_python(source) is True


def test_looks_like_python_rejects_plain_csv():
    """A CSV's rows are coincidentally valid Python (each line parses as
    a harmless, unused tuple expression via bare commas), so a plain
    ast.parse() success is not enough -- must require a real def/class/
    import to count as Python. Without this, every CSV misdetects as
    Python."""
    source = "id,name,salary\n1,Alice,50000\n2,Bob,60000\n"
    assert looks_like_python(source) is False


def test_looks_like_python_rejects_non_python_syntax():
    source = "SELECT * FROM employees WHERE salary > 50000;\n"
    assert looks_like_python(source) is False


def test_locate_candidates_by_body_finds_a_keyword_only_in_a_functions_body():
    """Real waste this closes: "remove the minio" referenced a
    "minio": {...} dict literal *inside* health()'s body, not its name
    or docstring -- name-only matching (locate_candidates) found
    nothing, and the request used to fall all the way through to a
    full-file regeneration just so the model could see where "minio"
    actually was."""
    source = (
        "def index():\n"
        "    return {'service': 'chatbot'}\n\n\n"
        "def health():\n"
        "    return {'status': 'ok', 'minio': {'endpoint': 'minio:9000'}}\n"
    )
    candidates = locate_candidates_by_body(source, "remove the minio")
    assert [c.name for c in candidates] == ["health"]


def test_locate_candidates_by_body_ranks_more_matching_words_first():
    source = (
        "def a():\n"
        "    return connect_minio_bucket()\n\n\n"
        "def b():\n"
        "    return 'minio' in some_config\n"
    )
    candidates = locate_candidates_by_body(source, "remove the minio bucket config")
    assert candidates[0].name == "a"  # matches both "minio" and "bucket"


def test_locate_candidates_by_body_finds_nothing_for_an_unrelated_request():
    source = "def add(a, b):\n    return a + b\n\n\ndef divide(a, b):\n    return a / b\n"
    assert locate_candidates_by_body(source, "completely unconnected topic xyz123") == []


def test_locate_candidates_by_body_prefers_the_nested_method_over_its_enclosing_class(tmp_path):
    """Real bug this closes: a class's own line span structurally
    CONTAINS every one of its methods, so a word matching only inside
    one nested method (e.g. "temperature" only inside a constructor)
    scored an independent "hit" for the ENCLOSING CLASS too -- entirely
    explained by that one method already matching, never a separate
    signal. Selecting both dragged every OTHER unrelated method in the
    class into context (and into whatever a REPLACE went on to restate).
    Real observed case: "make the default temperature 0.5" against a
    TypeScript class whose constructor sets it caused the whole class --
    three unrelated methods included -- to be selected instead of just
    the constructor. Verified live end-to-end this same session: fixing
    this took the resulting edit from a 3009-token whole-class REPLACE
    down to a 993-token constructor-only one, touching only the one
    line that needed to change."""
    source = (
        "class RagModel:\n"
        "    def __init__(self, temperature=0.3):\n"
        "        self.temperature = temperature\n\n"
        "    def ingest_documents(self, documents):\n"
        "        return len(documents)\n\n"
        "    def ingest_text(self, text):\n"
        "        return text\n"
    )
    candidates = locate_candidates_by_body(source, "change the default temperature to 0.5")
    assert [c.name for c in candidates] == ["__init__"]  # never "RagModel" alongside it


def test_build_context_merges_a_second_target_found_only_by_body_content():
    """Real bug this closes: a compound request naming one target
    literally ("rename ask_all") while a second clause ("change the
    temperature to 0.5") only shares a word with a DIFFERENT symbol's
    body, never its name/docstring. Gating locate_candidates_by_body
    behind "no name-based candidate found at all" meant that second
    target's content was never shown to the model once the first was
    found by name -- not escalated, not refused, silently never
    addressed at all. Must merge both into context."""
    source = (
        "class Bot:\n"
        "    def ask_model(self, x):\n"
        "        return self._call(x, temperature=0.7)\n\n"
        "    def ask_all(self, x):\n"
        "        return self.ask_model(x)\n"
    )
    ctx = build_context(source, "change the temperature to 0.5\nalso rename ask_all to ask_everyone")
    assert set(ctx["candidate_symbols"]) == {"ask_model", "ask_all"}


def test_build_context_never_adds_a_second_occurrence_of_an_already_selected_name():
    """Real regression this guards: merging in body-content matches must
    never add a SECOND entry sharing a name locate_candidates already
    resolved (e.g. the disambiguated occurrence of a duplicated method
    name across two classes) -- candidate_lines assumes at most one
    occurrence per name, and a second, different same-named symbol
    silently overwrote which line the already-correct resolution
    pointed at, corrupting which occurrence a REPLACE actually applies
    to."""
    ctx = build_context(TWO_CLASS_SAME_METHOD_SOURCE, "fix Hyperbolic's tan method to validate input")
    # "Hyperbolic" the class is its own separate, legitimately distinct
    # name-based candidate (the request also literally names the class) --
    # what must never happen is a SECOND "tan" entry from Trig's own
    # occurrence sneaking in via the body-content merge.
    assert ctx["candidate_symbols"].count("tan") == 1
    # Resolved to Hyperbolic's own occurrence (line 8), not Trig's (line 2).
    assert ctx["candidate_lines"]["tan"] == 8


def test_defined_symbol_name_resolves_a_bare_class_method_snippet_via_wrapper():
    """A bare method-shorthand snippet (JS/TS/Java/C#-style) is only
    valid syntax as a class member, never standalone -- direct parsing
    finds zero symbols, not because the content is ambiguous but because
    it has no class context to parse inside. Real bug this caught: a
    REPLACE that renamed a JS class method (askAll -> askEveryone) had
    its new name silently undetectable this way, which meant the
    "sweep every stale reference elsewhere in the file" rename-safety
    check (api/run_pipeline.py) never even ran for JS/TS/Java/C#."""
    content = "askEveryone(input) {\n    return this.askModel(input);\n  }"
    assert defined_symbol_name(content, "javascript") == "askEveryone"


def test_defined_symbol_name_still_returns_none_for_genuinely_ambiguous_content():
    content = "askEveryone(input) {\n    return 1;\n  }\n  askOther(x) {\n    return 2;\n  }"
    assert defined_symbol_name(content, "javascript") is None


def test_is_whole_file_delete_target_matches_bare_file_reference():
    assert is_whole_file_delete_target("calculator.py", "delete this file") is True
    assert is_whole_file_delete_target("calculator.py", "delete the file") is True


def test_is_whole_file_delete_target_matches_the_files_own_real_name():
    assert is_whole_file_delete_target("new.db", "delete new.db") is True
    assert is_whole_file_delete_target("sample_project/calculator.py", "remove calculator.py") is True


def test_is_whole_file_delete_target_rejects_a_request_naming_something_else():
    """Must not misfire as a whole-file delete just because a delete verb
    is present -- "delete the add function" names a real symbol, not the
    file, so it must fall through to the symbol-delete path instead."""
    assert is_whole_file_delete_target("calculator.py", "delete the add function") is False
    assert is_whole_file_delete_target("calculator.py", "remove the subtract method") is False
