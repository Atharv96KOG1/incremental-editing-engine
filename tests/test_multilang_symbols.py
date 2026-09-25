"""Offline tests for language_has_symbol_concept -- real Tree-sitter
grammar introspection (local, no network), not a hardcoded per-language
list.
"""

from incremental_editing.retrieval.multilang_symbols import extract_symbols, language_has_symbol_concept


def test_language_has_symbol_concept_true_for_real_programming_languages():
    for language in ("python", "javascript", "java", "go"):
        assert language_has_symbol_concept(language) is True


def test_language_has_symbol_concept_false_for_prose_and_data_formats():
    """Real motivation: STRUCTURED_EDIT's classification call can only
    ever answer "escalate" for these -- every single time -- since
    their grammars have no function/class concept at all, regardless of
    what any request asks for or how much content the file has."""
    for language in ("markdown", "json", "yaml", "toml", "csv", "html", "css", "ini", "dotenv"):
        assert language_has_symbol_concept(language) is False, language


def test_language_has_symbol_concept_is_cached(monkeypatch):
    """Real cost this avoids: grammar introspection (enumerating every
    node kind) on every single request would be wasted repeated work --
    the grammar itself never changes mid-process."""
    from incremental_editing.retrieval import multilang_symbols

    calls = []
    real_get_parser = multilang_symbols.get_parser

    def _counting_get_parser(language):
        calls.append(language)
        return real_get_parser(language)

    monkeypatch.setattr(multilang_symbols, "_symbol_concept_cache", {})
    monkeypatch.setattr(multilang_symbols, "get_parser", _counting_get_parser)

    language_has_symbol_concept("markdown")
    language_has_symbol_concept("markdown")
    assert calls == ["markdown"]  # second call served from cache, no second parser lookup


def test_extract_symbols_finds_a_plain_c_function():
    """Real bug this closes: C's function_definition has no "name" field
    at all -- the identifier is nested inside its own function_declarator
    child -- so child_by_field_name("name") always returned None and
    every C function was silently dropped, full stop."""
    source = "int add(int a, int b) {\n    return a + b;\n}\n"
    symbols = extract_symbols(source, "c")
    assert [(s.name, s.symbol_type) for s in symbols] == [("add", "function")]


def test_extract_symbols_finds_cpp_free_functions_and_class_methods():
    """Real bug this closes: C++ inherits C's function_definition
    nesting issue, so a free function AND every method inside a class
    were both silently dropped -- only the class itself (class_specifier,
    which does have a real "name" field) was ever found."""
    source = (
        "int add(int a, int b) {\n    return a + b;\n}\n\n"
        "class Calculator {\npublic:\n    int add(int a, int b) { return a + b; }\n};\n"
    )
    symbols = extract_symbols(source, "cpp")
    assert [(s.name, s.symbol_type) for s in symbols] == [
        ("add", "function"),
        ("Calculator", "class"),
        ("add", "function"),
    ]


def test_extract_symbols_finds_rust_functions_structs_and_impl_methods():
    """Real bug this closes: Rust's function_item/struct_item node types
    contain none of "declaration"/"definition"/"specifier" -- the only
    substrings _classify used to require -- so every Rust symbol was
    silently dropped, full stop."""
    source = (
        "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n"
        "struct Calc;\nimpl Calc {\n    fn add(&self, a: i32, b: i32) -> i32 { a + b }\n}\n"
    )
    symbols = extract_symbols(source, "rust")
    assert [(s.name, s.symbol_type) for s in symbols] == [
        ("add", "function"),
        ("Calc", "class"),
        ("add", "function"),
    ]


def test_extract_symbols_finds_ruby_methods_and_classes():
    """Real bug this closes: Ruby's method/class definitions are bare
    "method"/"class" node types with no declaration/definition/specifier
    suffix at all, so they never passed the old gate -- every Ruby
    symbol was silently dropped, full stop."""
    source = "class Calc\n  def add(a, b)\n    a + b\n  end\nend\n"
    symbols = extract_symbols(source, "ruby")
    assert [(s.name, s.symbol_type) for s in symbols] == [("Calc", "class"), ("add", "function")]


def test_extract_symbols_bare_type_keyword_widening_does_not_misfire_elsewhere():
    """Widening _classify to accept bare "function"/"method"/"class" node
    types (needed for Ruby) must not suddenly start matching the literal
    keyword *token* itself -- several grammars (Python, JS, C++) expose
    that keyword as its own leaf node sharing the exact same bare type
    name a real definition container also uses. Already covered
    indirectly by every other passing test in this file/test_engine.py,
    asserted directly here: a plain Python class must still yield exactly
    one symbol, not two (one real, one from the "class" keyword token)."""
    source = "class Foo:\n    pass\n"
    symbols = extract_symbols(source, "python")
    assert [(s.name, s.symbol_type) for s in symbols] == [("Foo", "class")]
