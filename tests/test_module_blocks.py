"""Offline tests for the LLM-free module-level-block locator
(analyzer/module_blocks.py) -- content outside every function/class's own
span, in a language that otherwise has a real symbol concept. No LLM call
anywhere.
"""

from incremental_editing.analyzer.module_blocks import index_module_level_blocks, locate_module_level_block

# Mirrors the real motivating case: a module-level config list ("gpt"
# text) that sits between an import block and a class, with a trailing
# __main__ guard after the class -- three independently-targetable
# module-level regions around two real symbols.
CHATBOT_SOURCE = (
    "import os\n"
    "import sys\n"
    "\n"
    "MODEL_SLOTS = [\n"
    '    {"name": "Reasoner", "default": "gpt-4o-mini"},\n'
    '    {"name": "Coder", "default": "gpt-4.1-mini"},\n'
    "]\n"
    "\n"
    "\n"
    "class Bot:\n"
    "    def run(self) -> None:\n"
    "        pass\n"
    "\n"
    "\n"
    "def main() -> None:\n"
    "    Bot().run()\n"
    "\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    main()\n"
)

# No real function/class symbol anywhere -- the module-level "gaps" would
# be the entire file. locate_module_level_block itself has no opinion on
# this (that guard lives in run_pipeline.py, see test_engine.py's own
# refusal tests) -- these tests only check the locator's own blocks/scoring.
NO_SYMBOL_SOURCE = "import json\n\nVALUE = json.dumps({'a': 1})\n"


def test_index_module_level_blocks_finds_each_gap_as_its_own_block():
    blocks = index_module_level_blocks(CHATBOT_SOURCE, "python")
    names = [b.name for b in blocks]
    assert any("import os" in n for n in names)
    assert any("MODEL_SLOTS" in n for n in names)
    assert any("__main__" in n for n in names)
    # The gap content inside the class/function spans themselves must
    # never show up as a module-level block.
    assert not any("Bot()" in n for n in names)


def test_locate_module_level_block_finds_the_matching_config_list():
    block = locate_module_level_block(CHATBOT_SOURCE, "remove gpt", "python")
    assert block is not None
    assert "MODEL_SLOTS" in block.name


def test_locate_module_level_block_returns_none_when_no_block_matches():
    block = locate_module_level_block(CHATBOT_SOURCE, "add rate limiting to the websocket handler", "python")
    assert block is None


def test_locate_module_level_block_returns_none_on_a_tied_score():
    # Both module-level statements mention "value" equally -- ambiguous,
    # must refuse rather than guess which one was meant.
    source = "VALUE_A = 'value'\n\nVALUE_B = 'value'\n"
    block = locate_module_level_block(source, "change value", "python")
    assert block is None


def test_index_module_level_blocks_empty_for_a_file_with_no_gaps():
    source = "def f() -> None:\n    pass\n"
    assert index_module_level_blocks(source, "python") == []
