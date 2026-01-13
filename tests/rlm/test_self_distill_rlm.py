import os
import tempfile
from unittest.mock import patch

import pytest

from self_distill.rlm.self_distill_rlm import (
    TOOL_CREATION_PROMPT,
    SelfDistillRLM,
    get_tool_registry_setup_code,
)


class TestToolCreationPrompt:
    def test_prompt_is_non_empty_string(self):
        assert isinstance(TOOL_CREATION_PROMPT, str)
        assert len(TOOL_CREATION_PROMPT) > 0

    def test_prompt_contains_repl_instructions(self):
        assert "```repl" in TOOL_CREATION_PROMPT
        assert "NOT ```python" in TOOL_CREATION_PROMPT

    def test_prompt_contains_architecture_sections(self):
        assert "pre_completion/" in TOOL_CREATION_PROMPT
        assert "replacements/" in TOOL_CREATION_PROMPT
        assert "utilities/" in TOOL_CREATION_PROMPT

    def test_prompt_contains_hook_protocol(self):
        # Updated: now uses check(text) -> bool
        assert "check(text" in TOOL_CREATION_PROMPT
        assert "-> bool" in TOOL_CREATION_PROMPT
        assert "True" in TOOL_CREATION_PROMPT
        assert "False" in TOOL_CREATION_PROMPT

    def test_prompt_contains_repl_functions(self):
        assert "list_all_tools()" in TOOL_CREATION_PROMPT
        assert "create_tool(" in TOOL_CREATION_PROMPT
        assert "write_to_tool(" in TOOL_CREATION_PROMPT
        assert "finish_tool()" in TOOL_CREATION_PROMPT
        assert "run_tool(" in TOOL_CREATION_PROMPT

    def test_prompt_contains_final_answer_format(self):
        assert "FINAL(" in TOOL_CREATION_PROMPT
        assert "FINAL_VAR(" in TOOL_CREATION_PROMPT


class TestGetToolRegistrySetupCode:
    def test_returns_string(self):
        code = get_tool_registry_setup_code("/tmp/test_tools")
        assert isinstance(code, str)
        assert len(code) > 0

    def test_includes_tools_dir_path(self):
        code = get_tool_registry_setup_code("/my/custom/path")
        assert 'TOOLS_DIR = "/my/custom/path"' in code

    def test_includes_categories(self):
        code = get_tool_registry_setup_code("/tmp/tools")
        assert 'CATEGORIES = ["pre_completion", "replacements", "utilities"]' in code

    def test_is_valid_python_syntax(self):
        code = get_tool_registry_setup_code("/tmp/tools")
        compile(code, "<string>", "exec")

    def test_defines_list_all_tools(self):
        code = get_tool_registry_setup_code("/tmp/tools")
        assert "def list_all_tools():" in code

    def test_defines_tool_management_functions(self):
        code = get_tool_registry_setup_code("/tmp/tools")
        assert "def load_tool(category, name):" in code
        assert "def create_tool(category, name, description=" in code
        assert "def write_to_tool(line):" in code
        assert "def finish_tool():" in code
        assert "def run_tool(category, name, input_text):" in code

    def test_defines_typo_aliases(self):
        code = get_tool_registry_setup_code("/tmp/tools")
        assert "write_to_text = write_to_tool" in code
        assert "add_line = write_to_tool" in code
        assert "write_line = write_to_tool" in code


class TestToolRegistryExecution:
    """Test the generated tool registry code by executing it."""

    @pytest.fixture
    def tools_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def registry_env(self, tools_dir):
        """Execute the registry code and return the namespace."""
        code = get_tool_registry_setup_code(tools_dir)
        namespace = {}
        exec(code, namespace)
        return namespace

    def test_creates_category_directories(self, registry_env, tools_dir):
        assert os.path.isdir(os.path.join(tools_dir, "pre_completion"))
        assert os.path.isdir(os.path.join(tools_dir, "replacements"))
        assert os.path.isdir(os.path.join(tools_dir, "utilities"))

    def test_list_all_tools_empty(self, registry_env):
        result = registry_env["list_all_tools"]()
        assert result == {
            "pre_completion": [],
            "replacements": [],
            "utilities": [],
        }

    def test_load_tool_nonexistent(self, registry_env):
        result = registry_env["load_tool"]("utilities", "nonexistent")
        assert result is None

    def test_create_tool_invalid_category(self, registry_env, capsys):
        registry_env["create_tool"]("invalid_category", "test_tool")
        captured = capsys.readouterr()
        assert "Error: Invalid category" in captured.out

    def test_write_to_tool_without_create(self, registry_env, capsys):
        registry_env["write_to_tool"]("print('hello')")
        captured = capsys.readouterr()
        assert "Error: No tool being created" in captured.out

    def test_finish_tool_without_create(self, registry_env, capsys):
        result = registry_env["finish_tool"]()
        captured = capsys.readouterr()
        assert result is None
        assert "Error: No tool being created" in captured.out

    def test_run_tool_nonexistent(self, registry_env):
        result = registry_env["run_tool"]("utilities", "nonexistent", "test")
        assert "error" in result
        assert "not found" in result["error"]

    def test_create_and_finish_utility_tool(self, registry_env, tools_dir):
        registry_env["create_tool"]("utilities", "my_util", "A test utility")
        registry_env["write_to_tool"]("def run(text):")
        registry_env["write_to_tool"]("    return text.upper()")
        name = registry_env["finish_tool"]()

        assert name == "my_util"
        tool_path = os.path.join(tools_dir, "utilities", "my_util.py")
        assert os.path.exists(tool_path)

        with open(tool_path) as f:
            content = f.read()
        assert "# A test utility" in content
        assert "def run(text):" in content

    def test_run_created_utility_tool(self, registry_env):
        registry_env["create_tool"]("utilities", "uppercaser", "Uppercase text")
        registry_env["write_to_tool"]("def run(text):")
        registry_env["write_to_tool"]("    return text.upper()")
        registry_env["finish_tool"]()

        result = registry_env["run_tool"]("utilities", "uppercaser", "hello")
        assert result == "HELLO"

    def test_create_pre_completion_hook(self, registry_env, tools_dir):
        registry_env["create_tool"]("pre_completion", "test_hook", "A test hook")
        registry_env["write_to_tool"]("def check(text: str) -> bool:")
        registry_env["write_to_tool"]("    return False")
        registry_env["finish_tool"]()

        # Verify file exists
        hook_path = os.path.join(tools_dir, "pre_completion", "test_hook.py")
        assert os.path.exists(hook_path)

    def test_create_replacement_tool(self, registry_env, tools_dir):
        registry_env["create_tool"](
            "replacements", "test_replacement", "A test replacement"
        )
        registry_env["write_to_tool"]("def run(text):")
        registry_env["write_to_tool"]("    return 'replaced: ' + text")
        registry_env["finish_tool"]()

        # Verify file exists
        repl_path = os.path.join(tools_dir, "replacements", "test_replacement.py")
        assert os.path.exists(repl_path)

    def test_run_tool_without_run_function(self, registry_env):
        registry_env["create_tool"]("utilities", "no_run", "Missing run")
        registry_env["write_to_tool"]("def other_func(): pass")
        registry_env["finish_tool"]()

        result = registry_env["run_tool"]("utilities", "no_run", "test")
        assert "error" in result
        assert "no run() function" in result["error"]

    def test_run_tool_with_exception(self, registry_env):
        registry_env["create_tool"]("utilities", "buggy", "Buggy tool")
        registry_env["write_to_tool"]("def run(text):")
        registry_env["write_to_tool"]("    raise ValueError('intentional error')")
        registry_env["finish_tool"]()

        result = registry_env["run_tool"]("utilities", "buggy", "test")
        assert "error" in result
        assert "intentional error" in result["error"]

    def test_typo_aliases_work(self, registry_env, tools_dir):
        registry_env["create_tool"]("utilities", "alias_test", "Test aliases")
        registry_env["write_to_text"]("def run(text):")
        registry_env["add_line"]("    return 'line1'")
        registry_env["write_line"]("    # comment")
        registry_env["finish_tool"]()

        tool_path = os.path.join(tools_dir, "utilities", "alias_test.py")
        assert os.path.exists(tool_path)


class TestSelfDistillRLMInit:
    """Test SelfDistillRLM initialization."""

    @pytest.fixture
    def tools_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @patch("self_distill.rlm.self_distill_rlm.RLM.__init__")
    def test_init_creates_tools_directory(self, mock_rlm_init, tools_dir):
        mock_rlm_init.return_value = None
        tools_path = os.path.join(tools_dir, "test_tools")

        SelfDistillRLM(tools_dir=tools_path)

        assert os.path.isdir(tools_path)
        assert os.path.isdir(os.path.join(tools_path, "pre_completion"))
        assert os.path.isdir(os.path.join(tools_path, "replacements"))
        assert os.path.isdir(os.path.join(tools_path, "utilities"))

    @patch("self_distill.rlm.self_distill_rlm.RLM.__init__")
    def test_init_default_model(self, mock_rlm_init, tools_dir):
        mock_rlm_init.return_value = None

        rlm = SelfDistillRLM(tools_dir=tools_dir)

        assert rlm.model == "ollama/llama3.2:3b"

    @patch("self_distill.rlm.self_distill_rlm.RLM.__init__")
    def test_init_custom_model(self, mock_rlm_init, tools_dir):
        mock_rlm_init.return_value = None

        rlm = SelfDistillRLM(model="ollama/mistral:7b", tools_dir=tools_dir)

        assert rlm.model == "ollama/mistral:7b"

    @patch("self_distill.rlm.self_distill_rlm.RLM.__init__")
    def test_init_tools_dir_is_absolute_path(self, mock_rlm_init, tools_dir):
        mock_rlm_init.return_value = None

        rlm = SelfDistillRLM(tools_dir="relative/path")

        assert rlm.tools_dir.is_absolute()

    @patch("self_distill.rlm.self_distill_rlm.RLM.__init__")
    def test_init_calls_parent_with_correct_args(self, mock_rlm_init, tools_dir):
        mock_rlm_init.return_value = None

        SelfDistillRLM(
            model="ollama/test:1b",
            tools_dir=tools_dir,
            max_iterations=20,
            verbose=True,
        )

        mock_rlm_init.assert_called_once()
        call_kwargs = mock_rlm_init.call_args[1]

        assert call_kwargs["backend"] == "litellm"
        assert call_kwargs["backend_kwargs"]["model_name"] == "ollama/test:1b"
        assert call_kwargs["backend_kwargs"]["api_base"] == "http://localhost:11434"
        assert call_kwargs["environment"] == "local"
        assert "setup_code" in call_kwargs["environment_kwargs"]
        assert call_kwargs["max_iterations"] == 20
        assert call_kwargs["custom_system_prompt"] == TOOL_CREATION_PROMPT
        assert call_kwargs["verbose"] is True

    @patch("self_distill.rlm.self_distill_rlm.RLM.__init__")
    def test_init_passes_extra_kwargs(self, mock_rlm_init, tools_dir):
        mock_rlm_init.return_value = None

        SelfDistillRLM(tools_dir=tools_dir, custom_kwarg="custom_value")

        call_kwargs = mock_rlm_init.call_args[1]
        assert call_kwargs["custom_kwarg"] == "custom_value"

    @patch("self_distill.rlm.self_distill_rlm.RLM.__init__")
    def test_init_initializes_metrics(self, mock_rlm_init, tools_dir):
        mock_rlm_init.return_value = None

        rlm = SelfDistillRLM(tools_dir=tools_dir)

        assert rlm._llm_calls_skipped == 0
        assert rlm._llm_calls_made == 0
        assert rlm._hook_executions == 0
        assert rlm._replacement_uses == 0


class TestSelfDistillRLMGetMetrics:
    """Test SelfDistillRLM.get_metrics()."""

    @pytest.fixture
    def tools_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def rlm(self, tools_dir):
        with patch("self_distill.rlm.self_distill_rlm.RLM.__init__") as mock_init:
            mock_init.return_value = None
            return SelfDistillRLM(tools_dir=tools_dir)

    def test_get_metrics_empty_tools(self, rlm):
        metrics = rlm.get_metrics()

        assert metrics["hooks"] == 0
        assert metrics["replacements"] == 0
        assert metrics["utilities"] == 0
        assert metrics["total_tools"] == 0
        assert metrics["hook_executions"] == 0
        assert metrics["replacement_uses"] == 0
        assert metrics["llm_calls_skipped"] == 0
        assert metrics["llm_calls_made"] == 0

    def test_get_metrics_with_hooks(self, rlm):
        hook_dir = rlm.tools_dir / "pre_completion"
        (hook_dir / "hook1.py").write_text("def check(text): return False")
        (hook_dir / "hook2.py").write_text("def check(text): return False")

        metrics = rlm.get_metrics()

        assert metrics["hooks"] == 2
        assert metrics["total_tools"] == 2

    def test_get_metrics_with_replacements(self, rlm):
        repl_dir = rlm.tools_dir / "replacements"
        (repl_dir / "repl1.py").write_text("def run(text): pass")

        metrics = rlm.get_metrics()

        assert metrics["replacements"] == 1
        assert metrics["total_tools"] == 1

    def test_get_metrics_with_utilities(self, rlm):
        util_dir = rlm.tools_dir / "utilities"
        (util_dir / "util1.py").write_text("def run(text): pass")
        (util_dir / "util2.py").write_text("def run(text): pass")
        (util_dir / "util3.py").write_text("def run(text): pass")

        metrics = rlm.get_metrics()

        assert metrics["utilities"] == 3
        assert metrics["total_tools"] == 3

    def test_get_metrics_mixed_tools(self, rlm):
        (rlm.tools_dir / "pre_completion" / "h1.py").write_text("")
        (rlm.tools_dir / "replacements" / "r1.py").write_text("")
        (rlm.tools_dir / "replacements" / "r2.py").write_text("")
        (rlm.tools_dir / "utilities" / "u1.py").write_text("")
        (rlm.tools_dir / "utilities" / "u2.py").write_text("")
        (rlm.tools_dir / "utilities" / "u3.py").write_text("")

        metrics = rlm.get_metrics()

        assert metrics["hooks"] == 1
        assert metrics["replacements"] == 2
        assert metrics["utilities"] == 3
        assert metrics["total_tools"] == 6

    def test_get_metrics_ignores_non_py_files(self, rlm):
        util_dir = rlm.tools_dir / "utilities"
        (util_dir / "tool.py").write_text("def run(text): pass")
        (util_dir / "readme.txt").write_text("not a tool")
        (util_dir / "data.json").write_text("{}")

        metrics = rlm.get_metrics()

        assert metrics["utilities"] == 1
        assert metrics["total_tools"] == 1

    def test_get_metrics_counts_all_py_files(self, rlm):
        util_dir = rlm.tools_dir / "utilities"
        (util_dir / "tool.py").write_text("def run(text): pass")
        (util_dir / "_private.py").write_text("def run(text): pass")
        (util_dir / "__init__.py").write_text("")

        metrics = rlm.get_metrics()

        # get_metrics counts all .py files (unlike _get_tools_in_category which filters)
        assert metrics["utilities"] == 3


class TestSelfDistillRLMPreCompletionHooks:
    """Test the pre-completion hook system."""

    @pytest.fixture
    def tools_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def rlm(self, tools_dir):
        with patch("self_distill.rlm.self_distill_rlm.RLM.__init__") as mock_init:
            mock_init.return_value = None
            return SelfDistillRLM(tools_dir=tools_dir)

    def test_no_hooks_returns_continue(self, rlm):
        result = rlm._run_pre_completion_hooks("test text")
        assert result["action"] == "continue"

    def test_hook_returns_false_continues(self, rlm):
        hook_dir = rlm.tools_dir / "pre_completion"
        (hook_dir / "always_false.py").write_text("def check(text): return False")

        result = rlm._run_pre_completion_hooks("test text")
        assert result["action"] == "continue"

    def test_hook_returns_true_with_replacement(self, rlm):
        # Create matching hook and replacement
        (rlm.tools_dir / "pre_completion" / "test_hook.py").write_text(
            "def check(text): return True"
        )
        (rlm.tools_dir / "replacements" / "test_hook.py").write_text(
            "def run(text): return 'replaced: ' + text"
        )

        result = rlm._run_pre_completion_hooks("hello")
        assert result["action"] == "replace"
        assert result["tool"] == "test_hook"
        assert result["result"] == "replaced: hello"

    def test_hook_returns_true_without_matching_replacement(self, rlm):
        # Create hook without matching replacement
        (rlm.tools_dir / "pre_completion" / "orphan_hook.py").write_text(
            "def check(text): return True"
        )

        result = rlm._run_pre_completion_hooks("test")
        # Should continue since no matching replacement exists
        assert result["action"] == "continue"

    def test_hook_without_check_function_skipped(self, rlm):
        (rlm.tools_dir / "pre_completion" / "no_check.py").write_text(
            "def other_func(): pass"
        )

        result = rlm._run_pre_completion_hooks("test")
        assert result["action"] == "continue"

    def test_hook_error_continues(self, rlm):
        (rlm.tools_dir / "pre_completion" / "buggy.py").write_text(
            "def check(text): raise ValueError('oops')"
        )

        result = rlm._run_pre_completion_hooks("test")
        assert result["action"] == "continue"

    def test_metrics_increment_on_hook_execution(self, rlm):
        (rlm.tools_dir / "pre_completion" / "hook1.py").write_text(
            "def check(text): return False"
        )

        assert rlm._hook_executions == 0
        rlm._run_pre_completion_hooks("test")
        assert rlm._hook_executions == 1

    def test_metrics_increment_on_replacement_use(self, rlm):
        (rlm.tools_dir / "pre_completion" / "hook1.py").write_text(
            "def check(text): return True"
        )
        (rlm.tools_dir / "replacements" / "hook1.py").write_text(
            "def run(text): return 'replaced'"
        )

        assert rlm._replacement_uses == 0
        rlm._run_pre_completion_hooks("test")
        assert rlm._replacement_uses == 1


class TestSelfDistillRLMHelperMethods:
    """Test helper methods on SelfDistillRLM."""

    @pytest.fixture
    def tools_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def rlm(self, tools_dir):
        with patch("self_distill.rlm.self_distill_rlm.RLM.__init__") as mock_init:
            mock_init.return_value = None
            return SelfDistillRLM(tools_dir=tools_dir)

    def test_get_tools_in_category_empty(self, rlm):
        tools = rlm._get_tools_in_category("utilities")
        assert tools == []

    def test_get_tools_in_category_with_tools(self, rlm):
        util_dir = rlm.tools_dir / "utilities"
        (util_dir / "tool1.py").write_text("")
        (util_dir / "tool2.py").write_text("")

        tools = rlm._get_tools_in_category("utilities")
        assert set(tools) == {"tool1", "tool2"}

    def test_get_tools_in_category_ignores_underscores(self, rlm):
        util_dir = rlm.tools_dir / "utilities"
        (util_dir / "tool.py").write_text("")
        (util_dir / "_private.py").write_text("")

        tools = rlm._get_tools_in_category("utilities")
        assert tools == ["tool"]

    def test_load_tool_module_exists(self, rlm):
        (rlm.tools_dir / "utilities" / "test_tool.py").write_text(
            "def run(text): return text.upper()"
        )

        module = rlm._load_tool_module("utilities", "test_tool")
        assert module is not None
        assert hasattr(module, "run")
        assert module.run("hello") == "HELLO"

    def test_load_tool_module_not_exists(self, rlm):
        module = rlm._load_tool_module("utilities", "nonexistent")
        assert module is None
