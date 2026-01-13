"""
Self-Distillation RLM: Extends RLM to create reusable Python tools with hooks.

Simplified Architecture:
- pre_completion/: Hooks with check(text) -> bool
- replacements/: Tools with run(text) -> str/dict (matched by name to hooks)
- utilities/: General helper tools with run(text) -> str/dict

Flow:
1. For each hook in pre_completion/, call check(text)
2. If True, find replacements/{hook_name}.py and run it instead of LLM
3. If all hooks return False, call LLM normally
"""

from pathlib import Path
from typing import Any

from rlm import RLM
from rlm.core.types import RLMChatCompletion


TOOL_CREATION_PROMPT = """You are a Self-Distilling Language Model. Your goal is to create reusable Python tools that reduce future LLM calls.

CRITICAL: ALL code MUST be inside ```repl blocks. Use EXACTLY this format:

```repl
your_code_here()
```

NOT ```python, NOT ``` alone - ONLY ```repl will execute!

## Tool System Architecture

Your tools directory has three subdirectories:

### 1. pre_completion/ - Detection hooks (return True/False)
These tools have a `check(text) -> bool` function.
- Return `True` if this task should be handled by a replacement tool
- Return `False` to let the LLM handle it

The hook name must MATCH the replacement tool name. If you create `pre_completion/grammar_checker.py`, there must be a matching `replacements/grammar_checker.py`.

### 2. replacements/ - Tools that REPLACE the LLM (return result)
These tools have a `run(text) -> str` or `run(text) -> dict` function.
When a pre_completion hook returns True, the matching replacement tool runs instead of the LLM.

### 3. utilities/ - Helper tools (return str or dict)
General purpose tools with `run(text)` function.

## REPL Functions Available

- `list_all_tools()` - List all tools in all directories
- `create_tool(category, name, description)` - Start creating a tool
- `write_to_tool(code_line)` - Write a line to current tool
- `finish_tool()` - Save and load the tool
- `run_tool(category, name, text)` - Execute a tool's run(text)

## Example: Creating a Grammar Checker

Step 1: Create the detection hook (returns bool):

```repl
create_tool("pre_completion", "grammar_checker", "Detects grammar tasks")
write_to_tool("def check(text: str) -> bool:")
write_to_tool("    '''Return True if this is a grammar task.'''")
write_to_tool("    keywords = ['grammar', 'grammatical', 'acceptable', 'sentence']")
write_to_tool("    text_lower = text.lower()")
write_to_tool("    return any(kw in text_lower for kw in keywords)")
finish_tool()
```

Step 2: Create the matching replacement tool (returns result):

```repl
create_tool("replacements", "grammar_checker", "Checks grammar without LLM")
write_to_tool("def run(text: str) -> dict:")
write_to_tool("    '''Check if sentence is grammatically acceptable.'''")
write_to_tool("    issues = []")
write_to_tool("    if text and not text[0].isupper():")
write_to_tool("        issues.append('Should start with capital')")
write_to_tool("    if text and not text.rstrip().endswith(('.', '!', '?')):")
write_to_tool("        issues.append('Should end with punctuation')")
write_to_tool("    return {'acceptable': len(issues) == 0, 'issues': issues}")
finish_tool()
```

## Example: Creating a PII Detector

Step 1: Create the detection hook:

```repl
create_tool("pre_completion", "pii_detector", "Detects PII in text")
write_to_tool("import re")
write_to_tool("")
write_to_tool("def check(text: str) -> bool:")
write_to_tool("    '''Return True if text contains PII.'''")
write_to_tool("    patterns = [")
write_to_tool("        r'[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}',  # email")
write_to_tool("        r'\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}',  # phone")
write_to_tool("        r'\\d{3}-\\d{2}-\\d{4}'  # SSN")
write_to_tool("    ]")
write_to_tool("    return any(re.search(p, text) for p in patterns)")
finish_tool()
```

Step 2: Create the matching replacement:

```repl
create_tool("replacements", "pii_detector", "Analyzes PII without LLM")
write_to_tool("import re")
write_to_tool("")
write_to_tool("def run(text: str) -> dict:")
write_to_tool("    '''Detect and report PII in text.'''")
write_to_tool("    found = []")
write_to_tool("    if re.search(r'[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}', text):")
write_to_tool("        found.append('email')")
write_to_tool("    if re.search(r'\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}', text):")
write_to_tool("        found.append('phone')")
write_to_tool("    if re.search(r'\\d{3}-\\d{2}-\\d{4}', text):")
write_to_tool("        found.append('ssn')")
write_to_tool("    return {'has_pii': len(found) > 0, 'pii_types': found}")
finish_tool()
```

## Workflow

1. Run `list_all_tools()` to see existing tools
2. For tasks you can solve with code, create BOTH:
   - A pre_completion hook that returns True when this task type is detected
   - A matching replacement tool that handles the task
3. Future tasks matching the hook will be handled WITHOUT calling the LLM!

## Important Rules

- Hook functions MUST return bool (True or False)
- Hook and replacement names MUST match exactly
- Replacement functions return the actual result (str or dict)
- Always create BOTH the hook AND the replacement together

## Final Answer

When done analyzing, return your answer using:
- FINAL(your answer here)
- FINAL_VAR(variable_name) to return a variable
"""


def get_tool_registry_setup_code(tools_dir: str) -> str:
    """Generate setup code that creates the tool registry in the REPL."""
    return f'''
import os
import sys
import importlib.util

TOOLS_DIR = "{tools_dir}"
CATEGORIES = ["pre_completion", "replacements", "utilities"]

# Create directory structure
for cat in CATEGORIES:
    os.makedirs(os.path.join(TOOLS_DIR, cat), exist_ok=True)

# State for tool creation
_current_tool_category = None
_current_tool_name = None
_current_tool_lines = []

def _get_tools_in_category(category):
    """Get list of tools in a category."""
    cat_dir = os.path.join(TOOLS_DIR, category)
    if not os.path.exists(cat_dir):
        return []
    return [f[:-3] for f in os.listdir(cat_dir) if f.endswith(".py") and not f.startswith("_")]

def list_all_tools():
    """List all tools in all categories."""
    result = {{}}
    for cat in CATEGORIES:
        tools = _get_tools_in_category(cat)
        result[cat] = tools
        print(f"{{cat}}: {{tools}}")
    return result

def load_tool(category, name):
    """Load and return a tool module."""
    path = os.path.join(TOOLS_DIR, category, f"{{name}}.py")
    if not os.path.exists(path):
        print(f"Tool not found: {{category}}/{{name}}")
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def create_tool(category, name, description=""):
    """Start creating a new tool in a category."""
    global _current_tool_category, _current_tool_name, _current_tool_lines
    if category not in CATEGORIES:
        print(f"Error: Invalid category '{{category}}'. Use one of: {{CATEGORIES}}")
        return
    _current_tool_category = category
    _current_tool_name = name
    _current_tool_lines = []
    if description:
        _current_tool_lines.append(f'# {{description}}')
        _current_tool_lines.append('')
    print(f"Creating tool: {{category}}/{{name}}")

def write_to_tool(line):
    """Write a line to the current tool."""
    global _current_tool_lines
    if _current_tool_name is None:
        print("Error: No tool being created. Call create_tool first.")
        return
    _current_tool_lines.append(line)

# Aliases for common LLM typos
write_to_text = write_to_tool
add_line = write_to_tool
write_line = write_to_tool

def finish_tool():
    """Finalize and save the current tool."""
    global _current_tool_category, _current_tool_name, _current_tool_lines
    if _current_tool_name is None:
        print("Error: No tool being created.")
        return None

    code = "\\n".join(_current_tool_lines)
    path = os.path.join(TOOLS_DIR, _current_tool_category, f"{{_current_tool_name}}.py")

    with open(path, "w") as f:
        f.write(code)

    print(f"Tool saved: {{path}}")

    # Verify the tool
    try:
        module = load_tool(_current_tool_category, _current_tool_name)
        if _current_tool_category == "pre_completion":
            if hasattr(module, 'check'):
                print(f"Hook {{_current_tool_name}} has check() -> bool - ready!")
            else:
                print(f"Warning: Hook {{_current_tool_name}} needs check(text) -> bool function")
        else:
            if hasattr(module, 'run'):
                print(f"Tool {{_current_tool_name}} has run() - ready!")
            else:
                print(f"Warning: Tool {{_current_tool_name}} needs run(text) function")
    except Exception as e:
        print(f"Warning: Could not load tool: {{e}}")

    name = _current_tool_name
    _current_tool_category = None
    _current_tool_name = None
    _current_tool_lines = []
    return name

def run_tool(category, name, input_text):
    """Execute a tool on input text."""
    module = load_tool(category, name)
    if module is None:
        return {{"error": f"Tool {{category}}/{{name}} not found"}}
    if not hasattr(module, 'run'):
        return {{"error": f"Tool {{category}}/{{name}} has no run() function"}}
    try:
        result = module.run(input_text)
        print(f"Tool {{name}} result: {{result}}")
        return result
    except Exception as e:
        return {{"error": str(e)}}

print("=== Self-Distill Tool Registry Ready ===")
print(f"Tools directory: {{TOOLS_DIR}}")
print()
list_all_tools()
'''


class SelfDistillRLM(RLM):
    """
    Extended RLM with simplified hook-based tool system.

    Tools are organized into:
    - pre_completion/: Hooks with check(text) -> bool
    - replacements/: Tools with run(text) -> result (matched by name to hooks)
    - utilities/: General helper tools
    """

    def __init__(
        self,
        model: str = "ollama/llama3.2:3b",
        tools_dir: Path | str = "tools",
        max_iterations: int = 10,
        verbose: bool = False,
        **kwargs,
    ):
        self.model = model
        self.tools_dir = Path(tools_dir).absolute()
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        # Metrics for tracking
        self._llm_calls_skipped = 0
        self._llm_calls_made = 0
        self._hook_executions = 0
        self._replacement_uses = 0

        # Create category subdirectories
        for category in ["pre_completion", "replacements", "utilities"]:
            (self.tools_dir / category).mkdir(exist_ok=True)

        # Generate setup code for tool registry
        setup_code = get_tool_registry_setup_code(str(self.tools_dir))

        # Initialize base RLM with litellm backend
        super().__init__(
            backend="litellm",
            backend_kwargs={
                "model_name": model,
                "api_base": "http://localhost:11434",
            },
            environment="local",
            environment_kwargs={"setup_code": setup_code},
            max_iterations=max_iterations,
            custom_system_prompt=TOOL_CREATION_PROMPT,
            verbose=verbose,
            **kwargs,
        )

    def _load_tool_module(self, category: str, name: str):
        """Load a tool module from disk."""
        import importlib.util

        path = self.tools_dir / category / f"{name}.py"
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _get_tools_in_category(self, category: str) -> list[str]:
        """Get list of tool names in a category."""
        cat_dir = self.tools_dir / category
        if not cat_dir.exists():
            return []
        return [f.stem for f in cat_dir.glob("*.py") if not f.name.startswith("_")]

    def _run_pre_completion_hooks(self, text: str) -> dict:
        """
        Run all pre_completion hooks on text BEFORE calling the LLM.

        Each hook's check(text) returns bool:
        - True: use matching replacement tool, skip LLM
        - False: continue checking other hooks

        Returns:
            {"action": "continue"} - no hook matched, call LLM
            {"action": "replace", "tool": "...", "result": ...} - skip LLM, use result
        """
        hooks = self._get_tools_in_category("pre_completion")

        for hook_name in hooks:
            hook_module = self._load_tool_module("pre_completion", hook_name)
            if hook_module is None or not hasattr(hook_module, "check"):
                continue

            self._hook_executions += 1

            try:
                # Call check(text) -> bool
                should_replace = hook_module.check(text)

                if should_replace:
                    # Look for matching replacement tool
                    replacement = self._load_tool_module("replacements", hook_name)
                    if replacement and hasattr(replacement, "run"):
                        self._replacement_uses += 1
                        result = replacement.run(text)
                        return {
                            "action": "replace",
                            "tool": hook_name,
                            "result": result,
                        }

            except Exception as e:
                print(f"Hook {hook_name} error: {e}")
                continue

        return {"action": "continue"}

    def completion(self, prompt: str | dict, **kwargs) -> RLMChatCompletion:
        """
        Override completion to run pre_completion hooks first.

        If a hook returns True and has a matching replacement, skip the LLM.
        Otherwise, proceed with the normal RLM completion.
        """
        # Extract text from prompt
        if isinstance(prompt, dict):
            text = prompt.get("text", prompt.get("content", str(prompt)))
        else:
            text = str(prompt)

        # Run pre-completion hooks
        hook_result = self._run_pre_completion_hooks(text)

        if hook_result["action"] == "replace":
            # Skip LLM - return a mock completion with the tool result
            self._llm_calls_skipped += 1

            # Create a minimal RLMChatCompletion-like response
            from dataclasses import dataclass

            @dataclass
            class MockUsageSummary:
                model_usage_summaries: dict

            @dataclass
            class MockCompletion:
                root_model: str
                prompt: str
                response: str
                usage_summary: MockUsageSummary
                execution_time: float
                replaced_by_tool: str
                tool_result: Any

            return MockCompletion(
                root_model=self.model,
                prompt=text,
                response=str(hook_result["result"]),
                usage_summary=MockUsageSummary(model_usage_summaries={}),
                execution_time=0.0,
                replaced_by_tool=hook_result["tool"],
                tool_result=hook_result["result"],
            )

        # Proceed with normal LLM completion
        self._llm_calls_made += 1
        return super().completion(prompt, **kwargs)

    def get_metrics(self) -> dict:
        """Get tool usage metrics."""
        hooks = len(list((self.tools_dir / "pre_completion").glob("*.py")))
        replacements = len(list((self.tools_dir / "replacements").glob("*.py")))
        utilities = len(list((self.tools_dir / "utilities").glob("*.py")))
        return {
            "hooks": hooks,
            "replacements": replacements,
            "utilities": utilities,
            "total_tools": hooks + replacements + utilities,
            "hook_executions": self._hook_executions,
            "replacement_uses": self._replacement_uses,
            "llm_calls_skipped": self._llm_calls_skipped,
            "llm_calls_made": self._llm_calls_made,
        }
