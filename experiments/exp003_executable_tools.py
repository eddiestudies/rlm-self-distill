"""
Experiment 003: Executable Python Tools

This experiment creates ACTUAL executable Python tools that:
1. Get saved to a local tools directory
2. Can be loaded and executed
3. Have a classifier tool that routes tasks to appropriate tools
4. Fall back to LLM if tool execution fails

Architecture:
- tools/classifiers/ - Tools that classify/route tasks
- tools/grammar/ - Grammar checking tools
- tools/pii/ - PII detection tools
- Tool registry that loads and executes tools
- LLM fallback when tools fail or don't exist
"""

import importlib.util
import json
import random
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from self_distill import (
    DATA,
    ExperimentTracker,
    OllamaClient,
    load_dataset,
)

from experiments.exp001_recursive_tool_creation import TaskItem


# === Tool Infrastructure ===


@dataclass
class ToolResult:
    """Result from executing a tool."""

    success: bool
    output: Any
    error: str | None = None
    tokens_used: int = 0  # If LLM fallback was used


@dataclass
class ExecutableTool:
    """An executable Python tool."""

    name: str
    tool_type: str  # "classifier", "grammar", "pii"
    description: str
    code: str
    file_path: Path
    created_at: datetime = field(default_factory=datetime.now)
    executions: int = 0
    failures: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.tool_type,
            "description": self.description,
            "code": self.code,
            "file_path": str(self.file_path),
            "executions": self.executions,
            "failures": self.failures,
        }


class ToolRegistry:
    """Registry for loading and executing Python tools."""

    def __init__(self, tools_dir: Path):
        self.tools_dir = tools_dir
        self.tools: dict[str, ExecutableTool] = {}
        self._loaded_modules: dict[str, Any] = {}

        # Create directory structure
        (tools_dir / "classifiers").mkdir(parents=True, exist_ok=True)
        (tools_dir / "grammar").mkdir(parents=True, exist_ok=True)
        (tools_dir / "pii").mkdir(parents=True, exist_ok=True)

    def register_tool(self, tool: ExecutableTool) -> bool:
        """Register and save a tool to disk."""
        try:
            # Save code to file
            with open(tool.file_path, "w") as f:
                f.write(tool.code)

            # Try to load it to validate
            self._load_module(tool)

            self.tools[tool.name] = tool
            return True
        except Exception as e:
            print(f"Failed to register tool {tool.name}: {e}")
            return False

    def _load_module(self, tool: ExecutableTool) -> Any:
        """Load a tool's Python module."""
        if tool.name in self._loaded_modules:
            # Reload to get latest version
            del self._loaded_modules[tool.name]

        spec = importlib.util.spec_from_file_location(tool.name, tool.file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[tool.name] = module
        spec.loader.exec_module(module)

        self._loaded_modules[tool.name] = module
        return module

    def execute_tool(self, tool_name: str, input_text: str) -> ToolResult:
        """Execute a tool on input text."""
        if tool_name not in self.tools:
            return ToolResult(
                success=False, output=None, error=f"Tool {tool_name} not found"
            )

        tool = self.tools[tool_name]
        tool.executions += 1

        try:
            module = self._load_module(tool)

            # All tools should have a 'run' function
            if not hasattr(module, "run"):
                return ToolResult(
                    success=False, output=None, error="Tool has no 'run' function"
                )

            result = module.run(input_text)
            return ToolResult(success=True, output=result)

        except Exception as e:
            tool.failures += 1
            return ToolResult(
                success=False,
                output=None,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            )

    def get_tools_by_type(self, tool_type: str) -> list[ExecutableTool]:
        """Get all tools of a specific type."""
        return [t for t in self.tools.values() if t.tool_type == tool_type]

    def get_tool_summary(self) -> str:
        """Get a summary of available tools for the LLM."""
        lines = ["Available Python tools:"]
        for tool in self.tools.values():
            lines.append(f"- {tool.name} ({tool.tool_type}): {tool.description}")
        return "\n".join(lines)


# === Tool Creation Prompts ===

TOOL_CREATION_SYSTEM_PROMPT = """You are a Python tool creator. When asked to create a tool, output ONLY valid Python code.

The code must:
1. Have a function called `run(text: str)` that takes input text and returns a result
2. Be self-contained (import any needed standard libraries at the top)
3. Handle errors gracefully
4. Return appropriate types:
   - Classifiers: return a string category like "grammar", "pii", "other"
   - Grammar tools: return dict {"acceptable": bool, "reason": str}
   - PII tools: return list of dicts [{"type": str, "text": str, "start": int, "end": int}]

Do NOT include markdown code blocks. Output raw Python code only."""

CLASSIFIER_CREATION_PROMPT = """Create a Python classifier tool that determines if text needs grammar checking or PII detection.

The function should:
1. Look for patterns indicating PII (emails, phone numbers, SSNs, IP addresses, etc.)
2. Otherwise assume it's a grammar task
3. Return "pii" if PII patterns found, "grammar" otherwise

Example patterns to check:
- Email: contains @ and .
- Phone: digits with dashes/parentheses in phone-like patterns
- SSN: XXX-XX-XXXX pattern
- IP: X.X.X.X pattern

Output only the Python code:"""

GRAMMAR_TOOL_PROMPT = """Create a Python grammar checking tool for this specific issue: {issue}

The function should:
1. Check for the specific grammar issue
2. Return {{"acceptable": True/False, "reason": "explanation"}}
3. Use simple pattern matching or heuristics

Common grammar checks:
- Subject-verb agreement (singular/plural)
- Sentence structure (has subject and verb)
- Common errors (their/there/they're, its/it's, etc.)

Output only the Python code:"""

PII_TOOL_PROMPT = """Create a Python PII detection tool for detecting: {pii_type}

The function should:
1. Find all instances of {pii_type} in the text
2. Return a list of dicts: [{{"type": "{pii_type}", "text": "found text", "start": int, "end": int}}]
3. Use regex patterns

Output only the Python code:"""


# === Main Experiment ===


class ExecutableToolExperiment:
    """Experiment with actual executable Python tools."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        output_dir: str = "experiment_outputs",
    ):
        self.model_name = model_name
        self.client = OllamaClient(model_name=model_name, host=host)
        self.tracker = ExperimentTracker("executable-tools")

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / f"exp003_{self.timestamp}"
        self.tools_dir = self.run_dir / "tools"

        self.registry = ToolRegistry(self.tools_dir)

        # Stats
        self.tool_tokens = 0  # Tokens used creating tools
        self.tool_executions = 0
        self.tool_failures = 0
        self.llm_fallbacks = 0
        self.llm_fallback_tokens = 0

    def create_tool_from_llm(
        self,
        prompt: str,
        tool_name: str,
        tool_type: str,
        description: str,
    ) -> ExecutableTool | None:
        """Have the LLM generate a Python tool."""
        messages = [
            {"role": "system", "content": TOOL_CREATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.client.completion(messages)
        usage = self.client.get_last_usage()
        self.tool_tokens += usage.total_input_tokens + usage.total_output_tokens

        # Clean up the response - remove markdown if present
        code = response.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        # Determine file path
        subdir = self.tools_dir / tool_type
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", tool_name.lower())
        file_path = subdir / f"{safe_name}.py"

        tool = ExecutableTool(
            name=tool_name,
            tool_type=tool_type,
            description=description,
            code=code,
            file_path=file_path,
        )

        if self.registry.register_tool(tool):
            print(f"  Created tool: {tool_name} ({tool_type})")
            return tool
        else:
            print(f"  Failed to create tool: {tool_name}")
            return None

    def create_base_tools(self):
        """Create the base classifier and common tools."""
        print("\nCreating base tools...")

        # 1. Task classifier
        self.create_tool_from_llm(
            CLASSIFIER_CREATION_PROMPT,
            "task_classifier",
            "classifiers",
            "Classifies text as grammar or PII task",
        )

        # 2. Basic grammar checker
        self.create_tool_from_llm(
            GRAMMAR_TOOL_PROMPT.format(
                issue="basic sentence structure - check if sentence has subject and verb"
            ),
            "basic_grammar",
            "grammar",
            "Checks basic sentence structure",
        )

        # 3. Email detector
        self.create_tool_from_llm(
            PII_TOOL_PROMPT.format(pii_type="EMAIL"),
            "email_detector",
            "pii",
            "Detects email addresses",
        )

        # 4. Phone detector
        self.create_tool_from_llm(
            PII_TOOL_PROMPT.format(pii_type="PHONE"),
            "phone_detector",
            "pii",
            "Detects phone numbers",
        )

        # 5. SSN detector
        self.create_tool_from_llm(
            PII_TOOL_PROMPT.format(pii_type="SSN"),
            "ssn_detector",
            "pii",
            "Detects Social Security Numbers",
        )

        # 6. IP address detector
        self.create_tool_from_llm(
            PII_TOOL_PROMPT.format(pii_type="IP_ADDRESS"),
            "ip_detector",
            "pii",
            "Detects IP addresses",
        )

        print(f"  Created {len(self.registry.tools)} base tools")

    def classify_task(self, text: str) -> str:
        """Use classifier tool to determine task type."""
        result = self.registry.execute_tool("task_classifier", text)
        if result.success and result.output in ["grammar", "pii"]:
            return result.output
        # Fallback heuristic
        if any(c in text for c in ["@", "XXX-XX", "555-", "(555)"]):
            return "pii"
        return "grammar"

    def run_grammar_tools(self, text: str) -> ToolResult:
        """Run grammar tools on text."""
        grammar_tools = self.registry.get_tools_by_type("grammar")

        for tool in grammar_tools:
            result = self.registry.execute_tool(tool.name, text)
            self.tool_executions += 1

            if result.success:
                return result
            else:
                self.tool_failures += 1

        # No tools worked
        return ToolResult(
            success=False, output=None, error="No grammar tools succeeded"
        )

    def run_pii_tools(self, text: str) -> ToolResult:
        """Run all PII tools and aggregate results."""
        pii_tools = self.registry.get_tools_by_type("pii")
        all_entities = []
        any_success = False

        for tool in pii_tools:
            result = self.registry.execute_tool(tool.name, text)
            self.tool_executions += 1

            if result.success and isinstance(result.output, list):
                all_entities.extend(result.output)
                any_success = True
            elif not result.success:
                self.tool_failures += 1

        if any_success:
            return ToolResult(success=True, output=all_entities)
        return ToolResult(success=False, output=None, error="No PII tools succeeded")

    def llm_fallback(self, task: TaskItem) -> tuple[str, int]:
        """Fall back to LLM when tools fail."""
        self.llm_fallbacks += 1

        if task.dataset_type == "cola":
            prompt = f'Is this sentence grammatically acceptable? Answer only 0 or 1: "{task.text}"'
        else:
            prompt = f'List PII in this text as JSON array: "{task.text}"'

        response = self.client.completion(prompt)
        usage = self.client.get_last_usage()
        tokens = usage.total_input_tokens + usage.total_output_tokens
        self.llm_fallback_tokens += tokens

        return response, tokens

    def process_task(self, task: TaskItem) -> dict:
        """Process a single task using tools with LLM fallback."""
        # Classify
        task_type = self.classify_task(task.text)

        # Run appropriate tools
        if task_type == "grammar":
            result = self.run_grammar_tools(task.text)
        else:
            result = self.run_pii_tools(task.text)

        # Check if we need LLM fallback
        if result.success:
            return {
                "task_type": task_type,
                "used_tool": True,
                "tool_output": result.output,
                "llm_fallback": False,
                "tokens_used": 0,
            }
        else:
            # Fallback to LLM
            response, tokens = self.llm_fallback(task)
            return {
                "task_type": task_type,
                "used_tool": False,
                "tool_error": result.error,
                "llm_fallback": True,
                "llm_response": response,
                "tokens_used": tokens,
            }

    def run_baseline(self, tasks: list[TaskItem]) -> dict:
        """Run baseline (direct LLM, no tools)."""
        results = []
        total_tokens = 0

        for i, task in enumerate(tqdm(tasks, desc="Baseline", unit="task")):
            if task.dataset_type == "cola":
                prompt = (
                    f'Is this grammatically acceptable? Answer 0 or 1: "{task.text}"'
                )
            else:
                prompt = f'List PII as JSON: "{task.text}"'

            response = self.client.completion(prompt)
            usage = self.client.get_last_usage()
            tokens = usage.total_input_tokens + usage.total_output_tokens
            total_tokens += tokens

            results.append(
                {
                    "task_index": i,
                    "dataset_type": task.dataset_type,
                    "tokens": tokens,
                    "response": response[:100],
                }
            )

        return {"results": results, "total_tokens": total_tokens}

    def run_with_tools(self, tasks: list[TaskItem]) -> dict:
        """Run with executable tools."""
        results = []

        pbar = tqdm(enumerate(tasks), total=len(tasks), desc="With Tools", unit="task")
        for i, task in pbar:
            result = self.process_task(task)
            result["task_index"] = i
            result["dataset_type"] = task.dataset_type
            results.append(result)

            tools_used = sum(1 for r in results if r.get("used_tool"))
            fallbacks = sum(1 for r in results if r.get("llm_fallback"))
            pbar.set_postfix(tools=tools_used, fallbacks=fallbacks)

        return {
            "results": results,
            "tool_tokens": self.tool_tokens,
            "llm_fallback_tokens": self.llm_fallback_tokens,
            "total_tokens": self.tool_tokens + self.llm_fallback_tokens,
            "tool_executions": self.tool_executions,
            "tool_failures": self.tool_failures,
            "llm_fallbacks": self.llm_fallbacks,
        }

    def generate_report(
        self, baseline: dict, with_tools: dict, tasks: list[TaskItem]
    ) -> Path:
        """Generate PDF report."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        filepath = self.run_dir / "report.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(
            Paragraph("Experiment 003: Executable Python Tools", styles["Heading1"])
        )
        story.append(
            Paragraph(
                f"Model: {self.model_name} | Tasks: {len(tasks)}", styles["Normal"]
            )
        )
        story.append(Spacer(1, 0.2 * inch))

        # Summary
        story.append(Paragraph("Token Comparison", styles["Heading2"]))

        tools_total = with_tools["total_tokens"]
        baseline_total = baseline["total_tokens"]
        savings = baseline_total - tools_total
        savings_pct = (savings / baseline_total * 100) if baseline_total > 0 else 0

        data = [
            ["Metric", "Baseline", "With Tools"],
            ["Total Tokens", f"{baseline_total:,}", f"{tools_total:,}"],
            ["Tool Creation", "-", f"{with_tools['tool_tokens']:,}"],
            ["LLM Fallbacks", "-", f"{with_tools['llm_fallback_tokens']:,}"],
            ["Savings", "-", f"{savings:,} ({savings_pct:+.1f}%)"],
        ]

        table = Table(data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        # Tool stats
        story.append(Paragraph("Tool Execution Stats", styles["Heading2"]))

        tool_data = [
            ["Metric", "Value"],
            ["Tools Created", str(len(self.registry.tools))],
            ["Tool Executions", str(with_tools["tool_executions"])],
            ["Tool Failures", str(with_tools["tool_failures"])],
            ["LLM Fallbacks", str(with_tools["llm_fallbacks"])],
        ]

        tool_table = Table(tool_data, colWidths=[2 * inch, 1.5 * inch])
        tool_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(tool_table)
        story.append(Spacer(1, 0.2 * inch))

        # Tools created
        story.append(Paragraph("Tools Created", styles["Heading2"]))
        for tool in self.registry.tools.values():
            story.append(
                Paragraph(
                    f"<b>{tool.name}</b> ({tool.tool_type}): {tool.description}",
                    styles["Normal"],
                )
            )
            story.append(
                Paragraph(
                    f"  Executions: {tool.executions}, Failures: {tool.failures}",
                    styles["Normal"],
                )
            )

        doc.build(story)
        return filepath

    def save_results(self, baseline: dict, with_tools: dict) -> Path:
        """Save results to JSON."""
        filepath = self.run_dir / "results.json"

        data = {
            "metadata": {
                "model": self.model_name,
                "timestamp": self.timestamp,
            },
            "baseline": {
                "total_tokens": baseline["total_tokens"],
                "results_count": len(baseline["results"]),
            },
            "with_tools": {
                "total_tokens": with_tools["total_tokens"],
                "tool_tokens": with_tools["tool_tokens"],
                "llm_fallback_tokens": with_tools["llm_fallback_tokens"],
                "tool_executions": with_tools["tool_executions"],
                "tool_failures": with_tools["tool_failures"],
                "llm_fallbacks": with_tools["llm_fallbacks"],
            },
            "tools": [t.to_dict() for t in self.registry.tools.values()],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath


def load_mixed_dataset(
    cola_count: int, pii_count: int, seed: int = 42
) -> list[TaskItem]:
    """Load mixed dataset."""
    tasks = []

    cola_data = list(load_dataset(DATA.COLA, "train"))
    for i in range(min(cola_count, len(cola_data))):
        item = cola_data[i]
        tasks.append(
            TaskItem(
                text=item.question,
                expected_answer=item.answer,
                dataset_type="cola",
                original_index=i,
            )
        )

    pii_data = list(load_dataset(DATA.PII_DETECTION, "train"))
    for i in range(min(pii_count, len(pii_data))):
        item = pii_data[i]
        tasks.append(
            TaskItem(
                text=item.question,
                expected_answer=item.answer,
                dataset_type="pii",
                original_index=i,
            )
        )

    random.seed(seed)
    random.shuffle(tasks)
    return tasks


def run_experiment(
    model_name: str = "llama3.2:3b",
    cola_count: int = 50,
    pii_count: int = 15,
) -> dict:
    """Run the executable tools experiment."""
    print("=" * 60)
    print("Experiment 003: Executable Python Tools")
    print("=" * 60)

    experiment = ExecutableToolExperiment(model_name=model_name)

    # Create base tools
    experiment.create_base_tools()

    # Load data
    print(f"\nLoading {cola_count} CoLA + {pii_count} PII tasks...")
    tasks = load_mixed_dataset(cola_count, pii_count)
    print(f"Loaded {len(tasks)} tasks")

    # Run baseline
    print("\n--- Running Baseline ---")
    baseline = experiment.run_baseline(tasks)
    print(f"Baseline tokens: {baseline['total_tokens']:,}")

    # Run with tools
    print("\n--- Running With Tools ---")
    with_tools = experiment.run_with_tools(tasks)
    print(f"Tool creation tokens: {with_tools['tool_tokens']:,}")
    print(f"LLM fallback tokens: {with_tools['llm_fallback_tokens']:,}")
    print(f"Total with tools: {with_tools['total_tokens']:,}")

    # Generate report
    print("\n--- Generating Report ---")
    report_path = experiment.generate_report(baseline, with_tools, tasks)
    experiment.save_results(baseline, with_tools)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    savings = baseline["total_tokens"] - with_tools["total_tokens"]
    savings_pct = (
        (savings / baseline["total_tokens"] * 100)
        if baseline["total_tokens"] > 0
        else 0
    )

    print(f"Baseline: {baseline['total_tokens']:,} tokens")
    print(f"With Tools: {with_tools['total_tokens']:,} tokens")
    print(f"Savings: {savings:,} tokens ({savings_pct:+.1f}%)")
    print(f"\nTool executions: {with_tools['tool_executions']}")
    print(f"Tool failures: {with_tools['tool_failures']}")
    print(f"LLM fallbacks: {with_tools['llm_fallbacks']}")
    print(f"\nReport: {report_path}")
    print(f"Tools directory: {experiment.tools_dir}")

    return {
        "baseline": baseline,
        "with_tools": with_tools,
        "report": str(report_path),
        "tools_dir": str(experiment.tools_dir),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--cola", type=int, default=50)
    parser.add_argument("--pii", type=int, default=15)

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        cola_count=args.cola,
        pii_count=args.pii,
    )
