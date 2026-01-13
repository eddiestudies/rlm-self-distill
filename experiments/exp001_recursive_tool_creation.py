"""
Experiment 001: Recursive Tool Creation for Efficiency and Safety

This experiment tests a recursive language model that:
1. Creates preprocessing tools for safety issues
2. Creates efficiency tools for repeated task patterns
3. Tracks tool creation and usage over time
4. Compares against baseline (no tools/rules)

Runs on mixed CoLA (grammaticality) and PII (detection) datasets.
"""

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from self_distill import (
    DATA,
    CallType,
    ExperimentTracker,
    OllamaClient,
    load_dataset,
)

# === System Prompt ===

RECURSIVE_TOOL_SYSTEM_PROMPT = """You are a recursive language model that wants to resolve your tasks well but needs to do so efficiently and safely.

When solving a task, if you notice there was a safety issue in reading the text to begin with. Add preprocessing tools that run before you look at the text. They should be a pipeline of classifiers for whether to run more involved tools after. The output from your tools will be passed into you to further complete the task. The recursive calls will not create tools unless they are instructed to.

While completing a task, if you notice there is an efficiency for that type of task, a tool and skill can be made to leverage later to answer correctly but more quickly.

The rules and tools created should keep the correctness as if you answered the question directly, but with performance and auditability.

IMPORTANT: When you create a tool, output it in this format:
<tool>
name: [tool_name]
type: [safety|efficiency]
description: [what the tool does]
implementation: [the rule or pattern to apply]
</tool>

When using a tool, reference it as: <use_tool name="tool_name" />

For the task itself, provide your answer after any tool definitions."""


# === Data Classes ===


@dataclass
class TaskItem:
    """A unified task item from mixed datasets."""

    text: str
    expected_answer: str
    dataset_type: str  # "cola" or "pii"
    original_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """A tool created by the model."""

    name: str
    tool_type: str  # "safety" or "efficiency"
    description: str
    implementation: str
    created_at_task: int
    creation_tokens: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.tool_type,
            "description": self.description,
            "implementation": self.implementation,
            "created_at_task": self.created_at_task,
            "creation_tokens": self.creation_tokens,
        }


@dataclass
class ExperimentResult:
    """Result of running the experiment."""

    # Task results
    total_tasks: int
    correct_predictions: int

    # Token accounting
    baseline_total_tokens: int
    rlm_total_tokens: int
    tool_creation_tokens: int
    tool_usage_tokens: int

    # Tool statistics
    tools_created: list[ToolDefinition]
    tool_usage_counts: dict[str, int]

    # Per-task breakdown
    task_results: list[dict[str, Any]]


# === Helper Functions ===


def load_mixed_dataset(
    cola_split: str = "train",
    pii_split: str = "train",
    cola_limit: int | None = None,
    pii_limit: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[TaskItem]:
    """Load and mix CoLA and PII datasets."""
    tasks = []

    # Load CoLA
    cola_data = load_dataset(DATA.COLA, cola_split)
    cola_items = list(cola_data)
    if cola_limit:
        cola_items = cola_items[:cola_limit]

    for i, item in enumerate(cola_items):
        tasks.append(
            TaskItem(
                text=item.question,
                expected_answer=item.answer,
                dataset_type="cola",
                original_index=i,
                metadata={"task": "grammaticality_judgment"},
            )
        )

    # Load PII Detection
    pii_data = load_dataset(DATA.PII_DETECTION, pii_split)
    pii_items = list(pii_data)
    if pii_limit:
        pii_items = pii_items[:pii_limit]

    for i, item in enumerate(pii_items):
        tasks.append(
            TaskItem(
                text=item.question,
                expected_answer=item.answer,
                dataset_type="pii",
                original_index=i,
                metadata={"task": "pii_detection"},
            )
        )

    if shuffle:
        random.seed(seed)
        random.shuffle(tasks)

    return tasks


def parse_tools_from_response(response: str) -> list[dict[str, str]]:
    """Extract tool definitions from model response."""
    tools = []
    import re

    tool_pattern = r"<tool>(.*?)</tool>"
    matches = re.findall(tool_pattern, response, re.DOTALL)

    for match in matches:
        tool = {}
        for line in match.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                tool[key.strip()] = value.strip()
        if tool:
            tools.append(tool)

    return tools


def parse_tool_usages(response: str) -> list[str]:
    """Extract tool usage references from response."""
    import re

    pattern = r'<use_tool\s+name="([^"]+)"\s*/>'
    return re.findall(pattern, response)


def format_task_prompt(task: TaskItem, task_type_context: str = "") -> str:
    """Format a task as a prompt."""
    if task.dataset_type == "cola":
        return f"""Task Type: Grammaticality Judgment
{task_type_context}
Text: "{task.text}"

Is this sentence grammatically acceptable? Answer with 1 (acceptable) or 0 (unacceptable)."""

    else:  # pii
        return f"""Task Type: PII Detection
{task_type_context}
Text: "{task.text}"

Identify all PII (Personally Identifiable Information) in the text. Return a JSON array of detected entities with their type, text, and position."""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average)."""
    return len(text) // 4


# === Output Management ===


class ExperimentOutputManager:
    """Manages experiment output directories and files."""

    def __init__(self, experiment_name: str, base_dir: str = "experiment_outputs"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{self.timestamp}"

        # Create directory structure
        self.base_path = Path(base_dir) / self.run_id
        self.tools_path = self.base_path / "tools"
        self.tools_safety_path = self.tools_path / "safety"
        self.tools_efficiency_path = self.tools_path / "efficiency"
        self.results_path = self.base_path / "results"
        self.reports_path = self.base_path / "reports"

        for path in [
            self.tools_safety_path,
            self.tools_efficiency_path,
            self.results_path,
            self.reports_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def save_tool(self, tool: ToolDefinition) -> Path:
        """Save a tool definition to the appropriate folder."""
        if tool.tool_type == "safety":
            folder = self.tools_safety_path
        else:
            folder = self.tools_efficiency_path

        # Sanitize tool name for filename
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in tool.name)
        filepath = folder / f"{safe_name}.json"

        with open(filepath, "w") as f:
            json.dump(tool.to_dict(), f, indent=2)

        # Also save the implementation as a separate text file for readability
        impl_path = folder / f"{safe_name}_implementation.txt"
        with open(impl_path, "w") as f:
            f.write(f"Tool: {tool.name}\n")
            f.write(f"Type: {tool.tool_type}\n")
            f.write(f"Description: {tool.description}\n")
            f.write(f"Created at task: {tool.created_at_task}\n")
            f.write(f"Creation tokens: {tool.creation_tokens}\n")
            f.write("\n--- Implementation ---\n")
            f.write(tool.implementation)

        return filepath

    def save_results(self, results: dict[str, Any], name: str) -> Path:
        """Save experiment results to JSON."""
        filepath = self.results_path / f"{name}.json"

        # Convert any non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, ToolDefinition):
                return obj.to_dict()
            elif isinstance(obj, TaskItem):
                return {
                    "text": obj.text,
                    "expected_answer": obj.expected_answer,
                    "dataset_type": obj.dataset_type,
                    "original_index": obj.original_index,
                    "metadata": obj.metadata,
                }
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            return obj

        serializable = make_serializable(results)

        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        return filepath

    def generate_pdf_report(self, all_results: dict[str, Any]) -> Path:
        """Generate a PDF report of the experiment results."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        filepath = self.reports_path / "experiment_report.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=18,
            spaceAfter=20,
        )
        story.append(Paragraph("Experiment 001: Recursive Tool Creation", title_style))
        story.append(Paragraph(f"Run ID: {self.run_id}", styles["Normal"]))
        story.append(
            Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.3 * inch))

        # Executive Summary
        story.append(Paragraph("Executive Summary", styles["Heading2"]))

        baseline = all_results.get("baseline", {})
        with_tools = all_results.get("with_tools", {})
        direct_lm = all_results.get("direct_lm", {})

        baseline_tokens = baseline.get("total_tokens", 0)
        tools_tokens = with_tools.get("total_tokens", 0)
        direct_tokens = direct_lm.get("total_tokens", 0)

        summary_data = [
            ["Metric", "Baseline", "With Tools", "Direct LM"],
            [
                "Total Tokens",
                f"{baseline_tokens:,}",
                f"{tools_tokens:,}",
                f"{direct_tokens:,}",
            ],
            [
                "Total Tasks",
                str(len(baseline.get("results", []))),
                str(len(with_tools.get("results", []))),
                str(len(direct_lm.get("results", []))),
            ],
        ]

        if baseline_tokens > 0 and tools_tokens > 0:
            savings = baseline_tokens - tools_tokens
            savings_pct = (savings / baseline_tokens) * 100
            summary_data.append(
                [
                    "Token Savings vs Baseline",
                    "-",
                    f"{savings:,} ({savings_pct:.1f}%)",
                    "-",
                ]
            )

        tools_created = with_tools.get("tools_created", [])
        summary_data.append(["Tools Created", "0", str(len(tools_created)), "0"])

        summary_table = Table(
            summary_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]
        )
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(summary_table)
        story.append(Spacer(1, 0.3 * inch))

        # Tools Created Section
        if tools_created:
            story.append(Paragraph("Tools Created", styles["Heading2"]))

            for tool in tools_created:
                if isinstance(tool, dict):
                    tool_name = tool.get("name", "Unknown")
                    tool_type = tool.get("type", "unknown")
                    tool_desc = tool.get("description", "No description")
                    tool_impl = tool.get("implementation", "")[:200]
                else:
                    tool_name = tool.name
                    tool_type = tool.tool_type
                    tool_desc = tool.description
                    tool_impl = tool.implementation[:200]

                story.append(
                    Paragraph(f"<b>{tool_name}</b> ({tool_type})", styles["Normal"])
                )
                story.append(Paragraph(f"Description: {tool_desc}", styles["Normal"]))
                story.append(
                    Paragraph(f"Implementation: {tool_impl}...", styles["Normal"])
                )
                story.append(Spacer(1, 0.1 * inch))

            story.append(Spacer(1, 0.2 * inch))

        # Token Breakdown by Mode
        story.append(Paragraph("Token Usage by Mode", styles["Heading2"]))

        modes = [
            ("Baseline (No Tools)", baseline),
            ("With Recursive Tools", with_tools),
            ("Direct LM", direct_lm),
        ]

        for mode_name, mode_results in modes:
            if mode_results:
                story.append(Paragraph(f"<b>{mode_name}</b>", styles["Normal"]))
                results_list = mode_results.get("results", [])
                if results_list:
                    total_input = sum(r.get("input_tokens", 0) for r in results_list)
                    total_output = sum(r.get("output_tokens", 0) for r in results_list)
                    story.append(
                        Paragraph(f"  Input tokens: {total_input:,}", styles["Normal"])
                    )
                    story.append(
                        Paragraph(
                            f"  Output tokens: {total_output:,}", styles["Normal"]
                        )
                    )
                    story.append(
                        Paragraph(
                            f"  Total: {total_input + total_output:,}", styles["Normal"]
                        )
                    )
                story.append(Spacer(1, 0.1 * inch))

        # Per-Task Analysis
        story.append(Paragraph("Per-Task Token Distribution", styles["Heading2"]))

        if baseline.get("results"):
            task_data = [["Task #", "Type", "Baseline", "With Tools", "Direct LM"]]
            baseline_results = baseline.get("results", [])
            tools_results = with_tools.get("results", [])
            direct_results = direct_lm.get("results", [])

            for i in range(min(10, len(baseline_results))):  # Show first 10
                b_tokens = (
                    baseline_results[i].get("input_tokens", 0)
                    + baseline_results[i].get("output_tokens", 0)
                    if i < len(baseline_results)
                    else 0
                )
                t_tokens = (
                    tools_results[i].get("input_tokens", 0)
                    + tools_results[i].get("output_tokens", 0)
                    if i < len(tools_results)
                    else 0
                )
                d_tokens = (
                    direct_results[i].get("input_tokens", 0)
                    + direct_results[i].get("output_tokens", 0)
                    if i < len(direct_results)
                    else 0
                )
                task_type = (
                    baseline_results[i].get("dataset_type", "unknown")
                    if i < len(baseline_results)
                    else "?"
                )

                task_data.append(
                    [str(i + 1), task_type, str(b_tokens), str(t_tokens), str(d_tokens)]
                )

            task_table = Table(
                task_data,
                colWidths=[0.7 * inch, 0.8 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch],
            )
            task_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(task_table)

        # Build PDF
        doc.build(story)
        return filepath


# === Main Experiment ===


class RecursiveToolExperiment:
    """
    Experiment runner for recursive tool creation.

    Tracks:
    - Tool creation and usage over time
    - Token savings from tool reuse
    - Baseline comparison (direct LLM without tools)
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        host: str = "http://localhost:11434",
        experiment_name: str = "recursive-tool-creation",
        output_manager: ExperimentOutputManager | None = None,
    ):
        self.model_name = model_name
        self.client = OllamaClient(model_name=model_name, host=host)
        self.tracker = ExperimentTracker(experiment_name)
        self.output_manager = output_manager

        # Tool registry
        self.tools: dict[str, ToolDefinition] = {}
        self.tool_usage_counts: dict[str, int] = {}

    def _build_system_with_tools(self) -> str:
        """Build system prompt including available tools."""
        if not self.tools:
            return RECURSIVE_TOOL_SYSTEM_PROMPT

        tools_section = "\n\nAVAILABLE TOOLS:\n"
        for name, tool in self.tools.items():
            tools_section += f"""
- {name} ({tool.tool_type}): {tool.description}
  Implementation: {tool.implementation}
"""
        return RECURSIVE_TOOL_SYSTEM_PROMPT + tools_section

    def run_direct_lm(
        self,
        tasks: list[TaskItem],
        run_name: str = "direct_lm",
    ) -> dict[str, Any]:
        """Run direct LM calls without any system prompt or guidance."""
        results = []

        with self.tracker.start_run(run_name=run_name, tags={"mode": "direct_lm"}):
            self.tracker.log_model_params(model_name=self.model_name, mode="direct_lm")
            self.tracker.log_dataset_info(
                "mixed_cola_pii",
                "train",
                indices=[t.original_index for t in tasks],
                size=len(tasks),
            )

            for i, task in enumerate(tqdm(tasks, desc="Direct LM", unit="task")):
                # Minimal prompt - just the task
                if task.dataset_type == "cola":
                    prompt = f'Is this sentence grammatically correct? Answer 1 for yes, 0 for no: "{task.text}"'
                else:
                    prompt = f'List any PII in this text as JSON: "{task.text}"'

                response = self.client.completion(prompt)
                usage = self.client.get_last_usage()

                input_tokens = usage.total_input_tokens
                output_tokens = usage.total_output_tokens

                self.tracker.track_call(
                    CallType.NO_RULE,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    dataset_index=i,
                )

                results.append(
                    {
                        "task_index": i,
                        "dataset_type": task.dataset_type,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "response": response,
                        "expected": task.expected_answer,
                    }
                )

        return {
            "results": results,
            "total_tokens": sum(
                r["input_tokens"] + r["output_tokens"] for r in results
            ),
            "summary": self.tracker.get_summary(),
        }

    def run_baseline(
        self,
        tasks: list[TaskItem],
        run_name: str = "baseline",
    ) -> dict[str, Any]:
        """Run baseline with system prompt but no tool creation."""
        results = []

        with self.tracker.start_run(run_name=run_name, tags={"mode": "baseline"}):
            self.tracker.log_model_params(model_name=self.model_name, mode="baseline")
            self.tracker.log_dataset_info(
                "mixed_cola_pii",
                "train",
                indices=[t.original_index for t in tasks],
                size=len(tasks),
            )

            for i, task in enumerate(tqdm(tasks, desc="Baseline", unit="task")):
                prompt = format_task_prompt(task)

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer the task accurately and concisely.",
                    },
                    {"role": "user", "content": prompt},
                ]

                response = self.client.completion(messages)
                usage = self.client.get_last_usage()

                input_tokens = usage.total_input_tokens
                output_tokens = usage.total_output_tokens

                self.tracker.track_call(
                    CallType.NO_RULE,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    dataset_index=i,
                )

                results.append(
                    {
                        "task_index": i,
                        "dataset_type": task.dataset_type,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "response": response,
                        "expected": task.expected_answer,
                    }
                )

        return {
            "results": results,
            "total_tokens": sum(
                r["input_tokens"] + r["output_tokens"] for r in results
            ),
            "summary": self.tracker.get_summary(),
        }

    def run_with_tools(
        self,
        tasks: list[TaskItem],
        run_name: str = "with_tools",
    ) -> dict[str, Any]:
        """Run with recursive tool creation enabled."""
        results = []
        self.tools = {}
        self.tool_usage_counts = {}

        with self.tracker.start_run(
            run_name=run_name, tags={"mode": "recursive_tools"}
        ):
            self.tracker.log_model_params(
                model_name=self.model_name,
                mode="recursive_tools",
                system_prompt_length=len(RECURSIVE_TOOL_SYSTEM_PROMPT),
            )
            self.tracker.log_dataset_info(
                "mixed_cola_pii",
                "train",
                indices=[t.original_index for t in tasks],
                size=len(tasks),
            )

            pbar = tqdm(
                enumerate(tasks), total=len(tasks), desc="With Tools", unit="task"
            )
            for i, task in pbar:
                # Build context with available tools
                system_prompt = self._build_system_with_tools()
                task_context = ""
                if self.tools:
                    task_context = f"(You have {len(self.tools)} tools available. Use them if appropriate.)"

                prompt = format_task_prompt(task, task_context)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

                response = self.client.completion(messages)
                usage = self.client.get_last_usage()

                input_tokens = usage.total_input_tokens
                output_tokens = usage.total_output_tokens

                # Parse any new tools created
                new_tools = parse_tools_from_response(response)
                tools_created_this_task = []

                for tool_def in new_tools:
                    if "name" in tool_def:
                        tool = ToolDefinition(
                            name=tool_def["name"],
                            tool_type=tool_def.get("type", "unknown"),
                            description=tool_def.get("description", ""),
                            implementation=tool_def.get("implementation", ""),
                            created_at_task=i,
                            creation_tokens=output_tokens,
                        )
                        self.tools[tool.name] = tool
                        self.tool_usage_counts[tool.name] = 0
                        tools_created_this_task.append(tool.name)

                        # Save tool to folder
                        if self.output_manager:
                            self.output_manager.save_tool(tool)

                        # Track rule creation
                        self.tracker.track_rule_creation_call(
                            rule_id=tool.name,
                            rule_text=tool.implementation,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            dataset_index=i,
                        )

                # Parse tool usages
                tools_used = parse_tool_usages(response)
                for tool_name in tools_used:
                    if tool_name in self.tool_usage_counts:
                        self.tool_usage_counts[tool_name] += 1
                        self.tracker.record_rule_usage(tool_name)

                # Track call type based on what happened
                if tools_created_this_task:
                    call_type = CallType.RULE_CREATION
                    rule_tokens = output_tokens
                elif tools_used:
                    call_type = CallType.RULE_USAGE
                    rule_tokens = sum(
                        estimate_tokens(self.tools[t].implementation)
                        for t in tools_used
                        if t in self.tools
                    )
                else:
                    call_type = CallType.NO_RULE
                    rule_tokens = 0

                if not tools_created_this_task:
                    self.tracker.track_call(
                        call_type,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        rule_tokens=rule_tokens,
                        dataset_index=i,
                    )

                results.append(
                    {
                        "task_index": i,
                        "dataset_type": task.dataset_type,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tools_created": tools_created_this_task,
                        "tools_used": tools_used,
                        "response": response,
                        "expected": task.expected_answer,
                    }
                )

                pbar.set_postfix(
                    tools=len(self.tools), tokens=input_tokens + output_tokens
                )

        return {
            "results": results,
            "total_tokens": sum(
                r["input_tokens"] + r["output_tokens"] for r in results
            ),
            "tools_created": [t.to_dict() for t in self.tools.values()],
            "tool_usage_counts": self.tool_usage_counts.copy(),
            "summary": self.tracker.get_summary(),
        }

    def run_throwaway_token_measurement(
        self,
        tasks: list[TaskItem],
        run_name: str = "throwaway_measurement",
    ) -> dict[str, Any]:
        """
        Run sub-LLM calls to measure potential token savings.
        Simulates lightweight classifier calls.
        """
        results = []

        with self.tracker.start_run(
            run_name=run_name, tags={"mode": "throwaway_measurement"}
        ):
            self.tracker.log_model_params(
                model_name=self.model_name, mode="throwaway_measurement"
            )

            for i, task in enumerate(tqdm(tasks, desc="Classifier", unit="task")):
                classifier_prompt = (
                    f"Classify in one word (cola/pii/other): '{task.text[:100]}'"
                )

                messages = [
                    {"role": "system", "content": "Respond with only one word."},
                    {"role": "user", "content": classifier_prompt},
                ]

                response = self.client.completion(messages)
                usage = self.client.get_last_usage()

                results.append(
                    {
                        "task_index": i,
                        "classification_tokens": usage.total_input_tokens
                        + usage.total_output_tokens,
                        "response": response,
                    }
                )

                self.tracker.track_call(
                    CallType.NO_RULE,
                    input_tokens=usage.total_input_tokens,
                    output_tokens=usage.total_output_tokens,
                    dataset_index=i,
                )

        return {
            "results": results,
            "total_classifier_tokens": sum(r["classification_tokens"] for r in results),
            "avg_classifier_tokens": sum(r["classification_tokens"] for r in results)
            / len(results)
            if results
            else 0,
        }


def run_full_experiment(
    model_name: str = "llama3.2",
    cola_limit: int = 20,
    pii_limit: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the complete experiment with all modes."""
    print("=" * 60)
    print("Experiment 001: Recursive Tool Creation")
    print("=" * 60)

    # Initialize output manager
    output_manager = ExperimentOutputManager("exp001")

    # Load mixed dataset
    print("\nLoading mixed dataset...")
    tasks = load_mixed_dataset(
        cola_limit=cola_limit,
        pii_limit=pii_limit,
        seed=seed,
    )
    print(f"Loaded {len(tasks)} tasks (CoLA + PII)")

    # Initialize experiment
    experiment = RecursiveToolExperiment(
        model_name=model_name,
        output_manager=output_manager,
    )

    # Run direct LM (simplest baseline)
    print("\n" + "-" * 40)
    print("Running DIRECT LM (minimal prompt)...")
    print("-" * 40)
    direct_results = experiment.run_direct_lm(tasks, run_name="exp001_direct_lm")

    # Run baseline with system prompt
    print("\n" + "-" * 40)
    print("Running BASELINE (system prompt, no tools)...")
    print("-" * 40)
    baseline_results = experiment.run_baseline(tasks, run_name="exp001_baseline")

    # Run with tools
    print("\n" + "-" * 40)
    print("Running WITH TOOLS (recursive creation)...")
    print("-" * 40)
    tools_results = experiment.run_with_tools(tasks, run_name="exp001_with_tools")

    # Run throwaway measurement
    print("\n" + "-" * 40)
    print("Running THROWAWAY TOKEN MEASUREMENT...")
    print("-" * 40)
    throwaway_results = experiment.run_throwaway_token_measurement(
        tasks, run_name="exp001_throwaway"
    )

    # Compile all results
    all_results = {
        "direct_lm": direct_results,
        "baseline": baseline_results,
        "with_tools": tools_results,
        "throwaway": throwaway_results,
        "metadata": {
            "model_name": model_name,
            "cola_limit": cola_limit,
            "pii_limit": pii_limit,
            "seed": seed,
            "total_tasks": len(tasks),
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)

    output_manager.save_results(direct_results, "direct_lm_results")
    output_manager.save_results(baseline_results, "baseline_results")
    output_manager.save_results(tools_results, "with_tools_results")
    output_manager.save_results(throwaway_results, "throwaway_results")
    output_manager.save_results(all_results, "all_results")

    # Generate PDF report
    print("Generating PDF report...")
    pdf_path = output_manager.generate_pdf_report(all_results)
    print(f"PDF report saved to: {pdf_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\nOutput directory: {output_manager.base_path}")
    print(f"Total tasks: {len(tasks)}")
    print(f"\nDirect LM total tokens: {direct_results['total_tokens']:,}")
    print(f"Baseline total tokens: {baseline_results['total_tokens']:,}")
    print(f"With tools total tokens: {tools_results['total_tokens']:,}")
    print(
        f"Throwaway classifier tokens: {throwaway_results['total_classifier_tokens']:,}"
    )

    if tools_results["tools_created"]:
        print(f"\nTools created: {len(tools_results['tools_created'])}")
        for tool in tools_results["tools_created"]:
            usages = tools_results["tool_usage_counts"].get(tool["name"], 0)
            print(f"  - {tool['name']} ({tool['type']}): used {usages} times")

    token_diff = baseline_results["total_tokens"] - tools_results["total_tokens"]
    if token_diff > 0:
        print(
            f"\nToken savings with tools vs baseline: {token_diff:,} ({token_diff / baseline_results['total_tokens'] * 100:.1f}%)"
        )
    else:
        print(
            f"\nToken overhead with tools vs baseline: {-token_diff:,} ({-token_diff / baseline_results['total_tokens'] * 100:.1f}%)"
        )

    print(f"\nPDF Report: {pdf_path}")
    print("MLflow UI: http://localhost:5000")

    return {
        "all_results": all_results,
        "output_path": str(output_manager.base_path),
        "pdf_path": str(pdf_path),
        "tasks": tasks,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run recursive tool creation experiment"
    )
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument(
        "--cola-limit", type=int, default=20, help="Number of CoLA samples"
    )
    parser.add_argument(
        "--pii-limit", type=int, default=10, help="Number of PII samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_full_experiment(
        model_name=args.model,
        cola_limit=args.cola_limit,
        pii_limit=args.pii_limit,
        seed=args.seed,
    )
