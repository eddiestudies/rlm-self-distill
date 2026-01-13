"""
Experiment 002: Scaling Analysis for Recursive Tool Creation

This experiment tests how tool creation costs amortize as dataset size increases.
Runs iteratively at 10x, 100x, 1000x scale to find the break-even point.

Generates:
- PDF report after each batch
- Joint summary PDF at the end
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from self_distill import (
    DATA,
    ExperimentTracker,
    OllamaClient,
    load_dataset,
)

# Import from exp001
from experiments.exp001_recursive_tool_creation import (
    RECURSIVE_TOOL_SYSTEM_PROMPT,
    TaskItem,
    ToolDefinition,
    format_task_prompt,
    parse_tool_usages,
    parse_tools_from_response,
)


def load_mixed_dataset_scaled(
    cola_count: int,
    pii_count: int,
    shuffle: bool = True,
    seed: int = 42,
) -> list[TaskItem]:
    """Load and mix CoLA and PII datasets with specified counts."""
    tasks = []

    # Load CoLA - cycle through if needed
    cola_data = load_dataset(DATA.COLA, "train")
    cola_items = list(cola_data)

    for i in range(cola_count):
        item = cola_items[i % len(cola_items)]
        tasks.append(
            TaskItem(
                text=item.question,
                expected_answer=item.answer,
                dataset_type="cola",
                original_index=i,
                metadata={"task": "grammaticality_judgment"},
            )
        )

    # Load PII Detection - cycle through if needed
    pii_data = load_dataset(DATA.PII_DETECTION, "train")
    pii_items = list(pii_data)

    for i in range(pii_count):
        item = pii_items[i % len(pii_items)]
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


@dataclass
class BatchResult:
    """Results from a single batch run."""

    batch_name: str
    scale: int
    total_tasks: int

    direct_lm_tokens: int
    baseline_tokens: int
    with_tools_tokens: int

    tools_created: int
    tool_usages: int

    direct_lm_results: list[dict]
    baseline_results: list[dict]
    with_tools_results: list[dict]
    tools: list[dict]
    tool_usage_counts: dict[str, int]

    @property
    def tokens_per_task_direct(self) -> float:
        return self.direct_lm_tokens / self.total_tasks if self.total_tasks > 0 else 0

    @property
    def tokens_per_task_baseline(self) -> float:
        return self.baseline_tokens / self.total_tasks if self.total_tasks > 0 else 0

    @property
    def tokens_per_task_tools(self) -> float:
        return self.with_tools_tokens / self.total_tasks if self.total_tasks > 0 else 0

    @property
    def overhead_vs_baseline(self) -> float:
        if self.baseline_tokens > 0:
            return (
                (self.with_tools_tokens - self.baseline_tokens) / self.baseline_tokens
            ) * 100
        return 0

    @property
    def overhead_vs_direct(self) -> float:
        if self.direct_lm_tokens > 0:
            return (
                (self.with_tools_tokens - self.direct_lm_tokens) / self.direct_lm_tokens
            ) * 100
        return 0


class ScalingExperiment:
    """Experiment runner for scaling analysis."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        experiment_name: str = "scaling-analysis",
        output_dir: str = "experiment_outputs",
    ):
        self.model_name = model_name
        self.client = OllamaClient(model_name=model_name, host=host)
        self.tracker = ExperimentTracker(experiment_name)
        self.output_dir = Path(output_dir)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"exp002_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Accumulated results across batches
        self.batch_results: list[BatchResult] = []

        # Persistent tools across batches (for tools mode)
        self.tools: dict[str, ToolDefinition] = {}
        self.tool_usage_counts: dict[str, int] = {}

    def run_direct_lm_batch(self, tasks: list[TaskItem], batch_name: str) -> list[dict]:
        """Run direct LM on a batch."""
        results = []

        for task in tqdm(tasks, desc="  Direct LM", unit="task"):
            if task.dataset_type == "cola":
                prompt = f'Is this sentence grammatically correct? Answer 1 for yes, 0 for no: "{task.text}"'
            else:
                prompt = f'List any PII in this text as JSON: "{task.text}"'

            response = self.client.completion(prompt)
            usage = self.client.get_last_usage()

            results.append(
                {
                    "task_index": len(results),
                    "dataset_type": task.dataset_type,
                    "input_tokens": usage.total_input_tokens,
                    "output_tokens": usage.total_output_tokens,
                    "response": response[:200],  # Truncate for storage
                }
            )

        return results

    def run_baseline_batch(self, tasks: list[TaskItem], batch_name: str) -> list[dict]:
        """Run baseline on a batch."""
        results = []

        for task in tqdm(tasks, desc="  Baseline", unit="task"):
            prompt = format_task_prompt(task)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer accurately and concisely.",
                },
                {"role": "user", "content": prompt},
            ]

            response = self.client.completion(messages)
            usage = self.client.get_last_usage()

            results.append(
                {
                    "task_index": len(results),
                    "dataset_type": task.dataset_type,
                    "input_tokens": usage.total_input_tokens,
                    "output_tokens": usage.total_output_tokens,
                    "response": response[:200],
                }
            )

        return results

    def _build_system_with_tools(self) -> str:
        """Build system prompt including available tools."""
        if not self.tools:
            return RECURSIVE_TOOL_SYSTEM_PROMPT

        tools_section = "\n\nAVAILABLE TOOLS:\n"
        for name, tool in self.tools.items():
            tools_section += f"- {name} ({tool.tool_type}): {tool.description}\n"
        return RECURSIVE_TOOL_SYSTEM_PROMPT + tools_section

    def run_with_tools_batch(
        self, tasks: list[TaskItem], batch_name: str
    ) -> tuple[list[dict], list[dict]]:
        """Run with tools on a batch. Returns (results, new_tools)."""
        results = []
        new_tools = []

        pbar = tqdm(
            enumerate(tasks), total=len(tasks), desc="  With Tools", unit="task"
        )
        for i, task in pbar:
            system_prompt = self._build_system_with_tools()
            task_context = (
                f"(You have {len(self.tools)} tools available.)" if self.tools else ""
            )
            prompt = format_task_prompt(task, task_context)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            response = self.client.completion(messages)
            usage = self.client.get_last_usage()

            # Parse new tools
            parsed_tools = parse_tools_from_response(response)
            tools_created = []

            for tool_def in parsed_tools:
                if "name" in tool_def and tool_def["name"] not in self.tools:
                    tool = ToolDefinition(
                        name=tool_def["name"],
                        tool_type=tool_def.get("type", "unknown"),
                        description=tool_def.get("description", ""),
                        implementation=tool_def.get("implementation", ""),
                        created_at_task=i,
                        creation_tokens=usage.total_output_tokens,
                    )
                    self.tools[tool.name] = tool
                    self.tool_usage_counts[tool.name] = 0
                    tools_created.append(tool.name)
                    new_tools.append(tool.to_dict())

            # Parse tool usages
            tools_used = parse_tool_usages(response)
            for tool_name in tools_used:
                if tool_name in self.tool_usage_counts:
                    self.tool_usage_counts[tool_name] += 1

            results.append(
                {
                    "task_index": i,
                    "dataset_type": task.dataset_type,
                    "input_tokens": usage.total_input_tokens,
                    "output_tokens": usage.total_output_tokens,
                    "tools_created": tools_created,
                    "tools_used": tools_used,
                    "response": response[:200],
                }
            )

            pbar.set_postfix(tools=len(self.tools))

        return results, new_tools

    def run_batch(self, scale: int, cola_count: int, pii_count: int) -> BatchResult:
        """Run a complete batch at the given scale."""
        batch_name = f"scale_{scale}x"
        total_tasks = cola_count + pii_count

        print(f"\n{'=' * 60}")
        print(f"BATCH: {batch_name} ({total_tasks} tasks)")
        print(f"{'=' * 60}")

        # Load data
        print(f"\nLoading {cola_count} CoLA + {pii_count} PII samples...")
        tasks = load_mixed_dataset_scaled(cola_count, pii_count)

        # Run direct LM
        print("\nRunning Direct LM...")
        direct_results = self.run_direct_lm_batch(tasks, batch_name)
        direct_tokens = sum(
            r["input_tokens"] + r["output_tokens"] for r in direct_results
        )

        # Run baseline
        print("\nRunning Baseline...")
        baseline_results = self.run_baseline_batch(tasks, batch_name)
        baseline_tokens = sum(
            r["input_tokens"] + r["output_tokens"] for r in baseline_results
        )

        # Run with tools (tools persist across batches!)
        print("\nRunning With Tools...")
        tools_results, new_tools = self.run_with_tools_batch(tasks, batch_name)
        tools_tokens = sum(
            r["input_tokens"] + r["output_tokens"] for r in tools_results
        )

        # Compile batch result
        result = BatchResult(
            batch_name=batch_name,
            scale=scale,
            total_tasks=total_tasks,
            direct_lm_tokens=direct_tokens,
            baseline_tokens=baseline_tokens,
            with_tools_tokens=tools_tokens,
            tools_created=len(new_tools),
            tool_usages=sum(self.tool_usage_counts.values()),
            direct_lm_results=direct_results,
            baseline_results=baseline_results,
            with_tools_results=tools_results,
            tools=[t.to_dict() for t in self.tools.values()],
            tool_usage_counts=self.tool_usage_counts.copy(),
        )

        self.batch_results.append(result)

        # Print summary
        print("\n--- Batch Summary ---")
        print(
            f"Direct LM: {direct_tokens:,} tokens ({result.tokens_per_task_direct:.1f}/task)"
        )
        print(
            f"Baseline: {baseline_tokens:,} tokens ({result.tokens_per_task_baseline:.1f}/task)"
        )
        print(
            f"With Tools: {tools_tokens:,} tokens ({result.tokens_per_task_tools:.1f}/task)"
        )
        print(f"Overhead vs Baseline: {result.overhead_vs_baseline:+.1f}%")
        print(f"Total tools: {len(self.tools)}, New tools this batch: {len(new_tools)}")

        return result

    def generate_batch_pdf(self, result: BatchResult) -> Path:
        """Generate PDF report for a single batch."""
        filepath = self.run_dir / f"report_{result.batch_name}.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "Title", parent=styles["Heading1"], fontSize=16, spaceAfter=12
        )
        story.append(
            Paragraph(
                f"Experiment 002: Scaling Analysis - {result.batch_name}", title_style
            )
        )
        story.append(
            Paragraph(
                f"Scale: {result.scale}x | Tasks: {result.total_tasks}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.2 * inch))

        # Summary table
        summary_data = [
            ["Metric", "Direct LM", "Baseline", "With Tools"],
            [
                "Total Tokens",
                f"{result.direct_lm_tokens:,}",
                f"{result.baseline_tokens:,}",
                f"{result.with_tools_tokens:,}",
            ],
            [
                "Tokens/Task",
                f"{result.tokens_per_task_direct:.1f}",
                f"{result.tokens_per_task_baseline:.1f}",
                f"{result.tokens_per_task_tools:.1f}",
            ],
            ["Overhead vs Baseline", "-", "-", f"{result.overhead_vs_baseline:+.1f}%"],
            ["Tools Created", "0", "0", str(result.tools_created)],
        ]

        table = Table(
            summary_data, colWidths=[1.8 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch]
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        # Tools summary
        if result.tools:
            story.append(Paragraph("Tools Available", styles["Heading2"]))
            for tool in result.tools:
                story.append(
                    Paragraph(
                        f"<b>{tool['name']}</b> ({tool['type']}): {tool['description'][:100]}...",
                        styles["Normal"],
                    )
                )
            story.append(Spacer(1, 0.1 * inch))

        doc.build(story)
        return filepath

    def generate_joint_pdf(self) -> Path:
        """Generate comprehensive PDF with all batch results."""
        filepath = self.run_dir / "report_joint_summary.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "Title", parent=styles["Heading1"], fontSize=18, spaceAfter=20
        )
        story.append(
            Paragraph("Experiment 002: Scaling Analysis - Joint Summary", title_style)
        )
        story.append(Paragraph(f"Run ID: exp002_{self.timestamp}", styles["Normal"]))
        story.append(Paragraph(f"Model: {self.model_name}", styles["Normal"]))
        story.append(Spacer(1, 0.3 * inch))

        # Scaling comparison table
        story.append(Paragraph("Scaling Comparison", styles["Heading2"]))

        header = ["Scale", "Tasks", "Direct LM", "Baseline", "With Tools", "Overhead %"]
        data = [header]

        for r in self.batch_results:
            data.append(
                [
                    f"{r.scale}x",
                    str(r.total_tasks),
                    f"{r.direct_lm_tokens:,}",
                    f"{r.baseline_tokens:,}",
                    f"{r.with_tools_tokens:,}",
                    f"{r.overhead_vs_baseline:+.1f}%",
                ]
            )

        table = Table(
            data,
            colWidths=[
                0.7 * inch,
                0.7 * inch,
                1.2 * inch,
                1.2 * inch,
                1.2 * inch,
                1 * inch,
            ],
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.3 * inch))

        # Tokens per task comparison
        story.append(Paragraph("Tokens Per Task by Scale", styles["Heading2"]))

        header2 = [
            "Scale",
            "Direct/Task",
            "Baseline/Task",
            "Tools/Task",
            "Tool Overhead/Task",
        ]
        data2 = [header2]

        for r in self.batch_results:
            overhead_per_task = r.tokens_per_task_tools - r.tokens_per_task_baseline
            data2.append(
                [
                    f"{r.scale}x",
                    f"{r.tokens_per_task_direct:.1f}",
                    f"{r.tokens_per_task_baseline:.1f}",
                    f"{r.tokens_per_task_tools:.1f}",
                    f"{overhead_per_task:+.1f}",
                ]
            )

        table2 = Table(
            data2,
            colWidths=[0.8 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.4 * inch],
        )
        table2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(table2)
        story.append(Spacer(1, 0.3 * inch))

        # Tool accumulation
        story.append(Paragraph("Tool Accumulation Across Batches", styles["Heading2"]))

        cumulative_tools = 0
        tool_data = [["Batch", "New Tools", "Cumulative Tools", "Total Usages"]]
        cumulative_usages = 0

        for r in self.batch_results:
            cumulative_tools += r.tools_created
            cumulative_usages = r.tool_usages
            tool_data.append(
                [
                    r.batch_name,
                    str(r.tools_created),
                    str(cumulative_tools),
                    str(cumulative_usages),
                ]
            )

        tool_table = Table(
            tool_data, colWidths=[1.5 * inch, 1.2 * inch, 1.5 * inch, 1.2 * inch]
        )
        tool_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(tool_table)
        story.append(Spacer(1, 0.3 * inch))

        # Final tools list
        story.append(PageBreak())
        story.append(Paragraph("All Tools Created", styles["Heading2"]))

        if self.batch_results and self.batch_results[-1].tools:
            for tool in self.batch_results[-1].tools:
                usages = self.batch_results[-1].tool_usage_counts.get(tool["name"], 0)
                story.append(
                    Paragraph(
                        f"<b>{tool['name']}</b> ({tool['type']}) - Used {usages} times",
                        styles["Normal"],
                    )
                )
                story.append(Paragraph(f"  {tool['description']}", styles["Normal"]))
                story.append(Spacer(1, 0.05 * inch))

        # Conclusions
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Analysis", styles["Heading2"]))

        if self.batch_results:
            first = self.batch_results[0]
            last = self.batch_results[-1]

            if last.overhead_vs_baseline < first.overhead_vs_baseline:
                trend = "decreasing (tools becoming more efficient)"
            else:
                trend = "increasing (tools not amortizing)"

            story.append(
                Paragraph(
                    f"Overhead trend: {first.overhead_vs_baseline:+.1f}% → {last.overhead_vs_baseline:+.1f}% ({trend})",
                    styles["Normal"],
                )
            )

            total_tool_tokens = sum(t.get("creation_tokens", 0) for t in last.tools)
            story.append(
                Paragraph(
                    f"Total tool creation overhead: ~{total_tool_tokens:,} tokens",
                    styles["Normal"],
                )
            )

        doc.build(story)
        return filepath

    def save_results(self) -> Path:
        """Save all results to JSON."""
        results_file = self.run_dir / "all_results.json"

        data = {
            "metadata": {
                "model_name": self.model_name,
                "timestamp": self.timestamp,
                "batches": len(self.batch_results),
            },
            "batches": [
                {
                    "batch_name": r.batch_name,
                    "scale": r.scale,
                    "total_tasks": r.total_tasks,
                    "direct_lm_tokens": r.direct_lm_tokens,
                    "baseline_tokens": r.baseline_tokens,
                    "with_tools_tokens": r.with_tools_tokens,
                    "tokens_per_task_direct": r.tokens_per_task_direct,
                    "tokens_per_task_baseline": r.tokens_per_task_baseline,
                    "tokens_per_task_tools": r.tokens_per_task_tools,
                    "overhead_vs_baseline": r.overhead_vs_baseline,
                    "tools_created": r.tools_created,
                    "tool_usages": r.tool_usages,
                }
                for r in self.batch_results
            ],
            "final_tools": self.batch_results[-1].tools if self.batch_results else [],
            "final_tool_usages": self.batch_results[-1].tool_usage_counts
            if self.batch_results
            else {},
        }

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

        return results_file


def run_scaling_experiment(
    model_name: str = "llama3.2:3b",
    base_cola: int = 10,
    base_pii: int = 5,
    scales: list[int] = [1, 10],  # 1x, 10x (100x takes too long)
) -> dict[str, Any]:
    """Run the complete scaling experiment."""
    print("=" * 60)
    print("Experiment 002: Scaling Analysis")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(
        f"Base size: {base_cola} CoLA + {base_pii} PII = {base_cola + base_pii} tasks"
    )
    print(f"Scales: {scales}")

    experiment = ScalingExperiment(model_name=model_name)

    # Run each scale
    for scale in tqdm(scales, desc="Scales", unit="scale"):
        cola_count = base_cola * scale
        pii_count = base_pii * scale

        result = experiment.run_batch(scale, cola_count, pii_count)

        # Generate PDF for this batch
        pdf_path = experiment.generate_batch_pdf(result)
        print(f"\nBatch PDF: {pdf_path}")

    # Generate joint summary
    print("\n" + "=" * 60)
    print("Generating Joint Summary...")
    print("=" * 60)

    joint_pdf = experiment.generate_joint_pdf()
    results_file = experiment.save_results()

    print(f"\nJoint PDF: {joint_pdf}")
    print(f"Results JSON: {results_file}")
    print(f"Output directory: {experiment.run_dir}")

    return {
        "run_dir": str(experiment.run_dir),
        "joint_pdf": str(joint_pdf),
        "batch_results": experiment.batch_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run scaling analysis experiment")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--base-cola", type=int, default=10, help="Base CoLA count")
    parser.add_argument("--base-pii", type=int, default=5, help="Base PII count")
    parser.add_argument(
        "--scales",
        type=str,
        default="1,10",
        help="Comma-separated scales (e.g., 1,10,100)",
    )

    args = parser.parse_args()
    scales = [int(s) for s in args.scales.split(",")]

    run_scaling_experiment(
        model_name=args.model,
        base_cola=args.base_cola,
        base_pii=args.base_pii,
        scales=scales,
    )
