#!/usr/bin/env python3
"""
Experiment 005: RLM-Based Tool Creation

This experiment uses the SelfDistillRLM class which leverages the RLM framework
to have the inner LLM create and manage tools, rather than us hardcoding the
tool creation logic.

Key features:
- LLM creates tools via RLM's REPL environment
- Tools are saved to disk and tracked
- Safety macros are created by the LLM
- PDF report shows all tools/macros with their code
"""

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Preformatted,
    PageBreak,
)

from self_distill.clients.ollama_client import OllamaClient
from self_distill.datasets import DATA, load_dataset, Split
from self_distill.rlm import SelfDistillRLM


def seed_tools(tools_dir: Path):
    """Create initial hook and replacement tools for testing.

    Simplified architecture:
    - Hooks return bool (True = use replacement, False = use LLM)
    - Hook and replacement names must MATCH exactly
    """
    # Create directories
    (tools_dir / "pre_completion").mkdir(parents=True, exist_ok=True)
    (tools_dir / "replacements").mkdir(parents=True, exist_ok=True)
    (tools_dir / "utilities").mkdir(parents=True, exist_ok=True)

    # PII detector hook - returns True if PII found
    pii_hook = '''# Detects PII in text - returns bool
import re

def check(text: str) -> bool:
    """Return True if text contains PII patterns."""
    patterns = [
        r'[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}',  # email
        r'\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}',  # phone
        r'\\d{3}-\\d{2}-\\d{4}'  # SSN
    ]
    return any(re.search(p, text) for p in patterns)
'''
    (tools_dir / "pre_completion" / "pii_detector.py").write_text(pii_hook)

    # Grammar checker hook - returns True if grammar task detected
    grammar_hook = '''# Detects grammar tasks - returns bool

def check(text: str) -> bool:
    """Return True if this is a grammar acceptability task."""
    keywords = ['grammatical', 'grammar', 'acceptable', 'sentence']
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)
'''
    (tools_dir / "pre_completion" / "grammar_checker.py").write_text(grammar_hook)

    # PII detector replacement - matching name!
    pii_replacement = '''# Analyzes PII - replaces LLM call
import re

def run(text: str) -> dict:
    """Detect and report PII in text."""
    found = []
    if re.search(r'[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}', text):
        found.append('email')
    if re.search(r'\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}', text):
        found.append('phone')
    if re.search(r'\\d{3}-\\d{2}-\\d{4}', text):
        found.append('ssn')
    return {
        'has_pii': len(found) > 0,
        'pii_types': found,
        'recommendation': 'Mask PII before processing' if found else 'No PII detected'
    }
'''
    (tools_dir / "replacements" / "pii_detector.py").write_text(pii_replacement)

    # Grammar checker replacement - matching name!
    grammar_replacement = '''# Checks grammar - replaces LLM call
import re

def run(text: str) -> dict:
    """Check if a sentence is grammatically acceptable."""
    # Extract the actual sentence from the prompt
    sentence = text
    if '"' in text:
        match = re.search(r'"([^"]+)"', text)
        if match:
            sentence = match.group(1)
    elif 'Analyze this text:' in text:
        sentence = text.split('Analyze this text:')[-1].strip()

    issues = []
    if sentence and not sentence[0].isupper():
        issues.append('Should start with capital letter')
    if sentence and not sentence.rstrip().endswith(('.', '!', '?')):
        issues.append('Should end with punctuation')

    return {'acceptable': len(issues) == 0, 'issues': issues}
'''
    (tools_dir / "replacements" / "grammar_checker.py").write_text(grammar_replacement)

    # PII masker utility
    pii_masker = '''# Masks PII in text
import re

def run(text: str) -> str:
    """Mask PII patterns in text."""
    text = re.sub(r'[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}', '[EMAIL]', text)
    text = re.sub(r'\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}', '[PHONE]', text)
    text = re.sub(r'\\d{3}-\\d{2}-\\d{4}', '[SSN]', text)
    return text
'''
    (tools_dir / "utilities" / "pii_masker.py").write_text(pii_masker)

    print(f"Seeded tools directory with:")
    print(f"  - pre_completion/pii_detector.py (check -> bool)")
    print(f"  - pre_completion/grammar_checker.py (check -> bool)")
    print(f"  - replacements/pii_detector.py (run -> dict)")
    print(f"  - replacements/grammar_checker.py (run -> dict)")
    print(f"  - utilities/pii_masker.py (run -> str)")


@dataclass
class ToolRecord:
    """Record of a tool created by the LLM."""

    name: str
    file_path: str
    description: str
    code: str
    created_at: str
    used_count: int = 0


@dataclass
class MacroRecord:
    """Record of a macro/code block executed by the LLM."""

    task_index: int
    task_type: str
    code: str
    output: str
    timestamp: str


@dataclass
class ExperimentResults:
    """Results from the experiment."""

    tools: list[ToolRecord] = field(default_factory=list)
    macros: list[MacroRecord] = field(default_factory=list)
    baseline_tokens: int = 0
    rlm_tokens: int = 0
    tasks_processed: int = 0
    tool_executions: int = 0
    llm_fallbacks: int = 0


def run_baseline(client: OllamaClient, model: str, tasks: list[dict]) -> int:
    """Run baseline LLM on all tasks."""
    total_tokens = 0

    for task in tqdm(tasks, desc="Baseline", unit="task"):
        prompt = f"""Analyze the following text:

Text: {task["text"]}

Task: {task["task_type"]}

Provide your analysis."""

        _ = client.completion(prompt, model)
        usage = client.get_last_usage()
        total_tokens += usage.total_input_tokens + usage.total_output_tokens

    return total_tokens


def run_with_rlm(
    rlm: SelfDistillRLM, tasks: list[dict], results: ExperimentResults
) -> int:
    """Run tasks using SelfDistillRLM."""
    total_tokens = 0

    pbar = tqdm(enumerate(tasks), total=len(tasks), desc="RLM", unit="task")
    for i, task in pbar:
        prompt = f"""Task Type: {task["task_type"]}

Analyze this text: {task["text"]}

If this is a grammar task, check if the sentence is grammatically acceptable.
If this is a PII task, detect any personally identifiable information.

First check if you have an existing tool for this task type.
If not, create one and save it for reuse.
If dealing with PII, apply the safety macro pattern first."""

        try:
            response = rlm.completion(prompt)

            # Track tokens from response (RLM uses usage_summary.model_usage_summaries)
            if hasattr(response, "usage_summary") and response.usage_summary:
                for (
                    model_name,
                    usage,
                ) in response.usage_summary.model_usage_summaries.items():
                    total_tokens += usage.total_input_tokens + usage.total_output_tokens

            # Record any code blocks (macros) that were executed
            if hasattr(response, "iterations"):
                for iteration in response.iterations:
                    if hasattr(iteration, "code_blocks"):
                        for block in iteration.code_blocks:
                            macro = MacroRecord(
                                task_index=i,
                                task_type=task["task_type"],
                                code=block.code
                                if hasattr(block, "code")
                                else str(block),
                                output=str(block.result)
                                if hasattr(block, "result")
                                else "",
                                timestamp=datetime.now().isoformat(),
                            )
                            results.macros.append(macro)

        except Exception as e:
            tqdm.write(f"  [RLM] Task {i} error: {e}")
            results.llm_fallbacks += 1

        # Update progress bar with tool counts
        if (i + 1) % 10 == 0:
            metrics = rlm.get_metrics()
            pbar.set_postfix(
                hooks=metrics["hooks"],
                repl=metrics["replacements"],
                util=metrics["utilities"],
            )

    results.tasks_processed = len(tasks)
    return total_tokens


def collect_tools(tools_dir: Path, results: ExperimentResults):
    """Collect all tools created during the experiment."""
    # Search in category subdirectories
    for category in ["pre_completion", "replacements", "utilities"]:
        cat_dir = tools_dir / category
        if not cat_dir.exists():
            continue
        for tool_file in cat_dir.glob("*.py"):
            code = tool_file.read_text()

            # Extract description from comment if present
            description = ""
            lines = code.split("\n")
            if lines and lines[0].startswith("# "):
                description = lines[0][2:].strip()

            tool = ToolRecord(
                name=f"{category}/{tool_file.stem}",
                file_path=str(tool_file),
                description=description,
                code=code,
                created_at=datetime.fromtimestamp(
                    tool_file.stat().st_mtime
                ).isoformat(),
            )
            results.tools.append(tool)


def generate_report(output_dir: Path, results: ExperimentResults, model: str) -> Path:
    """Generate PDF report with tools and macros."""
    pdf_path = output_dir / "report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=18, spaceAfter=20
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontSize=14, spaceAfter=12
    )
    subhead_style = ParagraphStyle(
        "SubHead", parent=styles["Heading3"], fontSize=12, spaceAfter=8
    )
    code_style = ParagraphStyle("Code", fontName="Courier", fontSize=8, leading=10)

    # Title
    story.append(Paragraph("Experiment 005: RLM-Based Tool Creation", title_style))
    story.append(Paragraph(f"Model: {model}", styles["Normal"]))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]
        )
    )
    story.append(Spacer(1, 20))

    # Summary
    story.append(Paragraph("Summary", heading_style))

    savings = results.baseline_tokens - results.rlm_tokens
    savings_pct = (
        (savings / results.baseline_tokens * 100) if results.baseline_tokens > 0 else 0
    )

    summary_data = [
        ["Metric", "Value"],
        ["Baseline Tokens", f"{results.baseline_tokens:,}"],
        ["RLM Tokens", f"{results.rlm_tokens:,}"],
        ["Token Savings", f"{savings:,} ({savings_pct:.1f}%)"],
        ["Tasks Processed", str(results.tasks_processed)],
        ["Tools Created", str(len(results.tools))],
        ["Macros Executed", str(len(results.macros))],
        ["LLM Fallbacks", str(results.llm_fallbacks)],
    ]

    table = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 30))

    # Tools Section
    story.append(Paragraph("Tools Created by LLM", heading_style))
    story.append(Spacer(1, 10))

    if results.tools:
        for i, tool in enumerate(results.tools):
            story.append(Paragraph(f"Tool {i + 1}: {tool.name}", subhead_style))
            story.append(Paragraph(f"<b>Path:</b> {tool.file_path}", styles["Normal"]))
            story.append(
                Paragraph(
                    f"<b>Description:</b> {tool.description or 'N/A'}", styles["Normal"]
                )
            )
            story.append(
                Paragraph(f"<b>Created:</b> {tool.created_at}", styles["Normal"])
            )
            story.append(Spacer(1, 5))
            story.append(Paragraph("<b>Code:</b>", styles["Normal"]))

            # Truncate long code for PDF
            code_preview = (
                tool.code[:1500] + "..." if len(tool.code) > 1500 else tool.code
            )
            story.append(Preformatted(code_preview, code_style))
            story.append(Spacer(1, 15))
    else:
        story.append(
            Paragraph("No tools were created during this experiment.", styles["Normal"])
        )

    story.append(PageBreak())

    # Macros Section
    story.append(Paragraph("Macros/Code Executed", heading_style))
    story.append(Spacer(1, 10))

    if results.macros:
        # Group by task type
        by_type: dict[str, list[MacroRecord]] = {}
        for macro in results.macros:
            if macro.task_type not in by_type:
                by_type[macro.task_type] = []
            by_type[macro.task_type].append(macro)

        for task_type, macros in by_type.items():
            story.append(
                Paragraph(
                    f"Task Type: {task_type} ({len(macros)} macros)", subhead_style
                )
            )

            # Show first few unique macros
            seen_codes = set()
            shown = 0
            for macro in macros:
                code_hash = hash(macro.code[:200])
                if code_hash not in seen_codes and shown < 3:
                    seen_codes.add(code_hash)
                    shown += 1

                    story.append(
                        Paragraph(f"<b>Task {macro.task_index}:</b>", styles["Normal"])
                    )
                    code_preview = (
                        macro.code[:800] + "..."
                        if len(macro.code) > 800
                        else macro.code
                    )
                    story.append(Preformatted(code_preview, code_style))

                    if macro.output:
                        output_preview = (
                            macro.output[:200] + "..."
                            if len(macro.output) > 200
                            else macro.output
                        )
                        story.append(
                            Paragraph(
                                f"<b>Output:</b> {output_preview}", styles["Normal"]
                            )
                        )
                    story.append(Spacer(1, 10))

            if len(macros) > shown:
                story.append(
                    Paragraph(
                        f"... and {len(macros) - shown} more similar macros",
                        styles["Normal"],
                    )
                )
            story.append(Spacer(1, 15))
    else:
        story.append(
            Paragraph(
                "No macros were executed during this experiment.", styles["Normal"]
            )
        )

    doc.build(story)
    return pdf_path


def save_results(output_dir: Path, results: ExperimentResults, model: str):
    """Save results to JSON."""
    data = {
        "metadata": {
            "model": model,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "baseline_tokens": results.baseline_tokens,
            "rlm_tokens": results.rlm_tokens,
            "tasks_processed": results.tasks_processed,
            "tools_created": len(results.tools),
            "macros_executed": len(results.macros),
            "llm_fallbacks": results.llm_fallbacks,
        },
        "tools": [
            {
                "name": t.name,
                "file_path": t.file_path,
                "description": t.description,
                "code": t.code,
                "created_at": t.created_at,
            }
            for t in results.tools
        ],
        "macros": [
            {
                "task_index": m.task_index,
                "task_type": m.task_type,
                "code": m.code,
                "output": m.output,
            }
            for m in results.macros[:50]  # Limit to first 50 for file size
        ],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(data, f, indent=2)


@dataclass
class BatchMetrics:
    """Metrics for a single batch of tasks."""

    batch_num: int
    tasks_in_batch: int
    cumulative_tasks: int
    baseline_tokens: int
    rlm_tokens: int
    hooks_count: int
    replacements_count: int
    llm_calls_made: int
    llm_calls_skipped: int
    cumulative_baseline: int = 0
    cumulative_rlm: int = 0


def run_with_batch_tracking(
    rlm: SelfDistillRLM,
    client: OllamaClient,
    base_model: str,
    tasks: list[dict],
    batch_size: int = 20,
    skip_baseline: bool = False,
) -> list[BatchMetrics]:
    """Run tasks in batches and track tool accumulation."""
    batch_metrics = []
    cumulative_baseline = 0
    cumulative_rlm = 0

    num_batches = (len(tasks) + batch_size - 1) // batch_size

    for batch_num in tqdm(range(num_batches), desc="Batches", unit="batch"):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(tasks))
        batch_tasks = tasks[start_idx:end_idx]

        tqdm.write(
            f"\n--- Batch {batch_num + 1}/{num_batches} (tasks {start_idx + 1}-{end_idx}) ---"
        )

        # Run baseline for this batch
        batch_baseline = 0
        if not skip_baseline:
            for task in tqdm(batch_tasks, desc="  Baseline", unit="task", leave=False):
                prompt = f"Analyze: {task['text']}\nTask: {task['task_type']}"
                _ = client.completion(prompt, base_model)
                usage = client.get_last_usage()
                batch_baseline += usage.total_input_tokens + usage.total_output_tokens

        # Run RLM for this batch
        batch_rlm = 0
        metrics_before = rlm.get_metrics()

        for task in tqdm(batch_tasks, desc="  RLM", unit="task", leave=False):
            prompt = f"""Task Type: {task["task_type"]}

Analyze this text: {task["text"]}

If this is a grammar task, check if the sentence is grammatically acceptable.
If this is a PII task, detect any personally identifiable information.

First check if you have an existing tool for this task type.
If not, create one and save it for reuse."""

            try:
                response = rlm.completion(prompt)
                if hasattr(response, "usage_summary") and response.usage_summary:
                    for (
                        model_name,
                        usage,
                    ) in response.usage_summary.model_usage_summaries.items():
                        batch_rlm += (
                            usage.total_input_tokens + usage.total_output_tokens
                        )
            except Exception as e:
                tqdm.write(f"  Error: {e}")

        metrics_after = rlm.get_metrics()
        cumulative_baseline += batch_baseline
        cumulative_rlm += batch_rlm

        batch_metric = BatchMetrics(
            batch_num=batch_num + 1,
            tasks_in_batch=len(batch_tasks),
            cumulative_tasks=end_idx,
            baseline_tokens=batch_baseline,
            rlm_tokens=batch_rlm,
            hooks_count=metrics_after["hooks"],
            replacements_count=metrics_after["replacements"],
            llm_calls_made=metrics_after["llm_calls_made"]
            - metrics_before.get("llm_calls_made", 0)
            if batch_num > 0
            else metrics_after["llm_calls_made"],
            llm_calls_skipped=metrics_after["llm_calls_skipped"]
            - metrics_before.get("llm_calls_skipped", 0)
            if batch_num > 0
            else metrics_after["llm_calls_skipped"],
            cumulative_baseline=cumulative_baseline,
            cumulative_rlm=cumulative_rlm,
        )
        batch_metrics.append(batch_metric)

        # Print batch summary
        savings_pct = (
            ((batch_baseline - batch_rlm) / batch_baseline * 100)
            if batch_baseline > 0
            else 0
        )
        tqdm.write(
            f"  Batch: baseline={batch_baseline:,} rlm={batch_rlm:,} ({savings_pct:+.1f}%)"
        )
        tqdm.write(
            f"  Tools: hooks={metrics_after['hooks']} replacements={metrics_after['replacements']}"
        )
        tqdm.write(
            f"  LLM: made={batch_metric.llm_calls_made} skipped={batch_metric.llm_calls_skipped}"
        )

    return batch_metrics


def main():
    parser = argparse.ArgumentParser(description="Experiment 005: RLM Tool Creation")
    parser.add_argument(
        "--model", default="ollama/llama3.2:3b", help="Model (litellm format)"
    )
    parser.add_argument("--cola", type=int, default=20, help="Number of CoLA samples")
    parser.add_argument("--pii", type=int, default=10, help="Number of PII samples")
    parser.add_argument("--sciq", type=int, default=0, help="Number of SciQ samples")
    parser.add_argument(
        "--all-train", action="store_true", help="Use all training data"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Batch size for tracking"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5, help="Max RLM iterations per task"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--tools-dir", type=str, default=None, help="Reuse existing tools directory"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip baseline run"
    )
    parser.add_argument(
        "--seed-tools",
        action="store_true",
        help="Seed tools directory with pre-built hooks",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 005: RLM-Based Tool Creation with Hooks")
    print("=" * 60)
    print()
    print(f"Model: {args.model}")
    if args.all_train:
        print("Mode: All training data with batch tracking")
    else:
        print(f"Tasks: {args.cola} CoLA + {args.pii} PII + {args.sciq} SciQ")
    print(f"Batch size: {args.batch_size}")
    if args.tools_dir:
        print(f"Reusing tools from: {args.tools_dir}")
    if args.seed_tools:
        print("Will seed tools directory with pre-built hooks")
    print()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiment_outputs/exp005_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use existing tools dir or create new one
    if args.tools_dir:
        tools_dir = Path(args.tools_dir)
        if not tools_dir.exists():
            print(f"Error: Tools directory {tools_dir} does not exist")
            return
    else:
        tools_dir = output_dir / "tools"
        tools_dir.mkdir(exist_ok=True)

    # Seed tools if requested
    if args.seed_tools:
        print("--- Seeding Tools Directory ---")
        seed_tools(tools_dir)
        print()

    results = ExperimentResults()

    # Load datasets
    print("Loading datasets...")
    tasks = []

    if args.all_train:
        # Load all training data
        cola_ds = load_dataset(DATA.COLA, Split.TRAIN)
        for item in tqdm(cola_ds, desc="  CoLA", unit="item"):
            tasks.append(
                {"text": item.question, "task_type": "grammar", "expected": item.answer}
            )

        pii_ds = load_dataset(DATA.PII_DETECTION, Split.TRAIN)
        pii_start = len(tasks)
        for item in tqdm(pii_ds, desc="  PII", unit="item"):
            tasks.append(
                {
                    "text": item.question,
                    "task_type": "pii_detection",
                    "expected": item.answer,
                }
            )

        sciq_ds = load_dataset(DATA.SCIQ, Split.TRAIN)
        sciq_start = len(tasks)
        for item in tqdm(sciq_ds, desc="  SciQ", unit="item"):
            tasks.append(
                {
                    "text": item.question,
                    "task_type": "science_qa",
                    "expected": item.answer,
                }
            )
    else:
        # CoLA tasks
        cola_ds = load_dataset(DATA.COLA, Split.DEV)
        for i, item in enumerate(cola_ds):
            if i >= args.cola:
                break
            tasks.append(
                {"text": item.question, "task_type": "grammar", "expected": item.answer}
            )

        # PII tasks
        if args.pii > 0:
            pii_ds = load_dataset(DATA.PII_DETECTION, Split.DEV)
            for i, item in enumerate(pii_ds):
                if i >= args.pii:
                    break
                tasks.append(
                    {
                        "text": item.question,
                        "task_type": "pii_detection",
                        "expected": item.answer,
                    }
                )

        # SciQ tasks
        if args.sciq > 0:
            sciq_ds = load_dataset(DATA.SCIQ, Split.DEV)
            for i, item in enumerate(sciq_ds):
                if i >= args.sciq:
                    break
                tasks.append(
                    {
                        "text": item.question,
                        "task_type": "science_qa",
                        "expected": item.answer,
                    }
                )

    print(f"Total: {len(tasks)} tasks")
    print()

    # Initialize RLM and client
    client = OllamaClient()
    base_model = args.model.replace("ollama/", "")

    rlm = SelfDistillRLM(
        model=args.model,
        tools_dir=tools_dir,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )

    # Run with batch tracking
    print("--- Running with Batch Tracking ---")
    batch_metrics = run_with_batch_tracking(
        rlm=rlm,
        client=client,
        base_model=base_model,
        tasks=tasks,
        batch_size=args.batch_size,
        skip_baseline=args.skip_baseline,
    )

    # Get final metrics
    metrics = rlm.get_metrics()
    results.tool_executions = metrics["total_tools"]

    # Accumulate totals from batches
    results.baseline_tokens = sum(b.baseline_tokens for b in batch_metrics)
    results.rlm_tokens = sum(b.rlm_tokens for b in batch_metrics)
    results.tasks_processed = len(tasks)

    print()
    print("=" * 60)
    print("BATCH ACCUMULATION SUMMARY")
    print("=" * 60)
    print(
        f"{'Batch':<6} {'Tasks':<8} {'Baseline':<12} {'RLM':<12} {'Savings':<10} {'Hooks':<6} {'Repl':<6}"
    )
    print("-" * 60)
    for b in batch_metrics:
        savings = b.baseline_tokens - b.rlm_tokens
        savings_pct = (
            (savings / b.baseline_tokens * 100) if b.baseline_tokens > 0 else 0
        )
        print(
            f"{b.batch_num:<6} {b.cumulative_tasks:<8} {b.cumulative_baseline:<12,} {b.cumulative_rlm:<12,} {savings_pct:>+8.1f}% {b.hooks_count:<6} {b.replacements_count:<6}"
        )
    print()

    # Save batch metrics to JSON
    batch_data = [
        {
            "batch": b.batch_num,
            "cumulative_tasks": b.cumulative_tasks,
            "baseline_tokens": b.baseline_tokens,
            "rlm_tokens": b.rlm_tokens,
            "cumulative_baseline": b.cumulative_baseline,
            "cumulative_rlm": b.cumulative_rlm,
            "hooks": b.hooks_count,
            "replacements": b.replacements_count,
            "llm_calls_made": b.llm_calls_made,
            "llm_calls_skipped": b.llm_calls_skipped,
        }
        for b in batch_metrics
    ]
    with open(output_dir / "batch_metrics.json", "w") as f:
        json.dump(batch_data, f, indent=2)

    print(
        f"Tools: hooks={metrics['hooks']}, replacements={metrics['replacements']}, utilities={metrics['utilities']}"
    )
    print(f"Hook executions: {metrics['hook_executions']}")
    print(
        f"LLM calls: made={metrics['llm_calls_made']}, skipped={metrics['llm_calls_skipped']}"
    )
    print(f"Replacement tool uses: {metrics['replacement_uses']}")
    print()

    # Collect tools
    print("Collecting tools...")
    collect_tools(tools_dir, results)
    print(f"Found {len(results.tools)} tools")
    print()

    # Generate report
    print("Generating report...")
    pdf_path = generate_report(output_dir, results, args.model)
    save_results(output_dir, results, args.model)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    savings = results.baseline_tokens - results.rlm_tokens
    savings_pct = (
        (savings / results.baseline_tokens * 100) if results.baseline_tokens > 0 else 0
    )

    print(f"Baseline: {results.baseline_tokens:,} tokens")
    print(f"RLM: {results.rlm_tokens:,} tokens")
    if savings > 0:
        print(f"Savings: {savings:,} tokens ({savings_pct:.1f}%)")
    else:
        print(f"Overhead: {-savings:,} tokens ({-savings_pct:.1f}%)")
    print()
    print(f"Tools created: {len(results.tools)}")
    print(f"  - Hooks (pre_completion): {metrics['hooks']}")
    print(f"  - Replacements: {metrics['replacements']}")
    print(f"  - Utilities: {metrics['utilities']}")
    print()
    print(f"Hook System Performance:")
    print(f"  - Hook executions: {metrics['hook_executions']}")
    print(f"  - LLM calls made: {metrics['llm_calls_made']}")
    print(f"  - LLM calls SKIPPED (replaced by tools): {metrics['llm_calls_skipped']}")
    print(f"  - Replacement tool uses: {metrics['replacement_uses']}")
    print()
    print(f"Report: {pdf_path}")
    print(f"Tools directory: {tools_dir}")


if __name__ == "__main__":
    main()
