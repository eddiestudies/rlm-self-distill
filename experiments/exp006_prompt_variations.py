#!/usr/bin/env python3
"""
Experiment 006: Prompt Variation Testing

This experiment tests different system prompt variations to find
the most effective approach for self-distillation.

Prompt Versions:
- v1_basic: Core principles with single example
- v2_cost: Cost-aware with explicit cost/benefit reasoning
- v3_pattern: Pattern learning focus
- v4_minimal: Concise instructions, maximum autonomy
- v5_reflective: Explicit reasoning about decisions

Metrics tracked per prompt:
- Tools created (hooks + replacements)
- Tool quality (correct contract implementation)
- Token usage (creation overhead vs savings)
- LLM calls skipped

Features:
- Checkpoint support: saves after each prompt, can resume with --resume
- Progress bars with ETA
"""

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
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
from self_distill.rlm.prompts import PROMPTS, PROMPT_DESCRIPTIONS


@dataclass
class PromptTestResult:
    """Results from testing a single prompt version."""
    prompt_version: str
    prompt_description: str

    # Tool metrics
    hooks_created: int = 0
    replacements_created: int = 0
    utilities_created: int = 0

    # Quality metrics
    tools_with_correct_contract: int = 0
    tools_with_errors: int = 0

    # Efficiency metrics
    baseline_tokens: int = 0
    rlm_tokens: int = 0
    llm_calls_made: int = 0
    llm_calls_skipped: int = 0

    # Task metrics
    tasks_processed: int = 0
    tasks_total: int = 0

    # Timing
    duration_seconds: float = 0.0
    start_time: str = ""
    end_time: str = ""

    # Tool code samples
    tool_samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_version": self.prompt_version,
            "prompt_description": self.prompt_description,
            "hooks_created": self.hooks_created,
            "replacements_created": self.replacements_created,
            "utilities_created": self.utilities_created,
            "tools_with_correct_contract": self.tools_with_correct_contract,
            "tools_with_errors": self.tools_with_errors,
            "baseline_tokens": self.baseline_tokens,
            "rlm_tokens": self.rlm_tokens,
            "tasks_processed": self.tasks_processed,
            "tasks_total": self.tasks_total,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tool_samples": self.tool_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PromptTestResult":
        """Create from dictionary."""
        return cls(
            prompt_version=data["prompt_version"],
            prompt_description=data["prompt_description"],
            hooks_created=data.get("hooks_created", 0),
            replacements_created=data.get("replacements_created", 0),
            utilities_created=data.get("utilities_created", 0),
            tools_with_correct_contract=data.get("tools_with_correct_contract", 0),
            tools_with_errors=data.get("tools_with_errors", 0),
            baseline_tokens=data.get("baseline_tokens", 0),
            rlm_tokens=data.get("rlm_tokens", 0),
            tasks_processed=data.get("tasks_processed", 0),
            tasks_total=data.get("tasks_total", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            tool_samples=data.get("tool_samples", []),
        )


def save_checkpoint(output_dir: Path, completed_prompts: list[str], results: list[PromptTestResult], metadata: dict):
    """Save checkpoint after each prompt completion."""
    checkpoint = {
        "metadata": metadata,
        "completed_prompts": completed_prompts,
        "results": [r.to_dict() for r in results],
        "checkpoint_time": datetime.now().isoformat(),
    }
    checkpoint_path = output_dir / "checkpoint.json"
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    return checkpoint_path


def load_checkpoint(output_dir: Path) -> tuple[list[str], list[PromptTestResult], dict] | None:
    """Load checkpoint if exists."""
    checkpoint_path = output_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    completed = checkpoint.get("completed_prompts", [])
    results = [PromptTestResult.from_dict(r) for r in checkpoint.get("results", [])]
    metadata = checkpoint.get("metadata", {})

    return completed, results, metadata


def create_rlm_with_prompt(prompt_version: str, tools_dir: Path, model: str, verbose: bool = False):
    """Create an RLM instance with a specific prompt version."""
    from rlm import RLM

    prompt = PROMPTS[prompt_version]

    # Generate setup code for tool registry (same as SelfDistillRLM)
    setup_code = f'''
import os
import sys
import importlib.util

TOOLS_DIR = "{tools_dir}"
CATEGORIES = ["pre_completion", "replacements", "utilities"]

for cat in CATEGORIES:
    os.makedirs(os.path.join(TOOLS_DIR, cat), exist_ok=True)

_current_tool_category = None
_current_tool_name = None
_current_tool_lines = []

def _get_tools_in_category(category):
    cat_dir = os.path.join(TOOLS_DIR, category)
    if not os.path.exists(cat_dir):
        return []
    return [f[:-3] for f in os.listdir(cat_dir) if f.endswith(".py") and not f.startswith("_")]

def list_all_tools():
    result = {{}}
    for cat in CATEGORIES:
        tools = _get_tools_in_category(cat)
        result[cat] = tools
        print(f"{{cat}}: {{tools}}")
    return result

def load_tool(category, name):
    path = os.path.join(TOOLS_DIR, category, f"{{name}}.py")
    if not os.path.exists(path):
        print(f"Tool not found: {{category}}/{{name}}")
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def create_tool(category, name, description=""):
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
    global _current_tool_lines
    if _current_tool_name is None:
        print("Error: No tool being created. Call create_tool first.")
        return
    _current_tool_lines.append(line)

write_to_text = write_to_tool
add_line = write_to_tool
write_line = write_to_tool

def finish_tool():
    global _current_tool_category, _current_tool_name, _current_tool_lines
    if _current_tool_name is None:
        print("Error: No tool being created.")
        return None

    code = "\\n".join(_current_tool_lines)
    path = os.path.join(TOOLS_DIR, _current_tool_category, f"{{_current_tool_name}}.py")

    with open(path, "w") as f:
        f.write(code)

    print(f"Tool saved: {{path}}")

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

print("=== Tool Registry Ready ===")
print(f"Tools directory: {{TOOLS_DIR}}")
list_all_tools()
'''

    rlm = RLM(
        backend="litellm",
        backend_kwargs={
            "model_name": model,
            "api_base": "http://localhost:11434",
        },
        environment="local",
        environment_kwargs={"setup_code": setup_code},
        max_iterations=5,
        custom_system_prompt=prompt,
        verbose=verbose,
    )

    return rlm


def validate_tool(tool_path: Path, category: str) -> dict:
    """Validate a tool has the correct contract."""
    import importlib.util

    result = {
        "path": str(tool_path),
        "valid": False,
        "has_correct_function": False,
        "error": None,
    }

    try:
        spec = importlib.util.spec_from_file_location(tool_path.stem, tool_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if category == "pre_completion":
            if hasattr(module, 'check'):
                result["has_correct_function"] = True
                # Test that it returns bool
                try:
                    ret = module.check("test text")
                    result["valid"] = isinstance(ret, bool)
                except Exception as e:
                    result["error"] = f"check() error: {e}"
        else:
            if hasattr(module, 'run'):
                result["has_correct_function"] = True
                result["valid"] = True  # Don't test run() as it may have side effects

    except Exception as e:
        result["error"] = str(e)

    return result


def run_prompt_test(
    prompt_version: str,
    model: str,
    tasks: list[dict],
    output_dir: Path,
    client: OllamaClient,
    base_model: str,
    verbose: bool = False,
) -> PromptTestResult:
    """Run a test with a specific prompt version."""

    result = PromptTestResult(
        prompt_version=prompt_version,
        prompt_description=PROMPT_DESCRIPTIONS[prompt_version],
        tasks_total=len(tasks),
        start_time=datetime.now().isoformat(),
    )

    # Create tools directory for this prompt
    tools_dir = output_dir / f"tools_{prompt_version}"
    tools_dir.mkdir(exist_ok=True)
    for cat in ["pre_completion", "replacements", "utilities"]:
        (tools_dir / cat).mkdir(exist_ok=True)

    # Create RLM with this prompt
    rlm = create_rlm_with_prompt(prompt_version, tools_dir, model, verbose)

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Testing: {prompt_version} - {PROMPT_DESCRIPTIONS[prompt_version]}")
    tqdm.write(f"{'='*60}")

    start_time = time.time()

    # Run baseline for comparison (quick sample)
    baseline_sample = min(5, len(tasks))
    for task in tqdm(tasks[:baseline_sample], desc=f"  [{prompt_version}] Baseline sample", unit="task", leave=False):
        prompt = f"Analyze: {task['text']}\nTask: {task['task_type']}"
        _ = client.completion(prompt, base_model)
        usage = client.get_last_usage()
        result.baseline_tokens += usage.total_input_tokens + usage.total_output_tokens

    # Scale up baseline estimate
    if baseline_sample > 0:
        result.baseline_tokens = int(result.baseline_tokens * len(tasks) / baseline_sample)

    # Run RLM with detailed progress
    pbar = tqdm(
        enumerate(tasks),
        total=len(tasks),
        desc=f"  [{prompt_version}] RLM",
        unit="task",
        dynamic_ncols=True,
    )

    for i, task in pbar:
        prompt = f"""Task Type: {task['task_type']}

Analyze this text: {task['text']}

First check if you have an existing tool for this task type.
If not, and this is a repeatable pattern, consider creating one."""

        task_start = time.time()
        try:
            response = rlm.completion(prompt)

            if hasattr(response, 'usage_summary') and response.usage_summary:
                for model_name, usage in response.usage_summary.model_usage_summaries.items():
                    result.rlm_tokens += usage.total_input_tokens + usage.total_output_tokens

            result.tasks_processed += 1

        except Exception as e:
            if verbose:
                tqdm.write(f"  Task {i} error: {e}")

        # Update progress bar with stats
        task_time = time.time() - task_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(tasks) - i - 1)

        pbar.set_postfix({
            "done": result.tasks_processed,
            "tokens": f"{result.rlm_tokens:,}",
            "eta": f"{remaining/60:.1f}m",
        })

    result.duration_seconds = time.time() - start_time
    result.end_time = datetime.now().isoformat()

    # Collect and validate tools
    for category in ["pre_completion", "replacements", "utilities"]:
        cat_dir = tools_dir / category
        for tool_file in cat_dir.glob("*.py"):
            if tool_file.name.startswith("_"):
                continue

            if category == "pre_completion":
                result.hooks_created += 1
            elif category == "replacements":
                result.replacements_created += 1
            else:
                result.utilities_created += 1

            # Validate
            validation = validate_tool(tool_file, category)
            if validation["valid"]:
                result.tools_with_correct_contract += 1
            else:
                result.tools_with_errors += 1

            # Sample tool code
            if len(result.tool_samples) < 5:
                result.tool_samples.append({
                    "name": f"{category}/{tool_file.stem}",
                    "code": tool_file.read_text()[:500],
                    "valid": validation["valid"],
                    "error": validation.get("error"),
                })

    tqdm.write(f"  Completed: {result.hooks_created} hooks, {result.replacements_created} replacements")
    tqdm.write(f"  Tokens: baseline~{result.baseline_tokens:,} rlm={result.rlm_tokens:,}")
    tqdm.write(f"  Duration: {result.duration_seconds/60:.1f} minutes")

    return result


def generate_report(output_dir: Path, results: list[PromptTestResult], model: str) -> Path:
    """Generate PDF comparison report."""
    pdf_path = output_dir / "prompt_comparison_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=12)
    code_style = ParagraphStyle('Code', fontName='Courier', fontSize=8, leading=10)

    # Title
    story.append(Paragraph("Experiment 006: Prompt Variation Testing", title_style))
    story.append(Paragraph(f"Model: {model}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Summary comparison table
    story.append(Paragraph("Prompt Comparison Summary", heading_style))

    table_data = [
        ["Prompt", "Hooks", "Repl", "Valid", "Baseline", "RLM", "Savings", "Time"]
    ]

    for r in results:
        savings = r.baseline_tokens - r.rlm_tokens
        savings_pct = (savings / r.baseline_tokens * 100) if r.baseline_tokens > 0 else 0
        table_data.append([
            r.prompt_version,
            str(r.hooks_created),
            str(r.replacements_created),
            str(r.tools_with_correct_contract),
            f"{r.baseline_tokens:,}",
            f"{r.rlm_tokens:,}",
            f"{savings_pct:+.1f}%",
            f"{r.duration_seconds/60:.1f}m",
        ])

    table = Table(table_data, colWidths=[1*inch, 0.5*inch, 0.5*inch, 0.5*inch, 0.9*inch, 0.9*inch, 0.7*inch, 0.6*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 30))

    # Detailed results per prompt
    for r in results:
        story.append(PageBreak())
        story.append(Paragraph(f"Prompt: {r.prompt_version}", heading_style))
        story.append(Paragraph(f"<b>Description:</b> {r.prompt_description}", styles['Normal']))
        story.append(Spacer(1, 10))

        # Metrics
        savings = r.baseline_tokens - r.rlm_tokens
        savings_pct = (savings / r.baseline_tokens * 100) if r.baseline_tokens > 0 else 0

        metrics_data = [
            ["Metric", "Value"],
            ["Hooks Created", str(r.hooks_created)],
            ["Replacements Created", str(r.replacements_created)],
            ["Tools with Correct Contract", str(r.tools_with_correct_contract)],
            ["Tools with Errors", str(r.tools_with_errors)],
            ["Baseline Tokens (est)", f"{r.baseline_tokens:,}"],
            ["RLM Tokens", f"{r.rlm_tokens:,}"],
            ["Token Savings", f"{savings:,} ({savings_pct:+.1f}%)"],
            ["Duration", f"{r.duration_seconds/60:.1f} minutes"],
            ["Tasks Processed", f"{r.tasks_processed}/{r.tasks_total}"],
        ]

        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 15))

        # Tool samples
        if r.tool_samples:
            story.append(Paragraph("Sample Tools Created:", styles['Normal']))
            for sample in r.tool_samples[:3]:
                story.append(Paragraph(f"<b>{sample['name']}</b> (valid={sample['valid']})", styles['Normal']))
                story.append(Preformatted(sample['code'][:400], code_style))
                story.append(Spacer(1, 10))

    doc.build(story)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Experiment 006: Prompt Variation Testing")
    parser.add_argument("--model", default="ollama/qwen2.5-coder:32b", help="Model to test")
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS.keys()), help="Prompts to test")
    parser.add_argument("--cola", type=int, default=30, help="CoLA samples per test")
    parser.add_argument("--pii", type=int, default=10, help="PII samples per test")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint directory (e.g., experiment_outputs/exp006_20260112_234253)")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 006: Prompt Variation Testing")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Prompts to test: {args.prompts}")
    print(f"Tasks per prompt: {args.cola} CoLA + {args.pii} PII")

    # Setup output directory
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            print(f"Error: Resume directory not found: {args.resume}")
            return
        print(f"Resuming from: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiment_outputs/exp006_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Load checkpoint if resuming
    completed_prompts = []
    results = []
    metadata = {
        "model": args.model,
        "prompts": args.prompts,
        "cola": args.cola,
        "pii": args.pii,
        "started": datetime.now().isoformat(),
    }

    if args.resume:
        checkpoint_data = load_checkpoint(output_dir)
        if checkpoint_data:
            completed_prompts, results, saved_metadata = checkpoint_data
            metadata.update(saved_metadata)
            print(f"Loaded checkpoint: {len(completed_prompts)} prompts already completed")
            print(f"  Completed: {completed_prompts}")
            remaining = [p for p in args.prompts if p not in completed_prompts]
            print(f"  Remaining: {remaining}")
            print()

    # Load tasks
    print("Loading datasets...")
    tasks = []

    cola_ds = load_dataset(DATA.COLA, Split.DEV)
    for i, item in enumerate(cola_ds):
        if i >= args.cola:
            break
        tasks.append({
            "text": item.question,
            "task_type": "grammar",
            "expected": item.answer
        })

    pii_ds = load_dataset(DATA.PII_DETECTION, Split.DEV)
    for i, item in enumerate(pii_ds):
        if i >= args.pii:
            break
        tasks.append({
            "text": item.question,
            "task_type": "pii_detection",
            "expected": item.answer
        })

    print(f"Loaded {len(tasks)} tasks")
    print()

    # Initialize client
    client = OllamaClient()
    base_model = args.model.replace("ollama/", "")

    # Calculate remaining prompts
    prompts_to_run = [p for p in args.prompts if p not in completed_prompts]

    if not prompts_to_run:
        print("All prompts already completed!")
    else:
        print(f"Running {len(prompts_to_run)} prompt(s): {prompts_to_run}")

        # Estimate total time based on completed results
        if results:
            avg_time = sum(r.duration_seconds for r in results) / len(results)
            est_remaining = avg_time * len(prompts_to_run)
            print(f"Estimated time remaining: {est_remaining/60:.1f} minutes")
        print()

    # Test each prompt with checkpoint saving
    for prompt_version in tqdm(prompts_to_run, desc="Prompts", unit="prompt"):
        if prompt_version not in PROMPTS:
            tqdm.write(f"Warning: Unknown prompt version {prompt_version}, skipping")
            continue

        result = run_prompt_test(
            prompt_version=prompt_version,
            model=args.model,
            tasks=tasks,
            output_dir=output_dir,
            client=client,
            base_model=base_model,
            verbose=args.verbose,
        )
        results.append(result)
        completed_prompts.append(prompt_version)

        # Save checkpoint after each prompt
        checkpoint_path = save_checkpoint(output_dir, completed_prompts, results, metadata)
        tqdm.write(f"  Checkpoint saved: {checkpoint_path}")

    # Generate report
    print("\nGenerating comparison report...")
    pdf_path = generate_report(output_dir, results, args.model)

    # Save final JSON results
    results_data = {
        "metadata": {
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "tasks_per_prompt": len(tasks),
            "total_duration_minutes": sum(r.duration_seconds for r in results) / 60,
        },
        "results": [r.to_dict() for r in results]
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Prompt':<15} {'Hooks':<8} {'Repl':<8} {'Valid':<8} {'Savings':<12} {'Time':<10}")
    print("-" * 70)

    total_time = 0
    for r in results:
        savings_pct = ((r.baseline_tokens - r.rlm_tokens) / r.baseline_tokens * 100) if r.baseline_tokens > 0 else 0
        print(f"{r.prompt_version:<15} {r.hooks_created:<8} {r.replacements_created:<8} {r.tools_with_correct_contract:<8} {savings_pct:>+10.1f}% {r.duration_seconds/60:>8.1f}m")
        total_time += r.duration_seconds

    print("-" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print()
    print(f"Report: {pdf_path}")
    print(f"Results: {output_dir / 'results.json'}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
