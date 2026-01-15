#!/usr/bin/env python3
"""
Experiment 007: Evidence-Guided Tool Creation

This experiment uses embedding-based clustering to detect patterns
in tasks BEFORE prompting the model. When a cluster is detected,
we add context to the prompt: "Found N similar tasks - consider creating a tool."

Key difference from exp006:
- exp006: Model decides if pattern exists (often misses patterns)
- exp007: We detect patterns with embeddings, tell model explicitly

Architecture:
    Task → Embed → Check cluster → If cluster found:
                                      Add "Found N similar tasks" to prompt
                                   → Process with RLM
                                   → Store embedding for future detection

Features:
- Checkpoint support with --resume
- Progress bars with ETA
- Saves evidence store for analysis
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
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

from self_distill.clients.ollama_client import OllamaClient
from self_distill.datasets import DATA, load_dataset, Split
from self_distill.evidence import EvidenceStore


@dataclass
class ExperimentResult:
    """Results from the experiment."""

    # Task metrics
    tasks_processed: int = 0
    tasks_total: int = 0

    # Pattern detection
    patterns_detected: int = 0
    patterns_triggered_tools: int = 0

    # Tool metrics
    hooks_created: int = 0
    replacements_created: int = 0

    # Token metrics
    baseline_tokens: int = 0
    rlm_tokens: int = 0

    # Timing
    duration_seconds: float = 0.0
    start_time: str = ""
    end_time: str = ""

    # Cluster analysis
    clusters_found: int = 0
    largest_cluster: int = 0
    avg_cluster_size: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tasks_processed": self.tasks_processed,
            "tasks_total": self.tasks_total,
            "patterns_detected": self.patterns_detected,
            "patterns_triggered_tools": self.patterns_triggered_tools,
            "hooks_created": self.hooks_created,
            "replacements_created": self.replacements_created,
            "baseline_tokens": self.baseline_tokens,
            "rlm_tokens": self.rlm_tokens,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "clusters_found": self.clusters_found,
            "largest_cluster": self.largest_cluster,
            "avg_cluster_size": self.avg_cluster_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def save_checkpoint(
    output_dir: Path, result: ExperimentResult, processed_ids: list[str], metadata: dict
):
    """Save checkpoint after each batch."""
    checkpoint = {
        "metadata": metadata,
        "processed_ids": processed_ids,
        "result": result.to_dict(),
        "checkpoint_time": datetime.now().isoformat(),
    }
    checkpoint_path = output_dir / "checkpoint.json"
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    return checkpoint_path


def load_checkpoint(
    output_dir: Path,
) -> tuple[list[str], ExperimentResult, dict] | None:
    """Load checkpoint if exists."""
    checkpoint_path = output_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    processed_ids = checkpoint.get("processed_ids", [])
    result = ExperimentResult.from_dict(checkpoint.get("result", {}))
    metadata = checkpoint.get("metadata", {})

    return processed_ids, result, metadata


def get_tool_registry_setup_code(tools_dir: Path) -> str:
    """Generate setup code for tool registry."""
    return f'''
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


SYSTEM_PROMPT = """You are a Self-Distilling Language Model. Your goal is to create reusable tools
for patterns you encounter repeatedly.

## Tool Architecture

You have access to a tools directory with two key categories:

### pre_completion/ - Detection hooks
Files here contain a `check(text: str) -> bool` function.
- Return `True` if this task should be handled by a tool
- Return `False` to let the LLM handle it normally

### replacements/ - Execution tools
Files here contain a `run(text: str) -> str | dict` function.
- Called when the matching hook returns True
- Returns the result directly, replacing the LLM call

**Contract**: Hook and replacement names MUST match exactly.

## REPL Functions

```repl
list_all_tools()                           # See existing tools
create_tool(category, name, description)   # Start a new tool
write_to_tool(code_line)                   # Add a line of code
finish_tool()                              # Save and activate
run_tool(category, name, text)             # Test a tool
```

## When to Create Tools

Create a tool when you see a PATTERN - multiple similar tasks that can be handled
with deterministic code. The system will tell you when it detects patterns.

## Example

```repl
create_tool("pre_completion", "email_detector", "Detects email addresses")
write_to_tool("import re")
write_to_tool("")
write_to_tool("def check(text: str) -> bool:")
write_to_tool("    return bool(re.search(r'[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}', text))")
finish_tool()
```

## Output

After analysis, return your answer using:
- FINAL(your answer)
- FINAL_VAR(variable_name)
"""


def run_experiment(
    model: str,
    tasks: list[dict],
    output_dir: Path,
    evidence_store: EvidenceStore,
    client: OllamaClient,
    result: ExperimentResult,
    processed_ids: set[str],
    verbose: bool = False,
) -> ExperimentResult:
    """Run the experiment with evidence-guided tool creation."""
    from rlm import RLM

    tools_dir = output_dir / "tools"
    tools_dir.mkdir(exist_ok=True)
    for cat in ["pre_completion", "replacements", "utilities"]:
        (tools_dir / cat).mkdir(exist_ok=True)

    setup_code = get_tool_registry_setup_code(tools_dir)

    rlm = RLM(
        backend="litellm",
        backend_kwargs={
            "model_name": model,
            "api_base": "http://localhost:11434",
        },
        environment="local",
        environment_kwargs={"setup_code": setup_code},
        max_iterations=5,
        custom_system_prompt=SYSTEM_PROMPT,
        verbose=verbose,
    )

    base_model = model.replace("ollama/", "")
    start_time = time.time()

    # Filter to unprocessed tasks
    remaining_tasks = [t for t in tasks if t["id"] not in processed_ids]

    tqdm.write(
        f"\nProcessing {len(remaining_tasks)} tasks ({len(processed_ids)} already done)"
    )

    # Run baseline sample
    if result.baseline_tokens == 0:
        baseline_sample = min(5, len(tasks))
        for task in tqdm(
            tasks[:baseline_sample], desc="Baseline sample", unit="task", leave=False
        ):
            prompt = f"Analyze: {task['text']}\nTask: {task['task_type']}"
            _ = client.completion(prompt, base_model)
            usage = client.get_last_usage()
            result.baseline_tokens += (
                usage.total_input_tokens + usage.total_output_tokens
            )

        if baseline_sample > 0:
            result.baseline_tokens = int(
                result.baseline_tokens * len(tasks) / baseline_sample
            )

    # Process tasks with evidence guidance
    pbar = tqdm(
        remaining_tasks,
        desc="Evidence-guided RLM",
        unit="task",
        dynamic_ncols=True,
    )

    checkpoint_interval = 10  # Save every 10 tasks

    for i, task in enumerate(pbar):
        task_id = task["id"]
        task_text = task["text"]
        task_type = task["task_type"]

        # Check for pattern in evidence store
        pattern_prompt = evidence_store.get_pattern_prompt(task_text, n_samples=3)

        if pattern_prompt:
            result.patterns_detected += 1
            tqdm.write(f"  Pattern detected for task {task_id}!")

        # Build prompt with optional pattern context
        prompt_parts = [f"Task Type: {task_type}", f"Analyze this text: {task_text}"]

        if pattern_prompt:
            prompt_parts.append(f"\n## Pattern Evidence\n{pattern_prompt}")

        prompt_parts.append(
            "\nFirst check if you have an existing tool. If a pattern was detected above, strongly consider creating a reusable tool."
        )

        prompt = "\n\n".join(prompt_parts)

        # Run RLM
        try:
            response = rlm.completion(prompt)

            if hasattr(response, "usage_summary") and response.usage_summary:
                for (
                    model_name,
                    usage,
                ) in response.usage_summary.model_usage_summaries.items():
                    result.rlm_tokens += (
                        usage.total_input_tokens + usage.total_output_tokens
                    )

            result.tasks_processed += 1

        except Exception as e:
            if verbose:
                tqdm.write(f"  Task {task_id} error: {e}")

        # Add to evidence store for future pattern detection
        evidence_store.add(task_id, task_text, metadata={"type": task_type})
        processed_ids.add(task_id)

        # Update progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(remaining_tasks) - i - 1)

        pbar.set_postfix(
            {
                "done": result.tasks_processed,
                "patterns": result.patterns_detected,
                "tokens": f"{result.rlm_tokens:,}",
                "eta": f"{remaining / 60:.1f}m",
            }
        )

        # Checkpoint periodically
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(output_dir, result, list(processed_ids), {"model": model})
            evidence_store.save(output_dir / "evidence_store")

    result.duration_seconds = time.time() - start_time
    result.end_time = datetime.now().isoformat()

    # Count created tools
    for category in ["pre_completion", "replacements"]:
        cat_dir = tools_dir / category
        tools = [f for f in cat_dir.glob("*.py") if not f.name.startswith("_")]
        if category == "pre_completion":
            result.hooks_created = len(tools)
        else:
            result.replacements_created = len(tools)

    # Analyze clusters
    stats = evidence_store.analyze()
    result.clusters_found = stats.num_clusters
    result.largest_cluster = stats.largest_cluster_size
    result.avg_cluster_size = stats.avg_cluster_size

    return result


def generate_report(output_dir: Path, result: ExperimentResult, model: str) -> Path:
    """Generate PDF report."""
    pdf_path = output_dir / "experiment_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=18, spaceAfter=20
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontSize=14, spaceAfter=12
    )

    story.append(
        Paragraph("Experiment 007: Evidence-Guided Tool Creation", title_style)
    )
    story.append(Paragraph(f"Model: {model}", styles["Normal"]))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]
        )
    )
    story.append(Spacer(1, 20))

    # Results table
    story.append(Paragraph("Results Summary", heading_style))

    savings = result.baseline_tokens - result.rlm_tokens
    savings_pct = (
        (savings / result.baseline_tokens * 100) if result.baseline_tokens > 0 else 0
    )

    data = [
        ["Metric", "Value"],
        ["Tasks Processed", f"{result.tasks_processed}/{result.tasks_total}"],
        ["Patterns Detected", str(result.patterns_detected)],
        ["Hooks Created", str(result.hooks_created)],
        ["Replacements Created", str(result.replacements_created)],
        ["Baseline Tokens (est)", f"{result.baseline_tokens:,}"],
        ["RLM Tokens", f"{result.rlm_tokens:,}"],
        ["Token Savings", f"{savings:,} ({savings_pct:+.1f}%)"],
        ["Duration", f"{result.duration_seconds / 60:.1f} minutes"],
        ["Clusters Found", str(result.clusters_found)],
        ["Largest Cluster", str(result.largest_cluster)],
        ["Avg Cluster Size", f"{result.avg_cluster_size:.1f}"],
    ]

    table = Table(data, colWidths=[2.5 * inch, 2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)

    doc.build(story)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 007: Evidence-Guided Tool Creation"
    )
    parser.add_argument(
        "--model", default="ollama/qwen2.5-coder:32b", help="Model to use"
    )
    parser.add_argument("--cola", type=int, default=50, help="CoLA samples")
    parser.add_argument("--pii", type=int, default=20, help="PII samples")
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.75,
        help="Similarity threshold for clustering",
    )
    parser.add_argument(
        "--min-cluster", type=int, default=3, help="Minimum cluster size to trigger"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint directory")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 007: Evidence-Guided Tool Creation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Tasks: {args.cola} CoLA + {args.pii} PII")
    print(f"Clustering: similarity={args.similarity}, min_cluster={args.min_cluster}")

    # Setup output directory
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            print(f"Error: Resume directory not found: {args.resume}")
            return
        print(f"Resuming from: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiment_outputs/exp007_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Load or create evidence store
    evidence_store_path = output_dir / "evidence_store"
    if evidence_store_path.exists():
        print("Loading existing evidence store...")
        evidence_store = EvidenceStore.load(evidence_store_path)
        print(f"  Loaded {len(evidence_store)} existing embeddings")
    else:
        print("Creating new evidence store...")
        evidence_store = EvidenceStore(
            similarity_threshold=args.similarity,
            min_cluster_size=args.min_cluster,
        )

    # Load checkpoint or create new result
    processed_ids = set()
    result = ExperimentResult(start_time=datetime.now().isoformat())

    if args.resume:
        checkpoint_data = load_checkpoint(output_dir)
        if checkpoint_data:
            loaded_ids, result, metadata = checkpoint_data
            processed_ids = set(loaded_ids)
            print(f"Loaded checkpoint: {len(processed_ids)} tasks already processed")

    # Load tasks
    print("\nLoading datasets...")
    tasks = []

    cola_ds = load_dataset(DATA.COLA, Split.DEV)
    for i, item in enumerate(cola_ds):
        if i >= args.cola:
            break
        tasks.append(
            {
                "id": f"cola_{i}",
                "text": item.question,
                "task_type": "grammar",
                "expected": item.answer,
            }
        )

    pii_ds = load_dataset(DATA.PII_DETECTION, Split.DEV)
    for i, item in enumerate(pii_ds):
        if i >= args.pii:
            break
        tasks.append(
            {
                "id": f"pii_{i}",
                "text": item.question,
                "task_type": "pii_detection",
                "expected": item.answer,
            }
        )

    result.tasks_total = len(tasks)
    print(f"Loaded {len(tasks)} tasks")

    # NOTE: We do NOT pre-embed tasks. Pattern detection happens incrementally:
    # 1. Check new task against previously-seen tasks
    # 2. If pattern found, add context to prompt
    # 3. Add task to store AFTER processing
    # This ensures patterns are detected as clusters form over time.

    # Initialize client
    client = OllamaClient()

    # Run experiment
    print("\n" + "=" * 60)
    print("Running experiment...")
    print("=" * 60)

    result = run_experiment(
        model=args.model,
        tasks=tasks,
        output_dir=output_dir,
        evidence_store=evidence_store,
        client=client,
        result=result,
        processed_ids=processed_ids,
        verbose=args.verbose,
    )

    # Save final state
    evidence_store.save(output_dir / "evidence_store")
    save_checkpoint(output_dir, result, list(processed_ids), {"model": args.model})

    # Generate report
    print("\nGenerating report...")
    pdf_path = generate_report(output_dir, result, args.model)

    # Save JSON results
    with open(output_dir / "results.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tasks processed: {result.tasks_processed}/{result.tasks_total}")
    print(f"Patterns detected: {result.patterns_detected}")
    print(
        f"Tools created: {result.hooks_created} hooks, {result.replacements_created} replacements"
    )

    savings_pct = (
        ((result.baseline_tokens - result.rlm_tokens) / result.baseline_tokens * 100)
        if result.baseline_tokens > 0
        else 0
    )
    print(f"Token savings: {savings_pct:+.1f}%")
    print(f"Duration: {result.duration_seconds / 60:.1f} minutes")
    print()
    print(f"Report: {pdf_path}")
    print(f"Results: {output_dir / 'results.json'}")
    print(f"Evidence store: {output_dir / 'evidence_store'}")


if __name__ == "__main__":
    main()
