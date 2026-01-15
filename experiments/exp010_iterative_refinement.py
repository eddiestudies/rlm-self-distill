#!/usr/bin/env python3
"""
Experiment 010: Iterative Tool Refinement

The model creates a tool, tests it, sees results, and iterates to improve.

Loop:
1. Model creates/refines a detection tool
2. Test against labeled dataset
3. Show model: true positives, false positives, false negatives
4. Model creates improved version
5. Repeat N times

Tracks precision/recall/F1 over iterations to see if model learns.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from self_distill.clients.ollama_client import OllamaClient
from self_distill.datasets import load_ai4privacy


@dataclass
class TestResult:
    """Result of testing a tool on a single sample."""

    sample_id: str
    text: str
    expected_types: set[str]
    detected_types: set[str]
    detected: list[dict]

    @property
    def true_positive_types(self) -> set[str]:
        """Types correctly detected."""
        return self.expected_types & self.detected_types

    @property
    def false_positive_types(self) -> set[str]:
        """Types detected but not expected."""
        return self.detected_types - self.expected_types

    @property
    def false_negative_types(self) -> set[str]:
        """Types expected but not detected."""
        return self.expected_types - self.detected_types


@dataclass
class IterationMetrics:
    """Metrics for one iteration."""

    iteration: int
    tool_code: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    error: str | None = None

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def to_dict(self):
        return {
            "iteration": self.iteration,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tn": self.true_negatives,
            "error": self.error,
        }


def extract_code_block(text: str) -> str | None:
    """Extract Python code from markdown."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def run_tool(code: str, text: str) -> tuple[list[dict], str | None]:
    """Execute tool code and return results."""
    try:
        namespace = {"__builtins__": __builtins__}
        exec(code, namespace)

        if "detect" not in namespace:
            return [], "No 'detect' function found"

        result = namespace["detect"](text)
        if not isinstance(result, list):
            return [], f"Expected list, got {type(result)}"

        return result, None
    except Exception as e:
        return [], f"{type(e).__name__}: {str(e)}"


def test_tool(
    code: str, samples: list[dict]
) -> tuple[IterationMetrics, list[TestResult]]:
    """Test a tool against all samples - evaluates by PII TYPE matching."""
    metrics = IterationMetrics(iteration=0, tool_code=code)
    results = []

    for sample in samples:
        detected, error = run_tool(code, sample["text"])

        if error and not metrics.error:
            metrics.error = error

        # Extract detected types
        detected_types = {d.get("type", "UNKNOWN") for d in detected}
        expected_types = set(sample["entities"])

        result = TestResult(
            sample_id=sample["id"],
            text=sample["text"],
            expected_types=expected_types,
            detected_types=detected_types,
            detected=detected,
        )
        results.append(result)

        # Count by TYPE, not by sample
        metrics.true_positives += len(result.true_positive_types)
        metrics.false_positives += len(result.false_positive_types)
        metrics.false_negatives += len(result.false_negative_types)

    return metrics, results


def format_feedback(
    metrics: IterationMetrics, results: list[TestResult], max_examples: int = 3
) -> str:
    """Format test results as feedback for the model."""

    # Aggregate type-level errors across all samples
    all_fp_types: dict[str, int] = {}  # type -> count of false positives
    all_fn_types: dict[str, int] = {}  # type -> count of false negatives
    all_tp_types: dict[str, int] = {}  # type -> count of true positives

    fp_examples: list[tuple[str, str, set]] = []  # (text, detected_types, fp_types)
    fn_examples: list[tuple[str, set, set]] = []  # (text, expected_types, fn_types)

    for r in results:
        for t in r.false_positive_types:
            all_fp_types[t] = all_fp_types.get(t, 0) + 1
        for t in r.false_negative_types:
            all_fn_types[t] = all_fn_types.get(t, 0) + 1
        for t in r.true_positive_types:
            all_tp_types[t] = all_tp_types.get(t, 0) + 1

        if r.false_positive_types and len(fp_examples) < max_examples:
            fp_examples.append((r.text, r.detected_types, r.false_positive_types))
        if r.false_negative_types and len(fn_examples) < max_examples:
            fn_examples.append((r.text, r.expected_types, r.false_negative_types))

    feedback = f"""## Test Results (Iteration {metrics.iteration})

**Metrics (by PII type, not by sample):**
- Precision: {metrics.precision:.1%} ({metrics.true_positives} correct type detections / {metrics.true_positives + metrics.false_positives} total type detections)
- Recall: {metrics.recall:.1%} ({metrics.true_positives} types detected / {metrics.true_positives + metrics.false_negatives} expected types)
- F1 Score: {metrics.f1:.3f}

**Type-level breakdown:**
- True Positives: {metrics.true_positives} (correctly detected the right PII type)
- False Positives: {metrics.false_positives} (detected wrong PII type)
- False Negatives: {metrics.false_negatives} (missed a PII type that was present)
"""

    if metrics.error:
        feedback += f"\n**Error in tool:** {metrics.error}\n"

    # Show which types are problematic
    if all_fp_types:
        sorted_fps = sorted(all_fp_types.items(), key=lambda x: -x[1])[:5]
        feedback += "\n**Most common FALSE POSITIVE types (detecting these when not present):**\n"
        for t, count in sorted_fps:
            feedback += f"  - {t}: {count} times\n"

    if all_fn_types:
        sorted_fns = sorted(all_fn_types.items(), key=lambda x: -x[1])[:5]
        feedback += (
            "\n**Most common MISSED types (not detecting these when present):**\n"
        )
        for t, count in sorted_fns:
            feedback += f"  - {t}: {count} times\n"

    # Show examples
    if fp_examples:
        feedback += "\n**False Positive Examples (wrong type detected):**\n"
        for text, detected, fp_types in fp_examples:
            feedback += f'- Text: "{text[:80]}..."\n'
            feedback += f"  Wrongly detected: {fp_types}\n"

    if fn_examples:
        feedback += "\n**False Negative Examples (type missed):**\n"
        for text, expected, fn_types in fn_examples:
            feedback += f'- Text: "{text[:80]}..."\n'
            feedback += f"  Missed types: {fn_types} (expected: {expected})\n"

    return feedback


def create_initial_tool(client: OllamaClient, model: str, samples: list[dict]) -> str:
    """Create initial tool from examples."""
    # Show a few examples
    examples = samples[:5]
    examples_text = "\n".join(
        [
            f'- Text: "{s["text"][:150]}..."\n  PII types: {s["entities"]}'
            for s in examples
        ]
    )

    prompt = f"""Create a Python function to detect PII (Personal Identifiable Information) in text.

Here are some example texts with their PII types:
{examples_text}

Requirements:
1. Function signature: `def detect(text: str) -> list[dict]`
2. Return format: `[{{"type": "PII_TYPE", "value": "matched", "start": int, "end": int}}]`
3. Use regex patterns that generalize (don't hardcode specific values)
4. Focus on common PII: emails, phone numbers, SSNs, names, addresses

Return ONLY the Python code in a ```python block, no explanation needed.

```python
import re

def detect(text: str) -> list[dict]:
    # Your code here
```"""

    response = client.completion(prompt, model)
    code = extract_code_block(response)
    return code or ""


def refine_tool(
    client: OllamaClient, model: str, current_code: str, feedback: str, iteration: int
) -> str:
    """Ask model to refine the tool based on feedback."""
    prompt = f"""You are iteratively improving a PII detection tool. Here is the current code:

```python
{current_code}
```

{feedback}

Based on these results, create an IMPROVED version of the tool.

**Guidelines:**
- If precision is low (many false positives): make patterns more specific
- If recall is low (many false negatives): make patterns more general or add new patterns
- Look at the false positive/negative examples to understand what's going wrong
- Don't hardcode specific values from examples - create generalizable patterns

Return ONLY the improved Python code in a ```python block.

```python
import re

def detect(text: str) -> list[dict]:
    # Your improved code here
```"""

    response = client.completion(prompt, model)
    code = extract_code_block(response)
    return code or current_code  # Fall back to current if extraction fails


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Exp010: Iterative Tool Refinement")
    parser.add_argument("--model", default="qwen2.5-coder:32b", help="Model to use")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument("--dataset-size", type=int, default=100, help="Dataset size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 010: Iterative Tool Refinement")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Iterations: {args.iterations}")
    print(f"Dataset size: {args.dataset_size}")
    print()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiment_outputs/exp010_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = output_dir / "tools"
    tools_dir.mkdir(exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = load_ai4privacy(limit=args.dataset_size)
    samples = [
        {
            "id": f"pii_{i}",
            "text": item.source_text,
            "entities": [e.label for e in item.entities],
        }
        for i, item in enumerate(dataset)
    ]
    print(f"Loaded {len(samples)} samples")

    # Initialize
    client = OllamaClient()
    all_metrics: list[IterationMetrics] = []

    # Create initial tool
    print("\nCreating initial tool...")
    current_code = create_initial_tool(client, args.model, samples)

    if not current_code:
        print("Failed to create initial tool!")
        return

    # Save initial tool
    (tools_dir / "iter_000.py").write_text(current_code)
    print("Initial tool created")

    # Iterative refinement loop
    print("\n" + "=" * 60)
    print("Starting iterative refinement...")
    print("=" * 60)

    best_f1 = 0.0
    best_iteration = 0
    best_code = current_code

    pbar = tqdm(range(args.iterations), desc="Refining")

    for i in pbar:
        # Test current tool
        metrics, results = test_tool(current_code, samples)
        metrics.iteration = i
        metrics.tool_code = current_code
        all_metrics.append(metrics)

        # Track best
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_iteration = i
            best_code = current_code

        # Update progress bar
        pbar.set_postfix(
            {
                "P": f"{metrics.precision:.1%}",
                "R": f"{metrics.recall:.1%}",
                "F1": f"{metrics.f1:.3f}",
                "best": f"{best_f1:.3f}@{best_iteration}",
            }
        )

        if args.verbose:
            tqdm.write(
                f"\nIteration {i}: P={metrics.precision:.1%} R={metrics.recall:.1%} F1={metrics.f1:.3f}"
            )
            if metrics.error:
                tqdm.write(f"  Error: {metrics.error}")

        # Save tool every 10 iterations
        if i % 10 == 0:
            (tools_dir / f"iter_{i:03d}.py").write_text(current_code)

        # Check for early stopping (perfect score or no improvement for 20 iterations)
        if metrics.f1 >= 0.99:
            tqdm.write(f"\nReached near-perfect F1 at iteration {i}!")
            break

        if i > 20 and all(m.f1 <= best_f1 for m in all_metrics[-20:]):
            tqdm.write("\nNo improvement for 20 iterations, stopping early.")
            break

        # Generate feedback and refine
        feedback = format_feedback(metrics, results)
        current_code = refine_tool(client, args.model, current_code, feedback, i)

    # Final test with best tool
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    final_metrics, _ = test_tool(best_code, samples)

    # Save best tool
    (tools_dir / "best.py").write_text(best_code)

    # Save results
    results = {
        "metadata": {
            "timestamp": timestamp,
            "model": args.model,
            "iterations": len(all_metrics),
            "dataset_size": len(samples),
        },
        "best": {
            "iteration": best_iteration,
            "f1": best_f1,
            "precision": final_metrics.precision,
            "recall": final_metrics.recall,
        },
        "history": [m.to_dict() for m in all_metrics],
    }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nBest result at iteration {best_iteration}:")
    print(f"  Precision: {final_metrics.precision:.1%}")
    print(f"  Recall: {final_metrics.recall:.1%}")
    print(f"  F1 Score: {best_f1:.3f}")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Best tool saved to: {tools_dir / 'best.py'}")

    # Print F1 progression
    print("\nF1 Score Progression:")
    step = max(1, len(all_metrics) // 20)
    for i in range(0, len(all_metrics), step):
        m = all_metrics[i]
        bar = "█" * int(m.f1 * 40)
        print(f"  {i:3d}: {bar} {m.f1:.3f}")


if __name__ == "__main__":
    main()
