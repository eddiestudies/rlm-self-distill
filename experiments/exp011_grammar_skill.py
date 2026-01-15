#!/usr/bin/env python3
"""
Experiment 011: Grammar Skill Development

Creates a single grammar skill through iterative refinement.
The model creates a function to determine if sentences are grammatically correct.

Architecture:
    1. Load CoLA dataset (grammaticality judgments)
    2. Model creates initial grammar skill
    3. Test against dataset, compute accuracy
    4. Show model the errors
    5. Model refines the skill
    6. Iterate until convergence or max iterations
"""

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from self_distill.clients.ollama_client import OllamaClient
from self_distill.datasets import CoLADataset, Split
from self_distill.skills.base import CodeSkill, AlwaysTrigger
from self_distill.skills.registry import SkillRegistry


@dataclass
class TestResult:
    """Result of testing on one sample."""
    sentence: str
    expected: bool  # True = grammatical
    predicted: bool | None
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.predicted == self.expected


@dataclass
class IterationMetrics:
    """Metrics for one iteration."""
    iteration: int
    accuracy: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    errors: int
    skill_code: str

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "accuracy": self.accuracy,
            "tp": self.true_positives,
            "tn": self.true_negatives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "errors": self.errors,
        }


def extract_code_block(text: str) -> str | None:
    """Extract Python code from markdown."""
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def test_skill(skill: CodeSkill, samples: list[dict]) -> tuple[IterationMetrics, list[TestResult]]:
    """Test a skill against all samples."""
    results = []
    tp = tn = fp = fn = errors = 0

    for sample in samples:
        sentence = sample["sentence"]
        expected = sample["label"]  # True = grammatical

        result = skill.run(sentence)

        if result.error:
            results.append(TestResult(
                sentence=sentence,
                expected=expected,
                predicted=None,
                error=result.error,
            ))
            errors += 1
            continue

        # Convert output to boolean
        predicted = bool(result.output)

        results.append(TestResult(
            sentence=sentence,
            expected=expected,
            predicted=predicted,
        ))

        if expected and predicted:
            tp += 1
        elif not expected and not predicted:
            tn += 1
        elif not expected and predicted:
            fp += 1
        else:
            fn += 1

    total = len(samples) - errors
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return IterationMetrics(
        iteration=0,
        accuracy=accuracy,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        errors=errors,
        skill_code=skill.code,
    ), results


def simple_pos_tag(sentence: str) -> str:
    """Simple POS tagger for feedback - helps model see patterns."""
    import re

    # Common word -> POS mappings
    determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every', 'each', 'all', 'both', 'few', 'many', 'much', 'most'}
    pronouns = {'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us', 'they', 'them', 'who', 'whom', 'what', 'which', 'that', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves'}
    auxiliaries = {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could', 'must'}
    prepositions = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over'}
    conjunctions = {'and', 'or', 'but', 'nor', 'yet', 'so', 'for', 'because', 'although', 'while', 'if', 'when', 'unless', 'since', 'though'}

    words = re.findall(r'\b\w+\b', sentence.lower())
    tags = []

    for word in words:
        if word in determiners:
            tags.append('DET')
        elif word in pronouns:
            tags.append('PRON')
        elif word in auxiliaries:
            tags.append('AUX')
        elif word in prepositions:
            tags.append('PREP')
        elif word in conjunctions:
            tags.append('CONJ')
        elif word.endswith('ly'):
            tags.append('ADV')
        elif word.endswith(('ing', 'ed', 's')) and len(word) > 4:
            tags.append('VERB')
        elif word.endswith(('tion', 'ness', 'ment', 'ity', 'er', 'or')):
            tags.append('NOUN')
        elif word.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible')):
            tags.append('ADJ')
        else:
            tags.append('?')  # Unknown

    return ' '.join(tags)


def format_feedback(metrics: IterationMetrics, results: list[TestResult], max_examples: int = 5) -> str:
    """Format test results as feedback for the model."""
    # Collect error examples
    fps = [r for r in results if r.predicted is not None and r.predicted and not r.expected][:max_examples]
    fns = [r for r in results if r.predicted is not None and not r.predicted and r.expected][:max_examples]
    errs = [r for r in results if r.error][:max_examples]

    feedback = f"""## Test Results (Iteration {metrics.iteration})

**Accuracy: {metrics.accuracy:.1%}** ({metrics.true_positives + metrics.true_negatives}/{metrics.true_positives + metrics.true_negatives + metrics.false_positives + metrics.false_negatives} correct)

**Breakdown:**
- True Positives: {metrics.true_positives} (correctly identified grammatical sentences)
- True Negatives: {metrics.true_negatives} (correctly identified ungrammatical sentences)
- False Positives: {metrics.false_positives} (said grammatical but was ungrammatical)
- False Negatives: {metrics.false_negatives} (said ungrammatical but was grammatical)
- Errors: {metrics.errors} (skill crashed)
"""

    if errs:
        feedback += "\n**Runtime Errors (fix these first!):**\n"
        for r in errs:
            feedback += f"- \"{r.sentence[:60]}...\"\n  Error: {r.error}\n"

    if fps:
        feedback += "\n**False Positives (your skill said grammatical, but these are UNGRAMMATICAL):**\n"
        for r in fps:
            pos = simple_pos_tag(r.sentence)
            feedback += f"- \"{r.sentence}\"\n  POS: {pos}\n"

    if fns:
        feedback += "\n**False Negatives (your skill said ungrammatical, but these are GRAMMATICAL):**\n"
        for r in fns:
            pos = simple_pos_tag(r.sentence)
            feedback += f"- \"{r.sentence}\"\n  POS: {pos}\n"

    return feedback


def create_initial_skill(client: OllamaClient, model: str, samples: list[dict]) -> str:
    """Ask model to create initial grammar skill."""
    prompt = """Create a Python function that determines if an English sentence is grammatically correct.

Your task is to build a general grammar checker that works on ANY English sentence. Do NOT rely on specific vocabulary - focus purely on GRAMMATICAL STRUCTURE.

**Recommended Approach - POS-based Grammar Patterns:**

1. Build a simple part-of-speech (POS) tagger using word patterns and suffixes:
   - Words ending in -ly are often adverbs
   - Words ending in -ed, -ing are often verbs
   - Words ending in -tion, -ness, -ment are often nouns
   - Use word lists for determiners (the, a, an), pronouns (he, she, it), auxiliaries (is, was, have, had)

2. Convert sentences to POS sequences like: DET NOUN VERB DET NOUN
   Example: "The cat chased the mouse" -> "DET NOUN VERB DET NOUN"

3. Check if the POS pattern is valid English grammar:
   - Valid: DET NOUN VERB, NOUN VERB NOUN, DET ADJ NOUN VERB
   - Invalid: DET DET NOUN, VERB VERB NOUN, NOUN NOUN VERB VERB

4. Check structural constraints:
   - Subject-verb agreement (singular/plural consistency)
   - Proper clause structure
   - No dangling modifiers or incomplete phrases

**Requirements:**
1. Function signature: `def solve(text: str) -> bool`
2. Return `True` if grammatically correct, `False` otherwise
3. ONLY use Python standard library (re, string, etc.) - NO external packages
4. Focus on grammar STRUCTURE, not vocabulary or meaning
5. Be robust - don't crash on edge cases

Return ONLY the Python code in a ```python block.

```python
def solve(text: str) -> bool:
    # Your implementation here
    pass
```"""

    response = client.completion(prompt, model)
    return extract_code_block(response) or ""


def refine_skill(client: OllamaClient, model: str, current_code: str, feedback: str) -> str:
    """Ask model to refine the skill based on feedback."""
    prompt = f"""You are improving a grammar checking function that uses POS-based pattern matching.

**Current code:**
```python
{current_code}
```

{feedback}

**Analyze the errors:**
- For false positives: What POS patterns are you incorrectly accepting as valid?
- For false negatives: What valid grammatical structures are you rejecting?

**Improvement strategy:**
1. Convert the example sentences to POS sequences to understand the patterns
2. Adjust your grammar rules based on the patterns you see
3. Remember: Focus on STRUCTURE (POS patterns), not specific words
4. Consider adding/removing pattern rules based on the errors

**Guidelines:**
- Fix runtime errors first
- ONLY use Python standard library (re, string, etc.) - NO external packages
- Improve your POS tagger if words are being misclassified
- Add new valid patterns or remove overly strict rules
- Balance precision and recall

Return ONLY the improved Python code in a ```python block.

```python
def solve(text: str) -> bool:
    # Your improved implementation
    pass
```"""

    response = client.completion(prompt, model)
    return extract_code_block(response) or current_code


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Exp011: Grammar Skill Development")
    parser.add_argument("--model", default="deepseek-r1:70b", help="Model to use")
    parser.add_argument("--iterations", type=int, default=50, help="Max iterations")
    parser.add_argument("--train-size", type=int, default=200, help="Training set size")
    parser.add_argument("--test-size", type=int, default=100, help="Test set size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 011: Grammar Skill Development")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Iterations: {args.iterations}")
    print(f"Train/Test: {args.train_size}/{args.test_size}")
    print()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiment_outputs/exp011_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    skills_dir = output_dir / "skills"
    skills_dir.mkdir(exist_ok=True)

    # Load dataset
    print("Loading CoLA dataset...")
    cola_train = CoLADataset(split=Split.TRAIN)
    cola_dev = CoLADataset(split=Split.DEV)

    # Prepare samples (CoLA: question is the sentence, answer is "0" or "1")
    train_samples = []
    for i, item in enumerate(cola_train):
        if i >= args.train_size:
            break
        train_samples.append({
            "sentence": item.question,  # The sentence itself
            "label": item.answer == "1",  # 1 = grammatical, 0 = ungrammatical
        })

    test_samples = []
    for i, item in enumerate(cola_dev):
        if i >= args.test_size:
            break
        test_samples.append({
            "sentence": item.question,
            "label": item.answer == "1",
        })

    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Count class distribution
    train_pos = sum(1 for s in train_samples if s["label"])
    test_pos = sum(1 for s in test_samples if s["label"])
    print(f"Train distribution: {train_pos} grammatical / {len(train_samples) - train_pos} ungrammatical")
    print(f"Test distribution: {test_pos} grammatical / {len(test_samples) - test_pos} ungrammatical")

    # Initialize
    client = OllamaClient()
    registry = SkillRegistry()
    all_metrics: list[IterationMetrics] = []

    # Create initial skill
    print("\nCreating initial grammar skill...")
    initial_code = create_initial_skill(client, args.model, train_samples)

    if not initial_code:
        print("Failed to create initial skill!")
        return

    skill = CodeSkill(
        name="grammar_checker",
        code=initial_code,
        description="Checks if sentences are grammatically correct",
    )

    # Save initial skill
    (skills_dir / "iter_000.py").write_text(initial_code)
    print(f"Initial skill created (valid: {skill.is_valid})")

    if not skill.is_valid:
        print(f"Compile error: {skill._compile_error}")

    # Iterative refinement
    print("\n" + "=" * 60)
    print("Starting iterative refinement...")
    print("=" * 60)

    best_accuracy = 0.0
    best_iteration = 0
    best_code = initial_code
    no_improvement_count = 0

    pbar = tqdm(range(args.iterations), desc="Refining")

    for i in pbar:
        # Test on training set
        metrics, results = test_skill(skill, train_samples)
        metrics.iteration = i
        all_metrics.append(metrics)

        # Track best (on training)
        if metrics.accuracy > best_accuracy:
            best_accuracy = metrics.accuracy
            best_iteration = i
            best_code = skill.code
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Update progress
        pbar.set_postfix({
            "acc": f"{metrics.accuracy:.1%}",
            "best": f"{best_accuracy:.1%}@{best_iteration}",
            "errs": metrics.errors,
        })

        if args.verbose:
            tqdm.write(f"\nIter {i}: Accuracy={metrics.accuracy:.1%} (TP={metrics.true_positives}, TN={metrics.true_negatives}, FP={metrics.false_positives}, FN={metrics.false_negatives}, Err={metrics.errors})")

        # Save skill every 10 iterations
        if i % 10 == 0:
            (skills_dir / f"iter_{i:03d}.py").write_text(skill.code)

        # Early stopping
        if metrics.accuracy >= 0.95:
            tqdm.write(f"\nReached 95% accuracy at iteration {i}!")
            break

        if no_improvement_count >= 15:
            tqdm.write(f"\nNo improvement for 15 iterations, stopping early.")
            break

        # Generate feedback and refine
        feedback = format_feedback(metrics, results)
        new_code = refine_skill(client, args.model, skill.code, feedback)
        skill.update_code(new_code)

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Use best skill
    best_skill = CodeSkill(
        name="grammar_checker_best",
        code=best_code,
        description="Best grammar checker",
    )

    final_metrics, final_results = test_skill(best_skill, test_samples)

    # Save best skill
    (skills_dir / "best.py").write_text(best_code)

    # Register in skill registry
    registry.register(best_skill, AlwaysTrigger())
    registry.save(output_dir)

    # Save results
    results = {
        "metadata": {
            "timestamp": timestamp,
            "model": args.model,
            "iterations": len(all_metrics),
            "train_size": len(train_samples),
            "test_size": len(test_samples),
        },
        "best": {
            "iteration": best_iteration,
            "train_accuracy": best_accuracy,
            "test_accuracy": final_metrics.accuracy,
        },
        "test_results": {
            "accuracy": final_metrics.accuracy,
            "tp": final_metrics.true_positives,
            "tn": final_metrics.true_negatives,
            "fp": final_metrics.false_positives,
            "fn": final_metrics.false_negatives,
            "errors": final_metrics.errors,
        },
        "history": [m.to_dict() for m in all_metrics],
    }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\nBest training accuracy: {best_accuracy:.1%} at iteration {best_iteration}")
    print(f"\nTest Results:")
    print(f"  Accuracy: {final_metrics.accuracy:.1%}")
    print(f"  TP: {final_metrics.true_positives}, TN: {final_metrics.true_negatives}")
    print(f"  FP: {final_metrics.false_positives}, FN: {final_metrics.false_negatives}")
    print(f"  Errors: {final_metrics.errors}")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Best skill saved to: {skills_dir / 'best.py'}")

    # Print accuracy progression
    print("\nAccuracy Progression (Training):")
    step = max(1, len(all_metrics) // 15)
    for i in range(0, len(all_metrics), step):
        m = all_metrics[i]
        bar = "█" * int(m.accuracy * 40)
        print(f"  {i:3d}: {bar} {m.accuracy:.1%}")


if __name__ == "__main__":
    main()
