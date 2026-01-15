#!/usr/bin/env python3
"""
Experiment 009: Isolated Tool Creation and Validation

Tests tool creation in isolation, without RLM complexity.

For each sample:
1. Ask LLM to create a tool that solves this type of problem
2. Ask LLM to generate a "near miss" adversarial example
3. Run tool against original sample (should pass)
4. Run tool against adversarial example (should handle correctly)
5. Test tool against full dataset to find coverage/overlap

This validates the fundamental tool creation mechanism works before
adding optimization complexity.
"""

import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from self_distill.clients.ollama_client import OllamaClient
from self_distill.datasets import load_ai4privacy


@dataclass
class ToolResult:
    """Result of running a tool on a sample."""
    sample_id: str
    sample_text: str
    tool_output: str | dict | None
    error: str | None
    passed: bool


@dataclass
class CreatedTool:
    """A tool created by the LLM."""
    name: str
    description: str
    code: str
    original_sample_id: str
    original_sample_text: str
    adversarial_example: str
    adversarial_expected: str

    # Validation results
    original_result: Optional[ToolResult] = None
    adversarial_result: Optional[ToolResult] = None
    coverage_results: list[ToolResult] = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "original_sample_id": self.original_sample_id,
            "original_sample_text": self.original_sample_text[:200],
            "adversarial_example": self.adversarial_example,
            "adversarial_expected": self.adversarial_expected,
            "original_passed": self.original_result.passed if self.original_result else None,
            "adversarial_passed": self.adversarial_result.passed if self.adversarial_result else None,
            "coverage_count": len([r for r in self.coverage_results if r.passed]),
            "coverage_total": len(self.coverage_results),
        }


def extract_code_block(text: str) -> str | None:
    """Extract Python code from markdown code block."""
    # Try ```python first
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ```
    match = re.search(r'```\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def extract_json_block(text: str) -> dict | None:
    """Extract JSON from response."""
    # Try ```json first
    match = re.search(r'```json\n(.*?)```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except:
            pass

    # Try to find JSON object directly
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return None


def create_tool_for_sample(client: OllamaClient, model: str, sample: dict) -> CreatedTool | None:
    """Ask LLM to create a tool for this sample type."""

    prompt = f"""You are creating a reusable Python tool for detecting PII (Personal Identifiable Information).

Given this example:
Text: {sample['text']}
Expected PII types: {sample.get('entities', [])}

Create a Python function that can detect THIS TYPE of PII pattern.

Requirements:
1. Function signature: `def detect(text: str) -> list[dict]`
2. Return format: `[{{"type": "PII_TYPE", "value": "matched_text", "start": int, "end": int}}]`
3. Use regex or string matching - be specific to the pattern you see
4. Focus on ONE specific pattern type (e.g., emails, SSNs, names in context)

Respond with:
1. Tool name (snake_case)
2. Brief description
3. The Python code in a ```python block

Example response format:
Name: email_detector
Description: Detects email addresses using regex
```python
import re

def detect(text: str) -> list[dict]:
    pattern = r'[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{{2,}}'
    results = []
    for match in re.finditer(pattern, text):
        results.append({{
            "type": "EMAIL",
            "value": match.group(),
            "start": match.start(),
            "end": match.end()
        }})
    return results
```"""

    response = client.completion(prompt, model)

    # Parse response
    lines = response.strip().split('\n')
    name = None
    description = None

    for line in lines:
        if line.lower().startswith('name:'):
            name = line.split(':', 1)[1].strip().replace(' ', '_').lower()
        elif line.lower().startswith('description:'):
            description = line.split(':', 1)[1].strip()

    code = extract_code_block(response)

    if not code or not name:
        return None

    return CreatedTool(
        name=name,
        description=description or "No description",
        code=code,
        original_sample_id=sample['id'],
        original_sample_text=sample['text'],
        adversarial_example="",
        adversarial_expected="",
    )


def generate_adversarial(client: OllamaClient, model: str, tool: CreatedTool) -> tuple[str, str]:
    """Generate a near-miss adversarial example for the tool."""

    prompt = f"""Given this PII detection tool:

Tool: {tool.name}
Description: {tool.description}
Original text it was designed for: {tool.original_sample_text[:300]}

Generate a "near miss" adversarial example - text that looks similar but should NOT trigger this detector (or should trigger differently).

For example:
- If it detects "john.smith@email.com", generate text with "john.smith[at]email.com" or "john.smith @ email . com"
- If it detects SSN "123-45-6789", generate text with "123-456-789" (wrong format)
- If it detects phone "(555) 234-5678", generate text with "555.234.5678" (different format)

Respond with JSON:
```json
{{
    "adversarial_text": "The near-miss text here",
    "expected_behavior": "Should not detect / Should detect as TYPE / etc"
}}
```"""

    response = client.completion(prompt, model)
    data = extract_json_block(response)

    if data:
        return data.get("adversarial_text", ""), data.get("expected_behavior", "")

    return "", ""


def run_tool(tool: CreatedTool, text: str, sample_id: str) -> ToolResult:
    """Execute a tool against a text sample."""
    try:
        # Create a namespace and execute the tool code
        namespace = {"__builtins__": __builtins__}
        exec(tool.code, namespace)

        if 'detect' not in namespace:
            return ToolResult(
                sample_id=sample_id,
                sample_text=text,
                tool_output=None,
                error="Tool has no 'detect' function",
                passed=False,
            )

        result = namespace['detect'](text)

        # Check if it returned valid results
        passed = isinstance(result, list) and len(result) > 0

        return ToolResult(
            sample_id=sample_id,
            sample_text=text,
            tool_output=result,
            error=None,
            passed=passed,
        )

    except Exception as e:
        return ToolResult(
            sample_id=sample_id,
            sample_text=text,
            tool_output=None,
            error=f"{type(e).__name__}: {str(e)}",
            passed=False,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Exp009: Tool Creation Validation")
    parser.add_argument("--model", default="qwen2.5-coder:32b", help="Model to use")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to create tools for")
    parser.add_argument("--dataset-size", type=int, default=100, help="Dataset size for coverage testing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 009: Isolated Tool Creation Validation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Creating tools for: {args.samples} samples")
    print(f"Testing coverage on: {args.dataset_size} samples")
    print()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiment_outputs/exp009_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = output_dir / "tools"
    tools_dir.mkdir(exist_ok=True)

    # Load dataset
    print("Loading AI4Privacy dataset...")
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

    # Initialize client
    client = OllamaClient()

    # Phase 1: Create tools
    print("\n" + "=" * 60)
    print("Phase 1: Creating Tools")
    print("=" * 60)

    created_tools: list[CreatedTool] = []

    for i in tqdm(range(min(args.samples, len(samples))), desc="Creating tools"):
        sample = samples[i]

        if args.verbose:
            tqdm.write(f"\nCreating tool for sample {sample['id']}...")

        tool = create_tool_for_sample(client, args.model, sample)

        if tool:
            # Save tool code
            tool_path = tools_dir / f"{tool.name}.py"
            tool_path.write_text(tool.code)

            created_tools.append(tool)
            tqdm.write(f"  Created: {tool.name}")
        else:
            tqdm.write(f"  Failed to create tool for {sample['id']}")

    print(f"\nCreated {len(created_tools)} tools")

    # Phase 2: Generate adversarial examples
    print("\n" + "=" * 60)
    print("Phase 2: Generating Adversarial Examples")
    print("=" * 60)

    for tool in tqdm(created_tools, desc="Generating adversarials"):
        adv_text, adv_expected = generate_adversarial(client, args.model, tool)
        tool.adversarial_example = adv_text
        tool.adversarial_expected = adv_expected

        if args.verbose and adv_text:
            tqdm.write(f"  {tool.name}: {adv_text[:50]}...")

    # Phase 3: Validate tools
    print("\n" + "=" * 60)
    print("Phase 3: Validating Tools")
    print("=" * 60)

    for tool in tqdm(created_tools, desc="Validating"):
        # Test on original sample
        original_sample = next((s for s in samples if s['id'] == tool.original_sample_id), None)
        if original_sample:
            tool.original_result = run_tool(tool, original_sample['text'], original_sample['id'])

        # Test on adversarial
        if tool.adversarial_example:
            tool.adversarial_result = run_tool(tool, tool.adversarial_example, "adversarial")

        if args.verbose:
            orig_status = "PASS" if tool.original_result and tool.original_result.passed else "FAIL"
            adv_status = "PASS" if tool.adversarial_result and not tool.adversarial_result.passed else "FAIL"
            tqdm.write(f"  {tool.name}: original={orig_status}, adversarial={adv_status}")

    # Phase 4: Test coverage across dataset
    print("\n" + "=" * 60)
    print("Phase 4: Testing Coverage Across Dataset")
    print("=" * 60)

    coverage_matrix = {}  # tool_name -> list of (sample_id, passed)

    for tool in tqdm(created_tools, desc="Testing coverage"):
        coverage_matrix[tool.name] = []

        for sample in samples:
            result = run_tool(tool, sample['text'], sample['id'])
            tool.coverage_results.append(result)
            coverage_matrix[tool.name].append((sample['id'], result.passed))

    # Phase 5: Analyze overlap
    print("\n" + "=" * 60)
    print("Phase 5: Analyzing Tool Overlap")
    print("=" * 60)

    overlap_analysis = {}

    for tool in created_tools:
        matching_samples = [r.sample_id for r in tool.coverage_results if r.passed]
        overlap_analysis[tool.name] = {
            "original": tool.original_sample_id,
            "matches": matching_samples,
            "match_count": len(matching_samples),
            "match_rate": len(matching_samples) / len(samples) * 100,
        }

        print(f"  {tool.name}: matches {len(matching_samples)}/{len(samples)} ({len(matching_samples)/len(samples)*100:.1f}%)")

    # Generate report
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    results = {
        "metadata": {
            "timestamp": timestamp,
            "model": args.model,
            "samples_for_tools": args.samples,
            "dataset_size": args.dataset_size,
        },
        "tools": [t.to_dict() for t in created_tools],
        "overlap_analysis": overlap_analysis,
        "summary": {
            "tools_created": len(created_tools),
            "original_pass_rate": sum(1 for t in created_tools if t.original_result and t.original_result.passed) / len(created_tools) * 100 if created_tools else 0,
            "adversarial_correct_rate": sum(1 for t in created_tools if t.adversarial_result and not t.adversarial_result.passed) / len(created_tools) * 100 if created_tools else 0,
            "avg_coverage": sum(len([r for r in t.coverage_results if r.passed]) for t in created_tools) / len(created_tools) if created_tools else 0,
        },
    }

    # Save results
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\nTools created: {results['summary']['tools_created']}")
    print(f"Original sample pass rate: {results['summary']['original_pass_rate']:.1f}%")
    print(f"Adversarial correct rate: {results['summary']['adversarial_correct_rate']:.1f}%")
    print(f"Average coverage per tool: {results['summary']['avg_coverage']:.1f} samples")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Tools saved to: {tools_dir}")

    # Print individual tool results
    print("\n" + "-" * 60)
    print("Individual Tool Results:")
    print("-" * 60)

    for tool in created_tools:
        orig = "PASS" if tool.original_result and tool.original_result.passed else "FAIL"
        adv = "CORRECT" if tool.adversarial_result and not tool.adversarial_result.passed else "WRONG"
        coverage = len([r for r in tool.coverage_results if r.passed])
        print(f"  {tool.name:30} | orig: {orig:4} | adv: {adv:7} | coverage: {coverage:3}/{len(samples)}")


if __name__ == "__main__":
    main()
