#!/usr/bin/env python3
"""
Experiment 004: Safety Macro Pipeline

Architecture:
1. Safety Macro runs FIRST on all inputs
   - Detects if input contains potentially sensitive data (PII)
   - If detected, runs safety suite to mask/redact BEFORE LLM sees it

2. Completion Rules Check
   - After safety preprocessing, check if existing tools can solve the task
   - Route to grammar tools, math tools, etc.

3. LLM Sub-calls (only when needed)
   - LLM only receives sanitized/masked text
   - Fall back for complex reasoning

Key Safety Benefit: LLM never directly sees raw PII data.
"""

import argparse
import json
import re
import importlib.util
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from tqdm import tqdm
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from self_distill.clients.ollama_client import OllamaClient
from self_distill.datasets import DATA, load_dataset, Split


@dataclass
class SafetyResult:
    """Result from safety macro processing."""

    is_safe: bool
    original_text: str
    sanitized_text: str
    pii_found: list[dict] = field(default_factory=list)
    masking_applied: bool = False


@dataclass
class CompletionResult:
    """Result from completion rule processing."""

    handled_by_tool: bool
    tool_name: str | None
    result: str
    tokens_used: int = 0


@dataclass
class PipelineResult:
    """Full pipeline execution result."""

    task_type: str
    safety_result: SafetyResult
    completion_result: CompletionResult
    llm_fallback_used: bool = False
    llm_fallback_tokens: int = 0
    total_tokens: int = 0


class SafetyMacro:
    """
    Safety macro that runs BEFORE any LLM processing.
    Detects and masks PII so the LLM never sees raw sensitive data.
    """

    def __init__(self):
        self.pii_patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        }

    def check_for_pii(self, text: str) -> list[dict]:
        """Check if text contains PII patterns."""
        found = []
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                found.append(
                    {
                        "type": pii_type,
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        return found

    def mask_pii(self, text: str, pii_items: list[dict]) -> str:
        """Mask PII in text with placeholder tokens."""
        # Sort by position (reverse) to avoid offset issues
        sorted_pii = sorted(pii_items, key=lambda x: x["start"], reverse=True)

        masked = text
        for item in sorted_pii:
            mask = f"[{item['type']}_MASKED]"
            masked = masked[: item["start"]] + mask + masked[item["end"] :]

        return masked

    def run(self, text: str) -> SafetyResult:
        """
        Run safety macro on input text.
        Returns sanitized text safe for LLM processing.
        """
        pii_found = self.check_for_pii(text)

        if pii_found:
            sanitized = self.mask_pii(text, pii_found)
            return SafetyResult(
                is_safe=False,
                original_text=text,
                sanitized_text=sanitized,
                pii_found=pii_found,
                masking_applied=True,
            )
        else:
            return SafetyResult(
                is_safe=True,
                original_text=text,
                sanitized_text=text,
                pii_found=[],
                masking_applied=False,
            )


class CompletionRules:
    """
    Completion rules that can handle tasks without LLM.
    Checks if existing tools can solve the task.
    """

    def __init__(self, tools_dir: Path):
        self.tools_dir = tools_dir
        self.tools: dict[str, Callable] = {}
        self._load_tools()

    def _load_tools(self):
        """Load all available tools from tools directory."""
        for category in ["grammar", "pii", "classifiers"]:
            category_dir = self.tools_dir / category
            if category_dir.exists():
                for tool_file in category_dir.glob("*.py"):
                    if tool_file.name.startswith("__"):
                        continue
                    tool_name = tool_file.stem
                    try:
                        spec = importlib.util.spec_from_file_location(
                            tool_name, tool_file
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            if hasattr(module, "run"):
                                self.tools[tool_name] = module.run
                    except Exception:
                        pass

    def classify_task(self, text: str, is_pii_task: bool = False) -> str:
        """Classify what type of task this is."""
        if is_pii_task:
            return "pii_detection"

        # Check for grammar-related keywords
        grammar_keywords = [
            "grammatical",
            "grammar",
            "acceptable",
            "sentence",
            "correct",
        ]
        if any(kw in text.lower() for kw in grammar_keywords):
            return "grammar"

        # Default to general
        return "general"

    def can_handle(self, task_type: str) -> bool:
        """Check if we have tools to handle this task type."""
        if task_type == "grammar" and "basic_grammar" in self.tools:
            return True
        if task_type == "pii_detection":
            pii_tools = [
                "email_detector",
                "phone_detector",
                "ssn_detector",
                "ip_detector",
            ]
            return any(t in self.tools for t in pii_tools)
        return False

    def execute(self, text: str, task_type: str) -> CompletionResult:
        """Execute completion rules for the given task."""
        if task_type == "grammar" and "basic_grammar" in self.tools:
            try:
                result = self.tools["basic_grammar"](text)
                return CompletionResult(
                    handled_by_tool=True,
                    tool_name="basic_grammar",
                    result=json.dumps(result)
                    if isinstance(result, dict)
                    else str(result),
                )
            except Exception as e:
                return CompletionResult(
                    handled_by_tool=False, tool_name=None, result=f"Tool error: {e}"
                )

        if task_type == "pii_detection":
            # Run all PII detectors
            all_detections = []
            tools_used = []
            for tool_name in [
                "email_detector",
                "phone_detector",
                "ssn_detector",
                "ip_detector",
            ]:
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name](text)
                        if result:
                            all_detections.extend(
                                result if isinstance(result, list) else [result]
                            )
                            tools_used.append(tool_name)
                    except Exception:
                        pass

            return CompletionResult(
                handled_by_tool=bool(tools_used),
                tool_name=",".join(tools_used) if tools_used else None,
                result=json.dumps(all_detections),
            )

        return CompletionResult(handled_by_tool=False, tool_name=None, result="")


class SafetyMacroPipeline:
    """
    Full pipeline: Safety Macro → Completion Rules → LLM Fallback

    Key principle: LLM never sees raw PII data.
    """

    def __init__(self, client: OllamaClient, model: str, tools_dir: Path):
        self.client = client
        self.model = model
        self.safety_macro = SafetyMacro()
        self.completion_rules = CompletionRules(tools_dir)

        # Metrics
        self.total_inputs = 0
        self.pii_masked_count = 0
        self.tool_handled_count = 0
        self.llm_fallback_count = 0
        self.total_tool_tokens = 0
        self.total_llm_tokens = 0

    def process(self, text: str, task_hint: str = "") -> PipelineResult:
        """
        Process input through the safety-first pipeline.

        1. Safety macro checks and masks PII
        2. Completion rules try to handle the task
        3. LLM fallback only sees sanitized text
        """
        self.total_inputs += 1

        # Step 1: Safety Macro (ALWAYS runs first)
        safety_result = self.safety_macro.run(text)
        if safety_result.masking_applied:
            self.pii_masked_count += 1

        # Determine task type
        is_pii_task = "pii" in task_hint.lower() or bool(safety_result.pii_found)
        task_type = self.completion_rules.classify_task(text, is_pii_task)

        # Step 2: Try Completion Rules
        completion_result = CompletionResult(
            handled_by_tool=False, tool_name=None, result=""
        )

        if self.completion_rules.can_handle(task_type):
            # For PII tasks, we already have the detection from safety macro
            if task_type == "pii_detection" and safety_result.pii_found:
                completion_result = CompletionResult(
                    handled_by_tool=True,
                    tool_name="safety_macro",
                    result=json.dumps(safety_result.pii_found),
                )
            else:
                completion_result = self.completion_rules.execute(
                    safety_result.sanitized_text,  # Use sanitized text!
                    task_type,
                )

        if completion_result.handled_by_tool:
            self.tool_handled_count += 1
            return PipelineResult(
                task_type=task_type,
                safety_result=safety_result,
                completion_result=completion_result,
                llm_fallback_used=False,
                total_tokens=0,  # No LLM tokens used
            )

        # Step 3: LLM Fallback (only sees SANITIZED text)
        self.llm_fallback_count += 1

        # Build prompt with sanitized text only
        prompt = f"""Analyze the following text and provide your assessment.
Note: Any [TYPE_MASKED] tokens indicate redacted sensitive information.

Text: {safety_result.sanitized_text}

Task: {task_hint if task_hint else "Provide analysis"}"""

        response = self.client.completion(prompt, self.model)
        usage = self.client.get_last_usage()
        llm_tokens = usage.total_input_tokens + usage.total_output_tokens
        self.total_llm_tokens += llm_tokens

        completion_result = CompletionResult(
            handled_by_tool=False,
            tool_name=None,
            result=response,
            tokens_used=llm_tokens,
        )

        return PipelineResult(
            task_type=task_type,
            safety_result=safety_result,
            completion_result=completion_result,
            llm_fallback_used=True,
            llm_fallback_tokens=llm_tokens,
            total_tokens=llm_tokens,
        )


def create_base_tools(tools_dir: Path, client: OllamaClient, model: str) -> int:
    """Create base tools using LLM and return tokens used."""
    total_tokens = 0

    tools_to_create = [
        {
            "name": "basic_grammar",
            "category": "grammar",
            "description": "Check if a sentence is grammatically acceptable",
            "prompt": """Create a Python function called 'run' that takes a string 'text' and returns a dict with:
- 'acceptable': bool - whether the sentence is grammatically correct
- 'reason': str - explanation if not acceptable

Check for: subject-verb agreement, proper punctuation, sentence structure.
Include: import re at top, def run(text: str) -> dict:""",
        },
        {
            "name": "email_detector",
            "category": "pii",
            "description": "Detect email addresses in text",
            "prompt": """Create a Python function called 'run' that takes a string 'text' and returns a list of dicts.
Each dict should have: 'type': 'EMAIL', 'text': the matched email, 'start': start index, 'end': end index.
Use regex to find email patterns. Include: import re at top, def run(text: str) -> list:""",
        },
        {
            "name": "phone_detector",
            "category": "pii",
            "description": "Detect phone numbers in text",
            "prompt": """Create a Python function called 'run' that takes a string 'text' and returns a list of dicts.
Each dict should have: 'type': 'PHONE', 'text': the matched phone, 'start': start index, 'end': end index.
Handle formats: (123) 456-7890, 123-456-7890, 123.456.7890. Include: import re, def run(text: str) -> list:""",
        },
        {
            "name": "ssn_detector",
            "category": "pii",
            "description": "Detect Social Security Numbers in text",
            "prompt": """Create a Python function called 'run' that takes a string 'text' and returns a list of dicts.
Each dict should have: 'type': 'SSN', 'text': the matched SSN, 'start': start index, 'end': end index.
SSN format: XXX-XX-XXXX. Include: import re at top, def run(text: str) -> list:""",
        },
        {
            "name": "ip_detector",
            "category": "pii",
            "description": "Detect IP addresses in text",
            "prompt": """Create a Python function called 'run' that takes a string 'text' and returns a list of dicts.
Each dict should have: 'type': 'IP', 'text': the matched IP, 'start': start index, 'end': end index.
IPv4 format: X.X.X.X where X is 0-255. Include: import re at top, def run(text: str) -> list:""",
        },
    ]

    for tool_info in tqdm(tools_to_create, desc="Creating tools", unit="tool"):
        category_dir = tools_dir / tool_info["category"]
        category_dir.mkdir(parents=True, exist_ok=True)

        tool_path = category_dir / f"{tool_info['name']}.py"

        prompt = f"""Write ONLY Python code, no markdown, no explanation.
{tool_info["prompt"]}

Output ONLY the Python code:"""

        code = client.completion(prompt, model)
        usage = client.get_last_usage()
        tokens = usage.total_input_tokens + usage.total_output_tokens
        total_tokens += tokens
        # Clean up code
        code = code.replace("```python", "").replace("```", "").strip()

        with open(tool_path, "w") as f:
            f.write(code)

    return total_tokens


def run_baseline(
    client: OllamaClient, model: str, tasks: list[dict]
) -> tuple[int, list]:
    """Run baseline LLM on all tasks (sees raw data)."""
    total_tokens = 0
    results = []

    for i, task in enumerate(tqdm(tasks, desc="Baseline", unit="task")):
        prompt = f"""Analyze the following text:

Text: {task["text"]}

Task: {task["task_type"]}

Provide your analysis."""

        _ = client.completion(prompt, model)
        usage = client.get_last_usage()
        tokens = usage.total_input_tokens + usage.total_output_tokens
        total_tokens += tokens

        results.append(
            {
                "task_idx": i,
                "task_type": task["task_type"],
                "tokens": tokens,
                "saw_raw_pii": True,  # Baseline sees everything
            }
        )

    return total_tokens, results


def generate_report(output_dir: Path, results: dict):
    """Generate PDF report."""
    pdf_path = output_dir / "report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], fontSize=18, spaceAfter=20
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontSize=14, spaceAfter=12
    )

    story.append(Paragraph("Experiment 004: Safety Macro Pipeline", title_style))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]
        )
    )
    story.append(Spacer(1, 20))

    # Key insight
    story.append(Paragraph("Key Safety Insight", heading_style))
    story.append(
        Paragraph(
            f"The safety macro masked PII in {results['pipeline']['pii_masked']} inputs before LLM processing. "
            f"The LLM never saw raw PII data in these cases.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 15))

    # Results table
    story.append(Paragraph("Token Usage Comparison", heading_style))

    data = [
        ["Metric", "Baseline", "Pipeline", "Difference"],
        [
            "Total Tokens",
            f"{results['baseline']['total_tokens']:,}",
            f"{results['pipeline']['total_tokens']:,}",
            f"{results['pipeline']['total_tokens'] - results['baseline']['total_tokens']:+,}",
        ],
        [
            "LLM Tokens",
            f"{results['baseline']['total_tokens']:,}",
            f"{results['pipeline']['llm_tokens']:,}",
            f"{results['pipeline']['llm_tokens'] - results['baseline']['total_tokens']:+,}",
        ],
        [
            "Raw PII Exposures",
            f"{results['baseline']['raw_pii_exposures']}",
            f"{results['pipeline']['raw_pii_exposures']}",
            f"{results['pipeline']['raw_pii_exposures'] - results['baseline']['raw_pii_exposures']:+}",
        ],
    ]

    table = Table(data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 20))

    # Pipeline stats
    story.append(Paragraph("Pipeline Statistics", heading_style))

    pipeline_data = [
        ["Metric", "Count"],
        ["Total Tasks", str(results["pipeline"]["total_tasks"])],
        ["PII Masked (Safety Macro)", str(results["pipeline"]["pii_masked"])],
        ["Handled by Tools", str(results["pipeline"]["tool_handled"])],
        ["LLM Fallbacks", str(results["pipeline"]["llm_fallbacks"])],
        ["Tool Creation Tokens", f"{results['pipeline']['tool_creation_tokens']:,}"],
    ]

    table2 = Table(pipeline_data, colWidths=[3 * inch, 2 * inch])
    table2.setStyle(
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
    story.append(table2)
    story.append(Spacer(1, 20))

    # Savings
    savings = results["baseline"]["total_tokens"] - results["pipeline"]["total_tokens"]
    savings_pct = (
        (savings / results["baseline"]["total_tokens"] * 100)
        if results["baseline"]["total_tokens"] > 0
        else 0
    )

    story.append(Paragraph("Summary", heading_style))
    if savings > 0:
        story.append(
            Paragraph(
                f"Token Savings: {savings:,} tokens ({savings_pct:.1f}%)",
                styles["Normal"],
            )
        )
    else:
        story.append(
            Paragraph(
                f"Token Overhead: {-savings:,} tokens ({-savings_pct:.1f}%)",
                styles["Normal"],
            )
        )

    story.append(
        Paragraph(
            f"PII Protection: {results['pipeline']['pii_masked']} inputs had PII masked before LLM processing",
            styles["Normal"],
        )
    )

    doc.build(story)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 004: Safety Macro Pipeline"
    )
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--cola", type=int, default=50, help="Number of CoLA samples")
    parser.add_argument("--pii", type=int, default=30, help="Number of PII samples")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 004: Safety Macro Pipeline")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  1. Safety Macro → Masks PII BEFORE LLM sees it")
    print("  2. Completion Rules → Try tools first")
    print("  3. LLM Fallback → Only sees sanitized text")
    print()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiment_outputs/exp004_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tools_dir = output_dir / "tools"

    client = OllamaClient()

    # Create base tools
    print("Creating base tools...")
    tool_creation_tokens = create_base_tools(tools_dir, client, args.model)
    print(f"  Tool creation tokens: {tool_creation_tokens:,}")
    print()

    # Load datasets
    print(f"Loading {args.cola} CoLA + {args.pii} PII tasks...")
    tasks = []

    # CoLA tasks
    cola_ds = load_dataset(DATA.COLA, Split.DEV)
    for i, item in enumerate(cola_ds):
        if i >= args.cola:
            break
        tasks.append(
            {"text": item.question, "task_type": "grammar", "expected": item.answer}
        )

    # PII tasks (synthetic with actual PII)
    pii_examples = [
        "Contact john.doe@example.com for more information.",
        "Call me at (555) 123-4567 tomorrow.",
        "My SSN is 123-45-6789, please keep it safe.",
        "Server IP: 192.168.1.100 is down.",
        "Email support@company.org or call 555-987-6543.",
        "Patient SSN: 987-65-4321 needs review.",
        "Connect to 10.0.0.1 for internal access.",
        "Reach out to admin@test.net for help.",
        "Emergency contact: (800) 555-0199",
        "Database at 172.16.0.50 contains records.",
    ]

    for i in range(args.pii):
        tasks.append(
            {
                "text": pii_examples[i % len(pii_examples)],
                "task_type": "pii_detection",
                "expected": "detect_pii",
            }
        )

    print(f"Loaded {len(tasks)} tasks")
    print()

    # Count PII in tasks
    safety_macro = SafetyMacro()
    tasks_with_pii = sum(1 for t in tasks if safety_macro.check_for_pii(t["text"]))
    print(f"Tasks containing PII: {tasks_with_pii}")
    print()

    # Run baseline (LLM sees raw data)
    print("--- Running Baseline (LLM sees raw PII) ---")
    baseline_tokens, baseline_results = run_baseline(client, args.model, tasks)
    baseline_pii_exposures = sum(
        1
        for r in baseline_results
        if r["saw_raw_pii"] and safety_macro.check_for_pii(tasks[r["task_idx"]]["text"])
    )
    print(f"Baseline tokens: {baseline_tokens:,}")
    print(f"Baseline raw PII exposures: {baseline_pii_exposures}")
    print()

    # Run pipeline
    print("--- Running Safety Macro Pipeline ---")
    pipeline = SafetyMacroPipeline(client, args.model, tools_dir)
    pipeline_results = []

    pbar = tqdm(enumerate(tasks), total=len(tasks), desc="Pipeline", unit="task")
    for i, task in pbar:
        result = pipeline.process(task["text"], task["task_type"])
        pipeline_results.append(result)
        pbar.set_postfix(
            masked=pipeline.pii_masked_count,
            tools=pipeline.tool_handled_count,
            llm=pipeline.llm_fallback_count,
        )

    # Calculate pipeline tokens (tool creation + LLM fallbacks)
    pipeline_total_tokens = tool_creation_tokens + pipeline.total_llm_tokens

    print()
    print(f"Pipeline tokens: {pipeline_total_tokens:,}")
    print(f"  - Tool creation: {tool_creation_tokens:,}")
    print(f"  - LLM fallbacks: {pipeline.total_llm_tokens:,}")
    print(f"PII masked before LLM: {pipeline.pii_masked_count}")
    print(f"Tasks handled by tools: {pipeline.tool_handled_count}")
    print()

    # Compile results
    results = {
        "metadata": {
            "model": args.model,
            "timestamp": timestamp,
            "cola_samples": args.cola,
            "pii_samples": args.pii,
        },
        "baseline": {
            "total_tokens": baseline_tokens,
            "raw_pii_exposures": baseline_pii_exposures,
        },
        "pipeline": {
            "total_tokens": pipeline_total_tokens,
            "tool_creation_tokens": tool_creation_tokens,
            "llm_tokens": pipeline.total_llm_tokens,
            "total_tasks": len(tasks),
            "pii_masked": pipeline.pii_masked_count,
            "tool_handled": pipeline.tool_handled_count,
            "llm_fallbacks": pipeline.llm_fallback_count,
            "raw_pii_exposures": 0,  # Pipeline never exposes raw PII to LLM
        },
    }

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate report
    print("--- Generating Report ---")
    pdf_path = generate_report(output_dir, results)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    savings = baseline_tokens - pipeline_total_tokens
    savings_pct = (savings / baseline_tokens * 100) if baseline_tokens > 0 else 0

    print(
        f"Baseline: {baseline_tokens:,} tokens (raw PII exposed: {baseline_pii_exposures})"
    )
    print(f"Pipeline: {pipeline_total_tokens:,} tokens (raw PII exposed: 0)")

    if savings > 0:
        print(f"Savings: {savings:,} tokens ({savings_pct:.1f}%)")
    else:
        print(f"Overhead: {-savings:,} tokens ({-savings_pct:.1f}%)")

    print()
    print(
        f"SAFETY: {pipeline.pii_masked_count} inputs had PII masked before LLM processing"
    )
    print("        The LLM never saw raw PII in these cases.")
    print()
    print(f"Report: {pdf_path}")
    print(f"Results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
