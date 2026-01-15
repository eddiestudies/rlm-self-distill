#!/usr/bin/env python3
"""
Extract experiment data for visualization dashboard.
Collects metadata, results, tools, and sample data from all experiments.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ToolInfo:
    name: str
    category: str
    code: str
    path: str


@dataclass
class ExperimentRun:
    experiment_id: str
    run_id: str
    timestamp: str
    output_dir: str
    results: dict = field(default_factory=dict)
    tools: list[ToolInfo] = field(default_factory=list)
    sample_data: list[dict] = field(default_factory=list)
    directory_structure: list[str] = field(default_factory=list)


@dataclass
class ExperimentDefinition:
    id: str
    name: str
    description: str
    architecture: str
    source_file: str
    dataset_info: str
    runs: list[ExperimentRun] = field(default_factory=list)


def extract_docstring(file_path: Path) -> tuple[str, str, str]:
    """Extract title, description and architecture from experiment docstring."""
    content = file_path.read_text()

    # Find the docstring
    match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if not match:
        return "", "", ""

    docstring = match.group(1).strip()
    lines = docstring.split("\n")

    # First line is usually the title
    title = lines[0].strip() if lines else ""

    # Find description (everything before Architecture:)
    desc_lines = []
    arch_lines = []
    in_architecture = False

    for line in lines[1:]:
        if "Architecture:" in line or "Key " in line:
            in_architecture = True
        if in_architecture:
            arch_lines.append(line)
        else:
            desc_lines.append(line)

    description = "\n".join(desc_lines).strip()
    architecture = "\n".join(arch_lines).strip()

    return title, description, architecture


def get_dataset_info(file_path: Path) -> str:
    """Extract dataset usage from experiment file."""
    content = file_path.read_text()

    datasets = []
    if "CoLA" in content or "cola" in content.lower():
        datasets.append("CoLA (Grammar)")
    if "PII" in content or "pii" in content.lower():
        datasets.append("PII Detection")
    if "GSM8K" in content or "gsm8k" in content.lower():
        datasets.append("GSM8K (Math)")
    if "AI4Privacy" in content or "ai4privacy" in content.lower():
        datasets.append("AI4Privacy (200K PII)")
    if "SciQ" in content or "sciq" in content.lower():
        datasets.append("SciQ (Science QA)")

    return ", ".join(datasets) if datasets else "Unknown"


def collect_tools(output_dir: Path) -> list[ToolInfo]:
    """Collect all tool files from an experiment output."""
    tools = []

    # Look for tools in various structures
    tool_patterns = [
        output_dir / "tools" / "**" / "*.py",
        output_dir / "*.py",
    ]

    for pattern in tool_patterns:
        for tool_path in output_dir.glob(str(pattern.relative_to(output_dir))):
            if tool_path.name.startswith("_"):
                continue

            # Determine category from path
            rel_path = tool_path.relative_to(output_dir)
            parts = rel_path.parts

            if "tools" in parts:
                idx = parts.index("tools")
                if len(parts) > idx + 2:
                    category = parts[idx + 1]
                else:
                    category = "general"
            else:
                category = "root"

            try:
                code = tool_path.read_text()
                tools.append(
                    ToolInfo(
                        name=tool_path.stem,
                        category=category,
                        code=code,
                        path=str(rel_path),
                    )
                )
            except Exception:
                pass

    return tools


def collect_sample_data(output_dir: Path) -> list[dict]:
    """Collect sample data from evidence store or checkpoints."""
    samples = []

    # Try evidence store
    metadata_path = output_dir / "evidence_store" / "vectors" / "metadata.json"
    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text())
            texts = data.get("texts", [])[:5]  # First 5 samples
            metadata = data.get("metadata", [])[:5]
            for i, (text, meta) in enumerate(zip(texts, metadata)):
                samples.append(
                    {
                        "id": data.get("ids", [f"sample_{i}"])[i]
                        if i < len(data.get("ids", []))
                        else f"sample_{i}",
                        "text": text[:200] + "..." if len(text) > 200 else text,
                        "type": meta.get("type", "unknown"),
                    }
                )
        except Exception:
            pass

    return samples


def get_directory_structure(output_dir: Path) -> list[str]:
    """Get directory structure as a list of paths."""
    structure = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rel_path = path.relative_to(output_dir)
            structure.append(str(rel_path))
    return structure


def process_experiment_outputs(base_dir: Path) -> dict[str, list[ExperimentRun]]:
    """Process all experiment output directories."""
    runs_by_exp: dict[str, list[ExperimentRun]] = {}

    for output_dir in sorted(base_dir.iterdir()):
        if not output_dir.is_dir():
            continue

        # Parse experiment ID and timestamp from directory name
        match = re.match(r"(exp\d+)_(\d{8}_\d{6})", output_dir.name)
        if not match:
            continue

        exp_id = match.group(1)
        timestamp_str = match.group(2)

        # Parse timestamp
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").isoformat()
        except ValueError:
            timestamp = timestamp_str

        # Load results if available
        results = {}
        results_path = output_dir / "results.json"
        if results_path.exists():
            try:
                results = json.loads(results_path.read_text())
            except Exception:
                pass

        # Collect tools
        tools = collect_tools(output_dir)

        # Collect sample data
        sample_data = collect_sample_data(output_dir)

        # Get directory structure
        dir_structure = get_directory_structure(output_dir)

        run = ExperimentRun(
            experiment_id=exp_id,
            run_id=output_dir.name,
            timestamp=timestamp,
            output_dir=str(output_dir),
            results=results,
            tools=[asdict(t) for t in tools],
            sample_data=sample_data,
            directory_structure=dir_structure,
        )

        if exp_id not in runs_by_exp:
            runs_by_exp[exp_id] = []
        runs_by_exp[exp_id].append(run)

    return runs_by_exp


def main():
    project_root = Path(__file__).parent.parent
    experiments_dir = project_root / "experiments"
    outputs_dir = project_root / "experiment_outputs"

    # Collect experiment definitions
    experiments = []

    for exp_file in sorted(experiments_dir.glob("exp*.py")):
        exp_id = exp_file.stem.split("_")[0]  # exp001, exp002, etc.

        title, description, architecture = extract_docstring(exp_file)
        dataset_info = get_dataset_info(exp_file)

        exp = ExperimentDefinition(
            id=exp_id,
            name=title or exp_file.stem,
            description=description,
            architecture=architecture,
            source_file=exp_file.name,
            dataset_info=dataset_info,
        )
        experiments.append(exp)

    # Collect runs for each experiment
    if outputs_dir.exists():
        runs_by_exp = process_experiment_outputs(outputs_dir)

        for exp in experiments:
            if exp.id in runs_by_exp:
                exp.runs = [asdict(r) for r in runs_by_exp[exp.id]]

    # Output as JSON
    output = {
        "generated_at": datetime.now().isoformat(),
        "experiments": [asdict(e) for e in experiments],
    }

    # Write to dashboard data file
    output_path = project_root / "dashboard" / "data.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(f"Extracted data for {len(experiments)} experiments")
    total_runs = sum(len(e.runs) for e in experiments)
    print(f"Total runs: {total_runs}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
