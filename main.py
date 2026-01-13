"""
RLM Self-Distillation: LLM Self-Distillation to Rule-Based Sub-Task Completion

This module provides a CLI entry point for running experiments comparing
baseline LLM inference against recursive language models with executable tools.

Usage:
    python main.py --experiment exp003 --model llama3.2:3b --cola 50 --pii 15
    python main.py --experiment exp004 --model llama3.2:3b --cola 50 --pii 30
"""

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "exp001": "exp001_recursive_tool_creation.py",
    "exp002": "exp002_scaling_analysis.py",
    "exp003": "exp003_executable_tools.py",
    "exp004": "exp004_safety_macro_pipeline.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="RLM Self-Distillation Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --list                          # List available experiments
  python main.py --experiment exp003             # Run exp003 with defaults
  python main.py --experiment exp004 --cola 100  # Run exp004 with 100 CoLA samples
        """,
    )
    parser.add_argument(
        "--list", action="store_true", help="List available experiments"
    )
    parser.add_argument(
        "--experiment", "-e", choices=EXPERIMENTS.keys(), help="Experiment to run"
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Ollama model name (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--cola", type=int, default=50, help="Number of CoLA samples (default: 50)"
    )
    parser.add_argument(
        "--pii", type=int, default=15, help="Number of PII samples (default: 15)"
    )

    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        print()
        for name, script in EXPERIMENTS.items():
            print(f"  {name}: {script}")
        print()
        print("Run with: python main.py --experiment <name>")
        return

    if not args.experiment:
        parser.print_help()
        return

    experiments_dir = Path(__file__).parent / "experiments"
    script = experiments_dir / EXPERIMENTS[args.experiment]

    if not script.exists():
        print(f"Error: Experiment script not found: {script}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--cola",
        str(args.cola),
        "--pii",
        str(args.pii),
    ]

    print(f"Running: {args.experiment}")
    print(f"Command: {' '.join(cmd)}")
    print()

    subprocess.run(cmd, env={"PYTHONPATH": "."})


if __name__ == "__main__":
    main()
