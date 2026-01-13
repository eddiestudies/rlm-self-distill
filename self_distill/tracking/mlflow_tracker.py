"""
MLflow experiment tracking for self-distillation experiments.

Tracks:
- Model configuration
- Dataset info (name, split, indices)
- Call statistics (rule creation vs rule usage)
- Token usage (total and rule-related overhead)
- Rules created and used
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mlflow


class CallType(Enum):
    """Type of LLM call for tracking purposes."""

    RULE_CREATION = "rule_creation"  # Call that creates/generates a rule
    RULE_USAGE = "rule_usage"  # Call that uses an existing rule
    NO_RULE = "no_rule"  # Call without any rule involvement


@dataclass
class TrackedCall:
    """A single tracked LLM call."""

    call_type: CallType
    input_tokens: int = 0
    output_tokens: int = 0
    rule_tokens: int = 0  # Extra tokens used for rule (in prompt or response)
    dataset_index: int | None = None
    rule_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RuleStats:
    """Statistics for a single rule."""

    rule_id: str
    rule_text: str
    creation_tokens: int = 0  # Tokens used to create this rule
    times_used: int = 0
    tokens_saved: int = 0  # Estimated tokens saved by using rule vs recreating


class ExperimentTracker:
    """
    MLflow-based experiment tracker for self-distillation runs.

    Usage:
        tracker = ExperimentTracker(experiment_name="gsm8k-distillation")

        with tracker.start_run(run_name="baseline"):
            tracker.log_model_params(model_name="llama3.2", temperature=0.7)
            tracker.log_dataset_info("gsm8k", "train", indices=[0, 1, 2])

            # Track calls as you make them
            tracker.track_call(CallType.RULE_CREATION, input_tokens=100, output_tokens=50, rule_tokens=50)
            tracker.track_call(CallType.RULE_USAGE, input_tokens=80, output_tokens=40, rule_tokens=30)

            # Register rules
            tracker.register_rule("rule_1", "Always show your work step by step")
            tracker.record_rule_usage("rule_1")

        # Run is automatically ended and metrics logged
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ):
        """
        Initialize the experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: Location for artifacts (default: MLflow default)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
        else:
            self.experiment_id = experiment.experiment_id

        # Run state
        self._active_run: mlflow.ActiveRun | None = None
        self._calls: list[TrackedCall] = []
        self._rules: dict[str, RuleStats] = {}
        self._dataset_indices: list[int] = []

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> "ExperimentTracker":
        """
        Start a new MLflow run.

        Can be used as a context manager:
            with tracker.start_run("my-run"):
                ...

        Args:
            run_name: Name for this run
            tags: Optional tags to add
            description: Optional run description

        Returns:
            self for context manager usage
        """
        self._calls = []
        self._rules = {}
        self._dataset_indices = []

        run_tags = tags or {}
        if description:
            run_tags["mlflow.note.content"] = description

        self._active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=run_tags,
        )
        return self

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run and log final metrics.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self._active_run is None:
            return

        # Log final metrics
        self._log_final_metrics()

        # Log artifacts
        self._log_artifacts()

        mlflow.end_run(status=status)
        self._active_run = None

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        status = "FINISHED" if exc_type is None else "FAILED"
        self.end_run(status=status)
        return False

    def log_model_params(
        self,
        model_name: str,
        **kwargs,
    ) -> None:
        """
        Log model parameters.

        Args:
            model_name: Name/identifier of the model
            **kwargs: Additional model parameters (temperature, top_p, etc.)
        """
        params = {"model_name": model_name, **kwargs}
        mlflow.log_params(params)

    def log_dataset_info(
        self,
        dataset_name: str,
        split: str,
        indices: list[int] | None = None,
        size: int | None = None,
        **kwargs,
    ) -> None:
        """
        Log dataset information.

        Args:
            dataset_name: Name of the dataset (e.g., "gsm8k", "cola")
            split: Dataset split (train, dev, test)
            indices: Specific indices used (for subset experiments)
            size: Total size of the dataset/subset
            **kwargs: Additional dataset parameters
        """
        params = {
            "dataset_name": dataset_name,
            "dataset_split": split,
            **kwargs,
        }

        if indices is not None:
            self._dataset_indices = indices
            params["dataset_indices_count"] = len(indices)
            # Log first/last few indices for reference
            if len(indices) <= 10:
                params["dataset_indices"] = str(indices)
            else:
                params["dataset_indices_sample"] = str(indices[:5] + ["..."] + indices[-5:])

        if size is not None:
            params["dataset_size"] = size

        mlflow.log_params(params)

    def track_call(
        self,
        call_type: CallType,
        input_tokens: int = 0,
        output_tokens: int = 0,
        rule_tokens: int = 0,
        dataset_index: int | None = None,
        rule_id: str | None = None,
        **metadata,
    ) -> TrackedCall:
        """
        Track an LLM call.

        Args:
            call_type: Type of call (RULE_CREATION, RULE_USAGE, NO_RULE)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            rule_tokens: Extra tokens used for rule guidance
            dataset_index: Index in dataset being processed
            rule_id: ID of rule being created or used
            **metadata: Additional metadata to store

        Returns:
            The TrackedCall object
        """
        call = TrackedCall(
            call_type=call_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            rule_tokens=rule_tokens,
            dataset_index=dataset_index,
            rule_id=rule_id,
            metadata=metadata,
        )
        self._calls.append(call)

        # Log incremental metrics
        call_count = len(self._calls)
        mlflow.log_metrics(
            {
                "total_calls": call_count,
                "total_input_tokens": sum(c.input_tokens for c in self._calls),
                "total_output_tokens": sum(c.output_tokens for c in self._calls),
                "total_rule_tokens": sum(c.rule_tokens for c in self._calls),
            },
            step=call_count,
        )

        return call

    def register_rule(
        self,
        rule_id: str,
        rule_text: str,
        creation_tokens: int = 0,
    ) -> RuleStats:
        """
        Register a new rule that was created.

        Args:
            rule_id: Unique identifier for the rule
            rule_text: The actual rule text
            creation_tokens: Tokens used to create this rule

        Returns:
            RuleStats object for this rule
        """
        stats = RuleStats(
            rule_id=rule_id,
            rule_text=rule_text,
            creation_tokens=creation_tokens,
        )
        self._rules[rule_id] = stats

        # Log rule count
        mlflow.log_metric("rules_created", len(self._rules))

        return stats

    def record_rule_usage(
        self,
        rule_id: str,
        tokens_saved: int = 0,
    ) -> None:
        """
        Record that a rule was used.

        Args:
            rule_id: ID of the rule that was used
            tokens_saved: Estimated tokens saved by using this rule
        """
        if rule_id in self._rules:
            self._rules[rule_id].times_used += 1
            self._rules[rule_id].tokens_saved += tokens_saved

    def log_custom_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log a custom metric."""
        mlflow.log_metric(key, value, step=step)

    def log_custom_params(self, params: dict[str, Any]) -> None:
        """Log custom parameters."""
        mlflow.log_params(params)

    def _log_final_metrics(self) -> None:
        """Log final aggregated metrics at end of run."""
        if not self._calls:
            return

        # Call type breakdown
        rule_creation_calls = [c for c in self._calls if c.call_type == CallType.RULE_CREATION]
        rule_usage_calls = [c for c in self._calls if c.call_type == CallType.RULE_USAGE]
        no_rule_calls = [c for c in self._calls if c.call_type == CallType.NO_RULE]

        metrics = {
            # Call counts
            "final_total_calls": len(self._calls),
            "final_rule_creation_calls": len(rule_creation_calls),
            "final_rule_usage_calls": len(rule_usage_calls),
            "final_no_rule_calls": len(no_rule_calls),
            # Token totals
            "final_total_input_tokens": sum(c.input_tokens for c in self._calls),
            "final_total_output_tokens": sum(c.output_tokens for c in self._calls),
            "final_total_tokens": sum(c.total_tokens for c in self._calls),
            "final_total_rule_tokens": sum(c.rule_tokens for c in self._calls),
            # Rule stats
            "final_rules_created": len(self._rules),
            "final_total_rule_usages": sum(r.times_used for r in self._rules.values()),
            "final_tokens_saved_by_rules": sum(r.tokens_saved for r in self._rules.values()),
        }

        # Average tokens per call type
        if rule_creation_calls:
            metrics["avg_tokens_rule_creation"] = sum(c.total_tokens for c in rule_creation_calls) / len(rule_creation_calls)
        if rule_usage_calls:
            metrics["avg_tokens_rule_usage"] = sum(c.total_tokens for c in rule_usage_calls) / len(rule_usage_calls)
        if no_rule_calls:
            metrics["avg_tokens_no_rule"] = sum(c.total_tokens for c in no_rule_calls) / len(no_rule_calls)

        # Rule overhead percentage
        total_tokens = sum(c.total_tokens for c in self._calls)
        if total_tokens > 0:
            rule_token_overhead = sum(c.rule_tokens for c in self._calls)
            metrics["rule_token_overhead_pct"] = (rule_token_overhead / total_tokens) * 100

        mlflow.log_metrics(metrics)

    def _log_artifacts(self) -> None:
        """Log artifacts (call log, rules, etc.) at end of run."""
        import json
        import tempfile
        import os

        # Log call history as JSON
        if self._calls:
            calls_data = [
                {
                    "call_type": c.call_type.value,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "rule_tokens": c.rule_tokens,
                    "total_tokens": c.total_tokens,
                    "dataset_index": c.dataset_index,
                    "rule_id": c.rule_id,
                    "metadata": c.metadata,
                }
                for c in self._calls
            ]

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(calls_data, f, indent=2)
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_path="logs")
            os.unlink(temp_path)

        # Log rules as JSON
        if self._rules:
            rules_data = {
                rule_id: {
                    "rule_text": stats.rule_text,
                    "creation_tokens": stats.creation_tokens,
                    "times_used": stats.times_used,
                    "tokens_saved": stats.tokens_saved,
                }
                for rule_id, stats in self._rules.items()
            }

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(rules_data, f, indent=2)
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_path="logs")
            os.unlink(temp_path)

        # Log dataset indices if available
        if self._dataset_indices:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(self._dataset_indices, f)
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_path="logs")
            os.unlink(temp_path)

    # Convenience methods for common patterns

    def track_rule_creation_call(
        self,
        rule_id: str,
        rule_text: str,
        input_tokens: int,
        output_tokens: int,
        dataset_index: int | None = None,
        **metadata,
    ) -> TrackedCall:
        """
        Convenience method to track a rule creation call and register the rule.

        Args:
            rule_id: Unique identifier for the rule
            rule_text: The generated rule text
            input_tokens: Input tokens for this call
            output_tokens: Output tokens for this call
            dataset_index: Index in dataset being processed
            **metadata: Additional metadata

        Returns:
            TrackedCall object
        """
        # Register the rule
        self.register_rule(rule_id, rule_text, creation_tokens=output_tokens)

        # Track the call
        return self.track_call(
            call_type=CallType.RULE_CREATION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            rule_tokens=output_tokens,  # All output tokens are "rule tokens" for creation
            dataset_index=dataset_index,
            rule_id=rule_id,
            **metadata,
        )

    def track_rule_usage_call(
        self,
        rule_id: str,
        input_tokens: int,
        output_tokens: int,
        rule_tokens_in_prompt: int,
        dataset_index: int | None = None,
        baseline_tokens: int | None = None,
        **metadata,
    ) -> TrackedCall:
        """
        Convenience method to track a call that uses an existing rule.

        Args:
            rule_id: ID of the rule being used
            input_tokens: Input tokens for this call
            output_tokens: Output tokens for this call
            rule_tokens_in_prompt: Tokens used for the rule in the prompt
            dataset_index: Index in dataset being processed
            baseline_tokens: Baseline tokens without rule (for savings calculation)
            **metadata: Additional metadata

        Returns:
            TrackedCall object
        """
        # Record rule usage
        tokens_saved = 0
        if baseline_tokens is not None:
            tokens_saved = max(0, baseline_tokens - (input_tokens + output_tokens))
        self.record_rule_usage(rule_id, tokens_saved=tokens_saved)

        # Track the call
        return self.track_call(
            call_type=CallType.RULE_USAGE,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            rule_tokens=rule_tokens_in_prompt,
            dataset_index=dataset_index,
            rule_id=rule_id,
            **metadata,
        )

    @property
    def calls(self) -> list[TrackedCall]:
        """Get all tracked calls."""
        return self._calls.copy()

    @property
    def rules(self) -> dict[str, RuleStats]:
        """Get all registered rules."""
        return self._rules.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the current run state."""
        return {
            "total_calls": len(self._calls),
            "rule_creation_calls": len([c for c in self._calls if c.call_type == CallType.RULE_CREATION]),
            "rule_usage_calls": len([c for c in self._calls if c.call_type == CallType.RULE_USAGE]),
            "no_rule_calls": len([c for c in self._calls if c.call_type == CallType.NO_RULE]),
            "total_tokens": sum(c.total_tokens for c in self._calls),
            "rule_tokens": sum(c.rule_tokens for c in self._calls),
            "rules_created": len(self._rules),
            "total_rule_usages": sum(r.times_used for r in self._rules.values()),
        }
