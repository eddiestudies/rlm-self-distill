import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from self_distill.tracking import CallType, ExperimentTracker, TrackedCall


class TestTrackedCall:
    def test_tracked_call_creation(self):
        call = TrackedCall(
            call_type=CallType.RULE_CREATION,
            input_tokens=100,
            output_tokens=50,
            rule_tokens=50,
        )
        assert call.call_type == CallType.RULE_CREATION
        assert call.input_tokens == 100
        assert call.output_tokens == 50
        assert call.rule_tokens == 50
        assert call.total_tokens == 150

    def test_tracked_call_with_metadata(self):
        call = TrackedCall(
            call_type=CallType.RULE_USAGE,
            input_tokens=80,
            output_tokens=40,
            dataset_index=5,
            rule_id="rule_1",
            metadata={"custom": "value"},
        )
        assert call.dataset_index == 5
        assert call.rule_id == "rule_1"
        assert call.metadata["custom"] == "value"


class TestCallType:
    def test_call_type_values(self):
        assert CallType.RULE_CREATION.value == "rule_creation"
        assert CallType.RULE_USAGE.value == "rule_usage"
        assert CallType.NO_RULE.value == "no_rule"


class TestExperimentTrackerInit:
    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_init_creates_experiment(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test-experiment")

        mock_mlflow.get_experiment_by_name.assert_called_once_with("test-experiment")
        mock_mlflow.create_experiment.assert_called_once()
        assert tracker.experiment_id == "exp_123"

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_init_uses_existing_experiment(self, mock_mlflow):
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "existing_123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        tracker = ExperimentTracker("existing-experiment")

        assert tracker.experiment_id == "existing_123"
        mock_mlflow.create_experiment.assert_not_called()

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_init_with_tracking_uri(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        ExperimentTracker("test", tracking_uri="http://localhost:5000")

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")


class TestExperimentTrackerRun:
    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_start_run(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.start_run(run_name="my-run", tags={"tag1": "value1"})

        mock_mlflow.start_run.assert_called_once()
        call_kwargs = mock_mlflow.start_run.call_args[1]
        assert call_kwargs["run_name"] == "my-run"
        assert "tag1" in call_kwargs["tags"]

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_end_run(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.start_run()
        tracker._active_run = MagicMock()
        tracker.end_run()

        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_context_manager(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")

        with tracker.start_run():
            tracker._active_run = MagicMock()
            pass

        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_context_manager_on_exception(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")

        with pytest.raises(ValueError):
            with tracker.start_run():
                tracker._active_run = MagicMock()
                raise ValueError("Test error")

        mock_mlflow.end_run.assert_called_once_with(status="FAILED")


class TestExperimentTrackerLogging:
    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_log_model_params(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.log_model_params(model_name="llama3.2", temperature=0.7)

        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["model_name"] == "llama3.2"
        assert params["temperature"] == 0.7

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_log_dataset_info(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.log_dataset_info("gsm8k", "train", indices=[0, 1, 2], size=100)

        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["dataset_name"] == "gsm8k"
        assert params["dataset_split"] == "train"
        assert params["dataset_size"] == 100

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_log_dataset_info_stores_indices(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.log_dataset_info("gsm8k", "train", indices=[0, 1, 2, 3, 4])

        assert tracker._dataset_indices == [0, 1, 2, 3, 4]


class TestExperimentTrackerCallTracking:
    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_track_call(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        call = tracker.track_call(
            CallType.RULE_CREATION,
            input_tokens=100,
            output_tokens=50,
            rule_tokens=50,
        )

        assert len(tracker._calls) == 1
        assert call.call_type == CallType.RULE_CREATION
        assert call.total_tokens == 150
        mock_mlflow.log_metrics.assert_called()

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_track_multiple_calls(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.track_call(CallType.RULE_CREATION, input_tokens=100, output_tokens=50)
        tracker.track_call(CallType.RULE_USAGE, input_tokens=80, output_tokens=40)
        tracker.track_call(CallType.NO_RULE, input_tokens=60, output_tokens=30)

        assert len(tracker._calls) == 3

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_calls_property_returns_copy(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.track_call(CallType.NO_RULE, input_tokens=100, output_tokens=50)

        calls = tracker.calls
        calls.append(TrackedCall(CallType.NO_RULE))

        assert len(tracker._calls) == 1  # Original unchanged


class TestExperimentTrackerRules:
    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_register_rule(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        stats = tracker.register_rule("rule_1", "Always show work", creation_tokens=50)

        assert "rule_1" in tracker._rules
        assert stats.rule_text == "Always show work"
        assert stats.creation_tokens == 50
        mock_mlflow.log_metric.assert_called_with("rules_created", 1)

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_record_rule_usage(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.register_rule("rule_1", "Test rule")
        tracker.record_rule_usage("rule_1", tokens_saved=20)
        tracker.record_rule_usage("rule_1", tokens_saved=15)

        assert tracker._rules["rule_1"].times_used == 2
        assert tracker._rules["rule_1"].tokens_saved == 35

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_rules_property_returns_copy(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.register_rule("rule_1", "Test")

        rules = tracker.rules
        rules["rule_2"] = MagicMock()

        assert "rule_2" not in tracker._rules


class TestExperimentTrackerConvenienceMethods:
    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_track_rule_creation_call(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        call = tracker.track_rule_creation_call(
            rule_id="rule_1",
            rule_text="Show your work",
            input_tokens=100,
            output_tokens=50,
            dataset_index=5,
        )

        assert call.call_type == CallType.RULE_CREATION
        assert "rule_1" in tracker._rules
        assert tracker._rules["rule_1"].rule_text == "Show your work"

    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_track_rule_usage_call(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.register_rule("rule_1", "Test rule")

        call = tracker.track_rule_usage_call(
            rule_id="rule_1",
            input_tokens=80,
            output_tokens=40,
            rule_tokens_in_prompt=30,
            dataset_index=10,
            baseline_tokens=150,
        )

        assert call.call_type == CallType.RULE_USAGE
        assert call.rule_tokens == 30
        assert tracker._rules["rule_1"].times_used == 1
        assert tracker._rules["rule_1"].tokens_saved == 30  # 150 - 120


class TestExperimentTrackerSummary:
    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_get_summary(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.track_call(
            CallType.RULE_CREATION, input_tokens=100, output_tokens=50, rule_tokens=50
        )
        tracker.track_call(
            CallType.RULE_USAGE, input_tokens=80, output_tokens=40, rule_tokens=30
        )
        tracker.track_call(CallType.NO_RULE, input_tokens=60, output_tokens=30)
        tracker.register_rule("rule_1", "Test")
        tracker.record_rule_usage("rule_1")

        summary = tracker.get_summary()

        assert summary["total_calls"] == 3
        assert summary["rule_creation_calls"] == 1
        assert summary["rule_usage_calls"] == 1
        assert summary["no_rule_calls"] == 1
        assert summary["total_tokens"] == 360  # 150 + 120 + 90
        assert summary["rule_tokens"] == 80  # 50 + 30
        assert summary["rules_created"] == 1
        assert summary["total_rule_usages"] == 1


class TestExperimentTrackerFinalMetrics:
    @patch("self_distill.tracking.mlflow_tracker.mlflow")
    def test_log_final_metrics(self, mock_mlflow):
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp_123"

        tracker = ExperimentTracker("test")
        tracker.track_call(
            CallType.RULE_CREATION, input_tokens=100, output_tokens=50, rule_tokens=50
        )
        tracker.track_call(
            CallType.RULE_USAGE, input_tokens=80, output_tokens=40, rule_tokens=30
        )
        tracker.register_rule("rule_1", "Test", creation_tokens=50)
        tracker.record_rule_usage("rule_1", tokens_saved=20)

        tracker._log_final_metrics()

        # Check that log_metrics was called with final metrics
        calls = mock_mlflow.log_metrics.call_args_list
        final_metrics_call = calls[-1][0][0]

        assert final_metrics_call["final_total_calls"] == 2
        assert final_metrics_call["final_rule_creation_calls"] == 1
        assert final_metrics_call["final_rule_usage_calls"] == 1
        assert final_metrics_call["final_rules_created"] == 1
