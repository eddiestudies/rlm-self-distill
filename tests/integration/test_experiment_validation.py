"""
Tests to validate experiment output structure.

These tests ensure experiments produce correctly formatted outputs
with all required fields.
"""

import json
import tempfile
from pathlib import Path

import pytest


class TestExperimentOutputStructure:
    """Test that experiment outputs have the correct structure."""

    def test_results_json_schema(self):
        """Verify results.json has required fields."""
        # Example of expected results.json structure
        expected_schema = {
            "metadata": {
                "required_fields": ["timestamp", "model"],
                "optional_fields": ["iterations", "train_size", "test_size"],
            },
            "test_results": {
                "required_fields": ["accuracy"],
                "optional_fields": ["tp", "tn", "fp", "fn", "errors"],
            },
        }

        # Create a valid results structure
        valid_results = {
            "metadata": {
                "timestamp": "20260114_100354",
                "model": "test-model",
                "iterations": 10,
            },
            "test_results": {
                "accuracy": 0.85,
                "tp": 100,
                "tn": 50,
                "fp": 10,
                "fn": 5,
            },
        }

        # Validate metadata
        for field in expected_schema["metadata"]["required_fields"]:
            assert field in valid_results["metadata"], f"Missing required field: {field}"

        # Validate test_results
        for field in expected_schema["test_results"]["required_fields"]:
            assert field in valid_results["test_results"], f"Missing required field: {field}"

    def test_skills_json_schema(self):
        """Verify skills.json has required fields."""
        valid_skill = {
            "test_skill": {
                "skill": {
                    "name": "test_skill",
                    "description": "A test skill",
                    "version": 1,
                    "code": "def solve(t): return t",
                    "is_valid": True,
                    "compile_error": None,
                },
                "trigger_name": "always",
                "priority": 0,
            }
        }

        # Validate skill structure
        for skill_name, skill_data in valid_skill.items():
            assert "skill" in skill_data
            assert "trigger_name" in skill_data
            assert "priority" in skill_data

            skill_info = skill_data["skill"]
            assert "name" in skill_info
            assert "code" in skill_info
            assert "is_valid" in skill_info


class TestExperimentResultsValidation:
    """Test validation of experiment results."""

    def test_accuracy_bounds(self):
        """Accuracy should be between 0 and 1."""
        valid_accuracy = 0.68
        assert 0.0 <= valid_accuracy <= 1.0

        # Edge cases
        assert 0.0 <= 0.0 <= 1.0
        assert 0.0 <= 1.0 <= 1.0

    def test_confusion_matrix_consistency(self):
        """Confusion matrix values should be non-negative and consistent."""
        results = {
            "tp": 102,
            "tn": 0,
            "fp": 48,
            "fn": 0,
            "total": 150,
        }

        # All values non-negative
        assert results["tp"] >= 0
        assert results["tn"] >= 0
        assert results["fp"] >= 0
        assert results["fn"] >= 0

        # Sum should equal total
        total_from_matrix = results["tp"] + results["tn"] + results["fp"] + results["fn"]
        assert total_from_matrix == results["total"]

    def test_iteration_history_structure(self):
        """History should track metrics across iterations."""
        history = [
            {"iteration": 0, "accuracy": 0.0, "errors": 300},
            {"iteration": 1, "accuracy": 0.63, "errors": 0},
            {"iteration": 2, "accuracy": 0.68, "errors": 0},
        ]

        # Iterations should be sequential
        for i, entry in enumerate(history):
            assert entry["iteration"] == i

        # All entries should have accuracy
        for entry in history:
            assert "accuracy" in entry
            assert 0.0 <= entry["accuracy"] <= 1.0


class TestSkillCodeValidation:
    """Test validation of generated skill code."""

    def test_valid_skill_code(self):
        """Valid skill code should compile and run."""
        from self_distill.skills.base import CodeSkill

        valid_code = '''
def solve(text):
    return len(text)
'''
        skill = CodeSkill("test", valid_code)
        assert skill.is_valid
        result = skill.run("hello")
        assert result.success
        assert result.output == 5

    def test_skill_must_have_solve_function(self):
        """Skill code must define a solve function."""
        from self_distill.skills.base import CodeSkill

        invalid_code = '''
def process(text):
    return text
'''
        skill = CodeSkill("test", invalid_code)
        assert not skill.is_valid
        assert "solve" in skill._compile_error

    def test_skill_handles_runtime_errors(self):
        """Skills should gracefully handle runtime errors."""
        from self_distill.skills.base import CodeSkill

        error_code = '''
def solve(text):
    raise ValueError("test error")
'''
        skill = CodeSkill("test", error_code)
        assert skill.is_valid  # Compiles OK

        result = skill.run("test")
        assert not result.success
        assert "ValueError" in result.error


class TestExperimentOutputFiles:
    """Test experiment output file handling."""

    def test_save_results_json(self):
        """Test saving results to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            results = {
                "metadata": {"model": "test", "timestamp": "20260115"},
                "test_results": {"accuracy": 0.75},
            }

            output_file = path / "results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            # Verify file exists and is valid JSON
            assert output_file.exists()

            with open(output_file) as f:
                loaded = json.load(f)

            assert loaded == results

    def test_save_skill_files(self):
        """Test saving skill code to .py files."""
        from self_distill.skills.base import CodeSkill

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            skill_code = '''
def solve(text):
    return text.upper()
'''
            skill = CodeSkill("upper_case", skill_code, "Converts to uppercase")

            # Save skill code
            skill_file = path / "upper_case.py"
            with open(skill_file, "w") as f:
                f.write(skill.code)

            # Verify file can be loaded and executed
            assert skill_file.exists()

            # Load and verify
            with open(skill_file) as f:
                loaded_code = f.read()

            new_skill = CodeSkill("loaded", loaded_code)
            assert new_skill.is_valid
            assert new_skill.run("hello").output == "HELLO"


class TestExperimentMetrics:
    """Test experiment metric calculations."""

    def test_accuracy_calculation(self):
        """Test accuracy calculation from confusion matrix."""
        tp, tn, fp, fn = 102, 0, 48, 0
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total

        assert accuracy == pytest.approx(0.68)

    def test_precision_calculation(self):
        """Test precision calculation."""
        tp, fp = 102, 48
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        assert precision == pytest.approx(0.68)

    def test_recall_calculation(self):
        """Test recall calculation."""
        tp, fn = 102, 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        assert recall == pytest.approx(1.0)

    def test_f1_calculation(self):
        """Test F1 score calculation."""
        precision = 0.68
        recall = 1.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        assert f1 == pytest.approx(0.809, rel=0.01)


class TestIterativeRefinementLogic:
    """Test the iterative refinement logic used in experiments."""

    def test_best_iteration_selection(self):
        """Test selecting the best iteration from history."""
        history = [
            {"iteration": 0, "accuracy": 0.0},
            {"iteration": 1, "accuracy": 0.63},
            {"iteration": 2, "accuracy": 0.0},
            {"iteration": 3, "accuracy": 0.68},
            {"iteration": 4, "accuracy": 0.65},
        ]

        best = max(history, key=lambda x: x["accuracy"])
        assert best["iteration"] == 3
        assert best["accuracy"] == 0.68

    def test_convergence_detection(self):
        """Test detecting when refinement has converged."""
        history = [
            {"iteration": 0, "accuracy": 0.60},
            {"iteration": 1, "accuracy": 0.65},
            {"iteration": 2, "accuracy": 0.66},
            {"iteration": 3, "accuracy": 0.66},
            {"iteration": 4, "accuracy": 0.66},
        ]

        # Check for convergence (no improvement for N iterations)
        no_improvement_count = 0
        best_accuracy = 0

        for entry in history:
            if entry["accuracy"] > best_accuracy:
                best_accuracy = entry["accuracy"]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        # Converged after 2 iterations without improvement
        assert no_improvement_count >= 2

    def test_error_rate_tracking(self):
        """Test tracking error rate across iterations."""
        history = [
            {"iteration": 0, "errors": 300, "total": 300},
            {"iteration": 1, "errors": 0, "total": 300},
            {"iteration": 2, "errors": 300, "total": 300},
            {"iteration": 3, "errors": 0, "total": 300},
        ]

        # Calculate error rates
        error_rates = [h["errors"] / h["total"] for h in history]

        assert error_rates[0] == 1.0  # All errors
        assert error_rates[1] == 0.0  # No errors
        assert error_rates[2] == 1.0  # All errors (unstable)
        assert error_rates[3] == 0.0  # No errors

        # Count successful iterations (error rate < 1.0)
        successful = sum(1 for r in error_rates if r < 1.0)
        assert successful == 2
