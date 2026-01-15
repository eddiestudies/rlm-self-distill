"""Tests for SkillRegistry."""

import tempfile
from pathlib import Path


from self_distill.skills.base import (
    CodeSkill,
    AlwaysTrigger,
    KeywordTrigger,
    PatternTrigger,
)
from self_distill.skills.registry import SkillRegistry, RegisteredSkill


class TestRegisteredSkill:
    """Tests for RegisteredSkill dataclass."""

    def test_creation(self):
        skill = CodeSkill("test", "def solve(t): return t")
        trigger = AlwaysTrigger()

        registered = RegisteredSkill(skill=skill, trigger=trigger)
        assert registered.skill == skill
        assert registered.trigger == trigger
        assert registered.priority == 0

    def test_with_priority(self):
        skill = CodeSkill("test", "def solve(t): return t")
        trigger = AlwaysTrigger()

        registered = RegisteredSkill(skill=skill, trigger=trigger, priority=10)
        assert registered.priority == 10


class TestSkillRegistryInit:
    """Tests for SkillRegistry initialization."""

    def test_default_init(self):
        registry = SkillRegistry()
        assert len(registry) == 0
        assert registry.confidence_threshold == 0.5

    def test_custom_threshold(self):
        registry = SkillRegistry(confidence_threshold=0.8)
        assert registry.confidence_threshold == 0.8


class TestSkillRegistryRegister:
    """Tests for registering skills."""

    def test_register_skill(self):
        registry = SkillRegistry()
        skill = CodeSkill("upper", "def solve(t): return t.upper()")
        trigger = AlwaysTrigger()

        registry.register(skill, trigger)
        assert "upper" in registry
        assert len(registry) == 1

    def test_register_with_priority(self):
        registry = SkillRegistry()
        skill = CodeSkill("test", "def solve(t): return t")
        trigger = AlwaysTrigger()

        registry.register(skill, trigger, priority=5)
        assert registry.skills["test"].priority == 5

    def test_register_multiple(self):
        registry = SkillRegistry()

        skill1 = CodeSkill("s1", "def solve(t): return 1")
        skill2 = CodeSkill("s2", "def solve(t): return 2")

        registry.register(skill1, AlwaysTrigger())
        registry.register(skill2, AlwaysTrigger())

        assert len(registry) == 2
        assert "s1" in registry
        assert "s2" in registry


class TestSkillRegistryUnregister:
    """Tests for unregistering skills."""

    def test_unregister_existing(self):
        registry = SkillRegistry()
        skill = CodeSkill("test", "def solve(t): return t")
        registry.register(skill, AlwaysTrigger())

        result = registry.unregister("test")
        assert result is True
        assert "test" not in registry

    def test_unregister_nonexistent(self):
        registry = SkillRegistry()
        result = registry.unregister("nonexistent")
        assert result is False


class TestSkillRegistryGetSkill:
    """Tests for getting skills."""

    def test_get_existing(self):
        registry = SkillRegistry()
        skill = CodeSkill("test", "def solve(t): return t")
        registry.register(skill, AlwaysTrigger())

        retrieved = registry.get_skill("test")
        assert retrieved == skill

    def test_get_nonexistent(self):
        registry = SkillRegistry()
        assert registry.get_skill("nonexistent") is None


class TestSkillRegistryCheckTriggers:
    """Tests for checking triggers."""

    def test_no_triggers(self):
        registry = SkillRegistry()
        result = registry.check_triggers("test")
        assert result == []

    def test_trigger_below_threshold(self):
        registry = SkillRegistry(confidence_threshold=0.8)
        skill = CodeSkill("test", "def solve(t): return t")
        # Keyword trigger returns partial confidence
        trigger = KeywordTrigger("kw", ["hello", "world", "foo", "bar"])
        registry.register(skill, trigger)

        # Only one keyword matches, confidence = 0.25 < 0.8
        result = registry.check_triggers("hello there")
        assert result == []

    def test_trigger_above_threshold(self):
        registry = SkillRegistry(confidence_threshold=0.5)
        skill = CodeSkill("test", "def solve(t): return t")
        trigger = AlwaysTrigger()
        registry.register(skill, trigger)

        result = registry.check_triggers("anything")
        assert len(result) == 1
        assert result[0] == ("test", 1.0)

    def test_multiple_triggers_sorted(self):
        registry = SkillRegistry(confidence_threshold=0.0)

        skill1 = CodeSkill("low", "def solve(t): return 1")
        skill2 = CodeSkill("high", "def solve(t): return 2")

        # Low confidence trigger
        trigger1 = KeywordTrigger("kw", ["rare", "unusual", "special", "unique"])
        # High confidence trigger
        trigger2 = AlwaysTrigger()

        registry.register(skill1, trigger1)
        registry.register(skill2, trigger2)

        result = registry.check_triggers("test")
        # High confidence first
        assert result[0][0] == "high"
        assert result[0][1] == 1.0


class TestSkillRegistryProcess:
    """Tests for processing text."""

    def test_process_no_match(self):
        registry = SkillRegistry()
        result = registry.process("test")
        assert result is None

    def test_process_with_match(self):
        registry = SkillRegistry()
        skill = CodeSkill("upper", "def solve(t): return t.upper()")
        registry.register(skill, AlwaysTrigger())

        result = registry.process("hello")
        assert result is not None
        assert result.output == "HELLO"
        assert result.metadata["skill_name"] == "upper"
        assert result.metadata["trigger_confidence"] == 1.0

    def test_process_uses_highest_confidence(self):
        registry = SkillRegistry(confidence_threshold=0.0)

        skill1 = CodeSkill("one", "def solve(t): return 1")
        skill2 = CodeSkill("two", "def solve(t): return 2")

        # skill2 has higher confidence trigger
        registry.register(skill1, KeywordTrigger("kw", ["a", "b", "c", "d"]))
        registry.register(skill2, AlwaysTrigger())

        result = registry.process("a")  # Only matches one keyword
        assert result.output == 2  # Uses skill2 (always triggers)

    def test_process_with_skill_error(self):
        registry = SkillRegistry()
        skill = CodeSkill("broken", "def solve(t): raise ValueError('oops')")
        registry.register(skill, AlwaysTrigger())

        result = registry.process("test")
        assert result is not None
        assert not result.success
        assert "ValueError" in result.error


class TestSkillRegistryProcessAll:
    """Tests for processing with all matching skills."""

    def test_process_all_no_match(self):
        registry = SkillRegistry()
        results = registry.process_all("test")
        assert results == []

    def test_process_all_single_match(self):
        registry = SkillRegistry()
        skill = CodeSkill("upper", "def solve(t): return t.upper()")
        registry.register(skill, AlwaysTrigger())

        results = registry.process_all("hello")
        assert len(results) == 1
        assert results[0].output == "HELLO"

    def test_process_all_multiple_matches(self):
        registry = SkillRegistry(confidence_threshold=0.0)

        skill1 = CodeSkill("one", "def solve(t): return 1")
        skill2 = CodeSkill("two", "def solve(t): return 2")

        registry.register(skill1, AlwaysTrigger())
        registry.register(skill2, AlwaysTrigger())

        results = registry.process_all("test")
        assert len(results) == 2
        outputs = {r.output for r in results}
        assert outputs == {1, 2}


class TestSkillRegistrySave:
    """Tests for saving registry."""

    def test_save_creates_file(self):
        registry = SkillRegistry()
        skill = CodeSkill("test", "def solve(t): return t", "Test skill")
        registry.register(skill, AlwaysTrigger())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry"
            registry.save(path)

            assert (path / "skills.json").exists()

    def test_save_content(self):
        import json

        registry = SkillRegistry()
        skill = CodeSkill("test", "def solve(t): return t", "Test skill")
        registry.register(skill, AlwaysTrigger(), priority=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry"
            registry.save(path)

            with open(path / "skills.json") as f:
                data = json.load(f)

            assert "test" in data
            assert data["test"]["skill"]["name"] == "test"
            assert data["test"]["trigger_name"] == "always"
            assert data["test"]["priority"] == 5


class TestSkillRegistryListSkills:
    """Tests for listing skills."""

    def test_list_empty(self):
        registry = SkillRegistry()
        assert registry.list_skills() == []

    def test_list_skills(self):
        registry = SkillRegistry()
        skill = CodeSkill("test", "def solve(t): return t", "Test description")
        registry.register(skill, AlwaysTrigger(), priority=3)

        skills = registry.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "test"
        assert skills[0]["description"] == "Test description"
        assert skills[0]["trigger"] == "always"
        assert skills[0]["priority"] == 3
        assert skills[0]["version"] == 1


class TestSkillRegistryIntegration:
    """Integration tests for SkillRegistry."""

    def test_grammar_checker_scenario(self):
        """Test a realistic grammar checking scenario."""
        registry = SkillRegistry(confidence_threshold=0.3)

        # Grammar skill that checks for common errors
        grammar_code = """
def solve(text):
    errors = []
    # Check for double spaces
    if "  " in text:
        errors.append("double space")
    # Check sentence starts with capital
    if text and text[0].islower():
        errors.append("lowercase start")
    return {"has_errors": len(errors) > 0, "errors": errors}
"""
        grammar_skill = CodeSkill("grammar", grammar_code, "Checks grammar")
        grammar_trigger = KeywordTrigger(
            "grammar_kw",
            ["grammar", "correct", "sentence", "check"],
        )

        registry.register(grammar_skill, grammar_trigger)

        # Test with triggering text
        result = registry.process("Is this sentence correct?")
        assert result is not None
        assert result.success
        assert result.output["has_errors"] is False

        # Test with error
        result2 = registry.process("check this  sentence")
        assert result2.output["has_errors"] is True
        assert "double space" in result2.output["errors"]

    def test_pii_detector_scenario(self):
        """Test a PII detection scenario."""
        registry = SkillRegistry()

        pii_code = """
import re

def solve(text):
    pii_found = []
    # SSN pattern
    ssn_pattern = r"\\b\\d{3}-\\d{2}-\\d{4}\\b"
    for match in re.finditer(ssn_pattern, text):
        pii_found.append({"type": "SSN", "value": match.group(), "start": match.start()})
    # Email pattern
    email_pattern = r"\\b[\\w.-]+@[\\w.-]+\\.\\w+\\b"
    for match in re.finditer(email_pattern, text):
        pii_found.append({"type": "EMAIL", "value": match.group(), "start": match.start()})
    return pii_found
"""
        pii_skill = CodeSkill("pii", pii_code, "Detects PII")
        pii_trigger = PatternTrigger(
            "pii_patterns",
            [r"\d{3}-\d{2}-\d{4}", r"[\w.-]+@[\w.-]+\.\w+"],
        )

        registry.register(pii_skill, pii_trigger)

        # Test with SSN
        result = registry.process("My SSN is 123-45-6789")
        assert result is not None
        assert len(result.output) == 1
        assert result.output[0]["type"] == "SSN"

        # Test with email
        result2 = registry.process("Email me at test@example.com")
        assert len(result2.output) == 1
        assert result2.output[0]["type"] == "EMAIL"

        # Test with no PII
        result3 = registry.process("Hello world")
        assert result3 is None  # No trigger match
