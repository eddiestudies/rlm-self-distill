"""Tests for skills base classes."""

import pytest

from self_distill.skills.base import (
    SkillResult,
    Trigger,
    Skill,
    CodeSkill,
    PatternTrigger,
    KeywordTrigger,
    AlwaysTrigger,
)


class TestSkillResult:
    """Tests for SkillResult dataclass."""

    def test_result_creation(self):
        result = SkillResult(output="test")
        assert result.output == "test"
        assert result.confidence == 1.0
        assert result.metadata == {}
        assert result.error is None

    def test_result_with_all_fields(self):
        result = SkillResult(
            output=42,
            confidence=0.9,
            metadata={"key": "value"},
            error=None,
        )
        assert result.output == 42
        assert result.confidence == 0.9
        assert result.metadata == {"key": "value"}

    def test_success_true_when_no_error(self):
        result = SkillResult(output="ok")
        assert result.success is True

    def test_success_false_when_error(self):
        result = SkillResult(output=None, error="Something went wrong")
        assert result.success is False

    def test_result_with_none_output(self):
        result = SkillResult(output=None, confidence=0.0)
        assert result.output is None
        assert result.success is True  # No error means success


class TestCodeSkill:
    """Tests for CodeSkill class."""

    def test_valid_code_compiles(self):
        code = '''
def solve(text):
    return len(text)
'''
        skill = CodeSkill("counter", code, "Counts characters")
        assert skill.is_valid
        assert skill._compile_error is None

    def test_missing_solve_function(self):
        code = '''
def process(text):
    return text
'''
        skill = CodeSkill("bad", code)
        assert not skill.is_valid
        assert "solve" in skill._compile_error

    def test_syntax_error_in_code(self):
        code = '''
def solve(text)
    return text
'''
        skill = CodeSkill("broken", code)
        assert not skill.is_valid
        assert "SyntaxError" in skill._compile_error

    def test_run_returns_result(self):
        code = '''
def solve(text):
    return text.upper()
'''
        skill = CodeSkill("upper", code)
        result = skill.run("hello")
        assert result.success
        assert result.output == "HELLO"
        assert result.confidence == 1.0

    def test_run_with_compile_error(self):
        skill = CodeSkill("bad", "not valid python {{{")
        result = skill.run("test")
        assert not result.success
        assert "Compile error" in result.error

    def test_run_with_runtime_error(self):
        code = '''
def solve(text):
    return 1 / 0
'''
        skill = CodeSkill("divzero", code)
        result = skill.run("test")
        assert not result.success
        assert "Runtime error" in result.error
        assert "ZeroDivisionError" in result.error

    def test_update_code_success(self):
        code_v1 = '''
def solve(text):
    return 1
'''
        code_v2 = '''
def solve(text):
    return 2
'''
        skill = CodeSkill("versioned", code_v1)
        assert skill.version == 1
        assert skill.run("x").output == 1

        success = skill.update_code(code_v2)
        assert success
        assert skill.version == 2
        assert skill.run("x").output == 2

    def test_update_code_failure(self):
        code_v1 = '''
def solve(text):
    return 1
'''
        skill = CodeSkill("versioned", code_v1)
        success = skill.update_code("invalid {{{")
        assert not success
        assert skill.version == 2  # Version still increments
        assert not skill.is_valid

    def test_to_dict(self):
        code = '''
def solve(text):
    return True
'''
        skill = CodeSkill("test_skill", code, "A test skill")
        d = skill.to_dict()

        assert d["name"] == "test_skill"
        assert d["description"] == "A test skill"
        assert d["version"] == 1
        assert d["code"] == code
        assert d["is_valid"] is True
        assert d["compile_error"] is None

    def test_to_dict_with_error(self):
        skill = CodeSkill("broken", "bad code {{{")
        d = skill.to_dict()

        assert d["is_valid"] is False
        assert d["compile_error"] is not None

    def test_repr(self):
        skill = CodeSkill("my_skill", "def solve(t): return t")
        assert "my_skill" in repr(skill)
        assert "v1" in repr(skill)


class TestPatternTrigger:
    """Tests for PatternTrigger class."""

    def test_single_pattern_match(self):
        trigger = PatternTrigger("email", [r"\b[\w.-]+@[\w.-]+\.\w+\b"])
        confidence = trigger.check("Contact me at test@example.com")
        assert confidence == 1.0

    def test_single_pattern_no_match(self):
        trigger = PatternTrigger("email", [r"\b[\w.-]+@[\w.-]+\.\w+\b"])
        confidence = trigger.check("No email here")
        assert confidence == 0.0

    def test_multiple_patterns_partial_match(self):
        trigger = PatternTrigger(
            "pii",
            [r"\d{3}-\d{2}-\d{4}", r"\b[\w.-]+@[\w.-]+\.\w+\b", r"\d{3}-\d{3}-\d{4}"],
        )
        # Only one pattern matches
        confidence = trigger.check("SSN: 123-45-6789")
        assert 0 < confidence < 1.0
        assert confidence == pytest.approx(1 / 3)

    def test_multiple_patterns_all_match(self):
        trigger = PatternTrigger("all", [r"hello", r"world"])
        confidence = trigger.check("hello world")
        assert confidence == 1.0

    def test_threshold_not_met(self):
        trigger = PatternTrigger("multi", [r"a", r"b", r"c"], threshold=2)
        confidence = trigger.check("only a here")
        assert confidence == 0.0

    def test_threshold_met(self):
        trigger = PatternTrigger("multi", [r"a", r"b", r"c"], threshold=2)
        confidence = trigger.check("has a and b")
        assert confidence > 0.0

    def test_case_insensitive(self):
        trigger = PatternTrigger("case", [r"HELLO"])
        confidence = trigger.check("hello world")
        assert confidence == 1.0

    def test_description(self):
        trigger = PatternTrigger("test", [r"a", r"b", r"c"])
        assert "3 patterns" in trigger.description

    def test_repr(self):
        trigger = PatternTrigger("my_trigger", [r"test"])
        assert "my_trigger" in repr(trigger)


class TestKeywordTrigger:
    """Tests for KeywordTrigger class."""

    def test_single_keyword_match(self):
        trigger = KeywordTrigger("grammar", ["grammatical"])
        confidence = trigger.check("Is this grammatical?")
        assert confidence == 1.0

    def test_single_keyword_no_match(self):
        trigger = KeywordTrigger("grammar", ["grammatical"])
        confidence = trigger.check("Is this correct?")
        assert confidence == 0.0

    def test_case_insensitive(self):
        trigger = KeywordTrigger("test", ["HELLO"])
        confidence = trigger.check("hello there")
        assert confidence == 1.0

    def test_multiple_keywords_partial(self):
        trigger = KeywordTrigger("pii", ["ssn", "email", "phone"])
        confidence = trigger.check("My ssn is private")
        assert 0 < confidence < 1.0

    def test_threshold_not_met(self):
        trigger = KeywordTrigger("multi", ["a", "b", "c"], threshold=2)
        confidence = trigger.check("only a here")
        assert confidence == 0.0

    def test_description(self):
        trigger = KeywordTrigger("test", ["a", "b", "c", "d", "e"])
        assert "a" in trigger.description


class TestAlwaysTrigger:
    """Tests for AlwaysTrigger class."""

    def test_always_returns_one(self):
        trigger = AlwaysTrigger()
        assert trigger.check("anything") == 1.0
        assert trigger.check("") == 1.0

    def test_name_and_description(self):
        trigger = AlwaysTrigger()
        assert trigger.name == "always"
        assert "Always" in trigger.description


class TestTriggerAbstract:
    """Tests for Trigger ABC."""

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            Trigger()

    def test_custom_trigger_implementation(self):
        class CustomTrigger(Trigger):
            name = "custom"

            def check(self, text: str) -> float:
                return 0.5 if "test" in text else 0.0

        trigger = CustomTrigger()
        assert trigger.check("this is a test") == 0.5
        assert trigger.check("nothing here") == 0.0


class TestSkillAbstract:
    """Tests for Skill ABC."""

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            Skill()

    def test_custom_skill_implementation(self):
        class CustomSkill(Skill):
            name = "custom"

            def run(self, text: str) -> SkillResult:
                return SkillResult(output=text.upper())

        skill = CustomSkill()
        result = skill.run("hello")
        assert result.output == "HELLO"
