"""
Base classes for Skills and Triggers.

Skills are functions that solve problems.
Triggers determine when to activate a skill.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import re


@dataclass
class SkillResult:
    """Result from running a skill."""
    output: Any
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class Trigger(ABC):
    """
    Base class for triggers.

    A trigger checks if a skill should be activated for given input.
    Returns a confidence score between 0.0 and 1.0.
    """

    name: str = "base_trigger"
    description: str = "Base trigger"

    @abstractmethod
    def check(self, text: str) -> float:
        """
        Check if this trigger should activate.

        Args:
            text: Input text to check

        Returns:
            Confidence score 0.0-1.0 (0 = don't activate, 1 = definitely activate)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class Skill(ABC):
    """
    Base class for skills.

    A skill is a function that solves a specific type of problem.
    """

    name: str = "base_skill"
    description: str = "Base skill"
    version: int = 1

    @abstractmethod
    def run(self, text: str) -> SkillResult:
        """
        Run the skill on input text.

        Args:
            text: Input text to process

        Returns:
            SkillResult with output and metadata
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, v{self.version})"


class CodeSkill(Skill):
    """
    A skill defined by Python code string.

    The code must define a function with signature:
        def solve(text: str) -> Any
    """

    def __init__(self, name: str, code: str, description: str = ""):
        self.name = name
        self.code = code
        self.description = description
        self.version = 1
        self._func: Optional[Callable] = None
        self._compile_error: Optional[str] = None
        self._compile()

    def _compile(self) -> None:
        """Compile the code and extract the solve function."""
        try:
            namespace = {"__builtins__": __builtins__}
            exec(self.code, namespace)

            if "solve" not in namespace:
                self._compile_error = "Code must define a 'solve(text: str)' function"
                return

            self._func = namespace["solve"]
            self._compile_error = None

        except Exception as e:
            self._compile_error = f"{type(e).__name__}: {str(e)}"
            self._func = None

    def run(self, text: str) -> SkillResult:
        """Run the skill."""
        if self._compile_error:
            return SkillResult(
                output=None,
                confidence=0.0,
                error=f"Compile error: {self._compile_error}",
            )

        try:
            result = self._func(text)
            return SkillResult(output=result, confidence=1.0)
        except Exception as e:
            return SkillResult(
                output=None,
                confidence=0.0,
                error=f"Runtime error: {type(e).__name__}: {str(e)}",
            )

    def update_code(self, new_code: str) -> bool:
        """Update the skill's code. Returns True if compilation succeeded."""
        self.code = new_code
        self.version += 1
        self._compile()
        return self._compile_error is None

    @property
    def is_valid(self) -> bool:
        return self._compile_error is None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "code": self.code,
            "is_valid": self.is_valid,
            "compile_error": self._compile_error,
        }


class PatternTrigger(Trigger):
    """
    A trigger based on regex patterns.

    Returns confidence based on how many patterns match.
    """

    def __init__(self, name: str, patterns: list[str], threshold: int = 1):
        """
        Args:
            name: Trigger name
            patterns: List of regex patterns
            threshold: Minimum matches to return confidence > 0
        """
        self.name = name
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.threshold = threshold
        self.description = f"Pattern trigger with {len(patterns)} patterns"

    def check(self, text: str) -> float:
        """Check patterns against text."""
        matches = sum(1 for p in self.patterns if p.search(text))

        if matches < self.threshold:
            return 0.0

        # Confidence scales with number of matches
        return min(1.0, matches / len(self.patterns))


class KeywordTrigger(Trigger):
    """Simple keyword-based trigger."""

    def __init__(self, name: str, keywords: list[str], threshold: int = 1):
        self.name = name
        self.keywords = [k.lower() for k in keywords]
        self.threshold = threshold
        self.description = f"Keyword trigger: {keywords[:3]}..."

    def check(self, text: str) -> float:
        text_lower = text.lower()
        matches = sum(1 for k in self.keywords if k in text_lower)

        if matches < self.threshold:
            return 0.0

        return min(1.0, matches / len(self.keywords))


class AlwaysTrigger(Trigger):
    """Trigger that always activates (for testing)."""

    name = "always"
    description = "Always triggers"

    def check(self, text: str) -> float:
        return 1.0
