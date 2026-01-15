"""
Skill Registry - manages skills and their triggers.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from self_distill.skills.base import Skill, Trigger, CodeSkill, SkillResult


@dataclass
class RegisteredSkill:
    """A skill with its trigger."""

    skill: Skill
    trigger: Trigger
    priority: int = 0  # Higher priority skills are checked first


class SkillRegistry:
    """
    Registry for managing skills and triggers.

    Usage:
        registry = SkillRegistry()
        registry.register(grammar_skill, grammar_trigger)

        # Process text
        result = registry.process("Is this sentence correct?")
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.skills: dict[str, RegisteredSkill] = {}
        self.confidence_threshold = confidence_threshold

    def register(
        self,
        skill: Skill,
        trigger: Trigger,
        priority: int = 0,
    ) -> None:
        """Register a skill with its trigger."""
        self.skills[skill.name] = RegisteredSkill(
            skill=skill,
            trigger=trigger,
            priority=priority,
        )

    def unregister(self, name: str) -> bool:
        """Remove a skill from the registry."""
        if name in self.skills:
            del self.skills[name]
            return True
        return False

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        if name in self.skills:
            return self.skills[name].skill
        return None

    def check_triggers(self, text: str) -> list[tuple[str, float]]:
        """
        Check all triggers and return skills that should activate.

        Returns:
            List of (skill_name, confidence) tuples, sorted by confidence
        """
        results = []

        for name, registered in self.skills.items():
            confidence = registered.trigger.check(text)
            if confidence >= self.confidence_threshold:
                results.append((name, confidence))

        # Sort by confidence (highest first), then by priority
        results.sort(key=lambda x: (-x[1], -self.skills[x[0]].priority))
        return results

    def process(self, text: str) -> Optional[SkillResult]:
        """
        Process text using the best matching skill.

        Returns:
            SkillResult from the highest-confidence skill, or None if no skill triggers
        """
        triggered = self.check_triggers(text)

        if not triggered:
            return None

        # Use the highest confidence skill
        skill_name, confidence = triggered[0]
        skill = self.skills[skill_name].skill

        result = skill.run(text)
        result.metadata["skill_name"] = skill_name
        result.metadata["trigger_confidence"] = confidence

        return result

    def process_all(self, text: str) -> list[SkillResult]:
        """Process text with ALL triggered skills."""
        triggered = self.check_triggers(text)
        results = []

        for skill_name, confidence in triggered:
            skill = self.skills[skill_name].skill
            result = skill.run(text)
            result.metadata["skill_name"] = skill_name
            result.metadata["trigger_confidence"] = confidence
            results.append(result)

        return results

    def save(self, path: Path) -> None:
        """Save registry to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save each CodeSkill
        skills_data = {}
        for name, registered in self.skills.items():
            if isinstance(registered.skill, CodeSkill):
                skills_data[name] = {
                    "skill": registered.skill.to_dict(),
                    "trigger_name": registered.trigger.name,
                    "priority": registered.priority,
                }

        with open(path / "skills.json", "w") as f:
            json.dump(skills_data, f, indent=2)

    def list_skills(self) -> list[dict]:
        """List all registered skills."""
        return [
            {
                "name": name,
                "description": reg.skill.description,
                "trigger": reg.trigger.name,
                "priority": reg.priority,
                "version": getattr(reg.skill, "version", 1),
            }
            for name, reg in self.skills.items()
        ]

    def __len__(self) -> int:
        return len(self.skills)

    def __contains__(self, name: str) -> bool:
        return name in self.skills
