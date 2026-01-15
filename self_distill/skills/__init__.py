"""
Skills and Triggers Architecture

A skill is a function that solves a specific type of problem.
A trigger determines when a skill should be activated.

Example:
    trigger = GrammarTrigger()  # Returns confidence 0.0-1.0
    skill = GrammarSkill()      # The actual solver function

    if trigger.check(text) > 0.7:
        result = skill.run(text)
"""

from self_distill.skills.base import Skill, Trigger, SkillResult
from self_distill.skills.registry import SkillRegistry

__all__ = [
    "Skill",
    "Trigger",
    "SkillResult",
    "SkillRegistry",
]
