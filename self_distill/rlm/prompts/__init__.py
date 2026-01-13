"""
Self-Distill RLM Prompt Versions

Each prompt version explores different approaches to guiding the LLM
in creating tools for self-distillation.

Versions:
- v1_basic_constitution: Core principles with single example
- v2_cost_aware: Explicit cost/benefit reasoning
- v3_pattern_learning: Focus on pattern recognition
- v4_minimal: Concise instructions, maximum autonomy
- v5_reflective: Explicit reasoning about decisions
"""

from self_distill.rlm.prompts.v1_basic_constitution import PROMPT_V1
from self_distill.rlm.prompts.v2_cost_aware import PROMPT_V2
from self_distill.rlm.prompts.v3_pattern_learning import PROMPT_V3
from self_distill.rlm.prompts.v4_minimal import PROMPT_V4
from self_distill.rlm.prompts.v5_reflective import PROMPT_V5

PROMPTS = {
    "v1_basic": PROMPT_V1,
    "v2_cost": PROMPT_V2,
    "v3_pattern": PROMPT_V3,
    "v4_minimal": PROMPT_V4,
    "v5_reflective": PROMPT_V5,
}

PROMPT_DESCRIPTIONS = {
    "v1_basic": "Basic constitution with core principles and single example",
    "v2_cost": "Cost-aware with explicit cost/benefit analysis for tool creation",
    "v3_pattern": "Pattern learning focus - observe, learn, protect, evolve",
    "v4_minimal": "Minimal instructions, maximum LLM autonomy",
    "v5_reflective": "Explicit reasoning and self-reflection before decisions",
}

__all__ = [
    "PROMPTS",
    "PROMPT_DESCRIPTIONS",
    "PROMPT_V1",
    "PROMPT_V2",
    "PROMPT_V3",
    "PROMPT_V4",
    "PROMPT_V5",
]
