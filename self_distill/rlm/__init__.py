"""RLM extensions for self-distillation with tool creation."""

from self_distill.rlm.self_distill_rlm import (
    SelfDistillRLM,
    TOOL_CREATION_PROMPT,
    get_tool_registry_setup_code,
)

__all__ = ["SelfDistillRLM", "TOOL_CREATION_PROMPT", "get_tool_registry_setup_code"]
