# Experiment Log: RLM Self-Distillation Research

Last updated: 2026-01-15

## Overview

This log documents experimental findings from the RLM (Recursive Language Models) self-distillation research project. The goal is to explore whether models can create reusable tools/skills through iterative refinement to improve performance on tasks.

## Key Findings

1. **Hand-built rules work and save tokens.** Experiments 003 and 004 showed 34-43% token savings using pre-built Python rules (not LLM-generated). The rules ran as pre-completion hooks, avoiding PII cases and handling grammar checks before the LLM was called.

2. **LLM-generated rules don't work with smaller models.** When asking models (llama, qwen, deepseek) to CREATE the rules, they fail to produce usable code. Only Claude Opus 4.5 showed promise for rule generation.

---

## Experiments Summary

### Exp001: Recursive Tool Creation (Baseline)
- **Model**: llama3.2:3b
- **Finding**: Direct LM approach used fewer tokens (1,478) vs baseline (2,448) vs with-tools (12,859)
- **Issue**: Tools created were not executable - just descriptions/concepts
- **Result**: No measurable improvement from tool creation

### Exp002: Scaling Analysis
- **Goal**: Test if more compute helps
- **Result**: Inconclusive - model limitations more significant than scale

### Exp003: Executable Tools ✅ SUCCESS
- **Goal**: Use hand-built executable Python tools as pre-completion hooks
- **Approach**: Pre-built rules (NOT LLM-generated) saved to disk, executed via `importlib`
- **Result**: **-34% token savings** vs baseline
- **Tools**: task_classifier, basic_grammar, email/phone/ssn/ip detectors
- **Key insight**: Tools executed locally with zero prompt overhead; LLM only called when tools fail

### Exp004: Safety Macro Pipeline ✅ SUCCESS
- **Goal**: Safety-first pipeline where LLM never sees raw PII
- **Approach**: Pre-built Safety Macro runs FIRST, masks PII before any LLM call
- **Result**: **-43% token savings** + 100% PII protection (0 raw exposures vs 30 in baseline)
- **Architecture**:
  1. Safety Macro detects/masks PII → `[EMAIL_MASKED]`, `[SSN_MASKED]`
  2. Completion rules check if tools can handle the task
  3. LLM fallback only receives sanitized text
- **Key insight**: Hand-built rules work; the question is whether LLMs can CREATE such rules

### Exp005: RLM Tool Creation (Multiple Runs)
- **Runs**: 20+ iterations with various parameters
- **Finding**: High variance in outputs, no consistent improvement pattern

### Exp006: Prompt Variations
- **Model**: qwen2.5-coder:32b
- **Duration**: ~7 hours (430 minutes)
- **Tasks per prompt**: 31
- **Prompts tested**: Multiple variations (v1_basic, v2_cost, etc.)
- **Result**:
  - 0 hooks created
  - 0 replacements created
  - 0 tools with correct contract
- **Conclusion**: Even with explicit cost/benefit prompts, model failed to create usable tools

### Exp007: Evidence-Guided Tools
- **Model**: ollama (likely qwen or llama variant)
- **Innovation**: Embedding-based clustering to detect patterns BEFORE prompting
- **Tasks**: 51
- **Duration**: ~71 minutes
- **Result**:
  - Patterns detected: 0
  - Patterns triggered tools: 0
  - Hooks/replacements created: 0
- **Token usage**: Baseline 12,607 vs RLM 330,877 (26x more!)
- **Conclusion**: Embedding detection worked but model couldn't capitalize on detected patterns

### Exp008: Large PII Evidence
- **Model**: ollama variant
- **Duration**: ~8.75 hours
- **Tasks**: 190/200 processed
- **Result**:
  - Patterns detected: 6
  - Patterns triggered tools: 0
  - Clusters found: 37
  - Largest cluster: 12 items
  - Avg cluster size: 4.2
- **Token usage**: Baseline 83,440 vs RLM 1,482,190 (17.8x more!)
- **Conclusion**: Clustering works well at scale, but model still fails to create useful tools

### Exp009: Tool Validation
- **Goal**: Validate generated tools actually work
- **Result**: Most generated tools non-functional

### Exp010: Iterative Refinement
- **Goal**: Refine tools based on error feedback
- **Result**: Limited improvement through iteration

### Exp011: Grammar Skill Development
- **Model**: deepseek-r1:70b
- **Task**: Create grammar correctness classifier for CoLA dataset
- **Iterations**: 25
- **Train/Test**: 300/150 samples
- **Best accuracy**: 68% (iteration 9)
- **Key issue**: 18/25 iterations failed (produced unparseable code)
- **Critical finding**: Best "skill" just returns True always (trivial classifier matching class imbalance)

```python
# The "best" skill literally just returns True
def solve(text):
    # ... some unused checks ...
    return True  # Always predicts "grammatically incorrect"
```

---

## Technical Observations

### Token Efficiency - What Works vs What Doesn't

**Hand-built rules (SUCCESS):**
| Approach | Tokens | vs Baseline |
|----------|--------|-------------|
| Baseline LLM | 6,435 | - |
| Executable Tools (exp003) | 4,229 | **-34% savings** |
| Safety Macro Pipeline (exp004) | 3,668* | **-43% savings** |

*Effective tokens after PII masking reduces payload size

**LLM-generated rules (FAILURE):**
| Experiment | Baseline Tokens | RLM Tokens | Ratio |
|------------|-----------------|------------|-------|
| Exp007 | 12,607 | 330,877 | 26x more |
| Exp008 | 83,440 | 1,482,190 | 17.8x more |

When LLMs try to CREATE rules, they use 17-26x more tokens with no accuracy improvement.

### Clustering (Evidence Module)
The embedding-based clustering successfully identifies similar tasks:
- Exp008 found 37 clusters with avg size 4.2
- Largest cluster had 12 similar items
- However, detected patterns did not translate to tool creation

### Code Generation Quality
- Small models (llama3.2:3b, qwen2.5-coder:32b) struggle with:
  - Syntactically correct Python
  - Following output format specifications
  - Creating tools that actually execute
- deepseek-r1:70b showed improvement but still 72% failure rate on iterations

---

## What Works

1. **Hand-built pre-completion rules**: 34-43% token savings (exp003, exp004)
2. **Safety Macro pattern**: PII masking before LLM call = 100% protection
3. **Embedding clustering**: Successfully detects similar tasks
4. **Architecture**: The pipeline structure is sound
5. **Opus 4.5 for rule generation**: Shows promise (needs more testing)

## What Doesn't Work (Yet)

1. **Simple prompts + weak models**: Cannot produce usable tools
2. **Iterative refinement with weak models**: High failure rate, trivial solutions
3. **Cost/benefit prompting**: Models don't internalize cost tradeoffs

---

## Next Steps to Consider

1. **Test with Opus 4.5**: The architecture may work with sufficient model capability
2. **Hybrid approach**: Use Opus 4.5 for tool creation, smaller models for execution
3. **Constrained generation**: Limit output format to reduce parsing failures
4. **Few-shot examples**: Provide working tool examples in prompts
5. **Verification loop**: Add execution verification before accepting tools

---

## Repository Structure

```
experiments/
  exp001_recursive_tool_creation.py
  exp002_scaling_analysis.py
  exp003_executable_tools.py
  exp004_safety_macro_pipeline.py
  exp005_rlm_tool_creation.py
  exp006_prompt_variations.py
  exp007_evidence_guided_tools.py
  exp008_large_pii_evidence.py
  exp009_tool_validation.py
  exp010_iterative_refinement.py
  exp011_grammar_skill.py

self_distill/
  evidence/       # Embedding-based pattern detection
  skills/         # Code skill creation and validation
  datasets/       # CoLA, PII, AI4Privacy, etc.
  clients/        # Ollama client wrapper
```
