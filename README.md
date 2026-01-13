# rlm-self-distill
Large Language Model Self Distillation to Rule based sub task completion


In this research we are measuring two types of Large Language Model code distillations.

- 1, Building off RLM work, we will measure if Large Langauge Models can create Macros that ensure stricter safety handling of tasks. An example is to have a PII macro run python code to scrub PII before calling the LLM recursively on the cleaned data.

- 2, Can the RLM system create tools that bypass LLM sub tasks with Rule based systems that are more auditable.


The primary setup will be in updating the RLM loop to have a suite of Rules, Macros, and Recursive calls. The RLM paper already showcased filtering mechanisms that reduced token use.

Motivations for the work are 3 fold.
- First, increased auditability, the more code run throughout the process the more interpretable the end result.
- Second, reduced costs by leveraging the right tool for the task.
- Third, performance based costs can reduce loops and LLM calls which can be slower overall than traditional methods for certain tasks.

## Preliminary Findings

### Experiment Overview

We conducted experiments comparing baseline LLM inference against a Recursive Language Model (RLM) system that creates and executes Python tools for task completion.

**Datasets:**
- **CoLA** (Corpus of Linguistic Acceptability): Grammar acceptability classification
- **PII Detection**: Synthetic dataset with emails, phone numbers, SSNs, and IP addresses

### Key Results

| Approach | Tokens | Change |
|----------|--------|--------|
| Baseline LLM | 6,435 | - |
| Text-based Tools (exp001) | 33,849 | +426% overhead |
| Text-based Tools at Scale (exp002) | 37,063 | +576% overhead |
| Executable Python Tools (exp003) | 4,229 | -34% savings |
| **Safety Macro Pipeline (exp004)** | **12,114** | **-43% savings** |

### Analysis

**Why text-based tools failed (exp001, exp002):**
- Tool definitions were embedded as text descriptions in the system prompt
- Each new tool increased prompt size, causing cumulative overhead
- No actual code execution - tools were just documentation

**Why executable tools succeeded (exp003):**
- Tools are actual Python files saved to disk (`tools/grammar/`, `tools/pii/`, `tools/classifiers/`)
- Executed locally via `importlib` - zero prompt overhead
- Task classifier routes inputs to appropriate tool pipelines
- LLM only called when tools fail (0 fallbacks in initial experiments)

**Safety Macro Pipeline (exp004):**
- **Key insight**: LLM never sees raw PII data
- Safety macro runs FIRST on all inputs, detecting and masking sensitive data
- Completion rules handle tasks that tools can solve
- LLM fallback only receives sanitized text with `[TYPE_MASKED]` placeholders
- Result: 43% token savings + 100% PII protection (0 raw exposures vs 30 in baseline)

### Tool Execution Statistics

| Tool | Type | Executions | Success Rate |
|------|------|------------|--------------|
| task_classifier | classifier | 65 | 100% |
| basic_grammar | grammar | 60 | 100% |
| email_detector | pii | 5 | 100% |
| phone_detector | pii | 5 | 100% |
| ssn_detector | pii | 5 | 100% |
| ip_detector | pii | 5 | 100% |

### Architecture

**Experiment 003: Tool-First Pipeline**
```
Input Text
    │
    ▼
┌─────────────────┐
│ Task Classifier │  ← Python tool (classifies grammar vs PII)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Grammar │ │  PII   │
│ Tools  │ │ Tools  │
└────────┘ └────────┘
    │         │
    ▼         ▼
┌─────────────────┐
│  LLM Fallback   │  ← Only if tools fail
└─────────────────┘
```

**Experiment 004: Safety Macro Pipeline**
```
Input Text (may contain PII)
    │
    ▼
┌─────────────────────────────┐
│      SAFETY MACRO           │  ← ALWAYS runs first
│  - Detect PII patterns      │
│  - Mask sensitive data      │
│  - [EMAIL_MASKED], etc.     │
└─────────────┬───────────────┘
              │
    Sanitized Text (PII masked)
              │
              ▼
┌─────────────────────────────┐
│    COMPLETION RULES         │  ← Check if tools can handle
│  - Grammar tools            │
│  - PII detection tools      │
└─────────────┬───────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
  Tool Result    LLM Fallback
                (sees ONLY sanitized text)
```

## Setup

```bash
# Install dependencies
uv sync

# Run experiments
PYTHONPATH=. uv run python experiments/exp003_executable_tools.py --model "llama3.2:3b" --cola 50 --pii 15

# View MLflow dashboard
mlflow ui
```

## Project Structure

```
self_distill/
├── datasets/          # Dataset loaders (CoLA, PII, GSM8K)
├── clients/           # Ollama client wrapper
└── tracking/          # MLflow experiment tracking

experiments/
├── exp001_recursive_tool_creation.py   # Text-based tools (baseline)
├── exp002_scaling_analysis.py          # Scale testing
├── exp003_executable_tools.py          # Executable Python tools
└── exp004_safety_macro_pipeline.py     # Safety-first with PII masking

tests/                 # Mirrors self_distill structure
```
