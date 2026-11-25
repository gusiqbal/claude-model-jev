# claude-model-jev

JEV-style typed decision system for selecting the optimal Claude model for a given task.

## What it does

Takes a task description, extracts signals (complexity, cost priority, speed, quality bar), and returns a probability distribution over Claude models using JEV decision concepts:

- **choice**: softmax probability distribution over model candidates
- **score**: ordered-level scoring per dimension (capability, cost, speed, quality)
- **noul**: yes/no gate checks (context window, capability threshold)
- **confidence**: gap between top-1 and top-2 probabilities

## Install

```bash
pip install .
```

## Usage

```python
from claude_model_jev import extract_signal, choose_model

signal = extract_signal("Build an autonomous agent that browses the web and writes a report")
decision = choose_model(signal)

print(decision.choice)        # e.g. claude-fable-5-2
print(decision.probabilities) # {model_id: probability, ...}
print(decision.reasoning)     # one-line rationale
```

### Direct signal construction

```python
from claude_model_jev import TaskSignal, Complexity, QualityBar, choose_model

signal = TaskSignal(
    complexity=Complexity.COMPLEX,
    quality_bar=QualityBar.HIGH,
    is_agentic=True,
)
decision = choose_model(signal)
```

### Filter candidates

```python
decision = choose_model(signal, candidates=["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
```

## CLI

```bash
claude-model-jev "Summarise this article"                      # keyword extraction (English)
claude-model-jev --complexity COMPLEX --quality HIGH --agentic  # explicit signals, any language
python3 -m claude_model_jev.router                              # demo across 7 task types
```

Prints JSON: `model_id`, `tier`, `name`, `effort`, `confidence`, `signal`, `top`. Flags override signals extracted from the text. `--candidates` limits the pool (comma-separated model IDs).

## Claude Code orchestration

Let an LLM classify the task, since keyword extraction only knows English. Pass the signals as flags, then dispatch a subagent with the returned `tier` as its model and `effort` as its effort. See `skills/route/SKILL.md` for a ready-made Claude Code skill.

## How it works

Each model is scored on 6 dimensions with adaptive weights. A power curve (`0.3 + 0.7 * cap^(1 + need*3)`) amplifies capability gaps for harder tasks. A hard gate at 0.5 prevents cost from compensating for insufficient capability. Scores go through softmax (temperature=0.15) for a decisive probability distribution.

## License

MIT
