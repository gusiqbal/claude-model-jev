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

print(decision.choice)        # claude-opus-5-5
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
claude-model-jev
```

Runs the built-in demo across 7 task types.

## How it works

Each model is scored on 6 dimensions with adaptive weights. A power curve (`0.3 + 0.7 * cap^(1 + need*3)`) amplifies capability gaps for harder tasks. A hard gate at 0.5 prevents cost from compensating for insufficient capability. Scores go through softmax (temperature=0.15) for a decisive probability distribution.

## License

MIT
