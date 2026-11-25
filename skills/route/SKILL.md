---
name: route
description: Orchestrate a task. Classify it, let the JEV router pick the Claude model and effort, run the work in a subagent on that model, then return the result. Use when the user types /route <task>, or asks to route or delegate a task to the right model ("route this", "delegasikan", "pilih model dan kerjakan").
---

# route

You are the orchestrator. Classify, route, delegate, check, report. Do not do the task yourself.

If no task was given, ask for one in a single line and stop.

## 1. Classify

Read the task (any language) and the conversation so far. Pick one value per signal:

| Signal | Values (low to high) | Guide |
|---|---|---|
| complexity | TRIVIAL, SIMPLE, MODERATE, COMPLEX, FRONTIER | label or route; summarise, translate, small edit; write, review or debug code; architecture, multi-file refactor, research; novel reasoning or proofs |
| quality | DRAFT, GOOD, HIGH, CRITICAL, FRONTIER | prototype; normal work; expert; production, security or money paths; pushing limits |
| cost | IGNORE, LOW, MEDIUM, HIGH, CRITICAL | MEDIUM unless the user says cheap or best quality |
| speed | NONE, LOW, MEDIUM, HIGH, CRITICAL | MEDIUM unless the user says fast or no rush |
| agentic | flag | the work needs many tool calls across files or steps |

## 2. Route

```bash
PYTHONPATH="$HOME/bagusfolder/claude-model-jev" python3 -m claude_model_jev \
  --complexity MODERATE --quality GOOD --cost MEDIUM --speed MEDIUM \
  --candidates claude-fable-5-2,claude-opus-5-5,claude-sonnet-5-5,claude-haiku-5-5
```

Add `--agentic` when it applies. The candidates are the latest model per tier, because the Agent tool selects a model by tier only.

Before delegating, tell the user one line built from the JSON, for example:
`→ Opus 5.5 · effort high · P 0.32 (runner-up Sonnet 5.5 0.32) · MODERATE/HIGH`

## 3. Delegate

Call the Agent tool with:

- `model`: the JSON `tier` (fable, opus, sonnet or haiku)
- `effort`: the JSON `effort`. This skill explicitly asks for it.
- `subagent_type`: `general-purpose`, or `Explore` when the task only reads code. Never `fork`: a fork ignores `model`.
- `description`: 3 to 5 words
- `prompt`: a self-contained brief. The subagent sees none of this conversation, so include the task in full, the working directory and file paths, anything already decided, constraints, whether to edit files or only research, and what to return.

If the task splits into independent parts of clearly different difficulty, classify and route each part on its own and send the Agent calls in one message.

## 4. Check and report

Treat the subagent's report as a claim. If it changed files, read the diff before reporting. Then give the user the result, ending with one line naming the model that did the work.
