"""
Claude Model JEV Router
========================
JEV-style typed decision system for selecting the optimal Claude model.

Core JEV concepts applied:
- choice: probability distribution over model candidates
- score: ordered-level scoring per dimension (complexity, cost, speed, quality)
- noul: yes/no gate checks (e.g. "does this need 1M context?")
- confidence: how certain the recommendation is
- validation: probabilities sum to 1, winner has max P
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


# --- Model Registry ---

@dataclass(frozen=True)
class ClaudeModel:
    id: str
    name: str
    tier: str           # fable, opus, sonnet, haiku
    context_tokens: int # max context window
    max_output: int     # max output tokens
    input_cost: float   # USD per 1M tokens
    output_cost: float  # USD per 1M tokens
    capability: float   # 0-1 normalized capability score
    speed: float        # 0-1 normalized speed (1 = fastest)

MODELS = [
    ClaudeModel("claude-fable-5-2",  "Fable 5.2",  "fable",  1_000_000, 128_000, 8.00, 40.00, 1.00, 0.30),
    ClaudeModel("claude-opus-5-5",   "Opus 5.5",   "opus",   1_000_000, 128_000, 4.00, 20.00, 0.92, 0.50),
    ClaudeModel("claude-opus-5",     "Opus 5",     "opus",   1_000_000, 128_000, 5.00, 25.00, 0.88, 0.48),
    ClaudeModel("claude-sonnet-5-5", "Sonnet 5.5", "sonnet", 1_000_000, 128_000, 2.00, 10.00, 0.82, 0.70),
    ClaudeModel("claude-sonnet-5",   "Sonnet 5",   "sonnet", 1_000_000, 128_000, 2.00, 10.00, 0.78, 0.68),
    ClaudeModel("claude-haiku-5-5",  "Haiku 5.5",  "haiku",  1_000_000, 128_000, 0.10,  0.50, 0.60, 0.95),
    ClaudeModel("claude-haiku-4-5",  "Haiku 4.5",  "haiku",    200_000, 128_000, 1.00,  5.00, 0.50, 0.90),
]

MODEL_BY_ID = {m.id: m for m in MODELS}


# --- Task Signal Extraction ---

class Complexity(Enum):
    TRIVIAL = 0     # classification, yes/no, extraction
    SIMPLE = 1      # summarisation, Q&A, translation
    MODERATE = 2    # code generation, analysis, multi-step
    COMPLEX = 3     # architecture, long-horizon agent, research
    FRONTIER = 4    # novel reasoning, math proof, hardest tasks

class CostPriority(Enum):
    IGNORE = 0      # quality at any price
    LOW = 1         # prefer cheaper if quality holds
    MEDIUM = 2      # balance cost and quality
    HIGH = 3        # minimise cost, accept quality trade-off
    CRITICAL = 4    # cheapest possible

class SpeedPriority(Enum):
    NONE = 0        # batch, no latency constraint
    LOW = 1         # background processing
    MEDIUM = 2      # interactive but patient
    HIGH = 3        # real-time chat
    CRITICAL = 4    # sub-second, autocomplete

class QualityBar(Enum):
    DRAFT = 0       # rough output acceptable
    GOOD = 1        # production-quality text
    HIGH = 2        # expert-level reasoning
    CRITICAL = 3    # zero-error tolerance
    FRONTIER = 4    # pushing model limits


@dataclass
class TaskSignal:
    """Extracted signals from a task description."""
    complexity: Complexity = Complexity.MODERATE
    cost_priority: CostPriority = CostPriority.MEDIUM
    speed_priority: SpeedPriority = SpeedPriority.MEDIUM
    quality_bar: QualityBar = QualityBar.GOOD
    estimated_input_tokens: int = 1000
    estimated_output_tokens: int = 1000
    needs_tools: bool = False
    needs_vision: bool = False
    is_agentic: bool = False
    is_batch: bool = False
    volume: int = 1  # number of requests


# --- JEV Decision Engine ---

@dataclass
class Decision:
    """JEV-style typed decision result."""
    choice: str                        # winning model ID
    probabilities: dict[str, float]    # P(model) for each candidate
    confidence: float                  # 0-1 overall confidence
    scores: dict[str, dict[str, float]]  # per-model dimension scores
    cost_estimate: float               # estimated USD for the task
    reasoning: str                     # one-line rationale
    effort_recommendation: str         # low/medium/high/xhigh/max


def _softmax(scores: dict[str, float], temperature: float = 0.5) -> dict[str, float]:
    """Convert raw scores to probability distribution via softmax."""
    max_s = max(scores.values())
    exps = {k: math.exp((v - max_s) / temperature) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _score_model(model: ClaudeModel, signal: TaskSignal) -> dict[str, float]:
    """Score a model on each dimension. Returns dict of dimension -> score [0,1]."""

    # Capability fit: does the model MEET the complexity bar?
    # JEV noul gate: hard threshold, then gradient above it
    complexity_need = signal.complexity.value / 4.0
    if model.capability >= complexity_need:
        # ponytail: power curve scales with complexity — harder tasks amplify capability gap
        exp = 1.0 + complexity_need * 3.0
        capability_fit = 0.3 + 0.7 * (model.capability ** exp)
    else:
        gap = complexity_need - model.capability
        capability_fit = max(0.0, 0.2 * math.exp(-gap * 8.0))

    # Cost efficiency: normalised so cheapest = 1.0, most expensive = 0.0
    # Use log scale so the massive Haiku price gap doesn't dominate everything
    costs = [(m.input_cost + m.output_cost) for m in MODELS]
    model_cost = model.input_cost + model.output_cost
    log_min = math.log(min(costs) + 0.01)
    log_max = math.log(max(costs) + 0.01)
    cost_efficiency = 1.0 - (math.log(model_cost + 0.01) - log_min) / (log_max - log_min)

    # Speed fit: model speed vs task need
    speed_need = signal.speed_priority.value / 4.0
    speed_fit = model.speed
    if speed_need > 0.5 and model.speed < 0.5:
        speed_fit *= 0.4  # harsh penalty for slow model when speed matters
    elif speed_need < 0.3:
        speed_fit = 0.5  # speed doesn't matter, flatten this dimension

    # Quality: same power curve, gated by quality bar
    quality_need = signal.quality_bar.value / 4.0
    if model.capability >= quality_need:
        exp = 1.0 + quality_need * 3.0
        quality_fit = 0.3 + 0.7 * (model.capability ** exp)
    else:
        gap = quality_need - model.capability
        quality_fit = max(0.0, 0.15 * math.exp(-gap * 8.0))

    # Context fit (noul-style hard gate)
    context_fit = 1.0
    total_tokens = signal.estimated_input_tokens + signal.estimated_output_tokens
    if total_tokens > model.context_tokens:
        context_fit = 0.0

    # Agentic bonus: agents need high capability
    agentic_bonus = 0.0
    if signal.is_agentic:
        if model.capability >= 0.85:
            agentic_bonus = 0.3
        elif model.capability >= 0.75:
            agentic_bonus = 0.1

    return {
        "capability": capability_fit,
        "cost": cost_efficiency,
        "speed": speed_fit,
        "quality": quality_fit,
        "context": context_fit,
        "agentic": agentic_bonus,
    }


def _compute_weights(signal: TaskSignal) -> dict[str, float]:
    """Dimension weights based on task priorities."""
    w = {
        "capability": 0.35,
        "cost": 0.10,
        "speed": 0.10,
        "quality": 0.30,
        "context": 0.05,
        "agentic": 0.10,
    }
    # Shift weights: when user cares about cost/speed, boost those
    cost_shift = signal.cost_priority.value / 4.0
    speed_shift = signal.speed_priority.value / 4.0

    w["cost"] += cost_shift * 0.20
    w["speed"] += speed_shift * 0.15
    # When cost/speed rise, capability/quality absorb the reduction
    w["capability"] -= (cost_shift * 0.10 + speed_shift * 0.05)
    w["quality"] -= (cost_shift * 0.10 + speed_shift * 0.10)

    # Normalise
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def _estimate_cost(model: ClaudeModel, signal: TaskSignal) -> float:
    """Estimate USD cost for the task."""
    input_cost = (signal.estimated_input_tokens / 1_000_000) * model.input_cost
    output_cost = (signal.estimated_output_tokens / 1_000_000) * model.output_cost
    return (input_cost + output_cost) * signal.volume


def _recommend_effort(signal: TaskSignal) -> str:
    """Map task complexity to effort level."""
    mapping = {
        Complexity.TRIVIAL: "low",
        Complexity.SIMPLE: "medium",
        Complexity.MODERATE: "high",
        Complexity.COMPLEX: "xhigh",
        Complexity.FRONTIER: "max",
    }
    return mapping[signal.complexity]


def choose_model(
    signal: TaskSignal,
    candidates: Optional[list[str]] = None,
    temperature: float = 0.15,
) -> Decision:
    """
    JEV-style model selection.

    Args:
        signal: extracted task signals
        candidates: optional list of model IDs to consider (default: all)
        temperature: softmax temperature (lower = more decisive)

    Returns:
        Decision with choice, probabilities, confidence, and reasoning.

    Raises:
        ValueError: if validation fails (JEV validation rules)
    """
    # Filter candidates
    pool = MODELS
    if candidates:
        pool = [m for m in MODELS if m.id in candidates]
        if not pool:
            raise ValueError(f"No valid candidates in {candidates}")

    # Score each model
    weights = _compute_weights(signal)
    all_scores: dict[str, dict[str, float]] = {}
    weighted_totals: dict[str, float] = {}

    for model in pool:
        dim_scores = _score_model(model, signal)
        all_scores[model.id] = dim_scores

        # Context gate: if context_fit is 0, the model is eliminated
        if dim_scores["context"] == 0.0:
            weighted_totals[model.id] = -10.0  # effectively eliminated
            continue

        total = sum(dim_scores[d] * weights[d] for d in weights)

        # JEV hard gate: cost cannot compensate for insufficient capability
        if dim_scores["capability"] < 0.5:
            total *= dim_scores["capability"]

        weighted_totals[model.id] = total

    # JEV choice: softmax over weighted totals
    probabilities = _softmax(weighted_totals, temperature)

    # JEV validation
    choice = max(probabilities, key=probabilities.get)
    assert abs(sum(probabilities.values()) - 1.0) < 0.02, "probabilities must sum to 1"
    assert probabilities[choice] == max(probabilities.values()), "choice must have max P"

    # Confidence: gap between top-1 and top-2
    sorted_probs = sorted(probabilities.values(), reverse=True)
    confidence = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0)
    confidence = min(1.0, max(0.0, confidence))

    # Cost estimate
    chosen_model = MODEL_BY_ID[choice]
    cost = _estimate_cost(chosen_model, signal)

    # Reasoning
    reasoning = (
        f"{chosen_model.name} (P={probabilities[choice]:.2f}, "
        f"confidence={confidence:.2f}) — "
        f"capability={all_scores[choice]['capability']:.2f}, "
        f"cost={all_scores[choice]['cost']:.2f}, "
        f"est. USD {cost:.4f}/req"
    )

    return Decision(
        choice=choice,
        probabilities=probabilities,
        confidence=confidence,
        scores=all_scores,
        cost_estimate=cost,
        reasoning=reasoning,
        effort_recommendation=_recommend_effort(signal),
    )


# --- Signal Extraction from Natural Language ---

COMPLEXITY_KEYWORDS = {
    Complexity.TRIVIAL: ["classify", "label", "yes/no", "true/false", "detect", "filter", "route"],
    Complexity.SIMPLE: ["summarise", "summarize", "translate", "extract", "format", "convert", "rewrite"],
    Complexity.MODERATE: ["generate", "code", "write", "analyse", "analyze", "implement", "review", "debug"],
    Complexity.COMPLEX: ["architect", "design", "research", "plan", "refactor", "migrate", "agent", "multi-step"],
    Complexity.FRONTIER: ["prove", "novel", "breakthrough", "frontier", "hardest", "unsolved", "math proof"],
}

COST_KEYWORDS = {
    CostPriority.CRITICAL: ["cheapest", "free", "minimal cost", "budget", "bulk"],
    CostPriority.HIGH: ["cheap", "cost-effective", "affordable", "save money"],
    CostPriority.LOW: ["best quality", "premium", "no budget limit", "money no object"],
    CostPriority.IGNORE: ["at any cost", "whatever it takes", "spare no expense"],
}

SPEED_KEYWORDS = {
    SpeedPriority.CRITICAL: ["instant", "autocomplete", "real-time", "sub-second", "streaming"],
    SpeedPriority.HIGH: ["fast", "quick", "responsive", "chat", "interactive"],
    SpeedPriority.LOW: ["background", "async", "overnight", "batch"],
    SpeedPriority.NONE: ["no rush", "whenever", "queue"],
}


def extract_signal(task_description: str) -> TaskSignal:
    """Extract TaskSignal from natural language task description."""
    desc = task_description.lower()
    signal = TaskSignal()

    # Complexity detection
    for level, keywords in COMPLEXITY_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            signal.complexity = level

    # Cost priority
    for level, keywords in COST_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            signal.cost_priority = level

    # Speed priority
    for level, keywords in SPEED_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            signal.speed_priority = level

    # Quality bar heuristics
    if any(kw in desc for kw in ["critical", "production", "zero error", "safety"]):
        signal.quality_bar = QualityBar.CRITICAL
    elif any(kw in desc for kw in ["high quality", "expert", "thorough", "prove", "proof", "frontier"]):
        signal.quality_bar = QualityBar.HIGH
    elif any(kw in desc for kw in ["draft", "rough", "quick and dirty", "prototype"]):
        signal.quality_bar = QualityBar.DRAFT

    # Auto-escalate quality for complex/frontier tasks
    if signal.complexity.value >= Complexity.COMPLEX.value:
        signal.quality_bar = max(signal.quality_bar, QualityBar.HIGH, key=lambda x: x.value)

    # Feature flags
    signal.is_agentic = any(kw in desc for kw in ["agent", "autonomous", "multi-step", "tool use"])
    signal.needs_tools = any(kw in desc for kw in ["tool", "function call", "api", "search"])
    signal.needs_vision = any(kw in desc for kw in ["image", "screenshot", "vision", "picture", "photo"])
    signal.is_batch = any(kw in desc for kw in ["batch", "bulk", "1000", "10000", "mass"])

    # Token estimates from context clues
    if any(kw in desc for kw in ["long document", "book", "large file", "100k", "codebase"]):
        signal.estimated_input_tokens = 100_000
    elif any(kw in desc for kw in ["article", "page", "file"]):
        signal.estimated_input_tokens = 5_000

    if any(kw in desc for kw in ["long output", "full code", "detailed report"]):
        signal.estimated_output_tokens = 10_000

    if signal.is_batch:
        signal.volume = 100  # default batch assumption

    return signal


# --- Demo ---

def demo():
    tasks = [
        "Classify these 10,000 support tickets into categories. Cheapest option, batch processing.",
        "Write a production-ready REST API in Python with authentication, rate limiting, and tests.",
        "Real-time chat autocomplete for a mobile app. Must be instant and cheap.",
        "Research and architect a distributed system for processing 1M events/second.",
        "Summarise this article in 3 bullet points.",
        "Build an autonomous agent that browses the web, extracts data, and writes a report.",
        "Prove this mathematical theorem about prime distribution in number fields.",
    ]

    for task in tasks:
        signal = extract_signal(task)
        decision = choose_model(signal)

        print(f"\n{'='*70}")
        print(f"TASK: {task[:65]}...")
        print(f"  Signal: complexity={signal.complexity.name}, "
              f"cost={signal.cost_priority.name}, speed={signal.speed_priority.name}")
        print(f"  → {decision.reasoning}")
        print(f"  Effort: {decision.effort_recommendation}")
        print(f"  Top 3:")
        sorted_p = sorted(decision.probabilities.items(), key=lambda x: -x[1])
        for model_id, prob in sorted_p[:3]:
            m = MODEL_BY_ID[model_id]
            print(f"    {m.name:12s}  P={prob:.3f}  "
                  f"(${m.input_cost}/{m.output_cost} per MTok)")


if __name__ == "__main__":
    demo()
