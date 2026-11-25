import argparse
import json

from .router import (
    MODEL_BY_ID,
    Complexity,
    CostPriority,
    QualityBar,
    SpeedPriority,
    choose_model,
    extract_signal,
)


def main(argv=None):
    p = argparse.ArgumentParser(prog="claude-model-jev", description="Pick a Claude model for a task. Flags override signals extracted from the text.")
    p.add_argument("text", nargs="?", default="", help="task description (English keyword extraction)")
    p.add_argument("--complexity", choices=[e.name for e in Complexity])
    p.add_argument("--quality", choices=[e.name for e in QualityBar])
    p.add_argument("--cost", choices=[e.name for e in CostPriority])
    p.add_argument("--speed", choices=[e.name for e in SpeedPriority])
    p.add_argument("--agentic", action="store_true")
    p.add_argument("--candidates", help="comma-separated model IDs")
    a = p.parse_args(argv)

    s = extract_signal(a.text)
    if a.complexity:
        s.complexity = Complexity[a.complexity]
    if a.quality:
        s.quality_bar = QualityBar[a.quality]
    if a.cost:
        s.cost_priority = CostPriority[a.cost]
    if a.speed:
        s.speed_priority = SpeedPriority[a.speed]
    if a.agentic:
        s.is_agentic = True

    d = choose_model(s, a.candidates.split(",") if a.candidates else None)
    m = MODEL_BY_ID[d.choice]
    top = sorted(d.probabilities.items(), key=lambda kv: -kv[1])[:4]
    print(json.dumps({
        "model_id": m.id,
        "tier": m.tier,
        "name": m.name,
        "effort": d.effort_recommendation,
        "confidence": round(d.confidence, 3),
        "cost_estimate_usd": round(d.cost_estimate, 6),
        "signal": {
            "complexity": s.complexity.name,
            "quality": s.quality_bar.name,
            "cost": s.cost_priority.name,
            "speed": s.speed_priority.name,
            "agentic": s.is_agentic,
        },
        "top": [{"name": MODEL_BY_ID[k].name, "p": round(v, 3)} for k, v in top],
    }, indent=2))


if __name__ == "__main__":
    main()
