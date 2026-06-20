#!/usr/bin/env python3
"""
Bidirectional steering for "Intrusive Thought" — ONE slider per concept, on a topic-forcing prompt:
  left  (negative) : FORBID the concept -> the model contorts to avoid naming what it's clearly thinking
  center (0)       : the thought itself (the prompt's natural completion)
  right (positive) : FORCE the concept  -> it can't stop / floods / breaks
Local (gemma-2-2b + Gemma Scope), no API. Writes data/steers.json in the {tours:[...]} shape the page reads.
"""
import os, sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_FILES = [ROOT / ".secrets.env", ROOT / ".env", ROOT.parent / ".secrets.env"]
LAYER, SOURCE = 20, "20-gemmascope-res-16k"
STR = [-420, -260, -190, 0, 190, 260, 420]          # one slider, suppress (−) … force (+)

# Each concept teaches ONE real, paper-grounded lesson about LLM internals (see _research/).
# id · chip label · feature index · feature label · topic-FORCING prompt · lesson · paper · url
CONCEPTS = [
    dict(id="golden_gate", label="Golden Gate Bridge", idx=1566, flabel="the Golden Gate landmark in San Francisco",
         prompt="The famous orange-red suspension bridge spanning the bay in San Francisco is the",
         lesson="One feature, turned up, can hijack the whole personality. Anthropic clamped this exact kind of feature high and the model began to insist it <i>was</i> the Golden Gate Bridge.",
         paper="Anthropic, Golden Gate Claude (2024)",
         url="https://www.anthropic.com/news/golden-gate-claude"),
    dict(id="drugs", label="drugs", idx=12140, flabel="illegal drugs and narcotics",
         prompt="The illegal narcotic substances people get hooked on and ruin their lives for are called",
         lesson="“Drugs” isn’t one idea in there. Force it hard and the single feature shatters into its finer pieces — cocaine, heroin, meth — a glimpse of <b>feature splitting</b>.",
         paper="Bricken et al., Towards Monosemanticity (2023)",
         url="https://transformer-circuits.pub/2023/monosemantic-features/"),
    dict(id="death", label="death", idx=9232, flabel="death and dying",
         prompt="The one thing that happens to every living creature at the very end of its life is",
         lesson="Forbid a thought and the model doesn’t fall silent — it reaches for a <b>euphemism</b>. The meaning is stored many ways; you blocked one route to the word, so it takes the next.",
         paper="Arditi et al., Refusal is mediated by a single direction (2024)",
         url="https://arxiv.org/abs/2406.11717"),
    dict(id="spanish", label="Spanish", idx=8590, flabel="the Spanish language",
         prompt="My favorite thing to do on a slow Sunday morning is to sit by the window with a warm drink and",
         lesson="This knob isn’t a word — it’s a whole <b>language</b>. Turn it up and the model drifts into fluent Spanish no matter what you asked it, because the concept lives above any one tongue.",
         paper="Templeton et al., Scaling Monosemanticity (2024)",
         url="https://transformer-circuits.pub/2024/scaling-monosemanticity/"),
    dict(id="scripture", label="scripture", idx=2655, flabel="faith and salvation through Jesus",
         prompt="After thinking about it for years, I’ve come to believe that the real secret to a good and meaningful life is",
         lesson="Force a feature and the model’s whole <b>voice</b> can change genre — turn this one up and ordinary chatter starts to <b>preach</b>, flooding with scripture and the name of the Lord. Less a topic than a whole register.",
         paper="Templeton et al., Scaling Monosemanticity (2024)",
         url="https://transformer-circuits.pub/2024/scaling-monosemanticity/"),
    dict(id="gratitude", label="gratitude", idx=12860, flabel="gratitude and good fortune",
         prompt="Looking back on everything that has happened to me this year, I have to say that",
         lesson="Not every knob is a thing — some are a <b>mood</b>. Force this one and the model can’t stop counting its blessings (“fortunately… luckily… thankfully…”) — proof that even a vibe is a real, movable direction inside.",
         paper="Park, Choe & Veitch, The Linear Representation Hypothesis (2024)",
         url="https://arxiv.org/abs/2311.03658"),
    dict(id="sycophancy", label="flattery", idx=7093, flabel="admiration, praise and flattery",
         prompt="When a reader asks me what I honestly think of their very first novel, I tell them",
         lesson="Not every knob is a <b>topic</b> — some are <b>behaviors</b>. Turn up flattery and the model gushes that their first novel is a masterpiece, whatever it really is: the <b>sycophancy</b> AI-safety folks worry about, laid bare as one dial.",
         paper="Sharma et al., Towards Understanding Sycophancy in LMs (2023)",
         url="https://arxiv.org/abs/2310.13548"),
]


def secret(name):
    v = os.environ.get(name)
    if v:
        return v
    for p in SECRET_FILES:
        if p.exists():
            for line in open(p):
                s = line.strip()
                if not s.startswith("#") and name in s and "=" in s:
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main():
    import torch
    from huggingface_hub import login
    login(token=secret("HF_TOKEN"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("loading gemma-2-2b + SAE…", flush=True)
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{LAYER}/width_16k/canonical", device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    Wdec = sae.W_dec.to(device).to(torch.bfloat16)

    steer = {"vec": None}
    model.model.layers[LAYER].register_forward_hook(
        lambda m, i, o: (((o[0] + steer["vec"],) + o[1:]) if isinstance(o, tuple) else (o + steer["vec"])) if steer["vec"] is not None else o)

    @torch.no_grad()
    def gen(prompt, idx=None, strength=0, n=72):
        torch.manual_seed(16)
        steer["vec"] = strength * Wdec[idx] if (idx is not None and strength) else None
        enc = tok(prompt, return_tensors="pt").to(device)
        ids = model.generate(**enc, max_new_tokens=n, do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
        steer["vec"] = None
        return tok.decode(ids[0], skip_special_tokens=True)

    # optional CLI ids => rebuild only those concepts, MERGING into existing data/steers.json
    only = set(sys.argv[1:])
    path = ROOT / "data/steers.json"
    existing = {t["id"]: t for t in json.load(open(path)).get("tours", [])} if (only and path.exists()) else {}

    built, t0, tours = {}, time.time(), []
    for c in CONCEPTS:
        if only and c["id"] not in only:
            continue
        cid, idx, prompt = c["id"], c["idx"], c["prompt"]
        default = gen(prompt)
        steps = [{"strength": s, "steered": (default if s == 0 else gen(prompt, idx, s))} for s in STR]
        built[cid] = {"id": cid, "label": c["label"], "prompt": prompt, "zeroAt": STR.index(0),
                      "feature": {"layer": SOURCE, "index": idx, "label": c["flabel"]},
                      "lesson": c["lesson"], "paper": c["paper"], "paperUrl": c["url"],
                      "default": default, "steps": steps}
        print(f"· {cid}:", flush=True)
        for st in steps:
            print(f"    {st['strength']:>5}: {st['steered'][len(prompt):][:62].strip()!r}", flush=True)
        tours = [t for t in (built.get(cc["id"], existing.get(cc["id"])) for cc in CONCEPTS) if t is not None]
        json.dump({"model": "gemma-2-2b", "source": "local: Gemma Scope 2b res L20 16k",
                   "built": time.strftime("%Y-%m-%d"), "bidirectional": True, "tours": tours},
                  open(path, "w"), indent=1)
    print(f"\nwrote data/steers.json — {len(tours)} concepts ({len(built)} rebuilt) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
