#!/usr/bin/env python3
"""Curation pass 4 — hunt for MORE distinct lessons (each must STEER, not just label cleanly).
Neuronpedia finds the labeled feature; we steer it locally forbid(−220)/force(+240,+360) on a
theme-appropriate prompt and print, to keep the ones that genuinely change the output."""
import os, json, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_FILES = [ROOT / ".secrets.env", ROOT / ".env", ROOT.parent / ".secrets.env"]
LAYER, SOURCE = 20, "20-gemmascope-res-16k"

NEUTRAL = "My favorite thing to do on a slow Sunday morning is to sit by the window with a warm drink and"
# theme, label, search query, label-keywords, prompt, strengths
CAND = [
    ("register", "the Bible / scripture", "the Bible, scripture, biblical verses, gospel and the Lord",
     "bible|scriptur|verse|gospel|lord|psalm|christ", NEUTRAL, [-220, 240, 360]),
    ("register", "Islam / the Quran", "Islam, the Quran, Muslim prayer, Allah and the mosque",
     "islam|muslim|quran|allah|mosque|prophet", NEUTRAL, [-220, 240, 360]),
    ("style", "ALL CAPS / shouting", "text written in all capital letters, shouting, loud emphasis",
     "capital|uppercase|shout|caps|emphas|loud", NEUTRAL, [240, 360, 460]),
    ("style", "questions", "asking questions, interrogative sentences that end with a question mark",
     "question|interrog|asking|inquir", NEUTRAL, [240, 360, 460]),
    ("style", "rhyming poetry", "rhyming poetry, verse, poems and stanzas",
     "poe|rhym|verse|stanza|rhyme", NEUTRAL, [240, 360, 460]),
    ("behavior", "flattery / sycophancy", "flattery, excessive praise, gushing compliments and sycophancy",
     "flatter|praise|complim|sycoph|gush|admir", "When a reader asks me what I honestly think of their very first novel, I tell them", [-240, 240, 360]),
    ("behavior", "uncertainty / hedging", "expressing uncertainty, hedging with maybe, perhaps and possibly",
     "uncertain|hedg|maybe|perhaps|possibly|might", NEUTRAL, [240, 360]),
    ("sentiment", "negativity / criticism", "negative sentiment, harsh criticism, complaints and disapproval",
     "negativ|critic|complain|disapprov|harsh", NEUTRAL, [240, 360]),
    ("emotion", "fear / dread", "fear, dread, anxiety, terror and being scared",
     "fear|dread|anxi|terror|scared|afraid", "As I walked alone down the dark, empty hallway at midnight, the only thing I could feel was", [-240, -160, 240, 360]),
]


def secret(name):
    v = os.environ.get(name)
    if v: return v
    for p in SECRET_FILES:
        if p.exists():
            for line in open(p):
                s = line.strip()
                if not s.startswith("#") and name in s and "=" in s:
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def find(query, keywords, key):
    body = json.dumps({"modelId": "gemma-2-2b", "sourceSet": "gemmascope-res-16k", "text": query,
                       "selectedLayers": [SOURCE], "sortIndexes": [], "ignoreBos": True, "numResults": 12}).encode()
    req = urllib.request.Request("https://www.neuronpedia.org/api/search-all", data=body,
                                 headers={"x-api-key": key, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"   search HTTP {e.code}"); return None
    kws = [k.strip() for k in keywords.split("|")]
    fs = [{"index": int(r["index"]), "max": float(r.get("maxValue", 0)),
           "label": (((r.get("neuron") or {}).get("explanations") or [{}])[0].get("description")) or ""} for r in d.get("result", [])]
    m = [f for f in fs if any(k in f["label"].lower() for k in kws)]
    return max(m, key=lambda f: f["max"]) if m else (fs[0] if fs else None)


def main():
    import torch
    from huggingface_hub import login
    login(token=secret("HF_TOKEN"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE
    key = secret("NEURONPEDIA_API_KEY")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("loading gemma-2-2b + SAE…", flush=True)
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", f"layer_{LAYER}/width_16k/canonical", device=device)
    if isinstance(sae, tuple): sae = sae[0]
    Wdec = sae.W_dec.to(device).to(torch.bfloat16)
    steer = {"vec": None}
    model.model.layers[LAYER].register_forward_hook(
        lambda m, i, o: (((o[0] + steer["vec"],) + o[1:]) if isinstance(o, tuple) else (o + steer["vec"])) if steer["vec"] is not None else o)

    @torch.no_grad()
    def gen(prompt, idx, strength, n=52):
        torch.manual_seed(16)
        steer["vec"] = strength * Wdec[idx] if strength else None
        enc = tok(prompt, return_tensors="pt").to(device)
        ids = model.generate(**enc, max_new_tokens=n, do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
        steer["vec"] = None
        return tok.decode(ids[0], skip_special_tokens=True)[len(prompt):].strip().replace("\n", " ")

    print("\n=== curation 4: hunting more steerable lessons ===", flush=True)
    for theme, label, q, kw, prompt, strs in CAND:
        f = find(q, kw, key)
        if not f:
            print(f"\n• [{theme}] {label}: no feature"); continue
        print(f"\n• [{theme}] {label}  feat {f['index']:>6} :: {f['label'][:46]}", flush=True)
        print(f"    base : {gen(prompt, f['index'], 0)[:110]!r}", flush=True)
        for s in strs:
            tag = "FORBID" if s < 0 else "FORCE "
            print(f"    {tag}{s:>5}: {gen(prompt, f['index'], s)[:110]!r}", flush=True)


if __name__ == "__main__":
    main()
