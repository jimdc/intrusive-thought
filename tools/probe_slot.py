#!/usr/bin/env python3
"""Fast probe for two slots that failed/were-weak in the build: the multilingual slot (does any language
feature cleanly flip output to that language?) and Python (needs a NON-code prompt so forcing shows code).
Loads the model once, prints a few strengths per (label, idx, prompt). No JSON written."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SECRET_FILES = [ROOT / ".secrets.env", ROOT / ".env", ROOT.parent / ".secrets.env"]
LAYER = 20

NEUTRAL = "My favorite thing to do on a slow Sunday morning is to sit by the window with a warm drink and"
STORY = "Let me tell you a little story about something that happened to me yesterday. I was walking down the street when"
# (slot, label, idx, prompt, strengths)
PROBES = [
    ("lang", "French",  12057, NEUTRAL, [0, 220, 320, 440]),
    ("lang", "Spanish", 8590,  NEUTRAL, [0, 220, 320, 440]),
    ("lang", "Spanish", 8590,  STORY,   [320, 440]),
    ("lang", "German",  9279,  NEUTRAL, [0, 320, 440]),
    ("lang", "Japanese",4174,  NEUTRAL, [320, 440]),
    ("py",   "Python",  8236,  "Let me tell you about my plans for this weekend. On Saturday morning, the first thing I am going to do is", [-260, -190, 0, 190, 260, 360]),
    ("py",   "Python",  8236,  "The recipe my grandmother taught me for the perfect cup of tea goes like this. First, you", [0, 190, 260, 360]),
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
    if isinstance(sae, tuple): sae = sae[0]
    Wdec = sae.W_dec.to(device).to(torch.bfloat16)
    steer = {"vec": None}
    model.model.layers[LAYER].register_forward_hook(
        lambda m, i, o: (((o[0] + steer["vec"],) + o[1:]) if isinstance(o, tuple) else (o + steer["vec"])) if steer["vec"] is not None else o)

    @torch.no_grad()
    def gen(prompt, idx, strength, n=60):
        torch.manual_seed(16)
        steer["vec"] = strength * Wdec[idx] if strength else None
        enc = tok(prompt, return_tensors="pt").to(device)
        ids = model.generate(**enc, max_new_tokens=n, do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
        steer["vec"] = None
        return tok.decode(ids[0], skip_special_tokens=True)[len(prompt):].strip().replace("\n", " ")

    last = None
    for slot, label, idx, prompt, strs in PROBES:
        if prompt != last:
            print(f"\n### prompt: …{prompt[-60:]!r}"); last = prompt
        print(f"  {label} ({idx}):", flush=True)
        for s in strs:
            print(f"     {s:>5}: {gen(prompt, idx, s)[:120]!r}", flush=True)


if __name__ == "__main__":
    main()
