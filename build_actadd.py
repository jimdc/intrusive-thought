#!/usr/bin/env python3
"""build_actadd.py — precompute ActAdd steering data for the "Intrusive Thought" demo's
ActAdd contrast-pair bank (data/steer-actadd.json).

Implements activation-addition steering (ActAdd; Turner et al. 2023,
arXiv:2308.10248): a steering vector is the difference between a model's
residual-stream activations on a contrast pair of prompts, taken at a
middle transformer layer. Adding `coefficient * vector` to that layer's
residual stream during generation nudges the model's output toward (or,
for negative coefficients, away from) the concept the pair encodes. No
weights are touched — the model is exactly itself again at coefficient 0.

Usage:
    python build_actadd.py                 # full run: loads google/gemma-2-2b,
                                             # writes data/steer-actadd.json
    python build_actadd.py --selftest      # vector-math self-test only.
                                             # No model, no download. CI gate.

The full run expects google/gemma-2-2b to already be present in the local
Hugging Face cache. Run with HF_HUB_OFFLINE=1 so the loader reads the cache
directly, without a network round-trip or an HF token:

    HF_HUB_OFFLINE=1 python build_actadd.py

Dependencies (full run only): torch, transformers. --selftest needs only
torch — no transformers import, no model weights, no network access.
"""

import argparse
import datetime
import json
import os
import sys

# One residual-stream direction per concept, built from a short contrast
# pair. `prompt` is the sentence completions are generated from; it is
# deliberately unrelated to the contrast pair itself, so any lean in the
# output is attributable to the steering vector, not the prompt content.
CONCEPTS = [
    {
        "id": "warmth",
        "label": "Warmth",
        "polePositive": "Love",
        "poleNegative": "Hate",
        "contrastPositive": "Love. Kindness. Warmth. Affection and tenderness.",
        "contrastNegative": "Hate. Cruelty. Coldness. Contempt and indifference.",
        "prompt": "My neighbor knocked on my door and asked to borrow a cup of sugar. I",
        "lesson": "Drag back to coefficient 0 and the completion is <b>byte-for-byte</b> the model's unsteered output — not similar, identical — because the vector was added to an activation, never written to a weight. That's the whole demo: nothing is left behind when you let go.",
        "paper": "Turner et al., Activation Addition (2023)",
        "paperUrl": "https://arxiv.org/abs/2308.10248",
    },
    {
        "id": "formality",
        "label": "Formality",
        "polePositive": "Formal",
        "poleNegative": "Casual",
        "contrastPositive": "Dear Sir or Madam, I am writing to formally request your consideration of the following matter.",
        "contrastNegative": "hey!! so basically I just wanted to ask you something real quick, no big deal lol",
        "prompt": "Let me tell you about my weekend.",
        "lesson": "There's no autoencoder here, no dictionary of 16,000 learned features to search — the direction is just the difference between two sentences someone wrote, one formal and one casual. Change the pair and you change the concept: <b>contrast-pair steering trades an SAE's precision for do-it-yourself reach</b>.",
        "paper": "Turner et al., Activation Addition (2023)",
        "paperUrl": "https://arxiv.org/abs/2308.10248",
    },
    {
        "id": "playfulness",
        "label": "Playfulness",
        "polePositive": "Playful",
        "poleNegative": "Serious",
        "contrastPositive": "Whee! Tag, you're it! Let's giggle and play and have silly fun!",
        "contrastNegative": "This is a serious, solemn matter requiring careful, formal analysis.",
        "prompt": "The meeting began, and the manager said,",
        "lesson": "The push is <b>pure vector arithmetic</b>: the same direction is just added at a positive or negative coefficient, so +6 and −6 displace the residual stream by exactly equal and opposite amounts — no learned asymmetry the way an SAE feature's force/forbid sides have.",
        "paper": "Turner et al., Activation Addition (2023)",
        "paperUrl": "https://arxiv.org/abs/2308.10248",
    },
]

# Both signs, through zero. Symmetric and odd-length, so the middle entry
# is always the unsteered (coefficient 0) baseline. Scaled empirically: at
# layer 13 the mean-pooled contrast direction has norm ~47 against a residual
# stream norm of ~293, so coefficient 6 already displaces ~96% of the
# residual's own magnitude (collapse) and 12 is ~2x it (pure noise). 2-4
# keeps displacement in the 15-65% range where the lean is visible without
# already having overwhelmed the sentence.
COEFFICIENT_LADDER = [-4, -2, 0, 2, 4]

DEFAULT_MODEL = "google/gemma-2-2b"
DEFAULT_LAYER = 13  # gemma-2-2b has 26 layers; 13 is the midpoint.
DEFAULT_MAX_NEW_TOKENS = 40


# ---------------------------------------------------------------------------
# Core ActAdd math. No torch-specific behavior beyond relying on tensors
# supporting +, -, and scalar * — kept isolated so --selftest can exercise
# it without importing transformers or touching any model weights.
# ---------------------------------------------------------------------------

def build_direction(positive_activation, negative_activation):
    """The steering vector: difference of the two contrast activations."""
    return positive_activation - negative_activation


def add_steering_vector(hidden_state, direction, coefficient):
    """Add `coefficient * direction` to a residual-stream hidden state.

    coefficient=0 must be exactly the identity — that's what makes the
    demo's "back to 0 = back to normal" claim true rather than aspirational.
    """
    if coefficient == 0:
        return hidden_state
    return hidden_state + coefficient * direction


def run_selftest():
    import torch

    torch.manual_seed(0)
    hidden = torch.randn(5, 16)  # (seq_len, hidden_dim), stand-in residual stream
    positive = torch.randn(16)
    negative = torch.randn(16)

    direction = build_direction(positive, negative)
    assert torch.allclose(direction, positive - negative), (
        "direction must equal positive activation minus negative activation"
    )

    steered_zero = add_steering_vector(hidden, direction, 0.0)
    assert torch.equal(steered_zero, hidden), "coefficient 0 must be the identity"

    steered_pos = add_steering_vector(hidden, direction, 3.0)
    steered_neg = add_steering_vector(hidden, direction, -3.0)
    assert torch.allclose(steered_pos - hidden, 3.0 * direction), (
        "positive coefficient must add coefficient * direction"
    )
    assert torch.allclose(steered_neg - hidden, -3.0 * direction), (
        "negative coefficient must subtract |coefficient| * direction"
    )
    assert torch.allclose(steered_pos - hidden, -(steered_neg - hidden)), (
        "+c and -c must displace the hidden state by equal and opposite amounts"
    )

    steered_double = add_steering_vector(hidden, direction, 6.0)
    assert torch.allclose(steered_double - hidden, 2 * (steered_pos - hidden)), (
        "steering must scale linearly in the coefficient"
    )

    print(
        "selftest passed: direction = positive - negative; "
        "coefficient 0 is identity; steering is linear and sign-symmetric"
    )
    return 0


# ---------------------------------------------------------------------------
# Model-backed pipeline. Only imported/used on a full run.
# ---------------------------------------------------------------------------

class SteeringHook:
    """A forward hook that adds a fixed steering vector to a decoder
    layer's output residual stream for the duration it is registered."""

    def __init__(self, direction, coefficient):
        self.direction = direction
        self.coefficient = coefficient

    def __call__(self, module, inputs, output):
        if self.coefficient == 0:
            return output
        if isinstance(output, tuple):
            hidden = add_steering_vector(output[0], self.direction, self.coefficient)
            return (hidden,) + output[1:]
        return add_steering_vector(output, self.direction, self.coefficient)


def get_residual_activation(model, tokenizer, text, layer, device):
    """Mean-pooled residual-stream activation for `text` at `layer`.

    hidden_states[0] is the embedding output, so hidden_states[layer] is
    the residual stream after `layer` transformer blocks have run.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer][0]  # (seq_len, hidden_dim)
    return hidden.mean(dim=0)


def build_steering_vector(model, tokenizer, positive_text, negative_text, layer, device):
    positive = get_residual_activation(model, tokenizer, positive_text, layer, device)
    negative = get_residual_activation(model, tokenizer, negative_text, layer, device)
    return build_direction(positive, negative)


def generate_completion(model, tokenizer, prompt, direction, coefficient, layer, device, max_new_tokens):
    import torch

    # hidden_states[layer] is the output of decoder block (layer - 1);
    # hook that block so its output is exactly the residual stream we steer.
    hook_module = model.model.layers[layer - 1]
    hook = SteeringHook(direction, coefficient)
    handle = hook_module.register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    finally:
        handle.remove()


def run_full_pipeline(model_name, layer, max_new_tokens, out_path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"loading {model_name} (offline cache, device={device})...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    concepts_out = []
    for concept in CONCEPTS:
        print(f"steering concept: {concept['id']}", file=sys.stderr)
        direction = build_steering_vector(
            model, tokenizer,
            concept["contrastPositive"], concept["contrastNegative"],
            layer, device,
        )
        ladder = []
        for coefficient in COEFFICIENT_LADDER:
            completion = generate_completion(
                model, tokenizer, concept["prompt"], direction, coefficient,
                layer, device, max_new_tokens,
            )
            ladder.append({"coefficient": coefficient, "completion": completion})
            print(f"  coefficient={coefficient:+d}: {completion[:80]!r}", file=sys.stderr)
        concepts_out.append({
            "id": concept["id"],
            "label": concept["label"],
            "polePositive": concept["polePositive"],
            "poleNegative": concept["poleNegative"],
            "contrastPositive": concept["contrastPositive"],
            "contrastNegative": concept["contrastNegative"],
            "prompt": concept["prompt"],
            "lesson": concept["lesson"],
            "paper": concept["paper"],
            "paperUrl": concept["paperUrl"],
            "zeroIndex": COEFFICIENT_LADDER.index(0),
            "ladder": ladder,
        })

    data = {
        "model": model_name,
        "method": "ActAdd (activation addition)",
        "citation": "Turner et al., 2023, arXiv:2308.10248",
        "layer": layer,
        "built": datetime.date.today().isoformat(),
        "sample": False,
        "concepts": concepts_out,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="run the vector-math self-test and exit (no model)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help=f"residual layer to steer (default: {DEFAULT_LAYER})")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--out", default=os.path.join("data", "steer-actadd.json"))
    args = parser.parse_args()

    if args.selftest:
        return run_selftest()
    return run_full_pipeline(args.model, args.layer, args.max_new_tokens, args.out)


if __name__ == "__main__":
    sys.exit(main())
