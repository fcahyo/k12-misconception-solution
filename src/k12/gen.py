"""Generative label scoring for Qwen3.5-family models: full distribution over all label strings.

P(label) = prod_t P(tok_t | prompt, tok_<t). Prompt is run once (batch 1, use_cache), the cache is
expanded to one row per label with reorder_cache, and all label continuations are scored in one batch.
"""
from __future__ import annotations

import torch


def label_token_ids(tokenizer, labels: list[str]) -> list[list[int]]:
    """Token ids of each label as the start of the assistant response (no leading space).
    The label set is prefix-free at token level (checked), so no end token is needed."""
    out = [tokenizer(c, add_special_tokens=False).input_ids for c in labels]
    for i, a in enumerate(out):
        for j, b in enumerate(out):
            if i != j and len(a) <= len(b) and b[: len(a)] == a:
                raise ValueError(f"label {labels[i]} is a token prefix of {labels[j]}")
    return out


def _inner(model):
    """Return the Qwen3_5ForConditionalGeneration under optional PEFT wrapping."""
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m


@torch.no_grad()
def score_labels(model, enc: dict, label_ids: list[list[int]], device, chunk: int = 12) -> torch.Tensor:
    """enc: processor output for ONE prompt (ends with generation prompt). Returns log-probs (n_labels,).

    Labels are grouped by their all-but-last-token prefix; one continuation row per distinct prefix
    gives the exact probability of every label (last-token distribution read at the row's final position)."""
    m = _inner(model)
    kw = {k: v.to(device) for k, v in enc.items() if k in
          ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")}
    m.model.rope_deltas = None  # avoid stale M-RoPE deltas from a previous (image) prompt
    out = m(**kw, use_cache=True, logits_to_keep=1)
    lp0 = torch.log_softmax(out.logits[0, -1].float(), -1)
    scores = torch.empty(len(label_ids), device=device)
    groups: dict[tuple, list[int]] = {}
    for i, ids in enumerate(label_ids):
        if len(ids) == 1:
            scores[i] = lp0[ids[0]]
        else:
            groups.setdefault(tuple(ids[:-1]), []).append(i)
    if not groups:
        return scores
    prefixes = list(groups)
    Lmax = max(len(p) for p in prefixes)
    base_cache = out.past_key_values
    import copy

    for c0 in range(0, len(prefixes), chunk):
        part = prefixes[c0:c0 + chunk]
        G = len(part)
        cache = copy.deepcopy(base_cache) if c0 + chunk < len(prefixes) else base_cache
        cache.reorder_cache(torch.zeros(G, dtype=torch.long, device=device))
        cont = torch.zeros((G, Lmax), dtype=torch.long, device=device)
        for g, pfx in enumerate(part):
            cont[g, : len(pfx)] = torch.tensor(pfx, device=device)
        h = m.model(input_ids=cont, past_key_values=cache, use_cache=False).last_hidden_state  # (G, Lmax, H)
        for g, pfx in enumerate(part):
            n = len(pfx)
            lp = torch.log_softmax(m.lm_head(h[g, :n]).float(), -1)  # (n, V)
            base = lp0[pfx[0]]
            for t in range(1, n):
                base = base + lp[t - 1, pfx[t]]
            for i in groups[pfx]:
                scores[i] = base + lp[n - 1, label_ids[i][-1]]
        del cache, h
    return scores
