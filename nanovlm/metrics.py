"""ROUGE-1 (unigram overlap), used as a memorization/diversity check only —
NOT a quality signal (paper §4; EXPERIMENT_GUIDE_encoder_ambiguity.md §6
warns ROUGE-1 trended opposite to judge quality as description length grew
in an earlier project on this line of work). Implemented directly to avoid
an extra heavyweight dependency for one metric.
"""

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def rouge1_f1(candidate: str, reference: str) -> float:
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    cand_counts = Counter(cand_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(cand_counts[w], ref_counts[w]) for w in cand_counts)

    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
