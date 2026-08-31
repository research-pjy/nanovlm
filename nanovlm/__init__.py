"""nanovlm: resolving the NanoVLMs paper's visual encoder ambiguity.

See DESIGN_DECISIONS.md at the repo root for every filled-in gap.
"""

from nanovlm.config import MODEL_SIZES, FIXED_HPARAMS, NanoVLMConfig
from nanovlm.model.nanovlm import NanoVLM

__all__ = ["MODEL_SIZES", "FIXED_HPARAMS", "NanoVLMConfig", "NanoVLM"]
