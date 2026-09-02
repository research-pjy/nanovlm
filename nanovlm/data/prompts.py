"""Prompt templates for the teacher (caption generation) and grader
(evaluation) LLM calls.

The paper's own prompts (Figure 2, Prompt1/Prompt2/Prompt3) are given only
as an image in the PDF, not as extractable text, and this project's teacher
model is llama3:8b, not the paper's GPT-4o (DGX_GUIDE_nanovlm.md §3) — so
these are written fresh to match the paper's *described* intent (simple,
child-like language; a 5-dimension judge rubric scored out of 10) rather
than transcribed. Both prompts are used identically for every image /
every model output — nothing here varies by encoder strategy or size.
"""

from nanovlm.config import FIXED_HPARAMS

_WORD_LO, _WORD_HI = FIXED_HPARAMS["caption_word_range"]
_EVAL_LO, _EVAL_HI = FIXED_HPARAMS["eval_prompt_word_range"]

RUBRIC_DIMENSIONS = ["grammar", "creativity", "consistency", "meaningfulness", "plot"]


def teacher_caption_prompt(all_captions: list[str]) -> str:
    """Prompt for llama3:8b to produce one LongDesc-style (60-70 word)
    child-simple description from an image's combined COCO captions.
    Combining all five captions per image mirrors the paper's own dataset
    prep (Figure 3) — see DESIGN_DECISIONS.md §1 for the single-format
    (LongDesc-only) decision.
    """
    captions_block = "\n".join(f"- {c.strip()}" for c in all_captions)
    return (
        "You are writing a picture description for a 3-4 year old child's "
        "storybook. Below are five human-written captions describing the "
        "same photo. Using ONLY the objects, people, animals, and actions "
        "they mention, write ONE description of the photo.\n\n"
        f"Reference captions:\n{captions_block}\n\n"
        "Rules:\n"
        f"- Length: {_WORD_LO}-{_WORD_HI} words, no more, no less.\n"
        "- Use simple words and short sentences a 3-4 year old could "
        "understand and say themselves.\n"
        "- Be concrete and visual: describe what is happening in the "
        "picture, not abstract ideas.\n"
        "- Do not mention that this is a photo, a caption, or an "
        "instruction. Just write the description itself.\n"
        "- Do not add anything not supported by the reference captions.\n\n"
        "Description:"
    )


def eval_partial_text_prompt(full_caption: str) -> str:
    """Truncates a generated caption to an 18-20 word partial-text prompt
    for the held-out completion task (paper §3; DESIGN_DECISIONS.md §1/§6
    — long-format length, consistent with LongDesc-style training data).
    """
    words = full_caption.strip().split()
    n = min(_EVAL_HI, max(_EVAL_LO, len(words) // 2))
    return " ".join(words[:n])


def grading_prompt(image_context: str, partial_prompt: str, completion: str) -> str:
    """Prompt for the llama3:8b grader (DGX_GUIDE_nanovlm.md §3) to score a
    model's text completion on the paper's 5-dimension rubric (§3),
    0-10 per dimension, returned as strict JSON so evaluate.py can parse it
    without free-text extraction.
    """
    dims = ", ".join(RUBRIC_DIMENSIONS)
    return (
        "You are grading a short image description written by a small "
        "language model, the way a teacher grades a student's story. The "
        "model was shown an image and this partial description, and asked "
        "to complete it:\n\n"
        f'Partial text given to the model: "{partial_prompt}"\n'
        f'Full model completion: "{completion}"\n\n'
        f"Image context (for your reference only): {image_context}\n\n"
        f"Score the completion from 0 to 10 (10 = best) on each of these "
        f"{len(RUBRIC_DIMENSIONS)} dimensions: {dims}.\n"
        "- grammar: is it grammatically correct English?\n"
        "- creativity: does it add engaging, non-generic detail?\n"
        "- consistency: is it internally consistent, no contradictions?\n"
        "- meaningfulness: does it make coherent sense as a description?\n"
        "- plot: does it flow as a coherent continuation of the partial text?\n\n"
        "Respond with ONLY a JSON object, no other text, in exactly this "
        "form: "
        '{"grammar": <0-10>, "creativity": <0-10>, "consistency": <0-10>, '
        '"meaningfulness": <0-10>, "plot": <0-10>}'
    )
