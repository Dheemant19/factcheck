"""Prompt and input builders for the LLM and the encoders (owner M3, T14; T07's zero-shot prompt).

Every way a claim reaches a model is built here, so runs, the app and the tests use the same
text:

    zero_shot_messages(claim, "liar")             # T07: task, label definitions, the claim
    knn_messages(claim, examples, "liar")         # T14: + similar claims already fact-checked
    rag_messages(claim, evidence, "pubhealth")    # T14: + numbered evidence sentences
    encoder_pair(claim, evidence)                 # T14: (claim, evidence) for a tokenizer pair

The LLM builders return chat messages, not a string: ``factcheck.models.llm`` applies the
model's chat template (Qwen3 with thinking off), so the same messages work for Llama too.
The answer words are the dataset's label names; the scorer reads the probability of each
label's first token (T07), so no prompt asks for letters or explanations.

This module needs no PyTorch, so the app and the tests can import it cheaply.
"""

from __future__ import annotations

from collections.abc import Sequence

# Per dataset: what the text is, and a one-line definition of each label (registry order).
# LIAR uses PolitiFact's own Truth-O-Meter definitions.
PROMPTS: dict[str, dict] = {
    "covid": {
        "task": "Decide whether this social media post about COVID-19 is real or fake news.",
        "labels": {
            "real": "the post states verified, factually accurate information.",
            "fake": "the post states false or misleading information.",
        },
    },
    "liar": {
        "task": "Rate this statement by a US politician on PolitiFact's six-point truth scale.",
        "labels": {
            "pants-fire": "the statement is not accurate and makes a ridiculous claim.",
            "false": "the statement is not accurate.",
            "barely-true": "the statement contains an element of truth but ignores critical facts "
            "that would give a different impression.",
            "half-true": "the statement is partially accurate but leaves out important details "
            "or takes things out of context.",
            "mostly-true": "the statement is accurate but needs clarification or additional "
            "information.",
            "true": "the statement is accurate and there is nothing significant missing.",
        },
    },
    "pubhealth": {
        "task": "Judge this public-health claim the way a professional fact-checker would.",
        "labels": {
            "false": "the claim is false.",
            "mixture": "the claim is partly true and partly false.",
            "true": "the claim is true.",
            "unproven": "there is not enough evidence to say whether the claim is true or false.",
        },
    },
    "isot": {
        "task": "Decide whether this news article is real news or fake news.",
        "labels": {
            "real": "the article is genuine reporting from a reliable outlet.",
            "fake": "the article is fabricated or comes from an unreliable source.",
        },
    },
    "climate_fever": {
        "task": "Decide whether the evidence supports or refutes this claim about climate.",
        "labels": {
            "supports": "the evidence shows the claim is true.",
            "refutes": "the evidence shows the claim is false.",
            "not_enough_info": "the evidence does not say whether the claim is true or false.",
            "disputed": "some evidence supports the claim and some refutes it.",
        },
    },
}

EXAMPLE_WORDS = 60  # kNN examples are cut to this many words (ISOT: title + start of article)


def label_definitions(dataset: str) -> str:
    return "\n".join(
        f"- {label}: {meaning}" for label, meaning in PROMPTS[dataset]["labels"].items()
    )


def answer_instruction(dataset: str) -> str:
    """The closing sentence of every prompt.

    It matters: without "Reply with the label only ..." Qwen3 puts about half its probability on
    "**" (it wants to bold the answer) on PUBHEALTH, and label coverage drops to 0.53 (T07).
    """
    return (
        f"Answer with exactly one label from: {', '.join(PROMPTS[dataset]['labels'])}. Reply "
        "with the label only, in lowercase, with no other text or formatting."
    )


def shorten(text: str, max_words: int = EXAMPLE_WORDS) -> str:
    """The first ``max_words`` words, with "..." when something was cut."""
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def user(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


def zero_shot_message(text: str, dataset: str) -> str:
    """T07's user message: task, label definitions, the text, and the answer format."""
    spec = PROMPTS[dataset]
    return (
        f"{spec['task']}\n\nLabels:\n{label_definitions(dataset)}\n\nText: {text}\n\n"
        f"{answer_instruction(dataset)}"
    )


def zero_shot_messages(text: str, dataset: str) -> list[dict]:
    return user(zero_shot_message(text, dataset))


def knn_messages(
    claim: str,
    examples: Sequence[tuple[str, str]],
    dataset: str,
    max_example_words: int = EXAMPLE_WORDS,
) -> list[dict]:
    """Few-shot prompt with retrieved, already fact-checked training claims.

    ``examples`` are (text, label name) pairs, MOST SIMILAR FIRST, as the retrieval index
    returns them (``factcheck.retrieval.knn_examples``). They are shown the other way round,
    so the most similar example sits right above the claim being rated. Long examples are cut
    to ``max_example_words`` words, which keeps 8 ISOT examples well under 2,000 tokens.
    """
    shots = "\n\n".join(
        f"Text: {shorten(text, max_example_words)}\nLabel: {label}"
        for text, label in reversed(examples)
    )
    spec = PROMPTS[dataset]
    return user(
        f"{spec['task']}\n\nLabels:\n{label_definitions(dataset)}\n\n"
        f"Here are similar texts that professional fact-checkers have already labelled:\n\n"
        f"{shots}\n\nNow label this text.\n\nText: {claim}\n\n{answer_instruction(dataset)}"
    )


def rag_messages(claim: str, evidence: Sequence[str], dataset: str) -> list[dict]:
    """Evidence prompt: the claim and numbered evidence sentences, judged on the evidence only.

    ``evidence`` comes from ``factcheck.retrieval.evidence_for`` (most similar first).
    """
    numbered = "\n".join(f"{i}. {sentence}" for i, sentence in enumerate(evidence, start=1))
    spec = PROMPTS[dataset]
    return user(
        f"{spec['task']}\n\nLabels:\n{label_definitions(dataset)}\n\n"
        f"Evidence:\n{numbered}\n\nJudge the text using only this evidence.\n\n"
        f"Text: {claim}\n\n{answer_instruction(dataset)}"
    )


def encoder_pair(claim: str, evidence: Sequence[str]) -> tuple[str, str]:
    """(claim, evidence) for an encoder, most similar sentence first.

    Pass both to the tokenizer as a pair, ``tokenizer(claim, evidence, truncation="only_second")``:
    it inserts the right separator for each model (BERT [SEP], RoBERTa </s></s>, ...) and cuts
    only the evidence, from its end, so the least similar sentence goes first.
    """
    return claim, " ".join(evidence)
