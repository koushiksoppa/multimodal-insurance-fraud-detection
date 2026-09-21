"""Render claim narratives from the narrative-channel fields.

No public corpus of fraud-labelled insurance claim narratives exists, so the
text modality has to be constructed. The construction is deliberately
constrained so that the resulting experiments stay meaningful:

1. **The narrative never sees the label.** Templates are selected from the
   narrative *fields* only. Nothing in the generator reads ``fraud_label``, so
   no lexical shortcut to the answer can be introduced. A test enforces this.

2. **The facts are real.** Every assertion in a narrative comes from a real
   value in the source table. The prose is constructed; the content is not.

3. **Prose is lossy.** A random subset of facts is omitted from each narrative,
   the way a real first-notice-of-loss report is incomplete. This means the
   text modality carries *less* information than the narrative fields do, which
   is the realistic case. The gap between a model reading the prose and a model
   reading the same fields as columns is measured directly, and reported as the
   information recovered from free text.

4. **No class-correlated vocabulary.** Because generation is label-blind, both
   classes draw from one shared pool of phrasings, registers and filler. There
   is no giveaway word for a classifier to latch onto.

The generator is seeded per claim, so a given claim always produces the same
narrative regardless of row order or parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.seeding import derive_seed

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# phrase banks
#
# Several phrasings per fact, so the transformer has to read meaning rather
# than match a fixed string. Every variant of a given fact is semantically
# equivalent; the variation is stylistic only.
# --------------------------------------------------------------------------

OPENERS = [
    "The policyholder reports that the incident occurred {when}.",
    "Loss reported {when}.",
    "According to the notification, the collision took place {when}.",
    "Incident date given as {when}.",
    "The insured states the accident happened {when}.",
    "Per the first notice of loss, the event occurred {when}.",
]

AREA_PHRASES = {
    "Urban": [
        "The location was within a built-up urban area.",
        "The incident took place on a city road.",
        "Scene recorded as urban.",
        "This occurred in a metropolitan area.",
    ],
    "Rural": [
        "The location was a rural road outside the town limits.",
        "The incident took place in a rural area.",
        "Scene recorded as rural.",
        "This occurred on a country road away from built-up areas.",
    ],
}

FAULT_PHRASES = {
    "Policy Holder": [
        "The insured accepts responsibility for the collision.",
        "Liability is attributed to our policyholder.",
        "Fault has been assigned to the insured driver.",
        "Our insured is recorded as the at-fault party.",
    ],
    "Third Party": [
        "The insured attributes responsibility to the other driver.",
        "Liability is attributed to a third party.",
        "Fault has been assigned to the other vehicle involved.",
        "A third party is recorded as the at-fault party.",
    ],
}

POLICE_PHRASES = {
    "Yes": [
        "A police report was filed at the scene.",
        "Attending officers produced a report.",
        "A police incident number has been supplied.",
        "Law enforcement attended and documented the collision.",
    ],
    "No": [
        "No police report was filed.",
        "The parties did not involve law enforcement.",
        "No police incident number is available.",
        "Officers did not attend the scene.",
    ],
}

WITNESS_PHRASES = {
    "Yes": [
        "An independent witness was present and has given a statement.",
        "A witness to the collision has been identified.",
        "There was a bystander who confirmed the sequence of events.",
        "Witness details were taken at the scene.",
    ],
    "No": [
        "No witnesses were present.",
        "No independent witness has been identified.",
        "There were no bystanders able to describe the collision.",
        "No witness details are on file.",
    ],
}

PAST_CLAIMS_PHRASES = {
    "none": [
        "The claimant has no prior claims on record.",
        "This is the first claim recorded for this claimant.",
        "No previous claim history is held.",
    ],
    "1": [
        "One earlier claim appears on the claimant's record.",
        "The claimant has a single prior claim.",
        "Claim history shows one previous loss.",
    ],
    "2 to 4": [
        "The claimant has between two and four prior claims.",
        "Claim history shows a handful of previous losses.",
        "Between two and four earlier claims are on record.",
    ],
    "more than 4": [
        "The claimant has more than four prior claims on record.",
        "Claim history shows an extensive record of previous losses.",
        "More than four earlier claims are recorded for this claimant.",
    ],
}

ADDRESS_CHANGE_PHRASES = {
    "no change": [
        "The registered address is unchanged.",
        "No address change is recorded on the policy.",
        "Correspondence address has remained the same.",
    ],
    "under 6 months": [
        "The policyholder changed address within the last six months.",
        "A recent address change, under six months old, is on file.",
        "The correspondence address was updated in the past six months.",
    ],
    "1 year": [
        "The policyholder changed address about a year ago.",
        "An address change roughly twelve months old is on file.",
        "The correspondence address was updated around a year ago.",
    ],
    "2 to 3 years": [
        "The policyholder last changed address two to three years ago.",
        "An address change from two to three years back is on file.",
        "The correspondence address was updated a few years ago.",
    ],
    "4 to 8 years": [
        "The policyholder last changed address several years ago.",
        "An address change from four to eight years back is on file.",
        "The correspondence address has been stable for several years.",
    ],
}

SUPPLEMENT_PHRASES = {
    "none": [
        "No supplementary documentation has been submitted.",
        "The file contains no supplements.",
        "No additional supporting documents accompany the claim.",
    ],
    "1 to 2": [
        "One or two supplementary documents accompany the claim.",
        "A small number of supplements have been submitted.",
        "The file contains a couple of additional documents.",
    ],
    "3 to 5": [
        "Between three and five supplementary documents have been submitted.",
        "Several supplements accompany the claim file.",
        "The file contains a moderate number of additional documents.",
    ],
    "more than 5": [
        "More than five supplementary documents have been submitted.",
        "An unusually large bundle of supplements accompanies the claim.",
        "The file contains many additional documents.",
    ],
}

POLICY_AGE_PHRASES = {
    "none": [
        "The policy was incepted immediately before the loss date.",
        "Cover had only just started when the incident occurred.",
        "There was effectively no gap between inception and the accident.",
    ],
    "1 to 7": [
        "The policy had been in force for under a week at the time of the loss.",
        "Cover began within a week of the incident.",
        "Only a few days separated policy inception from the accident.",
    ],
    "8 to 15": [
        "The policy had been in force for one to two weeks.",
        "Cover began roughly a fortnight before the incident.",
        "The policy was about two weeks old at the date of loss.",
    ],
    "15 to 30": [
        "The policy had been in force for two to four weeks.",
        "Cover began within the month preceding the incident.",
        "The policy was a few weeks old at the date of loss.",
    ],
    "more than 30": [
        "The policy had been in force for more than thirty days.",
        "Cover had been established well before the incident.",
        "The policy was already established at the date of loss.",
    ],
}

DELAY_PHRASES = [
    (0.5, [
        "The claim was notified almost immediately after the incident.",
        "Notification reached us within days of the loss.",
        "The loss was reported promptly.",
    ]),
    (4.0, [
        "The claim was notified within a few weeks of the incident.",
        "Notification followed the loss after a short interval.",
        "The loss was reported within the month.",
    ]),
    (12.0, [
        "The claim was notified some weeks after the incident.",
        "There was a noticeable interval between the loss and its notification.",
        "Notification came a couple of months after the event.",
    ]),
    (float("inf"), [
        "The claim was notified a considerable time after the incident.",
        "A long interval separated the loss from its notification.",
        "Notification was substantially delayed relative to the loss date.",
    ]),
]

#: Register-setting filler. Carries no information about any field, and is
#: drawn identically regardless of the claim's content, so it cannot become a
#: class signal. Its purpose is to stop every narrative reading like a form.
FILLER = [
    "The vehicle was recovered to an approved repairer for assessment.",
    "Photographs of the damage have been requested from the insured.",
    "The claimant was contacted by telephone to confirm the details above.",
    "An engineer's inspection has been scheduled.",
    "The file has been passed to the motor claims team for review.",
    "Correspondence with the insured is ongoing.",
    "Repair estimates are awaited.",
    "The insured confirmed that no injuries were sustained.",
    "Further documentation will be added to the file as it arrives.",
    "Contact details on the policy were verified during the call.",
]

CLOSERS = [
    "Assessment pending.",
    "Handler to review.",
    "File open.",
    "Awaiting further information.",
    "Under review by the claims team.",
]


@dataclass(frozen=True)
class GenerationConfig:
    """Knobs for narrative construction."""

    #: Probability that any given optional fact is omitted from the narrative.
    #: Models the incompleteness of a real first notice of loss, and makes the
    #: text modality strictly lossy relative to the narrative fields.
    omission_rate: float = 0.15

    #: Number of information-free filler sentences to mix in.
    min_filler: int = 0
    max_filler: int = 2

    #: Shuffle the order of the fact sentences, so position carries no meaning.
    shuffle_facts: bool = True

    include_closer_prob: float = 0.5


DEFAULT_CONFIG = GenerationConfig()


def _pick(generator: np.random.Generator, options: list[str]) -> str:
    return str(options[generator.integers(len(options))])


def _when_phrase(row: pd.Series, generator: np.random.Generator) -> str:
    """Describe the incident timing without inventing a precise date.

    The source has only a month, a week within that month and a day name, so
    the narrative says exactly that and no more.
    """
    day = row.get("incident_day_of_week")
    month = row.get("incident_month")
    week = row.get("incident_week_of_month")
    ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    try:
        week_word = ordinals.get(int(week), "")
    except (TypeError, ValueError):
        week_word = ""

    variants = [
        f"on a {day} in {month}",
        f"on {day} during the {week_word} week of {month}" if week_word else f"on {day} in {month}",
        f"in {month}, on a {day}",
    ]
    return _pick(generator, variants)


def _delay_phrase(row: pd.Series, generator: np.random.Generator) -> str | None:
    delay = row.get("report_delay_weeks")
    if delay is None or pd.isna(delay):
        return None
    for threshold, options in DELAY_PHRASES:
        if float(delay) <= threshold:
            return _pick(generator, options)
    return None


def _lookup(bank: dict[str, list[str]], value, generator: np.random.Generator) -> str | None:
    """Fetch a phrasing, returning None for values the bank does not cover.

    Unknown values are dropped rather than paraphrased, so the generator never
    asserts something the source data does not say.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    options = bank.get(str(value))
    return _pick(generator, options) if options else None


def render_claim_text(
    row: pd.Series,
    *,
    seed: int,
    config: GenerationConfig = DEFAULT_CONFIG,
) -> str:
    """Render one claim narrative.

    ``row`` must contain the narrative-channel fields. ``fraud_label`` is never
    read, even if present.
    """
    generator = np.random.default_rng(seed)

    opener = _pick(generator, OPENERS).format(when=_when_phrase(row, generator))

    facts = [
        _lookup(AREA_PHRASES, row.get("accident_area"), generator),
        _lookup(FAULT_PHRASES, row.get("fault"), generator),
        _lookup(POLICE_PHRASES, row.get("police_report_filed"), generator),
        _lookup(WITNESS_PHRASES, row.get("witness_present"), generator),
        _lookup(PAST_CLAIMS_PHRASES, row.get("past_number_of_claims"), generator),
        _lookup(ADDRESS_CHANGE_PHRASES, row.get("address_change_claim"), generator),
        _lookup(SUPPLEMENT_PHRASES, row.get("number_of_supplements"), generator),
        _lookup(POLICY_AGE_PHRASES, row.get("days_policy_accident"), generator),
        _delay_phrase(row, generator),
    ]
    facts = [f for f in facts if f]

    # Drop a random subset: a real report does not record every fact.
    kept = [f for f in facts if generator.random() >= config.omission_rate]
    if not kept and facts:
        # Never emit a narrative with no content at all.
        kept = [facts[int(generator.integers(len(facts)))]]

    n_filler = int(generator.integers(config.min_filler, config.max_filler + 1))
    if n_filler:
        chosen = generator.choice(len(FILLER), size=n_filler, replace=False)
        kept.extend(FILLER[i] for i in chosen)

    if config.shuffle_facts:
        order = generator.permutation(len(kept))
        kept = [kept[i] for i in order]

    parts = [opener, *kept]
    if generator.random() < config.include_closer_prob:
        parts.append(_pick(generator, CLOSERS))

    return " ".join(parts)


def generate_narratives(
    df: pd.DataFrame,
    *,
    base_seed: int,
    config: GenerationConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Render a narrative for every claim.

    Each claim is seeded from its own ``claim_id``, so narratives are stable
    under reordering, resampling and parallel execution.
    """
    # Guard rather than comment: the renderer must not be able to see the label
    # even by accident.
    source = df.drop(columns=["fraud_label"]) if "fraud_label" in df.columns else df

    texts = [
        render_claim_text(
            row,
            seed=derive_seed(base_seed, f"narrative:{row['claim_id']}"),
            config=config,
        )
        for _, row in source.iterrows()
    ]

    series = pd.Series(texts, index=df.index, name="claim_text")
    word_counts = series.str.split().str.len()
    logger.info(
        "generated %d narratives (words: median %.0f, p95 %.0f, max %d)",
        len(series),
        word_counts.median(),
        word_counts.quantile(0.95),
        word_counts.max(),
    )
    return series
