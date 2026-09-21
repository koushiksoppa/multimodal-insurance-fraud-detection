# Datasets

Every number on this page was measured by running the pipeline, not quoted from
a dataset card. Regenerate them with:

```bash
python scripts/download_data.py
python scripts/build_dataset.py
```

The build writes `evaluation/reports/build_report_real-v1.json`, which contains
the full machine-readable version of what follows.

---

## The honest summary first

There is no public dataset that pairs a fraud-labelled insurance claim with its
own damage photograph and its own claim narrative. This project therefore
assembles a corpus, and the assembly is where the credibility of every
downstream result is won or lost. The short version:

| Modality | Status | What is real | What is constructed |
|---|---|---|---|
| Structured | **Real** | 15,420 claim records, 33 attributes, real fraud labels, real class imbalance | Nothing |
| Image | **Real** | 2,300 photographs with real human-assigned damage labels | The *attachment* of a photograph to a claim |
| Text | **Constructed** | Every fact asserted in a narrative is a real value from the claim record | The prose that expresses those facts |
| Fraud label | **Real** | `FraudFound_P`, as distributed | Nothing |

The one thing that is neither real nor arbitrary is the image-to-claim
attachment, and it is the reason this project runs two experimental tracks
rather than one. See [Why two tracks](#why-two-tracks).

---

## Source 1 — structured claims (real)

**Name** Vehicle Insurance Claim Fraud Detection, circulating as `carclaims.txt`,
`fraud_oracle.csv` and `claims_oracle.csv`.

**Origin** A vehicle insurance claims table distributed with the Angoss
KnowledgeSeeker software and used in the academic insurance-fraud literature,
for example Phua, C., Alahakoon, D., & Lee, V. (2004), *Minority report in fraud
detection: classification of skewed data*, ACM SIGKDD Explorations 6(1), 50–59.

**Retrieved from** a public GitHub mirror, pinned in `src/data/sources.py`.
SHA-256 of the file as downloaded is recorded in `data/raw/manifest.json`.

**Licence** Not formally stated by the original distributor. The table is widely
redistributed for research and teaching and contains no identifiable
individuals. **This repository does not redistribute it** — the file is
downloaded at build time and is git-ignored. Treat it as research-use-only
rather than as freely licensed data. If you need a formally licensed
alternative, the pipeline is not tied to this file: replace the source record in
`src/data/sources.py` and adjust `SOURCE_RENAMES` in `src/data/schema.py`.

### Measured properties

| Property | Value |
|---|---|
| Rows | 15,420 |
| Columns | 33 |
| Fraud positives | 923 |
| Fraud rate | 5.986 % |
| Imbalance ratio | 15.7 : 1 |
| Fully duplicated rows | 0 |
| Duplicate feature vectors | 0 |
| Calendar years covered | 1994, 1995, 1996 |

Fraud rate by year — note the decline, which is why the headline protocol is a
temporal split:

| Year | Claims | Fraud | Rate |
|---|---:|---:|---:|
| 1994 | 6,142 | 409 | 6.659 % |
| 1995 | 5,195 | 301 | 5.794 % |
| 1996 | 4,083 | 213 | 5.217 % |

### Data quality findings

These are defects in the source data, found by `src/data/quality.py` and by the
profiling in `notebooks/02_data_quality.ipynb`. They are reported, never
silently repaired.

1. **`Age` is zero for 320 rows.** Every one of those rows carries
   `AgeOfPolicyHolder = "16 to 17"`. This is a coding artefact, not a driver
   aged zero. The pipeline converts it to an explicit missing value and adds a
   `driver_age_missing` indicator, so a model can use the *fact* of missingness
   without being told the driver is a newborn.

2. **`PolicyType` disagrees with `VehicleCategory` in 32.3 % of rows.**
   `PolicyType` embeds a vehicle class in its own string, and the two agree on
   only 67.66 % of records. The bulk of the disagreement is 4,987 rows labelled
   `"Sedan - Liability"` whose `VehicleCategory` is `Sport`. This matters
   because `PolicyType` is the single most fraud-associated column in the table
   (Cramér's V = 0.168), so a model leaning on it is leaning on a field with a
   known encoding defect. The inconsistency is exposed as the engineered
   feature `policy_type_vehicle_mismatch` rather than hidden or patched.

3. **One row has `MonthClaimed = "0"` and `DayOfWeekClaimed = "0"`.** A single
   malformed record. Its derived report delay becomes missing.

4. **Three rows have a negative implied report delay** — the claim month
   precedes the incident month by more than a year-boundary wrap can explain.
   Retained as missing rather than clipped to zero.

5. **Manufacturer names contain typographical errors** in the source:
   `Accura`, `Mecedes`, `Nisson`, `Porche`. Consistent within the file, so they
   are left as-is; correcting them would change nothing and would misrepresent
   the source.

6. **`PolicyNumber` is a monotonic row index**, 1 … 15,420, not a repeated
   customer key. Two consequences: customer-level leakage cannot occur in this
   table (there are no repeated customers to leak), and `PolicyNumber` must
   never be a feature, because under a temporal split it is a direct proxy for
   time. It is declared an identifier in the schema and a test enforces that
   identifiers stay out of the feature set.

### What this dataset does *not* contain

Stated explicitly, because inventing these would be the easiest way to make the
project look richer than it is:

- **No claim amount.** The table has `VehiclePrice` and `Deductible` but no
  settled or requested amount. The unified schema therefore has no
  `claim_amount` column on the real track, despite it being an obvious field for
  a fraud model.
- **No dates.** Only a calendar year plus (month, week-of-month, day-name) pairs.
  All temporal quantities in this project are approximations derived from those,
  and are labelled as derived in the schema.
- **No free text.** See source 3.
- **No images.** See source 2.
- **No injury, no repair cost, no adjuster outcome.**

---

## Source 2 — damage photographs (real images, real labels)

**Name** Comprehensive car damage classification dataset

**Retrieved from** `https://huggingface.co/datasets/DrBimmer/comprehensive-car-damage`

**Licence** MIT, declared on the dataset card. Not redistributed here;
downloaded at build time.

### Measured properties

2,300 images, all of which decode successfully (0 corrupt files). Six classes,
formed by damage location crossed with condition:

| Class | Location | Condition | Images |
|---|---|---|---:|
| `F_Normal` | front | normal | 500 |
| `F_Breakage` | front | breakage | 500 |
| `F_Crushed` | front | crushed | 400 |
| `R_Normal` | rear | normal | 300 |
| `R_Breakage` | rear | breakage | 300 |
| `R_Crushed` | rear | crushed | 300 |

These labels are **real and human-assigned**. They are used directly as the
damage-intelligence targets. No damage label anywhere in this project is
invented.

Condition is treated as ordinal — `normal` 0, `breakage` 1, `crushed` 2 —
on the grounds that a broken component (a lamp, a bumper) is a lesser loss than
structural deformation of a panel, which is how the two classes look in the
source images.

---

## Source 3 — claim narratives (constructed)

No public corpus of fraud-labelled insurance claim narratives exists. The text
modality is therefore generated by `src/data/text_generation.py`, under four
constraints that exist to keep the experiments meaningful:

1. **The generator never sees the label.** It reads narrative-channel *fields*
   only. `tests/test_text_generation.py` flips every fraud label and asserts
   that not one character of output changes. Without this, a text model's
   performance would measure the data construction rather than anything else.

2. **Every asserted fact is real.** Each sentence renders a real value from the
   source table. The prose is constructed; the content is not. Values the phrase
   bank does not cover are *omitted* rather than paraphrased, so the generator
   never asserts something the source does not say.

3. **Prose is deliberately lossy.** 15 % of facts are dropped at random per
   claim, the way a real first notice of loss is incomplete. The text modality
   therefore carries strictly *less* information than the narrative fields do.
   The size of that gap is measured directly — see
   [the narrative ceiling](#the-narrative-ceiling).

4. **No class-correlated vocabulary.** Because generation is label-blind, both
   classes draw from one shared pool of phrasings and filler. There is no
   giveaway token.

Measured output: **median 79 words, 95th percentile 98, maximum 116.**

---

## The channel partition

This is the most important design decision in the dataset, and the one most
worth challenging.

If claim narratives were generated from *all* the structured fields, the text
modality would be a lossy re-encoding of the tabular modality. Any
"multimodal gain" from combining them would be an artefact of the construction,
not a real finding. So the 30 real feature columns are partitioned into two
**disjoint** channels:

- **Tabular channel** (15 fields) — policy and vehicle attributes: the
  structured facts an insurer holds independently of how the incident was
  described. Make, vehicle category, price band, vehicle age, driver age, sex,
  marital status, policyholder age band, policy type, base policy, deductible,
  driver rating, number of cars, agent type, year.

- **Narrative channel** (15 fields) — circumstantial detail about the incident
  and the reporting process, the kind of thing a person writes in a report.
  Accident area, fault, police report filed, witness present, days from
  inception to accident and to claim, prior claim count, address change,
  supplement count, incident and claim timing, derived report delay.

**The partition rule is "what kind of information is this?", not "how
predictive is this?"** The split was written and committed before any
per-feature association with the label was measured — commit `1af4334`, which
predates the profiling. That ordering is checkable in the git history, and it is
what stops the partition from being quietly tuned to flatter one modality.

Measured association after the fact (Cramér's V with the fraud label):

| Channel | Fields | Strongest field | Max V | Mean V |
|---|---:|---|---:|---:|
| Tabular | 15 | `policy_type` | 0.168 | 0.063 |
| Narrative | 15 | `fault` | 0.131 | 0.038 |

Both channels carry real signal, and the tabular channel carries more. That is
an outcome, not a target.

`src/data/schema.py::assert_disjoint_channels` enforces disjointness at build
time and in the test suite.

### The narrative ceiling

Because the narrative channel exists in two forms — as clean columns and as
prose — the project can measure something most text pipelines cannot: how much
of the available information a transformer actually recovers from free text.
`narrative_spec()` in `src/features/tabular.py` builds a model on the narrative
fields *as columns*. The gap between that model and the transformer reading the
generated prose is the cost of expressing facts in language and reading them
back out. It is reported as a diagnostic, not as one of the seven ablation cells.

---

## Image-to-claim attachment, and why two tracks

The photographs and the claim records come from different sources. Attaching
them requires a rule, and the rule determines what the image arm measures.

### The rule on the real-label track

A photograph is attached to at most one claim — **never reused** — and claims
are chosen uniformly at random within each label class, with the covered
fraction held equal across classes.

Measured result: **2,300 of 15,420 claims carry a photograph (14.92 %)**. Fraud
prevalence is 5.986 % in the full table and 6.000 % in the image-covered subset,
so precision–recall metrics stay comparable between them.

Three consequences, stated plainly:

1. The photograph is **not a photograph of that claim's vehicle**. It carries
   real damage labels but no information about whether *that* claim was
   fraudulent.
2. The image arm on the real-label track is therefore a **negative control**. A
   correctly built fusion model should extract no fraud signal from it. If it
   appears to, something is wrong — a leak, a bug, or an over-fitted evaluation
   — and the ablation table is where that would surface.
3. Because each image is used once and claims are disjoint across splits, **no
   photograph can appear on both sides of a split by construction**. That is the
   strongest available guarantee against image leakage, stronger than any
   after-the-fact duplicate check.

### Why two tracks

The negative control answers "does this data contain multimodal signal?" It
cannot answer "can this architecture exploit multimodal signal when it is
there?" Conflating those two questions is how unfounded multimodal results get
published, so they are separated:

- **Track A — real (`real-v1`).** Real labels, label-independent image
  attachment. The primary result. Honest measurement of what each modality
  contributes to *this* data.
- **Track B — synthetic.** A controlled benchmark with a documented generating
  process that places known, tunable signal in all three modalities, including
  a real fraud typology (damage exaggeration: claimed severity inconsistent
  with visible damage). Verifies that the fusion architecture recovers signal
  that is known to be present, and lets fusion strategies be compared against
  ground truth. Labelled synthetic everywhere and never presented as real
  insurance data.

Both tracks share the unified schema, so the same models, splits and evaluation
code run unchanged on either.

---

## Splits

Two strategies, with the difference between them reported as a result.

**Temporal (default, and the headline protocol).** Train on the earliest
claims, validate on the next period, test on the latest — the only protocol a
deployed fraud model ever actually faces.

| Split | Period | Claims | Fraud | Rate | With image | Fraud + image |
|---|---|---:|---:|---:|---:|---:|
| train | before 1995-07 | 8,762 | 550 | 6.28 % | 1,313 (14.99 %) | 81 |
| val | 1995-07 to 1995-12 | 2,575 | 160 | 6.21 % | 402 (15.61 %) | 25 |
| test | 1996 | 4,083 | 213 | 5.22 % | 585 (14.33 %) | 32 |

Note the test period has a *lower* fraud rate than the training period. That is
real distribution drift, and it means PR-AUC is not directly comparable between
splits; every PR-AUC in this project is reported with its prevalence and its
lift over prevalence.

Under this strategy `policy_year` is dropped from the feature set: it is
constant within the test period, so it would hand the model a feature whose
test-time value it never saw in training.

**Stratified random.** Label-stratified, time-ignoring. Reported alongside the
temporal result to quantify the optimism that ignoring time buys.

### Leakage controls

Each is enforced structurally and re-checked by `src/data/splits.py::audit_splits`,
which is wired into the test suite:

| Route | Control | Status |
|---|---|---|
| Same claim in two splits | Splits partition claims | checked |
| Same image in two splits | Each image attached to exactly one claim | checked |
| Same policy in two splits | Policy number is unique per row | checked |
| Duplicate records across splits | Exact feature-vector comparison | checked |
| Future data in training | Temporal ordering assertion | checked |
| Preprocessing fitted on test | Scalers/encoders fitted inside an sklearn pipeline on the training split only | structural |
| Threshold tuned on test | Thresholds selected on validation only | structural |
| Augmentation leakage | Augmentation applied to the training split only, inside the training loop | structural |

All checks pass on `real-v1`. The build refuses to write a dataset that fails
any of them.

---

## Limitations

Read this section before quoting any result from this project.

- **The claim data is from 1994–1996.** Fraud patterns, vehicle technology,
  reporting practice and the regulatory environment have all changed. Nothing
  here should be read as a statement about fraud rates today.
- **The narratives are constructed.** A real first notice of loss contains
  spelling errors, contradictions, emotion, hedging and irrelevant detail. This
  generator produces none of those. A text model that works here would need
  revalidation on real narratives before it meant anything.
- **The photographs do not belong to these claims on the real track.** This is
  by design, and it is why the image arm there is a negative control rather than
  evidence about images.
- **The source licence is unclear** for the structured table. Research use only.
- **Fairness analysis is limited to the attributes that exist** — sex, marital
  status, policyholder age band. There is no race, ethnicity, income or
  geography beyond urban/rural, so no fairness claim can be made along those
  dimensions, and none is made.
- **The image-covered subset has few positives.** 14.92 % coverage over a 6 %
  base rate leaves exactly 32 fraud positives in the temporal test split with a
  photograph attached. Image-arm estimates are correspondingly wide, and are
  always reported with bootstrap confidence intervals. Any conclusion drawn
  from that arm is a statement about a 585-claim subset, not about the dataset.
