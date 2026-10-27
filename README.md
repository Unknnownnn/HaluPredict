# HaluPredict — LLM Hallucination Guardrail

A machine-learning guardrail that checks whether an AI's answer is actually supported by the
source text it was given, and flags it when it is not.

Built on 64,507 labelled examples from [HaluEval](https://github.com/RUCAIBox/HaluEval) as a
predictive-analytics project.

---

## The problem, in plain terms

You give a language model a document and ask a question. It answers fluently. But fluent is not
the same as correct — it may invent a date, a number, a name, or an entire fact that the document
never says. That is a **hallucination**.

Here is a concrete case:

> **Context:** The Eiffel Tower is located in Paris, France. It was completed in 1889.
>
> **Question:** Where is the Eiffel Tower and when was it built?
>
> **Answer:** The Eiffel Tower is in Rome and was completed in 1950.

Every word is confident. Two facts are wrong. A human reader catches it; a downstream system
does not.

This project scores that answer **before it reaches the user**. It answers one question:

> *Is this response consistent with the context it was given?*

Note what it is **not**: it is not a fact-checker against the real world. If the source document
itself is wrong, a faithful answer is still "correct" by this definition. The task is measuring
**groundedness**, not truth. That distinction is deliberate and worth stating up front.

The output is a **risk score** between 0 and 1, plus a verdict and the name of the path that
produced it.

---

## How it works

The system runs in four steps.

### 1. Read the data

HaluEval ships four subsets. Three of them contain *paired* examples: the same context with one
correct answer and one hallucinated answer. The loader emits both as separate rows, which is why
10,000 raw examples become 20,000 training rows and why the label balance is naturally close to
50/50.

| Subset | Raw examples | Rows after pairing |
|---|---|---|
| Dialogue | 10,000 | 20,000 |
| Question answering | 10,000 | 20,000 |
| Summarisation | 10,000 | 20,000 |
| General | 4,507 | 4,507 |
| **Total** | | **64,507** |

Each row is reduced to a common schema: `context`, `prompt`, `response`, `is_hallucination`,
`task_type`, `group_id`.

### 2. Turn each example into 8 numbers

Models cannot read text, so every example becomes a small vector of interpretable features.

| Feature | What it measures | Intuition |
|---|---|---|
| `cosine_similarity` | semantic distance between context and response | do they talk about the same thing? |
| `prompt_response_cosine` | semantic distance between question and response | does the answer actually address the question? |
| `reference_length` | size of the source text | how much material there was to be faithful to |
| `response_length` | size of the answer | longer answers have more room to drift |
| `length_ratio` | response ÷ reference | is this a summary or a rant? |
| `lexical_overlap` | Jaccard overlap of word sets | are the same words reused? |
| `unsupported_number_ratio` | share of numbers in the response absent from the source | invented figures are a strong signal |
| `context_missing` | 1 if no context was supplied | the "general" subset runs in a different mode |

Semantic features come from `all-MiniLM-L6-v2`, a 384-dimension sentence embedding model. Only the
*similarity* is kept, not the raw vectors — cheaper, interpretable, and impossible to overfit 384
dimensions on 8-feature tabular models.

### 3. Train a fast model to handle the easy cases

Five different classifiers look at those 8 numbers and vote:

| Model | Why it is in the ensemble |
|---|---|
| Logistic Regression | linear baseline, highly interpretable |
| Random Forest (100 trees) | captures non-linear splits, robust to outliers |
| Decision Tree | simple, axis-aligned rules |
| SVM (RBF kernel) | smooth non-linear boundary |
| **ANN (PyTorch MLP 8 → 128 → 64 → 1)** | learns feature interactions the others cannot express |

Their predicted probabilities are averaged (**soft voting**). The value of an ensemble is not
that any single member is excellent — it is that the members fail on *different* rows.

The ANN is a standard feed-forward network with LayerNorm, ReLU and Dropout, trained by
backpropagation with AdamW on a class-weighted binary cross-entropy loss, stopping early on
validation average precision. It is wrapped to look like a scikit-learn estimator, so it drops
into the existing ensemble and serialisation code without any changes elsewhere.

That combination produces a **risk score in roughly 35 ms on GPU**.

### 4. Ask a second, slower model only when the fast one is unsure

This is the **cascade**, and it is the core design idea.

If the fast model's score is confidently low or confidently high — outside the band `0.35` to
`0.65` — the answer is returned immediately. That covers about **75% of traffic**.

If the score lands in that uncertain middle band, a much more expensive model is consulted: an
NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) that reads the context and the response
together and estimates how likely the response is to be *entailed* by the context. Hallucination
score is `1 − P(entailment)`.

The two scores are then blended:

```
final = 0.4 × tier1_score + 0.6 × tier2_score
```

A response is flagged when the final score crosses the cascade threshold.

**Why bother with two tiers?** Because the expensive model is about 20× slower on CPU. Paying that
cost on every request — including the three-quarters that were never ambiguous — is waste. The
cascade buys close to the same quality for a fraction of the average latency.

---

## Evaluation

The split is **group-disjoint**: because each context appears twice (once with the correct answer,
once with the hallucinated twin), the two twins must never end up on opposite sides of the
train/test boundary. A normal random split leaks information and inflates every metric.
`StratifiedGroupKFold` grouped on `group_id` prevents that.

Thresholds are tuned on **out-of-fold** predictions only — every training row is scored by a model
that never saw it — so the operating point is not fitted to data the model has memorised.

### Last complete run

Held-out test set of 12,902 examples:

| Metric | Safe | Hallucination |
|---|---|---|
| Precision | 0.801 | 0.661 |
| Recall | 0.609 | **0.834** |
| F1 | 0.692 | 0.738 |

Average Precision **0.746** · accuracy 0.717 · TN 4,104 · FP 2,635 · FN 1,022 · TP 5,141

Tier-1 resolved 74.8% of traffic; 25.2% escalated to Tier-2.
Latency: **~35 ms** Tier-1, **~53 ms** end-to-end on GPU, ~234 ms on CPU.

**Read the trade-off correctly.** Hallucination recall is 83% while precision is 66% — the system
accepts a lot of false alarms to avoid missing real ones. For a guardrail that is the right
operating point: a spurious warning costs a glance, a missed hallucination ships to the user. The
threshold is a dial, and the PR curve shows every other setting on it.

> These numbers come from an earlier 5-feature, randomly-split version of the pipeline. The
> current notebook uses 8 features and the stricter group-disjoint split, so they will move.
> **Re-run the evaluation cell before quoting any figure.**

---

## Project structure

```
HaluPredict/
├── Copy_of_PredictiveProject.ipynb            Main notebook — the whole pipeline end to end
├── ProjectFinalDeepseek.ipynb                 Earlier working version, kept for provenance
├── Predictive_Analysis_LLM_Hallucination_
│   Guardrail_Report.docx                      Written report
├── guardrail_deployment_package.zip           Trained artifacts + standalone Python modules
├── docs/
│   ├── architecture.mmd                       System diagram (Mermaid source)
│   └── cascade-flow.mmd                       Cascade diagram (Mermaid source)
├── README.md
└── .gitignore
```

The `.mmd` files are plain text; paste either into [mermaid.live](https://mermaid.live) to render it.

---

## What is inside the notebook

The notebook is ordered so it can be run top to bottom in Google Colab.

| Stage | What happens |
|---|---|
| **Setup** | Install PyTorch and `sentence-transformers`, clone the HaluEval repository, check for a GPU |
| **Data loading** | Parse the four JSONL subsets with per-line error recovery, expand paired examples, unify the schema into one DataFrame |
| **Feature engineering** | Normalise text, compute embeddings, build the 8 feature columns |
| **ANN definition** | The `TorchMLPClassifier` wrapper class |
| **Splitting** | Group-disjoint 80/20 hold-out via `StratifiedGroupKFold` |
| **Training** | Fit the scaler, train all five Tier-1 models, generate out-of-fold scores, tune both thresholds, save artifacts |
| **Diagnostics** | Per-model average precision table and a standalone ANN scorecard |
| **Pipeline class** | `CascadedGuardrailPipeline` — loads artifacts and runs the two-tier prediction at inference time |
| **Benchmarking** | Latency measurement with proper GPU synchronisation, P50/P95/P99 |
| **Evaluation** | Held-out precision, recall, F1, confusion matrix, AP |
| **Demo** | Single-example checks, sentence-level streaming scan, interactive UI |
| **Deployment** | Artifact export/import, ask-anything helpers, ipywidgets form, Gradio app |

---

## How to run

### Full retrain (about 30–40 minutes)

1. Open `Copy_of_PredictiveProject.ipynb` in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**.
4. Artifacts are written to `./guardrail_artifacts`.

In Colab that folder is temporary — it disappears when the runtime disconnects. To keep it, point
`ARTIFACT_DIR` at Drive before training:

```python
ARTIFACT_DIR = "/content/drive/MyDrive/guardrail_artifacts"
```

### Load a trained model instead (no dataset, no retraining)

Only three files are needed:

```
guardrail_artifacts/
  guardrail_metadata.json          feature order, thresholds, versions
  guardrail_scaler.joblib          the fitted StandardScaler
  tier1_voting_ensemble.joblib     the five trained models
```

The two HuggingFace models (`all-MiniLM-L6-v2`, `nli-deberta-v3-small`) download automatically on
first use — they are not in the zip.

Run order in a fresh session:

1. PyTorch setup cell
2. `pip install sentence-transformers`
3. **ANN definition cell** — this must run before anything loads a `.joblib`
4. Pipeline class cell
5. Artifacts cell

```python
import_guardrail(archive_path="guardrail_artifacts.zip")
load_guardrail("./guardrail_artifacts")
smoke_test()

check_and_report(
    context="The JWST launched on 25 December 2021 from French Guiana.",
    prompt="When and where did JWST launch?",
    response="It launched in 1999 from the Soviet Union.",
)
```

**The one gotcha:** the ensemble pickle stores `TorchMLPClassifier` *by reference*, not by value.
Load it without that class defined and you get
`AttributeError: Can't get attribute 'TorchMLPClassifier'`. Run the ANN cell first and it never
happens.

### Interactive interfaces

Two are included, both at the end of the notebook:

- **ipywidgets form** — three boxes and a *Check* button, works inside Colab.
- **Gradio app** — a browser UI with single check, sentence-by-sentence scan, and a simulated
  live-typing mode that scores the answer word by word as it appears.

---

## Key design decisions

| Decision | Reason |
|---|---|
| Average Precision as headline metric | Accuracy hides the precision/recall trade-off and is dominated by the majority class; AP summarises the whole curve |
| Group-disjoint split | Prevents the correct/hallucinated twins of one example from leaking across train and test |
| Thresholds tuned on OOF scores | Stops the operating point being fitted to rows the model already memorised |
| Scaler fitted inside each fold | No preprocessing leakage |
| ANN wrapped as an sklearn estimator | Fits the existing ensemble, OOF loop and serialisation unchanged — pure addition, no refactor |
| Metadata file carries the schema version | A stale artifact fails loudly instead of silently mispredicting with the wrong feature order |
| Cascade instead of a single model | Most traffic is easy; spend the expensive compute only where it changes the answer |
| `predict_streaming()` | The product need is to flag the *sentence* where a response goes wrong, not the whole paragraph |

---

## Known limitations

**The gate is symmetric in score, not in correctness.** If the fast model is confidently wrong,
the request never escalates. In the last run, "JWST was launched in 1999 by the Soviet Union" was
scored safe by the fast path. That is a calibration failure, not a plumbing failure, and
calibration is the natural next step.

**HaluEval hallucinations are machine-generated.** Perturbed summaries and ChatGPT-written wrong
answers may be easier to spot than organic human hallucinations, so these numbers likely overstate
real-world performance.

**The "general" subset has no context at all.** Those 4,507 rows are scored against the prompt
instead, which is a different task wearing the same label. The dataset is not homogeneous, and the
per-task breakdown should be read with that in mind.

**English only, response-level only.** Sentence-level hallucination labels exist in the dataset but
are unused here.

---

## Tech stack

Python · PyTorch · scikit-learn · sentence-transformers · pandas · NumPy · Gradio · ipywidgets
