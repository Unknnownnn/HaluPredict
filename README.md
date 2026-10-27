# HaluPredict

A machine-learning guardrail that checks whether an AI's answer is actually supported by the
source text it was given, and flags it when it is not.

Built on 64,507 labelled examples from [HaluEval](https://github.com/RUCAIBox/HaluEval) as a
predictive-analytics project.

---

## The problem, in simple terms

You give a language model a document and ask a question. It answers fluently. But fluent is not
the same as correct, it may invent a date, a number, a name, or an entire fact the document never
contains. That is a **hallucination**.

A concrete case:

> **Context:** The Eiffel Tower is located in Paris, France. It was completed in 1889.
>
> **Question:** Where is the Eiffel Tower and when was it built?
>
> **Answer:** The Eiffel Tower is in Rome and was completed in 1950.

Every word is confident. Two facts are wrong. A human reader catches it; a downstream system does
not.

This project scores that answer **before it reaches the user**. It answers one question:

> *Is this response consistent with the context it was given?*

Note what it is **not**: it is not a fact-checker against the real world. If the source document
itself is wrong, a faithful answer is still "correct" by this definition. The task measures
**groundedness**, not truth. That distinction is deliberate and worth stating up front.

The output is a **risk score** from 0 to 1, a verdict, and the name of the path that produced it.

---

## How it works

The system runs in four steps.

### 1. Read the data

HaluEval ships four subsets. Three of them contain *paired* examples: the same context with one
correct answer and one hallucinated answer. The loader emits both as separate rows, which is why
10,000 raw examples become 20,000 training rows and why the label balance lands close to 50/50
without any resampling.

| Subset | Raw examples | Rows after pairing |
|---|---|---|
| Dialogue | 10,000 | 20,000 |
| Question answering | 10,000 | 20,000 |
| Summarisation | 10,000 | 20,000 |
| General | 4,507 | 4,507 |
| **Total** | | **64,507** |

Every row is reduced to one common schema: `context`, `prompt`, `response`, `is_hallucination`,
`task_type`, `group_id`.

### 2. Turn each example into 8 numbers

Models cannot read text, so each example becomes a small vector of interpretable features.

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

The two semantic features come from `all-MiniLM-L6-v2`, a 384-dimension sentence embedding model.
Only the *similarity* is kept, not the raw vectors - cheaper, interpretable, and impossible to
overfit on a tabular model with 8 inputs.

### 3. Train a fast model to handle the easy cases

Five different classifiers look at those 8 numbers and vote:

| Model | Why it is in the ensemble |
|---|---|
| Logistic Regression | linear baseline, highly interpretable |
| Random Forest (100 trees) | captures non-linear splits, robust to outliers |
| Decision Tree | simple, axis-aligned rules |
| SVM (RBF kernel) | smooth non-linear boundary |
| **ANN (PyTorch MLP 8 → 128 → 64 → 1)** | learns feature interactions the others cannot express |

Their predicted probabilities are averaged - *soft voting*. The value of an ensemble is not that
any single member is excellent; it is that the members fail on *different* rows.

The ANN is a standard feed-forward network with LayerNorm, ReLU and Dropout between the hidden
layers, trained by backpropagation with AdamW on a class-weighted binary cross-entropy loss, and
stopped early on validation average precision. It is wrapped to look like a scikit-learn
estimator, so it drops into the existing ensemble, cross-validation loop and serialisation code
with no changes anywhere else.

That stage produces a **risk score in about 37 ms on GPU**.

### 4. Ask a second, slower model only when the fast one is unsure

This is the **cascade**, and it is the core design idea.

If the fast model's score is confidently low or confidently high - outside the band **0.35 to
0.65** - the answer is returned immediately. That covers about **75% of traffic**.

If the score lands in that uncertain middle band, a much more expensive model is consulted: an
NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) that reads the context and the response
together and estimates how likely the response is to be *entailed* by the context. Its
hallucination score is `1 − P(entailment)`.

The two scores are blended:

```
final = 0.4 × tier1_score + 0.6 × tier2_score
```

A response is flagged when the final score crosses the cascade threshold.

**Why bother with two tiers?** Because the expensive model is far slower. Paying that cost on every
request - including the three-quarters that were never ambiguous - is waste. The cascade gets
close to the same quality for a fraction of the average latency.

---

## Evaluation

The split is **group-disjoint**. Because each context appears twice - once with the correct answer,
once with the hallucinated twin - the two twins must never end up on opposite sides of the
train/test boundary. A normal random split leaks information and quietly inflates every metric.
`StratifiedGroupKFold` grouped on `group_id` prevents that.

Thresholds are tuned on **out-of-fold** predictions only, so the operating point is never fitted to
rows the model has already memorised.

### Held-out test set - 12,901 examples

| Metric | Safe | Hallucination |
|---|---|---|
| Precision | 0.721 | **0.870** |
| Recall | **0.916** | 0.615 |
| F1 | 0.807 | 0.721 |

**Average Precision 0.855** · accuracy 0.772

| | Count |
|---|---|
| True negatives (safe, correctly cleared) | 6,153 |
| False positives (safe, wrongly flagged) | 568 |
| False negatives (hallucinations missed) | 2,377 |
| True positives (hallucinations caught) | 3,803 |

**Routing:** Tier-1 resolved 75.0% of requests in one pass (9,682); 25.0% escalated to the NLI
verifier (3,219). Thresholds in this run: Tier-1 `0.433`, cascade `0.860`.

### What this operating point means

At the tuned threshold the guardrail catches **62% of hallucinations** while flagging only
**568 of 6,721 safe answers** (8.5%). That is a precision-favouring setting: when it raises an
alarm you can trust it, at the cost of missing some bad answers.

The threshold is a dial, not a fixed truth. Lowering it catches more hallucinations and produces
more false alarms; the precision-recall curve in the notebook shows every setting on the menu. The
choice depends on whether a failed check or a false alarm costs more in your application.

### Per-model comparison on the held-out set

| Model | Average Precision |
|---|---|
| **ANN** | **0.891** |
| Random Forest | 0.883 |
| Ensemble (all five) | 0.876 |
| SVM | 0.854 |
| Logistic Regression | 0.815 |
| Decision Tree | 0.648 |

Worth reading honestly: on these 8 features the **ANN is the strongest single model**, and the
ensemble sits slightly *below* it. With only 8 inputs the members are highly correlated, so voting
averages away some of the ANN's edge rather than building on it. The ensemble is kept because it
is more stable across subgroups and less sensitive to any one model's failure mode - but if the
goal were purely to maximise this one number, shipping the ANN alone would be the better call.
On a richer input representation (the full embedding pair rather than one cosine scalar) the
ensemble's advantage would likely return.

### Latency

| Stage | GPU mean | GPU P95 | CPU mean |
|---|---|---|---|
| Tier-1 only | 36.7 ms | 38.7 ms | 41.7 ms |
| Tier-2 (when triggered) | 27.5 ms | 46.3 ms | 644 ms |
| **End to end** | **59.9 ms** | **100.7 ms** | **347 ms** |

First inference is slower - about 85 ms on GPU - because the models must be loaded into memory.
After that, a full check takes roughly a fifteenth of a second.

---

## What is inside the notebook

The notebook is ordered so it can be run top to bottom in Google Colab.

| Stage | What happens |
|---|---|
| **Setup** | Install PyTorch and `sentence-transformers`, clone HaluEval, check for a GPU |
| **Data loading** | Parse the four JSONL subsets with per-line error recovery, expand paired examples, unify the schema |
| **Feature engineering** | Normalise text, compute MiniLM embeddings, build the 8 feature columns |
| **ANN definition** | The `TorchMLPClassifier` wrapper class |
| **Splitting** | Group-disjoint 80/20 hold-out via `StratifiedGroupKFold` |
| **Training** | Fit the scaler, train all five Tier-1 models, generate out-of-fold scores, tune both thresholds, save artifacts |
| **Diagnostics** | Per-model average precision table and a standalone ANN scorecard |
| **Pipeline class** | `CascadedGuardrailPipeline` - loads artifacts and runs the two-tier prediction |
| **Benchmarking** | Latency measurement with GPU synchronisation, P50 / P95 / P99 |
| **Evaluation** | Held-out precision, recall, F1, confusion matrix, average precision |
| **Visualisations** | Confusion matrix, ROC curve and precision-recall curve |
| **Demo** | Single-example checks, sentence-level streaming scan |
| **Packaging** | Writes the standalone modules and zips the artifacts for download |
| **Deployment** | Artifact import/export, ask-anything helpers, ipywidgets form, Gradio app |

---

## How to run

### Full retrain (about 30–40 minutes)

1. Open `HaluPredict.ipynb` in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**.
4. Artifacts are written to `./guardrail_artifacts`.

In Colab that folder is temporary - it disappears when the runtime disconnects. To keep it, point
`ARTIFACT_DIR` at Drive before training:

```python
ARTIFACT_DIR = "/content/drive/MyDrive/guardrail_artifacts"
```

### Load a trained model instead (no dataset, no retraining)

Only three files are needed:

```
guardrail_artifacts/
  guardrail_metadata.json       feature order, thresholds, versions
  guardrail_scaler.joblib       the fitted StandardScaler
  tier1_voting_ensemble.joblib  the five trained models
```

The two HuggingFace models (`all-MiniLM-L6-v2`, `nli-deberta-v3-small`) download automatically on
first use - they are not in the zip.

Run order in a fresh session:

1. PyTorch setup cell
2. `pip install sentence-transformers`
3. **ANN definition cell** - this must run before anything loads a `.joblib`
4. Pipeline class cell
5. Deployment cell

```python
import_guardrail(archive_path="guardrail_deployment_package.zip")
load_guardrail("./guardrail_artifacts")
smoke_test()

check_and_report(
    context="The JWST launched on 25 December 2021 from French Guiana.",
    prompt="When and where did JWST launch?",
    response="It launched in 1999 from the Soviet Union.",
)
```

**The one gotcha:** the ensemble pickle stores `TorchMLPClassifier` *by reference*, not by value.
Load it without that class defined and you get `AttributeError: Can't get attribute
'TorchMLPClassifier'`. Run the ANN cell first and it never happens.

### Interactive interfaces

Two are included, both at the end of the notebook:

- **ipywidgets form** - three boxes and a *Check* button, works inside Colab.
- **Gradio app** - a browser UI with a single check, a sentence-by-sentence scan, and a simulated
  live-typing mode that rescores the answer word by word as it appears.

---

## Design decisions

| Decision | Reason |
|---|---|
| Average Precision as the headline metric | Accuracy hides the precision/recall trade-off and is dominated by the majority class; AP summarises the whole curve |
| Group-disjoint split | Stops the correct/hallucinated twins of one example leaking across train and test |
| Thresholds tuned on out-of-fold scores | Keeps the operating point away from rows the model already memorised |
| Scaler fitted inside each fold | No preprocessing leakage |
| ANN wrapped as an sklearn estimator | Fits the existing ensemble, CV loop and serialisation unchanged - an addition, not a refactor |
| Metadata file carries a schema version | A stale artifact fails loudly instead of silently mispredicting with the wrong feature order |
| Cascade instead of one model | Most traffic is easy; spend the expensive compute only where it changes the answer |
| `predict_streaming()` | The real product need is to flag the *sentence* where a response goes wrong, not the whole paragraph |

---
rmers · pandas · NumPy · Matplotlib · Gradio ·
ipywidgets
