# LLM Memory Interplay: Causal Tracing of Parametric vs. Contextual Memory

> Master Thesis Research · Causal Mediation Analysis on Large Language Models

This project investigates **how large language models (LLMs) balance parametric memory (knowledge stored in weights) and contextual memory (information provided in the prompt)** when answering factual questions. Using causal tracing / activation patching, we measure the layerwise and token-level causal contributions of hidden states, attention heads, and MLP sublayers to the model's final prediction.

---

## Method Overview

The core technique is **causal mediation analysis via activation patching**:

1. **Clean run** — feed the model a prompt with a supporting context passage and record the output probability for the correct answer.
2. **Corrupted run** — inject Gaussian noise into the embeddings of a target span (e.g., the object mention in the context) and observe the drop in probability.
3. **Restoration run** — for each `(token position, layer)` pair, restore the clean hidden state while keeping all other positions corrupted, then measure how much probability is recovered.

The **Indirect Effect (IE)** at each cell quantifies how much causal information flows through that specific component. Averaged across many examples, this produces a heatmap revealing which parts of the model and which token regions are responsible for factual recall.


---

## Results

### Causal Mediation Analysis — Layerwise Restoration

The heatmap below shows the average IE across hidden states, attention layers, and MLP layers after corrupting the object mention in the context. Each row corresponds to a token segment; each column is a transformer layer. Brighter cells indicate stronger causal contribution to the correct answer.

![Restoration Impact](src/results/joint9.jpg)

---

## Usage

### Running the Causal Tracer

```bash
cd src
python run_causal_tracer.py \
  --model_name /path/to/llama \
  --experiment_type 1 \
  --data_path ../data/data/peq/all-data.jsonl \
  --output_dir ../experiments/ct/llama/popqa/e1 \
  --prompt_format "C: {context} Q: {prompt} A:" \
  --samples 10 \
  --window 10 \
  --max_cf 3
```

| Argument | Description | Default |
|---|---|---|
| `--model_name` | Path or HuggingFace ID of the model | required |
| `--experiment_type` | Experiment design (`1`, `2`, or `3`) | `1` |
| `--data_path` | Path to JSONL evaluation file | required |
| `--output_dir` | Directory for `.bin` result files and PDFs | required |
| `--samples` | Number of counterfactual noise samples | `10` |
| `--window` | Layer window size for attention/MLP restoration | `10` |
| `--max_cf` | Maximum counterfactuals per example | `3` |
| `--replace` | Replace embeddings instead of adding noise (`0`/`1`) | `0` |
| `--reverse_patching` | Patch from noisy → clean direction (`0`/`1`) | `0` |
| `--max_datapoints` | Limit number of processed examples (`0` = all) | `0` |

---

## Experiment Designs

All three experiments apply **causal mediation analysis** (measuring Average Total Effect, ATE, and Average Indirect Effect, AIE) to isolate how different components of a subject–object–relation triple are processed across the model's layers and token positions.

### Experiment 1 — Subject Causal Mediation
Target the **subject** span in the context (e.g., *"Juan"* in *"Juan was written in the Spanish language."*). Compute ATE and AIE to reveal which layers and token positions mediate the model's use of subject identity when forming factual predictions.  
**Research question:** *How does the model encode and propagate subject information from context?*

### Experiment 2 — Object Causal Mediation
Target the **object** span in the context (e.g., *"Spanish"*). Compute ATE and AIE to reveal where and how the object attribute is read from context and routed to the output.  
**Research question:** *How does the model encode and propagate object information from context?*

### Experiment 3 — Relation Causal Mediation
Target the **relation** tokens in the context (the predicate/relational phrase binding subject to object). Compute ATE and AIE to assess the causal role of relational structure in the model's factual recall.  
**Research question:** *How does the model represent and use the relation linking subject and object?*

---

## Data Format

Each line of the JSONL dataset contains:

```json
{
  "question": "Which language was Juan written in?",
  "answers": ["Spanish"],
  "subj": "Juan",
  "prop": "P407",
  "obj": "Spanish",
  "passages": [{"text": "Juan was written in the Spanish language.", "title": ""}],
  "prompt_wo_ctx": "The language Juan was written in is",
  "prompt_with_ctx": "C: Juan was written in the Spanish language. The language Juan was written in is",
  "obj_cf": ["French", "German", "Italian"]
}
```

The `obj_cf` field provides counterfactual objects used as noise substitutes and baseline comparisons.

---

## Key Concepts

| Term | Definition |
|---|---|
| **Parametric memory** | Facts encoded in the model's weights during pretraining |
| **Contextual memory** | Information supplied in the input context/passage |
| **Total Effect (TE)** | Overall causal effect of the corrupted span on the output |
| **Indirect Effect (IE)** | Causal effect mediated through a specific `(token, layer)` component |
| **Activation patching** | Replacing a corrupted activation with its clean counterpart to measure causal contribution |
