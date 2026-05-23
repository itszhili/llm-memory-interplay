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

**Supported models:** GPT-2 · GPT-J · LLaMA

---

## Results

### Causal Mediation Analysis — Layerwise Restoration

The heatmap below shows the average IE across hidden states, attention layers, and MLP layers after corrupting the object mention in the context. Each row corresponds to a token segment; each column is a transformer layer. Brighter cells indicate stronger causal contribution to the correct answer.

![Restoration Impact](src/results/joint9.jpg)

---

## Project Structure

```
llm-memory-interplay/
├── data/
│   └── data/peq/               # PopQA-style evaluation datasets (JSONL)
│       ├── all-data.jsonl       # Full dataset
│       └── all/                 # Per-relation subsets (P17, P106, ...)
├── src/
│   ├── run_causal_tracer.py     # Main entry point
│   ├── take_avg_test_e1.py      # Averaging & plotting for Experiment 1
│   ├── take_avg_test_e2.py      # Averaging & plotting for Experiment 2
│   ├── take_avg_test_e3.py      # Averaging & plotting for Experiment 3
│   ├── causal_tracing/
│   │   ├── causal_tracer.py     # Core CausalTracer class
│   │   ├── layer_config.py      # Layer name configs for GPT-2 / GPT-J / LLaMA
│   │   ├── metrics.py           # Total Effect (TE) & Indirect Effect (IE)
│   │   ├── noise_level.py       # Adaptive noise calibration
│   │   └── plot_hidden_flow_heatmap.py  # Heatmap visualization
│   ├── utils/
│   │   ├── constants.py         # Device & prompt format constants
│   │   ├── layer_matching.py    # Layer name resolution helpers
│   │   ├── token_utils.py       # Tokenization, joint prob computation
│   │   ├── trace.py             # Single-layer activation hook
│   │   ├── trace_dict.py        # Multi-layer activation hook context manager
│   │   └── torch_utils.py       # Misc PyTorch helpers
│   └── results/
│       ├── joint9.jpg           # Joint heatmap (main result figure)
│       ├── heatmaps/            # Per-experiment individual heatmaps (PDF)
│       └── final plots/         # Publication-ready figures (PDF)
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/zhi-li-research/llm-memory-interplay.git
cd llm-memory-interplay

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers numpy tqdm matplotlib
```

> **Note:** A CUDA-capable GPU is strongly recommended. The tracer defaults to CUDA if available, otherwise falls back to CPU.

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

### Generating Averaged Heatmaps

After running the tracer, aggregate results across examples:

```bash
cd src
python take_avg_test_e1.py   # Experiment 1 averaged heatmaps
python take_avg_test_e2.py   # Experiment 2 averaged heatmaps
python take_avg_test_e3.py   # Experiment 3 averaged heatmaps
```

---

## Experiment Designs

### Experiment 1 — Object Corruption in Context
Corrupt the **object mention** in the context passage (e.g., "Spanish" in *"Juan was written in the Spanish language."*). Measure how strongly each layer and token position can restore the correct answer when patched back.  
**Research question:** *Where does the model read the factual object from the context?*

### Experiment 2 — Counterfactual Object, Subject Tracing
Replace the object in the context with a **counterfactual** (e.g., "French" instead of "Spanish"), then corrupt the **subject tokens**. Trace how the subject's representation causally mediates the answer.  
**Research question:** *How does subject identity interact with contextual object information?*

### Experiment 3 — Joint Subject + Object Corruption
Corrupt **both subject and object** spans simultaneously and measure the combined causal pathways.  
**Research question:** *What happens when both entity anchors are disrupted together?*

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

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{li2025llm,
  title   = {LLM Memory Interplay: Causal Tracing of Parametric vs. Contextual Memory},
  author  = {Zhi Li},
  year    = {2025},
  school  = {}
}
```
