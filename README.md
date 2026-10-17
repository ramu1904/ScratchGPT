# ScratchGPT

**A GPT-style language model built, trained, and tested from scratch in PyTorch.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-from%20scratch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-268%20passing-brightgreen)

> **Please note:** ScratchGPT is a small **base language model**, not an instruction-tuned chatbot. It was pretrained on TinyStories, a dataset of children's stories, and learns to predict the next token in story-like text. It was not trained to follow instructions or behave like ChatGPT, so a prompt such as `Hi` will usually continue as a story rather than reply as an assistant.

---

## Highlights

| | |
|---|---|
| **Parameters** | 10,101,760 (~10.1M) |
| **Architecture** | Decoder-only Transformer, 6 layers, 8 heads, 320-dim embeddings |
| **Tokenizer** | Custom byte-level BPE, 8,192 tokens, trained on the project's own data |
| **Context length** | 256 tokens |
| **Training** | 10 epochs on a cleaned TinyStories subset, free Google Colab GPU (Tesla T4) |
| **Validation loss** | 3.1917 → **2.1014** (down ~34.2%) |
| **Tests** | 268 passing, code formatted with Black |
| **Interface** | Local Streamlit app, no paid API needed |

No pretrained GPT weights and no hosted language-model API are used. Every component was implemented and trained here.

---

## Contents

[What Was Built](#what-was-built) · [Model Card](#model-card) · [Architecture](#architecture) · [Tokenizer](#tokenizer) · [Data](#dataset-and-preprocessing) · [Training](#training-methodology) · [Results](#training-results) · [Graphs](#training-graphs) · [Quick Start](#quick-start) · [Reproducing Training](#reproducing-training) · [Tests](#tests-and-code-quality) · [Structure](#repository-structure) · [Limitations](#limitations) · [Roadmap](#roadmap) · [Licensing](#data-licensing-and-model-artifacts) · [Skills](#skills-demonstrated)

---

## What Was Built

| Component | Purpose |
|---|---|
| Byte-level BPE tokenizer | Learns merge rules from the training text and maps text to token IDs |
| GPT configuration | Stores and validates architecture hyperparameters |
| Token and positional embeddings | Represent token identity and position in the sequence |
| Causal self-attention | Lets each position attend only to earlier positions |
| Transformer blocks | Combine attention with feed-forward layers |
| GPT model | Produces vocabulary logits for next-token prediction |
| Dataset and dataloaders | Turn encoded text into training and validation sequences |
| Loss and training utilities | Next-token loss, optimization steps, epochs, validation |
| Checkpoint utilities | Save model, configuration, optimizer state, and training state |
| Generation utilities | Greedy decoding and sampling (temperature, top-k) |
| Scripts | Tokenizer training, model training, diagnostics |
| Unit tests | Cover the core components |
| Streamlit app | Local prompt-and-generate interface |

---

## Model Card

| Property | Details |
|---|---|
| Model family | GPT-style decoder-only Transformer |
| Framework | PyTorch |
| Parameters | 10,101,760 (tied embedding and output weights) |
| Vocabulary size | 8,192 |
| Context length | 256 tokens |
| Embedding dimension | 320 |
| Layers / heads | 6 / 8 |
| Feed-forward dimension | 1,280 |
| Dropout | 0.1 |
| Training data | TinyStories subset |
| Final checkpoint | `scratchgpt_cleaned_10epoch.pt` |
| Instruction-tuned / chat fine-tuned | No / No |
| RAG or external API | No / No |
| Training environment | Google Colab free GPU (Tesla T4 observed) |
| Local development | Windows, Python 3.10, CPU PyTorch |

---

## Architecture

```text
TinyStories text
      |  cleaning and preprocessing
      v
Custom byte-level BPE tokenizer  ->  token IDs, fixed-length sequences
      v
Token embeddings + positional embeddings
      v
6 x causal Transformer blocks (causal self-attention + feed-forward)
      v
Vocabulary logits (8,192 per position)
      v
Next-token cross-entropy loss  ->  optimizer updates, checkpoints, history
      v
Autoregressive generation  ->  local Streamlit interface
```

Causal self-attention stops a position from using future tokens. During inference a token is selected or sampled, appended to the input, and the process repeats until the token limit is reached.

```python
GPTConfig(
    vocab_size=8192,
    context_length=256,
    embedding_dim=320,
    num_layers=6,
    num_heads=8,
    feed_forward_dim=1280,
    dropout=0.1,
)
```

---

## Tokenizer

Implemented in `src/scratchgpt/tokenizer.py`.

- Starts from the 256 byte values and repeatedly merges the most frequent adjacent pairs until it reaches 8,192 tokens (7,936 learned merges).
- Byte-level, so any UTF-8 text can be encoded.
- Uses a GPT-2-style regular-expression pre-tokenization pattern. The merge rules are learned for this project, and pretrained GPT-2 merges are not used.
- Merge rules are saved to JSON and rebuilt on load.
- Trained on the **training split only**, so validation text never influenced the vocabulary.

The tokenizer is part of the model artifact. It must match the checkpoint, because swapping it would make token IDs inconsistent with the learned embedding table.

---

## Dataset and Preprocessing

The model was trained on a subset of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories): **10,000 stories for training** (about 8.69M characters) and **500 for validation** (about 406K characters). Counts vary slightly with normalization and newline handling.

**Cleaning:** inspection found a few text-encoding artifacts, such as malformed quotation marks and some accented characters. A cleaned copy corrected these in both splits. The cleaned files were compared against the originals to confirm only the intended replacements changed, and story order, line layout, and blank-line structure were preserved.

**Why TinyStories?** It makes a compact language-model experiment feasible. It also shapes behavior strongly: a model trained on stories learns story-like continuations, not question answering or conversation. Raw data files are not committed to Git, so keep the dataset source and license available and download the data separately.

---

## Training Methodology

ScratchGPT uses causal next-token prediction. The output logits are compared with the target sequence shifted by one position, using cross-entropy loss.

```text
Input:  The little girl went to the
Target: little girl went to the garden
```

| Setting | Value |
|---|---:|
| Sequence length | 256 tokens |
| Batch size | 8 |
| Stride (sequence-count calculation) | 256 tokens |
| Training sequences (cleaned) | ~8,310 |
| Validation sequences | ~388 |
| Epochs | 10 (fresh run, not a continuation of the earlier 3-epoch checkpoint) |
| Hardware | Google Colab free GPU, Tesla T4 |

Sequence counts depend on the cleaned text, tokenizer, and configuration.

**Artifacts:**

```text
checkpoints/
├── scratchgpt_cleaned_10epoch.pt
├── tokenizer_cleaned.json
└── training_history_cleaned_10epoch.json
```

The checkpoint stores model state and configuration alongside training state and history. Keep the tokenizer JSON with it.

---

## Training Results

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 4.0441 | 3.1917 |
| 2 | 3.1567 | 2.7536 |
| 3 | 2.8279 | 2.5195 |
| 4 | 2.6171 | 2.3783 |
| 5 | 2.4617 | 2.2810 |
| 6 | 2.3401 | 2.2186 |
| 7 | 2.2415 | 2.1703 |
| 8 | 2.1594 | 2.1390 |
| 9 | 2.0882 | 2.1176 |
| 10 | 2.0257 | **2.1014** |

| Metric | Epoch 1 | Epoch 10 | Change |
|---|---:|---:|---:|
| Training loss | 4.0441 | 2.0257 | ~49.9% lower |
| Validation loss | 3.1917 | 2.1014 | ~34.2% lower |

Validation loss improved at every epoch, with smaller gains toward the end. The lowest validation loss in this run was 2.1014, at epoch 10.

### How to read these numbers

1. **Did the software work?** The tests and successful local generation show the tested components and integration work.
2. **Did the model learn from the data?** The falling training and validation loss show it improved at next-token prediction on held-out TinyStories text. Training loss ended below validation loss, which is common and does not by itself prove overfitting or broad generalization.
3. **Is it a helpful assistant?** Not established. No instruction-tuning or instruction-following benchmark was performed, and lower story loss does not imply better reasoning or factual accuracy.

The original-data and cleaned-data runs used different text normalization and tokenization with independent initializations, so their losses are not a controlled test of the effect of cleaning alone.

---

## Training Graphs

<!--
  TO ADD YOUR GRAPHS:
  1. Open this README on GitHub and click the pencil icon (Edit).
  2. Drag each graph screenshot into the editor. GitHub creates the image link for you.
  3. Replace the "coming soon" lines below with the generated image lines.
-->

**Training and validation loss**

*Graph screenshot coming soon.*

**Validation loss**

*Graph screenshot coming soon.*

---

## Quick Start

**Requirements:** Python 3.10+, Git, PyTorch (build suited to your hardware), Streamlit, and the checkpoint and tokenizer files. A GPU is not needed for short generations, but CPU inference can be slow.

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ScratchGPT

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

python -m pip install -e ".[dev,training]"
python -m pip install "streamlit>=1.40"
```

Place these files in `checkpoints/`:

```text
checkpoints/scratchgpt_cleaned_10epoch.pt
checkpoints/tokenizer_cleaned.json
```

Run the app:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). Stop it with `Ctrl+C`. The app uses CUDA when available and falls back to CPU.

**App features:** chat-style display, max-new-tokens slider, sampling toggle, temperature, top-k, new-conversation button, and device, parameter-count and context-length info. It is a local inference demo and does not turn the base model into an assistant.

---

## Reproducing Training

The ten-epoch run above is already complete, so retraining is not needed to use the app. To reproduce the experiment:

1. Download or prepare the TinyStories subset.
2. Clean the identified text-encoding artifacts.
3. Train the tokenizer on the training split only.
4. Encode training and validation text with that tokenizer.
5. Train the GPT model and record validation loss.
6. Save the checkpoint and history JSON, then plot the losses from that history.
7. Load the checkpoint and tokenizer together for inference.

```powershell
python scripts/train_tokenizer.py --help
python scripts/train.py --help
```

Use the same data, tokenizer, configuration, random seed, and arguments as the experiment record, since different settings give different results.

---

## Tests and Code Quality

Tests cover the tokenizer, attention, embeddings, Transformer blocks, model configuration, generation, loss, datasets, dataloaders, checkpoints, and training utilities.

```powershell
python -m black --check src tests scripts app
python -m pytest -q
```

Last verified in the development environment: `268 passed in 10.29s` and `37 files would be left unchanged`. Re-run both after any change. Passing tests show the covered behavior works; they are not a measure of language quality.

---

## Repository Structure

```text
ScratchGPT/
├── app/streamlit_app.py
├── checkpoints/              # model, tokenizer, training history
├── configs/
├── data/                     # local datasets, not committed by default
├── docs/images/              # graph and app screenshots
├── notebooks/
├── scripts/                  # train, train_tokenizer, diagnose_loss, download_tinystories, ...
├── src/scratchgpt/           # attention, checkpoint, config, dataloaders, dataset, embeddings,
│                             # generation, loss, model, tokenizer, training, transformer_block
├── tests/
├── pyproject.toml
└── README.md
```

---

## Limitations

- **Not instruction-tuned:** it does not follow instructions or respond like a chatbot.
- **Narrow domain:** TinyStories strongly shapes what it generates.
- **Small and short-context:** 10.1M parameters and 256 tokens, far below modern general-purpose LLMs.
- **No factuality guarantee:** outputs can be repetitive, incorrect, or incoherent.
- **No RAG, no tools:** it is not grounded in documents and does not browse or run agent workflows.
- **No conversational benchmark and no end-of-turn token:** generation stops at a token limit.
- **Speed depends on hardware:** device, PyTorch build, and sampling settings all matter.

Do not use it as the sole source for medical, legal, financial, safety-critical, or other high-stakes decisions.

---

## Roadmap

Ideas for follow-on experiments, none of which are completed features:

1. A fixed prompt suite with repeatable saved generation samples.
2. An evaluation script that reports metrics separately from subjective examples.
3. Better stopping behavior and sampling controls.
4. Controlled comparisons of context length and model size.
5. Supervised instruction fine-tuning on appropriately licensed data, with assistant-response-only loss masking.
6. Evaluating a tuned checkpoint on held-out instruction-following prompts.
7. Structured experiment logs for configurations and results.

Instruction-tuning would be a separate experiment and should only be described as done once it is implemented, trained, and evaluated.

---

## Data, Licensing, and Model Artifacts

- **Dataset:** TinyStories remains subject to its own terms. Review the [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories) before redistributing any data.
- **Code license:** add a `LICENSE` file before publishing. Without one, others do not automatically have the right to reuse or redistribute the code.
- **Large checkpoint:** the cleaned checkpoint is about **121 MB**, over GitHub's regular 100 MiB per-file limit, so do not commit it as a normal file. Options are Git LFS, a GitHub Release asset, or Hugging Face Hub (each has its own limits). Once hosted, add the real download link here.
- Keep virtual environments, caches, raw datasets, and temporary checkpoints out of Git.

---

## Skills Demonstrated

- Implementing a decoder-only Transformer, causal self-attention, and autoregressive generation in PyTorch
- Building a byte-level BPE tokenizer and serializing its merge rules
- Splitting data so tokenizer merges are learned from training text only
- Training with next-token cross-entropy and reading train and validation loss trends
- Managing checkpoints together with tokenizer and configuration
- Writing tests, keeping code formatted, and shipping a local inference UI
- Reporting limitations honestly, including the difference between **pretraining** (learning next-token statistics) and **instruction-tuning** (training on instruction and response examples). ScratchGPT demonstrates the former.

---

## Acknowledgements

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) for the corpus, [PyTorch](https://pytorch.org/) for training, and [Streamlit](https://streamlit.io/) for the interface.