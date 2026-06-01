# ARCHITECTURE.md

## 1. Overview
The end-to-end pipeline begins with loading the raw AG News dataset, followed by tokenisation using GPT-2's BPE tokenizer. A custom PyTorch dataset construction feeds into dataloaders that dynamically batch and pad sequences. The model architecture is a from-scratch PyTorch implementation of GPT-2 (124M), into which pretrained weights are loaded from an original OpenAI TensorFlow checkpoint. During training, transfer learning is achieved by fine-tuning only the classification head and the final transformer blocks while freezing the rest. Evaluation tracks loss and accuracy, leading to a robust inference pipeline that classifies new articles.

## 2. Model Architecture
The full GPT-2 124M architecture is implemented from scratch in PyTorch. 

```text
    Input Tokens
         |
   +-----+-----+
   |           |
 Token      Positional
Embedding   Embedding
   |           |
   +-----+-----+
         |
         v
+-------------------+
| Transformer Block | x 12
+-------------------+
         |
         v
  Final Layer Norm
         |
         v
Classification Head
```

Every component built from scratch includes:
- **Token and positional embeddings**: Maps discrete tokens and their positions into 768-dimensional continuous spaces.
- **Multi-head causal self-attention**: Computes attention weights across 12 heads, including the QKV projection and a causal mask to prevent peeking ahead.
- **Layer normalisation**: Stabilises training by normalising activations before attention and feed-forward blocks.
- **GELU activation**: Provides a smooth non-linearity used in the feed-forward network.
- **Feed-forward block (768 → 3072 → 768)**: Expands and contracts the hidden representation for complex feature extraction.
- **Residual connections**: Allows gradients to flow easily through the network by adding the input of a sub-layer to its output.
- **Classification head (768 → 4)**: Maps the final token's hidden state to 4 class logits.

**Last Token Position**: The last token position is used for classification because GPT-2 employs causal self-attention. This means the last token in the sequence is the only token that has attended to every preceding token, accumulating the context of the entire input text.

## 3. Weight Loading Pipeline
OpenAI's pretrained GPT-2 weights are distributed as TensorFlow checkpoints. The pipeline downloads these files and maps each variable manually to its corresponding PyTorch parameter. 

**QKV Matrix Splitting**: The TensorFlow checkpoint stores a single combined QKV projection matrix of shape 768×2304. To match our modular from-scratch PyTorch implementation, this matrix is explicitly split into three separate 768×768 matrices for the Query, Key, and Value projections.

**Weight Tying**: The token embedding matrix (50,257×768) is shared (tied) with the language modelling output head in the original architecture. This means the same matrix used to embed input tokens is reused to project final representations back into vocabulary scores. After `replace_classification_head()` or `configure_transfer_learning()`, the LM head is replaced with an untied 768→4 classification layer and tying no longer applies.

## 4. Transfer Learning Strategy
To preserve the language understanding capabilities of GPT-2 while adapting it for classification, the majority of the model is frozen. Only the final layers are unfrozen and trained.

This logic is implemented in `src/models/gpt2_classifier.py` as `configure_transfer_learning()` (freeze all parameters, replace the LM head, unfreeze the last *N* blocks, final layer norm, and classification head). The notebook calls that helper after pretrained weights are loaded.

| Component                    | Frozen | Trainable |
|------------------------------|--------|-----------|
| Transformer blocks 0–8       | Yes    | No        |
| Transformer blocks 9–11      | No     | Yes       |
| Final layer norm             | No     | Yes       |
| Original LM head (768→50257) | Removed| —         |
| New classification head (768→4)| No   | Yes       |

- **Total parameter count**: ~124M
- **Trainable parameter count**: ~21M

## 5. Data Pipeline
The data flow begins with downloading AG News from Hugging Face (via `prepare_datasets()` in `src/data/preprocessing.py`), writing stratified train/validation/test CSV splits, and loading them through `load_data()`. Frame validation (`validate_ag_news_frame()`) checks required columns and label range before training.

- **Stratified Splitting**: The dataset is split into Train, Validation, and Test sets using stratified splitting to ensure the class distribution remains perfectly balanced across all sets.
- **Tokenisation**: Text is tokenised using `tiktoken` with GPT-2's exact BPE vocabulary (shared constants in `src/data/columns.py`).
- **Sequence Truncation and Padding**: Sequences exceeding the context window are truncated. Instead of padding every sequence globally to the maximum model length (1024), we use **dynamic batch padding**.
- **Custom `custom_collate_fn`**: The PyTorch DataLoader uses `custom_collate_fn` in `src/data/dataloader.py`, which pads sequences only to the maximum length found *within the current batch*. This significantly reduces unnecessary computation on padding tokens.

## 6. Training Loop
The training loop runs step by step:
1. **Forward Pass**: The batch of tokenised text and attention masks are passed through the model.
2. **Loss Computation**: Cross-entropy loss is computed using only the logit from the *last token position* of the sequence. Other positions are ignored.
3. **Backward Pass**: Gradients are computed through the unfrozen layers.
4. **Gradient Clipping**: Gradients are clipped to a maximum norm to prevent exploding gradients.
5. **Optimizer Step**: The AdamW optimizer updates the weights.

**Evaluation Strategy**: During training, intermediate loss is computed every N steps using a small subset of **training** batches (`eval_iter`) and the **full validation** loader. At each epoch end, training accuracy is estimated on 20% of train batches (`train_acc_batch_fraction` in CONFIG) and validation accuracy on the full validation set. A final evaluation on all three splits is performed when training completes.

**Pre-training baseline**: After loading pretrained weights, the notebook calls `configure_transfer_learning()` (random 4-class head, top layers unfrozen) and then measures accuracy on a subset of batches *before* the training loop. This is a random-head baseline, not LM zero-shot classification.

| Hyperparameter              | Value         |
|-----------------------------|---------------|
| Optimizer                   | AdamW         |
| Learning rate               | 9e-6          |
| Weight decay                | 0.1           |
| Batch size                  | 32            |
| Epochs                      | 3             |
| Gradient clipping           | 1.0           |
| Eval batches (intermediate) | 5             |
| Evaluations per epoch       | 4 (`evals_per_epoch` in CONFIG) |
| Train acc batches (epoch)   | 20% of train loader (`train_acc_batch_fraction`) |
| Unfrozen transformer blocks | Last 3        |
| Loss function               | Cross-entropy |

## 7. Inference Pipeline
The model must use a 4-class output head (`replace_classification_head()` or `configure_transfer_learning()`) before inference. The `classify_news` function in `src/inference/predict.py` orchestrates inference step by step:

1. **Validate**: Reject empty text; ensure `out_head` output size matches `CONFIG["data"]["num_classes"]`.
2. **Tokenise**: Convert the raw input string into integer tokens using BPE (`TOKENIZER_ALLOWED_SPECIAL` from `src/data/columns.py`).
3. **Truncate**: Cap the token sequence at `max_length` if given (defaults to model context length). This is a truncate-only cap, not a padding target.
4. **Model Pass**: Pass a batch of shape `(1, seq_len)` at the **actual sequence length** — no padding to a fixed width (unlike training batches, which use dynamic padding in `custom_collate_fn`).
5. **Logit Extraction**: Extract the output logit at the *last real token* via `extract_last_token_logits()`.
6. **Argmax**: Apply `argmax` to the logits to determine the predicted class index.
7. **Mapping**: Map the integer index back to a human-readable category label (e.g., "Sci/Tech") via `LABELS`.

## 8. File and Directory Reference

```text
fine-tuned-gpt-2-news-classifier/
│
├── README.md                                 — project overview and quickstart guide
├── requirements.txt                          — Python dependency declarations
├── requirements-tf.txt                       — optional TensorFlow (GPT-2 checkpoint loading only)
├── .gitignore                                — git exclusion rules
├── LICENSE                                   — project license file
│
├── notebooks/
│   └── fine-tuned-gpt-2-news-classifier.ipynb — main project notebook (orchestration layer)
│
├── src/                                      — modularized source code
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py                        — CONFIG, LABELS, logger, validate_config()
│   ├── data/
│   │   ├── __init__.py
│   │   ├── columns.py                       — shared CSV column names and tokenizer constants
│   │   ├── preprocessing.py                 — prepare_datasets, load_data, validate_ag_news_frame
│   │   ├── dataset.py                       — NewsDataset class (tokenisation and encoding)
│   │   └── dataloader.py                    — DataLoader factory with custom_collate_fn (dynamic padding)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── layers.py                        — LayerNorm, GELU, and FeedForward built from scratch
│   │   ├── attention.py                     — MultiHeadAttention with causal masking
│   │   ├── transformer_block.py             — single TransformerBlock (attention + feed-forward + residuals)
│   │   └── gpt2_classifier.py              — GPTModel, replace_classification_head, configure_transfer_learning
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                         — train_classifier_simple loop (tqdm, TPU support, checkpointing)
│   │   ├── evaluate.py                      — calc_loss_batch, calc_loss_loader, evaluate_model
│   │   ├── logits.py                        — extract_last_token_logits shared helper
│   │   └── metrics.py                       — calc_accuracy_loader, save_loss_plot, save_accuracy_plot, generate_classification_metrics
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py                       — classify_news function for single-article inference
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                          — set_seed for reproducibility (torch, numpy, random)
│       └── helpers.py                       — GPT-2 weight download, TF checkpoint loading, QKV splitting
│
├── tests/                                    — pytest test suite
│   ├── conftest.py                          — shared fixtures (mini GPT config, classification model, sample data)
│   ├── test_config.py                       — tests for CONFIG structure, LABELS, and validate_config
│   ├── test_data.py                         — tests for NewsDataset, custom_collate_fn, and dataloader behaviour
│   ├── test_preprocessing.py                — tests for AG News frame validation helpers
│   ├── test_models.py                       — tests for GPTModel, attention, layers, transformer block
│   ├── test_training.py                     — tests for loss computation and logits extraction
│   ├── test_metrics.py                      — tests for loss plot saving helpers
│   ├── test_inference.py                    — tests for classify_news inference pipeline
│   └── test_seed.py                         — tests for seed reproducibility
│
├── docs/
│   └── ARCHITECTURE.md                       — deep technical reference for the pipeline
│
├── data/                                     — [GITIGNORED] generated CSV splits
│   ├── news_train.csv                        — training split of AG News
│   ├── news_validation.csv                   — validation split of AG News
│   └── news_test.csv                         — test split of AG News
│
├── gpt2/                                     — [GITIGNORED] downloaded pretrained weights (~500MB)
│   └── 124M/
│       ├── checkpoint                        — TensorFlow checkpoint state tracker
│       ├── encoder.json                      — BPE vocabulary mapping (tokens to strings)
│       ├── hparams.json                      — GPT-2 architecture hyperparameters
│       ├── model.ckpt.data-00000-of-00001    — raw pretrained weights data
│       ├── model.ckpt.index                  — weight indexing mapping
│       ├── model.ckpt.meta                   — checkpoint metadata
│       └── vocab.bpe                         — BPE byte-pair merges
│
├── models/                                   — [GITIGNORED] saved fine-tuned model
│   ├── News_classifier.pth                   — saved PyTorch model state dictionary
│   ├── News_classifier_metadata.json         — training metadata (date, epoch, val loss, config)
│   └── training_metrics.json                 — recorded losses and accuracies per step/epoch
│
└── plots/                                    — [GITIGNORED] generated by notebook
    ├── loss-plot.pdf                         — training and validation loss curve visualisation
    ├── accuracy-plot.pdf                     — training and validation accuracy progression
    └── confusion_matrix.png                  — per-class confusion matrix heatmap
```

