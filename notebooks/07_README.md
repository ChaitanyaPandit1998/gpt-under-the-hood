# Notebook 07: Instruction Fine-Tuning

## Overview

This notebook implements **instruction fine-tuning** (supervised fine-tuning) to transform a base language model into a question-answering system. It demonstrates the second stage of modern LLM training pipelines used by models like GPT-4, Claude, and Llama.

## What You'll Learn

1. **The Three-Stage LLM Pipeline**
   - Stage 1: Pre-training (Notebook 06)
   - Stage 2: Supervised Fine-Tuning (This notebook)
   - Stage 3: RLHF (Future work)

2. **Key Concepts**
   - Creating instruction-following datasets
   - Loss masking (only train on answer tokens)
   - Fine-tuning vs pre-training differences
   - Evaluating instruction-following capabilities

3. **Practical Implementation**
   - Q&A dataset generation from Shakespeare text
   - InstructionDataset class with loss masking
   - Masked cross-entropy loss function
   - Fine-tuning training loop
   - Before/after evaluation

## Quick Start

### Prerequisites

- Complete Notebook 06 (Pre-training)
- Python 3.9+
- PyTorch, NumPy, Matplotlib

### Running the Notebook

1. **Activate your environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook notebooks/07_instruction_finetuning.ipynb
   ```

3. **Run cells in order:**
   - The notebook is designed to run sequentially
   - Each section builds on the previous one
   - Expected runtime: ~5-15 minutes on CPU

## Key Components

### 1. Q&A Dataset Generation

The notebook generates ~500-1000 Q&A pairs using:
- **Template-based generation** for common questions
- **Text parsing** to extract characters and quotes
- **Shakespeare-specific knowledge** encoded in templates

Example Q&A pair:
```
Q: Who wrote these plays?
A: These plays were written by William Shakespeare.
```

### 2. InstructionDataset Class

Formats data as:
```
Question: [user question]
Answer: [model response]
```

**Key innovation: Loss Masking**
- Loss mask = 0 for question tokens (don't train on input)
- Loss mask = 1 for answer tokens (train on output)

This ensures the model learns to generate answers, not questions.

### 3. Masked Loss Function

```python
def compute_instruction_loss(logits, targets, loss_mask):
    # Compute loss only on answer tokens
    loss_per_token = F.cross_entropy(logits, targets, reduction='none')
    masked_loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()
    return masked_loss
```

### 4. Fine-Tuning Configuration

| Parameter | Pre-Training | Fine-Tuning |
|-----------|--------------|-------------|
| Learning Rate | 3e-4 | 1e-5 |
| Epochs | 10-50 | 1-3 |
| Data | Raw text | Q&A pairs |
| Loss | All tokens | Answer tokens only |

## File Structure

```
notebooks/
  └── 07_instruction_finetuning.ipynb  # Main notebook (35 cells)

data/
  └── shakespeare_qa.json              # Generated Q&A dataset (created by notebook)

test_notebook_07.py                    # Test script for components
```

## Generated Outputs

When you run the notebook, it will create:

1. **shakespeare_qa.json** - Q&A dataset (~500-1000 pairs)
2. **Training plots** - Loss curves showing fine-tuning progress
3. **Model evaluation** - Before/after comparison of Q&A capability

## Example Results

**Before Fine-Tuning:**
```
Prompt: "Question: Who wrote these plays?\nAnswer:"
Output: [Random Shakespeare-style text, not an answer]
```

**After Fine-Tuning:**
```
Prompt: "Question: Who wrote these plays?\nAnswer:"
Output: "These plays were written by William Shakespeare."
```

## Architecture

The notebook uses the same MiniGPT architecture from Notebook 06:
- 6 transformer layers
- 8 attention heads
- 256 embedding dimensions
- ~2M parameters

Only the training objective changes - not the model architecture.

## Educational Value

This notebook demonstrates:

1. **How ChatGPT-style models are created**
   - Pre-training gives language understanding
   - Fine-tuning adds instruction-following capability

2. **Why loss masking matters**
   - Training on everything → model continues text
   - Training on answers only → model responds to questions

3. **Data efficiency**
   - 500 examples can teach a new behavior
   - Quality matters more than quantity

4. **Transfer learning in action**
   - Pre-trained knowledge is preserved
   - New capability is added on top

## Limitations

This is an educational implementation with:
- Small dataset (~500 pairs vs millions in real systems)
- Limited knowledge (only what's in the dataset)
- No reasoning (pattern matching, not understanding)
- Potential hallucination (plausible but incorrect answers)

Real systems like GPT-4 use:
- Millions of Q&A pairs
- Much larger models (billions of parameters)
- Additional RLHF training
- Safety filters and alignment techniques

## Next Steps

To extend this notebook:

1. **Expand the dataset**
   - Add more question types
   - Include plot summaries and analysis
   - Add context from multiple plays

2. **Improve generation**
   - Implement beam search
   - Add repetition penalty
   - Use better sampling strategies

3. **Add evaluation metrics**
   - BLEU score for answer quality
   - Perplexity on held-out set
   - Human evaluation rubric

4. **Implement RLHF (Stage 3)**
   - Collect human preferences
   - Train reward model
   - Use PPO for alignment

## Testing

To verify the implementation without running the full notebook:

```bash
python test_notebook_07.py
```

This tests:
- ✓ Tokenizer functionality
- ✓ InstructionDataset class
- ✓ Masked loss computation
- ✓ Q&A generation patterns
- ✓ Character extraction from text

## References

This notebook implements concepts from:
- InstructGPT paper (OpenAI, 2022)
- LLaMA 2 paper (Meta, 2023)
- General instruction fine-tuning literature

## Questions?

Common issues:

**Q: The model gives random answers**
A: This is expected with a small dataset. The model learns patterns but has limited knowledge. Increase dataset size or training epochs.

**Q: Training loss doesn't decrease**
A: Check learning rate (should be low, ~1e-5), verify loss mask is correct, ensure data is properly formatted.

**Q: Out of memory errors**
A: Reduce batch size, reduce max_seq_len, or use gradient checkpointing.

**Q: Answers are repetitive**
A: Adjust temperature in generation, add repetition penalty, or use nucleus sampling (top-p).

## Summary

This notebook completes the core LLM training pipeline:

```
Notebook 05: Tokenization
    ↓
Notebook 06: Pre-training
    ↓
Notebook 07: Instruction Fine-Tuning ← You are here
    ↓
Future: RLHF & Alignment
```

You now understand how modern AI assistants like GPT-4 and Claude are trained!
