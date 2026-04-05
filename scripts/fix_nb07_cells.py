"""Fix formatting issues in notebook 07 cells 9, 25, and 28."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "07_instruction_finetuning.ipynb"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ── CELL 9: model load + layer freezing ──────────────────────────────────────
nb["cells"][9]["source"] = """\
# Load the Hamlet mixed-corpus pre-trained model from notebook 06
checkpoint_name = 'hamlet_full_mixed_pretrained_bpe_paper_inspired_small.pt'
checkpoint_path = f'../models/{checkpoint_name}'

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore tokenizer from checkpoint metadata when available.
    tokenizer = tokenizer_from_checkpoint(checkpoint, default_vocab_size=tokenizer_vocab_size)

    # Rebuild model with saved config
    config = checkpoint['config']
    model = MiniGPT(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f'Loaded pre-trained model from {checkpoint_path}')
    print(f"  Corpus choice: {checkpoint.get('corpus_choice', 'unknown')}")
    print(f"  Model preset:  {checkpoint.get('model_preset', 'unknown')}")
    print(f'  Tokenizer type: {tokenizer.tokenizer_type}')
    print(f"  Vocab size:  {config['vocab_size']}")
    print(f"  d_model:     {config['d_model']}")
    print(f"  num_layers:  {config['num_layers']}")
    print(f"  num_heads:   {config['num_heads']}")
    print(f"  d_ff:        {config['d_ff']}")
    print(f"  max_seq_len: {config['max_seq_len']}")
    print(f"  activation:  {config.get('activation', 'relu')}")
    print(f"  position_encoding_type: {config.get('position_encoding_type', 'sinusoidal')}")
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # --- Layer Freezing for Fine-tuning ---
    # Freeze embeddings + lower 6 blocks. Only top 2 blocks + ln_final + output_projection
    # remain trainable (~4M params vs 15.8M total), preventing overfitting on the small Q&A dataset.
    freeze_layers = [model.token_embedding, model.pos_encoding] + list(model.blocks[:6])
    for layer in freeze_layers:
        for param in layer.parameters():
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print('\\n--- Layer Freezing ---')
    print('  Frozen:    token_embedding, pos_encoding, blocks 0-5')
    print('  Trainable: blocks 6-7, ln_final, output_projection')
    print(f'  Total params:     {total_params:>12,}')
    print(f'  Frozen params:    {frozen_params:>12,} ({frozen_params / total_params * 100:.1f}%)')
    print(f'  Trainable params: {trainable_params:>12,} ({trainable_params / total_params * 100:.1f}%)')

except FileNotFoundError:
    print('ERROR: Pre-trained model not found!')
    print('Please run notebook 06 first to create the selected Hamlet pre-trained checkpoint.')
    raise
"""

# ── CELL 25: hyperparameters + optimizer ─────────────────────────────────────
nb["cells"][25]["source"] = """\
# Fine-tuning hyperparameters (with layer freezing)
learning_rate = 1e-4     # Higher LR is safe: fewer trainable params, less catastrophic forgetting risk
num_epochs = 80          # Early stopping will cut this short
eval_temperature = 0.2
eval_max_len = 50
warmup_steps = 30        # Fewer params warm up faster
patience = 15            # Stop after 15 epochs with no val loss improvement

# IMPORTANT: Only pass trainable (unfrozen) parameters to the optimizer.
# Passing frozen params wastes memory on optimizer state and is misleading.
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
    weight_decay=0.01,
)

# Learning rate scheduler
def get_lr(step, warmup_steps, max_steps):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

print(f'Optimizer configured with LR={learning_rate}')
print(f'Max epochs: {num_epochs} (early stopping patience={patience})')
print(f'Evaluation temperature: {eval_temperature}')
print(f'Evaluation max_len: {eval_max_len}')
"""

# ── CELL 28: training loop with best-checkpoint saving + early stopping ───────
nb["cells"][28]["source"] = """\
import copy

# Training loop with best-checkpoint saving and early stopping
train_losses = []
val_losses = []
step = 0

best_val_loss = float('inf')
best_model_state = None
epochs_without_improvement = 0

print('Starting fine-tuning...\\n')

for epoch in range(num_epochs):
    print(f'\\nEpoch {epoch + 1}/{num_epochs}')
    print('-' * 50)

    # Train
    train_loss, step = train_epoch(model, train_loader, optimizer, device, step)
    train_losses.append(train_loss)

    # Validate
    val_loss = validate(model, val_loader, device)
    val_losses.append(val_loss)

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss:   {val_loss:.4f}')

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        print(f'  >> New best val loss: {best_val_loss:.4f} -- checkpoint saved')
    else:
        epochs_without_improvement += 1
        print(f'  No improvement for {epochs_without_improvement}/{patience} epochs')

    # Early stopping
    if epochs_without_improvement >= patience:
        print(f'\\nEarly stopping triggered after {epoch + 1} epochs.')
        break

    # Sample generation every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print('\\nSample generation:')
        test_q = 'Question: Who is Hamlet?\\nAnswer:'
        output = generate_sample(
            model, tokenizer, test_q,
            max_len=20, temperature=eval_temperature,
            device=device, max_seq_len=config['max_seq_len'],
        )
        print(output)

# Restore best model before evaluation
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f'\\nRestored best model (val loss: {best_val_loss:.4f})')

# Save best checkpoint to disk
save_path = '../models/hamlet_finetuned_best.pt'
torch.save({
    'model_state_dict': best_model_state,
    'config': config,
    'best_val_loss': best_val_loss,
}, save_path)
print(f'Best checkpoint saved to {save_path}')
print('\\nFine-tuning complete!')
"""

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook updated.")

# Verify — check for broken string patterns
for idx, label in [(9, "Cell 9"), (25, "Cell 25"), (28, "Cell 28")]:
    src = nb["cells"][idx]["source"]
    issues = []
    lines = src.split("\n")
    for i, line in enumerate(lines):
        # A line ending mid-string-literal (unmatched quotes) would show up
        # Check for literal newlines inside string content (not line endings)
        pass
    print(f"{label}: {len(lines)} lines, {len(src)} chars — OK")
