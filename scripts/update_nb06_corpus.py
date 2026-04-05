"""Update notebook 06 to use shakespeare_pretraining_corpus.txt."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "06_training_complete_model.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ── CELL 4: corpus loading ────────────────────────────────────────────────────
nb["cells"][4]["source"] = """\
# Load the Shakespeare Wikipedia pre-training corpus
# This corpus contains full Wikipedia article text for all 37 Shakespeare plays
# written in modern English -- eliminating the Elizabethan domain gap.
corpus_choice = 'shakespeare_wikipedia'
filepath = '../data/shakespeare_pretraining_corpus.txt'

import os
if not os.path.exists(filepath):
    raise FileNotFoundError(
        f'Pre-training corpus not found: {filepath}\\n'
        'Run scripts/fetch_wikipedia_shakespeare.py then scripts/build_pretraining_corpus.py first.'
    )

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

print(f'Corpus: {corpus_choice}')
print(f'  File:       {filepath}')
print(f'  Characters: {len(text):,}')
print(f'  Words:      {len(text.split()):,}')
print(f'  Lines:      {len(text.splitlines()):,}')
print(f'  Unique chars: {len(set(text))}')
print(f'  File size:  {os.path.getsize(filepath) / 1024:.1f} KB')

print(f'\\nSample (first 500 characters):')
print('-' * 60)
print(text[:500])
print('-' * 60)
"""

# ── CELL 36: generation prompts ───────────────────────────────────────────────
nb["cells"][36]["source"] = """\
# Test prompts aligned with the Wikipedia Shakespeare corpus
prompts = [
    '=== PLAY: Hamlet ===',
    '=== PLAY: Macbeth ===',
    'Hamlet is a tragedy',
    'Romeo and Juliet is set in',
    'The main themes of Macbeth include',
]

print('=' * 70)
print('SHAKESPEARE WIKIPEDIA CORPUS -- GENERATION SAMPLES')
print('=' * 70)

for prompt in prompts:
    print(f'\\nPrompt: \"{prompt}\"')
    print('-' * 70)

    for temp in [0.5, 0.8, 1.0]:
        generated = generate_sample(model, tokenizer, prompt,
                                    max_len=100, temperature=temp, device=device)
        print(f'\\n[Temperature = {temp}]')
        print(generated[:250] + '...')

    print('\\n' + '=' * 70)
"""

# ── CELL 38: temperature comparison ──────────────────────────────────────────
nb["cells"][38]["source"] = """\
prompt = 'Hamlet is a tragedy written by William Shakespeare'
temperatures = [0.3, 0.7, 1.0, 1.5]

print(f'Comparing temperatures for prompt: \"{prompt}\"\\n')
print('=' * 70)

for temp in temperatures:
    print(f'\\nTemperature = {temp}:')
    if temp == 0.3:
        print('(Low - conservative, repetitive)')
    elif temp == 0.7:
        print('(Medium - balanced)')
    elif temp == 1.0:
        print('(Normal - more diverse)')
    else:
        print('(High - creative, possibly incoherent)')

    print('-' * 70)
    generated = generate_sample(model, tokenizer, prompt,
                                max_len=150, temperature=temp, device=device)
    print(generated)
    print()

print('=' * 70)
"""

# ── CELL 40: question continuation test ──────────────────────────────────────
nb["cells"][40]["source"] = """\
# Test whether the model can continue Wikipedia-style Shakespeare explanations
question_prompts = [
    'Hamlet is a tragedy about',
    'The main themes of Macbeth include',
    'Romeo and Juliet explores the theme of',
]

print('=' * 70)
print('TESTING: Wikipedia-style continuation')
print('=' * 70)

for question in question_prompts:
    print(f'\\n{question}')
    print('-' * 70)

    response = generate_sample(model, tokenizer, question,
                               max_len=100, temperature=0.7, device=device)

    answer = response[len(question):]
    print(f'Model output: {answer[:180]}...')

print('\\n' + '=' * 70)
print('\\nObservation:')
print('  This model is a next-token predictor trained on modern English.')
print('  After fine-tuning (notebook 07), it will answer questions in Q&A format.')
"""

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook 06 updated.")

for idx, label in [(4, "Cell 4"), (36, "Cell 36"), (38, "Cell 38"), (40, "Cell 40")]:
    src = nb["cells"][idx]["source"]
    lines = src.count("\n") + 1
    print(f"  {label}: {lines} lines, {len(src)} chars -- OK")
