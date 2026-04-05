"""
generate_hamlet_qa.py

Parses data/hamlet_aligned_corpus.txt and generates Q&A pairs for Hamlet fine-tuning.
Deduplicates against data/hamlet_qa.json (existing 90 pairs).
Outputs data/hamlet_qa_generated.json.
"""

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ALIGNED_CORPUS = DATA_DIR / "hamlet_aligned_corpus.txt"
EXISTING_QA = DATA_DIR / "hamlet_qa.json"
OUTPUT_QA = DATA_DIR / "hamlet_qa_generated.json"

SPEAKER_NORMALIZE = {
    "QUEEN": "Gertrude",
    "QUEEN GERTRUDE": "Gertrude",
    "GERTRUDE": "Gertrude",
    "HAMLET": "Hamlet",
    "CLAUDIUS": "Claudius",
    "HORATIO": "Horatio",
    "OPHELIA": "Ophelia",
    "LAERTES": "Laertes",
    "POLONIUS": "Polonius",
    "GHOST": "the Ghost",
    "BERNARDO": "Bernardo",
    "FRANCISCO": "Francisco",
    "ROSENCRANTZ": "Rosencrantz",
    "FIRST PLAYER": "the First Player",
}


def normalize_speaker(raw):
    key = raw.upper().strip()
    return SPEAKER_NORMALIZE.get(key, raw.title())


def parse_aligned_corpus(path):
    """Parse aligned corpus into passage blocks and background paragraphs."""
    text = path.read_text(encoding="utf-8")
    blocks = []

    # Split on [PLAY] tags
    segments = re.split(r'\[PLAY\]', text)

    for seg in segments[1:]:
        if '[SUMMARY]' not in seg:
            continue

        play_part, rest = seg.split('[SUMMARY]', 1)

        if '[CHARACTERS]' not in rest:
            continue

        summary_part, chars_part = rest.split('[CHARACTERS]', 1)

        speakers = []
        focus_tags = []
        for line in chars_part.strip().split('\n'):
            line = line.strip()
            if line.startswith('Speakers:'):
                raw = line.replace('Speakers:', '').strip()
                speakers = [s.strip() for s in raw.split(',')]
            elif line.startswith('Focus:'):
                raw = line.replace('Focus:', '').strip()
                focus_tags = [t.strip() for t in raw.split(',')]

        # Parse play lines: find speakers and their first dialogue line
        play_lines = play_part.strip().split('\n')
        first_speaker = None
        first_dialogue = None

        for line in play_lines:
            line = line.strip()
            if not line:
                continue
            # Speaker marker: ALL CAPS optionally with spaces, ending with colon
            if re.match(r'^[A-Z][A-Z\s]+:$', line):
                if first_speaker is None:
                    first_speaker = line.rstrip(':').strip()
            elif first_speaker and first_dialogue is None and line:
                first_dialogue = line

        blocks.append({
            'play_text': play_part.strip(),
            'summary': summary_part.strip(),
            'speakers': speakers,
            'focus_tags': focus_tags,
            'first_speaker': first_speaker,
            'first_dialogue': first_dialogue,
        })

    # Parse [BACKGROUND] paragraphs
    backgrounds = re.findall(r'\[BACKGROUND\]\n(.+)', text)

    return blocks, backgrounds


def generate_quote_attribution(blocks):
    """Generate 'Who says X?' pairs from passage dialogue lines."""
    pairs = []

    q_templates = [
        lambda q, sp: (f"Who says '{q}' in Hamlet?", f"{sp} speaks these words in Hamlet."),
        lambda q, sp: (f"Which character speaks the line '{q}'?", f"{sp} says this line in Hamlet."),
        lambda q, sp: (f"Who utters the words '{q}'?", f"These words are spoken by {sp} in Hamlet."),
    ]

    for i, block in enumerate(blocks):
        dialogue = block['first_dialogue']
        speaker_raw = block['first_speaker']

        if not dialogue or not speaker_raw:
            continue

        # Skip very short or trivial lines
        if len(dialogue) < 20:
            continue

        # Prefer lines up to 80 chars; truncate longer ones at punctuation
        quote = dialogue
        if len(quote) > 80:
            # Try to cut at a comma or colon
            cut = re.search(r'[,;:]', quote[:75])
            if cut:
                quote = quote[:cut.start()].strip()
            else:
                quote = quote[:75].strip()

        if len(quote) < 15:
            continue

        speaker = normalize_speaker(speaker_raw)
        template = q_templates[i % len(q_templates)]
        q, a = template(quote, speaker)

        if len(q) <= 150 and len(a) <= 150:
            pairs.append([q, a])

    return pairs


def generate_summary_factual(blocks):
    """Generate factual questions derived from block summaries."""
    pairs = []

    for block in blocks:
        summary = block['summary']
        focus = block['focus_tags']
        speakers = block['speakers']

        if not summary or not focus or not speakers:
            continue

        # Get clean speaker name
        speaker = normalize_speaker(speakers[0])
        primary_focus = focus[0]

        # Trim summary to fit in 150 chars
        answer = summary
        if len(answer) > 150:
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            trimmed = ''
            for s in sentences:
                candidate = (trimmed + ' ' + s).strip() if trimmed else s
                if len(candidate) <= 148:
                    trimmed = candidate
                else:
                    break
            answer = trimmed if trimmed else answer[:148] + '.'

        if len(answer) < 20:
            continue

        # Choose question template based on focus tags
        focus_set = set(focus)
        if 'guilt' in focus_set and 'confession' in focus_set:
            q = f"What does {speaker}'s private confession reveal in Hamlet?"
        elif 'guilt' in focus_set:
            q = f"What does this passage reveal about {speaker}'s guilt in Hamlet?"
        elif 'feigned madness' in focus_set or 'strategy' in focus_set:
            q = f"What does this passage reveal about Hamlet's feigned madness?"
        elif 'delay' in focus_set or 'hesitation' in focus_set:
            q = f"What does this scene show about Hamlet's delay in taking revenge?"
        elif 'mortality' in focus_set and speaker == 'Hamlet':
            q = f"What does Hamlet reflect on about mortality in this passage?"
        elif 'madness' in focus_set and speaker == 'Ophelia':
            q = f"What does this passage reveal about Ophelia's madness?"
        elif 'grief' in focus_set and speaker == 'Ophelia':
            q = f"How does Ophelia express her grief in this scene?"
        elif 'appearance versus reality' in focus_set or 'sincerity' in focus_set:
            q = f"How does this passage introduce the theme of appearance versus reality?"
        elif 'fate' in focus_set or 'acceptance' in focus_set:
            q = f"What attitude toward fate does Hamlet express before the final duel?"
        elif 'surveillance' in focus_set or 'distrust' in focus_set:
            q = f"How does this scene develop the theme of surveillance in Hamlet?"
        elif 'proof' in focus_set or 'exposure' in focus_set:
            q = f"How does this scene provide proof of Claudius's guilt?"
        elif 'tragedy' in focus_set and speaker == 'Horatio':
            q = f"What do Horatio's final words reveal about the tragedy of Hamlet?"
        elif 'death' in focus_set and 'closure' in focus_set:
            q = f"What is the significance of Hamlet's final words?"
        elif 'advice' in focus_set and speaker == 'Polonius':
            q = f"What kind of advice does Polonius give in this passage?"
        elif 'irony' in focus_set and speaker == 'Polonius':
            q = f"What is ironic about Polonius's speech on brevity?"
        elif 'theater' in focus_set or 'truth through performance' in focus_set:
            q = f"What does Hamlet's advice to the actors reveal about his character?"
        elif 'performance' in focus_set and speaker != 'Hamlet':
            q = f"How does the player's speech affect Hamlet in this scene?"
        elif 'self-criticism' in focus_set:
            q = f"How does Hamlet criticize himself in this soliloquy?"
        elif 'public image' in focus_set:
            q = f"How does Claudius present himself to the court after the old king's death?"
        elif 'caution' in focus_set and speaker == 'Laertes':
            q = f"What warning does Laertes give Ophelia about Hamlet?"
        elif 'revenge' in focus_set and 'contrast with hamlet' in ' '.join(focus).lower():
            q = f"How does Laertes contrast with Hamlet in his approach to revenge?"
        elif 'triumph' in focus_set:
            q = f"How does Hamlet feel after Claudius's reaction to the Mousetrap?"
        elif 'failed repentance' in focus_set:
            q = f"Why is Claudius unable to truly repent in this scene?"
        elif 'impulsive violence' in focus_set:
            q = f"What does Hamlet's killing of Polonius reveal about his character?"
        elif 'moral ambiguity' in focus_set:
            q = f"What does Hamlet mean when he says he must be cruel only to be kind?"
        elif 'crisis' in focus_set or 'accumulation of disaster' in focus_set:
            q = f"What does Claudius's speech after Polonius's death reveal?"
        elif 'despair' in focus_set:
            q = f"What state of mind does Hamlet express in the 'solid flesh' soliloquy?"
        elif 'human nature' in focus_set:
            q = f"What does Hamlet's 'piece of work is a man' speech express?"
        elif 'love' in focus_set and speaker == 'Hamlet' and 'grief' in focus_set:
            q = f"How does Hamlet declare his love for Ophelia at her grave?"
        elif 'revelation' in focus_set and speaker == 'Ghost':
            q = f"What does the Ghost reveal to Hamlet in this passage?"
        elif 'murder' in focus_set and 'betrayal' in focus_set:
            q = f"What does the Ghost tell Hamlet about the murder?"
        elif 'duty' in focus_set and 'burden' in focus_set:
            q = f"How does Hamlet react to learning he must avenge his father?"
        elif 'supernatural sign' in focus_set:
            q = f"What does Horatio's confrontation with the Ghost suggest?"
        elif 'family conflict' in focus_set and speaker == 'Gertrude':
            q = f"How does Gertrude react when Hamlet confronts her in the closet scene?"
        else:
            q = f"What happens in the scene where {speaker} speaks about {primary_focus} in Hamlet?"

        if len(q) <= 150:
            pairs.append([q, answer])

    return pairs


def generate_paraphrases(existing):
    """Generate paraphrase variants of existing Q&A pairs using pattern matching."""
    pairs = []
    existing_qs_lower = {q.lower().strip() for q, _ in existing}
    seen = set(existing_qs_lower)

    def try_add(q, a):
        key = q.lower().strip()
        if key not in seen and len(q) <= 150 and len(a) <= 150:
            seen.add(key)
            pairs.append([q, a])

    for question, answer in existing:
        # "Who is X?" → "Tell me about X." / "Describe X's character in Hamlet."
        m = re.match(r'^Who is (\w+)\?$', question)
        if m:
            name = m.group(1)
            try_add(f"Tell me about {name} in Hamlet.", answer)
            try_add(f"Describe {name}'s character in Hamlet.", answer)

        # "Who is Hamlet's X?" → "What is Hamlet's X's name?"
        m = re.match(r"^Who is Hamlet's (\w+)\?$", question)
        if m:
            rel = m.group(1)
            try_add(f"What is the name of Hamlet's {rel}?", answer)

        # "Who is X's Y?" → "Name X's Y in Hamlet."
        m = re.match(r"^Who is (\w+)'s (\w+)\?$", question)
        if m:
            person, rel = m.group(1), m.group(2)
            try_add(f"Name {person}'s {rel} in Hamlet.", answer)

        # "What happens to X?" → "What is X's fate in Hamlet?" / "How does X's story end?"
        m = re.match(r"^What happens to (\w+)\?$", question)
        if m:
            name = m.group(1)
            try_add(f"What is {name}'s fate in Hamlet?", answer)
            try_add(f"How does {name}'s story end in Hamlet?", answer)

        # "Why does X Y?" → "What causes X to Y?" / "For what reason does X Y?"
        m = re.match(r"^Why does (\w+) (.+)\?$", question)
        if m:
            name, action = m.group(1), m.group(2)
            try_add(f"What causes {name} to {action}?", answer)
            try_add(f"For what reason does {name} {action}?", answer)

        # "Why is X important in Hamlet?" → "What makes X significant in Hamlet?"
        m = re.match(r"^Why is (\w+) important(?: in Hamlet)?\?$", question)
        if m:
            name = m.group(1)
            try_add(f"What makes {name} significant in Hamlet?", answer)
            try_add(f"What is {name}'s importance in the play?", answer)

        # "What role does X play in Hamlet?" → "What is X's function in Hamlet?"
        m = re.match(r"^What role does (\w+) play in Hamlet\?$", question)
        if m:
            name = m.group(1)
            try_add(f"What is {name}'s function in Hamlet?", answer)

        # "What kind of person is X?" → "How would you describe X's character?"
        m = re.match(r"^What kind of person is (\w+)\?$", question)
        if m:
            name = m.group(1)
            try_add(f"How would you describe {name}'s character?", answer)
            try_add(f"What is {name}'s personality like in Hamlet?", answer)

        # "What is X's relation to Y?" → "How is X related to Y in Hamlet?"
        m = re.match(r"^What is (\w+'s) relation to (\w+)\?$", question)
        if m:
            a_part, b_part = m.group(1), m.group(2)
            try_add(f"How is {a_part} character related to {b_part} in Hamlet?", answer)

        # "What does X think about Y?" → "What are X's thoughts on Y?"
        m = re.match(r"^What does (\w+) think about (.+)\?$", question)
        if m:
            name, topic = m.group(1), m.group(2)
            try_add(f"What are {name}'s thoughts on {topic} in Hamlet?", answer)

        # "What themes appear in Hamlet?" → "What are the main themes of Hamlet?"
        if question == "What themes appear in Hamlet?":
            try_add("What are the main themes of Hamlet?", answer)
            try_add("What subjects does Hamlet explore?", answer)

        # "What does 'To be, or not to be' mean?" → "Explain the 'To be or not to be' soliloquy."
        if "To be" in question and "mean" in question:
            try_add("What is the meaning of Hamlet's 'To be or not to be' speech?", answer)

        # "What is the main conflict in Hamlet?" variants
        if "main conflict" in question:
            try_add("What is the central struggle in Hamlet?", answer)

    return pairs


# Hard-coded structural and factual pairs covering gaps in existing data
STRUCTURAL_QA = [
    # Acts and scenes
    ["In which act does Hamlet deliver the 'To be or not to be' soliloquy?",
     "Hamlet delivers the 'To be or not to be' soliloquy in Act 3, Scene 1."],
    ["In which act is Polonius killed?",
     "Polonius is killed in Act 3, Scene 4."],
    ["In which act does Ophelia go mad?",
     "Ophelia's madness is shown in Act 4."],
    ["In which act does the final duel take place?",
     "The final duel takes place in Act 5, Scene 2."],
    ["In which act does the Ghost appear to Hamlet?",
     "The Ghost appears to Hamlet in Act 1, Scene 5."],
    ["How many acts does Hamlet have?",
     "Hamlet has five acts."],

    # Deaths
    ["How does Ophelia die?",
     "Ophelia drowns in a stream after falling from a willow tree branch over the water."],
    ["How does Gertrude die?",
     "Gertrude dies after drinking poisoned wine that Claudius had prepared for Hamlet."],
    ["How does Laertes die?",
     "Laertes is wounded by his own poisoned sword during the duel with Hamlet."],
    ["How does Hamlet kill Claudius?",
     "Hamlet stabs Claudius with the poisoned sword and forces him to drink the poisoned wine."],
    ["How was King Hamlet murdered?",
     "King Hamlet was murdered by Claudius, who poured poison into his ear while he slept in the garden."],
    ["Who dies in the final scene of Hamlet?",
     "Gertrude, Laertes, Claudius, and Hamlet all die in the final scene."],

    # Minor characters
    ["Who are Rosencrantz and Guildenstern?",
     "Rosencrantz and Guildenstern are Hamlet's former schoolfellows who are sent by Claudius to spy on him."],
    ["What is Rosencrantz's role in Hamlet?",
     "Rosencrantz spies on Hamlet at Claudius's request alongside Guildenstern."],
    ["Who is Fortinbras?",
     "Fortinbras is the prince of Norway who arrives at the end of Hamlet to claim Denmark."],
    ["What role does Fortinbras play in Hamlet?",
     "Fortinbras arrives at the end and restores political order after the Danish court's collapse."],
    ["Who are the gravediggers in Hamlet?",
     "The gravediggers are two clowns who dig Ophelia's grave and joke about death in Act 5."],
    ["What is the gravedigger's significance in Hamlet?",
     "The gravedigger provides dark humor and prompts Hamlet to reflect on death and equality."],
    ["Who is Osric in Hamlet?",
     "Osric is a foolish courtier who delivers the challenge for the final duel to Hamlet."],
    ["Who is Marcellus in Hamlet?",
     "Marcellus is a guard who witnesses the Ghost with Horatio and speaks the famous 'rotten' line."],

    # Famous lines
    ["What are Hamlet's last words?",
     "Hamlet's last words are 'The rest is silence.'"],
    ["What does Horatio say when Hamlet dies?",
     "Horatio says 'Good night sweet prince: and flights of angels sing thee to thy rest.'"],
    ["Who says 'Something is rotten in the state of Denmark'?",
     "Marcellus says this in Act 1, expressing unease after seeing the Ghost."],
    ["What does 'The rest is silence' mean?",
     "It is Hamlet's final acceptance of death, suggesting nothing more needs to be said after all the violence."],
    ["Who says 'Neither a borrower nor a lender be'?",
     "Polonius says this as part of his advice to Laertes before he departs for France."],
    ["Who says 'Brevity is the soul of wit'?",
     "Polonius says this, ironically while speaking at great length."],
    ["Who says 'The lady doth protest too much'?",
     "Gertrude says this while watching the play within the play in Act 3."],
    ["Who says 'To thine own self be true'?",
     "Polonius says this to Laertes as part of his farewell advice before Laertes leaves for France."],
    ["Who says 'Good night sweet prince'?",
     "Horatio says these words as Hamlet dies, mourning his friend with deep tenderness."],
    ["Who says 'O, what a rogue and peasant slave am I'?",
     "Hamlet says this in a soliloquy after watching the player perform with genuine emotion."],
    ["Who says 'There is special providence in the fall of a sparrow'?",
     "Hamlet says this before the final duel, expressing his acceptance of fate."],

    # Plot mechanics
    ["What is the Mousetrap in Hamlet?",
     "The Mousetrap is the play Hamlet stages to test whether Claudius murdered his father."],
    ["What is the play within the play in Hamlet called?",
     "The play within the play is called 'The Mousetrap.'"],
    ["Why does Hamlet go to England?",
     "Claudius sends Hamlet to England with secret orders for his execution after Polonius's death."],
    ["What happens to Rosencrantz and Guildenstern in England?",
     "Rosencrantz and Guildenstern are executed after Hamlet replaces the order with one bearing their names."],
    ["What does Hamlet ask Horatio to do before dying?",
     "Hamlet asks Horatio to survive and tell his story truthfully to the world."],
    ["Why does Laertes use a poisoned sword in the duel?",
     "Laertes agrees to Claudius's plan to poison his sword blade to ensure Hamlet dies in the duel."],
    ["What is the significance of poison in Hamlet?",
     "Poison appears repeatedly as a symbol of corruption, betrayal, and the hidden evil at the heart of the court."],
    ["Why does Claudius try to pray?",
     "Claudius tries to pray out of guilt for murdering his brother, but he cannot truly repent."],
    ["What advice does Hamlet give the players?",
     "Hamlet tells the actors to speak naturally and avoid exaggeration to hold the mirror up to nature."],
    ["Where is Elsinore?",
     "Elsinore is the castle in Denmark where Hamlet takes place."],
    ["What is the significance of the ghost appearing on the battlements?",
     "The Ghost appearing at night on the battlements creates an atmosphere of danger and hidden political crisis."],
]


def generate_structural_questions():
    return [[q, a] for q, a in STRUCTURAL_QA]


def deduplicate(new_pairs, existing_pairs):
    """Remove new pairs whose questions duplicate existing ones (case-insensitive)."""
    existing_qs = {q.lower().strip() for q, _ in existing_pairs}
    seen = set()
    result = []
    for q, a in new_pairs:
        key = q.lower().strip()
        if key not in existing_qs and key not in seen:
            seen.add(key)
            result.append([q, a])
    return result


def validate_pair(q, a, max_answer_chars=150):
    return len(a) <= max_answer_chars and len(q) <= 150 and len(a) >= 10


def main():
    existing = json.loads(EXISTING_QA.read_text(encoding="utf-8"))
    blocks, backgrounds = parse_aligned_corpus(ALIGNED_CORPUS)
    print(f"Parsed {len(blocks)} passage blocks, {len(backgrounds)} background paragraphs")

    quote_pairs = generate_quote_attribution(blocks)
    print(f"  Quote attribution: {len(quote_pairs)} pairs")

    summary_pairs = generate_summary_factual(blocks)
    print(f"  Summary-based factual: {len(summary_pairs)} pairs")

    structural_pairs = generate_structural_questions()
    print(f"  Structural/hard-coded: {len(structural_pairs)} pairs")

    paraphrase_pairs = generate_paraphrases(existing)
    print(f"  Paraphrase variants: {len(paraphrase_pairs)} pairs")

    all_generated = quote_pairs + summary_pairs + structural_pairs + paraphrase_pairs
    valid = [[q, a] for q, a in all_generated if validate_pair(q, a)]
    print(f"\nAfter validation: {len(valid)} pairs")

    deduped = deduplicate(valid, existing)
    print(f"After deduplication against existing {len(existing)}: {len(deduped)} pairs")

    random.Random(42).shuffle(deduped)

    OUTPUT_QA.write_text(json.dumps(deduped, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(deduped)} pairs to {OUTPUT_QA}")


if __name__ == "__main__":
    main()
