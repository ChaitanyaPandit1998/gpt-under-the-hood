"""
generate_shakespeare_qa.py

Template-based Q&A generation for all 37 Shakespeare plays.
No external API required.

Uses:
  - PLAY_FACTS: hand-curated structured data (characters, themes, plot points)
  - Template functions that generate question/answer pairs from this data
  - Wikipedia raw text for plays not in PLAY_FACTS (basic genre/setting questions)

Writes to:
  data/shakespeare_pipeline/qa/<slug>.json   (per-play)
  data/shakespeare_pipeline/qa/qa_index.json (manifest)

Usage:
    python scripts/generate_shakespeare_qa.py
    python scripts/generate_shakespeare_qa.py --play Hamlet   # single play
    python scripts/generate_shakespeare_qa.py --stats         # count only
"""

import argparse
import json
import re
import random
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent / "data"
PIPELINE_DIR = DATA_DIR / "shakespeare_pipeline"
RAW_DIR      = PIPELINE_DIR / "raw_wikipedia"
QA_DIR       = PIPELINE_DIR / "qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

MAX_ANSWER_CHARS = 150

# ── Structured play facts ─────────────────────────────────────────────────────
# Each entry:
#   genre, setting, characters: {name: description}, themes: [str],
#   plot_points: [(question, answer)], ending: str
PLAY_FACTS = {
    "Hamlet": {
        "genre": "tragedy", "setting": "Denmark",
        "characters": {
            "Hamlet":   "the Prince of Denmark and the play's protagonist",
            "Claudius": "Hamlet's uncle, the King of Denmark, and the play's villain",
            "Gertrude": "Hamlet's mother and the Queen of Denmark",
            "Ophelia":  "a noblewoman loved by Hamlet",
            "Horatio":  "Hamlet's loyal and trusted friend",
            "Laertes":  "Ophelia's brother and the son of Polonius",
            "Polonius": "a court adviser and the father of Ophelia and Laertes",
            "The Ghost": "the spirit of Hamlet's murdered father",
        },
        "relations": [
            ("Hamlet", "Claudius", "uncle and stepfather"),
            ("Hamlet", "Gertrude", "mother"),
            ("Hamlet", "Horatio", "loyal friend"),
            ("Ophelia", "Laertes", "brother"),
            ("Ophelia", "Polonius", "father"),
            ("Laertes", "Polonius", "father"),
        ],
        "themes": ["revenge", "madness", "mortality", "deception", "moral conflict"],
        "plot_points": [
            ("What does the Ghost tell Hamlet?",
             "The Ghost tells Hamlet that Claudius murdered his father by pouring poison in his ear."),
            ("Why does Hamlet delay taking revenge?",
             "Hamlet delays because he is morally conflicted, uncertain, and prone to deep reflection."),
            ("What is the play within the play?",
             "Hamlet stages a play re-enacting his father's murder to test Claudius's guilt."),
            ("What happens to Polonius?",
             "Hamlet kills Polonius by mistake, stabbing him through a curtain while he hides."),
            ("Why does Ophelia go mad?",
             "Ophelia goes mad from grief after her father's death and Hamlet's rejection."),
            ("What happens in the final duel?",
             "Hamlet and Laertes duel with a poisoned blade; both are wounded and die."),
            ("How does Claudius die?",
             "Hamlet forces Claudius to drink poisoned wine and stabs him with the poisoned sword."),
            ("Who survives at the end of Hamlet?",
             "Horatio survives to tell the story of Hamlet's tragedy."),
            ("What is the 'To be, or not to be' speech about?",
             "It is Hamlet's reflection on whether it is nobler to endure suffering or to end one's life."),
            ("What happens in the graveyard scene?",
             "Hamlet reflects on death and mortality while holding the skull of Yorick, a former jester."),
        ],
        "ending": "Hamlet, Laertes, Gertrude, and Claudius all die in the final scene.",
        "famous_line": "To be, or not to be, that is the question.",
        "famous_line_speaker": "Hamlet",
    },
    "Macbeth": {
        "genre": "tragedy", "setting": "Scotland",
        "characters": {
            "Macbeth":       "a Scottish general who murders the king to seize power",
            "Lady Macbeth":  "Macbeth's ambitious wife who urges him to kill Duncan",
            "Duncan":        "the King of Scotland, murdered by Macbeth",
            "Banquo":        "a general and Macbeth's friend, murdered on Macbeth's orders",
            "Macduff":       "a Scottish nobleman who ultimately kills Macbeth",
            "Malcolm":       "Duncan's son and the rightful heir to the throne",
            "The Witches":   "three supernatural figures who prophesy Macbeth's rise and fall",
        },
        "relations": [
            ("Macbeth", "Lady Macbeth", "wife"),
            ("Macbeth", "Banquo", "friend and fellow general"),
            ("Malcolm", "Duncan", "father"),
        ],
        "themes": ["ambition", "guilt", "power", "fate", "the supernatural"],
        "plot_points": [
            ("What do the witches prophesy to Macbeth?",
             "They prophesy that Macbeth will become King of Scotland."),
            ("How does Macbeth become king?",
             "Macbeth murders King Duncan in his sleep and blames the guards."),
            ("Why does Macbeth have Banquo killed?",
             "Macbeth kills Banquo because the witches prophesied that Banquo's descendants would be kings."),
            ("What is Lady Macbeth's role in the murder?",
             "Lady Macbeth plans the murder of Duncan and pressures Macbeth to carry it out."),
            ("What is Lady Macbeth's fate?",
             "Overwhelmed by guilt, Lady Macbeth loses her sanity and dies, likely by suicide."),
            ("What does Macbeth see at the banquet?",
             "Macbeth sees the ghost of Banquo sitting at the banquet table, which terrifies him."),
            ("Who kills Macbeth?",
             "Macduff kills Macbeth in single combat after Malcolm's army invades Scotland."),
            ("What is the significance of Birnam Wood?",
             "The witches said Macbeth would fall when Birnam Wood moved to Dunsinane, which happens when soldiers carry branches."),
        ],
        "ending": "Macbeth is killed by Macduff, and Malcolm is restored as the rightful King of Scotland.",
        "famous_line": "Out, damned spot!",
        "famous_line_speaker": "Lady Macbeth",
    },
    "Othello": {
        "genre": "tragedy", "setting": "Venice and Cyprus",
        "characters": {
            "Othello":    "a Moorish general in the Venetian army and the play's protagonist",
            "Iago":       "Othello's ensign who manipulates him out of jealousy and resentment",
            "Desdemona":  "Othello's wife, falsely accused of infidelity",
            "Cassio":     "Othello's lieutenant, framed by Iago as Desdemona's lover",
            "Emilia":     "Iago's wife and Desdemona's loyal attendant",
            "Roderigo":   "a Venetian gentleman manipulated by Iago",
            "Brabantio":  "Desdemona's father, who objects to her marriage to Othello",
        },
        "relations": [
            ("Othello", "Desdemona", "wife"),
            ("Iago", "Emilia", "wife"),
            ("Desdemona", "Brabantio", "father"),
            ("Othello", "Cassio", "lieutenant"),
            ("Othello", "Iago", "ensign"),
        ],
        "themes": ["jealousy", "manipulation", "race", "trust", "appearance and reality"],
        "plot_points": [
            ("Why does Iago hate Othello?",
             "Iago hates Othello because Othello promoted Cassio over him and he suspects Othello of sleeping with Emilia."),
            ("How does Iago manipulate Othello?",
             "Iago plants false evidence and whispers lies to make Othello believe Desdemona is unfaithful."),
            ("What is the significance of the handkerchief?",
             "Iago plants Othello's handkerchief with Cassio as false proof of Desdemona's infidelity."),
            ("How does Othello die?",
             "After killing Desdemona, Othello learns the truth and kills himself out of grief and guilt."),
            ("How does Desdemona die?",
             "Othello smothers Desdemona in their bed, convinced by Iago that she was unfaithful."),
            ("How is Iago exposed?",
             "Emilia reveals Iago's deception after Desdemona's murder, exposing his plot."),
        ],
        "ending": "Desdemona and Othello both die; Iago is arrested but shows no remorse.",
        "famous_line": "O, beware, my lord, of jealousy! It is the green-eyed monster.",
        "famous_line_speaker": "Iago",
    },
    "King Lear": {
        "genre": "tragedy", "setting": "Britain",
        "characters": {
            "King Lear":  "an aging king who divides his kingdom among his daughters",
            "Cordelia":   "Lear's youngest and most loyal daughter",
            "Goneril":    "Lear's eldest daughter, who betrays him after gaining power",
            "Regan":      "Lear's second daughter, equally treacherous",
            "Edmund":     "the illegitimate son of Gloucester who schemes against his brother",
            "Edgar":      "Gloucester's legitimate son, disguised as Poor Tom",
            "Gloucester": "a loyal earl blinded by Regan and Cornwall",
            "Kent":       "a loyal nobleman who serves Lear in disguise",
            "The Fool":   "Lear's court jester who speaks the truth through jokes",
        },
        "relations": [
            ("Cordelia", "King Lear", "father"),
            ("Goneril", "King Lear", "father"),
            ("Regan", "King Lear", "father"),
            ("Edmund", "Gloucester", "father"),
            ("Edgar", "Gloucester", "father"),
        ],
        "themes": ["power", "loyalty", "age and decline", "madness", "family betrayal"],
        "plot_points": [
            ("How does Lear divide his kingdom?",
             "Lear divides his kingdom among his three daughters based on how much they declare their love for him."),
            ("Why does Lear banish Cordelia?",
             "Lear banishes Cordelia because she refuses to flatter him and speaks plainly about her love."),
            ("What happens to Lear after he divides the kingdom?",
             "Goneril and Regan strip him of his knights and power, driving him mad on the storm-swept moors."),
            ("What does Edmund do to his father?",
             "Edmund betrays his father Gloucester to Cornwall, resulting in Gloucester being blinded."),
            ("How does Gloucester lose his eyes?",
             "Cornwall gouges out Gloucester's eyes as punishment for his loyalty to Lear."),
            ("What happens to Cordelia?",
             "Cordelia is hanged in prison on Edmund's secret orders after being captured."),
            ("How does Lear die?",
             "Lear dies of grief after discovering that Cordelia has been killed."),
        ],
        "ending": "Cordelia, Lear, Goneril, Regan, and Edmund all die; Edgar survives to rule.",
        "famous_line": "How sharper than a serpent's tooth it is to have a thankless child.",
        "famous_line_speaker": "King Lear",
    },
    "Romeo and Juliet": {
        "genre": "tragedy", "setting": "Verona, Italy",
        "characters": {
            "Romeo":           "a young man from the Montague family who falls in love with Juliet",
            "Juliet":          "a young woman from the Capulet family who falls in love with Romeo",
            "Friar Lawrence":  "a friar who secretly marries Romeo and Juliet",
            "Mercutio":        "Romeo's witty and loyal friend",
            "Tybalt":          "Juliet's hot-tempered cousin",
            "The Nurse":       "Juliet's loyal caretaker and confidante",
            "Lord Capulet":    "Juliet's father",
            "Benvolio":        "Romeo's cousin and peacemaker",
        },
        "relations": [
            ("Romeo", "Juliet", "wife"),
            ("Juliet", "Lord Capulet", "father"),
            ("Juliet", "Tybalt", "cousin"),
            ("Romeo", "Mercutio", "close friend"),
            ("Romeo", "Benvolio", "cousin"),
        ],
        "themes": ["love", "fate", "family conflict", "youth and impulsiveness", "death"],
        "plot_points": [
            ("Why can Romeo and Juliet not be together openly?",
             "Their families, the Montagues and the Capulets, are bitter enemies locked in an ancient feud."),
            ("How do Romeo and Juliet meet?",
             "They meet at a Capulet party, not knowing each other's family name."),
            ("Who kills Tybalt?",
             "Romeo kills Tybalt after Tybalt kills Mercutio."),
            ("What is Friar Lawrence's plan?",
             "Friar Lawrence gives Juliet a potion to fake her death so she can escape with Romeo."),
            ("Why does Romeo kill himself?",
             "Romeo finds Juliet apparently dead in the tomb and kills himself with poison in despair."),
            ("How does Juliet die?",
             "Juliet wakes to find Romeo dead and kills herself with his dagger."),
            ("What ends the feud between the families?",
             "The deaths of Romeo and Juliet reconcile the Montague and Capulet families."),
        ],
        "ending": "Both Romeo and Juliet die in the tomb; their deaths end the feud between their families.",
        "famous_line": "What's in a name? That which we call a rose by any other name would smell as sweet.",
        "famous_line_speaker": "Juliet",
    },
    "Julius Caesar": {
        "genre": "tragedy", "setting": "Rome",
        "characters": {
            "Julius Caesar":  "the Roman dictator whose assassination drives the plot",
            "Brutus":         "a senator who joins the conspiracy out of love for Rome",
            "Cassius":        "the cunning leader of the conspiracy against Caesar",
            "Mark Antony":    "Caesar's loyal friend who turns the people against the conspirators",
            "Casca":          "a senator who joins the conspiracy",
            "Portia":         "Brutus's devoted wife",
            "Calphurnia":     "Caesar's wife, who warns him of the dangers",
            "Octavius":       "Caesar's heir, who fights against the conspirators",
        },
        "relations": [
            ("Brutus", "Portia", "wife"),
            ("Caesar", "Calphurnia", "wife"),
            ("Caesar", "Mark Antony", "loyal friend"),
            ("Caesar", "Brutus", "trusted friend"),
        ],
        "themes": ["power", "betrayal", "rhetoric", "honour", "political ambition"],
        "plot_points": [
            ("Why do the conspirators kill Caesar?",
             "They fear Caesar's ambition will destroy the Roman Republic and make him a tyrant."),
            ("What are the Ides of March?",
             "The Ides of March is March 15th, the date of Caesar's assassination."),
            ("Who leads the conspiracy against Caesar?",
             "Cassius organises the conspiracy and persuades Brutus to join."),
            ("What does Mark Antony do after Caesar's death?",
             "Antony gives a speech at Caesar's funeral that turns the Roman mob against the conspirators."),
            ("How does Brutus die?",
             "Brutus kills himself after his defeat at the Battle of Philippi."),
            ("How does Cassius die?",
             "Cassius orders his servant to kill him after wrongly believing the battle is lost."),
        ],
        "ending": "Brutus and Cassius die; Antony and Octavius defeat the conspirators.",
        "famous_line": "Et tu, Brute?",
        "famous_line_speaker": "Julius Caesar",
    },
    "A Midsummer Night's Dream": {
        "genre": "comedy", "setting": "Athens and an enchanted forest",
        "characters": {
            "Puck":       "a mischievous fairy servant to Oberon",
            "Oberon":     "the king of the fairies",
            "Titania":    "the queen of the fairies",
            "Hermia":     "a young Athenian woman in love with Lysander",
            "Helena":     "Hermia's friend who loves Demetrius",
            "Lysander":   "a young man in love with Hermia",
            "Demetrius":  "a young man who pursues Hermia but is loved by Helena",
            "Bottom":     "a weaver who is temporarily transformed with a donkey's head",
            "Theseus":    "the Duke of Athens",
            "Hippolyta":  "the Queen of the Amazons, betrothed to Theseus",
        },
        "relations": [
            ("Hermia", "Lysander", "lover"),
            ("Helena", "Demetrius", "lover"),
            ("Oberon", "Titania", "wife"),
            ("Puck", "Oberon", "servant"),
        ],
        "themes": ["love", "magic", "dreams", "transformation", "illusion and reality"],
        "plot_points": [
            ("What is the love potion in the play?",
             "Puck applies a magical flower juice to sleeping characters, making them fall in love with whoever they see first."),
            ("Why does Puck apply the potion to the wrong person?",
             "Puck mistakes Lysander for Demetrius in the dark forest."),
            ("What happens to Bottom?",
             "Puck transforms Bottom's head into a donkey's head, and the enchanted Titania falls in love with him."),
            ("How is the confusion resolved?",
             "Oberon reverses the enchantments, restoring everyone's correct affections."),
            ("What conflict exists between Oberon and Titania?",
             "Oberon and Titania quarrel over custody of an orphaned Indian boy, causing discord in the fairy world."),
            ("Why do the lovers flee to the forest?",
             "Hermia flees with Lysander to escape an arranged marriage to Demetrius; Helena follows to win Demetrius back."),
            ("Who is Puck and what does he do?",
             "Puck is a mischievous fairy who serves Oberon and accidentally applies the love potion to the wrong Athenian."),
        ],
        "ending": "The lovers are properly paired, Bottom is restored, and all celebrate three weddings.",
        "famous_line": "The course of true love never did run smooth.",
        "famous_line_speaker": "Lysander",
    },
    "Much Ado About Nothing": {
        "genre": "comedy", "setting": "Messina, Sicily",
        "characters": {
            "Benedick":  "a witty soldier who swears he will never marry",
            "Beatrice":  "a sharp-tongued woman who claims to despise Benedick",
            "Claudio":   "a young soldier who falls in love with Hero",
            "Hero":      "a gentle noblewoman loved by Claudio",
            "Don John":  "the illegitimate brother of Don Pedro who plots against the lovers",
            "Don Pedro": "a prince who helps orchestrate the match between Benedick and Beatrice",
            "Dogberry":  "a bumbling but well-meaning constable",
            "Leonato":   "the Governor of Messina and Hero's father",
        },
        "relations": [
            ("Hero", "Leonato", "father"),
            ("Hero", "Beatrice", "cousin"),
            ("Benedick", "Claudio", "fellow soldier"),
        ],
        "themes": ["deception", "honour", "love and wit", "social expectation", "forgiveness"],
        "plot_points": [
            ("How are Benedick and Beatrice tricked into falling in love?",
             "Their friends stage conversations for each to overhear, convincing each that the other is secretly in love with them."),
            ("What does Don John do to Hero?",
             "Don John frames Hero as unfaithful by arranging for Claudio to see another woman at her window."),
            ("What happens when Claudio publicly shames Hero?",
             "Hero collapses and the Friar advises the family to pretend she has died."),
            ("How is the plot against Hero discovered?",
             "Dogberry's watchmen accidentally overhear Don John's accomplices and expose the plot."),
            ("How does Benedick react to overhearing that Beatrice loves him?",
             "Benedick decides to return Beatrice's affections and abandons his vow never to marry."),
            ("How does Beatrice react to overhearing that Benedick loves her?",
             "Beatrice resolves to return Benedick's love and surrender her pride against marriage."),
            ("What does Benedick do to prove his love for Beatrice?",
             "Benedick challenges his friend Claudio to a duel after Claudio shames Hero, siding with Beatrice."),
        ],
        "ending": "Don John is captured; Hero and Claudio reconcile and marry alongside Benedick and Beatrice.",
        "famous_line": "I do love nothing in the world so well as you.",
        "famous_line_speaker": "Benedick",
    },
    "The Merchant of Venice": {
        "genre": "comedy", "setting": "Venice and Belmont",
        "characters": {
            "Shylock":    "a Jewish moneylender who demands a pound of flesh as bond",
            "Portia":     "a wealthy heiress who disguises herself as a lawyer",
            "Antonio":    "a merchant who borrows money from Shylock for his friend",
            "Bassanio":   "Antonio's friend who wishes to court Portia",
            "Jessica":    "Shylock's daughter who elopes with Lorenzo",
            "Lorenzo":    "a Christian gentleman who elopes with Jessica",
            "Gratiano":   "a talkative friend of Bassanio",
            "Nerissa":    "Portia's waiting-maid",
        },
        "relations": [
            ("Antonio", "Bassanio", "close friend"),
            ("Portia", "Bassanio", "suitor and husband"),
            ("Jessica", "Shylock", "father"),
            ("Jessica", "Lorenzo", "husband"),
        ],
        "themes": ["justice and mercy", "prejudice", "wealth", "appearance and reality", "loyalty"],
        "plot_points": [
            ("What is Shylock's bond?",
             "Shylock loans Antonio money on the condition that if unpaid, he may take a pound of Antonio's flesh."),
            ("What is the casket test?",
             "Portia's suitors must choose between gold, silver, and lead caskets; the correct choice wins her hand."),
            ("How does Portia save Antonio?",
             "Disguised as a lawyer, Portia argues that Shylock can take flesh but not blood, making the bond impossible to fulfil."),
            ("What happens to Shylock?",
             "Shylock is forced to convert to Christianity and give half his estate to Antonio and Jessica."),
            ("Why does Antonio borrow money from Shylock?",
             "Antonio borrows money so his friend Bassanio can travel to Belmont to court Portia."),
            ("What happens to Jessica in The Merchant of Venice?",
             "Jessica elopes with Lorenzo, converts to Christianity, and takes a portion of her father's money."),
        ],
        "ending": "Antonio's ships return safely; Bassanio marries Portia; Shylock loses his daughter and wealth.",
        "famous_line": "The quality of mercy is not strained.",
        "famous_line_speaker": "Portia",
    },
    "Twelfth Night": {
        "genre": "comedy", "setting": "Illyria",
        "characters": {
            "Viola":      "a shipwrecked woman who disguises herself as the page Cesario",
            "Orsino":     "the Duke of Illyria, in love with Olivia",
            "Olivia":     "a countess who falls in love with Cesario (Viola in disguise)",
            "Malvolio":   "Olivia's pompous steward, tricked into thinking Olivia loves him",
            "Sir Toby":   "Olivia's boisterous uncle",
            "Feste":      "a wise court jester in Olivia's household",
            "Sebastian":  "Viola's twin brother, believed drowned",
            "Maria":      "Olivia's witty waiting-woman who tricks Malvolio",
        },
        "relations": [
            ("Viola", "Sebastian", "twin brother"),
            ("Olivia", "Sir Toby", "uncle"),
            ("Orsino", "Viola", "lover"),
        ],
        "themes": ["love", "disguise and identity", "gender", "folly", "class"],
        "plot_points": [
            ("Why does Viola disguise herself?",
             "Viola dresses as a man named Cesario to find employment in Orsino's court after her shipwreck."),
            ("How is Malvolio tricked?",
             "Maria writes a forged letter in Olivia's handwriting convincing Malvolio that Olivia loves him."),
            ("How is the love triangle resolved?",
             "Sebastian arrives and is mistaken for Cesario; Olivia marries Sebastian while Viola reveals herself to Orsino."),
            ("Who does Viola fall in love with?",
             "Viola, disguised as Cesario, falls in love with her employer Duke Orsino."),
            ("What happens to Malvolio at the end of Twelfth Night?",
             "Malvolio is humiliated, briefly imprisoned as a madman, and leaves vowing revenge on his tormentors."),
            ("What causes the comedy of mistaken identity in Twelfth Night?",
             "Viola's twin brother Sebastian arrives, and people mistake him for Cesario, causing comic confusion."),
        ],
        "ending": "Viola and Orsino marry, Olivia and Sebastian marry, and Malvolio leaves in humiliation.",
        "famous_line": "If music be the food of love, play on.",
        "famous_line_speaker": "Orsino",
    },
    "The Tempest": {
        "genre": "comedy", "setting": "a remote island",
        "characters": {
            "Prospero":   "the rightful Duke of Milan who uses magic to control the island",
            "Miranda":    "Prospero's daughter, raised on the island",
            "Ariel":      "a spirit who serves Prospero and longs for freedom",
            "Caliban":    "a native of the island, enslaved by Prospero",
            "Ferdinand":  "the Prince of Naples who falls in love with Miranda",
            "Gonzalo":    "an honest counsellor loyal to Prospero",
            "Antonio":    "Prospero's treacherous brother who usurped his dukedom",
            "Alonso":     "the King of Naples who helped Antonio",
        },
        "relations": [
            ("Prospero", "Miranda", "daughter"),
            ("Ariel", "Prospero", "master"),
            ("Caliban", "Prospero", "enslaved by"),
            ("Miranda", "Ferdinand", "husband"),
            ("Prospero", "Antonio", "brother who betrayed him"),
        ],
        "themes": ["power and control", "forgiveness", "freedom", "colonialism", "magic"],
        "plot_points": [
            ("How did Prospero lose his dukedom?",
             "Prospero's brother Antonio usurped his title as Duke of Milan while Prospero was absorbed in his studies."),
            ("Why does Prospero create the tempest?",
             "Prospero conjures the storm to shipwreck Antonio and the King of Naples on the island."),
            ("What does Prospero promise Ariel?",
             "Prospero promises Ariel freedom in return for service."),
            ("How does the play end?",
             "Prospero forgives his enemies, breaks his staff, and renounces his magic before sailing back to Milan."),
            ("What is Caliban's complaint against Prospero?",
             "Caliban claims the island was rightfully his by inheritance and that Prospero enslaved him."),
            ("What was Ariel's situation before Prospero arrived?",
             "Ariel was trapped inside a pine tree by the witch Sycorax and freed by Prospero in exchange for service."),
            ("What is the significance of Prospero breaking his staff?",
             "Prospero's act of breaking his staff and drowning his books represents a renunciation of power and magic."),
        ],
        "ending": "Prospero forgives his enemies and regains his dukedom; Ariel is freed.",
        "famous_line": "We are such stuff as dreams are made on.",
        "famous_line_speaker": "Prospero",
    },
    "As You Like It": {
        "genre": "comedy", "setting": "the Forest of Arden",
        "characters": {
            "Rosalind":   "the protagonist who disguises herself as the shepherd Ganymede",
            "Orlando":    "a young nobleman who falls in love with Rosalind",
            "Celia":      "Rosalind's devoted cousin",
            "Jaques":     "a melancholy lord famous for his philosophical speeches",
            "Duke Senior": "the rightful duke who lives in exile in the Forest of Arden",
            "Duke Frederick": "the usurping duke who banishes Rosalind",
            "Touchstone": "a witty court fool who accompanies Rosalind",
            "Oliver":     "Orlando's cruel elder brother",
        },
        "relations": [
            ("Rosalind", "Duke Senior", "father"),
            ("Rosalind", "Celia", "cousin"),
            ("Orlando", "Oliver", "brother"),
        ],
        "themes": ["love", "nature vs. court life", "disguise", "pastoral life", "self-discovery"],
        "plot_points": [
            ("Why does Rosalind disguise herself?",
             "Rosalind disguises herself as a young man named Ganymede after being banished by Duke Frederick."),
            ("What is Jaques famous for?",
             "Jaques is famous for his speech comparing human life to the seven ages of man."),
            ("How does the play end?",
             "Four couples marry, Duke Frederick converts and restores power, and Duke Senior returns from exile."),
            ("Why does Orlando go to the Forest of Arden?",
             "Orlando flees to the forest to escape his brother Oliver's plot to have him killed."),
            ("How does Orlando express his love for Rosalind?",
             "Orlando hangs love poems addressed to Rosalind on trees throughout the Forest of Arden."),
            ("What is the Forest of Arden in As You Like It?",
             "The Forest of Arden is an idealized pastoral refuge where the exiled court finds wisdom away from society."),
        ],
        "ending": "Multiple couples marry, the rightful duke is restored, and harmony is achieved.",
        "famous_line": "All the world's a stage, and all the men and women merely players.",
        "famous_line_speaker": "Jaques",
    },
    "Richard III": {
        "genre": "history", "setting": "England",
        "characters": {
            "Richard III":  "the Duke of Gloucester who manipulates his way to the throne",
            "Lady Anne":    "a noblewoman Richard woos and marries despite her hatred of him",
            "Buckingham":   "Richard's chief ally who eventually turns against him",
            "Richmond":     "the Earl of Richmond who defeats Richard at Bosworth",
            "Queen Margaret": "the former queen who curses Richard",
            "Clarence":     "Richard's brother, murdered on Richard's orders",
        },
        "relations": [
            ("Richard III", "Clarence", "brother"),
            ("Richard III", "Lady Anne", "wife"),
            ("Richard III", "Buckingham", "chief ally"),
        ],
        "themes": ["ambition", "tyranny", "deception", "fate", "guilt"],
        "plot_points": [
            ("How does Richard become king?",
             "Richard has his brother Clarence killed, manipulates events, and eliminates rivals until he is crowned."),
            ("How does Richard woo Lady Anne?",
             "Richard courts Lady Anne over the coffin of her father-in-law, whom Richard murdered."),
            ("What is Richard's downfall?",
             "Richard is haunted by the ghosts of his victims and is defeated and killed at the Battle of Bosworth Field."),
            ("Who kills Richard III?",
             "Richmond, later Henry VII, kills Richard III at the Battle of Bosworth Field."),
            ("Who are the Princes in the Tower?",
             "The young King Edward V and his brother Richard, Duke of York, are imprisoned in the Tower on Richard's orders."),
            ("How does Richard describe himself in Richard III?",
             "Richard describes himself as deformed and unfinished, using his physical difference to justify his villainy."),
        ],
        "ending": "Richmond kills Richard and becomes Henry VII, founding the Tudor dynasty.",
        "famous_line": "A horse! A horse! My kingdom for a horse!",
        "famous_line_speaker": "Richard III",
    },
    "Henry V": {
        "genre": "history", "setting": "England and France",
        "characters": {
            "Henry V":       "the King of England who leads the campaign to conquer France",
            "Katharine":     "the Princess of France who eventually marries Henry",
            "Fluellen":      "a loyal Welsh captain in Henry's army",
            "Pistol":        "a boastful soldier from Henry's past as Prince Hal",
            "Archbishop of Canterbury": "the churchman who supports Henry's claim to France",
            "Montjoy":       "the French herald",
            "The Chorus":    "a narrative figure who sets the scene",
        },
        "relations": [
            ("Henry V", "Katharine", "wife"),
        ],
        "themes": ["leadership", "war and honour", "nationalism", "kingship", "common humanity"],
        "plot_points": [
            ("Why does Henry invade France?",
             "Henry claims the French throne through his ancestry and invades to assert this right."),
            ("What happens at the Battle of Agincourt?",
             "Henry's vastly outnumbered English army defeats the French at Agincourt."),
            ("How does Henry win Katharine?",
             "After defeating France, Henry courts Katharine in a charming scene and wins her hand in marriage."),
            ("What is the St Crispin's Day speech in Henry V?",
             "Henry's speech before Agincourt rallies his outnumbered army, calling them a band of brothers."),
            ("How outnumbered is Henry's army at Agincourt?",
             "The English are vastly outnumbered by the French at Agincourt yet win a decisive victory."),
        ],
        "ending": "England wins at Agincourt; Henry marries Katharine, uniting the English and French crowns.",
        "famous_line": "We few, we happy few, we band of brothers.",
        "famous_line_speaker": "Henry V",
    },
    "Antony and Cleopatra": {
        "genre": "tragedy", "setting": "Rome and Egypt",
        "characters": {
            "Mark Antony":   "a Roman general in love with Cleopatra",
            "Cleopatra":     "the Queen of Egypt who loves Antony",
            "Octavius Caesar": "Antony's rival and the future Emperor Augustus",
            "Enobarbus":     "Antony's loyal lieutenant who eventually deserts him",
            "Octavia":       "Octavius's sister whom Antony marries for political reasons",
            "Charmian":      "Cleopatra's devoted attendant",
        },
        "relations": [
            ("Antony", "Cleopatra", "lover and queen"),
            ("Antony", "Octavius", "political rival"),
            ("Antony", "Octavia", "political wife"),
            ("Antony", "Enobarbus", "loyal lieutenant"),
        ],
        "themes": ["love vs. duty", "empire", "honour", "self-destruction", "political power"],
        "plot_points": [
            ("Why does Antony marry Octavia?",
             "Antony marries Octavia as a political alliance with Octavius, though his heart remains with Cleopatra."),
            ("How does Antony die?",
             "Antony stabs himself after receiving a false report that Cleopatra is dead, then dies in her arms."),
            ("How does Cleopatra die?",
             "Cleopatra kills herself with an asp to avoid being paraded as a captive in Rome."),
            ("What is the central conflict in Antony and Cleopatra?",
             "Antony must choose between his duty as a Roman triumvir and his love for Cleopatra, Queen of Egypt."),
            ("Why does Enobarbus desert Antony?",
             "Enobarbus deserts after Antony's judgment is compromised by passion, but dies of grief over his betrayal."),
        ],
        "ending": "Both Antony and Cleopatra die; Octavius becomes the sole ruler of Rome.",
        "famous_line": "Age cannot wither her, nor custom stale her infinite variety.",
        "famous_line_speaker": "Enobarbus",
    },
    "The Taming of the Shrew": {
        "genre": "comedy", "setting": "Padua, Italy",
        "characters": {
            "Katherina":  "a sharp-tongued woman known as 'the shrew'",
            "Petruchio":  "a man from Verona who sets out to marry and tame Katherina",
            "Bianca":     "Katherina's gentle younger sister",
            "Lucentio":   "a student who falls in love with Bianca",
            "Baptista":   "the wealthy father of Katherina and Bianca",
            "Grumio":     "Petruchio's comic servant",
        },
        "relations": [
            ("Katherina", "Baptista", "father"),
            ("Bianca", "Baptista", "father"),
            ("Katherina", "Bianca", "sister"),
            ("Petruchio", "Katherina", "wife"),
            ("Lucentio", "Bianca", "wife"),
        ],
        "themes": ["gender and power", "marriage", "appearance and reality", "social conformity"],
        "plot_points": [
            ("Why does Petruchio want to marry Katherina?",
             "Petruchio seeks a wealthy wife and is attracted to the challenge of Katherina's fierce spirit."),
            ("How does Petruchio tame Katherina?",
             "Petruchio uses sleep deprivation, hunger, and psychological tactics to wear down her defiance."),
            ("Who is Bianca in The Taming of the Shrew?",
             "Bianca is Katherina's gentle younger sister who cannot marry until Katherina has found a husband first."),
            ("What is the wager at the end of The Taming of the Shrew?",
             "Three husbands bet on whose wife is most obedient; Petruchio wins when Katherina comes at his call."),
            ("How does Lucentio woo Bianca?",
             "Lucentio disguises himself as a tutor to gain access to Bianca and court her while teaching."),
            ("Why must Bianca wait to marry in The Taming of the Shrew?",
             "Baptista will not allow Bianca to marry before her elder sister Katherina has found a husband."),
        ],
        "ending": "Katherina delivers a speech on wifely obedience; Petruchio wins a wager on her compliance.",
        "famous_line": "I come to wive it wealthily in Padua.",
        "famous_line_speaker": "Petruchio",
    },
    "The Winter's Tale": {
        "genre": "comedy", "setting": "Sicily and Bohemia",
        "characters": {
            "Leontes":    "the King of Sicily who is consumed by unfounded jealousy",
            "Hermione":   "Leontes's faithful wife, falsely accused of infidelity",
            "Perdita":    "Leontes's daughter, abandoned at birth and raised as a shepherdess",
            "Polixenes":  "the King of Bohemia, wrongly suspected by Leontes",
            "Florizel":   "Polixenes's son who falls in love with Perdita",
            "Paulina":    "a noblewoman who challenges Leontes and protects Hermione",
            "Antigonus":  "Paulina's husband, who abandons Perdita in Bohemia",
        },
        "relations": [
            ("Leontes", "Hermione", "wife"),
            ("Leontes", "Perdita", "daughter"),
            ("Perdita", "Florizel", "lover"),
            ("Paulina", "Antigonus", "husband"),
        ],
        "themes": ["jealousy", "forgiveness", "time", "redemption", "family"],
        "plot_points": [
            ("What does Leontes accuse Hermione of?",
             "Leontes falsely accuses Hermione of having an affair with his friend Polixenes."),
            ("What happens to Hermione?",
             "Hermione apparently dies of grief, but is revealed to have been hidden by Paulina."),
            ("How does the play end?",
             "After sixteen years, Perdita returns, Leontes repents, and Hermione is revealed to be alive."),
            ("Why is Leontes's jealousy striking?",
             "Leontes's jealousy is entirely self-generated with no real evidence of Hermione's infidelity."),
            ("What happens to the infant Perdita?",
             "Leontes orders Perdita abandoned; she is left in Bohemia where a shepherd finds and raises her."),
            ("What is the statue scene in The Winter's Tale?",
             "Paulina unveils what appears to be a statue of Hermione that comes to life, revealed as Hermione herself."),
        ],
        "ending": "Leontes is reunited with Perdita and Hermione after sixteen years of guilt and loss.",
        "famous_line": "A sad tale's best for winter.",
        "famous_line_speaker": "Mamillius",
    },
    "Measure for Measure": {
        "genre": "comedy", "setting": "Vienna",
        "characters": {
            "Isabella":   "a novice nun who pleads for her brother's life",
            "Angelo":     "the strict deputy ruler who hypocritically tries to coerce Isabella",
            "Claudio":    "Isabella's brother, sentenced to death for fornication",
            "The Duke":   "the Duke of Vienna who disguises himself as a friar",
            "Lucio":      "a witty but dissolute friend of Claudio",
            "Mariana":    "Angelo's former betrothed, used in the bed trick",
        },
        "relations": [
            ("Isabella", "Claudio", "brother"),
            ("Angelo", "Mariana", "former betrothed"),
        ],
        "themes": ["justice and mercy", "hypocrisy", "sexuality and morality", "power", "forgiveness"],
        "plot_points": [
            ("Why is Claudio condemned?",
             "Claudio is condemned to death by Angelo for getting his betrothed pregnant before marriage."),
            ("What does Angelo demand of Isabella?",
             "Angelo tells Isabella he will spare Claudio only if she sleeps with him."),
            ("What is the bed trick?",
             "The Duke arranges for Mariana to take Isabella's place in Angelo's bed at night."),
            ("How is Angelo punished?",
             "Angelo is forced to marry Mariana and is ultimately pardoned by the Duke."),
            ("Why does the Duke disguise himself in Measure for Measure?",
             "The Duke disguises himself as a friar to observe how his strict deputy Angelo governs Vienna."),
            ("What happens to Claudio at the end of Measure for Measure?",
             "Claudio is reprieved from execution and reunited with his betrothed Juliet when the Duke intervenes."),
        ],
        "ending": "The Duke pardons everyone; Angelo is forced to marry Mariana; the Duke proposes to Isabella.",
        "famous_line": "The law hath not been dead, though it hath slept.",
        "famous_line_speaker": "Angelo",
    },
    "Coriolanus": {
        "genre": "tragedy", "setting": "Rome and its enemies",
        "characters": {
            "Coriolanus":   "a great Roman general who despises the common people",
            "Volumnia":     "Coriolanus's proud and influential mother",
            "Virgilia":     "Coriolanus's gentle wife",
            "Menenius":     "an older senator and friend to Coriolanus",
            "Aufidius":     "the Volscian general, Coriolanus's great rival",
            "Sicinius":     "a tribune who turns the people against Coriolanus",
        },
        "relations": [
            ("Coriolanus", "Volumnia", "mother"),
            ("Coriolanus", "Virgilia", "wife"),
            ("Coriolanus", "Aufidius", "enemy and rival"),
        ],
        "themes": ["pride", "democracy vs. aristocracy", "war", "honour", "loyalty"],
        "plot_points": [
            ("Why is Coriolanus banished from Rome?",
             "Coriolanus is banished for his arrogance and contempt toward the common citizens of Rome."),
            ("What does Coriolanus do after exile?",
             "Coriolanus joins his enemy Aufidius and leads the Volscian army against Rome."),
            ("What stops Coriolanus from destroying Rome?",
             "Coriolanus spares Rome after his mother Volumnia pleads with him not to attack."),
            ("How does Coriolanus die?",
             "Aufidius has Coriolanus killed after accusing him of betrayal."),
            ("What does Coriolanus refuse to do for the Roman consulship?",
             "Coriolanus refuses to show his battle wounds to commoners and ask for their votes as tradition required."),
            ("Why is Coriolanus's pride his downfall?",
             "Coriolanus cannot humble himself before the people, making him easy to exile and ultimately leading to his death."),
        ],
        "ending": "Coriolanus is killed by Aufidius after sparing Rome at his mother's request.",
        "famous_line": "There is a world elsewhere.",
        "famous_line_speaker": "Coriolanus",
    },
}

# ── Generic facts for plays without full PLAY_FACTS entries ──────────────────
# Loaded from the raw Wikipedia JSON
MINOR_PLAY_INFO = {
    "Timon of Athens":            {"genre": "tragedy",  "setting": "Athens"},
    "Titus Andronicus":           {"genre": "tragedy",  "setting": "Rome"},
    "The Comedy of Errors":       {"genre": "comedy",   "setting": "Ephesus"},
    "Love's Labour's Lost":       {"genre": "comedy",   "setting": "Navarre"},
    "The Two Gentlemen of Verona":{"genre": "comedy",   "setting": "Verona and Milan"},
    "The Merry Wives of Windsor": {"genre": "comedy",   "setting": "Windsor, England"},
    "Richard II":                 {"genre": "history",  "setting": "England"},
    "Henry IV, Part 1":           {"genre": "history",  "setting": "England"},
    "Henry IV, Part 2":           {"genre": "history",  "setting": "England"},
    "Henry VI, Part 1":           {"genre": "history",  "setting": "England and France"},
    "Henry VI, Part 2":           {"genre": "history",  "setting": "England"},
    "Henry VI, Part 3":           {"genre": "history",  "setting": "England"},
    "Henry VIII":                 {"genre": "history",  "setting": "England"},
    "King John":                  {"genre": "history",  "setting": "England and France"},
    "All's Well That Ends Well":  {"genre": "comedy",   "setting": "France and Italy"},
    "Troilus and Cressida":       {"genre": "tragedy",  "setting": "Troy"},
    "Cymbeline":                  {"genre": "comedy",   "setting": "Britain and Rome"},
    "Pericles":                   {"genre": "comedy",   "setting": "the ancient Mediterranean"},
}


# ── Q&A template generators ───────────────────────────────────────────────────

def qa_genre(title, facts):
    genre = facts["genre"]
    return [
        (f"What genre is {title}?",
         f"{title} is a {genre} by William Shakespeare."),
        (f"What kind of play is {title}?",
         f"{title} is a {genre}."),
        (f"Did Shakespeare write {title}?",
         f"Yes, {title} was written by William Shakespeare."),
    ]


def qa_setting(title, facts):
    setting = facts["setting"]
    return [
        (f"Where is {title} set?",
         f"{title} is set in {setting}."),
        (f"What is the setting of {title}?",
         f"The setting of {title} is {setting}."),
        (f"In what location does {title} take place?",
         f"{title} takes place in {setting}."),
    ]


def qa_characters(title, facts):
    pairs = []
    for name, desc in facts.get("characters", {}).items():
        pairs += [
            (f"Who is {name} in {title}?",
             f"{name} is {desc}."),
            (f"What is {name}'s role in {title}?",
             f"{name} is {desc}."),
            (f"Describe {name} in {title}.",
             f"{name} is {desc}."),
        ]
    return pairs


def qa_relations(title, facts):
    pairs = []
    for person, other, relation in facts.get("relations", []):
        pairs += [
            (f"What is {person}'s relationship to {other} in {title}?",
             f"{person}'s {relation} is {other}."),
            (f"Who is {person}'s {relation} in {title}?",
             f"{person}'s {relation} is {other}."),
        ]
    return pairs


def qa_themes(title, facts):
    themes = facts.get("themes", [])
    if not themes:
        return []
    theme_str = ", ".join(themes[:-1]) + f", and {themes[-1]}" if len(themes) > 1 else themes[0]
    pairs = [
        (f"What are the main themes of {title}?",
         f"The main themes of {title} include {theme_str}."),
        (f"What does {title} explore?",
         f"{title} explores themes of {theme_str}."),
    ]
    for theme in themes[:3]:
        pairs.append((
            f"Is {theme} a theme in {title}?",
            f"Yes, {theme} is a central theme in {title}.",
        ))
    return pairs


def qa_plot_points(title, facts):
    return list(facts.get("plot_points", []))


def qa_ending(title, facts):
    ending = facts.get("ending", "")
    if not ending:
        return []
    genre = facts["genre"]
    pairs = [
        (f"How does {title} end?",
         ending),
        (f"What is the ending of {title}?",
         ending),
    ]
    if genre == "tragedy":
        pairs.append((
            f"Why is {title} a tragedy?",
            f"{title} is a tragedy because {ending.lower()}",
        ))
    return pairs


def qa_famous_line(title, facts):
    line = facts.get("famous_line", "")
    speaker = facts.get("famous_line_speaker", "")
    if not line or not speaker:
        return []
    # Truncate to fit answer limit
    short = line if len(line) <= 100 else line[:97] + "..."
    return [
        (f"Who says \"{short}\" in {title}?",
         f"{speaker} says this famous line in {title}."),
        (f"What is a famous line from {title}?",
         f'A famous line from {title} is: "{short}"'),
    ]


def qa_generic(title, info):
    """Basic Q&A for plays without full PLAY_FACTS (uses MINOR_PLAY_INFO)."""
    genre   = info.get("genre", "play")
    setting = info.get("setting", "")
    pairs   = [
        (f"Who wrote {title}?",
         f"{title} was written by William Shakespeare."),
        (f"What genre is {title}?",
         f"{title} is a {genre} by William Shakespeare."),
        (f"What kind of play is {title}?",
         f"{title} is a Shakespeare {genre}."),
    ]
    if setting:
        pairs += [
            (f"Where is {title} set?",
             f"{title} is set in {setting}."),
        ]
    return pairs


def generate_for_play(title: str, facts: dict | None, minor_info: dict | None) -> list[list[str]]:
    """Generate all Q&A for a single play."""
    pairs = []

    if facts:
        pairs += qa_genre(title, facts)
        pairs += qa_setting(title, facts)
        pairs += qa_characters(title, facts)
        pairs += qa_relations(title, facts)
        pairs += qa_themes(title, facts)
        pairs += qa_plot_points(title, facts)
        pairs += qa_ending(title, facts)
        pairs += qa_famous_line(title, facts)
    elif minor_info:
        pairs += qa_generic(title, minor_info)
    else:
        pairs += [
            (f"Who wrote {title}?",
             f"{title} was written by William Shakespeare."),
        ]

    # Deduplicate
    seen = set()
    unique = []
    for q, a in pairs:
        key = q.lower().strip()
        if key not in seen and len(a) <= MAX_ANSWER_CHARS:
            seen.add(key)
            unique.append([q, a])

    return unique


# ── Main ───────────────────────────────────────────────────────────────────────

def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play",  help="Process only this play title")
    parser.add_argument("--stats", action="store_true", help="Count pairs only")
    args = parser.parse_args()

    raw_files = sorted(RAW_DIR.glob("*.json"))
    if not raw_files:
        print("ERROR: No raw Wikipedia files found. Run fetch_wikipedia_shakespeare.py first.")
        return

    plays_data = {}
    for f in raw_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        plays_data[d["title"]] = d

    titles = list(plays_data.keys())
    if args.play:
        if args.play not in plays_data:
            print(f"ERROR: '{args.play}' not found. Available: {', '.join(sorted(titles))}")
            return
        titles = [args.play]

    qa_index   = []
    total_pairs = 0

    for title in sorted(titles):
        facts      = PLAY_FACTS.get(title)
        minor_info = MINOR_PLAY_INFO.get(title)

        pairs = generate_for_play(title, facts, minor_info)
        n     = len(pairs)
        total_pairs += n

        tier = plays_data[title].get("tier", "minor")
        source = "full" if facts else ("basic" if minor_info else "minimal")
        print(f"  {title:42s}  {n:3d} pairs  [{source}]")

        if not args.stats:
            out_path = QA_DIR / f"{slug(title)}.json"
            result = {
                "title":   title,
                "tier":    tier,
                "genre":   plays_data[title].get("genre", "unknown"),
                "pairs":   pairs,
                "n_pairs": n,
            }
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        qa_index.append({"title": title, "file": f"{slug(title)}.json",
                         "status": "ok", "n_pairs": n})

    if not args.stats:
        idx_path = QA_DIR / "qa_index.json"
        idx_path.write_text(json.dumps(qa_index, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Index saved: {idx_path}")

    print(f"\n--- Summary ---")
    print(f"  Plays:       {len(titles)}")
    print(f"  Total pairs: {total_pairs}")
    if not args.stats:
        print(f"\nRun build_sft_dataset.py to merge into train/val split.")


if __name__ == "__main__":
    main()
