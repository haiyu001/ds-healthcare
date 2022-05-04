sentiment_negation = {
    "barely": "both",
    "hardly": "after",
    "lack": "before",
    "least": "before",
    "neither": "before",
    "never": "before",
    "nor": "before",
    "not": "before",
    "n't": "before",
    "without": "before",
    "non": "before",
    "no": "before",
    "nt": "before",
    "aint": "before",
    "cannot": "before",
    "cant": "before",
    "dont": "before",
    "arent": "before",
    "couldnt": "before",
    "didnt": "before",
    "darent": "before",
    "hadnt": "before",
    "hasnt": "before",
    "isnt": "before",
    "mightnt": "before",
    "mustnt": "before",
    "neednt": "before",
    "oughtnt": "before",
    "shant": "before",
    "shouldnt": "before",
    "wasnt": "before",
    "werent": "before",
    "wont": "before",
    "wouldnt": "before",
}

sentiment_negation_social = {**sentiment_negation, **{
    "none": "before",
    "nope": "before",
    "nothing": "before",
    "nowhere": "before",
    "uhuh": "before",
    "uh-uh": "before",
    "rarely": "before",
    "seldom": "before",
    "despite": "before",
}}

sentiment_pseudo = {
    "never so": "before",
    "never this": "before",
    "without doubt": "before",
    "at least": "before",
    "very least": "before",
    "not only": "before",
}

proposition_pseudo = [
    "no further",
    "not able to be",
    "not certain if",
    "not certain whether",
    "not necessarily",
    "without any further",
    "without difficulty",
    "without further",
    "might not",
    "not only",
    "no increase",
    "no significant change",
    "no change",
    "no definite change",
    "not extend",
    "not cause",
]

proposition_preceding = [
    "absence of",
    "declined",
    "denied",
    "denies",
    "denying",
    "no sign of",
    "no signs of",
    "not",
    "not demonstrate",
    "symptoms atypical",
    "doubt",
    "negative for",
    "no",
    "versus",
    "without",
    "doesn't",
    "doesnt",
    "don't",
    "dont",
    "didn't",
    "didnt",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "cannot",
    "can't",
    "cant",
    "couldn't",
    "couldnt",
    "never",
]

proposition_following = [
    "declined",
    "unlikely",
    "was not",
    "were not",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
]

proposition_termination = [
    "although",
    "apart from",
    "as there are",
    "aside from",
    "but",
    "except",
    "however",
    "involving",
    "nevertheless",
    "still",
    "though",
    "which",
    "yet",
]

proposition_chunk_prefix = [
    "no",
]

pseudo_clinical = proposition_pseudo + [
    "gram negative",
    "not rule out",
    "not ruled out",
    "not been ruled out",
    "not drain",
    "no suspicious change",
    "no interval change",
    "no significant interval change",
]

proposition_preceding_clinical = proposition_preceding + [
    "patient was not",
    "without indication of",
    "without sign of",
    "without signs of",
    "without any reactions or signs of",
    "no complaints of",
    "no evidence of",
    "no cause of",
    "evaluate for",
    "fails to reveal",
    "free of",
    "never developed",
    "never had",
    "did not exhibit",
    "rules out",
    "rule out",
    "rule him out",
    "rule her out",
    "rule patient out",
    "rule the patient out",
    "ruled out",
    "ruled him out",
    "ruled her out",
    "ruled patient out",
    "ruled the patient out",
    "r/o",
    "ro",
    "lack of",           # from UMLS NEG LEXICON
    "exclude",           # from UMLS NEG LEXICON
    "excluding",         # from UMLS NEG LEXICON
    "nonexistence",      # from UMLS NEG LEXICON
    "inexistent",        # from UMLS NEG LEXICON
    "infeasible",        # from UMLS NEG LEXICON
]

proposition_following_clinical = proposition_following + [
    "was ruled out",
    "were ruled out",
    "free",
    "disappear",         # from UMLS NEG LEXICON
]

proposition_termination_clinical = proposition_termination + [
    "cause for",
    "cause of",
    "causes for",
    "causes of",
    "etiology for",
    "etiology of",
    "origin for",
    "origin of",
    "origins for",
    "origins of",
    "other possibilities of",
    "reason for",
    "reason of",
    "reasons for",
    "reasons of",
    "secondary to",
    "source for",
    "source of",
    "sources for",
    "sources of",
    "trigger event for",
]

proposition_preceding_clinical_sensitive = proposition_preceding_clinical + [
    "concern for",
    "supposed",
    "which causes",
    "leads to",
    "h/o",
    "history of",
    "instead of",
    "if you experience",
    "if you get",
    "teaching the patient",
    "taught the patient",
    "teach the patient",
    "educated the patient",
    "educate the patient",
    "educating the patient",
    "monitored for",
    "monitor for",
    "test for",
    "tested for",
]