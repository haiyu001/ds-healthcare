from enum import Enum


# opinion polarity
class Polarity(Enum):
    NEG = "NEG"
    POS = "POS"
    UNK = "UNK"


# aspect and opinion relations
class RelCategory(Enum):
    SUBJ = [
        "nsubj",       # nominal subject
        "nsubjpass",   # passive nominal subject
    ]
    MOD = [
        "amod",        # adjective modifier
        "appos",       # appositional modifier
        "neg",         # negation modifier
        "nmod",        # nominal modifier
        "acl",         # clausal modifier
        "relcl",       # relative clause modifier
        "advcl",       # adverbial clause modifier
        "advmod",      # adverbial modifier
        "npadvmod",    # noun phrase adverbial modifier
    ]
    OBJ = [
        "dobj",        # direct object
        "dative",      # indirect object
    ]


# direct dependency rules
class RuleType(Enum):
    O_O = "O_O"
    O_A = "O_A"
    A_O = "A_O"
    A_A = "A_A"
    O_X_O = "O_X_O"
    O_X_A = "O_X_A"
    A_X_O = "A_X_O"
    A_X_A = "A_X_A"


# map Spacy tag to ABSA POS
class POS(Enum):
    ADJ = ["JJ", "JJR", "JJS", "AFX"]
    ADV = ["RB", "RBR", "RBS"]
    CC = ["CC"]
    DET = ["DT", "PDT"]
    EX = ["EX"]
    FW = ["FW"]
    INTJ = ["UH"]
    LS = ["LS"]
    MD = ["MD"]
    NN = ["NN", "NNS"]
    NNP = ["NNP", "NNPS"]
    NUM = ["CD"]
    POS = ["POS"]
    PREP = ["IN", "TO"]
    RP = ["RP"]
    SYM = ["SYM"]
    VB = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    PRON = ["PRP$", "PRP"]
    WH_ADV = ["WRB"]
    WH_DET = ["WDT"]
    WH_PRON = ["WP$", "WP"]
    OTHER = ["``", "''", ",", "-LRB-", "-RRB-", ".", ":", "HYPH", "NFP", "SP", "_SP", "$", "ADD", "GW", "NIL", "XX"]