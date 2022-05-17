from enum import Enum


class Polarity(Enum):
    NEG = "NEG"
    POS = "POS"
    UNK = "UNK"


class RelCategory(Enum):
    SUBJ = ["nsubj", "nsubjpass", "csubj", "csubjpass"]
    MOD = ["amod", "acl", "appos", "neg", "nmod", "advcl", "npadvmod", "advmod", "relcl"]
    OBJ = ["dobj", "dative"]


class RuleType(Enum):
    O_O = "O_O"
    O_X_O = "O_X_O"
    A_O = "A_O"
    A_X_O = "A_X_O"
    O_A = "O_A"
    O_X_A = "O_X_A"
    A_A = "A_A"
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