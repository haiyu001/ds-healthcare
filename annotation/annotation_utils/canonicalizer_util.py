prefixes = [
    "re",
    "dis",
    "over",
    "un",
    "mis",
    "out",
    "co",
    "de",
    "fore",
    "inter",
    "pre",
    "sub",
    "trans",
    "under",
    "anti",
    "auto",
    "bi",
    "counter",
    "ex",
    "hyper",
    "kilo",
    "mal",
    "mega",
    "mini",
    "mono",
    "neo",
    "poly",
    "pseudo",
    "semi",
    "sur",
    "tele",
    "tri",
    "ultra",
    "vice",
    "non",
    # "super",  # consider "super" as intensifier
]

prefix_trigram_base_word_pos = [
    "ADJ",
    "ADV",
    "VERB",
    "NOUN",
]

ampersand_trigram_word_pos = [
    "ADJ", 
    "NOUN", 
    "PROPN",
]

hyphen_trigram_two_words_pos = [
    ("NOUN", "NOUN"),
    ("NOUN", "ADJ"),
    ("ADJ", "ADJ"),
    ("ADV", "ADJ"),
    ("ADV", "VERB")
]

hyphen_trigram_blacklist = [
    "day - old",
    "days - old",
    "week - old",
    "weeks - old",
    "month - old",
    "months - old",
    "year - old",
    "years - old",
]