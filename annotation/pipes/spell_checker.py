from typing import Tuple, Dict, Any, List, Set
from utils.general_util import load_json_file
from utils.resource_util import get_model_filepath
from hunspell.hunspell import HunspellWrap
from hunspell import Hunspell
from collections import defaultdict
from spacy.tokens import Token, Doc
import re


class SpellChecker(object):
    word_regex = "^[A-Za-z0-9-]{3,}$"

    def __init__(self, attrs: Tuple[str, str, str] = ("spell_is_correct", "suggest_spellings", "misspellings")):
        self._spell_is_correct, self._suggest_spellings, self._misspellings = attrs
        self.hunspell_checker = get_hunspell_checker()
        self.common_misspellings = self._load_common_misspellings()
        self.enwiktionary_words = self._load_enwiktionary_words()
        self.umls_lexicon_words = self._load_umls_lexicon_words()
        Token.set_extension(self._spell_is_correct, getter=self.spell_is_correct, force=True)
        Token.set_extension(self._suggest_spellings, getter=self.get_suggest_spellings, force=True)
        Doc.set_extension(self._misspellings, getter=self.get_misspellings, force=True)

    def __call__(self, doc: Doc) -> Doc:
        return doc

    def _load_common_misspellings(self) -> Dict[str, str]:
        common_misspellings_filepath = get_model_filepath("lexicon", "common_misspelling.json")
        return load_json_file(common_misspellings_filepath)

    def _load_enwiktionary_words(self) -> Set[str]:
        enwiktionary_filepath = get_model_filepath("lexicon", "enwiktionary.json")
        enwiktionary_list = load_json_file(enwiktionary_filepath)
        return set(enwiktionary_list)

    def _load_umls_lexicon_words(self) -> Set[str]:
        umls_lexicon_filepath = get_model_filepath("UMLS", "LEX", "LEXICON_WRD.json")
        umls_lexicon_dict = load_json_file(umls_lexicon_filepath)
        return set([i.lower() for i in umls_lexicon_dict])

    def _need_spell_checking(self, token: Token) -> bool:
        return re.match(self.word_regex, token.text) and not token.is_stop and \
               token.lower_ not in self.enwiktionary_words and token.lower_ not in self.umls_lexicon_words

    def spell_is_correct(self, token: Token) -> bool:
        if not self._need_spell_checking(token):
            return True
        try:
            spell = self.hunspell_checker.spell(token.text)
        except UnicodeEncodeError:
            spell = True
        return spell

    def get_suggest_spellings(self, token: Token) -> List[str]:
        if not self._need_spell_checking(token):
            return []
        if token.lower_ in self.common_misspellings:
            suggestions = [self.common_misspellings[token.lower_]]
        else:
            try:
                suggestions = self.hunspell_checker.suggest(token.text)
            except UnicodeEncodeError:
                suggestions = []
        return suggestions

    def get_misspellings(self, doc: Doc) -> List[Dict[str, Any]]:
        misspelled_texts = []
        misspelled_text_suggestions = {}
        misspelled_text_ids = defaultdict(list)
        for token in doc:
            if not self.spell_is_correct(token):
                suggestion = self.get_suggest_spellings(token)
                if suggestion:
                    token_text = token.text
                    if token_text not in misspelled_texts:
                        misspelled_texts.append(token_text)
                        misspelled_text_suggestions[token_text] = list(suggestion)
                    misspelled_text_ids[token_text].append(token.i)
        misspellings = [{"text": misspelled_text,
                         "suggestions": misspelled_text_suggestions[misspelled_text],
                         "ids": misspelled_text_ids[misspelled_text], }
                        for misspelled_text in misspelled_texts]
        return misspellings


def get_hunspell_checker() -> HunspellWrap:
    hunspell_dictionary_dir = get_model_filepath("model", "hunspell")
    return Hunspell(hunspell_data_dir=hunspell_dictionary_dir)
