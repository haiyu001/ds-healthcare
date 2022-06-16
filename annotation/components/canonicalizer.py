from typing import Tuple, Dict

from lexicon.canonicalization_lexicon import prefixes, hyphen_trigram_blacklist, \
    prefix_trigram_base_word_pos, hyphen_trigram_two_words_pos, ampersand_trigram_word_pos, intensifiers
from annotation.tokenization.preprocessor import REPLACE_EMAIL, REPLACE_URL, REPLACE_HASHTAG, REPLACE_HANDLE
from spellchecker import SpellChecker
from utils.general_util import save_pdf, dump_json_file, split_filepath
import pandas as pd
import operator
import json
import os


def _get_bigram_canonicalization(unigram: str, bigram: str, unigram_count: int, bigram_count: int,
                                 spell_checker: SpellChecker) -> Tuple[str, str]:
    if unigram_count > bigram_count:
        return bigram, unigram
    elif unigram_count < bigram_count:
        return unigram, bigram
    elif unigram in spell_checker:
        return bigram, unigram
    else:
        return unigram, bigram


def _get_merge_and_replace_dict(trigram_words_to_count: Dict[Tuple[str], int],
                                unigram_to_count: Dict[str, int],
                                bigram_to_count: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    merge_dict, replace_dict = {}, {}
    for trigram_words, trigram_count in trigram_words_to_count.items():
        first_word, second_word = trigram_words[0], trigram_words[-1]
        unigram = first_word + second_word
        bigram = first_word + " " + second_word
        trigram = " ".join(trigram_words)
        unigram_count = unigram_to_count.get(unigram, 0)
        bigram_count = bigram_to_count.get(bigram, 0)
        if unigram_count > bigram_count and unigram_count > trigram_count:
            merge_dict[trigram] = unigram
            if bigram_count > 0:
                merge_dict[bigram] = unigram
        else:
            hyphen_word = "".join(trigram_words)
            merge_dict[trigram] = hyphen_word
            if bigram_count > 0:
                merge_dict[bigram] = hyphen_word
            if unigram_count > 0:
                replace_dict[unigram] = hyphen_word
    return merge_dict, replace_dict


def _load_unigram_bigram_trigram_dict(unigram_filepath: str,
                                      bigram_filepath: str,
                                      trigram_filepath: str,
                                      conjunction_trigram_canonicalization_filter_min_count: int) -> \
        Tuple[Dict[str, str], Dict[str, int], Dict[str, int], Dict[Tuple[str, str, str], int]]:
    vocab_pdf = pd.read_csv(unigram_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    bigram_pdf = pd.read_csv(bigram_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    trigram_pdf = pd.read_csv(trigram_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    vocab_pdf = vocab_pdf[vocab_pdf["count"] >= conjunction_trigram_canonicalization_filter_min_count]
    bigram_pdf = bigram_pdf[bigram_pdf["count"] >= conjunction_trigram_canonicalization_filter_min_count]
    trigram_pdf = trigram_pdf[trigram_pdf["count"] >= conjunction_trigram_canonicalization_filter_min_count]

    vocab_pdf["pos"] = vocab_pdf["top_three_pos"].apply(
        lambda x: max(json.loads(x).items(), key=operator.itemgetter(1))[0])
    vocab_to_pos = dict(zip(vocab_pdf["word"], vocab_pdf["pos"]))
    vocab_to_count = dict(zip(vocab_pdf["word"], vocab_pdf["count"]))
    bigram_to_count = dict(zip(bigram_pdf["ngram"], bigram_pdf["count"]))
    trigram_words_to_count = dict(zip(trigram_pdf["ngram"].apply(lambda x: tuple(x.split())), trigram_pdf["count"]))
    return vocab_to_pos, vocab_to_count, bigram_to_count, trigram_words_to_count


def _filter_canonicalization_dict(data_dict: Dict[str, Dict[str, str]],
                                  canonicalization_dict: Dict[str, str],
                                  canonicalization_type: str,
                                  canonicalization_source: str,
                                  case_insensitive: bool = True) -> Dict[str, Dict[str, str]]:
    filtered_canonicalization_dict = {}
    for canonicalization_key, canonicalization_value in canonicalization_dict.items():
        canonicalization_key_lower = canonicalization_key.lower()
        if canonicalization_key_lower not in data_dict:
            filtered_canonicalization_dict[canonicalization_key_lower] = {
                "key": canonicalization_key,
                "value": canonicalization_value,
                "type": canonicalization_type,
                "source": canonicalization_source,
                "case_insensitive": case_insensitive,
            }
    return filtered_canonicalization_dict


def _filter_preprocesor_replacement(data_dict: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    filtered_data_dict = {}
    for key, val in data_dict.items():
        if REPLACE_EMAIL.lower() not in key and REPLACE_URL not in key and \
                REPLACE_HASHTAG not in key and REPLACE_HANDLE not in key:
            filtered_data_dict[key] = val
    return filtered_data_dict


def get_bigram_and_spell_canonicalization(bigram_canonicalization_filepath: str,
                                          spell_canonicalization_filepath: str) -> \
        Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    spell_canonicalization_pdf = pd.read_csv(
        spell_canonicalization_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    bigram_canonicalization_pdf = pd.read_csv(
        bigram_canonicalization_filepath, encoding="utf-8", keep_default_na=False, na_values="")

    if not bigram_canonicalization_pdf.empty:
        bigram_canonicalization = dict(bigram_canonicalization_pdf.apply(
            lambda x: _get_bigram_canonicalization(x["unigram"], x["bigram"], x["unigram_count"], x["bigram_count"],
                                                   SpellChecker()), axis=1).tolist())
    else:
        bigram_canonicalization = {}
    unigram_to_bigram_count = dict(zip(bigram_canonicalization_pdf["unigram"],
                                       bigram_canonicalization_pdf["bigram_count"]))
    spell_replace_dict = dict(zip(spell_canonicalization_pdf["misspelling"],
                                  spell_canonicalization_pdf["correction"]))
    misspelling_to_correction_count = dict(zip(spell_canonicalization_pdf["misspelling"],
                                               spell_canonicalization_pdf["correction_count"]))
    common_words = set(spell_replace_dict.keys()).intersection(set(bigram_canonicalization.keys()))
    for common_word in common_words:
        if unigram_to_bigram_count[common_word] >= misspelling_to_correction_count[common_word]:
            del spell_replace_dict[common_word]
        else:
            del bigram_canonicalization[common_word]
    bigram_merge_dict = {k: v for k, v in bigram_canonicalization.items() if " " in k}
    bigram_split_dict = {k: v for k, v in bigram_canonicalization.items() if " " not in k}
    return bigram_merge_dict, bigram_split_dict, spell_replace_dict


def get_prefix_and_hyphen_canonicalization(unigram_to_pos: Dict[str, str],
                                           unigram_to_count: Dict[str, int],
                                           bigram_to_count: Dict[str, int],
                                           trigram_words_to_count: Dict[Tuple[str, str, str], int]) -> \
        Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    hyphen_words_to_count = {words: count for words, count in trigram_words_to_count.items() if words[1] == "-"}
    prefix_words_to_count = {words: count for words, count in hyphen_words_to_count.items() if words[0] in prefixes and
                             unigram_to_pos.get(words[-1]) in prefix_trigram_base_word_pos}
    hyphen_words_to_count = {words: count for words, count in hyphen_words_to_count.items() if
                             words[0] not in intensifiers and words[0] not in prefixes and
                             " ".join(words) not in hyphen_trigram_blacklist and
                             (unigram_to_pos[words[0]], unigram_to_pos[words[-1]]) in hyphen_trigram_two_words_pos}
    prefix_merge_dict, prefix_replace_dict = \
        _get_merge_and_replace_dict(prefix_words_to_count, unigram_to_count, bigram_to_count)
    hyphen_merge_dict, hyphen_replace_dict = \
        _get_merge_and_replace_dict(hyphen_words_to_count, unigram_to_count, bigram_to_count)
    return prefix_merge_dict, prefix_replace_dict, hyphen_merge_dict, hyphen_replace_dict


def get_ampersand_canonicalization(unigram_to_pos: Dict[str, str],
                                   trigram_words_to_count: Dict[Tuple[str, str, str], int]) -> Dict[str, str]:
    ampersand_merge_dict = {}
    ampersand_words_list = [words for words in trigram_words_to_count if words[1] == "&"
                            and unigram_to_pos.get(words[0]) in ampersand_trigram_word_pos and
                            unigram_to_pos.get(words[-1]) in ampersand_trigram_word_pos]
    for ampersand_words in ampersand_words_list:
        ampersand_word = "".join(ampersand_words)
        ampersand_merge_dict[" ".join(ampersand_words)] = ampersand_word
        and_trigram_words = (ampersand_words[0], "and", ampersand_words[-1])
        if and_trigram_words in trigram_words_to_count:
            ampersand_merge_dict[" ".join(and_trigram_words)] = ampersand_word
    return ampersand_merge_dict


def get_canonicalization(bigram_canonicalization_filepath: str,
                         spell_canonicalization_filepath: str,
                         unigram_filepath: str,
                         bigram_filepath: str,
                         trigram_filepath: str,
                         canonicalization_filepath: str,
                         conjunction_trigram_canonicalization_filter_min_count: int):

    merge_dict, split_dict, replace_dict = {}, {}, {}

    # TODO: add opinion multi word expression canonicalization

    # TODO: add multi word expression canonicalization

    # add prefix, hyphen and ampersand canonicalization
    unigram_to_pos, unigram_to_count, bigram_to_count, trigram_words_to_count = _load_unigram_bigram_trigram_dict(
        unigram_filepath, bigram_filepath, trigram_filepath, conjunction_trigram_canonicalization_filter_min_count)
    prefix_merge_dict, prefix_replace_dict, hyphen_merge_dict, hyphen_replace_dict = \
        get_prefix_and_hyphen_canonicalization(unigram_to_pos, unigram_to_count, bigram_to_count,
                                               trigram_words_to_count)
    ampersand_merge_dict = get_ampersand_canonicalization(unigram_to_pos, trigram_words_to_count)

    merge_dict.update(_filter_canonicalization_dict(merge_dict, prefix_merge_dict, "merge", "prefix", True))
    merge_dict.update(_filter_canonicalization_dict(merge_dict, hyphen_merge_dict, "merge", "hyphen", True))
    merge_dict.update(_filter_canonicalization_dict(merge_dict, ampersand_merge_dict, "merge", "ampersand", True))
    replace_dict.update(_filter_canonicalization_dict(replace_dict, prefix_replace_dict, "replace", "prefix", True))
    replace_dict.update(_filter_canonicalization_dict(replace_dict, hyphen_replace_dict, "replace", "hyphen", True))

    # add bigram and spell canonicalization
    bigram_merge_dict, bigram_split_dict, spell_replace_dict = \
        get_bigram_and_spell_canonicalization(bigram_canonicalization_filepath, spell_canonicalization_filepath)

    merge_dict.update(_filter_canonicalization_dict(merge_dict, bigram_merge_dict, "merge", "bigram", True))
    replace_dict.update(_filter_canonicalization_dict(replace_dict, spell_replace_dict, "replace", "spell", True))
    split_dict.update(_filter_canonicalization_dict(replace_dict, bigram_split_dict, "split", "bigram", True))

    # filter preprocessor replacement
    replace_dict = _filter_preprocesor_replacement(replace_dict)
    merge_dict = _filter_preprocesor_replacement(merge_dict)
    split_dict = _filter_preprocesor_replacement(split_dict)

    # dump canonicalization to json
    canonicalization_dict = {**replace_dict, **merge_dict, **split_dict}
    file_dir, file_name, file_format = split_filepath(canonicalization_filepath)
    dump_json_file(canonicalization_dict, canonicalization_filepath)

    # save canonicalization to csv
    canonicalization_csv_filepath = os.path.join(file_dir, f"{file_name}.csv")
    canonicalization_pdf = pd.DataFrame(canonicalization_dict).transpose().reset_index()
    canonicalization_pdf = canonicalization_pdf.rename(columns={"index": "id"})
    canonicalization_pdf = canonicalization_pdf.sort_values(by=["type", "source"])
    save_pdf(canonicalization_pdf, canonicalization_csv_filepath)
