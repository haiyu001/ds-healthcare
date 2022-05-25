from typing import Tuple, Dict, List
from word_vector.wv_space import ConceptNetWordVec, load_txt_vecs_to_pdf
from double_propagation.sentiment_subjectivity.model_building import get_sentiment_features_pdf, \
    get_model_prediction_pdf
from scipy.stats import hmean
import pandas as pd
import collections
import operator
import random
import json
import re


def _normalize_underscore_ampersand_lower_lemma(lower_lemma: str, word_to_lemma: Dict[str, str]) -> str:
    if "_" in lower_lemma or "&" in lower_lemma:
        words = re.split(r"_|&", lower_lemma)
        separator = "_" if "_" in lower_lemma else "&"
        lower_lemma = separator.join([word_to_lemma.get(word, word) for word in words])
    return lower_lemma


def _get_dom_pos(text: str, word_to_pos: Dict[str, str]) -> str:
    if " " in text:
        text = text.split()[-1]
    return word_to_pos[text.lower()]


def _get_noun_phrases_ids(pos_list: List[str], noun_phrase_max_words_count: int = 4) -> List[Tuple[int, int]]:
    res = []
    noun_propn_ids = [i for i, pos in enumerate(pos_list) if pos == "NOUN" or pos == "PROPN"]
    size = len(noun_propn_ids)
    if size >= 2:
        start = end = noun_propn_ids[0]
        i = 1
        while i <= size:
            if i != size and noun_propn_ids[i] == noun_propn_ids[i - 1] + 1:
                end = noun_propn_ids[i]
            else:
                if 2 <= end - start + 1 <= noun_phrase_max_words_count:
                    res.append((start, end + 1))
                start = end = noun_propn_ids[i] if i < size else -1
            i += 1
    return res


def load_word_to_dom_lemma_and_pos(unigram_filepath: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    unigram_pdf = pd.read_csv(unigram_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    unigram_pdf["top_three_lemma"] = unigram_pdf["top_three_lemma"].apply(json.loads)
    unigram_pdf["lemma"] = unigram_pdf["top_three_lemma"].apply(lambda x: max(x.items(), key=operator.itemgetter(1))[0])
    unigram_pdf["top_three_pos"] = unigram_pdf["top_three_pos"].apply(json.loads)
    unigram_pdf["pos"] = unigram_pdf["top_three_pos"].apply(lambda x: max(x.items(), key=operator.itemgetter(1))[0])
    word_to_dom_lemma = dict(zip(unigram_pdf["word"], unigram_pdf["lemma"]))
    word_to_dom_pos = dict(zip(unigram_pdf["word"], unigram_pdf["pos"]))
    return word_to_dom_lemma, word_to_dom_pos


def get_noun_phrases_pdf(phrase_filepath: str, noun_phrase_words_max_count: int = 4) -> pd.DataFrame:
    phrase_pdf = pd.read_csv(phrase_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    phrase_pdf["phrase_words"] = phrase_pdf["phrase"].str.split()
    phrase_pdf["phrase_poses"] = phrase_pdf["phrase_poses"].apply(json.loads)
    phrase_pdf["phrase_lemmas"] = phrase_pdf["phrase_lemmas"].apply(json.loads)
    phrase_pdf["phrase_deps"] = phrase_pdf["phrase_deps"].apply(json.loads)
    noun_phrase_to_count = collections.defaultdict(int)
    noun_phrase_to_lemma_list = collections.defaultdict(list)
    noun_phrase_to_dep_list = collections.defaultdict(list)
    for _, row in phrase_pdf.iterrows():
        phrase_words, phrase_poses, phrase_lemmas, phrase_deps = row["phrase_words"], row["phrase_poses"], \
                                                                 row["phrase_lemmas"], row["phrase_deps"]
        for start, end in _get_noun_phrases_ids(phrase_poses, noun_phrase_words_max_count):
            noun_phrase = tuple(phrase_words[start: end])
            noun_phrase_to_count[noun_phrase] += row["count"]
            noun_phrase_to_lemma_list[noun_phrase].append(tuple(phrase_lemmas[start: end]))
            noun_phrase_to_dep_list[noun_phrase].append(tuple(phrase_deps[start: end]))
    noun_phrase_to_word_lemmas = {phrase: collections.Counter(lemma_list).most_common(1)[0][0]
                                  for phrase, lemma_list in noun_phrase_to_lemma_list.items()}
    noun_phrase_to_word_deps = {phrase: collections.Counter(dep_list).most_common(1)[0][0]
                                for phrase, dep_list in noun_phrase_to_dep_list.items()}
    noun_phrase_record_list = []
    for noun_phrase in noun_phrase_to_count:
        word_lemmas, word_deps = noun_phrase_to_word_lemmas[noun_phrase], noun_phrase_to_word_deps[noun_phrase]
        if word_deps[0] == "amod" or word_deps[0] == "nummod":
            continue
        noun_phrase_words = []
        for i, word_lemma in enumerate(word_lemmas):
            if word_lemma.istitle():
                noun_phrase_words.append(noun_phrase[i].title())
            elif word_lemma.isupper():
                noun_phrase_words.append(noun_phrase[i].upper())
            elif word_lemma.islower():
                noun_phrase_words.append(noun_phrase[i].lower())
            else:
                noun_phrase_words.append(word_lemma if len(word_lemma) == len(noun_phrase[i]) else noun_phrase[i])
        noun_phrase_record_list.append({
            "noun_phrase": " ".join(noun_phrase_words),
            "lemma": " ".join(word_lemmas),
            "deps": word_deps,
            "count": noun_phrase_to_count[noun_phrase],
        })
    noun_phrases_pdf = pd.DataFrame(noun_phrase_record_list).sort_values(by="count", ascending=False)
    return noun_phrases_pdf


def get_aspect_merge_pdf(aspect_candidates_filepath: str,
                         aspect_ranking_vecs_filepath: str,
                         word_to_dom_lemma: Dict[str, str],
                         word_to_dom_pos: Dict[str, str],
                         aspect_opinion_num_samples: int) -> pd.DataFrame:
    aspect_candidates_pdf = pd.read_csv(
        aspect_candidates_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    aspect_candidates_pdf["samples"] = aspect_candidates_pdf["samples"].apply(json.loads)
    aspect_candidates_pdf["lower_lemma"] = aspect_candidates_pdf["lemma"].str.lower()
    aspect_candidates_pdf["lower_lemma"] = aspect_candidates_pdf["lower_lemma"].apply(
        _normalize_underscore_ampersand_lower_lemma, word_to_lemma=word_to_dom_lemma)
    aspect_lemma_merge_list = []
    for lower_lemma, aspect_group_pdf in aspect_candidates_pdf.groupby("lower_lemma"):
        members = aspect_group_pdf["text"].tolist()
        text = sorted(members, key=len)[0]
        count = sum(aspect_group_pdf["count"])
        samples = sum(aspect_group_pdf["samples"].tolist(), [])
        samples = random.sample(samples, min(aspect_opinion_num_samples, len(samples)))
        non_lower_lemmas = [lemma for lemma in aspect_group_pdf["lemma"] if not lemma.islower()]
        lemma = non_lower_lemmas[0] if non_lower_lemmas else lower_lemma
        aspect_lemma_merge_list.append({"text": text,
                                        "members": json.dumps(members, ensure_ascii=False),
                                        "count": count,
                                        "lemma": lemma,
                                        "samples": json.dumps(samples, ensure_ascii=False)})
    aspect_lemma_merge_pdf = pd.DataFrame(aspect_lemma_merge_list)
    aspect_lemma_merge_pdf["pos"] = aspect_lemma_merge_pdf["text"].str.lower().str.split().str[-1] \
        .apply(lambda x: word_to_dom_pos[x])
    aspect_lemma_merge_pdf["index"] = ["_".join(i.lower().split()) for i in aspect_lemma_merge_pdf["text"]]
    aspect_lemma_merge_pdf = aspect_lemma_merge_pdf.set_index("index")

    conceptnet_vecs_filepath = get_model_filepath("model", "conceptnet", "numberbatch-en-19.08.txt")
    conceptnet_wordvec = ConceptNetWordVec(conceptnet_vecs_filepath, use_oov_strategy=True)
    aspects = aspect_lemma_merge_pdf.index.tolist()
    conceptnet_wordvec.extract_txt_vecs(aspects, aspect_ranking_vecs_filepath)
    concreteness_features_pdf = load_txt_vecs_to_pdf(aspect_ranking_vecs_filepath)
    concreteness_model_filepath = get_model_filepath("model", "concreteness", "concreteness.hdf5")
    aspect_subjectivity_scores_pdf = get_model_prediction_pdf(concreteness_features_pdf, concreteness_model_filepath,
                                                              predicted_score_col="concreteness_score")
    aspect_merge_pdf = pd.concat([aspect_lemma_merge_pdf, aspect_subjectivity_scores_pdf], axis=1)
    aspect_merge_pdf = aspect_merge_pdf[["text", "count", "concreteness_score", "pos", "lemma", "members", "samples"]]
    aspect_merge_pdf = aspect_merge_pdf.sort_values(by="count", ascending=False)
    return aspect_merge_pdf


def save_aspect_ranking_pdf(aspect_candidates_filepath: str,
                            aspect_ranking_vecs_filepath: str,
                            aspect_ranking_filepath: str,
                            phrase_filepath: str,
                            word_to_dom_lemma: Dict[str, str],
                            word_to_dom_pos: Dict[str, str],
                            aspect_opinion_num_samples: int,
                            noun_phrase_min_count: int,
                            noun_phrase_max_words_count: int):
    noun_phrases_pdf = get_noun_phrases_pdf(phrase_filepath, noun_phrase_max_words_count)
    noun_phrases_pdf = noun_phrases_pdf[noun_phrases_pdf["count"] >= noun_phrase_min_count]
    noun_phrase_lemma_to_noun_phrase = dict(zip(noun_phrases_pdf["lemma"], noun_phrases_pdf["noun_phrase"]))

    aspect_merge_pdf = get_aspect_merge_pdf(aspect_candidates_filepath,
                                            aspect_ranking_vecs_filepath,
                                            word_to_dom_lemma,
                                            word_to_dom_pos,
                                            aspect_opinion_num_samples)
    aspect_ranking_pdf = aspect_merge_pdf.reset_index()
    aspect_ranking_pdf["members"] = aspect_ranking_pdf["members"].apply(json.loads)
    all_members = {member.lower() for members in aspect_ranking_pdf["members"] for member in members}
    expanded_noun_phrases = set()
    aspect_noun_phrases_list = []
    for i, row in aspect_ranking_pdf.iterrows():
        aspect_noun_phrases = []
        for noun_phrase_lemma, noun_phrase in noun_phrase_lemma_to_noun_phrase.items():
            if noun_phrase_lemma.endswith(row["lemma"]) and noun_phrase.lower() not in all_members and \
                    noun_phrase not in expanded_noun_phrases:
                aspect_noun_phrases.append(noun_phrase)
                expanded_noun_phrases.add(noun_phrase)
        aspect_noun_phrases_list.append(json.dumps(aspect_noun_phrases, ensure_ascii=False)
                                        if aspect_noun_phrases else None)
    aspect_ranking_pdf["noun_phrases"] = pd.Series(aspect_noun_phrases_list)
    aspect_ranking_pdf["members"] = aspect_ranking_pdf["members"].apply(json.dumps, ensure_ascii=False)
    aspect_ranking_pdf = aspect_ranking_pdf[["text", "count", "concreteness_score", "pos",
                                        "lemma", "members", "noun_phrases", "samples"]]
    save_pdf(aspect_ranking_pdf, aspect_ranking_filepath)


def save_opinion_ranking_pdf(opinion_candidates_filepath: str,
                             opinion_ranking_vecs_filepath: str,
                             opinion_ranking_filepath: str,
                             word_to_dom_lemma: Dict[str, str],
                             word_to_dom_pos: Dict[str, str]):
    opinion_candidates_pdf = pd.read_csv(
        opinion_candidates_filepath, encoding="utf-8", keep_default_na=False, na_values="")
    opinion_candidates_pdf["lemma"] = opinion_candidates_pdf["text"].apply(lambda x: word_to_dom_lemma[x])
    opinion_candidates_pdf["pos"] = opinion_candidates_pdf["text"].apply(lambda x: word_to_dom_pos[x])
    opinion_candidates_pdf = opinion_candidates_pdf.set_index("text")
    opinions = opinion_candidates_pdf.index.tolist()
    # extract opinion vecs
    conceptnet_vecs_filepath = get_model_filepath("model", "conceptnet", "numberbatch-en-19.08.txt")
    conceptnet_wordvec = ConceptNetWordVec(conceptnet_vecs_filepath, use_oov_strategy=True)
    conceptnet_wordvec.extract_txt_vecs(opinions, opinion_ranking_vecs_filepath)
    # run sentiment model prediction
    sentiment_features_pdf = get_sentiment_features_pdf(opinion_ranking_vecs_filepath)
    sentiment_model_filepath = get_model_filepath("model", "sentiment", "sentiment.hdf5")
    opinion_sentiment_scores_pdf = get_model_prediction_pdf(sentiment_features_pdf, sentiment_model_filepath,
                                                            predicted_score_col="sentiment_score")
    # run subjectivity model prediction
    subjectivity_features_pdf = load_txt_vecs_to_pdf(opinion_ranking_vecs_filepath)
    subjectivity_model_filepath = get_model_filepath("model", "subjectivity", "subjectivity.hdf5")
    opinion_subjectivity_scores_pdf = get_model_prediction_pdf(subjectivity_features_pdf, subjectivity_model_filepath,
                                                               predicted_score_col="subjectivity_score")
    # get opinion sentiment subjectivity scores
    opinion_sentiment_subjectivity_scores_pdf = \
        opinion_sentiment_scores_pdf.merge(opinion_subjectivity_scores_pdf, on="word").rename(columns={"word": "text"})
    opinion_sentiment_subjectivity_scores_pdf["hmean_score"] = hmean(opinion_sentiment_subjectivity_scores_pdf, axis=1)

    opinion_ranking_pdf = pd.concat([opinion_sentiment_subjectivity_scores_pdf,
                                  sentiment_features_pdf[["neg_avg", "pos_avg"]],
                                  opinion_candidates_pdf], axis=1)
    opinion_ranking_pdf = opinion_ranking_pdf[
        ["count", "polarity", "neg_avg", "pos_avg", "sentiment_score", "subjectivity_score", "hmean_score",
         "lemma", "pos", "samples"]]
    opinion_ranking_pdf = opinion_ranking_pdf.sort_values(by="hmean_score", ascending=False)
    save_pdf(opinion_ranking_pdf, opinion_ranking_filepath, csv_index_label="text", csv_index=True)


if __name__ == "__main__":
    from utils.general_util import setup_logger, save_pdf, make_dir, dump_json_file
    from utils.config_util import read_config_to_dict
    from utils.resource_util import get_repo_dir, get_data_filepath, get_model_filepath
    import os

    setup_logger()

    absa_config_filepath = os.path.join(get_repo_dir(), "double_propagation", "pipelines", "conf/absa_template.cfg")
    absa_config = read_config_to_dict(absa_config_filepath)

    domain_dir = get_data_filepath(absa_config["domain"])
    absa_dir = os.path.join(domain_dir, absa_config["absa_folder"])
    annotation_dir = os.path.join(domain_dir, absa_config["annotation_folder"])
    extraction_dir = os.path.join(domain_dir, absa_config["extraction_folder"])
    absa_aspect_dir = make_dir(os.path.join(absa_dir, "aspect"))
    absa_opinion_dir = make_dir(os.path.join(absa_dir, "opinion"))
    aspect_candidates_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_candidates_filename"])
    opinion_candidates_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_candidates_filename"])
    aspect_ranking_vecs_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_ranking_vecs_filename"])
    aspect_ranking_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_ranking_filename"])
    opinion_ranking_vecs_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_ranking_vecs_filename"])
    opinion_ranking_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_ranking_filename"])
    unigram_filepath = os.path.join(extraction_dir, absa_config["unigram_filename"])
    phrase_filepath = os.path.join(extraction_dir, absa_config["phrase_filename"])

    word_to_dom_lemma, word_to_dom_pos = load_word_to_dom_lemma_and_pos(unigram_filepath)

    save_aspect_ranking_pdf(aspect_candidates_filepath,
                            aspect_ranking_vecs_filepath,
                            aspect_ranking_filepath,
                            phrase_filepath,
                            word_to_dom_lemma,
                            word_to_dom_pos,
                            absa_config["aspect_opinion_num_samples"],
                            absa_config["noun_phrase_min_count"],
                            absa_config["noun_phrase_max_words_count"])

    save_opinion_ranking_pdf(opinion_candidates_filepath,
                             opinion_ranking_vecs_filepath,
                             opinion_ranking_filepath,
                             word_to_dom_lemma,
                             word_to_dom_pos)
