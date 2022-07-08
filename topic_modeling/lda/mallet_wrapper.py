from typing import List, Tuple, Optional, Union, Iterator
from gensim import utils, matutils
from gensim.corpora import Dictionary
from gensim.models import basemodel
from gensim.models.ldamodel import LdaModel
from gensim.utils import check_output, revdict
from io import TextIOWrapper
from itertools import chain
import numpy as np
import logging


class LdaMallet(utils.SaveLoad, basemodel.BaseTopicModel):
    def __init__(self,
                 mallet_path: str,
                 corpus: List[Tuple[str, List[Tuple[int, int]]]],
                 id2word: Dictionary,
                 workers: int = 8,
                 iterations: int = 1000,
                 optimize_interval: int = 0,
                 num_topics: int = 100,
                 topic_alpha: float = 0.5,
                 prefix: Optional[str] = None,
                 topic_threshold: float = 0.0,
                 random_seed: int = 0):
        """
        :param mallet_path: path to the mallet binary, e.g. `/home/username/mallet-2.0.8/bin/mallet`
        :param corpus: Collection of doc id and texts in BoW format
        :param id2word: Mapping between tokens ids and words from corpus
        :param num_topics: number of topics
        :param topic_alpha: alpha parameter of LDA
        :param workers: number of threads that will be used for training
        :param optimize_interval: optimize hyperparameters every `optimize_interval` iterations
        :param iterations: number of training iterations
        :param topic_threshold: threshold of the probability above which we consider a topic
        :param random_seed: random seed to ensure consistent results, if 0 - use system clock
        :param prefix: prefix for produced temporary files
        """
        self.mallet_path = mallet_path
        self.id2word = id2word
        self.vocab_size = 1 + max(self.id2word.keys())
        self.workers = workers
        self.iterations = iterations
        self.optimize_interval = optimize_interval
        self.num_topics = num_topics
        self.topic_alpha = topic_alpha
        self.alpha = topic_alpha * num_topics
        self.topic_threshold = topic_threshold
        self.random_seed = random_seed
        self.prefix = prefix
        self.num_terms = max(self.id2word.keys()) + 1
        if corpus is not None:
            self.train(corpus)

    def get_state_filepath(self) -> str:
        """Get path to temporary file."""
        return self.prefix + "state.mallet.gz"

    def get_inferencer_filepath(self) -> str:
        """Get path to inferencer.mallet file."""
        return self.prefix + "inferencer.mallet"

    def get_topickeys_filepath(self) -> str:
        """Get path to topic keys text file."""
        return self.prefix + "topickeys.txt"

    def get_doctopics_filepath(self) -> str:
        """Get path to document topic text file."""
        return self.prefix + "doctopics.txt"

    def get_corpustxt_filepath(self) -> str:
        """Get path to corpus text file"""
        return self.prefix + "corpus.txt"

    def get_corpusmallet_filepath(self) -> str:
        """Get path to corpus.mallet file."""
        return self.prefix + "corpus.mallet"

    def __getitem__(self,
                    bow: Union[Tuple[str, List[Tuple[int, int]]], List[Tuple[str, List[Tuple[int, int]]]]],
                    iterations: int = 100) -> Union[List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
        """
        Get vector for document(s).
        :param bow: list of (int, int) if document OR list of list of (int, int) if corpus in BoW format
        :param iterations: number of iterations that will be used for inferring
        :return: list of (int, float) - LDA vector for document as sequence of (topic_id, topic_probability) OR
                 list of list of (int, float) - LDA vectors for corpus in same format
        """
        is_corpus = True
        if isinstance(bow, tuple):
            is_corpus = False
            bow = [bow]

        self.convert_input(bow, infer=True)  # will create corpusmallet infer file - "corpus.mallet.infer"

        cmd = f"{self.mallet_path} infer-topics " \
              f"--input {self.get_corpusmallet_filepath() + '.infer'} " \
              f"--inferencer {self.get_inferencer_filepath()} " \
              f"--output-doc-topics {self.get_doctopics_filepath() + '.infer'} " \
              f"--num-iterations {iterations} " \
              f"--doc-topics-threshold {self.topic_threshold} " \
              f"--random-seed {self.random_seed}"

        logging.info(f"inferring topics with MALLET LDA '{cmd}'")
        check_output(args=cmd, shell=True)

        result = list(self.read_doctopics(self.get_doctopics_filepath() + ".infer"))
        return result if is_corpus else result[0]

    def corpus2mallet(self, corpus: List[Tuple[str, List[Tuple[int, int]]]], file_like: TextIOWrapper):
        """
        Convert `corpus` to Mallet format and write it to `file_like` descriptor.
        Mallet Format : document id[SPACE]label (not used)[SPACE]whitespace delimited utf8-encoded tokens[NEWLINE]
        """
        for doc_id, doc in corpus:
            tokens = chain.from_iterable([self.id2word[tokenid]] * int(cnt) for tokenid, cnt in doc)
            file_like.write(utils.to_utf8(f"{doc_id} 0 {' '.join(tokens)}\n"))

    def convert_input(self,
                      corpus: List[Tuple[str, List[Tuple[int, int]]]],
                      infer: bool = False,
                      serialize_corpus: bool = True):
        """Convert corpus to Mallet format and save it to a temporary text file."""
        corpustxt_filepath = self.get_corpustxt_filepath() + ".infer" if infer else self.get_corpustxt_filepath()
        if serialize_corpus:
            logging.info(f"serializing temporary corpus to {corpustxt_filepath}")
            with utils.open(corpustxt_filepath, "wb") as fout:
                self.corpus2mallet(corpus, fout)
        # convert the text file above into MALLET"s internal format
        cmd = f"{self.mallet_path} import-file " \
              f"--preserve-case " \
              f"--keep-sequence " \
              f"--remove-stopwords " \
              f"--token-regex \"\\S+\" " \
              f"--input {corpustxt_filepath} " \
              f"--output {self.get_corpusmallet_filepath()}"
        if infer:
            cmd += f".infer "
            cmd += f"--use-pipe-from {self.get_corpusmallet_filepath()}"

        logging.info(f"converting temporary corpus to MALLET format with '{cmd}'")
        check_output(args=cmd, shell=True)

    def train(self, corpus: List[Tuple[str, List[Tuple[int, int]]]]):
        """Train Mallet LDA."""
        self.convert_input(corpus, infer=False)
        cmd = f"{self.mallet_path} train-topics " \
              f"--input {self.get_corpusmallet_filepath()} " \
              f"--num-topics {self.num_topics} " \
              f"--alpha {self.alpha} " \
              f"--optimize-interval {self.optimize_interval} " \
              f"--num-threads {self.workers} " \
              f"--output-state {self.get_state_filepath()} " \
              f"--output-doc-topics {self.get_doctopics_filepath()} " \
              f"--output-topic-keys {self.get_topickeys_filepath()} " \
              f"--num-iterations {self.iterations} " \
              f"--inferencer-filename {self.get_inferencer_filepath()} " \
              f"--doc-topics-threshold {self.topic_threshold} " \
              f"--random-seed {self.random_seed} "

        logging.info(f"training MALLET LDA with '{cmd}'")
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

    def load_word_topics(self) -> np.ndarray:
        """Load topics X words matrix from `get_state_filepath` file."""
        logging.info(f"loading assigned topics from {self.get_state_filepath()}")
        word_topics = np.zeros((self.num_topics, self.num_terms), dtype=np.float64)
        if hasattr(self.id2word, "token2id"):
            word2id = self.id2word.token2id
        else:
            word2id = revdict(self.id2word)

        with utils.open(self.get_state_filepath(), "rb") as fin:
            _ = next(fin)  # skip header
            self.alpha = np.fromiter(next(fin).split()[2:], dtype=float)
            assert len(self.alpha) == self.num_topics, "mismatch between MALLET vs. requested topics"
            _ = next(fin)  # noqa:F841 beta
            for lineno, line in enumerate(fin):
                line = utils.to_unicode(line)
                doc, source, pos, typeindex, token, topic = line.split(" ")
                if token not in word2id:
                    continue
                tokenid = word2id[token]
                word_topics[int(topic), tokenid] += 1.0
        return word_topics

    def get_topics(self) -> np.numarray:
        """Get topics X words matrix."""
        topics = self.word_topics
        return topics / topics.sum(axis=1)[:, None]

    def show_topic(self, topicid: int, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Get `num_words` most probable words for the given `topicid`.
        :param topicid: id of topic
        :param topn: top number of topics that you"ll receive.
        :return: list of (str, float) -
                 sequence of probable words, as a list of `(word, word_probability)` for `topicid` topic.
        """
        if self.word_topics is None:
            logging.warning("Run train or load_word_topics before showing topics.")
        topic = self.word_topics[topicid]
        topic = topic / topic.sum()  # normalize to probability dist
        bestn = matutils.argsort(topic, topn, reverse=True)
        beststr = [(self.id2word[idx], topic[idx]) for idx in bestn]
        return beststr

    def show_topics(self, num_topics: int = 10, num_words: int = 10, formatted: bool = True) -> \
            Union[List[Tuple[int, str]], List[Tuple[int, List[Tuple[str, float]]]]]:
        """
        Get the `num_words` most probable words for `num_topics` number of topics.
        :param num_topics: number of topics to return, set `-1` to get all topics
        :param num_words: number of words
        :param formatted: If `True` - return the topics as a list of strings, otherwise as lists of (weight, word) pairs
        :return: list of tuple (topic_id, topic_formatted_str) (if formatted=True) OR
                 list of tuple (topic id, list of tuple (weight, word)) (if formatted=False)
        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)
            # add a little random jitter, to randomize results around the same alpha
            sort_alpha = self.alpha + 0.0001 * np.random.rand(len(self.alpha))
            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[: num_topics // 2] + sorted_topics[-num_topics // 2:]
        shown = []
        for i in chosen_topics:
            if formatted:
                topic = self.print_topic(i, topn=num_words)
            else:
                topic = self.show_topic(i, topn=num_words)
            shown.append((i, topic))
        return shown

    def read_doctopics(self, doctopics_filepath: str, renorm: bool = True) -> Iterator[List[Tuple[int, float]]]:
        """
        Get document topic vectors from MALLET"s "doc-topics" format, as sparse gensim vectors.
        :param doctopics_filepath: path to input file with document topics
        :param eps: threshold for probabilities
        :param renorm: if `True` - explicitly re-normalize distribution
        :return: list of (int, float) - list of (topic_id, topic probability)
        """
        with utils.open(doctopics_filepath, "rb") as input:
            for lineno, line in enumerate(input):
                if lineno == 0 and line.startswith(b"#doc "):
                    continue  # skip the header line if it exists

                parts = line.split()[2:]  # skip "doc" and "source" columns
                if len(parts) == self.num_topics:
                    doc = [(topic_id, float(weight)) for topic_id, weight in enumerate(parts)]
                else:
                    raise RuntimeError(f"invalid doc topics format at line {lineno + 1} in {doctopics_filepath}")
                if renorm:
                    # explicitly normalize weights to sum up to 1.0, just to be sure...
                    total_weight = float(sum(weight for _, weight in doc))
                    if total_weight:
                        doc = [(id_, float(weight) / total_weight) for id_, weight in doc]
                yield doc

    def load_document_topics(self) -> Iterator[List[Tuple[int, float]]]:
        """Shortcut for `LdaMallet.read_doctopics`."""
        return self.read_doctopics(self.get_doctopics_filepath())

    @classmethod
    def load(cls, *args, **kwargs) -> LdaModel:
        """
        Load a previously saved LdaMallet class. Handles backwards compatibility from
        older LdaMallet versions which did not use random_seed parameter.
        """
        model = super().load(*args, **kwargs)
        if not hasattr(model, "random_seed"):
            model.random_seed = 0
        return model

    @classmethod
    def update_prefix(cls, model_filepath: str, prefix: str):
        model = cls.load(model_filepath)
        if model.prefix != prefix:
            model.prefix = prefix
        model.save(model_filepath)


def malletmodel2ldamodel(mallet_model: LdaMallet, gamma_threshold: float = 0.001, iterations: int = 50) -> LdaModel:
    """
    Convert :class:`LdaMallet` to :class:`~gensim.models.ldamodel.LdaModel`. This works by copying the training
    model weights (alpha, beta...) from a trained mallet model into the gensim model.
    :param mallet_model: Trained Mallet model
    :param gamma_threshold: to be used for inference in the new LdaModel
    :param iterations: number of iterations to be used for inference in the new LdaModel
    :return: :class:`~gensim.models.ldamodel.LdaModel` Gensim native LDA
    """
    model_gensim = LdaModel(
        id2word=mallet_model.id2word,
        num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha,
        eta=0,
        iterations=iterations,
        gamma_threshold=gamma_threshold,
        dtype=np.float64  # don"t loose precision when converting from MALLET
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim
