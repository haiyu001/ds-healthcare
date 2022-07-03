from typing import List, Tuple, Optional, Union
from gensim import utils, matutils
from gensim.corpora import Dictionary
from gensim.models import basemodel
from gensim.models.ldamodel import LdaModel
from gensim.utils import check_output, revdict
from io import TextIOWrapper
from itertools import chain
import numpy as np
import logging
import tempfile
import random
import os


class LdaMallet(utils.SaveLoad, basemodel.BaseTopicModel):
    def __init__(self,
                 mallet_path: str,
                 corpus: List[List[Tuple[int, int]]],
                 id2word: Dictionary,
                 num_topics: int = 100,
                 alpha: float = 50.0,
                 workers: int = 8,
                 optimize_interval: int = 0,
                 iterations: int = 1000,
                 topic_threshold: float = 0.0,
                 random_seed: int = 0,
                 prefix: Optional[str] = None):
        """
        :param mallet_path: path to the mallet binary, e.g. `/home/username/mallet-2.0.8/bin/mallet`
        :param corpus: Collection of texts in BoW format
        :param id2word: Mapping between tokens ids and words from corpus
        :param num_topics: number of topics
        :param alpha: alpha parameter of LDA
        :param workers: number of threads that will be used for training
        :param optimize_interval: optimize hyperparameters every `optimize_interval` iterations
        :param iterations: number of training iterations
        :param topic_threshold: threshold of the probability above which we consider a topic
        :param random_seed: random seed to ensure consistent results, if 0 - use system clock
        :param prefix: prefix for produced temporary files
        """
        self.mallet_path = mallet_path
        self.corpus = corpus
        self.id2word = id2word
        self.vocab_size = 1 + max(self.id2word.keys())
        self.num_topics = num_topics
        self.alpha = alpha
        self.workers = workers
        self.optimize_interval = optimize_interval
        self.iterations = iterations
        self.topic_threshold = topic_threshold
        self.random_seed = random_seed
        if prefix is None:
            rand_prefix = hex(random.randint(0, 0xffffff))[2:] + "_"
            prefix = os.path.join(tempfile.gettempdir(), rand_prefix)
        self.prefix = prefix

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
                    bow: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]],
                    iterations: int = 100) -> Union[List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
        """
        Get vector for document(s).
        :param bow: list of (int, int) if document OR list of list of (int, int) if corpus in BoW format
        :param iterations: number of iterations that will be used for inferring
        :return: list of (int, float) - LDA vector for document as sequence of (topic_id, topic_probability) OR
                 list of list of (int, float) - LDA vectors for corpus in same format
        """
        is_corpus, corpus = utils.is_corpus(bow)
        if not is_corpus:
            # query is a single document => make a corpus out of it
            bow = [bow]

        self.convert_input(bow, infer=True)  # will create corpusmallet infer file - "corpus.mallet.infer"
        cmd = self.mallet_path + " infer-topics --input %s --inferencer %s " \
                                 "--output-doc-topics %s --num-iterations %s --doc-topics-threshold %s --random-seed %s"
        cmd = cmd % (self.get_corpusmallet_filepath() + ".infer",
                     self.get_inferencer_filepath(),
                     self.get_doctopics_filepath() + ".infer",
                     iterations,
                     self.topic_threshold,
                     str(self.random_seed))
        logging.info("inferring topics with MALLET LDA '%s'", cmd)
        check_output(args=cmd, shell=True)
        result = list(self.read_doctopics(self.get_doctopics_filepath() + ".infer"))
        return result if is_corpus else result[0]

    def corpus2mallet(self, corpus: List[List[Tuple[int, int]]], file_like: TextIOWrapper):
        """
        Convert `corpus` to Mallet format and write it to `file_like` descriptor.
        Mallet Format : document id[SPACE]label (not used)[SPACE]whitespace delimited utf8-encoded tokens[NEWLINE]
        """
        for doc_id, doc in enumerate(corpus):
            tokens = chain.from_iterable([self.id2word[tokenid]] * int(cnt) for tokenid, cnt in doc)
            file_like.write(utils.to_utf8(f"{doc_id} 0 {' '.join(tokens)}\n"))

    def convert_input(self, corpus: List[List[Tuple[int, int]]], infer: bool = False, serialize_corpus: bool = True):
        """Convert corpus to Mallet format and save it to a temporary text file."""
        if serialize_corpus:
            logging.info("serializing temporary corpus to %s", self.get_corpustxt_filepath())
            with utils.open(self.get_corpustxt_filepath(), "wb") as fout:
                self.corpus2mallet(corpus, fout)
        # convert the text file above into MALLET"s internal format
        cmd = self.mallet_path + \
              " import-file --preserve-case --keep-sequence " \
              "--remove-stopwords --token-regex \"\\S+\" --input %s --output %s"
        if infer:
            cmd += " --use-pipe-from " + self.get_corpusmallet_filepath()
            cmd = cmd % (self.get_corpustxt_filepath(),
                         self.get_corpusmallet_filepath() + ".infer")
        else:
            cmd = cmd % (self.get_corpustxt_filepath(),
                         self.get_corpusmallet_filepath())
        logging.info("converting temporary corpus to MALLET format with %s", cmd)
        check_output(args=cmd, shell=True)

    def train(self, corpus: List[List[Tuple[int, int]]]):
        """Train Mallet LDA."""
        self.convert_input(corpus, infer=False)
        cmd = self.mallet_path + \
              " train-topics --input %s --num-topics %s --alpha %s --optimize-interval %s " \
              "--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s " \
              "--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s --random-seed %s"

        cmd = cmd % (
            self.get_corpusmallet_filepath(), self.num_topics, self.alpha, self.optimize_interval,
            self.workers, self.get_state_filepath(), self.get_doctopics_filepath(), self.get_topickeys_filepath(),
            self.iterations,
            self.get_inferencer_filepath(), self.topic_threshold, str(self.random_seed)
        )
        logging.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

    def load_word_topics(self) -> np.ndarray:
        """Load topics X words matrix from `get_state_filepath` file."""
        logging.info("loading assigned topics from %s", self.get_state_filepath())
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

    def show_topics(self, num_topics: int = 10, num_words: int = 10, log: bool = False, formatted: bool = True) -> \
            Union[List[str], List[Tuple[float, int]]]:
        """
        Get the `num_words` most probable words for `num_topics` number of topics.
        :param num_topics: number of topics to return, set `-1` to get all topics
        :param iterations: number of words
        :param log: if `True` - write topic with logging too, used for debug proposes
        :param formatted: If `True` - return the topics as a list of strings, otherwise as lists of (weight, word) pairs
        :return: list of str - Topics as a list of strings (if formatted=True) OR
                 list of (float, str) - Topics as list of (weight, word) pairs (if formatted=False)
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
            if log:
                logging.info("topic #%i (%.3f): %s", i, self.alpha[i], topic)
        return shown

    def read_doctopics(self, doctopics_filepath: str, eps: float = 1e-6, renorm: bool = True) -> List[Tuple[int, float]]:
        """
        Get document topic vectors from MALLET"s "doc-topics" format, as sparse gensim vectors.
        :param doctopics_filepath: path to input file with document topics
        :param eps: threshold for probabilities
        :param renorm: if `True` - explicitly re-normalize distribution
        :return: list of (int, float) - LDA vectors for document
        """
        with utils.open(doctopics_filepath, "rb") as fin:
            for lineno, line in enumerate(fin):
                if lineno == 0 and line.startswith(b"#doc "):
                    continue  # skip the header line if it exists
                parts = line.split()[2:]  # skip "doc" and "source" columns
                if len(parts) == 2 * self.num_topics:
                    doc = [(int(id_), float(weight)) for id_, weight in zip(*[iter(parts)] * 2)
                           if abs(float(weight)) > eps]
                elif len(parts) == self.num_topics:
                    doc = [(id_, float(weight)) for id_, weight in enumerate(parts) if abs(float(weight)) > eps]
                else:
                    raise RuntimeError("invalid doc topics format at line %i in %s" % (lineno + 1, doctopics_filepath))
                if renorm:
                    # explicitly normalize weights to sum up to 1.0, just to be sure...
                    total_weight = float(sum(weight for _, weight in doc))
                    if total_weight:
                        doc = [(id_, float(weight) / total_weight) for id_, weight in doc]
                yield doc

    @classmethod
    def load(cls, *args, **kwargs):
        """
        Load a previously saved LdaMallet class. Handles backwards compatibility from
        older LdaMallet versions which did not use random_seed parameter.
        """
        model = super().load(*args, **kwargs)
        if not hasattr(model, "random_seed"):
            model.random_seed = 0
        return model


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
