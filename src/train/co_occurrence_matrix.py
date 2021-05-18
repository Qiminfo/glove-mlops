"""Data processing module.
This module aims at preparing and shaping the data in order to make it suitable for the subsequent training step.
"""

import sys
import logging
import pickle as pkl
from pathlib import Path
from typing import Iterable, Tuple, List, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

from pandas import read_json
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer

formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
stdout_handler.setFormatter(formatter)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)
stderr_handler.setFormatter(formatter)

logging.basicConfig(handlers=[stdout_handler, stderr_handler])
logger = logging.getLogger("CoOccurrence")
logger.setLevel(logging.INFO)


class CoOccurrenceMatrixTransformer(TransformerMixin, BaseEstimator):
    """Processes a dataset to make it suitable to train a GloVe word embedding."""

    def __init__(self, **params):
        """CooccurrenceMatrixTransformer
        Transformer that counts the number of
        """
        super(CoOccurrenceMatrixTransformer, self).__init__()
        self.counts = None
        self.count_params = params
        if "lowercase" not in params:
            self.count_params["lowercase"] = False
        self.fit_params = None
        self.counter = None
        self.vocabulary = None
        logger.debug(f"Class {self.__class__} is initialized")

    def fit_transform(self, documents, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to documents and y
        and returns a co-occurrence matrix.

        Parameters
        ----------
        documents : {array-like, sparse matrix, dataframe} of shape \
                (n_documents,) where each element is the content of a document.

        y : ndarray of shape (n_samples,), default=None
            Ignored.

        Returns
        -------
        cooccurrences : ndarray array of shape (n_tokens, n_tokens)
            Co-occurrence matrix of tokens sorted by frequency.
        """
        if hasattr(documents, "shape"):
            logger.debug("Shape of documents: %s", documents.shape)
        else:
            logger.debug("Shape of documents: %s", (len(documents),))
        logger.debug("Fitting parameters: %s", fit_params)
        coocurences = self.fit(documents, y, **fit_params).transform(self.counts)
        return coocurences

    @staticmethod
    def _flatmap(func: callable, arr: Iterable) -> List[str]:
        output = []
        for item in arr:
            for elem in func(item):
                output.extend([elem])
        return output

    def fit(self, documents, y=None, **fit_params):
        """
        Fit to data.

        Fits transformer to documents and returns the transformer object instance.

        Parameters
        ----------
        documents : {array-like, sparse matrix, dataframe} of shape \
                (n_documents,) where each element is the content of a document.

        y : ndarray of shape (n_samples,), default=None
            Ignored.

        Returns
        -------
        self : an instance of CoOccurrenceMatrixTransformer
        """
        count_params = self.count_params.copy()
        count_params.update(fit_params)
        # Need to consider ngrams rather than whole documents for cooccurrence
        ctx = count_params.pop("context_size", None)
        if ctx is not None:
            # build a proper analyzer that will behave as our counter
            analyzer_params = count_params.copy()
            analyzer_params.update({"ngram_range": (ctx, ctx)})
            analyzer = CountVectorizer(**analyzer_params).build_analyzer()
            # transform documents as list of ngrams
            documents = self._flatmap(analyzer, documents)

        self.fit_params = count_params.copy()
        self.fit_params["context_size"] = ctx  # saved to enable audit

        counter = CountVectorizer(**count_params)
        logger.debug("Perform counting using CountVectorizer...")
        self.counts = counter.fit_transform(documents, y)
        logger.debug("Counting done...")
        self.counter = counter
        self.vocabulary = counter.get_feature_names()
        logger.debug("Size of vocabulary: %s", len(self.vocabulary))
        return self

    def transform(self, count_matrix=None):
        """
        Transform data after transformer has been fitted. Returns a co-occurrence matrix.

        Parameters
        ----------
        count_matrix : {array-like, sparse matrix, dataframe} of shape \
                (n_documents, n_tokens) obtained with a CountVectorizer

        Returns
        -------
        cooccurrences : ndarray array of shape (n_tokens, n_tokens)
            Co-occurrence matrix of tokens sorted by frequency.
        """
        if count_matrix is not None:
            counts = count_matrix
        elif self.counts is not None:
            counts = self.counts
        else:
            raise ValueError(
                "CoOccurrenceMatrixTransformer instance hasn't been fitted. "
                "Please provide a count matrix count_matrix."
            )

        n_docs, n_tokens = counts.shape
        output = (counts.T * counts) - (
            np.eye(n_tokens, n_tokens) * counts.sum(axis=0).getA1()
        )
        return csr_matrix(output)

    def get_fit_params(self):
        if self.fit_params is None:
            return self.count_params
        else:
            return self.fit_params


def build_cooccurrence(
    cleansed_cvs: Union[str, Path],
    output_path: Union[str, Path],
    text_colname: str = "text",
    language_colname: str = "main_language",
    lang: Optional[str] = None,
    input: str = "content",
    encoding: str = "utf-8",
    decode_error: str = "strict",
    strip_accents: Optional[str] = None,
    lowercase: bool = True,
    preprocessor: Optional[callable] = None,
    tokenizer: Optional[callable] = None,
    stop_words: Optional[Union[str, List[str]]] = None,
    token_pattern: str = r"(?u)\b\w\w+\b",
    ngram_range: Tuple[int] = (1, 1),
    context_size: Optional[int] = None,
    analyzer: str = "word",
    max_df: float = 1.0,
    min_df: int = 1,
    max_features: int = None,
    vocabulary: List[str] = None,
    binary: bool = False,
    dtype=np.int64,
):
    """Cooccurrence building utility

    Use CoOccurrenceMatrixTransformer on the text provided as a json file produced by pandas.
    The DataFrame resulting from reading the json should contain at least two columns: 'text' and 'main_language'

    Parameters
    ----------
    cleansed_cvs: {str, Path}. Path to the cleansed cvs saved as a Dataframe serialized in json.
    output_path: {str, Path}. Path where to write the cooccurrence matrix as a numpy compressed matrix.
    kwargs : arguments to be passed to the CoOccurrenceMatrixTransformer constructor.

    Returns
    -------
    status : int. 0 indicates no error.
    """
    output_path = Path(output_path)

    df = read_json(cleansed_cvs)
    if lang is not None:
        df = df.query(f"{language_colname} == '{lang}'")

    cooc_matrix_tfx = CoOccurrenceMatrixTransformer(
        encoding=encoding,
        decode_error=decode_error,
        strip_accents=strip_accents,
        lowercase=lowercase,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        stop_words=stop_words,
        token_pattern=token_pattern,
        ngram_range=ngram_range,
        context_size=context_size,
        analyzer=analyzer,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        vocabulary=vocabulary,
        binary=binary,
        dtype=dtype,
    )

    cooc_matrix = cooc_matrix_tfx.fit_transform(df[text_colname])
    cooc_matrix_path = output_path / "cooccurrence_matrix.npz"

    np.savez_compressed(
        cooc_matrix_path,
        data=cooc_matrix.data,
        indices=cooc_matrix.indices,
        indptr=cooc_matrix.indptr,
        shape=cooc_matrix.shape,
    )

    counter_path = output_path / "counter.pkl"
    with open(counter_path, "wb") as ostream:
        pkl.dump(cooc_matrix_tfx.counter, ostream)

    if output_path.exists():
        return 0
    return 1
