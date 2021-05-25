"""Data processing module.
This module aims at preparing and shaping the resources in order to make it suitable for the subsequent training step.
"""
import sys
import logging
import json
import pickle as pkl
from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from pandas import DataFrame
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import torch
import torch.nn as nn

import glove

formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
stdout_handler.setFormatter(formatter)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)
stderr_handler.setFormatter(formatter)

logging.basicConfig(handlers=[stdout_handler, stderr_handler])
logger = logging.getLogger("GloVe")
logger.setLevel(logging.DEBUG)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info("Device used: %s", device)


class GloVe(nn.Module):
    """Processes a dataset to make it suitable to train a GloVe word embedding."""

    def __init__(self, cooccurrences, embedding_size, x_max=100, alpha=0.75):
        """
        GloVe embedding represented as a NN in Pytorch.

        Parameters
        ----------
        cooccurrences : sparse tensor of shape (n_tokens, n_tokens)
            Co-occurrence matrix of tokens sorted by frequency.
        embedding_size: int. Embedding size
        x_max: int. Saturation of the weighting function
        alpha: float. Exponant of the weighting function
        """
        super(GloVe, self).__init__()

        self.embed_size = embedding_size
        self.x_max = x_max
        self.alpha = alpha

        self.cooccurrences = cooccurrences
        self.n_tokens, _ = self.cooccurrences.shape

        # word embedding
        self.embedding = nn.Embedding(self.n_tokens, self.embed_size)
        self.bias = nn.Embedding(self.n_tokens, 1)

        # context embedding
        self.embedding_tilde = nn.Embedding(self.n_tokens, self.embed_size)
        self.bias_tilde = nn.Embedding(self.n_tokens, 1)

        # Initialization
        half_range = 0.5 / self.embed_size
        self.embedding.weight.data.uniform_(-half_range, half_range)
        self.embedding_tilde.weight.data.uniform_(-half_range, half_range)

        self.bias.weight.data.uniform_(-half_range, half_range)
        self.bias_tilde.weight.data.uniform_(-half_range, half_range)

    def forward(self, indices):
        """
        Forward pass.

        Parameters
        ----------
        indices: LongTensor of shape (batch_size,).
            Indices considered in this batch.

        Returns:
        --------
        loss: Loss estimate for Global Vectors word representations
                 defined in nlp.stanford.edu/pubs/glove.pdf
        """
        rows = self.cooccurrences._indices()[0][indices]
        cols = self.cooccurrences._indices()[1][indices]
        coocs = self.cooccurrences._values()[indices]
        weights = torch.pow(torch.minimum(coocs / self.x_max, torch.as_tensor(1.)), self.alpha)
        w, w_tilde = self.embedding(rows), self.embedding_tilde(cols)
        b, b_tilde = self.bias(rows), self.bias_tilde(cols)
        out = torch.sum(w * w_tilde, axis=1, keepdims=True) + b + b_tilde
        loss = torch.sum(weights * torch.pow(out.T - torch.log(coocs), 2)) * 0.5
        return loss

    def embeddings(self):
        return self.embedding.weight.data.cpu().numpy()


class GloveEmbeddingTransformer(TransformerMixin, BaseEstimator):
    """Processes a dataset to make it suitable to train a GloVe word embedding."""

    def __init__(
        self,
        counter: CountVectorizer,
        alpha=0.75,
        x_max=100.0,
        d=50,
        seed=1234,
        step_size=0.05,
        batch_size=50,
        max_epochs=25,
        workers=9,
        text_colname="text",
        clip_gradient=True,
    ):
        super(GloveEmbeddingTransformer, self).__init__()
        self.text_colname = text_colname
        self.embedding_params = {
            "alpha": alpha,
            "x_max": x_max,
            "d": d,
            "seed": seed,
            "step_size": step_size,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "workers": workers,
        }
        self.clip_gradient = clip_gradient
        if counter.vocabulary_ is None:
            raise ValueError("counter must be fitted.")
        self.counter = counter
        print(json.dumps({
            "min_df": self.counter.min_df,
            "max_df": self.counter.max_df,
            "max_features": self.counter.max_features,
        }))
        self.fit_params = None
        self.embeddings = None
        logger.debug(f"{self.__class__} initialized.")

    def fit_transform(self, cooccurrences_m, documents, y=None, **fit_params):
        """
        Fit to cooccurrence matrix and then transforms documents with fitted model.
        Fits transformer to cooccurrence matrix and returns the transformed documents.

        Parameters
        ----------
        cooccurrences_m : sparse matrix of shape (n_tokens, n_tokens).
        documents : {array-like, sparse matrix, dataframe} of shape \
                (n_documents,) where each element is the content of a document.
        y : Ignored. Used for compatibility with sklearn interface

        Returns
        -------
        embeddings : ndarray array of shape (n_tokens, n_tokens)
            Embedding of the words in documents.
        """
        embeddings = self.fit(cooccurrences_m, y, **fit_params).transform(documents)
        return embeddings

    def fit(self, cooccurrences_m, y=None, **fit_params):
        print(json.dumps(fit_params))
        impl = fit_params.pop('implementation', 'pytorch')
        self.fit_params = self._validate_fit_params(fit_params=fit_params)
        if impl == 'pytorch':
            return self._pytorch_fit(cooccurrences_m, y)
        elif impl == 'cython':
            return self._cython_fit(cooccurrences_m, y)
        else:
            raise ValueError("Implementation must be one of `pytorch` or `cython`.")

    def _cython_fit(self, cooccurrences_m, y=None):
        """
        Fit to resources.

        Fits transformer to documents and returns the transformer object instance.

        Parameters
        ----------
        cooccurrences_m : sparse matrix of shape (n_tokens, n_tokens).

        y : ndarray of shape (n_samples,), default=None
            Ignored.

        Returns
        -------
        self : an instance of GloveEmbeddingTransformer
        """
        logger.info(
            f"Co-occurrence matrix: Number of non-zero elements {cooccurrences_m.count_nonzero()} / "
            f"{cooccurrences_m.shape[0] * cooccurrences_m.shape[1]}"
        )

        # Compute embeddings
        logger.info("Compute embeddings...")
        model = glove.Glove(
            cooccurrences_m,
            d=self.fit_params.get("d"),
            alpha=self.fit_params.get("alpha"),
            x_max=self.fit_params.get("x_max"),
        )

        for epoch in range(self.fit_params.get("max_epochs")):
            err = model.train(
                step_size=self.fit_params.get("step_size"),
                batch_size=self.fit_params.get("batch_size"),
                workers=self.fit_params.get("workers"),
            )
            logger.debug(f"Error for epoch {epoch}: {err}")
            print(json.dumps({"implementation": "cython", "epoch": epoch, "total_loss": err}))

        logger.info("Embedding done.")
        self.embeddings = model.W.copy()
        return self

    def _pytorch_fit(self, cooccurrences_m, y=None):
        """
        Fit to resources.
        Fits transformer to cooccurrence matrix and returns the transformer object instance.

        Parameters
        ----------
        cooccurrences_m : sparse matrix of shape (n_tokens, n_tokens).
        y : Ignored. Used for compatibility with sklearn interface

        Returns
        -------
        self : an instance of GloveEmbeddingTransformer
        """
        logger.info("Starting to fit embedding...")

        # Set seed for reproducibility.
        torch.manual_seed(self.fit_params.get("seed"))
        np.random.seed(self.fit_params.get("seed"))

        logger.info("Compute embeddings...")

        logger.debug("Load cooccurrences as tensor to {device}...")
        cooccurrences_t = torch.sparse_coo_tensor(list(cooccurrences_m.nonzero()), cooccurrences_m.data,
                                                  cooccurrences_m.shape, device=device)
        logger.debug("Finished loading of cooccurrences.")

        logger.debug("Initialize model...")
        model = GloVe(cooccurrences_t, self.fit_params["d"], self.fit_params["x_max"], self.fit_params["alpha"]).to(
            device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=self.fit_params["step_size"])
        logger.debug("Model and optimizer initialized.")

        for epoch in tqdm(range(1, self.fit_params.get("max_epochs") + 1)):
            total_loss = 0.0
            for batch in self.get_next_batch(cooccurrences_m.nnz, self.fit_params["batch_size"]):
                batch = batch.to(device)
                loss = model(batch)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                if self.clip_gradient:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            print(json.dumps({"implementation": "pytorch", "epoch": epoch,
                              "total_loss": total_loss / cooccurrences_m.count_nonzero()}))
        logger.info("Embedding done.")
        self.embeddings = model.embeddings()
        return self

    def get_next_batch(self, total_els, batch_size):
        _idx = list(range(total_els))
        np.random.shuffle(_idx)
        while len(_idx):
            selection = []
            for i in range(batch_size):
                try:
                    selection.append(_idx.pop())
                except:
                    pass
            yield torch.LongTensor(selection)

    def _validate_fit_params(self, fit_params=None):
        valid_params = self.embedding_params.copy()
        if fit_params is not None:
            for key in fit_params:
                if key not in valid_params.keys():
                    logger.warning(
                        f"Unknown parameter {key} will be ignored. Valid parameters are: {valid_params.keys()}."
                    )
                if fit_params[key] is None:
                    raise ValueError(
                        f"Parameter {key} is set to None, which is forbidden."
                    )
            valid_params.update(fit_params)
        return valid_params

    def transform(self, documents=None):
        """
        Transform resources after transformer has been fitted. Returns a coocurence matrix.

        Parameters
        ----------
        documents : {array-like, sparse matrix, dataframe} of shape \
                (n_documents,) where each element is the content of a document.

        Returns
        -------
        embeddings : ndarray array of shape (n_tokens, n_tokens)
            Embedding of the words in documents.
        """
        if self.embeddings is None:
            raise ValueError("GloveEmbeddingTransformer instance hasn't been fitted.")

        if documents is None:
            return self.embeddings

        if isinstance(documents, DataFrame):
            doc_index = documents.index.copy()
            documents = documents[self.text_colname]
        else:
            doc_index = range(len(documents))
        counts = self.counter.transform(documents)

        normalizer = counts.sum(axis=1)
        embedded_docs = ((counts * self.embeddings) + 1.) / (normalizer + 1.)
        return DataFrame(data=embedded_docs, index=doc_index)

    def get_fit_params(self):
        if self.fit_params is None:
            return self._validate_fit_params()
        else:
            return self.fit_params


def train_glove(
        cooccurrence_matrix: Union[str, Path],
        counter: Union[str, Path],
        output_path: Union[str, Path],
        text_colname: str = "text",
        max_epochs: int = 5,
        nb_dims: int = 100,
        batch_size: int = 1024,
        implementation: str = "pytorch",
        **kwargs,
):
    """GloVe training utility

    Train GloVe on the text provided as a json file produced by pandas.
    The DataFrame resulting from reading the json should contain at least two columns: 'text' and 'main_language'

    Parameters
    ----------
    cooccurrence_matrix: {str, Path}. Path to a compressed numpy matrix.
    counter: {str, Path}. Path to a CountVectorizer serialized with pickle.
    output_path: {str, Path}. Path where to output the pickled embedder.
    text_colname: str.
        Name of the column of the dataframe to be used for transformation. Defaults to `text`.
    max_epochs: int. Number of epochs to train the embeddings. Defaults to 25.
    nb_dims: int. Number of dimensions of the resulting embeddings. Defaults to 100.
    batch_size: int. Size of the batches to train the embedder. Defaults to 1024.
    implementation: Type of implementation to be used during training: "pytorch" or "cython"
    kwargs : arguments to be passed to the DataCleaner constructor.

    Returns
    -------
    status : int. 0 indicates no error.
    """
    # Load cooccurrences from file
    cooccurrence_matrix = Path(cooccurrence_matrix)
    loader = np.load(cooccurrence_matrix)
    cooccurrences = csr_matrix(
        (loader['resources'], loader['indices'], loader['indptr']), shape=loader['shape'])

    # Load CountVectorizer from file
    counter = Path(counter)
    with open(counter, "rb") as istream:
        counter = pkl.load(istream)

    output_path = Path(output_path)

    glove_embedder = GloveEmbeddingTransformer(
        counter=counter,
        text_colname=text_colname,
        max_epochs=max_epochs, d=nb_dims, batch_size=batch_size, **kwargs
    )

    glove_embedder.fit(cooccurrences, implementation=implementation)
    with open(output_path, "wb") as ostream:
        pkl.dump(glove_embedder, ostream)

    if output_path.exists():
        return 0
    return 1
