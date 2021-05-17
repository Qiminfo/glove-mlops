from pathlib import Path
from typing import List, Optional, Union

import nltk
from nltk.corpus import stopwords
from pandas import DataFrame, Series, read_json
from sklearn.base import BaseEstimator, TransformerMixin
from unidecode import unidecode
import json

from . import text_utils as tu


class DataCleaner(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(
        self,
        replace_urls: bool = True,
        replace_dates: bool = True,
        replace_digits: bool = True,
        replace_mails: bool = True,
        normalize_text: bool = True,
        replace_placeholders: bool = True,
        custom_stop_words: Optional[List[str]] = None,
        text_colname: str = "text",
        lang_colname: str = "main_language",
    ):

        # ref colname to be transformed
        self._text_colname = text_colname
        self._lang_colname = lang_colname

        # cleaning params
        self.replace_urls = replace_urls
        self.replace_dates = replace_dates
        self.replace_digits = replace_digits
        self.replace_mails = replace_mails
        self.normalize_text = normalize_text
        self.replace_placeholders = replace_placeholders
        self.custom_stop_words = [] if custom_stop_words is None else custom_stop_words
        self.stopwords = {"fr": self.stopwords_fr(), "en": self.stopwords_en()}

        # word_counts
        self.preprocessing_w_counts = None
        self.postprocessing_w_counts = None

    def stopwords_fr(self):
        stopwords_fr = self.custom_stop_words + stopwords.words("french")
        if self.normalize_text:
            return list(map(unidecode, stopwords_fr))
        return stopwords_fr

    def stopwords_en(self):
        removed_english_stopwords = ["it"]
        stopwords_en = list(
            filter(
                lambda w: w not in removed_english_stopwords, stopwords.words("english")
            )
        )
        stopwords_en = self.custom_stop_words + stopwords_en
        if self.normalize_text:
            return list(map(unidecode, stopwords_en))
        return stopwords_en

    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        print("Cleaning text...")

        # todo delete file with less than n lines in text (parsR failed)
        X[self._text_colname] = X[self._text_colname].str.lower()
        self.preprocessing_w_counts = self.get_word_counts(X, self._text_colname)
        print(
            json.dumps(
                {f"pre_processing_vocab_size": self.preprocessing_w_counts.shape[0]}
            )
        )

        if self.replace_dates:
            X[self._text_colname] = X[self._text_colname].apply(tu.replace_dates)

        if self.replace_mails:
            X[self._text_colname] = X[self._text_colname].apply(tu.replace_mails)

        if self.replace_urls:
            X[self._text_colname] = X[self._text_colname].apply(tu.replace_dotnet)
            X[self._text_colname] = X[self._text_colname].apply(tu.replace_urls)

        if self.normalize_text:
            X[self._text_colname] = X[self._text_colname].apply(unidecode)

        # tmp tokenize to filter stop words
        X[self._text_colname] = X[self._text_colname].apply(
            lambda t: nltk.regexp_tokenize(t, pattern=r"(?u)\b\w\w+\b")
        )

        # replace digits
        if self.replace_digits:
            X[self._text_colname] = X[self._text_colname].apply(
                lambda l: list(map(tu.replace_digits, l))
            )

        if self.replace_placeholders:
            X[self._text_colname] = X[self._text_colname].apply(
                lambda l: list(map(tu.strip_markers, l))
            )

        # remove stop words
        def filter_stopwords(row: Series):
            if row[self._lang_colname] == "fr":
                return list(
                    filter(
                        lambda w: w not in self.stopwords["fr"], row[self._text_colname]
                    )
                )
            elif row[self._lang_colname] == "en":
                return list(
                    filter(
                        lambda w: w not in self.stopwords["en"], row[self._text_colname]
                    )
                )
            else:
                return row[self._text_colname]

        print(
            json.dumps(
                {
                    "pre_stop_words_filter_unique_count": X[self._text_colname]
                    .explode()
                    .nunique()
                }
            )
        )

        X[self._text_colname] = X.apply(filter_stopwords, axis=1)
        print(
            json.dumps(
                {
                    "post_stop_words_filter_unique_count": X[self._text_colname]
                    .explode()
                    .nunique()
                }
            )
        )

        # regroup list of tokens back to text
        X[self._text_colname] = X[self._text_colname].apply(lambda l: " ".join(l))

        self.postprocessing_w_counts = self.get_word_counts(X, self._text_colname)
        print(
            json.dumps(
                {"post_processing_vocab_size": self.postprocessing_w_counts.shape[0]}
            )
        )

        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        # todo set self.checkpoint_outputpath to None after fit_transform in order to play inference without
        #  checkpointing inputs

        return self.transform(X, y)

    # utilitary function to check results
    def get_word_counts(self, df: DataFrame, text_colname: str):

        tokenized_data = df[text_colname].copy()
        tokenized_data = tokenized_data.apply(
            lambda t: nltk.regexp_tokenize(t, pattern=r"(?u)\b\w\w+\b")
        )

        tokenized_data = (
            tokenized_data.explode()
            .value_counts()
            .sort_values(ascending=False)
            .to_frame()
        )

        tokenized_data["rank"] = tokenized_data.text.sort_values(ascending=False).rank()
        tokenized_data = tokenized_data.reset_index()
        tokenized_data.columns = ["token", "freq", "rank"]
        tokenized_data = tokenized_data.set_index("token")
        tokenized_data = tokenized_data.sort_values("rank", ascending=False)

        return tokenized_data

    def record_based_stopwords_filter(self, tokens: List[str], language: str):
        stopwords = self.stopwords[language]
        return list(filter(lambda t: t not in stopwords, tokens))


def clean_text(data_file: Union[str, Path], output_path: Union[str, Path], **kwargs):
    """Text cleaning utility

    Clean the text provided as a json file produced by pandas.
    The DataFrame resulting from reading the json should contain at least two columns: 'text' and 'lang'

    Parameters
    ----------
    data_file: {str, Path}. Path to a Dataframe serialized as json.
    kwargs : arguments to be passed to the DataCleaner constructor.

    Returns
    -------
    status : int. 0 indicates no error.
    """
    output_path = Path(output_path)
    print(json.dumps(kwargs))
    df = read_json(data_file)
    cleaner = DataCleaner(**kwargs)
    cleansed = cleaner.transform(df)
    cleansed.to_json(output_path)
    if output_path.exists():
        return 0
    return 1
