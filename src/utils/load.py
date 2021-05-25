import logging
import sys
from pathlib import Path
from typing import Union, Tuple

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelBinarizer

formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
stdout_handler.setFormatter(formatter)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)
stderr_handler.setFormatter(formatter)
logging.basicConfig(handlers=[stdout_handler, stderr_handler])
logger = logging.getLogger("Evaluation")
logger.setLevel(logging.INFO)


def load_data(output_path: Union[Path, str] = "./resources") -> Union[Tuple, str]:
    """
    Loads 20Newsgroups as demo dataset and returns it formatted as needed for later use as pipeline input
    :return: pd.DataFrame
    """

    newsgroups_train = fetch_20newsgroups(subset="train")
    df_train = pd.DataFrame(index=newsgroups_train.filenames, data=newsgroups_train.data, columns=["text"])
    df_train["main_language"] = "en"
    df_train = df_train.sample(n=500).copy()
    logger.info(f"Kept Train Sample DF with {df_train.shape[0]} samples")

    newsgroups_test = fetch_20newsgroups(subset="test")
    df_test = pd.DataFrame(index=newsgroups_test.filenames, data=newsgroups_test.data, columns=["text"])
    df_test["main_language"] = "en"
    df_test["label"] = newsgroups_test.target

    df_test = df_test.sample(n=100).copy()
    logger.info(f"Kept Test Sample DF with {df_test.shape[0]} samples")

    encoder = LabelBinarizer()
    t = pd.DataFrame(encoder.fit_transform(df_test.label))
    df_test.reset_index(inplace=True)
    df_test = pd.merge(df_test, t, how="inner", left_index=True, right_index=True)
    df_test.set_index("index", inplace=True)

    output_path = Path(output_path)
    train_fp = output_path / "demo_data_train.json"
    test_fp = output_path / "demo_data_test.json"
    df_train.to_json(train_fp)
    df_test.to_json(test_fp)

    logger.info(f"Train and Test datasets saved to {output_path}")

    return train_fp, test_fp
