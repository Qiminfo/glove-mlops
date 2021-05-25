"""Utilities to make proper evaluation"""
import sys
import json
import logging
from pathlib import Path
from typing import Tuple, Any, Union, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from pathlib import Path
import io

import pickle as pkl

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


def get_cooc_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute coocurences matrix from a Dataframe composed of profiles (rows) and their pertainencies to IT professional
    categories (Columns; values: 1/0).
    Args:
        df: pd.DatFrame, labeled dataset

    Returns: pd.DataFrame representing the matrix of coocurences (number of categories in common between each pair)

    """
    cooc = np.matmul(df.__array__(), df.__array__().T)
    cooc_df = pd.DataFrame(data=cooc, index=df.index, columns=df.index)
    return cooc_df


def relevant_items(idx: str, cooc: pd.DataFrame) -> pd.Series:
    """
    Returns (sorted or not) cvs (profiles) that have at least 1 categories in common for a give profile (idx)
    Args:
        idx: index of the target profile
        cooc: coocurence matrix (see above)

    Returns: pd.Series, relevant profiles

    """

    return cooc.loc[idx] > 0


def compute_distance_matrix(
    embedded_docs: pd.DataFrame, doc_idx: pd.Series = None, metric: str = "cosine"
) -> pd.DataFrame:
    """
    Compute distance between vectors representing documents (cvs)
    Args:
        embedded_docs: vectors of embedded documents (cvs)
        doc_idx: documents index
        metric: metric to be used for distance computation

    Returns: pd.DataFrame, distance matrix

    """
    if not isinstance(embedded_docs, pd.DataFrame):
        embedded_docs_ = np.asmatrix(embedded_docs)
        if doc_idx is None:
            raise ValueError(
                "doc_idx should be provided if you do not provide a DataFrame."
            )
    else:
        embedded_docs_ = embedded_docs.values
        doc_idx = embedded_docs.index
    distance_matrix = cdist(embedded_docs_, embedded_docs_, metric)
    return pd.DataFrame(data=distance_matrix, columns=doc_idx, index=doc_idx)


def average_precision(
    ranked: pd.Series, relevants: pd.Series, k: Optional[int] = None
) -> pd.Series:
    """
    Compute mean average precision (MAP) given an recommended (infered) ranking of similair profiles
    and its relevant profiles (labeled resources)
    Args:
        ranked: most similar profiles ranking given a target cv
        relevants: actually similar profiles
        k: max number of recommandation to be taken from the ranking

    Returns:

    """
    precisions, _ = precisions_recalls(ranked, relevants, k)

    # average precision does NOT take into account precision where result is not relevant
    # for explanations about usage of intersection:
    # See https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    avg_precision = precisions.loc[
        precisions.index.intersection(relevants.index)
    ].mean()

    return avg_precision


def get_rank(distance_scores: pd.Series) -> pd.Series:
    """
    Get the ranking given a Series containing distances
    Args:
        distance_scores: pd.Series, computed distances

    Returns: Series sorted by distance from least distant to most distant)

    """
    ranked = distance_scores.sort_values(ascending=True)
    ranked = ranked[1:]  # first element is the same document
    return ranked


def get_random_rank(doc_idx: pd.Index or pd.Series) -> pd.Series:
    """
    Generate a random ranking given an index of documents (cvs)
    Args:
        doc_idx: pd.Series of documents (cvs)

    Returns: pd.Series, Random ranking

    """
    if isinstance(doc_idx, pd.Index):
        rank = doc_idx.to_series()
    else:
        rank = doc_idx.copy()
    return rank.sample(frac=1.0)


def precisions_recalls(
    ranked: pd.Series, relevants: pd.Series, k: Optional[int] = None
) -> Tuple[Any, Union[float, Any]]:
    """
    Compute mean average precisions for every points of recall given a computed ranking for a document (cv) and
    its groundtruth (similar documents)
    Args:
        ranked: Computed ranking
        relevants: Relevant documents
        k: max number of recommandation from the ranking to be taken into account

    Returns: tuple(pd.Series: precisions, pd.Sereis: recalls)

    """
    if k is None:
        k = len(ranked)
    norm = np.arange(1, len(ranked) + 1)
    precisions = relevants.loc[ranked.index].cumsum() / norm
    recalls = relevants.loc[ranked.index].cumsum() / min(relevants.sum(), k)
    return precisions, recalls


# pylint: disable=invalid-name
def interp_pr(
    p: pd.Series, r: pd.Series, n: int
) -> Tuple[Any, Union[np.ndarray, Tuple[np.ndarray, Optional[float]]]]:
    """
    Interpolate precision values for each point of n recall intervals
    Args:
        p: points of precisions
        r: points of recalls
        n: nb of intervals

    Returns: tuple(pd.Series: interpolated precision points, pd.Series: recalls)

    """
    precisions_max = np.maximum.accumulate(p.__array__()[::-1])[::-1]
    recalls_i = np.linspace(0, 1, num=n)
    precisions_i = np.interp(recalls_i, r.__array__(), precisions_max)
    return precisions_i, recalls_i


# pylint: disable=invalid-name, too-many-locals
def map_precision_recall_curves(
    distance_matrix: pd.DataFrame,
    relevants: pd.Series,
    n: int = 10,
    k: Optional[int] = None,
    random: bool = False,
    interp: bool = True,
) -> pd.DataFrame:
    """Compute mean, std, max and min values for each kth recommendation over a set of queries"""
    precisions = []
    recalls = []
    for i, d in distance_matrix.iterrows():
        if random:
            rank = get_random_rank(distance_matrix.index.drop(i))
        else:
            rank = get_rank(d)

        p, r = precisions_recalls(rank, relevants.loc[d.name], k)

        if interp:
            p, r = interp_pr(p, r, n)

        precisions.append(p)
        recalls.append(r)

    precisions = np.vstack(precisions)
    recalls = np.vstack(recalls)

    mean_precisions = np.mean(precisions, axis=0)
    std_precisions = np.std(precisions, axis=0)
    mean_plus_stdev = np.add(mean_precisions, std_precisions)
    mean_minus_stdev = np.subtract(mean_precisions, std_precisions)
    max_precisions = np.max(precisions, axis=0)
    min_precisions = np.min(precisions, axis=0)
    mean_recalls = np.mean(recalls, axis=0)

    return pd.DataFrame(
        {
            "mean_p": mean_precisions,
            "mean_r": mean_recalls,
            "std_p": std_precisions,
            "max_p": max_precisions,
            "min_p": min_precisions,
            "mean_plus_stdev_p": mean_plus_stdev,
            "mean_minus_stdev_p": mean_minus_stdev,
        }
    )


# todo unit test
def compute_feedback_precisions(
    feedback_df: pd.DataFrame,
    ranking_colname: str = "results_ranking",
    feedbacks_colname: str = "results_feedbacks",
):
    """
    Compute mean average precision points for each datapoint of the dataset given a dataframe
    containing rankings and their related users' feedback
    Args:
        feedback_df: pd.DataFrame: each row is corresponding to one query
        ranking_colname: name of the column containing the rankings
        feedbacks_colname: name of the column containing user feedback

    Returns:
        pd.Dataframe with mean averages precisions for each query

    """
    precisions = feedback_df.apply(
        lambda query: precisions_recalls(
            pd.Series(query[ranking_colname]), pd.Series(query[feedbacks_colname])
        )[0],
        axis=1,
    )

    return precisions


def compute_feedback_precisions_per_quantiles(
    feedback_df: pd.DataFrame,
    k_limit: int,
    ranking_colname: str = "results_ranking",
    feedbacks_colname: str = "results_feedbacks",
    feedbacks_nb_colname: str = "feedbacks_nb",
):
    """
    Compute mean average precision points for top k ranking's recommendation from a dataframe
    containing rankings and their related users' feedback. Precision points are computed for each decile or lesser
    depending on the minimum number of feedback over every queries.
    Args:
        feedback_df: pd.DataFrame: each row is corresponding to one query
        k_limit: max number of recommandation from the ranking to be taken into account
        ranking_colname: name of the column containing the rankings
        feedbacks_colname: name of the column containing user feedback
        feedbacks_nb_colname: name of the column containing number of user feedback

    Returns:
        tuple(pd.Series: precisions for each resources point, pd.Series: precisions for each quantile,
        pd.Series: chosen quantiles)

    """
    precisions = compute_feedback_precisions(
        feedback_df,
        ranking_colname=ranking_colname,
        feedbacks_colname=feedbacks_colname,
    )

    quantiles = np.arange(0.0, 1.1, 0.1)

    # make sure that no record has zero feedback
    assert (
        feedback_df[feedbacks_nb_colname].min() != 0
    ), "One or more record with no feedback: check selected resources."

    if k_limit != 0:
        quantiles_nb = (
            k_limit
            if not feedback_df[feedbacks_nb_colname].min() < k_limit
            else feedback_df[feedbacks_nb_colname].min()
        )
        k_to_keep = (
            feedback_df[feedbacks_nb_colname].min()
            if feedback_df[feedbacks_nb_colname].min() > quantiles_nb
            else quantiles_nb
        )
        quantiles = pd.qcut(quantiles, quantiles_nb).categories.right.tolist()

    else:
        k_to_keep = feedback_df[feedbacks_nb_colname].min()
        if k_to_keep < 10:
            quantiles_nb = k_to_keep
            quantiles = pd.qcut(quantiles, quantiles_nb).categories.right.tolist()

    truncated_precisions = precisions.iloc[:, :k_to_keep].copy()

    precisions_quantiles = truncated_precisions.apply(
        lambda r: r.quantile(quantiles), axis=1
    )

    return truncated_precisions, precisions_quantiles, quantiles


def parse_feedback_df(df: pd.DataFrame, json_colname: str = "text") -> pd.DataFrame:
    """
    Parsing function be applied on a df returned by the 'download_all_files' func in 's3_utils'
    Content of each query's feedback (ie, the json resources) should be found in the 'json_colname' column of the input df.
    """
    df[json_colname] = df[json_colname].apply(json.loads)
    df["query_ts"] = df[json_colname].apply(lambda x: x.get("query_ts"))
    df["query_str"] = df[json_colname].apply(lambda x: x.get("query_str"))
    df["results"] = df[json_colname].apply(lambda x: x.get("results"))
    df["results_ranking"] = df.results.apply(lambda x: [e.get("name") for e in x])
    df["results_feedbacks"] = df.results.apply(lambda x: [e.get("value") for e in x])
    df["results_views"] = df.results.apply(lambda x: [e.get("viewed") for e in x])
    df["similarity_scores"] = df.results.apply(
        lambda x: [e.get("similarity") for e in x]
    )
    df["comments"] = df.results.apply(lambda x: [e.get("comment") for e in x])
    df["feedbacks_nb"] = df.results_ranking.apply(len)

    df.drop(columns=[json_colname, "results"], inplace=True)

    # filter out records with no feedbacks
    df = df.loc[df["feedbacks_nb"] != 0].copy()

    return df


def apply(
    glove_embedder: Union[str, Path],
    labeled_docs_path: Union[str, Path],
    lang: str = "",
    language_colname: str = "main_language",
    output_path: Union[str, Path] = "/resources/eval/embedded_cvs_labeled_df.json",
):
    """
    Apply GloVe to the dataframe in input and save the average embeddings in output_path.
    :param glove_embedder: filepath of trained Embedder (Transformer-like) to be used on labeled dataset.
    :param output_path: filepath of labeled dataset embeddings
    :param labeled_docs_path: filepath of labeled resources set; expected to be a json exported pandas dataframe
    :param lang: Language to consider from the output.
    Defaults to empty string, which means no distinction.
    :param language_colname: str. Name of the column indicating document's main language.
    :return: int. 0 if no error > 0 otherwise.
    """

    glove_embedder = Path(glove_embedder)
    output_path = Path(output_path)

    print(json.dumps({"lang": lang, "labeled_docs_path": str(labeled_docs_path)}))

    logger.info("If %s exists, it will be backed up.", output_path)
    try:
        output_path.rename(output_path.with_suffix(".bkp"))
    except FileNotFoundError:
        logger.debug("File %s did not exists, no back-up needed", output_path)

    try:
        with open(glove_embedder, "rb") as stream:
            model = pkl.load(stream)
    except FileNotFoundError as err:
        logger.error(err)
        return 1

    labeled_df = pd.read_json(labeled_docs_path)

    try:
        labeled_df.set_index("filename", inplace=True)
    except KeyError:
        logger.warning("Column `filename` is absent")

    if lang != "":
        labeled_df = labeled_df.query(f"{language_colname} == '{lang}'")

    embeddings = model.transform(labeled_df)
    embeddings.to_json(output_path, orient="index")

    logger.info(f"Embedded documents stored at: {output_path}")

    if output_path.exists():
        return 0
    return 1

    return 0


# pylint: disable=too-many-locals
def evaluation(
    embedded_cvs_labeled: Union[str, Path],
    labeled_docs_path: Union[str, Path],
    lang: str = "",
    language_colname: str = "main_language",
    ranking_limit: int = None,
    random_samples_nb: int = 100,
    output_path: Union[str, Path] = "/resources/eval",
):
    """Document retrieval performance evaluation.
    :param embedded_cvs_labeled: Input file path for the embedded dataset.
                               The input is expected to be a json file made with Pandas.
    :param labeled_docs_path: Input file path with the labeled documents.
                              The input is expected to be a json file made with Pandas.
    :param lang: Language to consider from the output.
                 Defaults to empty string, which means no distinction.
    :param language_colname: str. Name of the column indicating document's main language.
    :param ranking_limit: int. Compute performance over first `ranking_limit` documents.
                          Defaults to None. If None, no limit is enforced.
    :param random_samples_nb: int. Number of samples to be drawn for performance comparison.
    :param output_path: Path of Directory where to output evaluation artifacts & metrics.
    :return: int. 0 if no error > 0 otherwise.
    """
    labeled_docs_input = Path(labeled_docs_path)
    embedded_docs_input = Path(embedded_cvs_labeled)

    print(
        json.dumps(
            {
                "lang": lang,
                "ranking_limit": ranking_limit,
                "random_samples_nb:": random_samples_nb,
                "labeled_docs_path": str(labeled_docs_path),
            }
        )
    )

    logger.info("Running the evaluation pipeline...")
    # compute some metric to assess model validity
    # first select relevant cvs according to language
    dataframe = pd.read_json(labeled_docs_input)

    # select all if lang not specified: "" in r'*.'
    labeled_examples_lang = dataframe[
        dataframe[language_colname].str.contains(lang)
    ].copy()

    # adapted to demo resources
    profile_categories_columns = [c for c in labeled_examples_lang.columns if c not in ["text", "main_language"]]

    # compute what is relevant and what is not for each example
    cooc = get_cooc_matrix(labeled_examples_lang[profile_categories_columns])
    doc_names = labeled_examples_lang.index.to_series()
    relevants = doc_names.apply(lambda doc: relevant_items(doc, cooc))

    # get labeled_examples doc embeddings
    logger.info("Open embedding database %s", embedded_docs_input)
    embedded_docs = pd.read_json(embedded_docs_input, orient="index")

    # compute similarities between documents
    distance_matrix = compute_distance_matrix(embedded_docs, doc_names)
    avg_precisions = distance_matrix.apply(
        lambda s: average_precision(
            ranked=get_rank(s),
            relevants=relevants.loc[s.name],
            k=ranking_limit,
        )
    )
    print(
        json.dumps(
            {"MAP_avg": avg_precisions.mean(), "MAP_stdev": avg_precisions.std()}
        )
    )

    # get curves: m.a.p. for each kth recommendation
    # aggregated over the set of queries/labeled resources points
    curves_df = map_precision_recall_curves(
        distance_matrix=distance_matrix, relevants=relevants, k=ranking_limit
    )
    for i, row in curves_df.iterrows():
        rank_data = {col: row[col] for col in curves_df.columns}
        rank_data["rank"] = i
        print(json.dumps(rank_data))

    random_curves = {metric: [] for metric in curves_df.columns}

    for _ in range(random_samples_nb):
        random_curves_df = map_precision_recall_curves(
            distance_matrix=distance_matrix,
            relevants=relevants,
            random=True,
            k=ranking_limit,
        )

        for metric in random_curves_df.columns:
            random_curves[metric].append(random_curves_df[metric])

    random_curves_agg = {}
    for metric, values in random_curves.items():
        random_curves_agg[metric + "_random"] = np.mean(np.vstack(values), axis=0)

    random_curves_df = pd.DataFrame(random_curves_agg)
    for i, row in random_curves_df.iterrows():
        rank_data = {col: row[col] for col in random_curves_df.columns}
        rank_data["rank"] = i
        print(json.dumps(rank_data))

    random_map_avg = random_curves_df.mean_p_random.mean()
    random_map_stdev = random_curves_df.std_p_random.mean()
    print(
        json.dumps(
            {
                "random_MAP_avg": random_map_avg,
                "random_MAP_stdev": random_map_stdev,
                "random_samples_nb": random_samples_nb,
            }
        )
    )
    logger.info("Evaluation pipeline finished.")
    return 0
