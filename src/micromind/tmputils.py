from cartesio.dataset import DatasetReader
from micromind.microcell.profile.cell_profile import CellStainingProfile, ProfilingInfo
from micromind.experiment import HeterogeneityExp
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.manifold import TSNE

import numpy as np
import pandas as pd


def run_experiment(dataset_path):
    reader = DatasetReader(dataset_path)
    infos_path = reader / "INFOS.json"
    infos = ProfilingInfo(infos_path)
    profiling = CellStainingProfile(infos)
    exp = HeterogeneityExp(profiling, reader)
    results = exp.run()
    return results


def save_with_infos(csv_name, results, drop_list=[]):
    dataset = pd.DataFrame(results["all"], index=range(1, len(results["all"]) + 1))
    dataset_with_infos = dataset.copy()
    dataset_with_infos["image"] = results["image"]
    dataset_with_infos["n"] = results["n"]
    dataset_with_infos["days"] = dataset_with_infos.apply(lambda row: int(row.image.split("-")[0].replace("D", "")),
                                                          axis=1)
    dataset_with_infos.to_csv(csv_name)
    dataset_num = dataset.drop(drop_list, axis=1)
    return dataset_num, dataset_with_infos


def classify_data(X, n_class, scale=False, embedding="UMAP", classifier="GMM", perplexity=30):
    if scale:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    if embedding == "UMAP":
        X_emb = umap.UMAP(n_neighbors=perplexity).fit_transform(X)
    elif embedding == "TSNE":
        X_emb = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)
    else:
        X_emb = X

    if classifier == "GMM":
        model = mixture.GaussianMixture(n_components=n_class, covariance_type="tied").fit(X_emb)
    p = model.predict(X_emb)

    if embedding is None:
        X_emb = umap.UMAP(n_neighbors=perplexity).fit_transform(X)

    return X_emb, p, model