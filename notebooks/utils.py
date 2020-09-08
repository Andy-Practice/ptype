import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_data(dataset_name, header=None):
    dataset_path = "../data/" + dataset_name + ".csv"
    return pd.read_csv(
        dataset_path,
        sep=",",
        encoding="ISO-8859-1",
        dtype=str,
        header=header,
        keep_default_na=False,
        skipinitialspace=True,
    )


def subsample_df(df, column_to_sample_from, sample_num):
    return df[[column_to_sample_from]].sample(n=sample_num, random_state=1)


def plot_bar(classes, values, title, xlabel=None, ylabel=None, y_lim_max=1.0):
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), classes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.0, y_lim_max)
    plt.title(title)
    plt.show()


def plot_column_type_posterior(p_t, types):
    # p_t has subtypes of date separately and is not ordered alphabetically
    organized_p_t = {}
    for i, t in types:
        # maps subtypes to types (date-iso-8601 to date)
        t_ = t.split("-")[0]

        # sum the subtypes of dates
        if t_ in organized_p_t:
            organized_p_t[t_] += p_t[i - 1]
        else:
            organized_p_t[t_] = p_t[i - 1]

    if len(np.unique(p_t)) == 1:
        organized_p_t = {t: 1 / len(organized_p_t) for t in organized_p_t}

    # sort the posteriors
    sorted_types = sorted(organized_p_t.keys(), key=lambda x: x.lower())
    sorted_posteriors = [organized_p_t[t] for t in sorted_types]

    plot_bar(
        sorted_types,
        sorted_posteriors,
        title="p(t|x): posterior dist. of column type",
        xlabel="type",
        ylabel="posterior probability",
    )


def plot_arff_type_posterior(
    arff_posterior, types=["date", "nominal", "numeric", "string"]
):
    plot_bar(
        types,
        arff_posterior,
        title="posterior dist. of column ARFF type",
        xlabel="type",
        ylabel="posterior probability",
    )


def plot_row_type_posterior(col, t="missing"):
    if t == "missing":
        i = 1
    elif t == "anomaly":
        i = 2
    plot_bar(
        col.unique_vals,
        col.p_z[:, i],
        title="p(z_i=" + t + "|t,x_i): posterior dist. for row type",
        xlabel="unique value",
        ylabel="posterior " + t + " probability",
    )
