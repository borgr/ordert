import argparse
import ast
import itertools
import json
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from load_data import NGRAM_RESULTS, CHALLENGES
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters, cohens_kappa
from sklearn.metrics import cohen_kappa_score

try:
    from ordert.transformers.borgr_code.load_data import LEN, SYN_DEPTH, HUMAN, blimp_human, blimp_gpt, blimp_txl, \
        blimp_lstm, blimp_5
except ImportError as e:
    import sys

    sys.path.append(os.path.dirname(__file__))
    from load_data import LEN, SYN_DEPTH, HUMAN

sns.set()
# plt.rc('legend', fontsize=20)
# plt.rc('labelsize', fontsize=20)
# plt.rc('xticks.labelsize', fontsize=20)
# # plt.rc('xticks.labelsize', fontsize=20)
params = {'legend.fontsize': 'large',
          # 'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)
# read from
BLIMP = "/cs/snapless/oabend/borgr/ordert/blimp/data"
POOL_SIZE = 16

BLIMP_SUPER_CAT = {"anaphor agreement": ["anaphor_gender_agreement", "anaphor_number_agreement"],
                   "argument structure": ["animate_subject_passive", "animate_subject_trans", "causative",
                                          "drop_argument", "inchoative", "intransitive", "passive_1", "passive_2",
                                          "transitive"],
                   "binding": ["principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
                               "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
                               "principle_A_reconstruction"],
                   "control/raising": ["existential_there_object_raising", "existential_there_subject_raising",
                                       "expletive_it_object_raising", "tough_vs_raising_1", "tough_vs_raising_2"],
                   "determiner noun agreement": ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
                                                 "determiner_noun_agreement_irregular_1",
                                                 "determiner_noun_agreement_irregular_2",
                                                 "determiner_noun_agreement_with_adj_1",
                                                 "determiner_noun_agreement_with_adj_2",
                                                 "determiner_noun_agreement_with_adj_irregular_1",
                                                 "determiner_noun_agreement_with_adj_irregular_2"],
                   "elipsis": ["ellipsis_n_bar_1", "ellipsis_n_bar_2"],
                   "filler gap": ["wh_questions_object_gap", "wh_questions_subject_gap",
                                  "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap",
                                  "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
                                  "wh_vs_that_with_gap_long_distance"],
                   "irregular forms": ["irregular_past_participle_adjectives", "irregular_past_participle_verbs"],
                   "island effects": ["adjunct_island", "complex_NP_ _island",
                                      "coordinate_structure_constraint_complex_left_branch",
                                      "coordinate_structure_constraint_object_extraction",
                                      "left_branch_island_echo_question", "left_branch_island_simple_question",
                                      "sentential_subject_island", "wh_island "],
                   "npi licensing": ["matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
                                     "only_npi_licensor_present", "only_npi_scope",
                                     "sentential_negation_npi_licensor_present", "sentential_negation_npi_scope"],
                   "quantifiers": ["existential_there_quantifiers_1", "existential_there_quantifiers_2",
                                   "superlative_quantifiers_1", "superlative_quantifiers_2"],
                   "subject verb agreement": ["distractor_agreement_relational_noun",
                                              "distractor_agreement_relative_clause",
                                              "irregular_plural_subject_verb_agreement_1",
                                              "irregular_plural_subject_verb_agreement_2",
                                              "regular_plural_subject_verb_agreement_1",
                                              "regular_plural_subject_verb_agreement_2"],
                   }


def accuracy_from_file(file):
    answers = correct_from_file(file)
    correct = sum(answers)
    wrong = len(answers) - correct
    accuracy = correct / len(answers) if answers else 0
    if wrong + correct == 0:
        print(f"corrupt file {file}")
    return accuracy


def average_correlation(orders, other_orders=None, pearson=True):
    corr = 0
    # ranks = orders
    ranks = []
    if not pearson:
        unique_orders = []
        for order in orders:
            unique_order = []
            for item in order:
                if item not in unique_order:
                    unique_order.append(item)
            unique_orders.append(unique_order)
        orders = unique_orders
    for order in orders:
        if pearson:
            ranks.append(order)
        else:
            ranks.append([orders[0].index(item) for item in order])
    if other_orders is None:
        pairs = itertools.combinations(ranks, 2)
    else:
        pairs = itertools.product(orders, other_orders)
    for pair_num, (rank_a, rank_b) in enumerate(pairs):
        if pearson:
            corr += pearsonr(rank_a, rank_b)[0]
        else:
            corr += spearmanr(rank_a, rank_b)[0]
    corr = corr / (pair_num + 1)
    return corr


def learnt_orders(df, scores, measure="steps", pearson=True):
    """
    returns the order of each challenge in each step as a list (model, steps, order)
    :param df:
    :param scores:
    :return:
    """
    orders = []
    df = df.drop_duplicates(["model", "challenge", measure])
    for model in df["model"].unique():
        if pearson:
            complexity_order = df[(df[measure] == scores) & (df["model"] == model)].sort_values("challenge")[
                "accuracy"].tolist()
        else:
            complexity_order = df[(df[measure] == scores) & (df["model"] == model)].sort_values("accuracy")[
                "challenge"].tolist()
        if complexity_order:
            if orders and len(orders[0][-1]) != len(complexity_order):
                print(
                    f"warning wrong lengths in scores {scores} model {model} and {df['model'].unique()[-1]}, skipping ")
            else:
                orders.append((model, scores, complexity_order))
    return orders


def learnt_perp_orders(df, perplexity, pearson=True):
    """
    returns the order of each challenge in each step as a list (model, perplexity, order)
    :param df:
    :param perplexity:
    :return:
    """
    orders = []
    df = df.drop_duplicates(["model", "challenge", measure])
    for model in df["model"].unique():
        close_perp = find_nearest(df[df["model"] == model]["perplexity"].unique(), perplexity)
        if pearson:
            complexity_order = df[(df["perplexity"] == close_perp) & (df["model"] == model)].sort_values("challenge")[
                "accuracy"].tolist()
        else:
            complexity_order = df[(df["perplexity"] == close_perp) & (df["model"] == model)].sort_values("accuracy")[
                "challenge"].tolist()
        if complexity_order:
            if len(complexity_order) != 67 or (orders and len(orders[0][-1]) != len(complexity_order)):
                print(
                    f"warning wrong lengths in perplexity {perplexity} model {model} and {df['model'].unique()[-1]}, skipping ")
                return learnt_perp_orders(df[~((df["perplexity"] == close_perp) & (df["model"] == model))], perplexity)
            else:
                orders.append((model, perplexity, complexity_order))
    return orders


def correlate_with_base(df_base, df, name="", pearson=True, y_min=None):
    if name:
        name = name + "_"
    correlations_by_steps = []
    for steps in set(df["steps"].unique()) & set(df_base["steps"].unique()):
        orders = learnt_orders(df, steps, pearson=pearson)
        orders = [order[-1] for order in orders]
        base_orders = learnt_orders(df_base, steps, pearson=pearson)
        base_orders = [order[-1] for order in base_orders]
        cor = average_correlation(orders, base_orders, pearson=pearson)
        correlations_by_steps.append((steps, cor))
    correlations_by_steps = pd.DataFrame(correlations_by_steps, columns=["steps", "correlation"])
    ax = sns.lineplot(x="steps", y="correlation", data=correlations_by_steps)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 10000), ',')))
    # plt.legend(loc="best")
    plt.xlabel("10K Steps")
    plt.ylabel("Correlation")
    plt.ylim(bottom=y_min)
    if pearson:
        name = f"pearson_{name}"
    # plt.title("average spearman correlation of challenges rank as a function of steps")
    plt.savefig(os.path.join(graphs_path, f"{name}correlation_with_base_by_steps.png"))
    plt.clf()

    correlations_by_perplexity = []

    base_perplexities = base_df["perplexity"].unique()
    perplexities_range = np.linspace(base_perplexities.min(), base_perplexities.max(), 15)
    for perplexity in perplexities_range:
        orders = learnt_perp_orders(df, perplexity)
        orders = [order[-1] for order in orders]
        base_orders = learnt_perp_orders(df_base, perplexity)
        base_orders = [order[-1] for order in base_orders]
        cor = average_correlation(orders, base_orders, pearson=pearson)
        correlations_by_perplexity.append((perplexity, cor))
    correlations_by_perplexity = pd.DataFrame(correlations_by_perplexity, columns=["perplexity", "correlation"])
    sns.lineplot(x="perplexity", y="correlation", data=correlations_by_perplexity)
    # plt.legend(loc="best")
    plt.xlabel("Preplexity")
    plt.ylabel("Correlation")
    plt.ylim(bottom=y_min)
    plt.gca().invert_xaxis()

    # plt.title("average spearman correlation of challenges rank as a function of perplexity")
    plt.savefig(os.path.join(graphs_path, f"{name}correlation_with_base_by_perplexity.png"))
    plt.clf()


def correlate_sets_of_models(dfs, name="", save=True, pearson=True):
    max_steps = min((min(df.groupby(["model"])["steps"].max()) for _, df in dfs))
    for df_name, df in dfs:
        correlations_by_steps = calc_correlation_by_step(df[df["steps"] <= max_steps], measure="steps", pearson=pearson)
        ax = sns.lineplot(x="steps", y="correlation", data=correlations_by_steps, label=df_name.capitalize())
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 10000), ',')))
    plt.xlabel("10K Steps")
    plt.ylabel("Correlation")
    # plt.legend(loc="best")
    # plt.title("average spearman correlation of challenges rank as a function of steps")
    if pearson:
        name = f"pearson_{name}"
    if save:
        plt.savefig(os.path.join(graphs_path, f"{name}correlation_by_steps.png"))
    plt.clf()


def calc_correlation_by_step(df, measure="steps", pearson=True):
    correlations_by_steps = []
    expected_number_of_models = len(df["model"].unique())
    for steps in df[measure].unique():
        orders = learnt_orders(df, steps)
        orders = [order[-1] for order in orders]
        if len(orders) == expected_number_of_models:
            cor = average_correlation(orders, pearson=pearson)
            correlations_by_steps.append((steps, cor))
    correlations_by_steps = pd.DataFrame(correlations_by_steps, columns=[measure, "correlation"])
    return correlations_by_steps


def correlate_models(df, name="", save=True):
    # calculate correlation between models on how hard each phenomenon is
    if name:
        name = name + "_"
    correlations_by_steps = calc_correlation_by_step(df, measure="steps")
    ax = sns.lineplot(x="steps", y="correlation", data=correlations_by_steps)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 10000), ',')))
    plt.xlabel("10K Steps")
    plt.ylabel("Correlation")
    # plt.legend(loc="best")
    # plt.title("average spearman correlation of challenges rank as a function of steps")
    if save:
        plt.savefig(os.path.join(graphs_path, f"{name}correlation_by_steps.png"))
    plt.clf()


def plot_fields(df):
    # plot per challenge together (on line per challenge)
    group = df.groupby(["steps", "challenge"]).mean()
    group = group["accuracy"].unstack()
    for field in df["field"].unique():
        field_df = df[df["field"] == field]
        for challenge in field_df["challenge"].unique():
            ax = sns.lineplot(data=group[challenge],
                         label=challenge.capitalize())
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 10000), ',')))
        plt.xlabel("10K Steps")
        plt.ylabel("Accuracy")
        # plt.title("Averaged")
        plt.legend(loc="best").remove()
        # Shrink current axis by 20%
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        # Put a legend to the right of the current axis
        l = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(graphs_path, f"{field.replace(os.sep, '+')}.png"), bbox_extra_artists=(l,), bbox_inches='tight')
        plt.clf()


def plot_categories(df):
    # plot per challenge together (on line per challenge)
    group = df.groupby(["steps", "challenge"]).mean()
    group = group["accuracy"].unstack()
    for category in BLIMP_SUPER_CAT:
        for challenge in df["challenge"].unique():
            if challenge in BLIMP_SUPER_CAT[category]:
                ax = sns.lineplot(data=group[challenge],
                             label=challenge.capitalize())
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 10000), ',')))
        plt.xlabel("10K Steps")
        plt.ylabel("Accuracy")
        # plt.title("averaged")
        plt.legend(loc="best").remove()
        # Shrink current axis by 20%
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        # Put a legend to the right of the current axis
        l = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(graphs_path, f"{category.replace(os.sep, '+')}.png"), bbox_extra_artists=(l,), bbox_inches='tight')
        plt.clf()


def all_challenges(df):
    # plot per challenge together (on line per challenge)
    group = df.groupby(["steps", "challenge"]).mean()
    group = group["accuracy"].unstack()
    for challenge in df["challenge"].unique():
        ax = sns.lineplot(data=group[challenge],
                     label=challenge.capitalize())
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 10000), ',')))
    plt.xlabel("10K Steps")
    plt.ylabel("Accuracy")
    # plt.title("averaged")
    plt.legend(loc="best").remove()
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    l = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 4})
    plt.savefig(os.path.join(graphs_path, f"aaggregation_steps.png"), bbox_extra_artists=(l,), bbox_inches='tight')
    plt.clf()


def rename_models(legend):
    for name in legend.get_texts():
        new_name = name.get_text()
        if new_name == "gpt2":
            new_name = "GPT2$_{small}$"
        elif "seed" in new_name and "gpt" in new_name:
            new_name = "GPT2$_{tiny}^" + new_name[-1] + "$"
        if "gpt2Small" in new_name:
            new_name = "GPT$_{tiny" + new_name[-1] + "}$"
        new_name = new_name.replace("gpt", "GPT")
        new_name = new_name.replace("xl", "TransformerXL")
        new_name = new_name.replace("TransformerXLSmallTransformerXL", "XL$_{Small}")
        name.set_text(new_name)


def average_accuracy(df, plot_steps=True, plot_perplexity=True, max_perp=30):
    out_path = os.path.join(graphs_path)
    os.makedirs(out_path, exist_ok=True)
    # df = df.groupby("challenge").mean()
    # for challenge in df["challenge"].unique():

    measure = "Steps" if plot_steps else "Perplexity"

    for model in sorted(df["model"].unique()):
        line = df[df["model"] == model].groupby([measure.lower()]).mean()
        ax = sns.lineplot(x=line.index, y="accuracy", data=line,
                     label=model.capitalize())
    # plt.title(challenge)
    l = plt.legend(loc="best",  bbox_to_anchor=(1, 0.5))
    rename_models(l)
    if measure == "Steps":
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 10000), ',')))
        plt.xlabel("10K Steps")
    else:
        plt.xlabel(measure)
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(out_path, f"average_{measure.lower()}.png"), bbox_extra_artists=(l,), bbox_inches='tight')
    plt.clf()


def per_challenge(df, plot_steps=True, plot_perplexity=True, max_perp=30):
    # Plot line per model (each challenge on a separate plot)
    out_path = os.path.join(graphs_path, "per_challenge")
    os.makedirs(out_path, exist_ok=True)
    for challenge in df["challenge"].unique():
        if plot_steps:
            for model in sorted(df["model"].unique()):
                ax = sns.lineplot(x="steps", y="accuracy", data=df[(df["model"] == model) & (df["challenge"] == challenge)],
                             label=model)
            # plt.title(challenge)
            l = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            rename_models(l)
            ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 10000), ',')))
            plt.xlabel("10K Steps")
            plt.ylabel("Accuracy")
            plt.savefig(os.path.join(out_path, f"{challenge}_steps.png"), bbox_extra_artists=(l,), bbox_inches='tight')
            plt.clf()
        if plot_perplexity:
            df_perp = df[df["perplexity"] < max_perp]
            # Initialize figure and ax
            fig, ax = plt.subplots()
            ax.set(xscale="log")

            for i, model in enumerate(sorted(df_perp["model"].unique())):
                sns.lineplot(x="perplexity", y="accuracy", ax=ax,
                             data=df_perp[(df_perp["model"] == model) & (df_perp["challenge"] == challenge)],
                             label=model)
            ax.invert_xaxis()
            # plt.title(challenge.capitalize())
            ax.grid(False)
            plt.xlabel("Perplexity")
            plt.ylabel("Accuracy")
            l = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            rename_models(l)
            plt.savefig(os.path.join(out_path, f"{challenge}_perplexity.png"), bbox_extra_artists=(l,), bbox_inches='tight')
            plt.clf()


def correct_from_file(file):
    res = []
    with open(file) as fl:
        for i, line in enumerate(fl):
            lm_loss = float(line.strip().strip("[]"))
            if i % 2 == 1:  # lm_loss is like perplexity (need e^ [loss * token num]), lower is better
                bad_loss = lm_loss
                if bad_loss > good_loss:
                    res.append(1)
                else:
                    res.append(0)
            else:
                good_loss = lm_loss
    if len(res) > 1000:
        print(f"Wrong number of lines, assuming to many time written {len(res)} {file}")
        res = res[:1000]
    assert len(res) == 1000, f"{len(res)} {file}"
    return res


def find_first(array, value):
    for i, arr_val in enumerate(array):
        if arr_val - value > 0:
            return arr_val
    return arr_val


def find_nearest(array, value):
    n = [abs(i - value) for i in array]
    idx = n.index(min(n))
    return array[idx]


def calculate_outer_agreement(models_df, base_df):
    # Plot line per model (each challenge on a separate plot)
    kappas = []
    base_perplexities = base_df["perplexity"].unique()
    perplexities_range = np.linspace(base_perplexities.min(), base_perplexities.max(), 10)
    print("Calculating outer agreement...")
    for challenge in models_df["challenge"].unique():
        for perplexity in perplexities_range:
            sub_df = models_df[(models_df["challenge"] == challenge)]
            corrects = []
            for model in models_df["model"].unique():
                close_perp = find_nearest(sub_df[sub_df["model"] == model]["perplexity"].unique(), perplexity)
                correct = sub_df[(sub_df["model"] == model) & (sub_df["perplexity"] == close_perp)]["correct"].tolist()
                if correct:
                    corrects.append(ast.literal_eval(correct[0]))
            correct = [1 if x > 0.5 else 0 for x in np.mean(corrects, axis=0)]
            close_perp = find_nearest(base_df["perplexity"].unique(), perplexity)
            base_correct = base_df[(base_df["perplexity"] == close_perp)]["correct"].tolist()
            base_correct = ast.literal_eval(base_correct[0])
            # raters = aggregate_raters(np.array(corrects).T, 2)[0]
            kappas.append((challenge, perplexity, cohen_kappa_score(base_correct, correct)))
    df = pd.DataFrame(kappas, columns=["challenge", "perplexity", "kappa"])
    group = df.groupby(["perplexity", "challenge"]).mean()
    print(group)
    print(group.mean())
    # df.groupby(["challenge"]).mean()["kappa"].to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/per_challenge_kappa.csv")
    # df.groupby(["challenge", "steps"]).mean()["kappa"].to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/per_challenge_steps_kappa.csv")
    # df.to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/kappas.csv")


def calculate_inner_agreement(df):
    # Plot line per model (each challenge on a separate plot)
    kappas = []
    print("Calculating inner agreement...")
    for challenge in df["challenge"].unique():
        for steps in df["steps"].unique():
            sub_df = df[(df["steps"] == steps) & (df["challenge"] == challenge)]
            corrects = []
            for model in df["model"].unique():
                correct = sub_df[sub_df["model"] == model]["correct"].tolist()
                if correct:
                    corrects.append(ast.literal_eval(correct[0]))
            if len(corrects) > 3:
                raters = aggregate_raters(np.array(corrects).T, 2)[0]
                kappas.append((challenge, steps, fleiss_kappa(raters)))
    df = pd.DataFrame(kappas, columns=["challenge", "steps", "kappa"])
    group = df.groupby(["steps", "challenge"]).mean()
    # df.groupby(["challenge"]).mean()["kappa"].to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/per_challenge_kappa.csv")
    # df.groupby(["challenge", "steps"]).mean()["kappa"].to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/per_challenge_steps_kappa.csv")
    # df.to_csv("/home/leshem/PycharmProjects/ordert/ordert/transformers/output/kappas.csv")


def get_properties(file, path):
    with open(os.path.join(path, file + ".jsonl")) as fl:
        line = json.loads(fl.readline())
        return line["field"], line["linguistics_term"]


def acquire_statistics_from_file(root, filename):
    accuracy = accuracy_from_file(os.path.join(root, filename))
    correct = correct_from_file(os.path.join(root, filename))
    challenge = filename[:-len(".txt")]
    field, phenomenon = get_properties(challenge, BLIMP)
    return phenomenon, challenge, field, accuracy, correct


def get_ranks_per_measure(df, measure="steps", pearson=True):
    average = df.groupby([measure, "challenge"]).mean()
    average["model"] = ["average_model"] * len(average)
    average.reset_index(level=average.index.names, inplace=True)
    orders = []
    for steps in average[measure].unique():
        orders.append(learnt_orders(average, steps, measure, pearson=pearson)[0][1:])
    orders.sort(key=lambda x: x[0])
    return orders


def correlate_dynamics(df, metrics, name="", measure="steps", mark_x=None, mark_text=None, pearson=True, xscale=1, bbox=True):
    """
    compare the per step correlation of the (average over the) df with each metric [{challenge:score}]
    :param df:
    :param metrics:
    :return:
    """
    if mark_x is None:
        mark_x = itertools.repeat([])
    if mark_text is None:
        mark_text = itertools.repeat([])

    step_ranks = get_ranks_per_measure(df, measure, pearson=pearson)
    for (metric_name, metric), marks, texts in zip(metrics, mark_x, mark_text):
        metric_order = sorted(CHALLENGES.copy())
        if pearson:
            metric_order = [metric[x] for x in metric_order]
        else:
            metric_order.sort(key=lambda x: metric[x])
        correlations = []
        steps = []
        for step, order in step_ranks:
            if pearson:
                rank = order
            else:
                rank = [metric_order.index(item) for item in order]
            if rank and len(rank) != len(metric_order):
                continue
            if pearson:
                corr = pearsonr(rank, metric_order)[0]
            else:
                corr = spearmanr(rank, list(range(len(metric_order))))[0]
            correlations.append(corr)
            steps.append(step)
        marks = [steps.index(mark) for mark in marks]
        if not marks:
            marks = None
            marker = None
        else:
            marker = "D"
        ax = sns.lineplot(steps, correlations, label=metric_name.capitalize(), legend=False, markevery=marks,
                          marker=marker)
        if marks and texts:
            for mark, text in zip(marks, texts):
                if int(text) < 1:
                    text *= 100
                text = int(text)
                ax.text(steps[mark], correlations[mark] + 0.01, f"{text}", color=ax.lines[-1].get_color())

    # plt.legend(loc="best")
    min_x, max_x = ax.get_xlim()
    ax.set_xlim(min_x, max_x * xscale)
    ax.locator_params(nbins=4, axis='y')
    if bbox:
        l = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        l = plt.legend(loc="best")
    plt.ylabel("Correlation")
    if measure.lower() != "steps":
        name = f"{measure.lower()}_{name}"
        plt.xlabel(f"{measure.capitalize()}")
    else:
        plt.xlabel(f"10K {measure.capitalize()}")
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / 10000), ',')))
    if pearson:
        name = f"pearson_{name}"
    # plt.title("spearman correlation with metrics as a function of steps")
    plt.savefig(os.path.join(graphs_path, f"{name}metrics.png"), bbox_extra_artists=(l,), bbox_inches='tight')
    plt.clf()


def get_results_df(args, path):
    heavy_path = os.path.join(path, "results.csv")
    if args.heavy:
        csv_path = heavy_path
    else:
        csv_path = os.path.join(path, "results_light.csv")

    base_model = "gpt2Small"
    # skip_models = ["short"]
    # force recalculation of csv
    force = args.force

    if force or ((not os.path.isfile(csv_path)) and not os.path.isfile(heavy_path)):
        metadatas = []
        paths = []
        for root, dirnames, filenames in os.walk(path):
            results_dir = os.path.basename(root)
            model_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
            if results_dir.startswith("steps"):
                steps = int(results_dir.split("_")[0][len("steps"):])
                perplexity = float(results_dir.split("perplexity")[1])
                for filename in filenames:
                    paths.append((root, filename))
                    metadatas.append([model_name, steps, perplexity])
                    # if filename == "wh_island.txt":
                    # print(res[-1])
                    # print((model_name, filename, steps, perplexity, acc))
        headers = ["model", "steps", "perplexity", "phenomenon", "challenge", "field", "accuracy", "correct"]
        res = []
        # for path, metadata in zip(paths, metadatas):
        #     vals = acquire_statistics_from_file(*path)
        #     res.append(metadata + vals)

        chunksize = 1
        if (len(metadatas) > 100):
            chunksize = int(len(metadatas) / POOL_SIZE / 10)
        pool = Pool(POOL_SIZE)
        res = pool.starmap(acquire_statistics_from_file, paths)
        res = [data + list(stats) for data, stats in zip(metadatas, res)]
        df = pd.DataFrame(res, columns=headers)
        if not args.heavy:
            df.drop(["correct"], axis=1, inplace=True)
        print(df)
        df.to_csv(csv_path, index=False)
        print(f"wrote to {csv_path}")
    elif os.path.isfile(csv_path):
        print(f"reading cached {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"reading heavy {heavy_path}")
        df = pd.read_csv(heavy_path)
        df.drop(["correct"], axis=1, inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"wrote light version to {csv_path}")
    return df


def data_from_name(name):
    if "open" in name:
        return "open"
    if "news" in name:
        return "news"
    if "webtxt" in name:
        return "webtxt"
    return "bert"


def get_best_models(df, measure="perplexity", maximum=False):
    best_models = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        best_model = []
        best_score = float("inf")
        # repeats in case best score evaluation halted in the middle of blimp eval
        while len(best_model) != 67 and not model_df.empty:
            model_df = model_df[model_df[measure] != best_score]
            if maximum:
                best_score = model_df[measure].max()
            else:
                best_score = model_df[measure].min()
            best_model = model_df[model_df[measure] == best_score]
        best_models.append(best_model)
    return pd.concat(best_models)


def challenge_accuracies(df):
    df = df.set_index("challenge")
    df = df["accuracy"]
    return df.to_dict()


def get_closest_accuracies_by_measure(df, accuracy_tuples, measure="steps"):
    accuracies = [np.mean(list(accuracy_dict.values())) for name, accuracy_dict in accuracy_tuples]
    averaged_df = df.groupby([measure]).mean()
    if measure == "blimpAvgAcc":
        df_accuracies = averaged_df.index
    else:
        df_accuracies = averaged_df["blimpAvgAcc"]
    matched_accs = [[find_nearest(df_accuracies.unique(), accuracy)] for accuracy
                    in accuracies]
    matched_steps = [sorted(averaged_df[df_accuracies == acc].index)[:1] for acc in matched_accs]

    formatted_accuracies = [[x] for x in accuracies]
    return matched_steps, matched_accs, formatted_accuracies


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "-f", "--force", action="store_true", help="Recompute."
    )
    parser.add_argument(
        "--heavy", action="store_true", help="Rely on all data 'heavy' run."
    )
    args = parser.parse_args()

    # Ignore if less than:
    min_steps = 999
    max_perp = 1000

    # write to:
    path = os.path.abspath(os.path.dirname(__file__)) + r"/../output"
    # os.makedirs(path, exist_ok=True)
    out_path = path

    print("creating df...")
    df = get_results_df(args, path)
    print("got results, analysing...")

    # add blimp average accuracy
    df["blimpAvgAcc"] = df.groupby(["steps", "model"]).transform("mean")["accuracy"]
    # filter df
    graphs_path = os.path.join(out_path, "graphs")
    os.makedirs(graphs_path, exist_ok=True)
    # base_name = "GPT2 small}$"
    # df["model"] = df["model"].apply(lambda name: name.replace("gpt2Smallseed", base_name).replace("gpt", "GPT"))
    xl_df = df[df["model"] == "xl"]
    gpt_df = df[(df["model"] != "xlSmallxl") & (df["model"] != "xl")]
    df["base_model"] = df["model"].apply(lambda x: "gpt2Smallseed" in x)
    df = df[df["steps"] > min_steps]
    df["data"] = df["model"].apply(data_from_name)
    other_gpt_smalls = df[(df["data"] != "bert") & (df["model"].str.contains("gpt2Small"))]
    gpt_per_data = pd.concat([other_gpt_smalls, df[df["model"] == "gpt2Smallseed5"]])
    base_df = df[df["base_model"]]
    base_df = base_df[base_df["model"] != "gpt2Smallseed5"]

    # clean irrelevant models
    df = df[df["model"] != "bowgpt"]
    df = df[df["model"] != "bowgpt2"]
    df = df[df["model"] != "bowgpt3"]

    # correlations
    model_metrics = [("Bi-gram", NGRAM_RESULTS[1]),
                     ("Tri-gram", NGRAM_RESULTS[2]), ("quad-gram", NGRAM_RESULTS[3]), ("five-gram", NGRAM_RESULTS[4])]
    metrics = [("Human", HUMAN), ("Len", LEN), ("Syntactic\nDepth", SYN_DEPTH), ("Uni-gram", NGRAM_RESULTS[0]),
               ("Bi-gram", NGRAM_RESULTS[1]),
               ("Tri-gram", NGRAM_RESULTS[2]), ("quad-gram", NGRAM_RESULTS[3]), ("five-gram", NGRAM_RESULTS[4])]
    bow_last = df[df["model"] == "bowfc"]["perplexity"].min()
    bow_last = df[(df["model"] == "bowfc") & (df["perplexity"] == bow_last)]
    metrics.append(("bow", {row["challenge"]: row["accuracy"] for _, row in bow_last.iterrows()}))
    model_metrics.append(metrics[-1])
    window_last = df[df["model"] == "bowwindow5fc"]["perplexity"].min()
    window_last = df[(df["model"] == "bowwindow5fc") & (df["perplexity"] == window_last)]
    metrics.append(("window-5", {row["challenge"]: row["accuracy"] for _, row in window_last.iterrows()}))
    model_metrics.append(metrics[-1])
    best_models = get_best_models(df)
    gpt2_best = ("GPT2$_{small}$", challenge_accuracies(best_models[best_models["model"] == "gpt2"]))
    xlSmall_best = ("xl$_{small}$", challenge_accuracies(best_models[best_models["model"] == "xlSmallxl"]))
    xl_best = ("xl$_{ours}$", challenge_accuracies(best_models[best_models["model"] == "xl"]))

    gptSmall_best = ("GPT2$_{tiny}$", challenge_accuracies(best_models[best_models["model"] == "gpt2Smallseed2"]))
    news_gpt2Small_best = (
        "News GPT2$_{tiny}$", challenge_accuracies(best_models[best_models["model"] == "news_gpt2Small"]))
    # gpt2_steps_metrics = [gpt2_best]
    gpt2_steps = [10, 50, 100, 150, 200, 400, 800]
    gpt2_steps_metrics = [
        (f"gpt2 {step}K", challenge_accuracies(df[(df["model"] == "gpt2") & (df["steps"] == step * 1000)]))
        for step in gpt2_steps]

    # print("metric", window_last, metrics[-1])
    # raise

    # plot_fields(df)
    challenge_df = pd.concat([base_df, df[df["model"] == "gpt2"], xl_df])
    per_challenge(challenge_df, max_perp=max_perp)
    all_challenges(challenge_df)
    plot_categories(base_df)
    print(f"models in df {challenge_df['model'].unique()}")

    if args.heavy:
        calculate_outer_agreement(base_df, df[df["model"] == "gpt2"])
        # calculate_inner_agreement(base_df)

    blimp_metrics = [("5 gram", blimp_5), ("LSTM", blimp_lstm), ("XL", blimp_txl), ("GPT2$_{large}$", blimp_gpt),
                     ("Human", blimp_human)]
    for a in blimp_metrics:
        for b in blimp_metrics:
            if a == b:
                continue
            accs_a = []
            accs_b = []
            for key in a[1]:
                accs_a.append(a[1][key])
                accs_b.append(b[1][key])
            print(a[0], b[0], spearmanr(accs_a, accs_b))
    average_accuracy(df, max_perp=max_perp)

    nn_metrics = [x for x in blimp_metrics if "gram" not in x[0]] + [xlSmall_best, xl_best,
                                                                     gptSmall_best]  # , gpt2_best, gptSmall_best] # news_gpt2Small_best
    web_df = df[df["data"] == "webtxt"]

    five_grams = [("WikiBooks", NGRAM_RESULTS[4]), ("Giga", blimp_5)]
    closest, base_accs, accs = get_closest_accuracies_by_measure(base_df, five_grams)
    correlate_dynamics(base_df, five_grams,
                       name="5gram", mark_x=closest, mark_text=accs, bbox=False)#, xscale=1.2)
    closest, base_accs, accs = get_closest_accuracies_by_measure(web_df, five_grams)
    correlate_dynamics(web_df, five_grams,
                       name="web5gram", mark_x=closest, mark_text=accs, bbox=False)#, xscale=1.2)


    closest, base_accs, accs = get_closest_accuracies_by_measure(base_df, nn_metrics)
    correlate_dynamics(base_df, nn_metrics,
                       name="best", mark_x=closest, mark_text=accs)#, xscale=1.2)
    closest, base_accs, accs = get_closest_accuracies_by_measure(base_df, gpt2_steps_metrics)
    correlate_dynamics(base_df, gpt2_steps_metrics,
                       name="gpt2steps", mark_x=closest, mark_text=accs)
    closest, base_accs, accs = get_closest_accuracies_by_measure(web_df, gpt2_steps_metrics)
    correlate_dynamics(web_df, gpt2_steps_metrics,
                       name="webtxtgpt2steps", mark_x=closest, mark_text=accs)
    measure = "blimpAvgAcc"
    closest, base_accs, accs = get_closest_accuracies_by_measure(base_df, nn_metrics, measure=measure)
    correlate_dynamics(base_df, nn_metrics,
                       name="best", measure=measure, mark_x=closest, mark_text=accs)
    closest, base_accs, accs = get_closest_accuracies_by_measure(base_df, model_metrics)
    correlate_dynamics(base_df, model_metrics, name="models", mark_x=closest, mark_text=accs, bbox=False)
    correlate_dynamics(base_df, metrics)
    correlate_dynamics(df[df["model"] == "gpt2"], metrics, name="gptbig")
    correlate_dynamics(base_df, blimp_metrics, name="blimp")
    correlate_dynamics(df[df["model"] == "gpt2"], blimp_metrics, name="gptbigBlimp")
    correlate_dynamics(xl_df, metrics, name="xl")#, xscale=1.3)
    correlate_with_base(base_df, other_gpt_smalls, name="datacompare")
    for model in other_gpt_smalls["model"].unique():
        name = model.split("_")[0]
        correlate_dynamics(df[df["model"] == model], blimp_metrics, name=name + "blimp")
        correlate_with_base(base_df, other_gpt_smalls[other_gpt_smalls["model"] == model], name=name)

    correlate_with_base(base_df, df[df["model"] == "xl"], "gptToxl")
    # correlate_with_base(df[df["model"] == "xlSmallxl"], df[df["model"] == "xl"], "xlTosmall")
    correlate_with_base(base_df, df[df["model"] == "gpt2"], y_min=0.5)

    correlate_sets_of_models([("Init", base_df), ("Data", gpt_per_data)], name="both")
    correlate_models(gpt_per_data, name="datacompare")
    correlate_models(other_gpt_smalls, name="otherdata")
    correlate_models(base_df)
