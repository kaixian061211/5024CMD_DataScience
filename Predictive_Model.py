"""
Predictive_Model.py
-------------------
5024CMD Data Science - Individual Portfolio

Report section 4: predictive modelling.

Reads the CleanedData table and builds a simple prediction model for
whether a client subscribes to a term deposit.

Four algorithms are compared so that the choice of model can be
justified rather than assumed:
    Decision Tree, Logistic Regression, Random Forest, Naive Bayes

Each is trained on two feature sets:
    A  WITH duration    - benchmark only. Call length is known only
                          after the call has ended, so it cannot be
                          used to decide whom to call.
    B  WITHOUT duration - the deployable model, which uses only
                          information available before contact.

The banded attributes created during preprocessing are used as the
inputs, so the discretisation carried out in stage 2 feeds directly
into the model.

A final check re-runs the Variant B setup with day and previous
added, because neither was banded in stage 2 and so neither entered
the original feature list.

Run from the Portfolio folder:  python Predictive_Model.py
"""

import os
import sqlite3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, \
    train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_text

DATABASE = "data/bank_marketing.db"
FIGURES = "output/figures"
REPORT = "output/predictive_model_report.txt"

os.makedirs(FIGURES, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.30

# Inputs available BEFORE the call is placed
BASE_FEATURES = [
    "age_group", "job_group", "marital", "education",
    "credit_default", "balance_group", "housing", "loan",
    "contact_type", "month_group", "campaign_group",
    "pdays_group", "contacted_before", "poutcome"
]

# Only known AFTER the call has ended
LEAKING_FEATURE = "duration_group"

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def load():
    if not os.path.exists(DATABASE):
        raise FileNotFoundError(
            f"Database not found at {DATABASE}. Run Init_Database.py "
            f"and Data_Preprocessing.py first."
        )

    connection = sqlite3.connect(DATABASE)
    data = pd.read_sql_query("SELECT * FROM CleanedData", connection)
    connection.close()
    return data


def build_models():
    """
    The four candidate algorithms.

    Each is deliberately kept simple, as the brief asks for a simple
    prediction model. The decision tree is depth-limited so that the
    rules it produces can be read and quoted in the report.
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50,
            random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Naive Bayes": BernoulliNB()
    }


def evaluate(models, x_train, x_test, y_train, y_test, x_all, y_all):
    """Train each model, then score it on the held-out test set."""
    folds = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    results = []
    fitted = {}
    fold_scores = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted[name] = model

        predicted = model.predict(x_test)
        probability = model.predict_proba(x_test)[:, 1]

        cross_validated = cross_val_score(
            model, x_all, y_all, cv=folds, scoring="f1"
        )
        # Same folds again (fixed random_state), scored on AUC, so the
        # stability of AUC can be judged as well as that of F1
        cross_validated_auc = cross_val_score(
            model, x_all, y_all, cv=folds, scoring="roc_auc"
        )
        fold_scores[name] = {"F1": cross_validated,
                             "AUC": cross_validated_auc}

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predicted),
            "Precision": precision_score(y_test, predicted),
            "Recall": recall_score(y_test, predicted),
            "F1": f1_score(y_test, predicted),
            "AUC": roc_auc_score(y_test, probability),
            "CV F1 mean": cross_validated.mean(),
            "CV F1 sd": cross_validated.std(),
            "CV AUC mean": cross_validated_auc.mean(),
            "CV AUC sd": cross_validated_auc.std()
        })

    return pd.DataFrame(results), fitted, fold_scores


def run_variant(data, features, label):
    """Encode, split, train and evaluate one feature set."""
    log("=" * 72)
    log(f" {label}")
    log("=" * 72)
    log(f"Input attributes: {len(features)}")
    log(f"   {', '.join(features)}")
    log()

    encoded = pd.get_dummies(data[features], drop_first=False)
    target = data["subscribed"]

    log(f"After one-hot encoding: {encoded.shape[1]} binary columns")

    x_train, x_test, y_train, y_test = train_test_split(
        encoded, target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE
    )

    log(f"Training rows: {len(x_train)}   Test rows: {len(x_test)}")
    log(f"Positive class in training: {y_train.mean() * 100:.1f}%   "
        f"in test: {y_test.mean() * 100:.1f}%")
    log()

    models = build_models()
    table, fitted, fold_scores = evaluate(
        models, x_train, x_test, y_train, y_test, encoded, target
    )

    table = table.sort_values("F1", ascending=False).reset_index(drop=True)

    log("Held-out test results (sorted by F1)")
    log(table.round(3).to_string(index=False))
    log()

    best = table.iloc[0]["Model"]
    log(f"Best by F1: {best}")
    log()

    return {
        "label": label,
        "table": table,
        "fitted": fitted,
        "fold_scores": fold_scores,
        "best": best,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "features": list(encoded.columns)
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================
def report_confusion(variant):
    model = variant["fitted"][variant["best"]]
    predicted = model.predict(variant["x_test"])
    matrix = confusion_matrix(variant["y_test"], predicted)

    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]

    log(f"Confusion matrix - {variant['best']} ({variant['label']})")
    log(f"                    predicted no   predicted yes")
    log(f"   actual no  {true_negative:12}   {false_positive:13}")
    log(f"   actual yes {false_negative:12}   {true_positive:13}")
    log()
    log(f"   Correctly identified subscribers: {true_positive} of "
        f"{true_positive + false_negative}")
    log(f"   Wasted calls (predicted yes, said no): {false_positive}")
    log(f"   Missed subscribers (predicted no, said yes): "
        f"{false_negative}")
    log()

    # The same matrix read as a call list
    clients = int(matrix.sum())
    subscribers = int(true_positive + false_negative)
    flagged = int(true_positive + false_positive)
    log("   As a call list:")
    log(f"   Clients flagged to call: {flagged} of {clients} "
        f"({flagged / clients * 100:.1f}% of the list)")
    log(f"   Subscribers among them: {true_positive} "
        f"(hit rate {true_positive / flagged * 100:.1f}%, against "
        f"{subscribers / clients * 100:.1f}% calling at random)")
    log(f"   Share of all subscribers reached: "
        f"{true_positive / subscribers * 100:.1f}%")
    log(f"   Calling everyone instead: {clients} calls, {subscribers} "
        f"subscribers, {clients - subscribers} wasted calls")
    log()

    return matrix


# ============================================================
# DECISION TREE RULES
# ============================================================
def report_tree_rules(variant, data):
    tree = variant["fitted"]["Decision Tree"]

    log("=" * 72)
    log(f" DECISION TREE RULES ({variant['label']})")
    log("=" * 72)
    log()

    rules = export_text(
        tree,
        feature_names=variant["features"],
        max_depth=3,
        decimals=0
    )
    log(rules)
    log()

    # Node counts for the top three levels, so the figures quoted in the
    # report come from this log rather than from the tree diagram.
    # Counts are recomputed from the training rows that reach each node.
    structure = tree.tree_
    path = tree.decision_path(variant["x_train"])
    target = variant["y_train"].to_numpy()
    log("Training records at each node of the top three levels:")
    log(f"   {'rule':78} {'records':>8} {'no':>6} {'yes':>6} {'rate':>7}")

    def walk(node, depth, rule):
        rows = path[:, node].toarray().ravel().astype(bool)
        yes = int(target[rows].sum())
        total = int(rows.sum())
        label = rule if rule else "all training records"
        log(f"   {label:78} {total:>8} {total - yes:>6} {yes:>6} "
            f"{yes / total * 100:6.1f}%")
        if depth == 3 or structure.children_left[node] == -1:
            return
        name = variant["features"][structure.feature[node]]
        joiner = " and " if rule else ""
        walk(structure.children_left[node], depth + 1,
             f"{rule}{joiner}{name} = 0")
        walk(structure.children_right[node], depth + 1,
             f"{rule}{joiner}{name} = 1")

    walk(0, 0, "")
    log()

    # Figure: the top two levels, drawn directly so the text stays
    # readable at page width and the counts are whole numbers. The
    # counts come from the same training rows as the table above.
    def describe(column):
        attribute = max((a for a in BASE_FEATURES + [LEAKING_FEATURE]
                         if column.startswith(a + "_")), key=len)
        return f"{attribute} = {column[len(attribute) + 1:]}?"

    def counts(node):
        rows = path[:, node].toarray().ravel().astype(bool)
        return int(rows.sum()), int(target[rows].sum())

    overall = target.mean()
    colour_map = plt.get_cmap("RdBu")

    def fill(rate):
        # Red below the overall rate, blue above it, pale near it
        shade = 0.5 + (rate - overall) / (2 * max(overall, 1 - overall))
        # kept within the lighter part of the map so black text stays
        # readable on every box
        return colour_map(min(max(shade, 0.28), 0.72))

    # node, x, y, label of the edge leading into it
    left = structure.children_left
    right = structure.children_right
    nodes = [(0, 0.5, 0.86, None),
             (left[0], 0.25, 0.52, "no"), (right[0], 0.75, 0.52, "yes")]
    for parent, x in [(left[0], 0.25), (right[0], 0.75)]:
        nodes.append((left[parent], x - 0.125, 0.16, "no"))
        nodes.append((right[parent], x + 0.125, 0.16, "yes"))
    position = {node: (x, y) for node, x, y, _ in nodes}
    parent_of = {left[0]: 0, right[0]: 0}
    for parent in (left[0], right[0]):
        parent_of[left[parent]] = parent
        parent_of[right[parent]] = parent

    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    for node, x, y, edge in nodes:
        if node in parent_of:
            px, py = position[parent_of[node]]
            axis.annotate("", xy=(x, y + 0.1), xytext=(px, py - 0.1),
                          arrowprops=dict(arrowstyle="-|>", color="0.35"))
            axis.text((x + px) / 2 + (0.012 if edge == "yes" else -0.012),
                      (y + py) / 2, edge, fontsize=12, style="italic",
                      ha="left" if edge == "yes" else "right",
                      va="center", color="0.25")

        total, yes = counts(node)
        rate = yes / total
        lines = [f"{total:,} clients", f"{yes:,} subscribed ({rate:.1%})"]
        is_split = node in (0, left[0], right[0])
        if is_split:
            lines.append(describe(variant["features"][structure.feature[node]]))
        else:
            lines.append("(splits further)")
        axis.text(x, y, "\n".join(lines), ha="center", va="center",
                  fontsize=12, linespacing=1.5,
                  fontweight="normal",
                  bbox=dict(boxstyle="round,pad=0.6", facecolor=fill(rate),
                            edgecolor="0.3"))

    axis.set_title(
        f"Decision tree, top two of five levels ({variant['label']})",
        fontsize=13
    )
    figure.text(0.5, 0.01,
                "Training records. Each box asks the question shown; "
                "'yes' branches to the right. Colour: blue above the "
                f"overall {overall:.1%} rate, red below it.",
                ha="center", fontsize=10)
    figure.tight_layout(rect=[0, 0.03, 1, 1])
    figure.savefig(
        os.path.join(FIGURES, "fig12_decision_tree.png"), dpi=150
    )
    plt.close(figure)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
def report_importance(variant):
    model = variant["fitted"]["Random Forest"]

    importance = pd.Series(
        model.feature_importances_, index=variant["features"]
    ).sort_values(ascending=False).head(15)

    log("=" * 72)
    log(f" FEATURE IMPORTANCE - Random Forest ({variant['label']})")
    log("=" * 72)
    log()
    for name, value in importance.items():
        log(f"   {name:34} {value:.4f}")
    log()

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(range(len(importance)), importance.values[::-1],
              color="#4C72B0")
    axis.set_yticks(range(len(importance)))
    axis.set_yticklabels(importance.index[::-1], fontsize=9)
    axis.set_xlabel("Importance")
    axis.set_title(
        f"Top 15 features ({variant['label']})"
    )
    axis.grid(axis="x", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        os.path.join(FIGURES, "fig11_feature_importance.png"), dpi=110
    )
    plt.close(figure)


# ============================================================
# LOGISTIC REGRESSION COEFFICIENTS
# ============================================================
def report_coefficients(variant, attributes, data):
    """
    Every level of every attribute has its own column (drop_first=False),
    so a single coefficient has no fixed reference point. The meaningful
    quantity is the difference between two levels of the same attribute:
    exp(difference) is the odds ratio between them, holding the other
    inputs fixed. Each level is compared both with the attribute's
    lowest level (so every odds ratio is 1 or more) and with its most
    common level (a conventional reference category).
    """
    model = variant["fitted"]["Logistic Regression"]
    coefficients = pd.Series(model.coef_[0], index=variant["features"])

    # Map each one-hot column back to its attribute. The longest matching
    # name wins, because some names contain underscores (contact_type).
    def attribute_of(column):
        matches = [a for a in attributes if column.startswith(a + "_")]
        return max(matches, key=len)

    groups = {}
    for column, value in coefficients.items():
        attribute = attribute_of(column)
        level = column[len(attribute) + 1:]
        groups.setdefault(attribute, {})[level] = value

    ranges = {a: max(v.values()) - min(v.values())
              for a, v in groups.items()}
    # Gaps for these two attributes share the never-contacted effect
    shared = {"contacted_before", "pdays_group"}
    order = sorted(ranges, key=ranges.get, reverse=True)

    log("=" * 72)
    log(f" COEFFICIENTS - Logistic Regression ({variant['label']})")
    log("=" * 72)
    log()
    log("Default L2 regularisation (C = 1.0), so coefficients are shrunk")
    log("towards zero. Odds ratios compare two levels of one attribute")
    log("with every other input held fixed. contacted_before, the")
    log("'Not contacted' level of pdays_group and the 'unknown' level of")
    log("poutcome describe almost the same clients, so the model shares")
    log("that one effect between them and none of the three can be read")
    log("on its own.")
    log()
    log(f"Intercept: {model.intercept_[0]:+.3f}")
    log()

    log("Attributes ranked by the gap between their highest and lowest")
    log("level:")
    log(f"   {'attribute':18} {'highest level':16} {'lowest level':16}"
        f" {'gap':>6} {'odds ratio':>11}")
    for attribute in order:
        levels = groups[attribute]
        high = max(levels, key=levels.get)
        low = min(levels, key=levels.get)
        name = attribute + (" *" if attribute in shared else "")
        log(f"   {name:18} {high:16} {low:16}"
            f" {ranges[attribute]:6.3f} {np.exp(ranges[attribute]):11.2f}")
    log("   * gap shares the never-contacted effect (see note above)")
    log()

    log("Every level, with its odds ratio against the lowest level and")
    log("against the most common level of its attribute:")
    for attribute in order:
        levels = pd.Series(groups[attribute]).sort_values(ascending=False)
        counts = data[attribute].astype(str).value_counts()
        common = counts.idxmax()
        lowest = levels.min()
        log(f"   {attribute}   (most common level: {common}, "
            f"n = {counts[common]})")
        log(f"      {'level':16} {'n':>6} {'coefficient':>11}"
            f" {'OR vs lowest':>13} {'OR vs common':>13}")
        for level, value in levels.items():
            log(f"      {level:16} {counts.get(level, 0):>6} {value:+11.3f}"
                f" {np.exp(value - lowest):13.2f}"
                f" {np.exp(value - levels[common]):13.2f}")
    log()

    return groups, order


def figure_odds_ratios(variant, groups, order, data):
    """
    One bar per level: its odds ratio against a reference level of the
    same attribute, holding every other input fixed.

    Reference: the attribute's most common level, except for poutcome,
    whose reference is 'failure' so that 'success' is compared within
    previously contacted clients. contacted_before, pdays_group and the
    'unknown' level of poutcome are left out because they share the
    never-contacted effect and cannot be read on their own.
    """
    left_out = {"contacted_before", "pdays_group"}
    reference = {"poutcome": "failure"}
    skip_levels = {("poutcome", "unknown")}

    rows = []   # (attribute, label, odds ratio), top to bottom
    for attribute in order:
        if attribute in left_out:
            continue
        levels = groups[attribute]
        if attribute in reference:
            base = reference[attribute]
        else:
            base = data[attribute].astype(str).value_counts().idxmax()
        bars = sorted(
            ((level, np.exp(value - levels[base]))
             for level, value in levels.items()
             if level != base and (attribute, level) not in skip_levels),
            key=lambda item: item[1], reverse=True
        )
        for level, ratio in bars:
            rows.append((attribute, f"{attribute}: {level}  (vs {base})",
                         ratio))

    log("Values plotted in fig13_lr_odds_ratios.png (odds ratio of each")
    log("level against the reference shown):")
    for _, label, ratio in rows:
        log(f"   {label:46} {ratio:5.2f}")
    log()

    # y positions, with a small gap between attributes
    positions, y, previous = [], 0.0, None
    for attribute, _, _ in rows:
        if previous is not None and attribute != previous:
            y += 0.6
        positions.append(y)
        y += 1.0
        previous = attribute

    ratios = [ratio for _, _, ratio in rows]
    colours = ["#4C72B0" if ratio >= 1 else "#C44E52" for ratio in ratios]

    figure, axis = plt.subplots(figsize=(10, 9))
    # Bars start at 1 (no effect) and run left or right from it
    axis.barh(positions, [ratio - 1 for ratio in ratios], left=1,
              color=colours, height=0.7)
    axis.set_xscale("log")
    axis.axvline(1, color="black", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels([label for _, label, _ in rows], fontsize=9)
    axis.invert_yaxis()
    for position, ratio in zip(positions, ratios):
        axis.text(ratio * (1.04 if ratio >= 1 else 0.96), position,
                  f"{ratio:.2f}", va="center",
                  ha="left" if ratio >= 1 else "right", fontsize=8)
    axis.set_xlim(0.2, 15)
    axis.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    axis.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    axis.set_xlabel("Odds ratio (log scale): above 1 = more likely to "
                    "subscribe than the reference level")
    axis.grid(axis="x", alpha=0.3)
    axis.set_title(f"Logistic Regression odds ratios ({variant['label']})")
    figure.text(
        0.01, 0.01,
        "Reference: each attribute's most common level; for poutcome, "
        "'failure'. Other inputs held fixed.\n"
        "Not shown: contacted_before, pdays_group and poutcome = unknown, "
        "which share the never-contacted effect.",
        fontsize=8, ha="left", va="bottom"
    )
    figure.tight_layout(rect=[0, 0.05, 1, 1])
    figure.savefig(os.path.join(FIGURES, "fig13_lr_odds_ratios.png"),
                   dpi=110)
    plt.close(figure)


def report_month_contact(data):
    """
    month_group Q2 has the highest Logistic Regression coefficient but the
    lowest raw subscription rate. This block shows why: almost all the
    'unknown' contact_type records fall in Q2.
    """
    log("=" * 72)
    log(" CONTEXT FOR THE month_group COEFFICIENT")
    log("=" * 72)
    log()
    counts = pd.crosstab(data["month_group"], data["contact_type"])
    log("Records by quarter and contact type:")
    log(counts.to_string())
    log()
    unknown = data["contact_type"] == "unknown"
    log(f"unknown contact_type records: {unknown.sum()}, of which "
        f"{(unknown & (data['month_group'] == 'Q2')).sum()} are in Q2")
    log()
    log("Subscription rate by quarter, split by whether the contact type")
    log("was recorded:")
    log(f"   {'quarter':8} {'all':>14} {'recorded':>16} {'unknown':>16}"
        f" {'share unknown':>14}")
    for quarter in sorted(data["month_group"].unique()):
        rows = data["month_group"] == quarter
        known = rows & ~unknown
        missing = rows & unknown
        log(f"   {quarter:8}"
            f" {data.loc[rows, 'subscribed'].mean() * 100:6.1f}% (n={rows.sum():4})"
            f" {data.loc[known, 'subscribed'].mean() * 100:6.1f}% (n={known.sum():4})"
            f" {data.loc[missing, 'subscribed'].mean() * 100:6.1f}% (n={missing.sum():4})"
            f" {missing.sum() / rows.sum() * 100:13.1f}%")
    log()


# ============================================================
# LOGISTIC REGRESSION vs RANDOM FOREST, FOLD BY FOLD
# ============================================================
def compare_on_folds(variant, first, second):
    """
    Both models were cross-validated on the same five folds, so their
    scores can be compared fold by fold, which is a fairer test of a
    small difference than comparing two means.
    """
    log("=" * 72)
    log(f" {first} vs {second}, same five folds")
    log(f" ({variant['label']})")
    log("=" * 72)
    log()

    for metric in ["F1", "AUC"]:
        a = variant["fold_scores"][first][metric]
        b = variant["fold_scores"][second][metric]
        difference = b - a
        log(f"CV {metric} by fold:")
        log(f"   {'fold':>4} {first:>21} {second:>15} {'difference':>11}")
        for fold, (x, y, z) in enumerate(zip(a, b, difference), start=1):
            log(f"   {fold:>4} {x:>21.3f} {y:>15.3f} {z:>+11.3f}")
        log(f"   mean difference ({second} minus {first}): "
            f"{difference.mean():+.3f}")
        log(f"   sd of the differences: {difference.std():.3f}")
        log(f"   folds where {second} is higher: "
            f"{(difference > 0).sum()} of {len(difference)}")
        log()


# ============================================================
# CHECK: DEPENDENCE ON contact_type (selected model)
# ============================================================
def check_contact_type(data, variant_b):
    """
    Much of contact_type's value lies in its 'unknown' level, a gap in
    the records (2.1, 4.4). Two views of how much the selected model
    depends on it: (1) refit without contact_type, same split, seed and
    settings; (2) score the Variant B model separately on test clients
    whose contact type was recorded and on those whose was not.
    """
    name = "Logistic Regression"
    log("=" * 72)
    log(f" CHECK: DEPENDENCE ON contact_type ({name}, Variant B)")
    log("=" * 72)
    log()

    # (1) without contact_type
    features = [f for f in BASE_FEATURES if f != "contact_type"]
    encoded = pd.get_dummies(data[features], drop_first=False)
    target = data["subscribed"]
    x_train, x_test, y_train, y_test = train_test_split(
        encoded, target, test_size=TEST_SIZE, stratify=target,
        random_state=RANDOM_STATE
    )
    table, _, _ = evaluate({name: build_models()[name]},
                           x_train, x_test, y_train, y_test,
                           encoded, target)
    without = table.set_index("Model").loc[name]
    with_ = variant_b["table"].set_index("Model").loc[name]
    log("(1) Refit without contact_type (13 inputs):")
    log(f"   {'measure':12} {'with':>8} {'without':>8} {'change':>8}")
    for measure in ["F1", "AUC", "CV F1 mean", "CV AUC mean"]:
        a = round(with_[measure], 3)
        b = round(without[measure], 3)
        log(f"   {measure:12} {a:8.3f} {b:8.3f} "
            f"{round(b - a, 3) + 0.0:+8.3f}")
    log(f"   CV F1 sd without contact_type: {without['CV F1 sd']:.3f}   "
        f"CV AUC sd: {without['CV AUC sd']:.3f}")
    log("   (changes are differences of the rounded values)")
    log()

    # (2) the Variant B model on recorded vs unknown contact type
    model = variant_b["fitted"][name]
    x_b = variant_b["x_test"]
    y_b = variant_b["y_test"].to_numpy()
    predicted = model.predict(x_b)
    probability = model.predict_proba(x_b)[:, 1]
    unknown = x_b["contact_type_unknown"].to_numpy().astype(bool)
    log("(2) Variant B model scored on test clients split by whether the")
    log("    contact type was recorded:")
    log(f"   {'group':10} {'clients':>8} {'subscribers':>12} {'rate':>7}"
        f" {'flagged':>8} {'precision':>10} {'recall':>7} {'F1':>6}"
        f" {'AUC':>6}")
    for label, rows in [("recorded", ~unknown), ("unknown", unknown),
                        ("all", np.ones_like(unknown))]:
        y = y_b[rows]; p = predicted[rows]
        log(f"   {label:10} {rows.sum():8} {int(y.sum()):12}"
            f" {y.mean() * 100:6.1f}% {int(p.sum()):8}"
            f" {precision_score(y, p):10.3f} {recall_score(y, p):7.3f}"
            f" {f1_score(y, p):6.3f}"
            f" {roc_auc_score(y, probability[rows]):6.3f}")
    log()


# ============================================================
# CHECK: ADDING day AND previous
# ============================================================
def check_day_previous(data, variant_b):
    """
    day and previous were not banded in stage 2, so they never entered
    BASE_FEATURES. This was not decided in advance; it is tested here
    after the fact, using the Variant B split, seed and models.
    """
    log("=" * 72)
    log(" CHECK: ADDING day AND previous TO VARIANT B")
    log("=" * 72)
    log()

    # Work on a copy so nothing above is affected
    extended = data.copy()
    # day: one level per calendar day (31 one-hot columns)
    extended["day_level"] = extended["day"].astype(int).astype(str)
    # previous: capped, levels 0, 1, 2 and 3+ (4 one-hot columns)
    capped = extended["previous"].astype(int).clip(upper=3)
    extended["previous_level"] = capped.map(
        {0: "0", 1: "1", 2: "2", 3: "3+"}
    )

    never = extended["contacted_before"] == "no"
    zero = extended["previous"] == 0
    log("Overlap of previous with contacted_before:")
    log(f"   previous = 0                     : {zero.sum()}")
    log(f"   contacted_before = no            : {never.sum()}")
    log(f"   rows where the two disagree      : "
        f"{(zero != never).sum()} of {len(extended)}")
    log("   so the previous = 0 column duplicates contacted_before = no")
    log()

    # Same column order as the Variant B results (sorted by F1)
    models = list(variant_b["table"]["Model"])
    target = extended["subscribed"]
    added_prefixes = ("day_level_", "previous_level_")

    feature_sets = [
        ("Variant B (14 inputs)", None),
        ("+ day", BASE_FEATURES + ["day_level"]),
        ("+ previous", BASE_FEATURES + ["previous_level"]),
        ("+ day + previous",
         BASE_FEATURES + ["day_level", "previous_level"])
    ]

    tables = {}
    columns = {}
    tree_splits = {}
    tree_changed = {}
    for name, features in feature_sets:
        if features is None:
            # The baseline is the Variant B result reported above
            tables[name] = variant_b["table"].set_index("Model")
            columns[name] = len(variant_b["features"])
            continue

        encoded = pd.get_dummies(extended[features], drop_first=False)
        x_train, x_test, y_train, y_test = train_test_split(
            encoded, target,
            test_size=TEST_SIZE,
            stratify=target,
            random_state=RANDOM_STATE
        )
        table, fitted, _ = evaluate(
            build_models(), x_train, x_test, y_train, y_test,
            encoded, target
        )
        tables[name] = table.set_index("Model")
        columns[name] = encoded.shape[1]

        # Does the depth-limited tree ever split on the added columns?
        structure = fitted["Decision Tree"].tree_
        used = [encoded.columns[index] for index in structure.feature
                if index >= 0]
        tree_splits[name] = sum(
            column.startswith(added_prefixes) for column in used
        )

        # ...and if it does, does that change any test prediction?
        # The split and seed are unchanged, so the test rows are the
        # same rows as in Variant B.
        assert list(x_test.index) == list(variant_b["x_test"].index)
        before = variant_b["fitted"]["Decision Tree"].predict(
            variant_b["x_test"])
        after = fitted["Decision Tree"].predict(x_test)
        tree_changed[name] = int((before != after).sum())

    def scores(name, model):
        table = tables[name]
        return (round(table.loc[model, "F1"], 3),
                round(table.loc[model, "CV F1 mean"], 3),
                round(table.loc[model, "CV F1 sd"], 3))

    header = f"   {'Feature set':24}" + "".join(
        f"{model:>22}" for model in models)

    log("One-hot columns per feature set:")
    for name, _ in feature_sets:
        log(f"   {name:24} {columns[name]}")
    log()

    log("Test F1, with 5-fold CV F1 mean in brackets:")
    log(header)
    for name, _ in feature_sets:
        cells = ""
        for model in models:
            test, cv, _ = scores(name, model)
            cells += f"{test:>14.3f} ({cv:.3f})"
        log(f"   {name:24}{cells}")
    log()

    log("CV F1 standard deviation:")
    log(header)
    for name, _ in feature_sets:
        cells = "".join(f"{scores(name, model)[2]:>22.3f}"
                        for model in models)
        log(f"   {name:24}{cells}")
    log()

    # Differences of the rounded values printed above, so that the
    # changes agree with a subtraction of the table entries
    log("Change from Variant B, test F1 (CV F1 mean in brackets),")
    log("as differences of the rounded values above:")
    log(header)
    for name, _ in feature_sets[1:]:
        cells = ""
        for model in models:
            test, cv, _ = scores(name, model)
            base_test, base_cv, _ = scores("Variant B (14 inputs)", model)
            change_test = round(test - base_test, 3) + 0.0
            change_cv = round(cv - base_cv, 3) + 0.0
            cells += f"{change_test:>+13.3f} ({change_cv:+.3f})"
        log(f"   {name:24}{cells}")
    log()

    log("Decision Tree: splits on a day or previous column, and test")
    log("predictions that differ from Variant B:")
    for name, _ in feature_sets[1:]:
        log(f"   {name:24} splits = {tree_splits[name]}   "
            f"predictions changed = {tree_changed[name]} of "
            f"{len(variant_b['x_test'])}")
    log()


# ============================================================
# FIGURES COMPARING THE TWO VARIANTS
# ============================================================
def figure_comparison(variant_a, variant_b):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    models = list(build_models().keys())

    figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for axis, variant in zip(axes, [variant_a, variant_b]):
        table = variant["table"].set_index("Model").reindex(models)
        width = 0.15
        positions = np.arange(len(models))

        for index, metric in enumerate(metrics):
            axis.bar(positions + index * width, table[metric],
                     width, label=metric)

        axis.set_xticks(positions + width * 2)
        axis.set_xticklabels(models, rotation=20, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(variant["label"])
        axis.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Score")
    # One legend for both panels, below the plots, so that it cannot
    # cover any bar
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(metrics),
                  fontsize=10, frameon=False)

    figure.suptitle(
        "Algorithm comparison, with and without duration",
        fontsize=13
    )
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    figure.savefig(
        os.path.join(FIGURES, "fig08_model_comparison.png"), dpi=110
    )
    plt.close(figure)


def figure_roc(variant_a, variant_b):
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    for axis, variant in zip(axes, [variant_a, variant_b]):
        for name, model in variant["fitted"].items():
            probability = model.predict_proba(variant["x_test"])[:, 1]
            false_rate, true_rate, _ = roc_curve(
                variant["y_test"], probability
            )
            area = roc_auc_score(variant["y_test"], probability)
            axis.plot(false_rate, true_rate,
                      label=f"{name} (AUC {area:.3f})", linewidth=1.5)

        axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_title(variant["label"])
        axis.legend(loc="lower right", fontsize=9)
        axis.grid(alpha=0.3)

    figure.suptitle("ROC curves", fontsize=13)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "fig09_roc_curves.png"), dpi=110)
    plt.close(figure)


def figure_confusion(variant_a, matrix_a, variant_b, matrix_b):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis, variant, matrix in [
        (axes[0], variant_a, matrix_a),
        (axes[1], variant_b, matrix_b)
    ]:
        axis.imshow(matrix, cmap="Blues")

        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}",
                          ha="center", va="center", fontsize=14,
                          color="white"
                          if matrix[row, column] > matrix.max() / 2
                          else "black")

        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(["predicted no", "predicted yes"])
        axis.set_yticklabels(["actual no", "actual yes"])
        axis.set_title(f"{variant['best']}\n{variant['label']}")

    figure.suptitle("Confusion matrices, best model each",
                    fontsize=13)
    figure.tight_layout()
    figure.savefig(
        os.path.join(FIGURES, "fig10_confusion_matrices.png"), dpi=110
    )
    plt.close(figure)


def main():
    data = load()

    log("=" * 72)
    log(" 04  PREDICTIVE MODELLING")
    log("=" * 72)
    log(f"Records: {len(data)}")
    log(f"Target: subscribed "
        f"({data['subscribed'].mean() * 100:.1f}% positive)")
    log(f"Split: {int((1 - TEST_SIZE) * 100)}% train / "
        f"{int(TEST_SIZE * 100)}% test, stratified, "
        f"random_state={RANDOM_STATE}")
    log("Cross-validation: 5-fold stratified, scored on F1 and AUC")
    log()

    variant_a = run_variant(
        data,
        BASE_FEATURES + [LEAKING_FEATURE],
        "Variant A: WITH duration (benchmark only)"
    )

    variant_b = run_variant(
        data,
        BASE_FEATURES,
        "Variant B: WITHOUT duration (deployable)"
    )
    compare_on_folds(variant_b, "Logistic Regression", "Random Forest")

    # ------------------------------------------------------------
    # The cost of removing the leaking attribute
    # ------------------------------------------------------------
    log("=" * 72)
    log(" EFFECT OF REMOVING duration")
    log("=" * 72)
    log()

    table_a = variant_a["table"].set_index("Model")
    table_b = variant_b["table"].set_index("Model")

    log(f"{'Model':22} {'F1 with':>9} {'F1 without':>11} "
        f"{'change':>8}")
    for name in table_a.index:
        with_duration = table_a.loc[name, "F1"]
        without = table_b.loc[name, "F1"]
        log(f"{name:22} {with_duration:9.3f} {without:11.3f} "
            f"{without - with_duration:+8.3f}")
    log()
    log("The gap is the share of apparent performance that depends on")
    log("information the bank cannot have before placing the call.")
    log()

    matrix_a = report_confusion(variant_a)
    matrix_b = report_confusion(variant_b)

    report_tree_rules(variant_b, data)
    report_importance(variant_b)
    groups, order = report_coefficients(variant_b, BASE_FEATURES, data)
    figure_odds_ratios(variant_b, groups, order, data)
    report_month_contact(data)
    check_contact_type(data, variant_b)
    check_day_previous(data, variant_b)

    figure_comparison(variant_a, variant_b)
    figure_roc(variant_a, variant_b)
    figure_confusion(variant_a, matrix_a, variant_b, matrix_b)

    log("=" * 72)
    log(" FIGURES WRITTEN")
    log("=" * 72)
    for name in ["fig08_model_comparison.png", "fig09_roc_curves.png",
                 "fig10_confusion_matrices.png",
                 "fig11_feature_importance.png",
                 "fig12_decision_tree.png",
                 "fig13_lr_odds_ratios.png"]:
        log(f"   {name}")

    with open(REPORT, "w") as handle:
        handle.write("\n".join(lines))

    print()
    print(f"Log written to {REPORT}")


if __name__ == "__main__":
    main()
