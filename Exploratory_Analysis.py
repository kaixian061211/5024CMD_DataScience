"""
Exploratory_Analysis.py
-----------------------
5024CMD Data Science - Individual Portfolio

Report section 3: exploratory data analysis.

Reads the CleanedData table from the database and produces the
summary statistics and figures required by the brief:
  - distribution of single variable data
  - outlier inspection
  - relationships between variables (scatterplots)
  - correlation coefficients, with a check on the pdays sentinel
  - relationships with the target (subscription rate by level,
    medians by class)

All figures are written to the output folder as PNG files.

Run from the Portfolio folder:  python Exploratory_Analysis.py
"""

import os
import sqlite3

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")          # write files instead of opening windows

DATABASE = "data/bank_marketing.db"
FIGURES = "output/figures"
REPORT = "output/exploratory_analysis_report.txt"

os.makedirs(FIGURES, exist_ok=True)

NUMERIC = ["age", "balance", "day", "duration",
           "campaign", "pdays", "previous"]

CATEGORICAL = ["job", "marital", "education", "credit_default",
               "housing", "loan", "contact_type", "month", "poutcome"]

# Attributes shown in Figure 6 (rates) and Figure 7 (by class). The log
# in section 4 uses the same lists, so the figures and the printed
# values cannot drift apart.
RATE_COLUMNS = ["job", "marital", "education", "housing",
                "loan", "contact_type", "month", "poutcome"]

BY_CLASS_COLUMNS = ["age", "balance", "duration", "campaign"]

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def load():
    if not os.path.exists(DATABASE):
        raise FileNotFoundError(
            f"Database not found at {DATABASE}. "
            f"Run the database build and preprocessing scripts first."
        )

    connection = sqlite3.connect(DATABASE)
    data = pd.read_sql_query("SELECT * FROM CleanedData", connection)
    connection.close()
    return data


# ============================================================
# 1  SUMMARY STATISTICS
# ============================================================
def summary_statistics(data):
    log("=" * 68)
    log(" 1  SUMMARY STATISTICS")
    log("=" * 68)
    log()

    summary = data[NUMERIC].describe().T
    summary["skew"] = data[NUMERIC].skew()
    summary = summary[["count", "mean", "std", "min",
                       "25%", "50%", "75%", "max", "skew"]]

    log(summary.round(2).to_string())
    log()

    log("Categorical attributes: level counts")
    for column in CATEGORICAL:
        log(f"   {column:15} {data[column].nunique():3} levels")
    log()


# ============================================================
# 2  OUTLIER DETECTION
# ============================================================
def outlier_table(data):
    log("=" * 68)
    log(" 2  OUTLIER DETECTION (1.5 x IQR rule)")
    log("=" * 68)
    log()
    log(f"{'attribute':12} {'lower':>10} {'upper':>10} "
        f"{'outliers':>9} {'percent':>8}")

    for column in NUMERIC:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((data[column] < lower) | (data[column] > upper)).sum()

        log(f"{column:12} {lower:10.1f} {upper:10.1f} "
            f"{count:9} {count / len(data) * 100:7.1f}%")
    log()


# ============================================================
# 3  CORRELATION
# ============================================================
def correlation(data):
    log("=" * 68)
    log(" 3  CORRELATION COEFFICIENTS (Pearson, numeric attributes)")
    log("=" * 68)
    log()

    matrix = data[NUMERIC].corr()
    log(matrix.round(3).to_string())
    log()

    # Report every pair above a modest threshold
    log("Pairs with |r| >= 0.10:")
    found = False
    for i, first in enumerate(NUMERIC):
        for second in NUMERIC[i + 1:]:
            value = matrix.loc[first, second]
            if abs(value) >= 0.10:
                log(f"   {first:10} - {second:10} r = {value:6.3f}")
                found = True
    if not found:
        log("   none")
    log()

    # The -1 / 0 sentinels place every never-contacted client at the
    # same point, which can create a correlation on its own. Recompute
    # pdays-previous using only clients with a real campaign history.
    at_sentinel = ((data["pdays"] == -1) & (data["previous"] == 0)).sum()
    contacted = data[data["contacted_before"] == "yes"]
    log("Sentinel check, pdays - previous:")
    log(f"   records at (pdays = -1, previous = 0): {at_sentinel}")
    log(f"   r, all records                      : "
        f"{matrix.loc['pdays', 'previous']:6.3f}  (n = {len(data)})")
    log(f"   r, contacted before only            : "
        f"{contacted['pdays'].corr(contacted['previous']):6.3f}  "
        f"(n = {len(contacted)})")
    log()

    return matrix


# ============================================================
# 4  RELATIONSHIPS WITH THE TARGET
# ============================================================
def target_relationships(data):
    log("=" * 68)
    log(" 4  RELATIONSHIPS WITH THE TARGET")
    log("=" * 68)
    log()

    overall = data["subscribed"].mean() * 100
    log(f"Overall subscription rate: {overall:.1f}%")
    log()

    log("Subscription rate by level (as plotted in Figure 6),")
    log("highest first:")
    for column in RATE_COLUMNS:
        table = data.groupby(column)["subscribed"].agg(["count", "mean"])
        table = table.sort_values("mean", ascending=False)
        spread = (table["mean"].max() - table["mean"].min()) * 100
        log(f"   {column}  (spread {spread:.1f} points)")
        for level, row in table.iterrows():
            log(f"      {str(level):15} n={int(row['count']):6}"
                f"   rate={row['mean'] * 100:5.1f}%")
    log()

    log("Medians by class (as plotted in Figure 7):")
    log(f"   {'attribute':10} {'no':>9} {'yes':>9}")
    for column in BY_CLASS_COLUMNS:
        declined = data.loc[data["subscribed"] == 0, column].median()
        subscribed = data.loc[data["subscribed"] == 1, column].median()
        log(f"   {column:10} {declined:9.1f} {subscribed:9.1f}")
    log()


# ============================================================
# FIGURE 1 - target distribution
# ============================================================
def figure_target(data):
    counts = data["deposit"].value_counts()

    figure, axis = plt.subplots(figsize=(6, 5))
    bars = axis.bar(counts.index, counts.values,
                    color=["#4C72B0", "#C44E52"])

    for bar, value in zip(bars, counts.values):
        axis.text(bar.get_x() + bar.get_width() / 2,
                  value + 60,
                  f"{value:,}\n({value / len(data) * 100:.1f}%)",
                  ha="center")

    axis.set_title("Distribution of the target attribute")
    axis.set_xlabel("Subscribed to term deposit")
    axis.set_ylabel("Number of clients")
    axis.set_ylim(0, counts.max() * 1.18)
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "fig01_target_distribution.png"), dpi=110)
    plt.close(figure)


# ============================================================
# FIGURE 2 - numeric distributions
# ============================================================
def figure_histograms(data):
    figure, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for index, column in enumerate(NUMERIC):
        axes[index].hist(data[column], bins=40, color="#4C72B0",
                         edgecolor="white", linewidth=0.4)
        axes[index].set_title(
            f"{column}  (skew = {data[column].skew():.2f})"
        )
        axes[index].grid(axis="y", alpha=0.3)

    for extra in range(len(NUMERIC), len(axes)):
        axes[extra].set_visible(False)

    figure.suptitle("Distribution of numeric attributes",
                    fontsize=13)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "fig02_numeric_distributions.png"), dpi=110)
    plt.close(figure)


# ============================================================
# FIGURE 3 - boxplots for outliers
# ============================================================
def figure_boxplots(data):
    figure, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()

    for index, column in enumerate(NUMERIC):
        axes[index].boxplot(data[column], vert=True, widths=0.5)
        axes[index].set_title(column)
        axes[index].grid(axis="y", alpha=0.3)

    for extra in range(len(NUMERIC), len(axes)):
        axes[extra].set_visible(False)

    figure.suptitle("Outlier inspection by boxplot",
                    fontsize=13)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "fig03_boxplots.png"), dpi=110)
    plt.close(figure)


# ============================================================
# FIGURE 4 - correlation heatmap
# ============================================================
def figure_heatmap(matrix):
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)

    axis.set_xticks(range(len(NUMERIC)))
    axis.set_yticks(range(len(NUMERIC)))
    axis.set_xticklabels(NUMERIC, rotation=45, ha="right")
    axis.set_yticklabels(NUMERIC)

    # Print the coefficient inside each cell
    for row in range(len(NUMERIC)):
        for column in range(len(NUMERIC)):
            value = matrix.iloc[row, column]
            label = f"{value:.2f}"
            if label == "-0.00":       # tiny negatives round to -0.00
                label = "0.00"
            axis.text(column, row, label, ha="center",
                      va="center",
                      color="white" if abs(value) > 0.5 else "black",
                      fontsize=9)

    figure.colorbar(image, ax=axis, shrink=0.8)
    axis.set_title("Correlation matrix of numeric attributes")

    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "fig04_correlation_heatmap.png"), dpi=110)
    plt.close(figure)


# ============================================================
# FIGURE 5 - scatterplots
# ============================================================
def figure_scatter(data):
    pairs = [
        ("pdays", "previous"),
        ("age", "balance"),
        ("duration", "campaign"),
        ("balance", "duration")
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for index, (x_column, y_column) in enumerate(pairs):
        subscribed = data[data["subscribed"] == 1]
        declined = data[data["subscribed"] == 0]

        axes[index].scatter(declined[x_column], declined[y_column],
                            s=6, alpha=0.3, label="no",
                            color="#C44E52")
        axes[index].scatter(subscribed[x_column], subscribed[y_column],
                            s=6, alpha=0.3, label="yes",
                            color="#4C72B0")

        r = data[x_column].corr(data[y_column])
        axes[index].set_title(f"{x_column} vs {y_column}  (r = {r:.3f})")
        axes[index].set_xlabel(x_column)
        axes[index].set_ylabel(y_column)
        axes[index].legend(markerscale=3)
        axes[index].grid(alpha=0.3)

    figure.suptitle("Relationships between numeric attributes",
                    fontsize=13)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "fig05_scatterplots.png"), dpi=110)
    plt.close(figure)


# ============================================================
# FIGURE 6 - subscription rate by categorical attribute
# ============================================================
def figure_categorical_rates(data):
    columns = RATE_COLUMNS

    overall = data["subscribed"].mean() * 100

    figure, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for index, column in enumerate(columns):
        rates = (data.groupby(column)["subscribed"].mean() * 100)
        rates = rates.sort_values(ascending=False)

        axes[index].bar(range(len(rates)), rates.values,
                        color="#4C72B0")
        axes[index].axhline(overall, color="#C44E52",
                            linestyle="--", linewidth=1.2)
        axes[index].set_xticks(range(len(rates)))
        axes[index].set_xticklabels(rates.index, rotation=45,
                                    ha="right", fontsize=8)
        axes[index].set_title(column)
        axes[index].set_ylabel("Subscription rate (%)")
        axes[index].grid(axis="y", alpha=0.3)

    figure.suptitle(
        f"Subscription rate by category "
        f"(dashed line = overall rate {overall:.1f}%)",
        fontsize=13
    )
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "fig06_categorical_rates.png"), dpi=110)
    plt.close(figure)


# ============================================================
# FIGURE 7 - numeric attributes by class
# ============================================================
def figure_by_class(data):
    columns = BY_CLASS_COLUMNS

    figure, axes = plt.subplots(1, 4, figsize=(16, 5))

    for index, column in enumerate(columns):
        groups = [
            data[data["subscribed"] == 0][column],
            data[data["subscribed"] == 1][column]
        ]
        axes[index].boxplot(groups, tick_labels=["no", "yes"], showfliers=False)
        axes[index].set_title(column)
        axes[index].set_xlabel("Subscribed")
        axes[index].grid(axis="y", alpha=0.3)

    figure.suptitle(
        "Numeric attributes by outcome (outliers hidden)",
        fontsize=13
    )
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, "fig07_numeric_by_class.png"), dpi=110)
    plt.close(figure)


def main():
    data = load()

    log("=" * 68)
    log(" 03  EXPLORATORY DATA ANALYSIS")
    log("=" * 68)
    log(f"Records: {len(data)}")
    # Class balance of the target, as shown in Figure 1.1
    for level, count in data["deposit"].value_counts().items():
        log(f"   deposit = {level:4} {count:6}  ({count / len(data) * 100:.1f}%)")
    log()

    summary_statistics(data)
    outlier_table(data)
    matrix = correlation(data)
    target_relationships(data)

    figure_target(data)
    figure_histograms(data)
    figure_boxplots(data)
    figure_heatmap(matrix)
    figure_scatter(data)
    figure_categorical_rates(data)
    figure_by_class(data)

    log("=" * 68)
    log(" FIGURES WRITTEN")
    log("=" * 68)
    # List only the figures this script produces, not everything
    # sitting in the folder from other stages.
    for name in ["fig01_target_distribution.png",
                 "fig02_numeric_distributions.png",
                 "fig03_boxplots.png",
                 "fig04_correlation_heatmap.png",
                 "fig05_scatterplots.png",
                 "fig06_categorical_rates.png",
                 "fig07_numeric_by_class.png"]:
        log(f"   {name}")

    with open(REPORT, "w") as handle:
        handle.write("\n".join(lines))

    print()
    print(f"Log written to {REPORT}")


if __name__ == "__main__":
    main()
