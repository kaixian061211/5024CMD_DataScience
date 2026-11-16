"""
Data_Preprocessing.py
---------------------
5024CMD Data Science - Individual Portfolio

Report section 2: checking, cleaning and preprocessing.

Reads the AnalysisSet view from the database, runs the data quality
checks, converts the numeric attributes to categorical bands, and
writes a cleaned table back to the database for the EDA and
modelling stages.

Run from the Portfolio folder:  python Data_Preprocessing.py
"""

import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

DATABASE = "data/bank_marketing.db"
REPORT = "output/data_preprocessing_report.txt"

os.makedirs("output", exist_ok=True)

lines = []


def log(text=""):
    """Print to screen and collect for the report file."""
    print(text)
    lines.append(str(text))


# ============================================================
# 1  DATA QUALITY CHECKS
# ============================================================
def quality_checks(data):
    log("=" * 62)
    log(" 1  DATA QUALITY CHECKS")
    log("=" * 62)
    log()

    log(f"Records: {len(data)}")
    log(f"Attributes: {len(data.columns)}")
    log(f"Missing (null) values: {data.isna().sum().sum()}")
    # client_id is generated per row, so it must be excluded or no
    # duplicate could ever be found.
    source_columns = [c for c in data.columns
                      if c not in ("client_id", "contacted_before")]
    duplicates = data.duplicated(subset=source_columns).sum()
    log(f"Exact duplicate rows (source columns only): {duplicates}")
    log()

    # 'unknown' is disguised missing data. It is not one phenomenon:
    # in poutcome it means the client has no prior campaign history,
    # while in job and education it is genuine missing data.
    log("Disguised missing values ('unknown')")
    total_unknown = 0
    for column in data.columns:
        count = (data[column] == "unknown").sum()
        if count:
            total_unknown += count
            log(f"   {column:14} {count:6}  ({count / len(data) * 100:5.1f}%)")
    log(f"   {'total':14} {total_unknown:6}")
    log()

    # pdays uses -1 as a sentinel meaning "never contacted before".
    sentinel = (data["pdays"] == -1).sum()
    log(f"pdays = -1 (never previously contacted): {sentinel} "
        f"({sentinel / len(data) * 100:.1f}%)")

    unknown_outcome = (data["poutcome"] == "unknown").sum()
    log(f"poutcome = 'unknown':                    {unknown_outcome} "
        f"({unknown_outcome / len(data) * 100:.1f}%)")

    never = (data["pdays"] == -1) & (data["poutcome"] == "unknown")
    contacted_no_outcome = (data["pdays"] != -1) & \
                           (data["poutcome"] == "unknown")

    log(f"   both conditions (never contacted):     {never.sum()}")
    log(f"   contacted, but outcome not recorded:   "
        f"{contacted_no_outcome.sum()}")
    log("The two groups almost coincide but are not identical: "
        "poutcome")
    log("'unknown' also covers a small number of clients who were "
        "contacted")
    log("but whose outcome was never recorded. Only those are "
        "genuinely missing.")
    log()

    # contact_type 'unknown' is a third, different case
    unknown_contact = (data["contact_type"] == "unknown").sum()
    log(f"contact_type = 'unknown':                {unknown_contact} "
        f"({unknown_contact / len(data) * 100:.1f}%)")
    log("The channel was not recorded for these calls. It is kept as "
        "its own")
    log("level rather than imputed, because its subscription rate "
        "differs")
    log("sharply from the recorded channels and so carries "
        "information.")
    log()

    negative = (data["balance"] < 0).sum()
    log(f"Negative balance (overdrawn accounts): {negative} "
        f"({negative / len(data) * 100:.1f}%)")
    log("Valid values, not errors, so they are retained.")
    log()

    log("Skew and outliers (1.5 x IQR rule)")
    for column in ["age", "balance", "duration", "campaign", "pdays", "previous"]:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outliers = ((data[column] < low) | (data[column] > high)).sum()

        log(f"   {column:10} skew={data[column].skew():6.2f}   "
            f"upper fence={high:8.1f}   "
            f"outliers={outliers:5} ({outliers / len(data) * 100:4.1f}%)")
    log()
    log("Note: the counts for pdays and previous are artefacts of the")
    log("sentinel. Because 74.6% of pdays values are -1, the quartiles")
    log("collapse towards the sentinel and the upper fence falls below")
    log("almost every genuine value, so nearly every client who WAS")
    log("contacted before is flagged. The same pile-up at zero inflates")
    log("previous. These are not extreme values and are not treated as")
    log("outliers.")
    log()


# ============================================================
# 2  DISCRETISATION
# ============================================================
def equal_width_test(series):
    """
    Apply the equal-width method and report the largest band share.

    Used to demonstrate why equal width is unsuitable for the
    skewed attributes.
    """
    minimum = series.min()
    maximum = series.max()
    interval = (maximum - minimum) / 3

    bands = pd.cut(
        series,
        bins=[minimum - 1, minimum + interval, minimum + 2 * interval,
              maximum],
        labels=["Low", "Medium", "High"]
    )

    share = bands.value_counts().max() / len(series) * 100
    return interval, share, bands


def discretise(data):
    log("=" * 62)
    log(" 2  CONVERTING NUMERIC ATTRIBUTES TO CATEGORICAL")
    log("=" * 62)
    log()

    log("Equal-width method, interval = (max - min) / 3")
    for column in ["age", "balance", "duration", "campaign"]:
        interval, share, _ = equal_width_test(data[column])
        verdict = "usable" if share < 80 else "UNUSABLE"
        log(f"   {column:10} interval={interval:9.1f}   "
            f"largest band holds {share:5.1f}%   {verdict}")
    log()
    log("A band holding almost every record cannot separate the")
    log("classes, so quantile bands are used where skew is severe.")
    log()

    # age is only mildly skewed, so the equal-width method is kept
    _, _, data["age_group"] = equal_width_test(data["age"])

    # balance and duration are severely skewed, so equal frequency
    data["balance_group"] = pd.qcut(
        data["balance"], 3, labels=["Low", "Medium", "High"]
    )
    data["duration_group"] = pd.qcut(
        data["duration"], 3, labels=["Low", "Medium", "High"]
    )

    # campaign has too many repeated values for quantile bands,
    # so fixed cut-points are used instead
    data["campaign_group"] = pd.cut(
        data["campaign"],
        bins=[0, 1, 3, data["campaign"].max()],
        labels=["Low", "Medium", "High"]
    )

    # pdays: the sentinel is separated rather than binned as a number
    data["contacted_before"] = np.where(data["pdays"] == -1, "no", "yes")
    data["pdays_group"] = np.where(
        data["pdays"] == -1, "Not contacted",
        np.where(data["pdays"] <= 180, "Recent", "Long ago")
    )

    # Category consolidation: 12 job levels into 4 groups
    job_groups = {
        "management": "White collar", "admin.": "White collar",
        "entrepreneur": "White collar", "self-employed": "White collar",
        "technician": "Skilled", "services": "Skilled",
        "blue-collar": "Skilled", "housemaid": "Skilled",
        "retired": "Non-working", "student": "Non-working",
        "unemployed": "Non-working", "unknown": "Unknown"
    }
    data["job_group"] = data["job"].map(job_groups)

    # 12 months into 4 quarters
    month_groups = {
        "jan": "Q1", "feb": "Q1", "mar": "Q1",
        "apr": "Q2", "may": "Q2", "jun": "Q2",
        "jul": "Q3", "aug": "Q3", "sep": "Q3",
        "oct": "Q4", "nov": "Q4", "dec": "Q4"
    }
    data["month_group"] = data["month"].map(month_groups)

    # Target as 0/1 for the modelling stage
    data["subscribed"] = (data["deposit"] == "yes").astype(int)

    return data


# ============================================================
# 3  VALIDATION - do the new bands separate the target?
# ============================================================
def validate_bands(data):
    log("=" * 62)
    log(" 3  DO THE NEW BANDS SEPARATE THE TARGET?")
    log("=" * 62)
    log()

    overall = data["subscribed"].mean() * 100
    log(f"Overall subscription rate: {overall:.1f}%")
    log()

    new_columns = [
        "age_group", "balance_group", "duration_group",
        "campaign_group", "pdays_group", "job_group", "month_group"
    ]

    for column in new_columns:
        log(column)
        grouped = data.groupby(column, observed=True)["subscribed"]

        # Source attribute each numeric band was derived from, so the
        # actual value range of every band can be reported
        source = {
            "age_group": "age", "balance_group": "balance",
            "duration_group": "duration", "campaign_group": "campaign",
            "pdays_group": "pdays"
        }.get(column)

        for band, values in grouped:
            line = (f"   {str(band):14} n={len(values):6} "
                    f"({len(values) / len(data) * 100:4.1f}%)   "
                    f"rate={values.mean() * 100:5.1f}%")
            if source:
                members = data.loc[data[column] == band, source]
                line += f"   range {members.min()} to {members.max()}"
            log(line)

        spread = grouped.mean().max() - grouped.mean().min()
        log(f"   spread = {spread * 100:.1f} percentage points")
        log()


def attribute_selection(data):
    """
    Rank every candidate attribute by mutual information with the
    target, then report the subscription rate for each level of the
    strongest categorical attributes.

    Mutual information is used rather than correlation because the
    target is binary and most inputs are nominal, so a Pearson
    coefficient would be meaningless for them.
    """
    log("=" * 62)
    log(" 4  ATTRIBUTE SELECTION")
    log("=" * 62)
    log()

    candidates = [
        "age", "job", "marital", "education", "credit_default",
        "balance", "housing", "loan", "contact_type", "day",
        "month", "duration", "campaign", "pdays", "previous",
        "poutcome"
    ]

    encoded = data[candidates].copy()
    categorical = []

    for column in candidates:
        if not pd.api.types.is_numeric_dtype(encoded[column]):
            encoded[column] = LabelEncoder().fit_transform(
                encoded[column]
            )
            categorical.append(True)
        else:
            # Treat low-cardinality numerics as discrete too
            categorical.append(encoded[column].nunique() < 40)

    scores = mutual_info_classif(
        encoded,
        data["subscribed"],
        discrete_features=categorical,
        random_state=0
    )

    ranking = pd.Series(scores, index=candidates)
    ranking = ranking.sort_values(ascending=False)

    log("Mutual information with the target, all candidates ranked")
    for name, value in ranking.items():
        log(f"   {name:16} {value:.4f}")
    log()

    # Subscription rate per level for the categorical attributes
    log("Subscription rate by level (categorical attributes)")
    log()

    for column in ["poutcome", "month", "housing", "loan",
                   "contact_type", "marital", "education"]:
        log(f"{column}")
        grouped = data.groupby(column, observed=True)["subscribed"]

        for level, values in grouped:
            log(f"   {str(level):14} n={len(values):6} "
                f"rate={values.mean() * 100:5.1f}%")

        spread = (grouped.mean().max() - grouped.mean().min()) * 100
        log(f"   spread = {spread:.1f} percentage points")
        log()


def main():
    connection = sqlite3.connect(DATABASE)
    data = pd.read_sql_query("SELECT * FROM AnalysisSet", connection)

    quality_checks(data)
    data = discretise(data)
    validate_bands(data)
    attribute_selection(data)

    # Write the cleaned table back to the database
    data.to_sql("CleanedData", connection, if_exists="replace", index=False)
    connection.commit()
    connection.close()

    with open(REPORT, "w") as handle:
        handle.write("\n".join(lines))

    log("=" * 62)
    log(f"Cleaned table written to database as 'CleanedData'")
    log(f"Log written to {REPORT}")
    log("=" * 62)


if __name__ == "__main__":
    main()
