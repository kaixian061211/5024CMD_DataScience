"""
Init_Database.py
----------------
5024CMD Data Science - Individual Portfolio

Report section 2: the data model.

Loads the raw bank.csv into a normalised SQLite database. The brief
requires that all data is stored in an appropriate database, so every
later stage reads from this database rather than from the CSV.

Run from the Portfolio folder:  python Init_Database.py
"""

import os
import sqlite3

import pandas as pd

SOURCE_CSV = "data/bank.csv"
DATABASE = "data/bank_marketing.db"
REPORT = "output/init_database_report.txt"

os.makedirs("output", exist_ok=True)

lines = []


def log(text=""):
    """Print to screen and collect for the report file."""
    print(text)
    lines.append(str(text))


def load_source():
    """Read the raw CSV and attach the surrogate keys."""
    data = pd.read_csv(SOURCE_CSV)
    data = data.reset_index(drop=True)

    # The source has no identifier column. One contact per row, and
    # each contact belongs to one client, so both keys are generated
    # from the row position. This is documented as a limitation of
    # the extract in the report.
    data["client_id"] = data.index + 1
    data["contact_id"] = data.index + 1

    return data


def create_schema(connection):
    """Create the four tables, dropping any earlier version first."""
    connection.executescript("""
        DROP VIEW  IF EXISTS AnalysisSet;
        DROP TABLE IF EXISTS Outcome;
        DROP TABLE IF EXISTS PreviousCampaign;
        DROP TABLE IF EXISTS Contact;
        DROP TABLE IF EXISTS Client;

        -- Attributes of the person, independent of any campaign
        CREATE TABLE Client (
            client_id      INTEGER PRIMARY KEY,
            age            INTEGER NOT NULL,
            job            TEXT    NOT NULL,
            marital        TEXT    NOT NULL,
            education      TEXT    NOT NULL,
            credit_default TEXT    NOT NULL
                           CHECK (credit_default IN ('yes','no')),
            balance        INTEGER NOT NULL,
            housing        TEXT    NOT NULL
                           CHECK (housing IN ('yes','no')),
            loan           TEXT    NOT NULL
                           CHECK (loan IN ('yes','no'))
        );

        -- Attributes of the call event, not of the person.
        -- duration lives here because it describes the call, which is
        -- why it cannot be known before the call is placed.
        CREATE TABLE Contact (
            contact_id   INTEGER PRIMARY KEY,
            client_id    INTEGER NOT NULL,
            contact_type TEXT    NOT NULL,
            day          INTEGER NOT NULL CHECK (day BETWEEN 1 AND 31),
            month        TEXT    NOT NULL,
            duration     INTEGER NOT NULL,
            campaign     INTEGER NOT NULL,
            FOREIGN KEY (client_id) REFERENCES Client (client_id)
        );

        -- Prior campaign history. Held separately and populated ONLY
        -- for the 2,838 clients who were contacted in an earlier
        -- campaign. The 8,324 never-contacted clients have no row here
        -- at all, rather than a row full of placeholder values.
        CREATE TABLE PreviousCampaign (
            client_id INTEGER PRIMARY KEY,
            pdays     INTEGER NOT NULL,
            previous  INTEGER NOT NULL,
            poutcome  TEXT    NOT NULL,
            FOREIGN KEY (client_id) REFERENCES Client (client_id)
        );

        -- The target, tied to the contact that produced it
        CREATE TABLE Outcome (
            contact_id INTEGER PRIMARY KEY,
            deposit    TEXT NOT NULL CHECK (deposit IN ('yes','no')),
            FOREIGN KEY (contact_id) REFERENCES Contact (contact_id)
        );
    """)


def populate(connection, data):
    """Split the flat file across the four tables."""
    client = data[[
        "client_id", "age", "job", "marital", "education",
        "default", "balance", "housing", "loan"
    ]].rename(columns={"default": "credit_default"})

    contact = data[[
        "contact_id", "client_id", "contact", "day",
        "month", "duration", "campaign"
    ]].rename(columns={"contact": "contact_type"})

    # Only clients with real prior-campaign history get a row.
    # pdays = -1 and previous = 0 identify the never-contacted clients
    # (8,324 of them); both conditions agree exactly.
    contacted_before = data["pdays"] != -1
    previous_campaign = data.loc[contacted_before, [
        "client_id", "pdays", "previous", "poutcome"
    ]]

    outcome = data[["contact_id", "deposit"]]

    tables = {
        "Client": client,
        "Contact": contact,
        "PreviousCampaign": previous_campaign,
        "Outcome": outcome
    }

    for name, frame in tables.items():
        frame.to_sql(name, connection, if_exists="append", index=False)

    return tables


def create_analysis_view(connection):
    """
    Join the four tables back into one flat view.

    Later stages read from this view, so the analysis has a single
    documented entry point into the database and is repeatable.
    """
    connection.executescript("""
        CREATE VIEW AnalysisSet AS
        SELECT cl.client_id,
               cl.age, cl.job, cl.marital, cl.education,
               cl.credit_default, cl.balance, cl.housing, cl.loan,
               co.contact_type, co.day, co.month,
               co.duration, co.campaign,
               COALESCE(pc.pdays, -1)         AS pdays,
               COALESCE(pc.previous, 0)       AS previous,
               COALESCE(pc.poutcome,'unknown') AS poutcome,
               CASE WHEN pc.client_id IS NULL THEN 'no' ELSE 'yes' END
                   AS contacted_before,
               ou.deposit
        FROM Client cl
        JOIN Contact co          ON co.client_id  = cl.client_id
        -- LEFT JOIN: clients with no prior campaign have no row in
        -- PreviousCampaign, so the sentinel values are restored here
        -- for the flat analysis view only.
        LEFT JOIN PreviousCampaign pc ON pc.client_id = cl.client_id
        JOIN Outcome ou          ON ou.contact_id = co.contact_id;
    """)


def verify(connection, expected_rows):
    """Row counts, referential integrity, and the view total."""
    cursor = connection.cursor()

    log("Table row counts")
    for table in ["Client", "Contact", "Outcome"]:
        count = cursor.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        status = "OK" if count == expected_rows else "MISMATCH"
        log(f"   {table:18} {count:6}   {status}")

    # PreviousCampaign is deliberately sparse
    history = cursor.execute(
        "SELECT COUNT(*) FROM PreviousCampaign"
    ).fetchone()[0]
    log(f"   {'PreviousCampaign':18} {history:6}   "
          f"(clients with prior history only)")

    orphans = cursor.execute("""
        SELECT COUNT(*) FROM Contact
        WHERE client_id NOT IN (SELECT client_id FROM Client)
    """).fetchone()[0]
    log(f"\nOrphan contact rows: {orphans}")

    view_rows = cursor.execute(
        "SELECT COUNT(*) FROM AnalysisSet"
    ).fetchone()[0]
    log(f"AnalysisSet view:    {view_rows} rows")


def main():
    log("=" * 62)
    log(" 01  BUILD DATABASE")
    log("=" * 62)
    log()

    data = load_source()
    log(f"Source file: {SOURCE_CSV}")
    log(f"Records read: {len(data)}")
    log(f"Columns read: {len(data.columns) - 2}")
    log()

    if os.path.exists(DATABASE):
        os.remove(DATABASE)

    connection = sqlite3.connect(DATABASE)

    create_schema(connection)
    populate(connection, data)
    create_analysis_view(connection)
    connection.commit()

    verify(connection, len(data))

    connection.close()

    log()
    log(f"Database written to: {DATABASE}")

    with open(REPORT, "w") as handle:
        handle.write("\n".join(lines))

    print(f"Log written to {REPORT}")


if __name__ == "__main__":
    main()
