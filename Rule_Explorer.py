"""
Rule_Explorer.py
----------------
5024CMD Data Science - Individual Portfolio

A small window for trying the prediction rules on one client at a time.

Each input of the deployable model (Variant B, without duration) has a
row of buttons, one per level. Choosing a client profile shows:
    - the Logistic Regression probability that the client subscribes,
      against the overall subscription rate, and whether to call
    - the decision tree rule the client falls under, with how many
      training clients reached that rule and how many subscribed
    - the business rules that apply to the profile

The models are rebuilt on start with the same inputs, split, seed and
settings as Predictive_Model.py, which they are imported from, so the
window shows the same models as Section 4 of the report.

On every run, a log of checks against predictive_model_report.txt and
of the example clients quoted in Section 4.7 is written to
output/rule_explorer_report.txt.

Run from the Portfolio folder, after Init_Database.py and
Data_Preprocessing.py:
    python Rule_Explorer.py            writes the log, opens the window
    python Rule_Explorer.py --check    writes the log only
"""

import random
import sys
import tkinter as tk
from tkinter import font as tkfont

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from Predictive_Model import (BASE_FEATURES, RANDOM_STATE, TEST_SIZE,
                              build_models, load)

# One row of buttons per input, top to bottom: (column, label, levels).
# contacted_before is not shown: it follows from the previous campaign.
PROFILE = [
    ("age_group", "Age", ["Low", "Medium", "High"]),
    ("job_group", "Job", ["White collar", "Skilled", "Non-working",
                          "Unknown"]),
    ("marital", "Marital status", ["married", "single", "divorced"]),
    ("education", "Education", ["primary", "secondary", "tertiary",
                                "unknown"]),
    ("credit_default", "Credit in default", ["no", "yes"]),
    ("balance_group", "Balance", ["Low", "Medium", "High"]),
    ("housing", "Housing loan", ["no", "yes"]),
    ("loan", "Personal loan", ["no", "yes"]),
    ("contact_type", "Contact type", ["cellular", "telephone", "unknown"]),
    ("month_group", "Quarter", ["Q1", "Q2", "Q3", "Q4"]),
    ("campaign_group", "Calls this campaign", ["Low", "Medium", "High"]),
    ("pdays_group", "Previous campaign", ["Not contacted", "Recent",
                                          "Long ago"]),
    ("poutcome", "Previous outcome", ["failure", "other", "success",
                                      "unknown"]),
]

# Banded attributes and the raw attribute each band was cut from, so the
# button captions show the real ranges rather than typed-in numbers
BAND_SOURCES = {"age_group": "age", "balance_group": "balance",
                "campaign_group": "campaign", "pdays_group": "pdays"}

QUARTERS = {"Q1": "Jan–Mar", "Q2": "Apr–Jun", "Q3": "Jul–Sep",
            "Q4": "Oct–Dec"}

NOT_CONTACTED = "Not contacted"

REPORT = "output/rule_explorer_report.txt"

# Colours
BACKGROUND = "#F7F7F7"
PANEL = "#FFFFFF"
CHOSEN = "#CFE3F7"
TEXT = "#222222"
MUTED = "#8A8A8A"
CALL = "#1B7A4A"
NO_CALL = "#B03A2E"
TRACK = "#E4E4E4"

lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


# ============================================================
# MODELS - rebuilt exactly as in Predictive_Model.py
# ============================================================
def build():
    data = load()
    encoded = pd.get_dummies(data[BASE_FEATURES], drop_first=False)
    target = data["subscribed"]

    x_train, x_test, y_train, y_test = train_test_split(
        encoded, target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE
    )

    models = build_models()
    tree = models["Decision Tree"].fit(x_train, y_train)
    regression = models["Logistic Regression"].fit(x_train, y_train)

    # Training clients reaching each leaf of the tree, and how many of
    # them subscribed
    leaves = pd.DataFrame({"leaf": tree.apply(x_train),
                           "yes": y_train.to_numpy()})
    leaves = leaves.groupby("leaf")["yes"].agg(["size", "sum"])

    scores = {}
    for name, model in [("Logistic Regression", regression),
                        ("Decision Tree", tree)]:
        scores[name] = (
            f1_score(y_test, model.predict(x_test)),
            roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
        )

    return {
        "data": data,
        "columns": list(encoded.columns),
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "tree": tree,
        "regression": regression,
        "leaves": leaves,
        "scores": scores,
        "overall": target.mean(),
    }


def split_column(column):
    """'pdays_group_Long ago' -> ('pdays_group', 'Long ago')."""
    attribute = max((a for a in BASE_FEATURES if column.startswith(a + "_")),
                    key=len)
    return attribute, column[len(attribute) + 1:]


def encode(profile, columns):
    """One client profile as a single one-hot row."""
    row = pd.DataFrame(False, index=[0], columns=columns)
    for attribute, level in profile.items():
        row[f"{attribute}_{level}"] = True
    return row


def most_common_profile(data):
    """The most common level of every input: the window's opening view."""
    return {column: data[column].mode()[0] for column, _, _ in PROFILE}


def assess(models, profile):
    """Both models' answers for one client profile."""
    profile = dict(profile)
    profile["contacted_before"] = (
        "no" if profile["pdays_group"] == NOT_CONTACTED else "yes"
    )
    row = encode(profile, models["columns"])

    regression = models["regression"]
    probability = regression.predict_proba(row)[0, 1]
    # Difference from the overall rate, taken from the rounded values as
    # in the other logs
    difference = (round(probability * 100, 1)
                  - round(models["overall"] * 100, 1))

    # The questions the tree asks this client, and the leaf it reaches
    tree = models["tree"]
    structure = tree.tree_
    nodes = tree.decision_path(row).indices
    steps = []
    for node in nodes[:-1]:
        column = models["columns"][structure.feature[node]]
        attribute, level = split_column(column)
        answer = bool(row.iloc[0, structure.feature[node]]
                      > structure.threshold[node])
        steps.append((column, attribute, level, answer))
    total, yes = models["leaves"].loc[nodes[-1]]

    return {
        "probability": probability,
        "difference": difference,
        "call": int(regression.predict(row)[0]) == 1,
        "steps": steps,
        "leaf_total": int(total),
        "leaf_yes": int(yes),
        "tree_says_yes": int(tree.predict(row)[0]) == 1,
    }


# ============================================================
# LOG - checks, and the example clients quoted in the report
# ============================================================
def report(models):
    data = models["data"]
    overall = models["overall"]

    log("=" * 62)
    log(" RULE EXPLORER - CHECKS AND EXAMPLES")
    log("=" * 62)
    log()
    log("Models rebuilt with the inputs, split, seed and settings of")
    log("Predictive_Model.py (compare predictive_model_report.txt)")
    log(f"   training clients: {len(models['y_train']):,}   "
        f"test clients: {len(models['x_test']):,}")
    for name, (f1, auc) in models["scores"].items():
        log(f"   {name:20} test F1 {f1:.3f}   AUC {auc:.3f}")

    # Training clients answering 'yes' at the root of the tree
    tree = models["tree"]
    root = tree.tree_.feature[0]
    yes_branch = tree.tree_.children_right[0]
    reached = tree.decision_path(models["x_train"])[:, yes_branch]
    reached = reached.toarray().ravel().astype(bool)
    subscribed = int(models["y_train"].to_numpy()[reached].sum())
    attribute, level = split_column(models["columns"][root])
    log(f"   tree root: {attribute} = {level}? yes -> "
        f"{int(reached.sum()):,} training clients, "
        f"{subscribed:,} subscribed")
    log(f"   decision tree rules (leaves): {tree.get_n_leaves()}")
    log()

    log("Never-contacted rule enforced by the window")
    never = data[data["pdays_group"] == NOT_CONTACTED]
    log(f"   pdays_group = {NOT_CONTACTED}: {len(never):,} clients")
    log(f"   of these, contacted_before = no: "
        f"{(never['contacted_before'] == 'no').sum():,}   "
        f"poutcome = unknown: {(never['poutcome'] == 'unknown').sum():,}")
    log()

    base = most_common_profile(data)
    examples = [
        ("Opening view: the most common level of every input", base),
        ("Opening view, with contact_type = unknown",
         {**base, "contact_type": "unknown"}),
        ("Opening view, contacted recently and the previous campaign "
         "succeeded", {**base, "pdays_group": "Recent",
                       "poutcome": "success"}),
    ]
    x_train = models["x_train"]
    y_train = models["y_train"]

    for number, (title, profile) in enumerate(examples, start=1):
        result = assess(models, profile)
        log(f"Example {number}: {title}")
        settings = [f"{column}={profile[column]}" for column, _, _ in PROFILE]
        log("   " + ", ".join(settings[:7]))
        log("   " + ", ".join(settings[7:]))

        direction = "above" if result["difference"] >= 0 else "below"
        log(f"   Logistic Regression: probability "
            f"{result['probability']:.1%}, "
            f"{abs(result['difference']):.1f} points {direction} the "
            f"{overall:.1%} overall rate -> "
            f"{'call' if result['call'] else 'do not call'}")

        log("   Decision tree:")
        for step, (_, attribute, level, answer) in enumerate(
                result["steps"], start=1):
            log(f"      {step}. {attribute} = {level}? "
                f"{'yes' if answer else 'no'}")
        total, yes = result["leaf_total"], result["leaf_yes"]
        log(f"      reached by {total:,} training clients, {yes:,} "
            f"subscribed ({yes / total:.1%}) -> "
            f"{'subscribes' if result['tree_says_yes'] else 'does not subscribe'}")

        # The same count found directly, by filtering the training rows
        match = pd.Series(True, index=x_train.index)
        for column, _, _, answer in result["steps"]:
            match &= x_train[column] == answer
        log(f"      check: training clients giving the same answers: "
            f"{int(match.sum()):,}, subscribed {int(y_train[match].sum()):,}")
        log()


# ============================================================
# WINDOW
# ============================================================
class RuleExplorer:
    def __init__(self, root, models):
        self.root = root
        self.models = models
        self.data = models["data"]
        self.random = random.Random()

        base = tkfont.nametofont("TkDefaultFont")
        family = base.actual("family")
        self.fonts = {
            "title": tkfont.Font(family=family, size=15, weight="bold"),
            "heading": tkfont.Font(family=family, size=11, weight="bold"),
            "body": tkfont.Font(family=family, size=10),
            "bold": tkfont.Font(family=family, size=10, weight="bold"),
            "small": tkfont.Font(family=family, size=9),
            "decision": tkfont.Font(family=family, size=20, weight="bold"),
        }

        root.title("Rule Explorer - 5024CMD Bank Marketing")
        root.configure(bg=BACKGROUND)

        self.choice = {column: tk.StringVar() for column, _, _ in PROFILE}
        self.buttons = {}
        self.loading = False          # True while several buttons change
        self.was_contacted = False

        self.build_header()
        self.build_profile()
        self.build_results()
        self.build_footer()

        f1, auc = models["scores"]["Logistic Regression"]
        self.model_note = (f"Models rebuilt with the Section 4 split: "
                           f"Logistic Regression test F1 {f1:.3f}, "
                           f"AUC {auc:.3f}.")

        for variable in self.choice.values():
            variable.trace_add("write", lambda *_: self.changed())
        self.most_common()

    # ---------- layout ----------
    def panel(self, parent, title):
        frame = tk.Frame(parent, bg=PANEL, highlightthickness=1,
                         highlightbackground="#D6D6D6", padx=12, pady=10)
        tk.Label(frame, text=title, font=self.fonts["heading"], bg=PANEL,
                 fg=TEXT, anchor="w").pack(fill="x", pady=(0, 6))
        return frame

    def build_header(self):
        header = tk.Frame(self.root, bg=BACKGROUND, padx=14, pady=10)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text="Rule Explorer: will this client subscribe "
                 "to a term deposit?", font=self.fonts["title"],
                 bg=BACKGROUND, fg=TEXT).pack(anchor="w")
        tk.Label(header, text="Choose a client with the buttons. Only "
                 "information known before the call is used (Variant B, "
                 "without duration).", font=self.fonts["body"],
                 bg=BACKGROUND, fg=MUTED).pack(anchor="w")

    def caption(self, column, level):
        if column == "month_group":
            return f"{level} ({QUARTERS[level]})"
        if column not in BAND_SOURCES or level == NOT_CONTACTED:
            return level
        values = self.data.loc[self.data[column] == level,
                               BAND_SOURCES[column]]
        low, high = int(values.min()), int(values.max())
        unit = " days" if column == "pdays_group" else ""
        if low == high:
            text = f"{low:,}"
        elif low < 0:
            text = f"up to {high:,}"
        elif level in ("High", "Long ago"):
            text = f"{low:,}+{unit}"
        else:
            text = f"{low:,}–{high:,}{unit}"
        return f"{level} ({text})"

    def build_profile(self):
        frame = self.panel(self.root, "1  Client profile")
        frame.grid(row=1, column=0, sticky="nsew", padx=(14, 7), pady=(0, 10))
        grid = tk.Frame(frame, bg=PANEL)
        grid.pack(fill="both")

        for row, (column, label, levels) in enumerate(PROFILE):
            names = tk.Frame(grid, bg=PANEL)
            names.grid(row=row, column=0, sticky="w", padx=(0, 10), pady=1)
            tk.Label(names, text=label, font=self.fonts["bold"], bg=PANEL,
                     fg=TEXT).pack(anchor="w")
            tk.Label(names, text=column, font=self.fonts["small"], bg=PANEL,
                     fg=MUTED).pack(anchor="w")

            self.buttons[column] = []
            for position, level in enumerate(levels, start=1):
                button = tk.Radiobutton(
                    grid, text=self.caption(column, level), value=level,
                    variable=self.choice[column], indicatoron=False,
                    font=self.fonts["body"], bg=PANEL, fg=TEXT,
                    selectcolor=CHOSEN, activebackground="#EEF4FB",
                    padx=6, pady=3, bd=1
                )
                button.grid(row=row, column=position, sticky="ew",
                            padx=2, pady=1)
                self.buttons[column].append(button)

        for position in range(1, 5):
            grid.columnconfigure(position, uniform="buttons")

    def build_results(self):
        side = tk.Frame(self.root, bg=BACKGROUND)
        side.grid(row=1, column=1, sticky="nsew", padx=(7, 14), pady=(0, 10))

        # 2  Logistic Regression
        frame = self.panel(side, "2  Logistic Regression (selected model)")
        frame.pack(fill="x")
        self.decision = tk.Label(frame, font=self.fonts["decision"],
                                 bg=PANEL, anchor="w")
        self.decision.pack(fill="x")
        self.probability = tk.Label(frame, font=self.fonts["body"], bg=PANEL,
                                    fg=TEXT, anchor="w", justify="left")
        self.probability.pack(fill="x", pady=(2, 6))
        self.bar = tk.Canvas(frame, width=440, height=48, bg=PANEL,
                             highlightthickness=0)
        self.bar.pack(anchor="w")

        # 3  Decision tree
        frame = self.panel(side, "3  Decision tree rule for this client")
        frame.pack(fill="x", pady=(10, 0))
        self.path = tk.Text(frame, width=56, height=8, bd=0, bg=PANEL,
                            fg=TEXT, font=self.fonts["body"], wrap="word",
                            cursor="arrow")
        self.path.tag_configure("bold", font=self.fonts["bold"])
        self.path.tag_configure("muted", foreground=MUTED)
        self.path.pack(fill="x")

        # 4  Business rules
        frame = self.panel(side, "4  Business rules")
        frame.pack(fill="x", pady=(10, 0))
        self.rules = []
        recent = self.data.loc[self.data["pdays_group"] == "Recent", "pdays"]
        texts = [
            "Only information known before the call is used: duration "
            "(call length) is left out, as it is known only after the call.",
            "A client never contacted before has no previous outcome: "
            "poutcome is fixed at 'unknown' and contacted_before is 'no'.",
            f"A previous contact up to {int(recent.max())} days ago is "
            "Recent; one longer ago is Long ago.",
            "The client is called when the Logistic Regression probability "
            "is above 50%.",
        ]
        for text in texts:
            label = tk.Label(frame, text="•  " + text, font=self.fonts["body"],
                             bg=PANEL, anchor="w", justify="left",
                             wraplength=430)
            label.pack(fill="x", pady=1)
            self.rules.append(label)

    def build_footer(self):
        footer = tk.Frame(self.root, bg=BACKGROUND, padx=14)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        tk.Button(footer, text="Most common client", command=self.most_common,
                  font=self.fonts["body"], padx=10).pack(side="left")
        tk.Button(footer, text="Random test client", command=self.test_client,
                  font=self.fonts["body"], padx=10).pack(side="left", padx=8)
        self.status = tk.Label(footer, font=self.fonts["small"],
                               bg=BACKGROUND, fg=MUTED, anchor="w",
                               justify="left")
        self.status.pack(side="left", fill="x", padx=(6, 0))

    # ---------- actions ----------
    def set_profile(self, values):
        self.loading = True
        for column, _, _ in PROFILE:
            self.choice[column].set(values[column])
        self.loading = False
        # A loaded client keeps its own previous outcome
        self.was_contacted = values["pdays_group"] != NOT_CONTACTED
        self.refresh()

    def most_common(self):
        self.set_profile(most_common_profile(self.data))
        self.status.config(text="Showing the most common level of every "
                                "input. " + self.model_note)

    def test_client(self):
        index = self.random.choice(list(self.models["x_test"].index))
        client = self.data.loc[index]
        self.set_profile({column: client[column] for column, _, _ in PROFILE})
        self.status.config(
            text=f"Test client {int(client['client_id']):,} (not used in "
                 f"training) actually subscribed: {client['deposit']}.")

    def changed(self):
        """A button was clicked."""
        if self.loading:
            return
        self.status.config(text=self.model_note)
        self.refresh()

    def refresh(self):
        # Rule 2: never contacted means no previous outcome
        contacted = self.choice["pdays_group"].get() != NOT_CONTACTED
        if not contacted:
            if self.choice["poutcome"].get() != "unknown":
                self.loading = True
                self.choice["poutcome"].set("unknown")
                self.loading = False
        elif self.choice["poutcome"].get() == "unknown" and \
                not self.was_contacted:
            # Leaving 'Not contacted': start from the most common outcome
            # among clients who were contacted
            mode = self.data.loc[self.data["contacted_before"] == "yes",
                                 "poutcome"].mode()[0]
            self.loading = True
            self.choice["poutcome"].set(mode)
            self.loading = False
        self.was_contacted = contacted
        for button in self.buttons["poutcome"]:
            button.config(state="normal" if contacted else "disabled")

        profile = {column: self.choice[column].get()
                   for column, _, _ in PROFILE}
        result = assess(self.models, profile)

        self.show_regression(result)
        self.show_tree(result)
        self.show_rules(contacted)

    def show_regression(self, result):
        probability = result["probability"]
        call = result["call"]
        overall = self.models["overall"]

        colour = CALL if call else NO_CALL
        self.decision.config(text="CALL" if call else "DO NOT CALL",
                             fg=colour)
        difference = result["difference"]
        direction = "above" if difference >= 0 else "below"
        self.probability.config(
            text=f"Probability of subscribing: {probability:.1%}, "
                 f"{abs(difference):.1f} points {direction} the "
                 f"{overall:.1%} overall rate.")

        bar = self.bar
        bar.delete("all")
        left, right, top, bottom = 4, 436, 6, 24
        width = right - left
        bar.create_rectangle(left, top, right, bottom, fill=TRACK, width=0)
        bar.create_rectangle(left, top, left + width * probability, bottom,
                             fill=colour, width=0)
        average = left + width * overall
        threshold = left + width * 0.5
        bar.create_line(average, top - 4, average, bottom + 4, fill=TEXT,
                        width=2)
        bar.create_line(threshold, top - 4, threshold, bottom + 4,
                        fill=TEXT, dash=(3, 2))
        bar.create_text(average - 4, bottom + 6, anchor="ne",
                        font=self.fonts["small"], fill=TEXT,
                        text=f"overall {overall:.1%}")
        bar.create_text(threshold + 4, bottom + 6, anchor="nw",
                        font=self.fonts["small"], fill=TEXT,
                        text="call above 50%")

    def show_tree(self, result):
        total, yes = result["leaf_total"], result["leaf_yes"]
        text = self.path
        text.config(state="normal")
        text.delete("1.0", "end")
        for number, (_, attribute, level, answer) in enumerate(
                result["steps"], start=1):
            text.insert("end", f"{number}.  {attribute} = {level}?  ")
            text.insert("end", "yes" if answer else "no", "bold")
            text.insert("end", "\n")
        text.insert("end", f"Reached by {total:,} training clients, "
                           f"{yes:,} subscribed ({yes / total:.1%}).\n")
        text.insert("end", "Tree predicts: ")
        text.insert("end", "subscribes" if result["tree_says_yes"]
                    else "does not subscribe", "bold")
        if result["call"] != result["tree_says_yes"]:
            text.insert("end", "  (the two models disagree here)", "muted")
        text.config(state="disabled")

    def show_rules(self, contacted):
        active = [True, not contacted, contacted, True]
        for label, on in zip(self.rules, active):
            label.config(fg=TEXT if on else MUTED,
                         font=self.fonts["bold" if on else "body"])


def main():
    models = build()
    report(models)
    with open(REPORT, "w") as handle:
        handle.write("\n".join(lines))
    print(f"Log written to {REPORT}")
    if "--check" in sys.argv:
        return
    root = tk.Tk()
    RuleExplorer(root, models)
    root.resizable(False, False)
    root.mainloop()


if __name__ == "__main__":
    main()
