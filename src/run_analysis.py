#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from figure_style import INK, BLUE, MISSING_FILL, MISSING_INK, heatmap, label_color, configure
from matplotlib.patches import Rectangle

SEED = 20260731
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data"
OUT = ROOT / "outputs"
TAB = OUT / "tables"
FIG = OUT / "figures"
LOG = OUT / "logs"

C1_LABELS = {
    1: "1 Appraisal",
    2: "2 Help-seeking",
    3: "3 Diagnostic work-up",
    4: "4 Pre-treatment",
}

POST_CODES = {
    "A1_Diagnostic labeling": "A1 Diagnostic labeling",
    "A2_Temporal information": "A2 Temporal information",
    "A5_emotional_rep": "A5 Emotional representation",
    "B1_severity": "B1 Severity",
    "B3_duration": "B3 Duration",
    "B4_age_illness": "B4 Age-illness link",
    "B7_stress_illness": "B7 Stress-illness link",
    "D1_symptom_interpretation": "D1 Symptom interpretation",
    "D2_risk_assessment": "D2 Risk assessment",
    "D3_emotional_relief": "D3 Emotional relief",
    "D4_experience_sharing": "D4 Experience sharing",
    "D6_medical process information seeking": "D6 Medical-process information",
}

E_CODES = {
    "E1_informational": "E1 Informational",
    "E2_emotional": "E2 Emotional",
    "E5_help_seeking_encouragement": "E5 Help-seeking encouragement",
    "E6_reassurance_normalization": "E6 Reassurance/normalization",
    "E7_experiential_information": "E7 Experiential information",
}

MAPPINGS = [
    ("D1→E1", "D1_symptom_interpretation", "E1_informational", "Symptom interpretation → informational support"),
    ("D2→E1", "D2_risk_assessment", "E1_informational", "Risk assessment → informational support"),
    ("D6→E1", "D6_medical process information seeking", "E1_informational", "Process information → informational support"),
    ("D3→E2", "D3_emotional_relief", "E2_emotional", "Emotional-relief need → emotional support"),
    ("D2→E5", "D2_risk_assessment", "E5_help_seeking_encouragement", "Risk assessment → help-seeking encouragement"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(x: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    z = 1.959963984540054
    p = x / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return cen - half, cen + half


def prop_row(x: int, n: int) -> dict:
    lo, hi = wilson(int(x), int(n))
    return {"n": int(n), "count": int(x), "proportion": x / n if n else math.nan,
            "ci_low": lo, "ci_high": hi}


def bh(pvals: list[float]) -> list[float]:
    return multipletests(pvals, method="fdr_bh")[1].tolist() if pvals else []


def holm(pvals: list[float]) -> list[float]:
    return multipletests(pvals, method="holm")[1].tolist() if pvals else []


def global_binary_test(df: pd.DataFrame, group: str, outcome: str) -> dict:
    tab = pd.crosstab(df[group], df[outcome]).reindex(columns=[0, 1], fill_value=0)
    if tab.shape[0] < 2 or tab[1].sum() == 0 or tab[0].sum() == 0:
        return {"statistic": math.nan, "df": math.nan, "p_value": math.nan, "method": "not estimable"}
    chi2, p, dof, exp = stats.chi2_contingency(tab.to_numpy(), correction=False)
    if (exp < 5).any():
        observed = chi2
        y = df[outcome].to_numpy().copy()
        g = df[group].to_numpy()
        exceed = 0
        B = 10000
        for _ in range(B):
            RNG.shuffle(y)
            pt = pd.crosstab(g, y).reindex(index=tab.index, columns=[0, 1], fill_value=0)
            s = stats.chi2_contingency(pt.to_numpy(), correction=False)[0]
            exceed += s >= observed - 1e-12
        p = (exceed + 1) / (B + 1)
        return {"statistic": observed, "df": dof, "p_value": p, "method": "10,000-permutation Pearson chi-square"}
    return {"statistic": chi2, "df": dof, "p_value": p, "method": "Pearson chi-square"}


def make_thread_e(threads: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    agg = comments.groupby("thread_id")[list(E_CODES)].max().reset_index()
    out = threads.merge(agg, on="thread_id", how="left")
    # No human response means support is known absent. Human response without retained analyzable
    # content means support type is unknown and remains missing.
    for e in E_CODES:
        out.loc[out["C3_has_human_response"] == 0, e] = 0
    out["support_content_known"] = ((out["C3_has_human_response"] == 0) |
                                    (out["actual_analyzable_comment"] == 1)).astype(int)
    return out


def c1_profiles(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = d[d.C1_interval.isin([1, 2, 3, 4])].copy()
    sample_rows = []
    for c1 in [1, 2, 3, 4]:
        s = primary[primary.C1_interval == c1]
        sample_rows.append({
            "C1": C1_LABELS[c1], "n_threads": len(s),
            "human_response_n": int(s.C3_has_human_response.sum()),
            "human_response_pct": s.C3_has_human_response.mean(),
            "analyzable_response_n": int(s.actual_analyzable_comment.sum()),
            "analyzable_response_pct": s.actual_analyzable_comment.mean(),
            "comments_median": s.n_analyzable_comments.median(),
            "comments_q1": s.n_analyzable_comments.quantile(.25),
            "comments_q3": s.n_analyzable_comments.quantile(.75),
        })
    post_rows, tests = [], []
    for code, label in POST_CODES.items():
        for c1 in [1, 2, 3, 4]:
            s = primary[primary.C1_interval == c1]
            r = prop_row(int(s[code].sum()), len(s))
            post_rows.append({"code": code, "label": label, "C1": C1_LABELS[c1], **r})
        t = global_binary_test(primary, "C1_interval", code)
        tests.append({"family": "post communication codes", "outcome": code, "label": label, **t})
    q = bh([r["p_value"] for r in tests])
    for r, qq in zip(tests, q): r["q_value"] = qq
    return pd.DataFrame(sample_rows), pd.DataFrame(post_rows), pd.DataFrame(tests)


def support_profiles(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = d[d.C1_interval.isin([1, 2, 3, 4]) & (d.support_content_known == 1)].copy()
    rows, tests = [], []
    for e, label in E_CODES.items():
        for c1 in [1, 2, 3, 4]:
            s = primary[primary.C1_interval == c1]
            r = prop_row(int(s[e].sum()), len(s))
            rows.append({"estimand": "all threads with known support content", "code": e,
                         "label": label, "C1": C1_LABELS[c1], **r})
            sr = s[s.actual_analyzable_comment == 1]
            rr = prop_row(int(sr[e].sum()), len(sr))
            rows.append({"estimand": "conditional on analyzable response", "code": e,
                         "label": label, "C1": C1_LABELS[c1], **rr})
        t = global_binary_test(primary, "C1_interval", e)
        tests.append({"family": "thread-level support codes", "outcome": e, "label": label, **t})
    q = bh([r["p_value"] for r in tests])
    for r, qq in zip(tests, q): r["q_value"] = qq
    return pd.DataFrame(rows), pd.DataFrame(tests)


def access_model(d: pd.DataFrame, outcome: str = "C3_has_human_response", add_year: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = d[d.C1_interval.isin([1, 2, 3, 4])].copy()
    X = pd.get_dummies(s.C1_interval.astype(int), prefix="c1", drop_first=True, dtype=float)
    X = sm.add_constant(X)
    if add_year:
        X["year_centered"] = s.year - s.year.mean()
    fit = sm.GLM(s[outcome].astype(float), X, family=sm.families.Poisson()).fit(cov_type="HC3")
    # Marginal probabilities from observed proportions are the clearest estimand.
    probs = []
    for c1 in [1, 2, 3, 4]:
        ss = s[s.C1_interval == c1]
        probs.append({"outcome": outcome, "model": "modified Poisson with HC3",
                      "C1": C1_LABELS[c1], **prop_row(int(ss[outcome].sum()), len(ss))})
    contrasts = []
    pairs = [(2, 1), (3, 2), (4, 3)]
    names = list(fit.params.index)
    for a, b in pairs:
        va = np.zeros(len(names)); vb = np.zeros(len(names))
        va[names.index("const")] = 1; vb[names.index("const")] = 1
        if a != 1: va[names.index(f"c1_{a}")] = 1
        if b != 1: vb[names.index(f"c1_{b}")] = 1
        diff = va - vb
        logpr = float(diff @ fit.params.to_numpy())
        se = float(np.sqrt(diff @ fit.cov_params().to_numpy() @ diff))
        z = logpr / se
        pa = s.loc[s.C1_interval == a, outcome].mean()
        pb = s.loc[s.C1_interval == b, outcome].mean()
        contrasts.append({
            "outcome": outcome, "model": "modified Poisson with HC3" + (" + year" if add_year else ""),
            "contrast": f"C1 {a} vs {b}", "PR": math.exp(logpr),
            "ci_low": math.exp(logpr - 1.96 * se), "ci_high": math.exp(logpr + 1.96 * se),
            "risk_difference": pa - pb, "p_value": 2 * stats.norm.sf(abs(z)),
        })
    adj = holm([r["p_value"] for r in contrasts])
    for r, q in zip(contrasts, adj): r["p_holm"] = q
    return pd.DataFrame(probs), pd.DataFrame(contrasts)


def pathway_row(s: pd.DataFrame, e: str) -> dict:
    n = len(s)
    human = int(s.C3_has_human_response.sum())
    known = s[s.support_content_known == 1]
    responders = s[s.actual_analyzable_comment == 1]
    target_known = int(known[e].fillna(0).sum())
    target_resp = int(responders[e].fillna(0).sum())
    unknown = n - len(known)
    return {
        "need_n": n,
        "human_response_n": human,
        "access_probability": human / n if n else math.nan,
        "analyzable_response_n": len(responders),
        "target_support_among_responders_n": target_resp,
        "conditional_match_probability": target_resp / len(responders) if len(responders) else math.nan,
        "known_content_n": len(known),
        "target_support_known_n": target_known,
        "overall_coverage_known": target_known / len(known) if len(known) else math.nan,
        "unknown_content_n": unknown,
        "coverage_lower_bound": target_known / n if n else math.nan,
        "coverage_upper_bound": (target_known + unknown) / n if n else math.nan,
    }


def core_pathways(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_all, rows_c1, tests = [], [], []
    primary = d[d.C1_interval.isin([1, 2, 3, 4])].copy()
    for mid, need, e, label in MAPPINGS:
        ss = primary[primary[need] == 1]
        rows_all.append({"mapping": mid, "label": label, **pathway_row(ss, e)})
        known_need = ss[ss.support_content_known == 1]
        t = global_binary_test(known_need, "C1_interval", e) if len(known_need) else {
            "statistic": math.nan, "df": math.nan, "p_value": math.nan, "method": "not estimable"}
        tests.append({"family": "core need-support coverage", "mapping": mid, "label": label, **t})
        for c1 in [1, 2, 3, 4]:
            sc = ss[ss.C1_interval == c1]
            rows_c1.append({"mapping": mid, "label": label, "C1": C1_LABELS[c1], **pathway_row(sc, e)})
    q = bh([r["p_value"] for r in tests])
    for r, qq in zip(tests, q): r["q_value"] = qq
    return pd.DataFrame(rows_all), pd.DataFrame(rows_c1), pd.DataFrame(tests)


def topic_within_c1_3(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s3 = d[d.C1_interval == 3].copy()
    eligible_topics = s3.groupby("topic_label").size()
    eligible_topics = eligible_topics[eligible_topics >= 20].index.tolist()
    rows, tests = [], []
    for topic in eligible_topics:
        st = s3[s3.topic_label == topic]
        rows.append({"mapping": "Any human response", "label": "Any human response",
                     "topic": topic, "topic_n": len(st), "need_n": len(st),
                     "human_response_n": int(st.C3_has_human_response.sum()),
                     "access_probability": st.C3_has_human_response.mean(),
                     "analyzable_response_n": int(st.actual_analyzable_comment.sum()),
                     "target_support_among_responders_n": math.nan,
                     "conditional_match_probability": math.nan,
                     "known_content_n": math.nan, "target_support_known_n": math.nan,
                     "overall_coverage_known": math.nan, "unknown_content_n": math.nan,
                     "coverage_lower_bound": math.nan, "coverage_upper_bound": math.nan})
    for mid, need, e, label in MAPPINGS:
        valid_topics = []
        for topic in eligible_topics:
            st = s3[(s3.topic_label == topic) & (s3[need] == 1)]
            if len(st) >= 10:
                rows.append({"mapping": mid, "label": label, "topic": topic,
                             "topic_n": int((s3.topic_label == topic).sum()), **pathway_row(st, e)})
                valid_topics.append(topic)
        formal = s3[(s3.topic_label.isin(valid_topics)) & (s3[need] == 1) & (s3.support_content_known == 1)]
        if len(valid_topics) >= 2:
            t = global_binary_test(formal, "topic_label", e)
            tests.append({"family": "within C1=3 topic coverage", "mapping": mid, "label": label,
                          "eligible_topics": len(valid_topics), **t})
    if tests:
        q = bh([r["p_value"] for r in tests])
        for r, qq in zip(tests, q): r["q_value"] = qq
    return pd.DataFrame(rows), pd.DataFrame(tests)


def topic_overlap_and_standardized_access(d: pd.DataFrame, B: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnose C1-topic overlap and directly standardize adjacent comparisons.

    This is a sensitivity analysis, not a substitute for the descriptive main estimand.
    Each eligible topic receives equal weight. Eligibility requires at least 10 threads in
    both pathway positions so estimates are not driven by nearly empty cells.
    """
    s = d[d.C1_interval.isin([1, 2, 3, 4])].copy()
    counts = s.groupby(["topic_label", "C1_interval"]).size().unstack(fill_value=0).reindex(columns=[1,2,3,4],fill_value=0)
    overlap_rows, std_rows = [], []
    for a,b in [(2,1),(3,2),(4,3)]:
        eligible = counts.index[(counts[a] >= 10) & (counts[b] >= 10)].tolist()
        for topic in counts.index:
            overlap_rows.append({"contrast": f"C1 {a} vs {b}", "topic": topic,
                                 "n_higher_position": int(counts.loc[topic,a]),
                                 "n_lower_position": int(counts.loc[topic,b]),
                                 "eligible_min_10_each": int(topic in eligible)})
        if not eligible:
            continue
        def std_prob(data: pd.DataFrame, c1: int) -> float:
            return float(np.mean([data[(data.topic_label==t)&(data.C1_interval==c1)].C3_has_human_response.mean() for t in eligible]))
        pa, pb = std_prob(s,a), std_prob(s,b)
        draws=[]
        for _ in range(B):
            vals={}
            for c1 in [a,b]:
                tp=[]
                for topic in eligible:
                    cell=s[(s.topic_label==topic)&(s.C1_interval==c1)].C3_has_human_response.to_numpy()
                    tp.append(RNG.choice(cell,size=len(cell),replace=True).mean())
                vals[c1]=float(np.mean(tp))
            draws.append(vals[a]-vals[b])
        std_rows.append({"contrast": f"C1 {a} vs {b}", "eligible_topics": len(eligible),
                         "topics": " | ".join(eligible), "equal_topic_weighted_higher": pa,
                         "equal_topic_weighted_lower": pb, "risk_difference": pa-pb,
                         "rd_ci_low": float(np.quantile(draws,.025)),
                         "rd_ci_high": float(np.quantile(draws,.975)),
                         "bootstrap_replicates": B})
    return pd.DataFrame(overlap_rows), pd.DataFrame(std_rows)


def sensitivity_rank(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in [50, 75, 100]:
        s = d[(d["rank"] <= k) & d.C1_interval.isin([1, 2, 3, 4])]
        for c1 in [1, 2, 3, 4]:
            sc = s[s.C1_interval == c1]
            rows.append({"rank_cutoff": k, "analysis": "human response", "mapping": "C3",
                         "C1": C1_LABELS[c1], **prop_row(int(sc.C3_has_human_response.sum()), len(sc))})
        for mid, need, e, label in MAPPINGS:
            sn = s[(s[need] == 1) & (s.support_content_known == 1)]
            rows.append({"rank_cutoff": k, "analysis": "overall coverage", "mapping": mid,
                         "C1": "All primary C1", **prop_row(int(sn[e].sum()), len(sn))})
    return pd.DataFrame(rows)


def stratified_bootstrap(d: pd.DataFrame, B: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = d[d.C1_interval.isin([1, 2, 3, 4])].copy()
    topics = sorted(primary.topic_id.unique())
    access_draws = {(c1): [] for c1 in [1, 2, 3, 4]}
    coverage_draws = {m[0]: [] for m in MAPPINGS}
    for _ in range(B):
        chunks = []
        for topic in topics:
            st = primary[primary.topic_id == topic]
            idx = RNG.integers(0, len(st), len(st))
            chunks.append(st.iloc[idx])
        b = pd.concat(chunks, ignore_index=True)
        for c1 in [1, 2, 3, 4]:
            bc = b[b.C1_interval == c1]
            access_draws[c1].append(bc.C3_has_human_response.mean() if len(bc) else math.nan)
        for mid, need, e, _ in MAPPINGS:
            bn = b[(b[need] == 1) & (b.support_content_known == 1)]
            coverage_draws[mid].append(bn[e].mean() if len(bn) else math.nan)
    ar, cr = [], []
    for c1, vals in access_draws.items():
        vals = np.asarray(vals, float)
        ar.append({"estimand": "human response probability", "C1": C1_LABELS[c1],
                   "bootstrap_replicates": B, "ci_low": np.nanquantile(vals, .025),
                   "ci_high": np.nanquantile(vals, .975)})
    for mid, vals in coverage_draws.items():
        vals = np.asarray(vals, float)
        cr.append({"estimand": "overall need-matched coverage", "mapping": mid,
                   "bootstrap_replicates": B, "ci_low": np.nanquantile(vals, .025),
                   "ci_high": np.nanquantile(vals, .975)})
    return pd.DataFrame(ar), pd.DataFrame(cr)


def clinical_report_audit(d: pd.DataFrame) -> pd.DataFrame:
    s = d[(d.C1_interval == 3) & (d.topic_label == "Reading Clinical Reports")]
    rows = []
    for mid, need, e, label in MAPPINGS[:3]:
        sn = s[s[need] == 1]
        rows.append({"mapping": mid, "label": label, "topic": "Reading Clinical Reports",
                     **pathway_row(sn, e)})
    return pd.DataFrame(rows)


def save_figures(d: pd.DataFrame, post: pd.DataFrame, support: pd.DataFrame,
                 pathways: pd.DataFrame, topic: pd.DataFrame) -> None:
    """Render auxiliary plots using the same palette and encodings as submission figures."""
    from render_figures import figure2, figure3, MAPPING_ORDER, TOPIC_ORDER, INFO_MAPPINGS
    configure(plt)
    def save(fig, name):
        for extension in ["png", "svg", "pdf"]:
            fig.savefig(FIG / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    def matrix_plot(matrix, name, title):
        cmap = heatmap()
        fig, ax = plt.subplots(figsize=(10, max(4.8, len(matrix) * 0.43 + 1.6)))
        im = ax.imshow(np.ma.masked_invalid(matrix.values), vmin=0, vmax=1,
                       cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(matrix.columns)), ["Appraisal", "Help-seeking", "Diagnostic\nwork-up", "Pre-treatment"])
        ax.set_yticks(range(len(matrix)), matrix.index)
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value * 100:.0f}%", ha="center", va="center",
                            color=label_color(cmap(value)[:3]), fontsize=10)
                else:
                    ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=MISSING_FILL,
                                           edgecolor=MISSING_INK,hatch="///",linewidth=0))
                    ax.text(j,i,"—",ha="center",va="center",color=MISSING_INK,fontsize=10)
        ax.set_title(title, fontsize=13, pad=16)
        fig.colorbar(im, ax=ax, label="Proportion of threads", fraction=.035, pad=.03)
        fig.tight_layout()
        save(fig, name)

    fig, ax = plt.subplots(figsize=(14, 4.0)); ax.axis("off")
    boxes = [(0.01, "Topic model corpus\n15,037 threads"),
             (0.205, "Equal-allocation\npurposive sample\n100 threads × 10 topics"),
             (0.40, "Frozen human coding\n1,000 threads"),
             (0.595, "Response access\nC3 on all threads"),
             (0.79, "Support-category coverage\n2,532 comments\nin 719 threads")]
    for x, text in boxes:
        ax.add_patch(plt.Rectangle((x,.34),.17,.36,facecolor="#F4F7F9",edgecolor=BLUE,lw=1.4))
        ax.text(x+.085,.52,text,ha="center",va="center",fontsize=10,color=INK)
    for x in [.18,.375,.57,.765]:
        ax.annotate("",xy=(x+.023,.52),xytext=(x,.52),arrowprops=dict(arrowstyle="->",color=INK))
    ax.text(.5,.12,"Primary axis: pathway position (C1). Topic refines context within a pathway position.",ha="center",color=INK)
    save(fig,"figure1_study_and_analysis_flow")

    matrix = post.pivot(index="label",columns="C1",values="proportion").reindex(columns=list(C1_LABELS.values()))
    matrix_plot(matrix,"figure2_c1_communication_heatmap","Communication features by pathway position")
    values = pathways.set_index("mapping").loc[MAPPING_ORDER]
    path = figure2(values, FIG)
    for extension in [".png", ".svg", ".pdf"]:
        path.with_suffix(extension).replace(FIG / ("figure3_two_stage_support_pathway" + extension))
    known = support[support.estimand == "all threads with known support content"]
    matrix = known.pivot(index="label",columns="C1",values="proportion").reindex(columns=list(C1_LABELS.values()))
    matrix_plot(matrix,"figure4_c1_support_heatmap","Support-category presence by pathway position")
    info = topic[topic.mapping.isin(INFO_MAPPINGS)]
    pivot = info.pivot(index="topic",columns="mapping",values="overall_coverage_known").reindex(index=TOPIC_ORDER,columns=INFO_MAPPINGS)
    path = figure3(pivot, FIG)
    for extension in [".png", ".svg", ".pdf"]:
        path.with_suffix(extension).replace(FIG / ("figure5_c1_3_topic_information_gaps" + extension))


def main() -> None:
    print("Running frozen statistical analysis (10,000 permutations; 2,000 bootstrap replicates)...", flush=True)
    for p in [TAB, FIG, LOG]: p.mkdir(parents=True, exist_ok=True)
    threads = pd.read_csv(INP / "threads.csv").sort_values(["topic_id", "rank"]).reset_index(drop=True)
    comments = pd.read_csv(INP / "comments.csv")
    d = make_thread_e(threads, comments)
    # Intermediate record-level tables are not exported; inputs contain the coded records.

    qc = {
        "threads": len(threads), "unique_threads": threads.thread_id.nunique(),
        "comments": len(comments), "unique_comments": comments.comment_id.nunique(),
        "comment_threads": comments.thread_id.nunique(),
        "unlinked_comments": int((~comments.thread_id.isin(threads.thread_id)).sum()),
        "C1_counts": threads.C1_interval.fillna("NA").astype(str).value_counts().to_dict(),
        "C3_human_response_threads": int(threads.C3_has_human_response.sum()),
        "analyzable_response_threads": int(threads.actual_analyzable_comment.sum()),
        "response_content_unknown_threads": int(((threads.C3_has_human_response==1)&(threads.actual_analyzable_comment==0)).sum()),
        "threads_sha256": sha256(INP/"threads.csv"), "comments_sha256": sha256(INP/"comments.csv"),
    }
    (LOG/"qc.json").write_text(json.dumps(qc,indent=2,ensure_ascii=False),encoding="utf-8")
    assert qc["threads"] == qc["unique_threads"] == 1000
    assert qc["comments"] == qc["unique_comments"] == 2532
    assert qc["comment_threads"] == 719 and qc["unlinked_comments"] == 0
    assert qc["response_content_unknown_threads"] == 11

    sample, post, post_tests = c1_profiles(d)
    support, support_tests = support_profiles(d)
    access_probs, access_contrasts = access_model(d)
    _, access_year = access_model(d, add_year=True)
    analyzable_probs, analyzable_contrasts = access_model(d, outcome="actual_analyzable_comment")
    pathway_all, pathway_c1, pathway_tests = core_pathways(d)
    print("Core models complete; running topic comparisons...", flush=True)
    topic, topic_tests = topic_within_c1_3(d)
    sens_rank = sensitivity_rank(d)
    print("Running topic-stratified bootstrap...", flush=True)
    boot_access, boot_coverage = stratified_bootstrap(d, B=2000)
    report_audit = clinical_report_audit(d)
    print("Running equal-topic-weight sensitivity bootstrap...", flush=True)
    overlap, std_access = topic_overlap_and_standardized_access(d, B=2000)
    all_c1_counts = (d.assign(C1=d.C1_interval.map(C1_LABELS).fillna("NA/indeterminate"))
                      .groupby("C1",dropna=False).agg(
                          n_threads=("thread_id","size"),
                          human_response_n=("C3_has_human_response","sum"),
                          analyzable_response_n=("actual_analyzable_comment","sum")).reset_index())
    all_c1_counts["human_response_pct"] = all_c1_counts.human_response_n / all_c1_counts.n_threads
    all_c1_counts["analyzable_response_pct"] = all_c1_counts.analyzable_response_n / all_c1_counts.n_threads
    primary_c1 = d[d.C1_interval.isin([1,2,3,4])]
    access_global = pd.DataFrame([
        {"outcome":"C3_has_human_response", **global_binary_test(primary_c1,"C1_interval","C3_has_human_response")},
        {"outcome":"actual_analyzable_comment", **global_binary_test(primary_c1,"C1_interval","actual_analyzable_comment")},
    ])

    topic_dist = (d.assign(C1=d.C1_interval.map(C1_LABELS).fillna("NA/indeterminate"))
                    .groupby(["topic_label","C1"],dropna=False).size().reset_index(name="n"))
    outputs = {
        "00_c1_counts_all.csv": all_c1_counts,
        "01_c1_sample_flow.csv": sample, "02_c1_topic_distribution.csv": topic_dist,
        "03_c1_post_code_profiles.csv": post, "04_post_code_global_tests.csv": post_tests,
        "05_c1_support_profiles.csv": support, "06_support_global_tests.csv": support_tests,
        "07_access_probabilities.csv": access_probs, "08_access_adjacent_contrasts.csv": access_contrasts,
        "09_core_pathway_overall.csv": pathway_all, "10_core_pathway_by_c1.csv": pathway_c1,
        "11_core_pathway_global_tests.csv": pathway_tests,
        "12_c1_3_topic_pathways.csv": topic, "13_c1_3_topic_global_tests.csv": topic_tests,
        "14_rank_sensitivity.csv": sens_rank, "15_access_year_adjusted.csv": access_year,
        "16_analyzable_response_sensitivity_probabilities.csv": analyzable_probs,
        "17_analyzable_response_sensitivity_contrasts.csv": analyzable_contrasts,
        "18_stratified_bootstrap_access.csv": boot_access,
        "19_stratified_bootstrap_coverage.csv": boot_coverage,
        "20_clinical_reports_data_informed_audit.csv": report_audit,
        "21_topic_overlap_diagnostics.csv": overlap,
        "22_topic_standardized_access_sensitivity.csv": std_access,
        "23_access_global_tests.csv": access_global,
    }
    for name, df in outputs.items(): df.to_csv(TAB/name,index=False)
    save_figures(d, post, support, pathway_all, topic)

    summary = {"qc": qc,
               "access": access_probs.to_dict("records"),
               "access_contrasts": access_contrasts.to_dict("records"),
               "pathways": pathway_all.to_dict("records"),
               "pathway_tests": pathway_tests.to_dict("records"),
               "clinical_reports": report_audit.to_dict("records")}
    (LOG/"results_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    print(json.dumps(qc, ensure_ascii=False))


if __name__ == "__main__":
    main()
