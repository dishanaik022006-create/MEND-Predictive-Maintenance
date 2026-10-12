"""MEND - Machine Equipment Navigator & Diagnostics
AI-Based Predictive Maintenance for Industrial Machines.
Run:  streamlit run app.py
Needs in the same folder: ai4i2020.csv, best_model.joblib (model_results.csv optional).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
G, A, R, TXT, MUT, BG, CARD, LINE = "#2fbf85", "#d9a441", "#d9534f", "#e8ecea", "#9aa7a1", "#111413", "#1a1f1d", "#2a312e"

st.set_page_config(page_title="MEND", page_icon="⚙", layout="wide")
st.markdown(f"""<style>
.stApp{{background:{BG};color:{TXT}}}
[data-testid=stSidebar]{{background:#0b0d0c;border-right:1px solid {LINE}}}
[data-testid=stSidebar] *{{color:{TXT}!important}}
[data-testid=stSidebar] label p{{font-size:.95rem}}
h1,h2,h3,h4,p,li,label,span,div{{color:{TXT}}}
.block-container{{padding-top:1.6rem;max-width:1250px}}
.card{{background:{CARD};border:1px solid {LINE};border-radius:8px;padding:16px 18px;margin-bottom:12px}}
.kpi .v{{font-size:1.9rem;font-weight:700;line-height:1.1}} .kpi .l{{color:{MUT};font-size:.8rem;text-transform:uppercase;letter-spacing:.06em}}
.kpi .s{{color:{MUT};font-size:.78rem;margin-top:4px}}
.tag{{display:inline-block;padding:2px 10px;border-radius:4px;font-size:.78rem;font-weight:600;border:1px solid}}
.src{{color:{MUT};font-size:.8rem;margin:-6px 0 10px}}
.stButton>button,.stDownloadButton>button{{background:{G};color:#06130d!important;border:0;font-weight:700;border-radius:6px}}
.stButton>button:hover{{background:#3ad596}}
[data-baseweb=input] input,[data-baseweb=select]>div{{background:{CARD}!important;color:{TXT}!important}}
[data-testid=stDataFrame]{{border:1px solid {LINE};border-radius:6px}}
@media(max-width:700px){{.kpi .v{{font-size:1.4rem}}}}
</style>""", unsafe_allow_html=True)

PCOL = {"Critical": R, "High": "#e07a3f", "Medium": A, "Low": G}


def kpi(label, value, sub="", color=TXT):
    return f'<div class="card kpi"><div class="l">{label}</div><div class="v" style="color:{color}">{value}</div><div class="s">{sub}</div></div>'


def style(fig, h=330):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TXT,
                      margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    fig.update_xaxes(gridcolor=LINE)
    fig.update_yaxes(gridcolor=LINE)
    return fig


# ---------------------------------------------------------------- data / model
def concept(name):
    n = str(name).lower()
    for key, c in [("diff", "tdiff"), ("delta", "tdiff"), ("power", "power"), ("strain", "strain"), ("air", "air"),
                   ("process", "proc"), ("rot", "rpm"), ("rpm", "rpm"), ("speed", "rpm"), ("torque", "torque"),
                   ("wear", "wear"), ("type", "type")]:
        if key in n:
            return c
    return None


@st.cache_data
def load_data():
    p = BASE / "ai4i2020.csv"
    if not p.exists():
        return None
    raw = pd.read_csv(p)
    d = pd.DataFrame({"type": raw.get("Type", "M")})
    for col in raw.columns:
        c = concept(col) if col != "Machine failure" else None
        if c in ("air", "proc", "rpm", "torque", "wear") and c not in d:
            d[c] = raw[col]
    d["fail"] = raw["Machine failure"].astype(int)
    for m in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        if m in raw:
            d[m] = raw[m].astype(int)
    d.insert(0, "id", raw["Product ID"] if "Product ID" in raw else raw.index.astype(str))
    return d


@st.cache_resource
def load_model():
    p = BASE / "best_model.joblib"
    if not p.exists():
        return None, f"best_model.joblib not found in {BASE}"
    try:
        return joblib.load(p), None
    except Exception as e:  # noqa
        return None, str(e)


def model_features(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "steps") and hasattr(m.steps[0][1], "feature_names_in_"):
        return list(m.steps[0][1].feature_names_in_)
    return None


def derived(d):
    d = d.copy()
    d["power"] = d["torque"] * d["rpm"] * 2 * np.pi / 60
    d["tdiff"] = d["proc"] - d["air"]
    d["strain"] = d["wear"] * d["torque"]
    return d


def build_X(d, m):
    feats = model_features(m)
    if feats is None:
        raise ValueError("Model does not expose its feature names; cannot build a safe input.")
    d = derived(d)
    cols = {}
    for f in feats:
        c = concept(f)
        if c is None or c not in d:
            raise ValueError(f"Model needs feature '{f}', which this app cannot supply.")
        s = d[c]
        if c == "type" and not hasattr(m, "steps"):  # bare estimator: assume LabelEncoder order H,L,M
            s = s.map({"H": 0, "L": 1, "M": 2})
        cols[f] = s.values
    return pd.DataFrame(cols, columns=feats)


def predict(d, m):
    X = build_X(d, m)
    pred = m.predict(X)
    proba = m.predict_proba(X) if hasattr(m, "predict_proba") else None
    return pred, proba, list(getattr(m, "classes_", []))


def label(v):
    return {0: "Normal", 1: "Failure risk", "0": "Normal", "1": "Failure risk"}.get(v, str(v))


# ---------------------------------------------------------------- rules
LIMIT = {"L": 11000, "M": 12000, "H": 13000}


def rules(d):
    d = derived(d)
    lim = d["type"].map(LIMIT).fillna(12000)
    P = d["power"]
    return [  # (mask, severity, reason, action)
        ((d["wear"] >= 200), "High", "Tool wear ≥ 200 min", "Replace or re-sharpen the tool at next stop."),
        (((d["tdiff"] < 8.6) & (d["rpm"] < 1380)), "High", "Low temperature gap with low speed (heat-dissipation risk)", "Check coolant flow, fans and heat exchangers."),
        (((P < 3500) | (P > 9000)), "High", "Mechanical power outside 3.5-9 kW band", "Inspect drive, load and motor condition."),
        ((d["strain"] > lim), "High", "Wear × torque above overstrain limit", "Reduce load and inspect tool/spindle."),
        (((d["wear"] >= 150) & (d["wear"] < 200)), "Medium", "Tool wear approaching 200 min", "Schedule tool change soon."),
        (((d["strain"] > 0.85 * lim) & (d["strain"] <= lim)), "Medium", "Wear × torque near overstrain limit", "Monitor load; plan an inspection."),
        ((((P >= 3500) & (P < 3850)) | ((P > 8100) & (P <= 9000))), "Medium", "Power close to its limits", "Trend power draw over the next shifts."),
        (((d["tdiff"] >= 8.6) & (d["tdiff"] < 9.5) & (d["rpm"] < 1500)), "Medium", "Temperature gap getting small", "Watch cooling performance."),
    ]


def prioritize(d):
    rank = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    pr = pd.Series("Low", index=d.index)
    why = pd.Series("", index=d.index)
    act = pd.Series("", index=d.index)
    for mask, sev, reason, action in rules(d):
        why = why.where(~mask, why + reason + "; ")
        act = act.where(~mask, act + action + " ")
        up = mask & (pr.map(rank) < rank[sev])
        pr = pr.where(~up, sev)
    if "fail" in d:
        rec = d["fail"] == 1
        pr = pr.where(~rec, "Critical")
        why = why.where(~rec, "Recorded failure in dataset; " + why)
        act = act.where(~rec, "Root-cause the failure and repair before restart. " + act)
    why = why.where(why != "", "No rule triggered")
    act = act.where(act != "", "Routine monitoring")
    return pr, why.str.rstrip("; "), act


@st.cache_data
def planner_table(_m, has_model):
    d = load_data()
    pr, why, act = prioritize(d)
    t = d.assign(priority=pr, reasons=why, action=act)
    t["model_failure_prob"] = np.nan
    if has_model:
        try:
            _, proba, classes = predict(d, _m)
            if proba is not None and 1 in classes:
                t["model_failure_prob"] = proba[:, classes.index(1)].round(3)
        except Exception:  # noqa
            pass
    return t


# ---------------------------------------------------------------- pages
data = load_data()
model, model_err = load_model()

st.sidebar.markdown(f"<h2 style='margin:0;color:{G}!important'>MEND</h2><div style='color:{MUT};font-size:.8rem;margin-bottom:14px'>Machine Equipment Navigator &amp; Diagnostics</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Plant Briefing", "Condition Lab", "Maintenance Planner", "Model Insights", "About & Method"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(("Model: loaded ✓" if model is not None else "Model: not loaded ✗") + (" | Data: loaded ✓" if data is not None else " | Data: missing ✗"))

if data is None:
    st.error("ai4i2020.csv not found next to app.py.")
    st.stop()

# ---- 1 Plant Briefing
if page == "Plant Briefing":
    st.title("Plant Briefing")
    st.markdown('<div class="src">Source: historical dataset records (AI4I 2020), not a live feed.</div>', unsafe_allow_html=True)
    n, f = len(data), int(data.fail.sum())
    modes = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in data]
    c = st.columns(4)
    c[0].markdown(kpi("Records", f"{n:,}", "machine operating records"), unsafe_allow_html=True)
    c[1].markdown(kpi("Recorded failures", f"{f:,}", "Machine failure = 1", R), unsafe_allow_html=True)
    c[2].markdown(kpi("Failure rate", f"{f / n:.2%}", "failures / records", A), unsafe_allow_html=True)
    c[3].markdown(kpi("Healthy records", f"{n - f:,}", f"{(n - f) / n:.2%} of dataset", G), unsafe_allow_html=True)
    a, b = st.columns(2)
    t = data.groupby("type").agg(records=("fail", "size"), failures=("fail", "sum")).reset_index()
    t["failure_rate_%"] = (t.failures / t.records * 100).round(2)
    a.plotly_chart(style(px.bar(t, x="type", y="failure_rate_%", title="Failure rate by product type (%)", color_discrete_sequence=[G])), use_container_width=True)
    if modes:
        mc = data[modes].sum().reset_index()
        mc.columns = ["mode", "count"]
        b.plotly_chart(style(px.bar(mc, x="mode", y="count", title="Failures by recorded failure mode", color_discrete_sequence=[A])), use_container_width=True)
    a, b = st.columns(2)
    d2 = data.assign(state=data.fail.map({0: "Normal", 1: "Failure"}))
    a.plotly_chart(style(px.histogram(d2, x="wear", color="state", barmode="overlay", opacity=.7, title="Tool wear (min) distribution",
                                      color_discrete_map={"Normal": G, "Failure": R})), use_container_width=True)
    b.plotly_chart(style(px.scatter(d2.sample(min(3000, n), random_state=1), x="rpm", y="torque", color="state", opacity=.6,
                                    title="Torque vs rotational speed (sample of 3,000)", color_discrete_map={"Normal": G, "Failure": R})), use_container_width=True)
    top = t.sort_values("failure_rate_%", ascending=False).iloc[0]
    msg = f"Of {n:,} records, {f:,} ({f / n:.2%}) are recorded failures. Product type <b>{top['type']}</b> has the highest failure rate ({top['failure_rate_%']}%)."
    if modes:
        mm = data[modes].sum().idxmax()
        msg += f" The most frequent recorded failure mode is <b>{mm}</b> ({int(data[mm].sum())} records)."
    avg = data.groupby("fail").wear.mean()
    if 1 in avg.index:
        msg += f" Failed records show mean tool wear of {avg[1]:.0f} min vs {avg[0]:.0f} min for normal ones."
    st.markdown(f'<div class="card"><b>Operational summary</b><br>{msg}</div>', unsafe_allow_html=True)

# ---- 2 Condition Lab
elif page == "Condition Lab":
    st.title("Condition Lab")
    st.markdown('<div class="src">Source: live input you enter below, scored by your trained model.</div>', unsafe_allow_html=True)
    if model is None:
        st.error(f"Model unavailable: {model_err}")
        st.stop()
    feats = model_features(model)
    st.caption("Model expects: " + (", ".join(feats) if feats else "unknown"))
    EX = {"Typical": dict(type="M", air=300.0, proc=310.0, rpm=1500, torque=40.0, wear=100),
          "High tool wear": dict(type="L", air=300.0, proc=310.0, rpm=1500, torque=45.0, wear=230),
          "Overload": dict(type="L", air=300.0, proc=310.5, rpm=1350, torque=65.0, wear=200)}
    lo = data[["air", "proc", "rpm", "torque", "wear"]].agg(["min", "max"])
    for k, v in EX["Typical"].items():
        st.session_state.setdefault("in_" + k, v)

    def setex(name):
        for k, v in EX[name].items():
            st.session_state["in_" + k] = v

    b = st.columns(4)
    b[0].button("Example: typical", on_click=setex, args=("Typical",))
    b[1].button("Example: high wear", on_click=setex, args=("High tool wear",))
    b[2].button("Example: overload", on_click=setex, args=("Overload",))
    b[3].button("Reset", on_click=setex, args=("Typical",))
    a, c = st.columns(2)
    typ = a.selectbox("Product quality type", ["L", "M", "H"], key="in_type", help="L = low, M = medium, H = high quality variant")
    air = c.number_input("Air temperature (K)", float(lo.air["min"]) - 5, float(lo.air["max"]) + 5, key="in_air", step=0.1, help="Ambient air around the machine")
    proc = a.number_input("Process temperature (K)", float(lo.proc["min"]) - 5, float(lo.proc["max"]) + 5, key="in_proc", step=0.1, help="Temperature of the process itself")
    rpm = c.number_input("Rotational speed (rpm)", 500, 4000, key="in_rpm", step=10, help="Spindle speed")
    torque = a.number_input("Torque (Nm)", 0.0, 100.0, key="in_torque", step=0.5, help="Load on the spindle")
    wear = c.number_input("Tool wear (min)", 0, 400, key="in_wear", step=1, help="Minutes the tool has been in use")
    go = st.button("Analyze Machine Condition", type="primary")
    if go:
        errs = []
        if proc <= air:
            errs.append("Process temperature should be higher than air temperature.")
        if not (lo.rpm["min"] * .5 <= rpm <= lo.rpm["max"] * 1.5):
            errs.append("Rotational speed is far outside the training range.")
        for e in errs:
            st.warning(e)
        row = pd.DataFrame([dict(type=typ, air=air, proc=proc, rpm=rpm, torque=torque, wear=wear)])
        try:
            pred, proba, classes = predict(row, model)
        except Exception as e:  # noqa
            st.error(f"Prediction failed: {e}")
            st.stop()
        lab = label(pred[0])
        risky = lab == "Failure risk"
        col = R if risky else G
        st.markdown(f'<div class="card" style="border-left:4px solid {col}"><div class="l" style="color:{MUT}">MODEL PREDICTION</div>'
                    f'<div style="font-size:1.8rem;font-weight:700;color:{col}">{lab}</div></div>', unsafe_allow_html=True)
        if proba is not None:
            pdf = pd.DataFrame({"class": [label(c) for c in classes], "probability": proba[0]})
            st.plotly_chart(style(px.bar(pdf, x="probability", y="class", orientation="h", range_x=[0, 1], title="Class probabilities (from the model)",
                                         color="class", color_discrete_map={"Normal": G, "Failure risk": R}), 220), use_container_width=True)
        st.markdown("**What this means:** " + ("The model sees a pattern similar to past failures. Treat it as a prompt to inspect, not a certainty." if risky else
                    "The inputs look similar to normal operating records. This does not guarantee nothing will go wrong."))
        pr, why, act = prioritize(row)
        st.markdown(f'**Rule-based suggestion** <span class="tag" style="color:{PCOL[pr[0]]};border-color:{PCOL[pr[0]]}">{pr[0]}</span>', unsafe_allow_html=True)
        st.write("Reasons: " + why[0])
        st.write("Suggested action: " + act[0])
        st.caption("Suggestions come from fixed threshold rules (AI4I failure-mode definitions), not from the model, and are not guaranteed diagnoses.")

# ---- 3 Maintenance Planner
elif page == "Maintenance Planner":
    st.title("Maintenance Planner")
    st.markdown('<div class="src">Three different things are shown: <b>recorded failures</b> (dataset), <b>model failure probability</b> (model), '
                'and <b>priority/actions</b> (fixed rules). This is not a live plant feed.</div>', unsafe_allow_html=True)
    t = planner_table(model, model is not None)
    cnt = t.priority.value_counts()
    c = st.columns(4)
    for i, p in enumerate(["Critical", "High", "Medium", "Low"]):
        c[i].markdown(kpi(p, f"{cnt.get(p, 0):,}", "records", PCOL[p]), unsafe_allow_html=True)
    with st.expander("Priority rules"):
        st.markdown("- **Critical**: dataset records a failure.\n- **High**: tool wear ≥ 200 min; temperature gap < 8.6 K with speed < 1380 rpm; power outside 3.5-9 kW; wear×torque above 11k/12k/13k (L/M/H).\n"
                    "- **Medium**: wear 150-199 min; within 15% of overstrain limit; power near limits; small temperature gap.\n- **Low**: no rule triggered.")
    a, b, d = st.columns(3)
    sel = a.multiselect("Priority", ["Critical", "High", "Medium", "Low"], default=["Critical", "High"])
    ty = b.multiselect("Product type", ["L", "M", "H"], default=["L", "M", "H"])
    sort = d.selectbox("Sort by", ["priority", "wear", "torque", "model_failure_prob"])
    v = t[t.priority.isin(sel) & t.type.isin(ty)].copy()
    v["_r"] = v.priority.map({"Critical": 0, "High": 1, "Medium": 2, "Low": 3})
    v = v.sort_values(["_r"] if sort == "priority" else [sort, "_r"], ascending=[True] if sort == "priority" else [False, True])
    show = v[["id", "type", "priority", "fail", "model_failure_prob", "wear", "torque", "rpm", "reasons", "action"]].rename(columns={"fail": "recorded_failure"})
    st.caption(f"{len(show):,} records match.")
    st.dataframe(show.head(1000), use_container_width=True, hide_index=True)
    st.download_button("Download maintenance report (CSV)", show.to_csv(index=False).encode(), "mend_maintenance_report.csv", "text/csv")

# ---- 4 Model Insights
elif page == "Model Insights":
    st.title("Model Insights")
    p = BASE / "model_results.csv"
    if p.exists():
        r = pd.read_csv(p)
        st.markdown('<div class="src">Source: your training run (model_results.csv).</div>', unsafe_allow_html=True)
        st.dataframe(r, use_container_width=True, hide_index=True)
        num = r.select_dtypes("number").columns.tolist()
        name = next((c for c in r.columns if r[c].dtype == object), None)
        if num and name:
            m = r.melt(id_vars=name, value_vars=num)
            st.plotly_chart(style(px.bar(m, x=name, y="value", color="variable", barmode="group", title="Model comparison")), use_container_width=True)
    else:
        st.info("model_results.csv not found.")
    if model is not None:
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        st.markdown(f"**Loaded model:** `{type(est).__name__}`")
        try:
            names = list(est.feature_names_in_) if hasattr(est, "feature_names_in_") else None
            if hasattr(model, "steps") and hasattr(model.steps[0][1], "get_feature_names_out"):
                names = list(model.steps[0][1].get_feature_names_out())
            if hasattr(est, "feature_importances_") and names and len(names) == len(est.feature_importances_):
                fi = pd.DataFrame({"feature": names, "importance": est.feature_importances_}).sort_values("importance")
                st.plotly_chart(style(px.bar(fi, x="importance", y="feature", orientation="h", title="Feature importance", color_discrete_sequence=[G])), use_container_width=True)
        except Exception:  # noqa
            pass

# ---- 5 About
else:
    st.title("About & Method")
    st.markdown("""
**MEND** is a predictive-maintenance demo. A scikit-learn model trained on the public **AI4I 2020** dataset estimates whether a machine
operating point resembles past failures.

**Important limits**
- The dataset is synthetic and historical; MEND has no live sensor connection.
- Model output is a statistical estimate, not a diagnosis.
- Priority levels and actions are fixed threshold rules from the AI4I failure-mode definitions (TWF, HDF, PWF, OSF), shown separately from model output.

**Run on Windows:** `pip install streamlit pandas numpy scikit-learn plotly joblib` then `streamlit run app.py`.
Use the same scikit-learn version that trained `best_model.joblib`.
""")
    st.subheader("Dataset preview")
    st.dataframe(data.head(50), use_container_width=True, hide_index=True) 