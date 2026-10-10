"""Petro Data Platform: 5 oil & gas data projects in one Streamlit app.
Run:  pip install -r requirements.txt   then   streamlit run app.py
Tabs 1-3 share ONE well (demo, or your LAS/CSV). Tabs 4-5 take their own data.
"""
import io
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="Petro Data Platform", layout="wide")


# ---------- demo data, so everything works before you download real datasets ----------
@st.cache_data
def demo_well(n=600, seed=0):
    rng = np.random.default_rng(seed)
    nb = n // 30
    litho = np.repeat(rng.choice([0, 1, 2], size=nb, p=[.4, .4, .2]), 30)  # 0 sand, 1 shale, 2 limestone
    mu = lambda v: np.array(v)[litho]
    hc = np.repeat(rng.random(nb) < .5, 30) & (litho == 0)  # some sands carry hydrocarbons
    return pd.DataFrame({
        "DEPTH": np.linspace(2000, 2300, n),
        "GR": mu([40, 110, 25]) + rng.normal(0, 8, n),
        "RHOB": mu([2.30, 2.55, 2.65]) + rng.normal(0, .03, n),
        "NPHI": mu([.18, .35, .08]) + rng.normal(0, .02, n),
        "RT": np.where(hc, 40, mu([4, 2, 40])) * np.exp(rng.normal(0, .15, n)),
        "DT": mu([85, 105, 55]) + rng.normal(0, 3, n),
        "LITHO": np.array(["Sand", "Shale", "Limestone"])[litho],
    })


@st.cache_data
def demo_seismic(nx=120, nt=300):
    rng = np.random.default_rng(1)
    x = np.arange(nx)
    refl = np.zeros((nx, nt))
    for base, dip, amp in [(60, .2, 1), (110, -.1, -.8), (160, .15, .9), (210, .05, -1), (250, -.2, .7)]:
        refl[x, (base + dip * x + 8 * (x > 60)).astype(int)] += amp  # dipping layers + a fault at trace 60
    a = (np.pi * 25 * np.arange(-20, 21) * 0.004) ** 2
    ricker = (1 - 2 * a) * np.exp(-a)  # 25 Hz Ricker wavelet
    return np.array([np.convolve(r, ricker, "same") for r in refl]) + rng.normal(0, .05, (nx, nt))


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def tracks(d, spec, flag=None):
    """Draw log tracks. spec = [(title, [(column, color)], log_x)]; flag shades the pay zones."""
    fig, ax = plt.subplots(1, len(spec), figsize=(1.7 * len(spec), 7), sharey=True)
    ax = np.atleast_1d(ax)
    for a, (title, curves, log) in zip(ax, spec):
        for col, color in curves:
            a.plot(d[col], d.DEPTH, color=color, lw=.8)
        if log:
            a.set_xscale("log")
        if flag is not None:
            a.fill_betweenx(d.DEPTH, *a.get_xlim(), where=flag.values, color="gold", alpha=.25)
        a.set_title(title, fontsize=8)
        a.grid(alpha=.3)
    ax[0].invert_yaxis()
    return fig


# ---------- sidebar: load ONE well and map its curves ----------
st.title("🛢️ Petro Data Platform")
st.sidebar.header("Well data")
up = st.sidebar.file_uploader("LAS or CSV (optional)", type=["las", "csv"])
if up is None:
    df = demo_well()
    st.sidebar.caption("Using the synthetic demo well.")
elif up.name.lower().endswith(".las"):
    import lasio
    df = lasio.read(io.StringIO(up.getvalue().decode("utf-8", "ignore"))).df().reset_index()
else:
    df = pd.read_csv(up)

if "WELL" in df.columns:  # e.g. the FORCE 2020 file holds many wells
    df = df[df["WELL"] == st.sidebar.selectbox("Well", df["WELL"].unique())]

cols = list(df.columns)


def pick(label, *names):
    idx = next((cols.index(n) for n in names if n in cols), 0)
    return st.sidebar.selectbox(label, cols, index=idx)


w = pd.DataFrame({
    "DEPTH": df[pick("Depth", "DEPTH", "DEPT", "DEPTH_MD")],
    "GR": df[pick("Gamma ray", "GR")],
    "RHOB": df[pick("Density", "RHOB")],
    "NPHI": df[pick("Neutron", "NPHI")],
    "RT": df[pick("Resistivity", "RT", "RDEP", "ILD", "RESD")],
    "DT": df[pick("Sonic", "DT", "DTC")],
})

tabs = st.tabs(["1 · Well logs", "2 · Lithology ML", "3 · Log prediction", "4 · Decline curves", "5 · Seismic"])

# ---------- 1. Petrophysics ----------
with tabs[0]:
    p = st.columns(4)
    gr0 = p[0].number_input("GR clean", value=float(w.GR.quantile(.05)))
    gr1 = p[1].number_input("GR shale", value=float(w.GR.quantile(.95)))
    rw = p[2].number_input("Rw (ohm.m)", value=0.15)
    m_exp = p[3].number_input("m", value=2.0)
    q = st.columns(3)
    vc = q[0].slider("Vsh cutoff <", 0.0, 1.0, 0.35)
    pc = q[1].slider("PHI cutoff >", 0.0, 0.3, 0.08)
    sc = q[2].slider("Sw cutoff <", 0.0, 1.0, 0.5)

    vsh = ((w.GR - gr0) / (gr1 - gr0)).clip(0, 1)  # linear gamma-ray index
    phi = (((2.65 - w.RHOB) / 1.65 + w.NPHI) / 2).clip(0.01, 0.4)  # avg of density & neutron porosity
    sw = ((rw / (phi ** m_exp * w.RT.clip(lower=0.01))) ** 0.5).clip(0, 1)  # Archie, a=1, n=2
    pay = (vsh < vc) & (phi > pc) & (sw < sc)
    r = w.assign(VSH=vsh, PHI=phi, SW=sw)

    dz = w.DEPTH.diff().median()
    k = st.columns(4)
    k[0].metric("Gross", f"{w.DEPTH.max() - w.DEPTH.min():.0f}")
    k[1].metric("Net pay", f"{pay.sum() * dz:.1f}")
    k[2].metric("Avg PHI", f"{phi[pay].mean():.2f}" if pay.any() else "-")
    k[3].metric("Avg Sw", f"{sw[pay].mean():.2f}" if pay.any() else "-")
    show(tracks(r, [("GR", [("GR", "green")], False), ("RT", [("RT", "red")], True),
                    ("RHOB", [("RHOB", "blue")], False), ("NPHI", [("NPHI", "purple")], False),
                    ("VSH / PHI", [("VSH", "gray"), ("PHI", "orange")], False),
                    ("Sw", [("SW", "teal")], False)], pay))
    st.caption("Gold = net pay. Always sanity-check: is the porosity geologically reasonable, and why these cutoffs?")

# ---------- 2. Lithology classification ----------
with tabs[1]:
    lab = st.selectbox("Lithology label column", cols, index=cols.index("LITHO") if "LITHO" in cols else len(cols) - 1)
    d = w.assign(Y=df[lab]).dropna()
    X = d[["GR", "RHOB", "NPHI", "RT", "DT"]].assign(RT=lambda t: np.log10(t.RT.clip(lower=0.01)))
    y, names = pd.factorize(d.Y)
    cut = int(len(d) * 0.7)  # split by depth (not random) to avoid leakage between neighbouring samples
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=0).fit(X.iloc[:cut], y[:cut])
    st.metric("Accuracy on the deeper 30% (blind test)", f"{accuracy_score(y[cut:], rf.predict(X.iloc[cut:])):.1%}")
    st.bar_chart(pd.Series(rf.feature_importances_, index=X.columns, name="importance"))
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 6), sharey=True)
    for a, v, t in zip(ax, (y, rf.predict(X)), ("True", "Predicted")):
        a.imshow(v[:, None], aspect="auto", cmap="tab20", vmin=0, vmax=max(len(names) - 1, 1),
                 interpolation="nearest", extent=[0, 1, d.DEPTH.iloc[-1], d.DEPTH.iloc[0]])
        a.axhline(d.DEPTH.iloc[cut], color="k", ls="--")
        a.set_title(t, fontsize=9)
        a.set_xticks([])
    show(fig)
    st.caption("Dashed line = train/test split. Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))

# ---------- 3. Missing-log prediction ----------
with tabs[2]:
    tgt = st.selectbox("Log to predict", ["DT", "RHOB", "NPHI", "GR", "RT"])
    feats = [c for c in ["GR", "RHOB", "NPHI", "RT", "DT"] if c != tgt]
    d = w.dropna(subset=feats + [tgt])
    cut = int(len(d) * 0.7)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0).fit(tr[feats], tr[tgt])
    pred = rf.predict(te[feats])
    st.metric("R² on the deeper, unseen section", f"{r2_score(te[tgt], pred):.3f}")
    fig, ax = plt.subplots(figsize=(3.5, 6))
    ax.plot(te[tgt], te.DEPTH, "k", lw=.8, label="Actual")
    ax.plot(pred, te.DEPTH, "r", lw=.8, label="Predicted")
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(tgt)
    show(fig)
    miss = w[tgt].isna() & w[feats].notna().all(axis=1)  # real gaps in your file
    out = w.copy()
    if miss.any():
        out.loc[miss, tgt] = rf.predict(w.loc[miss, feats])
    st.download_button(f"Download logs with {int(miss.sum())} gaps filled (CSV)", out.to_csv(index=False), "filled_logs.csv")

# ---------- 4. Decline curve analysis (Arps) ----------
with tabs[3]:
    up2 = st.file_uploader("Production CSV (optional): a date column + an oil-rate column", type="csv", key="prod")
    if up2:
        pr = pd.read_csv(up2)
        tc = st.selectbox("Date column", pr.columns)
        qc = st.selectbox("Rate column (bbl/d)", pr.columns, index=min(1, len(pr.columns) - 1))
        pr = pr[[tc, qc]].dropna()
        pr[tc] = pd.to_datetime(pr[tc])
        pr = pr[pr[qc] > 0].sort_values(tc)
        t = (pr[tc] - pr[tc].iloc[0]).dt.days.values / 30.44  # months
        qd = pr[qc].values
    else:
        t = np.arange(60.0)
        qd = 1000 / (1 + 0.5 * 0.08 * t) ** 2 * np.exp(np.random.default_rng(2).normal(0, .04, 60))
        st.caption("Demo well: synthetic hyperbolic decline.")
    ql = st.number_input("Economic limit (bbl/d)", value=50.0)


    def arps(t, qi, di, b):
        return qi / (1 + b * di * t) ** (1 / b)


    try:
        (qi, di, b), _ = curve_fit(arps, t, qd, p0=[qd.max(), 0.05, 0.5],
                                   bounds=([0, 1e-4, 0.01], [np.inf, 2, 2]), maxfev=20000)
    except Exception as e:
        st.error(f"Fit failed: {e}")
    else:
        tf = np.arange(0, 600)
        qf = arps(tf, qi, di, b)
        life = qf >= ql
        eur = (qf[life] * 30.44).sum()  # bbl, summed monthly from t=0 until the economic limit
        k = st.columns(4)
        k[0].metric("EUR (MMbbl)", f"{eur / 1e6:.2f}")
        k[1].metric("qi (bbl/d)", f"{qi:.0f}")
        k[2].metric("Di (1/yr)", f"{di * 12:.2f}")
        k[3].metric("b", f"{b:.2f}")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(t, qd, "o", ms=3, label="Data")
        ax.semilogy(tf[life], qf[life], "r", label="Arps fit")
        ax.axhline(ql, color="k", ls="--", lw=.8)
        ax.set_xlabel("Months")
        ax.set_ylabel("bbl/d")
        ax.legend()
        show(fig)

# ---------- 5. Seismic ----------
with tabs[4]:
    sg = st.file_uploader("SEG-Y (optional): first 400 traces are shown", type=["sgy", "segy"])
    if sg:
        import segyio
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "u.sgy")
            with open(path, "wb") as fh:
                fh.write(sg.getvalue())
            with segyio.open(path, ignore_geometry=True) as f:
                data = np.asarray(f.trace.raw[:400])
    else:
        data = demo_seismic()
        st.caption("Demo section: dipping layers + a fault, convolved with a Ricker wavelet.")
    attr = st.radio("Attribute", ["Amplitude", "Envelope", "RMS"], horizontal=True)
    img = {"Amplitude": data, "Envelope": np.abs(hilbert(data, axis=1)),
           "RMS": np.sqrt(uniform_filter1d(data ** 2, 15, axis=1))}[attr]
    v = np.percentile(np.abs(img), 99)
    kw = dict(cmap="seismic", vmin=-v, vmax=v) if attr == "Amplitude" else dict(cmap="magma", vmin=0, vmax=v)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img.T, aspect="auto", **kw)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time (samples)")
    show(fig)
