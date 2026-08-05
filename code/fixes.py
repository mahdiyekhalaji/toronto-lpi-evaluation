"""
Toronto LPI evaluation - methodological fixes.

Addresses six problems identified in review:

  1. Multiple testing. A primary family of hypotheses is declared and
     Holm-Bonferroni family-wise adjusted p-values are reported. Everything
     outside the family is labelled exploratory.

  2. Event study endpoints. Site-years outside the window were dropped, which
     is not innocuous. They are now binned into endpoint indicators.

  3. Mechanism comparison on a common sample. Separate models ran on different
     sets of intersections. Outcomes are now stacked and the difference between
     mechanism and falsification effects is tested directly.

  4. Spatial autocorrelation. Moran's I on estimated fixed effects is attenuated
     by estimation noise, so a null result there proves nothing. Moran's I is
     now computed on observed collision totals, which are not estimated, and
     inference is protected with spatially clustered errors at four bandwidths.

  5. Spillover power. The null result is quantified with a minimum detectable
     effect rather than asserted.

  6. Empirical Bayes interval. The closed-form variance is checked against a
     site-level bootstrap.

Also: sensitivity to dropping 2025, whose collision-configuration coding is
only 87.5 percent complete.
"""

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from scipy import stats

import spatial as S

LAT_M = 111_320.0
LON_M = 111_320.0 * np.cos(np.radians(43.7))
rng = np.random.default_rng(20260805)
OUT = []


def fit(d, outcome, rhs="treated", unit="px", time="year", cluster="px"):
    keep = d.groupby(unit)[outcome].sum()
    d = d[d[unit].isin(keep[keep > 0].index)].copy()
    return pf.fepois(f"{outcome} ~ {rhs} | {unit} + {time}", data=d,
                     vcov={"CRV1": cluster}), d


def eff(m, term="treated"):
    t = m.tidy()
    b, se = t.loc[term, "Estimate"], t.loc[term, "Std. Error"]
    return ((np.exp(b) - 1) * 100, (np.exp(b - 1.96 * se) - 1) * 100,
            (np.exp(b + 1.96 * se) - 1) * 100, t.loc[term, "Pr(>|t|)"], b, se)


# =================================================== 1. multiple testing

PRIMARY = [
    ("ped_ksi",      "Pedestrian KSI (main effect)"),
    ("ped_turn_row", "Turning driver, ped has right-of-way (mechanism)"),
    ("ped_no_row",   "Ped without right-of-way (falsification)"),
    ("veh_only",     "Vehicle occupants only (placebo)"),
    ("ped_older",    "Pedestrian aged 65+ (pre-specified subgroup)"),
    ("cyc_ksi",      "Cyclist KSI (pre-specified subgroup)"),
]


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def multiple_testing(p):
    print("\n1. MULTIPLE TESTING  (Holm-Bonferroni over a declared primary family "
          f"of {len(PRIMARY)})")
    rows = []
    for out, lab in PRIMARY:
        m, d = fit(p, out)
        pct, lo, hi, pv, b, se = eff(m)
        rows.append(dict(outcome=out, label=lab, pct=pct, lo=lo, hi=hi, p_raw=pv,
                         events=int(d[out].sum())))
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p_raw.values)
    for x in r.itertuples():
        star = "  significant" if x.p_holm < 0.05 else ""
        print(f"  {x.label:<50}{x.pct:+7.1f}%  raw p={x.p_raw:.4f}  "
              f"Holm p={x.p_holm:.4f}{star}")
    OUT.append(("primary_family", r))
    print("  All other specifications in the study are exploratory and are")
    print("  reported with unadjusted p-values.")
    return r


# =================================================== 2. event study binning

def event_study_binned(p, outcome="ped_ksi", lo=-6, hi=6):
    print(f"\n2. EVENT STUDY WITH BINNED ENDPOINTS  (window {lo} to +{hi})")
    d = p.copy()
    terms = []
    # interior periods
    for t in range(lo, hi + 1):
        if t == -1:
            continue
        col = f"k_{'m' if t < 0 else 'p'}{abs(t)}"
        d[col] = ((d.ever == 1) & (d.evt == t)).astype(int)
        terms.append((t, col))
    # endpoint bins: everything beyond the window, kept rather than dropped
    d["k_pre_bin"] = ((d.ever == 1) & (d.evt < lo)).astype(int)
    d["k_post_bin"] = ((d.ever == 1) & (d.evt > hi)).astype(int)
    rhs = " + ".join([c for _, c in terms] + ["k_pre_bin", "k_post_bin"])

    m, used = fit(d, outcome, rhs=rhs)
    tid = m.tidy()
    rows = [dict(evt=-1, pct=0.0, lo=0.0, hi=0.0, p=np.nan)]
    for t, col in terms:
        b, se = tid.loc[col, "Estimate"], tid.loc[col, "Std. Error"]
        rows.append(dict(evt=t, pct=(np.exp(b) - 1) * 100,
                         lo=(np.exp(b - 1.96 * se) - 1) * 100,
                         hi=(np.exp(b + 1.96 * se) - 1) * 100,
                         p=tid.loc[col, "Pr(>|t|)"]))
    es = pd.DataFrame(rows).sort_values("evt").reset_index(drop=True)

    n_dropped_before = ((p.ever == 1) & (~p.evt.between(lo, hi))).sum()
    print(f"  site-years previously dropped, now binned: {n_dropped_before:,}")

    pre = [c for t, c in terms if t < -1] + ["k_pre_bin"]
    idx = [list(tid.index).index(c) for c in pre]
    V = m._vcov[np.ix_(idx, idx)]
    b = tid.Estimate.values[idx]
    stat = float(b @ np.linalg.solve(V, b))
    pv = 1 - stats.chi2.cdf(stat, len(pre))
    print(f"  joint pre-trend test on {len(pre)} leads incl. bin: "
          f"chi2({len(pre)}) = {stat:.2f}, p = {pv:.3f}")
    print(es.round(1).to_string(index=False))
    OUT.append(("event_study_binned", es))
    return es, pv


# =================================================== 3. common-sample mechanism

def mechanism_common_sample(p):
    print("\n3. MECHANISM vs FALSIFICATION ON A COMMON SAMPLE")
    a = p[["px", "year", "treated", "ped_turn_row"]].rename(
        columns={"ped_turn_row": "y"}).assign(kind="mech")
    b = p[["px", "year", "treated", "ped_no_row"]].rename(
        columns={"ped_no_row": "y"}).assign(kind="fals")
    d = pd.concat([a, b], ignore_index=True)
    d["unit"] = d.px + "_" + d.kind          # site-by-outcome fixed effect
    d["ty"] = d.kind + "_" + d.year.astype(str)   # outcome-specific year effects
    d["mech"] = (d.kind == "mech").astype(int)
    d["t_mech"] = d.treated * d.mech

    keep = d.groupby("unit").y.sum()
    d = d[d.unit.isin(keep[keep > 0].index)].copy()
    m = pf.fepois("y ~ treated + t_mech | unit + ty", data=d,
                  vcov={"CRV1": "px"})
    t = m.tidy()
    for term, lab in [("treated", "Falsification outcome (ped without ROW)"),
                      ("t_mech", "Mechanism minus falsification (difference)")]:
        bb, se = t.loc[term, "Estimate"], t.loc[term, "Std. Error"]
        print(f"  {lab:<50}{(np.exp(bb)-1)*100:+7.1f}%  "
              f"[{(np.exp(bb-1.96*se)-1)*100:+6.1f},{(np.exp(bb+1.96*se)-1)*100:+6.1f}]  "
              f"p={t.loc[term,'Pr(>|t|)']:.4f}")
    print(f"  common sample: {d.px.nunique()} intersections, "
          f"{int(d.y.sum())} collisions across both outcome types")
    OUT.append(("mechanism_common", t.reset_index()))
    return m


# =================================================== 4. spatial autocorrelation

def morans_observed(p, sites, radius=1000):
    """Moran's I on OBSERVED collision totals per site. No estimation noise,
    so no attenuation. This tests whether risk itself clusters spatially."""
    print("\n4. SPATIAL AUTOCORRELATION, MEASURED WITHOUT ATTENUATION")
    tot = p.groupby("px").ped_ksi.sum()
    s = sites.set_index("px").loc[tot.index]
    coords = np.c_[s.lat * LAT_M, s.lon * LON_M]
    I, pv = S.morans_i(tot.values.astype(float), coords, radius=radius)
    print(f"  Moran's I on observed pedestrian KSI totals ({radius} m): "
          f"I = {I:+.4f}, expected {-1/(len(tot)-1):+.4f}, permutation p = {pv:.3f}")
    if pv < 0.05:
        print("  -> collision risk IS spatially clustered. The earlier null on")
        print("     estimated fixed effects was attenuation, not absence.")
    OUT.append(("morans_observed", pd.DataFrame([dict(I=I, p=pv, radius=radius)])))
    return I, pv


def spatial_se_bandwidths(p, sites):
    print("\n   Spatially clustered standard errors at four bandwidths")
    s = sites.set_index("px")
    rows = []
    for km in (0.5, 1.0, 2.0, 5.0):
        d = p.copy()
        d["gx"] = (d.px.map(s.lon) * LON_M / (km * 1000)).round().astype("Int64")
        d["gy"] = (d.px.map(s.lat) * LAT_M / (km * 1000)).round().astype("Int64")
        d["cell"] = d.gx.astype(str) + "_" + d.gy.astype(str)
        keep = d.groupby("px").ped_ksi.sum()
        d = d[d.px.isin(keep[keep > 0].index)]
        m = pf.fepois("ped_ksi ~ treated | px + year", data=d, vcov={"CRV1": "cell"})
        pct, lo, hi, pv, b, se = eff(m)
        rows.append(dict(km=km, cells=d.cell.nunique(), pct=pct, lo=lo, hi=hi, p=pv))
        print(f"   {km:>4.1f} km grid ({d.cell.nunique():>4} cells): "
              f"{pct:+6.1f}%  [{lo:+6.1f},{hi:+6.1f}]  p={pv:.3f}")
    OUT.append(("spatial_se", pd.DataFrame(rows)))


# =================================================== 5. spillover power

def spillover_power(p, sites, radius=500):
    print(f"\n5. SPILLOVER: MINIMUM DETECTABLE EFFECT ({radius} m)")
    nt, ntot = S.neighbour_treatment(p, sites, radius)
    d = p.assign(nb_treated=nt, nb_total=ntot)
    d["nb_share"] = np.where(d.nb_total > 0, d.nb_treated / d.nb_total, 0.0)

    for label, sub in [("all sites, controlling own treatment", d),
                       ("never-treated sites only", d[d.ever == 0])]:
        rhs = "treated + nb_share" if "all" in label else "nb_share"
        m, used = fit(sub, "ped_ksi", rhs)
        pct, lo, hi, pv, b, se = eff(m, "nb_share")
        # 80% power, two-sided 5%: |effect| detectable at 2.80 * SE
        mde = (np.exp(2.80 * se) - 1) * 100
        print(f"  {label:<40}{pct:+7.1f}%  p={pv:.3f}")
        print(f"       SE {se:.3f} -> minimum detectable spillover "
              f"±{mde:.0f}% at 80% power")
        OUT.append(("spillover_power",
                    pd.DataFrame([dict(sample=label, pct=pct, se=se, mde=mde, p=pv)])))


# =================================================== 6. EB bootstrap

def eb_bootstrap(p, reps=2000):
    print(f"\n6. EMPIRICAL BAYES INTERVAL, CLOSED FORM vs BOOTSTRAP ({reps} reps)")
    d = p.dropna(subset=["log_ped", "log_veh"]).copy()
    ref = d[d.treated == 0]
    nb = smf.negativebinomial("ped_ksi ~ log_ped + log_veh + C(year)",
                              data=ref).fit(method="nm", maxiter=4000, disp=0)
    alpha = float(nb.params["alpha"]); k = 1.0 / alpha
    spf = smf.glm("ped_ksi ~ log_ped + log_veh + C(year)", data=ref,
                  family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    d["mu"] = spf.predict(d)

    t = d[(d.ever == 1) & d.lpi_year.between(2013, 2023)]
    sites = []
    for px, g in t.groupby("px"):
        b, a = g[g.treated == 0], g[g.treated == 1]
        if len(b) < 3 or len(a) < 2:
            continue
        P_b, P_a = b.mu.sum(), a.mu.sum()
        if P_b <= 0:
            continue
        O_b, O_a = b.ped_ksi.sum(), a.ped_ksi.sum()
        w = 1.0 / (1.0 + P_b / k)
        Eb = w * P_b + (1 - w) * O_b
        Ea = Eb * (P_a / P_b)
        Va = Ea * (P_a / P_b) * (1 - w)
        sites.append((O_a, Ea, Va))
    arr = np.array(sites)

    def theta_of(a):
        O, E, V = a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum()
        return (O / E) / (1 + V / E**2)

    th = theta_of(arr)
    O, E, V = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()
    se_cf = np.sqrt(th**2 * ((1 / O) + (V / E**2)) / (1 + V / E**2)**2)
    cf = ((th - 1) * 100, (th - 1.96 * se_cf - 1) * 100, (th + 1.96 * se_cf - 1) * 100)

    n = len(arr)
    boot = np.array([theta_of(arr[rng.integers(0, n, n)]) for _ in range(reps)])
    bl, bh = np.percentile(boot, [2.5, 97.5])

    print(f"  {n} treated sites | observed after {O:.0f}, expected {E:.1f}")
    print(f"  closed form  CMF {th:.3f}  effect {cf[0]:+.1f}%  "
          f"[{cf[1]:+.1f},{cf[2]:+.1f}]")
    print(f"  bootstrap    CMF {boot.mean():.3f}  effect {(th-1)*100:+.1f}%  "
          f"[{(bl-1)*100:+.1f},{(bh-1)*100:+.1f}]")
    agree = abs((bl - 1) * 100 - cf[1]) < 6 and abs((bh - 1) * 100 - cf[2]) < 6
    print("  -> intervals agree; closed-form variance is sound" if agree
          else "  -> intervals disagree; report the bootstrap interval")
    OUT.append(("eb_check", pd.DataFrame([dict(
        cmf=th, cf_lo=cf[1], cf_hi=cf[2], boot_lo=(bl-1)*100, boot_hi=(bh-1)*100)])))
    return th, cf, ((bl - 1) * 100, (bh - 1) * 100)


# =================================================== 7. drop 2025

def drop_last_year(p):
    print("\n7. SENSITIVITY: DROP 2025 (configuration coding 87.5% complete)")
    for lab, d in [("full panel 2010-2025", p), ("2010-2024 only", p[p.year <= 2024])]:
        for out in ("ped_ksi", "ped_turn_row"):
            m, used = fit(d, out)
            pct, lo, hi, pv, b, se = eff(m)
            print(f"  {lab:<24}{out:<16}{pct:+7.1f}%  "
                  f"[{lo:+6.1f},{hi:+6.1f}]  p={pv:.3f}")


if __name__ == "__main__":
    p = pd.read_csv("panel_full.csv", low_memory=False)
    p["px"] = p.px.astype(str).str.zfill(4)
    sites = (p.groupby("px").agg(lat=("lat", "first"), lon=("lon", "first"),
                                 lpi_year=("lpi_year", "first")).reset_index())
    sites = S.add_geography(p, sites)

    fam = multiple_testing(p)
    es, pre_p = event_study_binned(p)
    mechanism_common_sample(p)
    morans_observed(p, sites)
    spatial_se_bandwidths(p, sites)
    spillover_power(p, sites)
    eb_bootstrap(p)
    drop_last_year(p)

    with pd.ExcelWriter("fixes_results.xlsx") as w:
        for name, df in OUT:
            df.to_excel(w, sheet_name=name[:31], index=False)
    fam.to_csv("primary_family.csv", index=False)
    es.to_csv("event_study_binned.csv", index=False)
    print("\nwrote fixes_results.xlsx, primary_family.csv, event_study_binned.csv")
