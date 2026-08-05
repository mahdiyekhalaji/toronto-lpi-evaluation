"""
Toronto LPI evaluation - second round of fixes.

  A. Time-varying exposure (the paper's stated 'most serious unresolved threat').
  B. Mechanism stress tests: stacked DiD + randomization inference on turning outcome.
  C. All-injury pedestrian outcome from police data, 2014-2025.
  D. Approach-level treatment for turning collisions (exploratory).
  E. Dose within four-leg intersections only.
  F. SPF validation with a CURE plot.
  G. Bandwidth sensitivity for spillover and Moran's I.
  H. Overdispersion check.
"""

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats

import spatial as S

LAT_M = 111_320.0
LON_M = 111_320.0 * np.cos(np.radians(43.7))
rng = np.random.default_rng(20260806)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 200})
INK, ACC, WARM, MUTED = "#14211c", "#0b6e4f", "#b4531f", "#98a09c"


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


def show(lab, m, term="treated", n=""):
    pct, lo, hi, pv, *_ = eff(m, term)
    print(f"  {lab:<52}{pct:+7.1f}%  [{lo:+6.1f},{hi:+6.1f}]  p={pv:.3f}  {n}")


# ================================================= A. time-varying exposure

def count_history():
    t = pd.read_csv("tmc_all.csv", low_memory=False)
    t = t.dropna(subset=["px"]).copy()
    t["px"] = t.px.astype(float).astype(int).astype(str).str.zfill(4)
    t["date"] = pd.to_datetime(t.count_date, errors="coerce")
    dur = pd.to_numeric(t.count_duration, errors="coerce").replace(0, np.nan)
    dur = dur.fillna(pd.Series(
        np.where(t.date < pd.Timestamp("2023-09-01"), 8.0, np.nan), index=t.index))
    t["pv"] = pd.to_numeric(t.total_pedestrian, errors="coerce") * 14 / dur
    t["vv"] = pd.to_numeric(t.total_vehicle, errors="coerce") * 14 / dur
    t["year"] = t.date.dt.year
    t = t[(t.pv > 0) & (t.vv > 0)]
    return t.groupby(["px", "year"])[["pv", "vv"]].mean().reset_index()


def volume_trend_test(p, hist):
    print("\nA1. DIFFERENTIAL VOLUME TRENDS AT TREATED SITES")
    meta = p.groupby("px").lpi_year.first()
    h = hist.merge(meta.rename("lpi_year"), left_on="px", right_index=True, how="left")
    h = h[h.year.between(2010, 2025)].copy()
    h["treated"] = (h.lpi_year.notna() & (h.lpi_year <= h.year)).astype(int)
    h["lpv"] = np.log(h.pv)
    h["lvv"] = np.log(h.vv)
    print(f"  {len(h):,} counts at {h.px.nunique():,} signals, 2010-2025")
    for out, lab in [("lpv", "log pedestrian volume"), ("lvv", "log vehicle volume")]:
        m = pf.feols(f"{out} ~ treated | px + year", data=h, vcov={"CRV1": "px"})
        t = m.tidy()
        b, se, pv = (t.loc["treated", "Estimate"], t.loc["treated", "Std. Error"],
                     t.loc["treated", "Pr(>|t|)"])
        print(f"  {lab:<28} change after LPI: {(np.exp(b)-1)*100:+6.1f}%  "
              f"[{(np.exp(b-1.96*se)-1)*100:+6.1f},{(np.exp(b+1.96*se)-1)*100:+6.1f}]"
              f"  p={pv:.3f}")
    return h


def interpolated_exposure(p, hist):
    print("\nA2. MAIN MODEL WITH TIME-VARYING VOLUME CONTROLS")
    years = np.arange(2010, 2026)
    frames = []
    for px, g in hist.groupby("px"):
        g = g.sort_values("year")
        frames.append(pd.DataFrame(dict(
            px=px, year=years,
            lp_t=np.interp(years, g.year, np.log(g.pv)),
            lv_t=np.interp(years, g.year, np.log(g.vv)))))
    ex = pd.concat(frames, ignore_index=True)
    d = p.merge(ex, on=["px", "year"], how="left")
    print(f"  interpolated volumes cover {d.lp_t.notna().mean()*100:.1f}% of site-years")
    m0, _ = fit(d[d.lp_t.notna()], "ped_ksi")
    show("Same sample, no volume controls", m0)
    m1, _ = fit(d[d.lp_t.notna()], "ped_ksi", "treated + lp_t + lv_t")
    show("With interpolated log ped + veh volumes", m1)
    t = m1.tidy()
    for term, lab in [("lp_t", "pedestrian volume elasticity"),
                      ("lv_t", "vehicle volume elasticity")]:
        print(f"       {lab}: {t.loc[term,'Estimate']:+.2f} "
              f"(p={t.loc[term,'Pr(>|t|)']:.3f})")


# ================================================= B. mechanism stress tests

def mech_stacked(p, outcome="ped_turn_row", cohorts=range(2016, 2023), win=3):
    print("\nB1. STACKED DiD ON THE MECHANISM OUTCOME")
    stacks = []
    for c in cohorts:
        treat = p[p.lpi_year == c].px.unique()
        ctrl = p[p.lpi_year.isna() | (p.lpi_year > c + win)].px.unique()
        d = p[p.px.isin(np.r_[treat, ctrl]) & p.year.between(c - win, c + win)].copy()
        d["treated"] = (d.px.isin(treat) & (d.year >= c)).astype(int)
        d["uid"] = d.px + "_" + str(c)
        d["sy"] = f"{c}_" + d.year.astype(str)
        stacks.append(d)
    d = pd.concat(stacks, ignore_index=True)
    m, used = fit(d, outcome, unit="uid", time="sy")
    show("Turning collisions, stacked DiD", m, n=f"{used.px.nunique()} sites")


def mech_ri(p, outcome="ped_turn_row", reps=200):
    print(f"\nB2. RANDOMIZATION INFERENCE ON THE MECHANISM OUTCOME ({reps} perms)")
    obs_m, _ = fit(p, outcome)
    obs = obs_m.tidy().loc["treated", "Estimate"]
    ever = p[p.ever == 1].groupby("px").lpi_year.first()
    null = []
    for _ in range(reps):
        perm = pd.Series(rng.permutation(ever.values), index=ever.index)
        d = p.copy()
        fake = d.px.map(perm)
        d["treated"] = (fake.notna() & (fake <= d.year)).astype(int)
        try:
            m, _ = fit(d, outcome)
            null.append(m.tidy().loc["treated", "Estimate"])
        except Exception:
            continue
    null = np.array(null)
    pv = (np.abs(null) >= abs(obs)).mean()
    print(f"  observed {(np.exp(obs)-1)*100:+.1f}% | null mean "
          f"{(np.exp(null.mean())-1)*100:+.1f}% | randomization p = {pv:.3f}")


# ================================================= C. all-injury outcome

def all_injury_panel(p):
    print("\nC. ALL-INJURY PEDESTRIAN OUTCOME (police data, 2014-2025)")
    k = pd.read_csv("tps_ped.csv", low_memory=False)
    k["year"] = pd.to_numeric(k.OCC_YEAR, errors="coerce")
    k = k[(k.INJURY_COLLISIONS == "YES") & k.year.between(2014, 2025)]
    k = k.rename(columns={"LAT_WGS84": "latitude", "LONG_WGS84": "longitude"})
    k["latitude"] = pd.to_numeric(k.latitude, errors="coerce")
    k["longitude"] = pd.to_numeric(k.longitude, errors="coerce")
    k = k.dropna(subset=["latitude", "longitude"])
    print(f"  pedestrian injury collisions citywide: {len(k):,}")

    sites = p.groupby("px").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
    tree = cKDTree(np.c_[sites.lat * LAT_M, sites.lon * LON_M])
    dist, idx = tree.query(np.c_[k.latitude * LAT_M, k.longitude * LON_M])
    k = k.assign(dist_m=dist, px=sites.px.values[idx])
    k = k[k.dist_m <= 30]
    print(f"  within 30 m of a signal: {len(k):,}")

    cnt = (k.groupby(["px", k.year.astype(int)]).size().rename("ped_inj").reset_index())
    d = p[p.year.between(2014, 2025)].merge(cnt, on=["px", "year"], how="left")
    d["ped_inj"] = d.ped_inj.fillna(0).astype(int)

    m, used = fit(d, "ped_inj")
    show("Pedestrian injury collisions (all severities)", m,
         n=f"{int(used.ped_inj.sum()):,} events, {used.px.nunique():,} sites")

    lo, hi = -5, 5
    e = d.copy()
    terms = []
    for t in range(lo, hi + 1):
        if t == -1:
            continue
        c = f"k_{'m' if t<0 else 'p'}{abs(t)}"
        e[c] = ((e.ever == 1) & (e.evt == t)).astype(int)
        terms.append((t, c))
    e["k_pre_bin"] = ((e.ever == 1) & (e.evt < lo)).astype(int)
    e["k_post_bin"] = ((e.ever == 1) & (e.evt > hi)).astype(int)
    m2, _ = fit(e, "ped_inj",
                " + ".join([c for _, c in terms] + ["k_pre_bin", "k_post_bin"]))
    tid = m2.tidy()
    pre = [c for t, c in terms if t < -1] + ["k_pre_bin"]
    ix = [list(tid.index).index(c) for c in pre]
    V = m2._vcov[np.ix_(ix, ix)]
    b = tid.Estimate.values[ix]
    st = float(b @ np.linalg.solve(V, b))
    print(f"  joint pre-trend test: chi2({len(pre)}) = {st:.2f}, "
          f"p = {1-stats.chi2.cdf(st, len(pre)):.3f}")
    rows = [dict(evt=-1, pct=0.0, lo=0.0, hi=0.0)]
    for t, c in terms:
        bb, se = tid.loc[c, "Estimate"], tid.loc[c, "Std. Error"]
        rows.append(dict(evt=t, pct=(np.exp(bb)-1)*100,
                         lo=(np.exp(bb-1.96*se)-1)*100,
                         hi=(np.exp(bb+1.96*se)-1)*100))
    es = pd.DataFrame(rows).sort_values("evt")
    print(es.round(1).to_string(index=False))
    es.to_csv("event_study_allinjury.csv", index=False)


# ================================================= D. approach-level

LEFT_CROSS = {"North": "EAST", "South": "WEST", "East": "SOUTH", "West": "NORTH"}
RIGHT_CROSS = {"North": "WEST", "South": "EAST", "East": "NORTH", "West": "SOUTH"}
DIRS = ("NORTH", "SOUTH", "EAST", "WEST")


def approach_level(p):
    print("\nD. APPROACH-LEVEL TREATMENT FOR TURNING COLLISIONS (exploratory)")
    k = pd.read_csv("ksi.csv", low_memory=False)
    k["year"] = pd.to_datetime(k.accdate, errors="coerce").dt.year
    ped = k[k.pedestrian.astype(str).str.upper().isin(["YES", "1", "TRUE"])]
    pt = ped.pedtype.astype(str)
    turn = ped[pt.str.contains(
        "turns (left|right) into pedestrian crossing with right-of-way",
        regex=True, na=False)].copy()
    turn["turn_dir"] = np.where(turn.pedtype.str.contains("left"), "L", "R")
    turn["initdir"] = turn.initdir.astype(str).str.strip().str.title()
    turn = turn[turn.initdir.isin(LEFT_CROSS)]
    turn = turn.dropna(subset=["latitude", "longitude", "year"]).drop_duplicates("collision_id")

    s = pd.read_csv("ts.geojson", low_memory=False)
    s["lon"] = s.geometry.str.extract(r"\[(-?\d+\.\d+),")[0].astype(float)
    s["lat"] = s.geometry.str.extract(r",\s*(\d+\.\d+)\]")[0].astype(float)
    s = s.dropna(subset=["lat", "lon"])
    s["px"] = s.PX.astype(str).str.zfill(4)
    for dd in DIRS:
        s[f"y_{dd}"] = pd.to_datetime(
            s[f"LPI_{dd}_IMPLEMENTATION_DATE"], errors="coerce").dt.year
    tree = cKDTree(np.c_[s.lat * LAT_M, s.lon * LON_M])
    dist, idx = tree.query(np.c_[turn.latitude * LAT_M, turn.longitude * LON_M])
    turn = turn.assign(dist_m=dist, px=s.px.values[idx]).query("dist_m <= 30")
    sy = s.drop_duplicates("px").set_index("px")

    print(f"  {len(turn)} turning KSI collisions with usable vehicle direction")
    for mode, name in [("cross", "convention 1: crosswalk being turned across"),
                       ("init", "convention 2: vehicle's own approach")]:
        hit = tot = 0
        for r in turn.itertuples():
            ys = [sy.at[r.px, f"y_{d}"] for d in DIRS]
            ys = [y for y in ys if not pd.isna(y)]
            if not ys or min(ys) > r.year:
                continue                     # intersection not yet treated
            tot += 1
            leg = ((LEFT_CROSS if r.turn_dir == "L" else RIGHT_CROSS)[r.initdir]
                   if mode == "cross" else r.initdir.upper())
            ly = sy.at[r.px, f"y_{leg}"]
            if not pd.isna(ly) and ly <= r.year:
                hit += 1
        print(f"  {name}")
        print(f"    of {tot} turning KSI at treated intersections, the specific leg "
              f"was treated in {hit} ({hit/max(tot,1)*100:.0f}%)")
    print("  (direction-naming convention unverified; descriptive only)")


# ================================================= E. dose within 4-leg

def dose_fourleg(p):
    print("\nE. DOSE-RESPONSE WITHIN FOUR-LEG INTERSECTIONS ONLY")
    d = p[p.n_approaches == 4].copy()
    d["d12"] = d.dose.isin([1, 2]).astype(int)
    d["d34"] = (d.dose >= 3).astype(int)
    m, used = fit(d, "ped_ksi", "d12 + d34")
    show("1-2 approaches treated (4-leg only)", m, "d12")
    show("3-4 approaches treated (4-leg only)", m, "d34")
    print(f"  {used.px.nunique()} four-leg intersections")


# ================================================= F. CURE plot

def cure_plot(p, path="f8_cure.png"):
    print("\nF. SPF VALIDATION: CURE PLOT")
    d = p.dropna(subset=["log_ped", "log_veh"]).copy()
    ref = d[d.treated == 0]
    nb = smf.negativebinomial("ped_ksi ~ log_ped + log_veh + C(year)",
                              data=ref).fit(method="nm", maxiter=4000, disp=0)
    alpha = float(nb.params["alpha"])
    spf = smf.glm("ped_ksi ~ log_ped + log_veh + C(year)", data=ref,
                  family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    ref = ref.assign(mu=spf.predict(ref))
    ref = ref.assign(r=ref.ped_ksi - ref.mu).sort_values("veh_vol")
    cum = ref.r.cumsum().values
    var = (ref.mu + alpha * ref.mu**2).cumsum().values
    lim = 1.96 * np.sqrt(var)
    inside = float(np.mean(np.abs(cum) <= lim)) * 100
    print(f"  cumulative residuals within 95% bounds over {inside:.1f}% of the "
          f"volume range ({'acceptable' if inside >= 95 else 'form suspect'})")

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = ref.veh_vol.values
    ax.fill_between(x, -lim, lim, color=ACC, alpha=0.14, lw=0, label="95% bounds")
    ax.plot(x, cum, color=INK, lw=1.1, label="cumulative residuals")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("14-hour vehicle volume (log scale)")
    ax.set_ylabel("Cumulative residual")
    ax.set_title("CURE plot for the safety performance function", loc="left",
                 fontsize=10, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return inside


# ================================================= G. bandwidths

def bandwidths(p, sites):
    print("\nG. BANDWIDTH SENSITIVITY")
    for r in (250, 500, 1000):
        nt, ntot = S.neighbour_treatment(p, sites, r)
        d = p.assign(nb_share=np.where(ntot > 0, nt / np.maximum(ntot, 1), 0.0))
        m, _ = fit(d, "ped_ksi", "treated + nb_share")
        o = eff(m, "treated"); nb = eff(m, "nb_share")
        print(f"  spillover {r:>4} m: own {o[0]:+6.1f}% (p={o[3]:.3f}) | "
              f"neighbours {nb[0]:+6.1f}% (p={nb[3]:.3f})")
    tot = p.groupby("px").ped_ksi.sum()
    s = sites.set_index("px").loc[tot.index]
    coords = np.c_[s.lat * LAT_M, s.lon * LON_M]
    for r in (500, 1000, 2000):
        I, pv = S.morans_i(tot.values.astype(float), coords, radius=r)
        print(f"  Moran's I {r:>4} m on observed totals: I = {I:+.4f}, p = {pv:.3f}")


# ================================================= H. overdispersion

def overdispersion(p):
    print("\nH. OVERDISPERSION CHECK")
    m, used = fit(p, "ped_ksi")
    mu = m.predict(type="response")
    pearson = np.sum((used.ped_ksi - mu)**2 / mu) / (len(used) - len(m.tidy()))
    print(f"  Pearson dispersion statistic: {pearson:.2f} (1.0 = equidispersed)")
    print("  Cluster-robust standard errors do not rely on the Poisson variance;")
    print("  inference is protected regardless.")


if __name__ == "__main__":
    p = pd.read_csv("panel_full.csv", low_memory=False)
    p["px"] = p.px.astype(str).str.zfill(4)
    sites = (p.groupby("px").agg(lat=("lat", "first"), lon=("lon", "first"),
                                 lpi_year=("lpi_year", "first")).reset_index())
    sites = S.add_geography(p, sites)

    hist = count_history()
    volume_trend_test(p, hist)
    interpolated_exposure(p, hist)
    mech_stacked(p)
    mech_ri(p)
    all_injury_panel(p)
    approach_level(p)
    dose_fourleg(p)
    cure_plot(p)
    bandwidths(p, sites)
    overdispersion(p)
