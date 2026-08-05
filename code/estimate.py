"""
Toronto LPI evaluation - estimation.

Runs the full specification set:
  1  main TWFE Poisson
  2  co-treatment adjustment (red light cameras, cycling infrastructure)
  3  dose-response in treated approaches
  4  mechanism split by collision type
  5  severity split
  6  road user and age subgroups
  7  placebo and falsification outcomes
  8  event study with a joint pre-trend test
  9  stacked difference-in-differences
 10  Empirical Bayes before-after with an SPF (Highway Safety Manual style)
 11  equity: effect and timing by Neighbourhood Improvement Area
 12  randomization inference
 13  snap-radius sensitivity
"""

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf

import build_panel as bp

rng = np.random.default_rng(20260804)
RESULTS = []


def rec(block, label, outcome, m, term="treated", extra=""):
    t = m.tidy()
    b, se, p = t.loc[term, "Estimate"], t.loc[term, "Std. Error"], t.loc[term, "Pr(>|t|)"]
    row = dict(block=block, label=label, outcome=outcome,
               pct=(np.exp(b) - 1) * 100,
               lo=(np.exp(b - 1.96 * se) - 1) * 100,
               hi=(np.exp(b + 1.96 * se) - 1) * 100,
               p=p, note=extra)
    RESULTS.append(row)
    return row


def fit(d, outcome, rhs="treated", unit="px", time="year", cluster="px"):
    keep = d.groupby(unit)[outcome].sum()
    d = d[d[unit].isin(keep[keep > 0].index)].copy()
    return pf.fepois(f"{outcome} ~ {rhs} | {unit} + {time}", data=d,
                     vcov={"CRV1": cluster}), d


def show(r, n=None):
    tag = f"  n={n}" if n else ""
    print(f"  {r['label']:<44}{r['pct']:+7.1f}%  "
          f"[{r['lo']:+7.1f},{r['hi']:+7.1f}]  p={r['p']:.3f}{tag}")


# ============================================================ 1-2 main + controls

def main_and_controls(p):
    print("\n1. MAIN ESTIMATE AND CO-TREATMENT ADJUSTMENT")
    m, d = fit(p, "ped_ksi")
    show(rec("main", "Pedestrian KSI, unadjusted", "ped_ksi", m),
         f"{d.px.nunique()} sites")

    m, d = fit(p, "ped_ksi", "treated + rlc")
    show(rec("main", "+ red light camera control", "ped_ksi", m))
    print(f"       (red light camera coefficient: "
          f"{(np.exp(m.tidy().loc['rlc','Estimate'])-1)*100:+.1f}%, "
          f"p={m.tidy().loc['rlc','Pr(>|t|)']:.3f})")

    m, d = fit(p, "ped_ksi", "treated + bike")
    show(rec("main", "+ cycling infrastructure control", "ped_ksi", m))

    m, d = fit(p, "ped_ksi", "treated + rlc + bike")
    show(rec("main", "+ both co-treatments", "ped_ksi", m))

    # how much overlap is there at all?
    ov = p.groupby("px")[["ever", "rlc", "bike"]].max()
    both = ov[(ov.ever == 1)]
    print(f"       treated sites also receiving RLC: {both.rlc.sum()} "
          f"({both.rlc.mean()*100:.1f}%), cycling infra: {both.bike.sum()} "
          f"({both.bike.mean()*100:.1f}%)")


# ============================================================ 3 dose

def dose(p):
    print("\n2. DOSE-RESPONSE")
    d = p.copy()
    d["d1"] = (d.dose == 1).astype(int)
    d["d2"] = (d.dose == 2).astype(int)
    d["d34"] = (d.dose >= 3).astype(int)
    m, _ = fit(d, "ped_ksi", "d1 + d2 + d34")
    for term, lab in [("d1", "1 approach treated"),
                      ("d2", "2 approaches treated"),
                      ("d34", "3-4 approaches treated")]:
        show(rec("dose", lab, "ped_ksi", m, term=term))
    m, _ = fit(d, "ped_ksi", "dose")
    r = rec("dose", "per additional treated approach", "ped_ksi", m, term="dose")
    show(r)
    n = p[p.treated == 1].groupby("dose").size()
    print("       treated site-years by dose:",
          ", ".join(f"{k}:{v}" for k, v in n.items()))


# ============================================================ 4-7 mechanism etc

def mechanism(p):
    print("\n3. MECHANISM, SEVERITY, SUBGROUPS, FALSIFICATION")
    spec = [
        ("mechanism", "Turning driver hits ped with ROW", "ped_turn_row"),
        ("mechanism", "Straight driver hits ped with ROW", "ped_straight_row"),
        ("falsification", "Ped crossing WITHOUT right-of-way", "ped_no_row"),
        ("falsification", "Ped on sidewalk or midblock", "ped_offpath"),
        ("severity", "Fatal pedestrian collisions", "ped_fatal"),
        ("severity", "Non-fatal pedestrian KSI", "ped_nonfatal"),
        ("subgroup", "Pedestrian, age 65+", "ped_older"),
        ("subgroup", "Pedestrian, school child", "ped_child"),
        ("subgroup", "Cyclist KSI", "cyc_ksi"),
        ("placebo", "Vehicle-occupant KSI", "veh_only"),
    ]
    for block, lab, out in spec:
        m, d = fit(p, out)
        show(rec(block, lab, out, m), f"{int(d[out].sum())} events")


# ============================================================ 8 event study

def event_study(p, outcome="ped_ksi", lo=-6, hi=6):
    print("\n4. EVENT STUDY")
    d = p[(p.ever == 0) | (p.evt.between(lo, hi))].copy()
    terms = []
    for t in range(lo, hi + 1):
        if t == -1:
            continue
        col = f"k_{'m' if t < 0 else 'p'}{abs(t)}"
        d[col] = ((d.ever == 1) & (d.evt == t)).astype(int)
        terms.append((t, col))
    m, _ = fit(d, outcome, rhs=" + ".join(c for _, c in terms))
    tid = m.tidy()

    rows = [dict(evt=-1, pct=0.0, lo=0.0, hi=0.0, p=np.nan)]
    for t, col in terms:
        b, se = tid.loc[col, "Estimate"], tid.loc[col, "Std. Error"]
        rows.append(dict(evt=t, pct=(np.exp(b) - 1) * 100,
                         lo=(np.exp(b - 1.96 * se) - 1) * 100,
                         hi=(np.exp(b + 1.96 * se) - 1) * 100,
                         p=tid.loc[col, "Pr(>|t|)"]))
    es = pd.DataFrame(rows).sort_values("evt").reset_index(drop=True)

    # joint test that all pre-treatment leads are zero
    pre = [c for t, c in terms if t < -1]
    try:
        w = m.wald_test(R=np.eye(len(m.tidy()))[
            [list(m.tidy().index).index(c) for c in pre]])
        print(f"  joint pre-trend test on {len(pre)} leads: p = {w['pvalue']:.3f}")
    except Exception:
        # fall back on a chi-square built from the coefficient subvector
        idx = [list(tid.index).index(c) for c in pre]
        V = m._vcov[np.ix_(idx, idx)]
        b = tid.Estimate.values[idx]
        stat = float(b @ np.linalg.pinv(V) @ b)
        from scipy import stats
        print(f"  joint pre-trend test on {len(pre)} leads: "
              f"chi2({len(pre)}) = {stat:.2f}, p = {1-stats.chi2.cdf(stat,len(pre)):.3f}")
    print(es.round(1).to_string(index=False))
    return es


# ============================================================ 9 stacked

def stacked(p, outcome="ped_ksi", cohorts=range(2016, 2023), win=3):
    print("\n5. STACKED DIFFERENCE-IN-DIFFERENCES")
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
    show(rec("robust", "Stacked DiD, not-yet-treated controls", outcome, m),
         f"{used.px.nunique()} sites")


# ============================================================ 10 empirical Bayes

def empirical_bayes(p):
    """Highway Safety Manual style before-after with an SPF and EB correction."""
    print("\n6. EMPIRICAL BAYES BEFORE-AFTER (HSM method)")
    d = p.dropna(subset=["log_ped", "log_veh"]).copy()

    # SPF estimated on untreated site-years only
    ref = d[d.treated == 0]
    # estimate the overdispersion parameter rather than assuming it
    nb = smf.negativebinomial("ped_ksi ~ log_ped + log_veh + C(year)",
                              data=ref).fit(method="nm", maxiter=4000, disp=0)
    alpha = float(nb.params["alpha"])
    k = 1.0 / alpha          # HSM overdispersion parameter, Var = mu + mu^2/k
    spf = smf.glm("ped_ksi ~ log_ped + log_veh + C(year)", data=ref,
                  family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    d["mu"] = spf.predict(d)

    print(f"  SPF on {len(ref):,} untreated site-years: "
          f"ped volume elasticity {spf.params['log_ped']:.2f}, "
          f"vehicle volume elasticity {spf.params['log_veh']:.2f}, "
          f"alpha = {alpha:.2f} (k = {k:.2f})")

    t = d[(d.ever == 1) & d.lpi_year.between(bp.START + 3, bp.END - 2)]
    out = []
    for px, g in t.groupby("px"):
        b, a = g[g.treated == 0], g[g.treated == 1]
        if len(b) < 3 or len(a) < 2:
            continue
        P_b, P_a = b.mu.sum(), a.mu.sum()
        O_b, O_a = b.ped_ksi.sum(), a.ped_ksi.sum()
        if P_b <= 0:
            continue
        w = 1.0 / (1.0 + P_b / k)
        Eb = w * P_b + (1 - w) * O_b
        Ea = Eb * (P_a / P_b)
        Va = Ea * (P_a / P_b) * (1 - w)
        out.append((O_a, Ea, Va))
    O = sum(o for o, _, _ in out)
    E = sum(e for _, e, _ in out)
    V = sum(v for _, _, v in out)
    theta = (O / E) / (1 + V / E**2)
    var_theta = theta**2 * ((1 / O) + (V / E**2)) / (1 + V / E**2)**2
    se = np.sqrt(var_theta)
    pct = (theta - 1) * 100
    lo, hi = (theta - 1.96 * se - 1) * 100, (theta + 1.96 * se - 1) * 100
    print(f"  {len(out)} treated sites with exposure and a usable before period")
    print(f"  observed after {O:.0f}, expected after {E:.1f}")
    RESULTS.append(dict(block="robust", label="Empirical Bayes before-after",
                        outcome="ped_ksi", pct=pct, lo=lo, hi=hi,
                        p=np.nan, note=f"{len(out)} sites"))
    print(f"  {'Empirical Bayes before-after':<44}{pct:+7.1f}%  "
          f"[{lo:+7.1f},{hi:+7.1f}]  CMF = {theta:.3f}")


# ============================================================ 11 equity

def equity(p):
    print("\n7. EQUITY")
    d = p.copy()
    d["treated_nia"] = d.treated * d.nia
    m, _ = fit(d, "ped_ksi", "treated + treated_nia")
    show(rec("equity", "Effect outside NIAs", "ped_ksi", m))
    show(rec("equity", "Extra effect inside NIAs", "ped_ksi", m, term="treated_nia"))

    # did equity-deserving neighbourhoods get LPIs later?
    s = d.groupby("px").agg(nia=("nia", "max"), lpi=("lpi_year", "first"),
                            ped=("ped_ksi", "sum"))
    a = s[(s.nia == 1) & s.lpi.notna()].lpi
    b = s[(s.nia == 0) & s.lpi.notna()].lpi
    from scipy import stats
    u = stats.mannwhitneyu(a, b)
    print(f"  median LPI year: NIA {a.median():.0f} (n={len(a)}), "
          f"non-NIA {b.median():.0f} (n={len(b)}), Mann-Whitney p={u.pvalue:.3f}")
    cov_a = s[s.nia == 1].lpi.notna().mean() * 100
    cov_b = s[s.nia == 0].lpi.notna().mean() * 100
    print(f"  share of signals with an LPI: NIA {cov_a:.1f}%, non-NIA {cov_b:.1f}%")


# ============================================================ 12 inference

def randomization_inference(p, outcome="ped_ksi", reps=200):
    print(f"\n8. RANDOMIZATION INFERENCE ({reps} permutations)")
    obs_m, _ = fit(p, outcome)
    obs = obs_m.tidy().loc["treated", "Estimate"]

    ever = p[p.ever == 1].groupby("px").lpi_year.first()
    sites = ever.index.values
    null = []
    for _ in range(reps):
        perm = pd.Series(rng.permutation(ever.values), index=sites)
        d = p.copy()
        fake = d.px.map(perm)
        d["treated"] = (fake.notna() & (fake <= d.year)).astype(int)
        try:
            m, _ = fit(d, outcome)
            null.append(m.tidy().loc["treated", "Estimate"])
        except Exception:
            continue
    null = np.array(null)
    pval = (np.abs(null) >= abs(obs)).mean()
    print(f"  observed {(np.exp(obs)-1)*100:+.1f}% | "
          f"null mean {(np.exp(null.mean())-1)*100:+.1f}%, "
          f"sd {null.std():.3f} | randomization p = {pval:.3f}")
    return null, obs


# ============================================================ 13 radius

def radius_sensitivity():
    print("\n9. SNAP RADIUS SENSITIVITY")
    for r in (20, 30, 50, 100):
        q, _, _ = bp.build(radius=r)
        m, d = fit(q, "ped_ksi")
        row = rec("robust", f"Snap radius {r} m", "ped_ksi", m)
        show(row, f"{int(d.ped_ksi.sum())} events")


if __name__ == "__main__":
    p = pd.read_csv("panel_full.csv", low_memory=False)
    p["px"] = p.px.astype(str).str.zfill(4)
    print(f"panel: {p.px.nunique():,} intersections, {len(p):,} site-years, "
          f"{p.ped_ksi.sum():,} pedestrian KSI")

    main_and_controls(p)
    dose(p)
    mechanism(p)
    es = event_study(p)
    stacked(p)
    empirical_bayes(p)
    equity(p)
    null, obs = randomization_inference(p)
    radius_sensitivity()

    pd.DataFrame(RESULTS).to_csv("results_full.csv", index=False)
    es.to_csv("event_study_full.csv", index=False)
    np.save("ri_null.npy", null)
    print("\nwrote results_full.csv, event_study_full.csv, ri_null.npy")
