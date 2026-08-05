"""
Toronto LPI evaluation - spatial analysis.

Three things the map should be doing rather than decorating:

  1. Spillover. Difference-in-differences assumes an untreated site is unaffected
     by treatment elsewhere. If drivers or pedestrians adapt across a corridor,
     untreated neighbours of treated sites are contaminated controls and the
     main estimate is biased toward zero.

  2. Spatial autocorrelation. Clustering by intersection allows correlation within
     a site over time but assumes independence between sites. Nearby intersections
     share traffic, drivers and enforcement. Moran's I on the residuals tests it;
     Conley standard errors relax it.

  3. Geographic heterogeneity. Downtown intersections differ from suburban arterials
     in speed, volume and pedestrian mix. The effect need not be the same.
"""

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.spatial import cKDTree
from scipy import stats

LAT_M = 111_320.0
LON_M = 111_320.0 * np.cos(np.radians(43.7))
CITY_HALL = (43.6534, -79.3839)

RESULTS = []


def xy(df):
    return np.c_[df.lat * LAT_M, df.lon * LON_M]


def fit(d, outcome, rhs="treated", unit="px", time="year"):
    keep = d.groupby(unit)[outcome].sum()
    d = d[d[unit].isin(keep[keep > 0].index)].copy()
    return pf.fepois(f"{outcome} ~ {rhs} | {unit} + {time}", data=d,
                     vcov={"CRV1": "px"}), d


def eff(m, term):
    t = m.tidy()
    b, se = t.loc[term, "Estimate"], t.loc[term, "Std. Error"]
    return ((np.exp(b) - 1) * 100, (np.exp(b - 1.96 * se) - 1) * 100,
            (np.exp(b + 1.96 * se) - 1) * 100, t.loc[term, "Pr(>|t|)"])


def rec(label, m, term="treated"):
    pct, lo, hi, p = eff(m, term)
    RESULTS.append(dict(label=label, pct=pct, lo=lo, hi=hi, p=p))
    print(f"  {label:<46}{pct:+7.1f}%  [{lo:+7.1f},{hi:+7.1f}]  p={p:.3f}")


# ---------------------------------------------------------------- geography

def add_geography(p, sites):
    """Distance to City Hall, and neighbour counts at several radii."""
    d = np.hypot((sites.lat - CITY_HALL[0]) * LAT_M,
                 (sites.lon - CITY_HALL[1]) * LON_M) / 1000.0
    sites = sites.assign(km_core=d)

    tree = cKDTree(xy(sites))
    for r in (250, 500, 1000):
        neigh = tree.query_ball_point(xy(sites), r)
        sites[f"nb{r}"] = [[j for j in n if j != i] for i, n in enumerate(neigh)]
    return sites


def neighbour_treatment(p, sites, radius=500):
    """For each site-year, how many neighbours within `radius` are already treated."""
    idx = {px: i for i, px in enumerate(sites.px)}
    lpi = sites.lpi_year.values
    nb = sites[f"nb{radius}"].values

    n_treated, n_total = [], []
    for px, yr in zip(p.px, p.year):
        i = idx.get(px)
        if i is None:
            n_treated.append(np.nan); n_total.append(np.nan); continue
        js = nb[i]
        if not js:
            n_treated.append(0); n_total.append(0); continue
        y = lpi[js]
        n_treated.append(int(np.nansum(y <= yr)))
        n_total.append(len(js))
    return np.array(n_treated, dtype=float), np.array(n_total, dtype=float)


# ---------------------------------------------------------------- 1 spillover

def spillover(p, sites, radius=500):
    print(f"\n1. SPILLOVER  (neighbours within {radius} m)")
    nt, ntot = neighbour_treatment(p, sites, radius)
    d = p.copy()
    d["nb_treated"] = nt
    d["nb_total"] = ntot
    d["nb_share"] = np.where(d.nb_total > 0, d.nb_treated / d.nb_total, 0.0)
    d["nb_any"] = (d.nb_treated > 0).astype(int)

    print(f"  median neighbours within {radius} m: {np.nanmedian(ntot):.0f}")

    # (a) main model with neighbour exposure added
    m, _ = fit(d, "ped_ksi", "treated + nb_share")
    rec("Own treatment, controlling for neighbours", m, "treated")
    rec("Share of neighbours treated", m, "nb_share")

    # (b) the clean test: never-treated sites only, does neighbour treatment matter?
    nev = d[d.ever == 0].copy()
    if nev.ped_ksi.sum() > 30:
        m2, used = fit(nev, "ped_ksi", "nb_share")
        print(f"  never-treated sites only "
              f"({used.px.nunique()} sites, {int(used.ped_ksi.sum())} collisions):")
        rec("  neighbour treatment at untreated sites", m2, "nb_share")

    # (c) does the mechanism outcome spill over?
    m3, _ = fit(d, "ped_turn_row", "treated + nb_share")
    rec("Turning collisions, own treatment", m3, "treated")
    rec("Turning collisions, neighbour share", m3, "nb_share")
    return d


# ---------------------------------------------------------------- 2 Moran's I

def morans_i(resid, coords, radius=1000):
    """Moran's I on site-level mean residuals, inverse-distance weights."""
    tree = cKDTree(coords)
    pairs = tree.query_pairs(radius)
    if not pairs:
        return np.nan, np.nan
    n = len(resid)
    z = resid - resid.mean()
    num = 0.0
    W = 0.0
    for i, j in pairs:
        dij = np.hypot(*(coords[i] - coords[j]))
        w = 1.0 / max(dij, 50.0)
        num += 2 * w * z[i] * z[j]
        W += 2 * w
    I = (n / W) * (num / np.sum(z**2))
    EI = -1.0 / (n - 1)
    # permutation reference distribution
    null = []
    rng = np.random.default_rng(7)
    for _ in range(199):
        zp = rng.permutation(z)
        num_p = sum(2 * (1.0 / max(np.hypot(*(coords[i] - coords[j])), 50.0))
                    * zp[i] * zp[j] for i, j in pairs)
        null.append((n / W) * (num_p / np.sum(zp**2)))
    null = np.array(null)
    p = (np.abs(null - EI) >= abs(I - EI)).mean()
    return I, p


def autocorrelation(p, sites):
    """Test whether unobserved site risk is spatially clustered.

    Note: mean residuals per site are zero by construction once intersection
    fixed effects are included, so they carry no information. The quantity that
    does is the estimated fixed effect itself, which is exactly the unobserved
    site-level risk the model absorbs. If those are spatially correlated, then
    nearby intersections are not independent and clustering on the intersection
    alone may understate uncertainty.
    """
    print("\n2. SPATIAL AUTOCORRELATION IN SITE-LEVEL RISK")
    m, used = fit(p, "ped_ksi")
    fe = m.fixef()
    key = [k for k in fe if k.startswith("C(px)") or k == "px"][0]
    site = pd.Series(fe[key])
    site.index = [str(i).zfill(4) for i in site.index]
    site = site[site.index.isin(sites.px)]

    s = sites.set_index("px").loc[site.index]
    coords = np.c_[s.lat * LAT_M, s.lon * LON_M]
    I, pv = morans_i(site.values, coords, radius=1000)
    print(f"  Moran's I on estimated site effects (1 km): I = {I:+.4f}, "
          f"expected {-1/(len(site)-1):+.4f}, permutation p = {pv:.3f}")
    if pv < 0.05:
        print("  -> nearby intersections share unobserved risk; the grid-clustered "
              "standard errors below are the relevant check")
    else:
        print("  -> no detectable spatial clustering in site risk")
    return I, pv


def conley_check(p, sites, cut_km=1.0):
    """Cluster on a spatial grid instead of the intersection, as a coarse
    alternative to Conley HAC standard errors."""
    print(f"\n3. SPATIALLY CLUSTERED STANDARD ERRORS ({cut_km:.0f} km grid)")
    s = sites.set_index("px")
    d = p.copy()
    d["gx"] = (d.px.map(s.lon) * LON_M / (cut_km * 1000)).round().astype("Int64")
    d["gy"] = (d.px.map(s.lat) * LAT_M / (cut_km * 1000)).round().astype("Int64")
    d["cell"] = d.gx.astype(str) + "_" + d.gy.astype(str)
    keep = d.groupby("px").ped_ksi.sum()
    d = d[d.px.isin(keep[keep > 0].index)]
    m = pf.fepois("ped_ksi ~ treated | px + year", data=d,
                  vcov={"CRV1": "cell"})
    print(f"  {d.cell.nunique()} spatial cells")
    rec("Clustered on 1 km grid cell", m)


# ---------------------------------------------------------------- 4 geography

def geographic_heterogeneity(p, sites):
    print("\n4. GEOGRAPHIC HETEROGENEITY")
    s = sites.set_index("px")
    d = p.copy()
    d["km_core"] = d.px.map(s.km_core)
    d["inner"] = (d.km_core <= 7).astype(int)      # roughly the pre-amalgamation city
    d["t_inner"] = d.treated * d.inner
    m, _ = fit(d, "ped_ksi", "treated + t_inner")
    rec("Effect in outer Toronto (>7 km from core)", m, "treated")
    rec("Additional effect in inner Toronto", m, "t_inner")

    n_in = d[d.inner == 1].px.nunique()
    n_out = d[d.inner == 0].px.nunique()
    print(f"  {n_in} inner intersections, {n_out} outer")

    # treated-site density: are effects larger where LPIs are dense?
    d["dense"] = (d.get("nb_share", pd.Series(0, index=d.index)) > 0.5).astype(int)
    return d


if __name__ == "__main__":
    p = pd.read_csv("panel_full.csv", low_memory=False)
    p["px"] = p.px.astype(str).str.zfill(4)

    sites = (p.groupby("px")
               .agg(lat=("lat", "first"), lon=("lon", "first"),
                    lpi_year=("lpi_year", "first"))
               .reset_index())
    sites = add_geography(p, sites)
    print(f"panel: {p.px.nunique():,} intersections, {len(p):,} site-years")

    d = spillover(p, sites, radius=500)
    autocorrelation(p, sites)
    conley_check(p, sites)
    geographic_heterogeneity(p, sites)

    pd.DataFrame(RESULTS).to_csv("results_spatial.csv", index=False)
    d[["px", "year", "nb_treated", "nb_total", "nb_share"]].to_csv(
        "spillover_panel.csv", index=False)
    print("\nwrote results_spatial.csv, spillover_panel.csv")
