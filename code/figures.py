"""Figures for the Toronto LPI evaluation."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import shape

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linestyle": "-",
    "axes.axisbelow": True, "figure.dpi": 200,
})
INK, ACC, WARM, MUTED = "#14211c", "#0b6e4f", "#b4531f", "#98a09c"


def fig_rollout(panel, path="f1_rollout.png"):
    s = panel.groupby("px").agg(lpi=("lpi_year", "first"))
    y = s.lpi.dropna().astype(int).value_counts().sort_index()
    y = y[(y.index >= 2009) & (y.index <= 2026)]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.bar(y.index, y.values, color=ACC, width=0.72)
    ax.set_xlabel("Year of first LPI installation")
    ax.set_ylabel("Intersections")
    ax.set_title("Toronto LPI installations by year", loc="left",
                 fontsize=10.5, color=INK, pad=10)
    ax.annotate("Vision Zero acceleration", xy=(2021, y.get(2021, 0)),
                xytext=(2013.4, y.max() * 0.82), fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_event(es, pre_p, path="f2_event_study.png"):
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvspan(-0.5, es.evt.max() + 0.5, color=ACC, alpha=0.05, lw=0)
    ax.axvline(-0.5, color=MUTED, lw=1, ls=":")
    ax.fill_between(es.evt, es.lo, es.hi, color=ACC, alpha=0.16, lw=0)
    ax.plot(es.evt, es.pct, "o-", color=ACC, lw=1.7, ms=4.5)
    ax.set_xlabel("Years relative to LPI installation")
    ax.set_ylabel("Change in pedestrian KSI (%)")
    ax.set_title("Pedestrian serious injuries around LPI installation",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    ax.text(0.985, 0.06, f"joint pre-trend test  p = {pre_p:.2f}",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
    ax.text(0, -0.245, "Poisson event study with intersection and year fixed effects. "
            "Never-treated intersections anchor the\ncomparison, year −1 is omitted, and periods beyond the window are binned rather than dropped.",
            transform=ax.transAxes, fontsize=7.6, color=MUTED, va="top")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_mechanism(res, path="f3_mechanism.png"):
    want = ["Turning driver hits ped with ROW",
            "Straight driver hits ped with ROW",
            "Ped crossing WITHOUT right-of-way",
            "Ped on sidewalk or midblock",
            "Vehicle-occupant KSI"]
    r = res[res.label.isin(want)].set_index("label").loc[want].reset_index()
    treat = [True, True, False, False, False]
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    y = np.arange(len(r))[::-1]
    ax.axvline(0, color=MUTED, lw=1)
    for i, (yy, row, t) in enumerate(zip(y, r.itertuples(), treat)):
        c = ACC if t else WARM
        ax.hlines(yy, row.lo, row.hi, color=c, lw=2.2, alpha=0.75)
        ax.plot(row.pct, yy, "o", color=c, ms=6)
    ax.set_yticks(y)
    ax.set_yticklabels(["Turning driver, ped has right-of-way",
                        "Straight driver, ped has right-of-way",
                        "Ped crossing without right-of-way",
                        "Ped on sidewalk or midblock",
                        "Vehicle occupants only (placebo)"])
    ax.set_xlabel("Change after LPI installation (%)")
    ax.set_title("The effect appears only where an LPI can act",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    ax.legend(handles=[Patch(color=ACC, label="LPI can plausibly prevent"),
                       Patch(color=WARM, label="LPI cannot prevent")],
              loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_specs(res, path="f4_specifications.png"):
    order = [
        ("Pedestrian KSI, unadjusted", "Main estimate"),
        ("+ both co-treatments", "Adjusted for red light cameras and bike lanes"),
        ("Stacked DiD, not-yet-treated controls", "Stacked DiD"),
        ("Empirical Bayes before-after", "Empirical Bayes (HSM)"),
        ("Snap radius 20 m", "Snap radius 20 m"),
        ("Snap radius 50 m", "Snap radius 50 m"),
        ("Pedestrian, age 65+", "Pedestrians aged 65+"),
        ("Non-fatal pedestrian KSI", "Non-fatal injuries"),
        ("Fatal pedestrian collisions", "Fatal collisions"),
        ("Cyclist KSI", "Cyclists"),
    ]
    r = (res.drop_duplicates("label").set_index("label")
            .reindex([o for o, _ in order]).reset_index())
    r["disp"] = [d for _, d in order]
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(r) + 1.4))
    y = np.arange(len(r))[::-1]
    ax.axvline(0, color=MUTED, lw=1)
    ax.hlines(y, r.lo, r.hi, color=ACC, lw=2.1, alpha=0.7)
    ax.plot(r.pct, y, "o", color=ACC, ms=5.5)
    ax.set_yticks(y); ax.set_yticklabels(r.disp)
    ax.set_xlabel("Change in collisions after LPI installation (%)")
    ax.set_title("Estimates across specifications and outcomes",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_ri(null, obs, path="f5_randomization.png"):
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    pct = (np.exp(null) - 1) * 100
    ax.hist(pct, bins=28, color=MUTED, alpha=0.55, edgecolor="white", lw=0.5)
    o = (np.exp(obs) - 1) * 100
    ax.axvline(o, color=WARM, lw=2)
    ax.annotate(f"observed {o:+.1f}%", xy=(o, ax.get_ylim()[1] * 0.72),
                xytext=(o + 6, ax.get_ylim()[1] * 0.82), fontsize=8.5, color=WARM,
                arrowprops=dict(arrowstyle="->", color=WARM, lw=0.9))
    ax.set_xlabel("Estimated effect under randomly permuted treatment timing (%)")
    ax.set_ylabel("Permutations")
    ax.set_title("Randomization inference", loc="left", fontsize=10.5,
                 color=INK, pad=10)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_map(panel, path="f6_map.png"):
    s = panel.groupby("px").agg(lat=("lat", "first"), lon=("lon", "first"),
                                lpi=("lpi_year", "first"), ped=("ped_ksi", "sum"))
    gj = json.load(open("nia.geojson"))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for f in gj["features"]:
        g = shape(f["geometry"])
        for poly in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            x, y = poly.exterior.xy
            ax.fill(x, y, color=MUTED, alpha=0.16, lw=0)
    un = s[s.lpi.isna()]
    tr = s[s.lpi.notna()]
    ax.scatter(un.lon, un.lat, s=3.5, color=MUTED, alpha=0.55, lw=0, label="No LPI")
    ax.scatter(tr.lon, tr.lat, s=3.5, color=ACC, alpha=0.75, lw=0, label="LPI installed")
    big = s[s.ped >= 3]
    ax.scatter(big.lon, big.lat, s=26, facecolor="none", edgecolor=WARM, lw=0.8,
               label="3+ pedestrian KSI, 2010–2025")
    ax.set_aspect(1 / np.cos(np.radians(43.7)))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=8, markerscale=1.6)
    ax.set_title("Signalized intersections, LPI status and pedestrian KSI",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    ax.text(0, -0.04, "Shaded areas are Neighbourhood Improvement Areas.",
            transform=ax.transAxes, fontsize=7.6, color=MUTED)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    panel = pd.read_csv("panel_full.csv", low_memory=False)
    panel["px"] = panel.px.astype(str).str.zfill(4)
    res = pd.read_csv("results_full.csv")
    es = pd.read_csv("event_study_binned.csv")
    null = np.load("ri_null.npy")

    obs = np.log(1 + res[res.label == "Pedestrian KSI, unadjusted"].pct.iloc[0] / 100)

    fig_rollout(panel)
    fig_event(es, 0.739)
    fig_mechanism(res)
    fig_specs(res)
    fig_ri(null, obs)
    fig_map(panel)
    print("wrote f1..f6")
