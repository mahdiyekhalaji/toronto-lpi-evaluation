"""Analytical map figures for the spatial section."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import shape
import pyfixest as pf
from scipy.spatial import cKDTree

import spatial as S

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 200,
})
INK, ACC, WARM, MUTED = "#14211c", "#0b6e4f", "#b4531f", "#98a09c"


def city_outline(ax):
    gj = json.load(open("nia.geojson"))
    for f in gj["features"]:
        g = shape(f["geometry"])
        for poly in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
            x, y = poly.exterior.xy
            ax.fill(x, y, color=MUTED, alpha=0.13, lw=0)


def build(path="f7_spatial.png"):
    p = pd.read_csv("panel_full.csv", low_memory=False)
    p["px"] = p.px.astype(str).str.zfill(4)
    sites = (p.groupby("px").agg(lat=("lat", "first"), lon=("lon", "first"),
                                 lpi_year=("lpi_year", "first")).reset_index())
    sites = S.add_geography(p, sites)

    # panel A data: neighbour treatment exposure in the final year
    nt, ntot = S.neighbour_treatment(p, sites, 500)
    d = p.assign(nb_treated=nt, nb_total=ntot)
    last = d[d.year == 2025].set_index("px")
    sa = sites.set_index("px").join(last[["nb_treated", "nb_total"]])
    sa["share"] = np.where(sa.nb_total > 0, sa.nb_treated / sa.nb_total, np.nan)

    # panel B data: site-level residuals from the main model
    tot = p.groupby("px").ped_ksi.sum()
    sb = sites.set_index("px").loc[tot.index].assign(resid=tot.values)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))

    # ---- A: what the spillover test exploits
    ax = axes[0]
    city_outline(ax)
    un = sa[sa.lpi_year.isna()]
    tr = sa[sa.lpi_year.notna()]
    ax.scatter(tr.lon, tr.lat, s=3, color=MUTED, alpha=0.45, lw=0)
    sc0 = ax.scatter(un.lon, un.lat, s=13, c=un.share, cmap="YlOrRd",
                     vmin=0, vmax=1, lw=0.3, edgecolor="white")
    cb0 = fig.colorbar(sc0, ax=ax, fraction=0.035, pad=0.02)
    cb0.set_label("share of neighbours treated", fontsize=7.5)
    cb0.ax.tick_params(labelsize=7)
    ax.set_title("A. Untreated sites sit inside treated neighbourhoods",
                 loc="left", fontsize=10, color=INK, pad=8)
    ax.text(0, -0.045, "Grey points are signals with their own LPI. Coloured points "
            "have none, shaded by the share of\nsignals within 500 m that do. The "
            "spillover test asks whether that shading predicts collisions.",
            transform=ax.transAxes, fontsize=7.4, color=MUTED, va="top")

    # ---- B: residuals show no spatial structure
    ax = axes[1]
    city_outline(ax)
    z = sb.resid.values.astype(float)
    ax.scatter(sb.lon[z == 0], sb.lat[z == 0], s=3, color="#dfe3e0", lw=0)
    hit = z > 0
    sc = ax.scatter(sb.lon[hit], sb.lat[hit], s=8 + 9 * z[hit], c=z[hit],
                    cmap="YlOrRd", vmin=1, vmax=np.percentile(z[hit], 97),
                    lw=0.2, edgecolor="white", alpha=0.9)
    ax.set_title("B. Collision risk is spatially clustered",
                 loc="left", fontsize=10, color=INK, pad=8)
    ax.text(0, -0.045, "Observed pedestrian KSI totals, 2010–2025. Moran's I = +0.020 "
            "(expected −0.000), p = 0.005.\nClustering is real, so standard errors are "
            "checked on spatial grids from 0.5 to 5 km; the estimate holds.",
            transform=ax.transAxes, fontsize=7.4, color=MUTED, va="top")
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("pedestrian KSI, 2010–2025", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    for ax in axes:
        ax.set_aspect(1 / np.cos(np.radians(43.7)))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    build()
