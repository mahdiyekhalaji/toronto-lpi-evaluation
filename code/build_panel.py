"""
Toronto LPI evaluation - data assembly.

Builds an intersection-year panel with:
  - outcomes split by collision mechanism, severity and road user
  - treatment timing and dose (number of treated approaches)
  - time-varying co-treatments (red light cameras, cycling infrastructure)
  - exposure (pedestrian and vehicle volumes from turning movement counts)
  - equity context (Neighbourhood Improvement Area)

Sources, all City of Toronto Open Data:
  ts.geojson      Traffic Signals (tabular)
  ksi.csv         Motor Vehicle Collisions involving KSI
  rlc.csv         Red Light Cameras (with activation dates)
  cyc.csv         Cycling Network (with installation years)
  tmc_summary.csv Traffic Volumes at Intersections, most recent count per site
  nia.geojson     Neighbourhood Improvement Areas
"""

import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point

START, END = 2010, 2025
SNAP_M = 30
LAT_M = 111_320.0
LON_M = 111_320.0 * np.cos(np.radians(43.7))

APPROACHES = ["NORTH", "SOUTH", "EAST", "WEST"]
LPI_COLS = [f"LPI_{d}_IMPLEMENTATION_DATE" for d in APPROACHES]


# ------------------------------------------------------------------ signals

def load_signals():
    s = pd.read_csv("ts.geojson", low_memory=False)
    s["lon"] = s.geometry.str.extract(r"\[(-?\d+\.\d+),")[0].astype(float)
    s["lat"] = s.geometry.str.extract(r",\s*(\d+\.\d+)\]")[0].astype(float)
    s = s.dropna(subset=["lat", "lon"]).copy()
    s["px"] = s.PX.astype(str).str.zfill(4)

    for c in LPI_COLS:
        s[c] = pd.to_datetime(s[c], errors="coerce")
    s["lpi_date"] = s[LPI_COLS].min(axis=1)
    s["lpi_year"] = s.lpi_date.dt.year
    s["activation"] = pd.to_datetime(s.ACTIVATIONDATE, errors="coerce")

    # dose: how many approaches had an LPI by each year
    for c, a in zip(LPI_COLS, APPROACHES):
        s[f"y_{a}"] = s[c].dt.year
    s["n_approaches"] = pd.to_numeric(s.NUMBEROFAPPROACHES, errors="coerce")
    return s


# ------------------------------------------------------------------ collisions

TURN_ROW = (
    "Driver turns left into pedestrian crossing with right-of-way (at intersection)",
    "Driver turns right into pedestrian crossing with right-of-way (at intersection)",
)
STRAIGHT_ROW = (
    "Driver going straight hits pedestrian crossing with right-of-way (at intersection)",
)
NO_ROW = (
    "Driver going straight hits pedestrian crossing without right-of-way (at intersection)",
    "Driver turns left into pedestrian crossing without right-of-way (at intersection)",
    "Driver turns right into pedestrian crossing without right-of-way (at intersection)",
)
OFF_PATH = (
    "Driver hits pedestrian on sidewalk or shoulder",
    "Driver hits pedestrian at midblock",
)


def load_ksi():
    k = pd.read_csv("ksi.csv", low_memory=False)
    k["date"] = pd.to_datetime(k.accdate, errors="coerce")
    k["year"] = k.date.dt.year
    f = lambda c: k[c].astype(str).str.upper().isin(["YES", "1", "TRUE"])
    for c in ["pedestrian", "cyclist", "motorcyclist", "older_adult",
              "school_child", "red_light", "heavy_truck"]:
        k[f"is_{c}"] = f(c)
    k["at_signal"] = k.traffictl.astype(str).str.contains("Signal", case=False, na=False)
    k["fatal"] = k.acclass.astype(str).str.strip().str.lower().eq("fatal injury")
    k["ped_age"] = pd.to_numeric(
        k.invage.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return k


def collision_sets(k):
    """Outcome definitions. Mechanism split is what identifies the LPI channel."""
    at = k[k.at_signal].dropna(subset=["latitude", "longitude", "year"])
    ped = at[at.is_pedestrian]
    pt = ped.pedtype.astype(str)

    return {
        # main
        "ped_ksi":        ped,
        # mechanism: what an LPI physically acts on
        "ped_turn_row":   ped[pt.isin(TURN_ROW)],
        "ped_straight_row": ped[pt.isin(STRAIGHT_ROW)],
        # falsification: LPI cannot plausibly prevent these
        "ped_no_row":     ped[pt.isin(NO_ROW)],
        "ped_offpath":    ped[pt.isin(OFF_PATH)],
        # severity
        "ped_fatal":      ped[ped.fatal],
        "ped_nonfatal":   ped[~ped.fatal],
        # road user and age
        "cyc_ksi":        at[at.is_cyclist],
        "ped_older":      ped[ped.is_older_adult],
        "ped_child":      ped[ped.is_school_child],
        # placebo
        "veh_only":       at[~at.is_pedestrian & ~at.is_cyclist],
    }


def snap(df, s, radius=SNAP_M):
    tree = cKDTree(np.c_[s.lat * LAT_M, s.lon * LON_M])
    d, i = tree.query(np.c_[df.latitude * LAT_M, df.longitude * LON_M])
    out = df.assign(dist_m=d, px=s.px.values[i])
    return out[out.dist_m <= radius]


# ------------------------------------------------------------------ controls

def red_light_cameras(s):
    r = pd.read_csv("rlc.csv", low_memory=False)
    r["px"] = pd.to_numeric(r.TCS, errors="coerce")
    r = r.dropna(subset=["px"])
    r["px"] = r.px.astype(int).astype(str).str.zfill(4)
    r["rlc_year"] = pd.to_datetime(r.ACTIVATION_DATE, errors="coerce").dt.year
    return r.groupby("px").rlc_year.min()


def cycling_infra(s, radius_m=150):
    """Year the first separated or painted cycling facility appeared near a signal."""
    c = pd.read_csv("cyc.csv", low_memory=False)
    c = c[c.INSTALLED.between(1980, 2026)]
    sep = c[c.INFRA_HIGHORDER.astype(str).str.contains(
        "Cycle Track|Bike Lane", case=False, na=False)].copy()

    # segment midpoints from the linestring geometry
    def midpoint(g):
        try:
            coords = json.loads(g)["coordinates"]
            flat = []
            def walk(x):
                if isinstance(x[0], (int, float)):
                    flat.append(x)
                else:
                    for y in x:
                        walk(y)
            walk(coords)
            a = np.array(flat, dtype=float)
            return a[:, 1].mean(), a[:, 0].mean()
        except Exception:
            return np.nan, np.nan

    mids = sep.geometry.map(midpoint)
    sep["lat"] = [m[0] for m in mids]
    sep["lon"] = [m[1] for m in mids]
    sep = sep.dropna(subset=["lat", "lon"])

    tree = cKDTree(np.c_[sep.lat * LAT_M, sep.lon * LON_M])
    out = {}
    for px, la, lo in zip(s.px, s.lat, s.lon):
        idx = tree.query_ball_point([la * LAT_M, lo * LON_M], radius_m)
        if idx:
            out[px] = int(sep.INSTALLED.values[idx].min())
    return pd.Series(out, name="bike_year")


def exposure():
    t = pd.read_csv("tmc_summary.csv", low_memory=False)
    t = t.dropna(subset=["px"])
    t["px"] = t.px.astype(int).astype(str).str.zfill(4)
    # count_duration is populated only for the 14-hour program that began in
    # September 2023; earlier counts are blank and were collected over 8 hours.
    dur = pd.to_numeric(t.count_duration, errors="coerce").replace(0, np.nan)
    cdate = pd.to_datetime(t.latest_count_date, errors="coerce")
    dur = dur.fillna(pd.Series(np.where(cdate < pd.Timestamp("2023-09-01"), 8.0, np.nan), index=t.index))
    # scale every count to a common 14-hour basis
    t["ped_vol"] = pd.to_numeric(t.total_pedestrian, errors="coerce") * 14 / dur
    t["veh_vol"] = pd.to_numeric(t.total_vehicle, errors="coerce") * 14 / dur
    t["bike_vol"] = pd.to_numeric(t.total_bike, errors="coerce") * 14 / dur
    return (t.groupby("px")[["ped_vol", "veh_vol", "bike_vol"]].median()
             .replace(0, np.nan).dropna())


def nia_flag(s):
    gj = json.load(open("nia.geojson"))
    polys = [shape(f["geometry"]) for f in gj["features"]]
    out = []
    for la, lo in zip(s.lat, s.lon):
        p = Point(lo, la)
        out.append(any(poly.contains(p) for poly in polys))
    return pd.Series(out, index=s.px.values, name="nia").astype(int)


# ------------------------------------------------------------------ panel

def build(radius=SNAP_M):
    s = load_signals()
    k = load_ksi()
    sets = collision_sets(k)

    years = np.arange(START, END + 1)
    p = pd.MultiIndex.from_product([s.px.unique(), years],
                                   names=["px", "year"]).to_frame(index=False)

    for name, df in sets.items():
        d = snap(df.drop_duplicates("collision_id"), s, radius)
        cnt = (d.groupby(["px", d.year.astype(int)]).collision_id.nunique()
                 .rename(name).reset_index())
        p = p.merge(cnt, on=["px", "year"], how="left")
    p[list(sets)] = p[list(sets)].fillna(0).astype(int)

    meta = s[["px", "lpi_year", "activation", "n_approaches", "lat", "lon"]
             + [f"y_{a}" for a in APPROACHES]]
    p = p.merge(meta, on="px", how="left")
    p = p[p.activation.dt.year.fillna(1900) <= p.year].copy()

    # treatment and dose
    p["treated"] = (p.lpi_year.notna() & (p.lpi_year <= p.year)).astype(int)
    p["ever"] = p.lpi_year.notna().astype(int)
    p["evt"] = np.where(p.lpi_year.notna(), p.year - p.lpi_year, np.nan)
    p["dose"] = sum((p[f"y_{a}"].notna() & (p[f"y_{a}"] <= p.year)).astype(int)
                    for a in APPROACHES)
    p["dose_frac"] = (p.dose / p.n_approaches.clip(lower=1)).clip(0, 1)

    # co-treatments, time varying
    rlc = red_light_cameras(s)
    p["rlc_year"] = p.px.map(rlc)
    p["rlc"] = (p.rlc_year.notna() & (p.rlc_year <= p.year)).astype(int)

    bike = cycling_infra(s)
    p["bike_year"] = p.px.map(bike)
    p["bike"] = (p.bike_year.notna() & (p.bike_year <= p.year)).astype(int)

    # exposure and equity, time invariant
    ex = exposure()
    p = p.merge(ex, left_on="px", right_index=True, how="left")
    p["log_ped"] = np.log(p.ped_vol)
    p["log_veh"] = np.log(p.veh_vol)
    p["nia"] = p.px.map(nia_flag(s))
    return p, s, k


if __name__ == "__main__":
    p, s, k = build()
    p.to_csv("panel_full.csv", index=False)
    print(f"{p.px.nunique():,} intersections x {START}-{END} = {len(p):,} site-years")
    print(f"ever treated {p.groupby('px').ever.max().sum():,} | "
          f"RLC {p.groupby('px').rlc.max().sum():,} | "
          f"bike infra {p.groupby('px').bike.max().sum():,} | "
          f"NIA {p.groupby('px').nia.max().sum():,}")
    print(f"exposure available for {p.groupby('px').ped_vol.first().notna().sum():,}")
    print()
    cols = [c for c in p.columns if c.startswith(("ped_", "cyc_", "veh_"))
            and c not in ("ped_vol", "veh_vol")]
    print(p[[c for c in cols if p[c].dtype != float]].sum().to_string())
