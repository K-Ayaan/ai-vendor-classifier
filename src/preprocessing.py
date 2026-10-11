"""Step 2 - build the cleaned vendor table (one row per unique source id).

Usage: python src/preprocessing.py
Reads data/raw/*.json (never modified) and writes:
  data/processed/vendors.csv   (UTF-8)
  reports/preprocessing.md     (verification counts)
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_CSV = Path("data/processed/vendors.csv")
OUT_REPORT = Path("reports/preprocessing.md")

COLUMNS = ["vendor_id", "source_place_id", "title", "address", "lat", "lon", "pincode",
           "pincode_imputed", "locality", "categories", "n_categories", "is_chain"]
PINCODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
VALID_PINCODE_RE = re.compile(r"^6\d{5}$")  # every Tamil Nadu pincode starts with 6
STATE = "Tamil Nadu"
# study area (South Chennai to Chengalpattu) with a margin around the audited ranges
LAT_RANGE, LON_RANGE = (12.5, 13.2), (79.8, 80.4)
CHAIN_MIN_REPEATS = 3
EARTH_RADIUS_KM = 6371.0088


def clean_text(s):
    """Unicode NFC (keeps Tamil intact), then trim and collapse whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def norm_title(t):
    """Lowercase, keep letters/digits/combining marks in any script, collapse spaces."""
    kept = "".join(c if unicodedata.category(c)[0] in "LNM" else " " for c in t.lower())
    return re.sub(r"\s+", " ", kept).strip()


def extract_pincode(address):
    # the pincode follows the state at the end; earlier 6-digit runs are door numbers
    found = PINCODE_RE.findall(address)
    return found[-1] if found else None


def extract_locality(address):
    """Address part immediately before the part containing the state."""
    parts = [p.strip() for p in address.split(",")]
    state_idx = [i for i, p in enumerate(parts) if STATE in p]
    if not state_idx or state_idx[-1] == 0:
        return None
    return parts[state_idx[-1] - 1] or None


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_records():
    rows, categories = [], []
    for f in sorted(RAW_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        categories.append(d["category"])
        for p in d["points"]:
            rows.append({"source_place_id": p["id"], "title": clean_text(p["title"]),
                         "address": clean_text(p["address"]), "lat": float(p["lat"]),
                         "lon": float(p["lon"]), "category": d["category"]})
    return pd.DataFrame(rows), sorted(categories)


def merge_by_id(records):
    """One row per source id. Fails if a repeated id disagrees on title or location."""
    conflicts = records.groupby("source_place_id")[["title", "address", "lat", "lon"]].nunique()
    bad = conflicts[(conflicts > 1).any(axis=1)]
    if len(bad):
        raise ValueError(f"{len(bad)} source ids have conflicting fields: {list(bad.index[:5])}")
    cats = records.groupby("source_place_id")["category"].agg(lambda s: sorted(set(s)))
    v = records.drop_duplicates("source_place_id").set_index("source_place_id")
    v = v.drop(columns="category").join(cats.rename("category_list")).reset_index()
    # deterministic ids: sort by source id, then number 1..n
    v = v.sort_values("source_place_id", kind="mergesort").reset_index(drop=True)
    width = max(4, len(str(len(v))))
    v.insert(0, "vendor_id", [f"V{i:0{width}d}" for i in range(1, len(v) + 1)])
    return v


def impute_missing_pincodes(v):
    """Give each vendor without a pincode the pincode of its nearest valid neighbour."""
    v["pincode_imputed"] = False
    donors = v[v.pincode.notna() & v.pincode.fillna("").str.match(VALID_PINCODE_RE)
               & v.lat.between(*LAT_RANGE) & v.lon.between(*LON_RANGE)]
    log = []
    for i in v.index[v.pincode.isna()]:
        if not (LAT_RANGE[0] <= v.at[i, "lat"] <= LAT_RANGE[1]
                and LON_RANGE[0] <= v.at[i, "lon"] <= LON_RANGE[1]):
            raise ValueError(f"{v.at[i, 'vendor_id']} has no pincode and an invalid location")
        d = haversine_km(v.at[i, "lat"], v.at[i, "lon"], donors.lat.values, donors.lon.values)
        j = donors.index[int(np.argmin(d))]
        v.at[i, "pincode"] = v.at[j, "pincode"]
        v.at[i, "pincode_imputed"] = True
        log.append({"vendor_id": v.at[i, "vendor_id"], "title": v.at[i, "title"],
                    "neighbour_vendor_id": v.at[j, "vendor_id"],
                    "neighbour_title": v.at[j, "title"], "pincode": v.at[j, "pincode"],
                    "distance_km": round(float(d.min()), 4)})
    return v, log


def validate(df, known_categories):
    """Schema validation. Raises ValueError listing every problem found."""
    errs = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if list(df.columns) != COLUMNS:
        errs.append(f"column order {list(df.columns)} != {COLUMNS}")
    for c in ["vendor_id", "source_place_id", "title", "lat", "lon", "pincode", "categories"]:
        if df[c].isna().any() or (df[c].astype(str).str.strip() == "").any():
            errs.append(f"{c}: null or empty values")
    for c in ["vendor_id", "source_place_id"]:
        if df[c].duplicated().any():
            errs.append(f"{c}: {int(df[c].duplicated().sum())} duplicates")
    if not df.vendor_id.astype(str).str.match(r"^V\d{4,}$").all():
        errs.append("vendor_id: bad format")
    if not df.lat.between(*LAT_RANGE).all() or not df.lon.between(*LON_RANGE).all():
        errs.append("lat/lon outside study area")
    if not np.allclose(df.lat, df.lat.round(7)) or not np.allclose(df.lon, df.lon.round(7)):
        errs.append("lat/lon not rounded to 7 decimals")
    if not df.pincode.astype(str).str.match(VALID_PINCODE_RE).all():
        errs.append("pincode: not a 6-digit Tamil Nadu pincode")
    for c in ["pincode_imputed", "is_chain"]:
        if not df[c].isin([True, False]).all():
            errs.append(f"{c}: not boolean")
    cat_lists = df.categories.astype(str).str.split(";")
    if not (cat_lists.str.len() == df.n_categories).all():
        errs.append("n_categories does not match categories")
    if not cat_lists.map(lambda c: c == sorted(set(c))).all():
        errs.append("categories not sorted/unique")
    unknown = set().union(*cat_lists.map(set)) - set(known_categories)
    if unknown:
        errs.append(f"unknown categories: {sorted(unknown)}")
    if (df.n_categories < 1).any():
        errs.append("vendor with no category")
    if errs:
        raise ValueError("vendor table failed validation:\n  - " + "\n  - ".join(errs))


def md_table(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    records, known_categories = load_records()
    v = merge_by_id(records)

    v["lat"], v["lon"] = v.lat.round(7), v.lon.round(7)
    v["pincode"] = v.address.map(extract_pincode)
    v, imputation_log = impute_missing_pincodes(v)
    v["locality"] = v.address.map(extract_locality)
    v["categories"] = v.category_list.map(";".join)
    v["n_categories"] = v.category_list.map(len)

    # locked rule: a normalised title repeated on >=3 vendors is a "chain".
    # Limitation: generic names ("ration shop", "fish market") also match.
    nt = v.title.map(norm_title)
    counts = nt.value_counts()
    v["is_chain"] = nt.map(counts).ge(CHAIN_MIN_REPEATS) & nt.ne("")

    v = v[COLUMNS]
    validate(v, known_categories)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    v.to_csv(OUT_CSV, index=False, encoding="utf-8", lineterminator="\n")

    # round-trip check: the file on disk must load back and pass the same validation
    back = pd.read_csv(OUT_CSV, encoding="utf-8", dtype={"pincode": str}, keep_default_na=False,
                       na_values={"locality": [""]})
    validate(back, known_categories)
    if not back.title.equals(v.title.reset_index(drop=True)):
        raise ValueError("titles changed on CSV round-trip (encoding problem)")

    # ---- verification summary ----
    n_records, n_vendors = len(records), len(v)
    prevalence = (records.groupby("category").source_place_id.nunique()
                  .reindex(known_categories).rename("n_vendors").reset_index())
    prevalence["prevalence"] = (prevalence.n_vendors / n_vendors).round(4)
    file_hash = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()
    mapping_hash = hashlib.sha256(
        "\n".join(v.vendor_id + "," + v.source_place_id).encode("utf-8")).hexdigest()

    print(f"raw records: {n_records} | vendors (unique source ids): {n_vendors}")
    print(f"category memberships: {int(v.n_categories.sum())} (must equal raw records)")
    print(f"vendor_id range: {v.vendor_id.iloc[0]}..{v.vendor_id.iloc[-1]}")
    print(f"pincodes imputed: {len(imputation_log)} {imputation_log}")
    print(f"locality null: {int(v.locality.isna().sum())} | distinct: {v.locality.nunique()}")
    print(f"is_chain True: {int(v.is_chain.sum())} vendors, "
          f"{int((counts >= CHAIN_MIN_REPEATS).sum())} distinct normalised titles")
    print(f"id mapping sha256: {mapping_hash}")
    print(f"vendors.csv sha256: {file_hash}")
    print("label prevalence per category:")
    print(prevalence.to_string(index=False))

    if int(v.n_categories.sum()) != n_records:
        raise ValueError("category memberships do not add up to raw record count")

    top_chain = (nt[v.is_chain].value_counts().head(15)
                 .rename_axis("normalised_title").reset_index(name="n_vendors"))
    report = [
        "# Preprocessing verification\n",
        f"Generated by `python src/preprocessing.py` from `{RAW_DIR}`.\n",
        "## Counts\n",
        f"- Raw point records: {n_records}",
        f"- Vendors (one row per unique source id): {n_vendors}",
        f"- Category memberships after merge: {int(v.n_categories.sum())} "
        f"(equals raw records: {int(v.n_categories.sum()) == n_records})",
        f"- Categories: {len(known_categories)}",
        f"- `vendor_id` range: {v.vendor_id.iloc[0]} to {v.vendor_id.iloc[-1]} "
        "(assigned in sorted `source_place_id` order)",
        f"- Vendors with empty address: {int((v.address == '').sum())}",
        f"- `locality` missing: {int(v.locality.isna().sum())}; distinct localities: "
        f"{v.locality.nunique()}",
        f"- Distinct pincodes: {v.pincode.nunique()}\n",
        "## Deterministic ID mapping\n",
        f"- SHA-256 of the `vendor_id,source_place_id` mapping: `{mapping_hash}`",
        f"- SHA-256 of `vendors.csv`: `{file_hash}`",
        "- Rerunning on unchanged raw data must reproduce both hashes.\n",
        "## Pincode imputation\n",
        "Vendors without a pincode in their address get the pincode of the nearest "
        "vendor (haversine distance) that has a valid location and a valid pincode "
        "taken from its own address. They are flagged `pincode_imputed = True`.\n",
        md_table(pd.DataFrame(imputation_log)) if imputation_log else "None.",
        "\n## Label prevalence per category\n",
        "`n_vendors` = vendors whose `categories` include the category; "
        "`prevalence` = n_vendors / total vendors. Categories overlap, so shares sum "
        "to more than 1.\n",
        md_table(prevalence),
        "\n## Chain flag (`is_chain`)\n",
        f"Rule: normalised title shared by >= {CHAIN_MIN_REPEATS} vendors. "
        f"{int(v.is_chain.sum())} vendors flagged.\n",
        "**Limitation:** generic names (e.g. \"ration shop\", \"fish market\") meet the "
        "rule without being chains. Do not treat `is_chain` as ground truth.\n",
        md_table(top_chain),
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV} and {OUT_REPORT}")


if __name__ == "__main__":
    main()
