"""Step 0 - read-only audit of the raw category JSON files.

Usage: python src/audit.py [RAW_DIR]   (default: data/raw)
Writes reports/audit.md. Never modifies the raw files.
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

RAW_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
OUT = Path("reports/audit.md")
FIELDS = ["id", "title", "address", "lat", "lon"]
PINCODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
PLUSCODE_RE = re.compile(r"\b[23456789CFGHJMPQRVWX]{4,8}\+[23456789CFGHJMPQRVWX]{2,3}\b")


def norm_title(t):
    # keep letters, digits and combining marks in any script (Tamil vowel signs are marks)
    kept = "".join(c if unicodedata.category(c)[0] in "LNM" else " " for c in str(t).lower())
    return re.sub(r"\s+", " ", kept).strip()


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main():
    files = sorted(RAW_DIR.glob("*.json"))
    rows, header_rows = [], []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        pts = d.get("points", [])
        ids = [p.get("id") for p in pts]
        header_rows.append({
            "file": f.name, "category": d.get("category"), "status": d.get("status"),
            "header_count": d.get("count"), "len_points": len(pts),
            "count_matches": d.get("count") == len(pts),
            "outside": d.get("outside"), "dup_ids_in_file": len(ids) - len(set(ids)),
            "updated_at_utc": d.get("updated_at_utc"),
        })
        for p in pts:
            rows.append({**{k: p.get(k) for k in FIELDS}, "category": d.get("category"),
                         "extra_keys": ",".join(sorted(set(p) - set(FIELDS)))})

    hdr = pd.DataFrame(header_rows)
    df = pd.DataFrame(rows)
    out = [f"# Raw data audit\n\nSource directory: `{RAW_DIR}` ({len(files)} files)\n"]

    out.append("## 1. Header `count` vs len(points), duplicates within file\n")
    out.append(md_table(hdr))
    out.append(f"\nFiles where count != len(points): {int((~hdr.count_matches).sum())}")
    out.append(f"Distinct updated_at_utc dates: {sorted(hdr.updated_at_utc.str[:10].unique())}")
    out.append("Note: `count` and `outside` are file metadata only; never used as features or targets.\n")

    out.append("## 2. Nulls and empty strings per field (all point records)\n")
    ns = []
    for c in FIELDS:
        s = df[c]
        empty = int(s.map(lambda v: isinstance(v, str) and v.strip() == "").sum())
        ns.append({"field": c, "null": int(s.isna().sum()), "empty_string": empty})
    out.append(md_table(pd.DataFrame(ns)))
    extra = df.extra_keys[df.extra_keys != ""].value_counts()
    out.append(f"\nRecords with unexpected extra keys: {int(extra.sum())} {dict(extra)}\n")

    out.append("## 3. Duplicate ids across files\n")
    n_rec, n_uid = len(df), df.id.nunique()
    out.append(f"- Total point records: {n_rec}")
    out.append(f"- Unique ids: {n_uid}")
    out.append(f"- Records that are repeats of an id seen elsewhere: {n_rec - n_uid}")
    per_id = df.groupby("id").agg(n_cat=("category", "nunique"), n_rec=("category", "size"),
                                  n_titles=("title", "nunique"), n_lat=("lat", "nunique"),
                                  n_lon=("lon", "nunique"))
    out.append(f"- Ids whose title differs between files: {int((per_id.n_titles > 1).sum())}")
    out.append(f"- Ids whose lat/lon differs between files: "
               f"{int(((per_id.n_lat > 1) | (per_id.n_lon > 1)).sum())}\n")

    out.append("## 4. Number of categories per unique id\n")
    dist = per_id.n_cat.value_counts().sort_index().rename_axis("n_categories").reset_index(name="n_ids")
    dist["share"] = (dist.n_ids / n_uid).round(4)
    out.append(md_table(dist))
    pairs = Counter()
    for cats in df.groupby("id").category.apply(lambda s: tuple(sorted(set(s)))):
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                pairs[(cats[i], cats[j])] += 1
    out.append("\nTop 10 co-occurring category pairs:\n")
    out.append(md_table(pd.DataFrame([{"pair": f"{a} + {b}", "n_ids": n}
                                      for (a, b), n in pairs.most_common(10)])))
    out.append("")

    out.append("## 5. lat/lon ranges (unique ids)\n")
    u = df.drop_duplicates("id")
    out.append(md_table(u[["lat", "lon"]].describe().round(6).reset_index()))
    out.append("")

    out.append("## 6. Address pincode / Plus Code coverage (unique ids)\n")
    addr = u.address.fillna("")
    has_pin = addr.str.contains(PINCODE_RE)
    has_plus = addr.str.contains(PLUSCODE_RE)
    out.append(f"- Addresses with a 6-digit pincode: {int(has_pin.sum())}/{len(u)} ({has_pin.mean():.2%})")
    out.append(f"- Addresses with a Plus Code: {int(has_plus.sum())}/{len(u)} ({has_plus.mean():.2%})")
    out.append(f"- Addresses with neither: {int((~has_pin & ~has_plus).sum())}")
    pins = addr.str.findall(PINCODE_RE).str[-1].dropna()
    out.append(f"- Distinct pincodes (last match per address): {pins.nunique()}\n")

    out.append("## 7. Top repeated titles (normalised, unique ids)\n")
    tc = u.title.map(norm_title).value_counts()
    out.append(f"Titles appearing >= 3 times: {int((tc >= 3).sum())} distinct titles "
               f"covering {int(tc[tc >= 3].sum())} ids\n")
    out.append(md_table(tc.head(25).rename_axis("normalised_title").reset_index(name="n_ids")))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(files)} files, {n_rec} records, {n_uid} unique ids")


if __name__ == "__main__":
    main()
