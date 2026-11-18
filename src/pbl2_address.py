"""Addresses as sequences, for the PBL-2 HMM (Unit IV).

An address is split on commas into parts ("segments"). Each segment is an OBSERVATION, reduced to
one of M discrete symbols by its surface form (digits, road words, "nagar" words, ...). The
HIDDEN STATE of a segment is its role in the address:

    PREMISE  door number, building, shop name, landmark, plus code
    STREET   a street or road
    AREA     a neighbourhood, nagar, colony or village inside a town
    CITY     the town / city / taluk
    STATE    "Tamil Nadu <pincode>"

The same surface form can play different roles ("Anna Nagar" is an AREA, "Maraimalai Nagar" is
a CITY), so the role depends on the neighbours: exactly what an HMM's transitions model.
"""
import re

import numpy as np

STATES = ["PREMISE", "STREET", "AREA", "CITY", "STATE"]
SYMBOLS = ["STATEPIN", "PLUSCODE", "NUMBER", "TOWN", "ROAD", "NAGAR", "LANDMARK",
           "BUILDING", "SHORT", "WORD"]
TOWN_MIN_COUNT = 30  # a name is a TOWN if it is the part before the state in >= 30 train addresses
S = {s: i for i, s in enumerate(STATES)}
O = {s: i for i, s in enumerate(SYMBOLS)}

ROAD_WORDS = {"st", "street", "rd", "road", "salai", "lane", "ln", "highway", "nh", "bypass",
              "cross", "theru", "veedhi", "path", "trunk", "gst", "omr", "ecr", "marg"}
NAGAR_WORDS = {"nagar", "colony", "layout", "garden", "gardens", "enclave", "avenue", "puram",
               "extension", "extn", "phase", "sector", "township", "village", "city", "town",
               "kuppam", "palayam", "pettai", "pet", "medu", "kottai", "salai_area", "park",
               "residency", "estate", "society", "quarters", "housing", "board"}
LANDMARK_WORDS = {"opp", "opposite", "near", "behind", "next", "beside", "adjacent", "bus",
                  "stop", "junction", "signal", "bridge", "temple", "koil", "school", "college",
                  "hospital", "bank", "station", "church", "masjid", "petrol", "bunk"}
BUILDING_WORDS = {"floor", "flr", "complex", "building", "bldg", "shop", "plaza", "mall",
                  "apartments", "apartment", "apts", "tower", "towers", "mansion", "block",
                  "store", "stores", "market", "mart", "door", "flat", "plot", "survey", "no",
                  "unit", "ground", "first", "second", "1st", "2nd", "3rd"}
PLUS_RE = re.compile(r"^[23456789CFGHJMPQRVWX]{4}\+[23456789CFGHJMPQRVWX]{2,3}\b")
NUM_RE = re.compile(r"^(no\.?\s*|old no\.?\s*|new no\.?\s*|plot no\.?\s*|door no\.?\s*)?"
                    r"[\d][\d/\-a-z\s.,&]*$", re.I)
WORD_RE = re.compile(r"[a-z0-9]+")


def segments(address):
    """Comma-separated, stripped, non-empty parts."""
    return [p.strip() for p in str(address).split(",") if p.strip()]


def town_gazetteer(addresses, min_count=TOWN_MIN_COUNT):
    """Lower-cased names that appear immediately before the state part in at least
    `min_count` of the given (training) addresses."""
    from collections import Counter
    c = Counter()
    for a in addresses:
        parts = segments(a)
        idx = [i for i, p in enumerate(parts) if "tamil nadu" in p.lower()]
        if idx and idx[-1] > 0:
            c[parts[idx[-1] - 1].lower()] += 1
    return frozenset(n for n, k in c.items() if k >= min_count)


def symbol(seg, towns=frozenset()):
    """Surface form of one segment -> symbol index. Rules are checked in a fixed order."""
    low = seg.lower()
    words = WORD_RE.findall(low)
    ws = set(words)
    if "tamil nadu" in low:
        return O["STATEPIN"]
    if PLUS_RE.match(seg):
        return O["PLUSCODE"]
    if NUM_RE.match(seg) or (words and words[0][0].isdigit() and len(words) <= 2):
        return O["NUMBER"]
    if low in towns:
        return O["TOWN"]
    if ws & ROAD_WORDS:
        return O["ROAD"]
    if ws & NAGAR_WORDS:
        return O["NAGAR"]
    if ws & LANDMARK_WORDS:
        return O["LANDMARK"]
    if ws & BUILDING_WORDS:
        return O["BUILDING"]
    if len(words) <= 1 and len(low) <= 4:
        return O["SHORT"]
    return O["WORD"]


def encode(address, towns=frozenset()):
    return np.array([symbol(s, towns) for s in segments(address)], dtype=int)


def rule_labels(address, towns=frozenset()):
    """Deterministic baseline ("silver") labels, written without looking at the gold set:
    STATEPIN -> STATE; the segment just before STATE, or a TOWN, -> CITY; numbers, plus codes, buildings,
    landmarks -> PREMISE; road words -> STREET; nagar words -> AREA; any other segment ->
    PREMISE if it comes before the first street, else AREA."""
    syms = encode(address, towns)
    lab = np.empty(len(syms), dtype=int)
    state_pos = [i for i, s in enumerate(syms) if s == O["STATEPIN"]]
    first_road = next((i for i, s in enumerate(syms) if s == O["ROAD"]), len(syms))
    for i, s in enumerate(syms):
        if s == O["STATEPIN"]:
            lab[i] = S["STATE"]
        elif (state_pos and i == state_pos[-1] - 1) or s == O["TOWN"]:
            lab[i] = S["CITY"]
        elif s in (O["NUMBER"], O["PLUSCODE"], O["BUILDING"], O["LANDMARK"]):
            lab[i] = S["PREMISE"]
        elif s == O["ROAD"]:
            lab[i] = S["STREET"]
        elif s == O["NAGAR"]:
            lab[i] = S["AREA"]
        else:
            lab[i] = S["PREMISE"] if i < first_road else S["AREA"]
    return lab


def parse_label_string(s):
    """'PREMISE|STREET|AREA|CITY|STATE' -> state indices."""
    return np.array([S[t] for t in s.split("|")], dtype=int)
