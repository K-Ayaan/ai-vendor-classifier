# Address-segment labelling guide (gold set for the PBL-2 HMM)

`address_gold.csv` holds 60 addresses drawn at random (seed 42) from the **test split** of the
vendor table (the same stratified 80/20 split as the Phase 1 classifier). They are never used to
fit any model. Each address is split on commas; every part gets exactly one label, in order,
separated by `|`.

Labels were assigned by an AI judge (Claude Code) reading the address, following the rules
below. They are not human ground truth.

| Label | Use for | Examples |
|---|---|---|
| `PREMISE` | door/plot numbers, building or shop names, landmarks (`near …`, `opp …`), plus codes, stray fragments (`Post`, a lone pincode) | `No.7`, `MS Complex`, `near new market`, `X57X+2C4 Ration Shop` |
| `STREET` | any street or road, including numbered streets and highway codes; a part that mixes a door number with a street name is `STREET` | `GST Rd`, `2nd St`, `SH 49`, `NH-1`, `No: 58 mudichur road`, `New` (from `New, Trunk Rd`) |
| `AREA` | a neighbourhood, nagar, colony, village or locality inside a town | `Gandhi Nagar`, `Adyar`, `Potheri`, `West Tambaram` (when not last) |
| `CITY` | the postal locality (the last place name before the state) **and** the four major towns Chennai, Chengalpattu, Tambaram, Kanchipuram wherever they appear | `Chennai`, `Maraimalai Nagar`, `Gokulapuram` (when last) |
| `STATE` | the `Tamil Nadu <pincode>` part | `Tamil Nadu 600042` |
