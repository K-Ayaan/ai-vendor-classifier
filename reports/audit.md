# Raw data audit

Source directory: `data\raw` (23 files)

## 1. Header `count` vs len(points), duplicates within file

| file | category | status | header_count | len_points | count_matches | outside | dup_ids_in_file | updated_at_utc |
|---|---|---|---|---|---|---|---|---|
| apparel_footwear.json | apparel_footwear | complete | 281 | 281 | True | 60 | 0 | 2026-09-27T13:51:03.603442+00:00 |
| auto_parts.json | auto_parts | complete | 186 | 186 | True | 51 | 0 | 2026-09-27T14:23:41.005131+00:00 |
| bakery_sweets.json | bakery_sweets | complete | 226 | 226 | True | 38 | 0 | 2026-09-27T12:55:39.023956+00:00 |
| beauty_personal_care.json | beauty_personal_care | complete | 138 | 138 | True | 42 | 0 | 2026-09-27T13:56:06.608936+00:00 |
| dairy_eggs.json | dairy_eggs | complete | 236 | 236 | True | 86 | 0 | 2026-09-27T12:34:59.050194+00:00 |
| electronics.json | electronics | complete | 217 | 217 | True | 38 | 0 | 2026-09-27T13:12:14.764006+00:00 |
| florists.json | florists | complete | 124 | 124 | True | 46 | 0 | 2026-09-27T14:08:58.909381+00:00 |
| food_stalls.json | food_stalls | complete | 216 | 216 | True | 65 | 0 | 2026-09-27T13:04:15.729872+00:00 |
| fruits.json | fruits | complete | 244 | 244 | True | 54 | 0 | 2026-09-27T10:40:29.485905+00:00 |
| grocery_supermarket.json | grocery_supermarket | complete | 243 | 243 | True | 38 | 0 | 2026-09-27T11:08:37.839828+00:00 |
| hardware.json | hardware | complete | 303 | 303 | True | 64 | 0 | 2026-09-27T13:34:30.039606+00:00 |
| home_appliances.json | home_appliances | complete | 168 | 168 | True | 18 | 0 | 2026-09-27T13:25:28.995269+00:00 |
| household_homeware.json | household_homeware | complete | 252 | 252 | True | 104 | 0 | 2026-09-27T14:04:16.296764+00:00 |
| kirana_provisions.json | kirana_provisions | complete | 405 | 405 | True | 100 | 0 | 2026-09-27T11:00:39.276164+00:00 |
| meat_fish.json | meat_fish | complete | 323 | 323 | True | 71 | 0 | 2026-09-27T12:47:16.077681+00:00 |
| mobile_accessories.json | mobile_accessories | complete | 205 | 205 | True | 51 | 0 | 2026-09-27T13:19:47.506858+00:00 |
| optical.json | optical | complete | 136 | 136 | True | 44 | 0 | 2026-09-27T14:13:38.286063+00:00 |
| pet_supplies.json | pet_supplies | complete | 119 | 119 | True | 23 | 0 | 2026-09-27T14:17:48.081257+00:00 |
| pharmacy.json | pharmacy | complete | 486 | 486 | True | 64 | 0 | 2026-09-27T12:23:25.393257+00:00 |
| ration_pds.json | ration_pds | complete | 173 | 173 | True | 23 | 0 | 2026-09-27T10:46:37.532607+00:00 |
| stationery_books.json | stationery_books | complete | 268 | 268 | True | 53 | 0 | 2026-09-27T13:42:14.342279+00:00 |
| street_markets.json | street_markets | complete | 227 | 227 | True | 42 | 0 | 2026-09-27T11:16:34.716458+00:00 |
| vegetables.json | vegetables | complete | 263 | 263 | True | 65 | 0 | 2026-09-27T10:30:38.166315+00:00 |

Files where count != len(points): 0
Distinct updated_at_utc dates: ['2026-09-27']
Note: `count` and `outside` are file metadata only; never used as features or targets.

## 2. Nulls and empty strings per field (all point records)

| field | null | empty_string |
|---|---|---|
| id | 0 | 0 |
| title | 0 | 0 |
| address | 0 | 1 |
| lat | 0 | 0 |
| lon | 0 | 0 |

Records with unexpected extra keys: 0 {}

## 3. Duplicate ids across files

- Total point records: 5439
- Unique ids: 4612
- Records that are repeats of an id seen elsewhere: 827
- Ids whose title differs between files: 0
- Ids whose lat/lon differs between files: 0

## 4. Number of categories per unique id

| n_categories | n_ids | share |
|---|---|---|
| 1 | 4094 | 0.8877 |
| 2 | 368 | 0.0798 |
| 3 | 86 | 0.0186 |
| 4 | 29 | 0.0063 |
| 5 | 8 | 0.0017 |
| 6 | 13 | 0.0028 |
| 7 | 3 | 0.0007 |
| 8 | 6 | 0.0013 |
| 9 | 2 | 0.0004 |
| 10 | 3 | 0.0007 |

Top 10 co-occurring category pairs:

| pair | n_ids |
|---|---|
| fruits + vegetables | 98 |
| grocery_supermarket + kirana_provisions | 94 |
| kirana_provisions + pet_supplies | 60 |
| street_markets + vegetables | 56 |
| grocery_supermarket + street_markets | 51 |
| kirana_provisions + street_markets | 46 |
| electronics + home_appliances | 45 |
| fruits + street_markets | 37 |
| electronics + mobile_accessories | 33 |
| grocery_supermarket + vegetables | 31 |

## 5. lat/lon ranges (unique ids)

| index | lat | lon |
|---|---|---|
| count | 4612.0 | 4612.0 |
| mean | 12.876064 | 80.114667 |
| std | 0.130203 | 0.104415 |
| min | 12.65098 | 79.904572 |
| 25% | 12.743294 | 80.004034 |
| 50% | 12.902067 | 80.116831 |
| 75% | 12.990647 | 80.213154 |
| max | 13.059984 | 80.280627 |

## 6. Address pincode / Plus Code coverage (unique ids)

- Addresses with a 6-digit pincode: 4611/4612 (99.98%)
- Addresses with a Plus Code: 551/4612 (11.95%)
- Addresses with neither: 1
- Distinct pincodes (last match per address): 106

## 7. Top repeated titles (normalised, unique ids)

Titles appearing >= 3 times: 46 distinct titles covering 234 ids

| normalised_title | n_ids |
|---|---|
| ration shop | 39 |
| fair price shop | 12 |
| hap daily | 9 |
| reliance digital | 9 |
| fish market | 9 |
| xerox shop | 9 |
| sri krishna sweets | 8 |
| hetero pharmacy | 6 |
| metro shoes | 5 |
| aavin milk shop | 5 |
| the chennai mobiles | 5 |
| lovely pet shop | 5 |
| pharmacy | 5 |
| ganga sweets | 4 |
| aavin parlour | 4 |
| lenovo exclusive store gbs systems services | 4 |
| vasanth co | 4 |
| flower shop | 4 |
| fruit market | 4 |
| reliance smart bazaar | 4 |
| sekar stores | 4 |
| government ration shop | 4 |
| mochi shoes | 3 |
| shree mithai | 3 |
| nandhiniee sweets | 3 |
