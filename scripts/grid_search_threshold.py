# Usage: python scripts/grid_search_thresholds.py [--config CONFIG] [--input INPUT] [--result-dir DIR]
#                                                 [--n-keys 10000] [--seed 42]
# Each pg_trgm pass is cached in --result-dir; reruns only query the missing thresholds.
import argparse
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mine_musicbrainz import QUERY_COMPLEX, calculate_similarity, connect

REPO = Path(__file__).resolve().parent.parent

PG_TRGM_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
DIFFLIB_GRID = [round(v, 2) for v in np.arange(0.40, 0.96, 0.05)]

MAX_RESULTS = 50
MAX_RELEASES = 20
WORK_MEM = "256MB"

# Joins (artist, track) into a JSON key; a control character never appears in names.
SEP = "\x01"


def load_relations(input_path, n_keys, seed):
    with open(input_path) as f:
        whosampled_data = json.load(f)

    sample_keys = random.Random(seed).sample(sorted(whosampled_data), n_keys)

    relations = []
    seen = {}
    for key in tqdm(sample_keys, desc="loading relations", unit=" key"):
        for ident, entry in whosampled_data[key].items():
            pair_from = (entry["by"][0]["name"], entry["track"]["name"])
            pair_to = (entry["other_track_by"][0]["name"], entry["other_track"]["name"])
            relations.append((key, ident, pair_from, pair_to))
            seen.setdefault(pair_from, None)
            seen.setdefault(pair_to, None)

    pairs = list(seen)
    print(f"{len(sample_keys):,} keys -> {len(relations):,} relations "
          f"-> {2 * len(relations):,} lookups")
    print(f"unique pairs to query: {len(pairs):,} "
          f"(dedup removes {100 * (1 - len(pairs) / (2 * len(relations))):.1f}%)")
    return relations, pairs


def query_pair(cur, artist, track):
    cur.execute(QUERY_COMPLEX, {"artist": artist, "track": track,
                                "max_results": MAX_RESULTS, "max_releases": MAX_RELEASES})
    return cur.fetchall()


def estimate_cost(cur, pairs):
    probe = random.Random(7).sample(pairs, min(40, len(pairs)))
    rows = []
    for p in tqdm(PG_TRGM_GRID, desc="estimating cost", unit=" threshold"):
        cur.execute("SELECT set_limit(%s);", (p,))
        t0 = time.perf_counter()
        n_rows = sum(len(query_pair(cur, a, t)) for a, t in probe)
        per_query = (time.perf_counter() - t0) / len(probe)
        rows.append({"pg_trgm": p, "s_per_query": per_query,
                     "rows_per_query": n_rows / len(probe),
                     "est_minutes": per_query * len(pairs) / 60})

    cost = pd.DataFrame(rows)
    print(cost.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\nestimated total for the grid: {cost['est_minutes'].sum() / 60:.1f} h")
    print("(much lower if some thresholds are already cached on disk)")
    return cost


def result_path(result_dir, p):
    return result_dir / f"trgm_{int(round(p * 100)):03d}.json.gz"


def collect(cur, pairs, result_dir, p, force=False):
    path = result_path(result_dir, p)
    if path.exists() and not force:
        return path

    cur.execute("SELECT set_limit(%s);", (p,))
    out = {}
    for artist, track in tqdm(pairs, desc=f"pg_trgm={p}", unit=" pair", smoothing=0.01):
        rows = query_pair(cur, artist, track)
        scored = []
        for row in rows:
            db_artist = row[0].get("name", "")
            db_song = row[2].get("name", "")
            a_sim = calculate_similarity(artist, db_artist)
            s_sim = calculate_similarity(track, db_song)
            scored.append((min(a_sim, s_sim), db_artist, db_song, a_sim, s_sim))
        scored.sort(key=lambda x: -x[0])
        out[SEP.join((artist, track))] = {
            "n_db": len(rows),
            "mins": [round(s[0], 4) for s in scored],
            "top": [[s[1], s[2], round(s[3], 4), round(s[4], 4)] for s in scored[:3]],
        }

    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    tmp.replace(path)
    return path


def load_pass(result_dir, p):
    with gzip.open(result_path(result_dir, p), "rt", encoding="utf-8") as f:
        return json.load(f)


def summarize(result_dir, relations, p):
    data = load_pass(result_dir, p)
    keys = list(data)
    kidx = {k: i for i, k in enumerate(keys)}
    T = np.asarray(DIFFLIB_GRID, dtype=float)

    # `mins` is stored in descending order; reversed to ascending for searchsorted.
    counts = np.empty((len(keys), len(T)), dtype=np.int32)
    best = np.empty(len(keys), dtype=float)
    n_db = np.empty(len(keys), dtype=float)
    for i, k in enumerate(keys):
        v = data[k]
        asc = np.asarray(v["mins"][::-1], dtype=float)
        counts[i] = len(asc) - np.searchsorted(asc, T, side="right")
        best[i] = v["mins"][0] if v["mins"] else -1.0
        n_db[i] = v["n_db"]

    def side_index(col):
        return np.fromiter((kidx[SEP.join(r[col])] for r in relations),
                           dtype=np.int64, count=len(relations))

    i_from, i_to = side_index(2), side_index(3)
    i_side = np.concatenate([i_from, i_to])

    best_from, best_to = best[i_from], best[i_to]
    sides = best[i_side]
    side_counts = counts[i_side]
    db_rows = float(n_db[i_side].mean())

    rows = []
    for j, t in enumerate(DIFFLIB_GRID):
        c = side_counts[:, j]
        matched = sides > t
        rows.append({
            "pg_trgm": p,
            "difflib": t,
            "matches": int(c.sum()),
            "sides_with_match": 100 * matched.mean(),
            "relations_ok": 100 * ((best_from > t) & (best_to > t)).mean(),
            # distinct recordings of the same song: catalog density, not error
            "recs_per_side": float(c[matched].mean()) if matched.any() else 0.0,
            "exact": 100 * (sides[matched] >= 0.999).mean() if matched.any() else 0.0,
            "db_rows": db_rows,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over pg_trgm and difflib thresholds for MusicBrainz matching."
    )
    parser.add_argument("--config", type=Path, default=REPO / "config.yaml")
    parser.add_argument("--input", type=Path,
                        default=REPO / "data" / "whosampled-data-samples-only.json")
    parser.add_argument("--result-dir", type=Path,
                        default=REPO / "data" / "grid_search_thresholds_final_samples_only")
    parser.add_argument("--n-keys", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    print(f"grid: {len(PG_TRGM_GRID)} x {len(DIFFLIB_GRID)} = "
          f"{len(PG_TRGM_GRID) * len(DIFFLIB_GRID)} pairs")
    print(f"database passes required: {len(PG_TRGM_GRID)}")

    relations, pairs = load_relations(args.input, args.n_keys, args.seed)

    conn, cur = connect(args.config, WORK_MEM, PG_TRGM_GRID[0], create_indexes=False)
    cost = estimate_cost(cur, pairs)

    missing = [p for p in PG_TRGM_GRID if not result_path(args.result_dir, p).exists()]
    if missing:
        hours = cost[cost.pg_trgm.isin(missing)]["est_minutes"].sum() / 60
        print(f"{len(missing)} thresholds left to collect: {missing}  (~{hours:.1f} h)")
    else:
        print(f"all {len(PG_TRGM_GRID)} thresholds already in {args.result_dir.name} "
              f"-- nothing to collect")

    for p in PG_TRGM_GRID:
        collect(cur, pairs, args.result_dir, p)

    print("\ncollection complete:")
    for p in PG_TRGM_GRID:
        print(f"  pg_trgm={p}  {result_path(args.result_dir, p).stat().st_size / 1e6:6.1f} MB")

    grid = pd.DataFrame([r for p in tqdm(PG_TRGM_GRID, desc="metrics", unit=" threshold")
                         for r in summarize(args.result_dir, relations, p)])
    grid.to_csv(args.result_dir / "grid.csv", index=False)
    print(grid.head(len(DIFFLIB_GRID)).to_string(index=False))


if __name__ == "__main__":
    main()