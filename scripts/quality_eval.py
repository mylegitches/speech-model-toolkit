"""Compare clip measurements (quality_report.py) with what a person kept, and propose
automatic accept rules.

    python scripts/quality_eval.py /data/quality/tonyv4.jsonl [/data/quality/chuckyv2.jsonl ...]

For each check: how well it separates kept from left-out clips, and how many of each
fail at a strict threshold. Then the combined rule: how many clips it accepts, and how
many of those the person also kept (the precision we want at 95% or more).
"""

import json
import sys

import numpy as np

from app.voice import quality

# (name, row -> value, True if higher is better)
CHECKS = [
    ("speaker", lambda r: r["speaker"], True),
    ("halves", lambda r: r["halves"] if r["halves"] is not None else 1.0, True),
    ("words_match", lambda r: r["words_match"], True),
    ("word_prob_min", lambda r: r["word_prob_min"], True),
    ("snr_db", lambda r: r["snr_db"], True),
    ("level_db", lambda r: r["level_db"], True),
    ("edge_db", lambda r: max(r["edge_start_db"], r["edge_end_db"]), False),
    ("holes", lambda r: r["holes"], False),
    ("stoi", lambda r: r["stoi"], True),
    ("pesq", lambda r: r["pesq"], True),
    ("sisdr", lambda r: r["sisdr"], True),
    ("seconds", lambda r: r["seconds"], True),
]


def separation(kept, out):
    a, b = np.asarray(kept), np.asarray(out)
    if not len(a) or not len(b):
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def load(path):
    rows = [json.loads(line) for line in open(path)]
    vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
    confirmed = np.array([r["confirmed"] for r in rows])
    centre = quality.robust_centroid(vectors[confirmed] if confirmed.any() else vectors)
    for r, v in zip(rows, vectors):
        r["speaker"] = float(v @ centre)
    return rows


def report(name, rows):
    kept = [r for r in rows if r["saved"]]
    # left out = the AI confirmed it, the person reviewed the file and didn't keep it
    out = [r for r in rows if r["confirmed"] and not r["saved"]]
    conf = [r for r in rows if r["confirmed"]]
    print(f"\n=== {name}: {len(kept)} kept, {len(out)} left out; "
          f"accepting every confirmed clip: {sum(r['saved'] for r in conf)}/{len(conf)} "
          f"= {sum(r['saved'] for r in conf) / max(1, len(conf)):.1%} kept ===")
    print(f"{'check':14}{'kept median':>12}{'left-out med':>14}{'separation':>12}"
          f"{'  cut at':>10}{'kept lost':>11}{'left-out caught':>17}")
    cuts = {}
    for check, get, higher in CHECKS:
        a = [get(r) for r in kept]; b = [get(r) for r in out]
        s = separation(a, b) if higher else separation([-x for x in a], [-x for x in b])
        # a strict-but-fair cut: lose at most 5% of what the person kept
        cut = np.percentile(a, 5 if higher else 95)
        fails = (lambda x, cut=cut: x < cut) if higher else (lambda x, cut=cut: x > cut)
        cuts[check] = (get, fails)
        print(f"{check:14}{np.median(a):12.3f}{np.median(b) if b else float('nan'):14.3f}{s:12.2f}"
              f"{cut:10.3f}{sum(map(fails, a)):6}/{len(a):<4}{sum(map(fails, b)):8}/{len(b)}")
    return kept, out, conf, cuts


def rule(rows, cuts, use):
    """Accept: confirmed by the AI and passing every check in `use`."""
    return [r for r in rows if r["confirmed"] and not any(cuts[c][1](cuts[c][0](r)) for c in use)]


def main():
    for path in sys.argv[1:]:
        rows = load(path)
        kept, out, conf, cuts = report(path.rsplit("/", 1)[-1], rows)
        print("\ncombined rules (confirmed + all listed checks pass):")
        for use in (
            ["speaker"], ["speaker", "halves"], ["speaker", "halves", "words_match"],
            ["speaker", "halves", "words_match", "word_prob_min"],
            ["speaker", "halves", "words_match", "word_prob_min", "snr_db", "edge_db", "holes"],
            ["speaker", "halves", "words_match", "word_prob_min", "snr_db", "edge_db", "holes", "stoi"],
        ):
            acc = rule(rows, cuts, use)
            good = sum(r["saved"] for r in acc)
            print(f"  {'+'.join(use):70} accepts {len(acc):5}  kept {good / max(1, len(acc)):6.1%}"
                  f"  left-out let through {len(acc) - good:3}/{len(out)}"
                  f"  median PESQ {np.median([r['pesq'] for r in acc]) if acc else 0:.2f}")


if __name__ == "__main__":
    main()
