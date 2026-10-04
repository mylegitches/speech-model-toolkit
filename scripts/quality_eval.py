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


def refresh_labels(rows, voice_dir):
    """The person's decisions as they are now (the review may have changed since measuring)."""
    from pathlib import Path
    takes = {}
    out = []
    for r in rows:
        if r["take"] not in takes:
            f = Path(voice_dir) / "freeform" / r["take"] / "take.json"
            takes[r["take"]] = json.loads(f.read_text())["segments"] if f.is_file() else None
        segs = takes[r["take"]]
        if segs is None or r["index"] >= len(segs):
            continue
        s = segs[r["index"]]
        r["saved"], r["confirmed"] = bool(s.get("saved")), bool(s.get("confirmed"))
        out.append(r)
    return out


def scores(rows, accepted):
    kept_all = sum(r["saved"] for r in rows)
    good = sum(r["saved"] for r in accepted)
    return good / max(1, len(accepted)), good / max(1, kept_all), len(accepted)


def tune(rows, base, beta=0.5, levels=(0, 1, 2, 5, 10, 15, 20, 30)):
    """Per check, how strict (a percentile of the kept clips' values) to maximise F-beta:
    beta 0.5 counts a wrongly accepted clip twice as bad as a good one missed."""
    kept = [r for r in rows if r["saved"]]
    cut_at = {}
    for check, get, higher in CHECKS:
        values = [get(r) for r in kept]
        cut_at[check] = {lv: (np.percentile(values, lv if higher else 100 - lv) if lv else None) for lv in levels}

    def accept(choice, pool):
        res = []
        for r in pool:
            if base == "confirmed" and not r["confirmed"]:
                continue
            ok = True
            for check, get, higher in CHECKS:
                cut = cut_at[check][choice[check]]
                if cut is not None and ((get(r) < cut) if higher else (get(r) > cut)):
                    ok = False
                    break
            if ok:
                res.append(r)
        return res

    def fbeta(choice):
        p, rcl, _ = scores(rows, accept(choice, rows))
        return (1 + beta ** 2) * p * rcl / max(1e-9, beta ** 2 * p + rcl)

    choice = {c: 0 for c, _, _ in CHECKS}
    best = fbeta(choice)
    for _ in range(4):  # coordinate ascent
        improved = False
        for check, _, _ in CHECKS:
            for lv in levels:
                trial = {**choice, check: lv}
                f = fbeta(trial)
                if f > best + 1e-6:
                    best, choice, improved = f, trial, True
        if not improved:
            break
    return choice, cut_at, accept


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    voices_root = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--voices=")), None)
    for path in args:
        rows = load(path)
        if voices_root:
            name = path.rsplit("/", 1)[-1].removesuffix(".jsonl")
            rows = refresh_labels(rows, f"{voices_root}/{name}")
        if "--tune" in sys.argv:
            files = sorted({r["take"] for r in rows})
            half_a = set(files[::2])
            train = [r for r in rows if r["take"] in half_a]
            test = [r for r in rows if r["take"] not in half_a]
            kept_n, all_n = sum(r["saved"] for r in rows), len(rows)
            conf = [r for r in rows if r["confirmed"]]
            print(f"\n=== {path.rsplit('/', 1)[-1]}: {all_n} clips measured, {kept_n} kept by you "
                  f"({kept_n / all_n:.0%}); AI-confirmed: {len(conf)}, of which you kept "
                  f"{sum(r['saved'] for r in conf) / max(1, len(conf)):.0%} ===")
            for base in ("confirmed", "all"):
                choice, cut_at, accept = tune(train, base)
                on = {c: lv for c, lv in choice.items() if lv}
                p_tr, r_tr, n_tr = scores(train, accept(choice, train))
                p_te, r_te, n_te = scores(test, accept(choice, test))
                print(f"\n  start from {'AI-confirmed clips' if base == 'confirmed' else 'every clip'}; tuned on half the episodes:")
                print(f"    rule: " + ", ".join(f"{c} (drop the worst {lv}%)" for c, lv in on.items()) if on else "    rule: no extra checks")
                print(f"    tuning half:  accepts {n_tr:4}, you'd keep {p_tr:6.1%}, finds {r_tr:6.1%} of your kept clips")
                print(f"    test half:    accepts {n_te:4}, you'd keep {p_te:6.1%}, finds {r_te:6.1%} of your kept clips")
            continue
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
