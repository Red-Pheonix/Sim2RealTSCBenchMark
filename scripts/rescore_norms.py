"""Norm-robustness check for the reward gap (fix-plan Task 8a).

Rescore every reward-gap run's final R_real under PERTURBED per-component norms, WITHOUT
rerunning anything, and report whether the method ranking (or any method-vs-naive sign)
changes. If all settings are STABLE, the hand-picked `component_norm` in base.yml is not
driving conclusions and this table is the appendix robustness evidence. If any setting is
AFFECTED, switch to data-driven norms (fix-plan Task 8b).

How it works. The DTL `components` column stores per-decision terms
    t_i = w*_i * phi_i / n_i   (the eval scorer's per-component contribution)
and the logged R_real = -sum_i t_i. Under an alternative norm n'_i, only the divisor
changes, so
    t'_i = t_i * n_i / n'_i ,   R'_real = -sum_i t'_i
-- pure arithmetic on the log, no simulator. For methods that SELECT a policy by real
R_real (reward_inference / morl_grid / dynamic_reward_shaping), we RE-SELECT the argmax
candidate under the perturbed norm (not just rescore the original winner), because a
different norm could pick a different checkpoint.

CAVEAT (first-order). This rescores the EVAL + SELECTION channels only. Training rewards
also used the norm, so a policy trained under n' might behave differently -- STABLE here
means "no evidence the norm choice flips conclusions", not a proof. Treat AFFECTED as a
firm signal to move to data-driven norms.

Usage:
    python scripts/rescore_norms.py                    # default glob under data/output_data
    python scripts/rescore_norms.py --glob 'raw_results/**/*_DTL.log'
    python scripts/rescore_norms.py --factor 2.0       # perturbation multiplier (and 1/f)
"""

import argparse
import glob as globmod
import os
from collections import defaultdict

# base.yml component_norm (authoritative; hasn't changed per network).
COMPONENT_NORM = {
    "queue": 10.0, "delay": 1.0, "waiting": 100.0, "pressure": 10.0, "switches": 1.0,
    "fairness": 50.0, "emission": 10.0, "fuel": 1.0, "emergency_stops": 1.0, "ssm_conflicts": 10.0,
    "collisions": 1.0, "safety": 1.0,
}

KNOWN_METHODS = [  # longest-first so e.g. reward_shaping matches before shaping
    "dynamic_reward_shaping", "reward_inference", "reward_shaping", "reward_random",
    "morl_grid", "pt_naive", "shield", "naive",
]
# Old names kept so historical logs still parse; new canonical names are
# REAL_TRAIN (budget-counted candidate rollouts) and REAL_TEST (scoring evals).
EVAL_MODES = {"TEST_REAL", "FINAL_TEST_REAL", "TRANSFER_REAL", "REAL_TRAIN", "REAL_TEST"}


def parse_components(cell):
    """`queue=0.7557;emission=5.4509` -> {'queue': 0.7557, 'emission': 5.4509}."""
    out = {}
    for part in cell.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


def method_of(exp_name):
    for m in KNOWN_METHODS:
        if exp_name.endswith("_" + m) or exp_name == m:
            return m
    return None


def network_agent_of(path):
    """Path: .../sim2real_rewards/{world}_{agent}/{network}/{prefix}/logger/file.
    Returns (network, agent) or (None, None)."""
    logger_dir = os.path.dirname(path)
    prefix_dir = os.path.dirname(logger_dir)
    network_dir = os.path.dirname(prefix_dir)
    world_agent_dir = os.path.dirname(network_dir)
    network = os.path.basename(network_dir)
    world_agent = os.path.basename(world_agent_dir)
    agent = world_agent.split("_", 1)[1] if "_" in world_agent else None
    return network, agent


def setting_of(exp_name, network, agent, method):
    """exp_name = {network}_{setting}_{agent}_{method}; slice out the setting."""
    if not (network and agent and method):
        return None
    prefix = network + "_"
    suffix = "_" + agent + "_" + method
    if exp_name.startswith(prefix) and exp_name.endswith(suffix):
        return exp_name[len(prefix):len(exp_name) - len(suffix)]
    return None


def rescore(terms, norm_override):
    """R'_real under perturbed norms: R' = -sum_i t_i * n_i / n'_i."""
    total = 0.0
    for c, t in terms.items():
        n = COMPONENT_NORM.get(c, 1.0)
        n2 = norm_override.get(c, n)
        total += t * (n / n2)
    return -total


def candidate_rows(method, rows):
    """The rows among which the DEPLOYED policy is (re)selected by real R_real.
    rows = list of dicts with keys detail, terms, order."""
    def has(prefixes):
        return [r for r in rows if any(r["detail"].startswith(p) for p in prefixes)]

    if method == "reward_inference":
        c = has(["final;"])
    elif method == "morl_grid":
        c = has(["grid;", "refine;"])
    elif method == "dynamic_reward_shaping":
        c = has(["init;", "bo;"])
    else:
        # no selection: the deployed policy is fixed. Use the last eval row (file
        # order) -- the final scoring eval, written after all training-curve rows.
        return [rows[-1]] if rows else []
    # Selection methods: if candidate tags are missing (older logs), fall back to all
    # eval rows so we still produce a number.
    return c or ([rows[-1]] if rows else [])


def deployed_score(method, rows, norm_override):
    """Rescored R_real of the deployed policy under `norm_override` (argmax re-select
    for selection methods; the fixed final row otherwise)."""
    cands = candidate_rows(method, rows)
    if not cands:
        return None
    return max(rescore(r["terms"], norm_override) for r in cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="data/output_data/sim2real_rewards/**/*_DTL.log",
        help="glob for DTL logs (recursive)",
    )
    ap.add_argument("--factor", type=float, default=2.0,
                    help="per-component norm perturbation multiplier (and its reciprocal)")
    ap.add_argument("--networks", default="tempe_1x1,cologne3",
                    help="comma-separated networks to include (empty = all)")
    args = ap.parse_args()
    keep_networks = {n for n in args.networks.split(",") if n} if args.networks else None

    files = globmod.glob(args.glob, recursive=True)
    if not files:
        print(f"no DTL logs matched: {args.glob}")
        return

    # run key -> {network, setting, method, rows:[...]}
    runs = {}
    skipped = []
    for path in files:
        network, agent = network_agent_of(path)
        if keep_networks is not None and network not in keep_networks:
            continue
        try:
            with open(path) as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        if not lines:
            continue
        start = 1 if lines[0].startswith("exp_name") else 0
        for order, line in enumerate(lines[start:]):
            cols = line.split("\t")
            if len(cols) < 12:
                continue
            exp_name, mode = cols[0], cols[1]
            if mode not in EVAL_MODES:
                continue
            method = method_of(exp_name)
            setting = setting_of(exp_name, network, agent, method)
            if method is None or setting is None:
                skipped.append(exp_name)
                continue
            if setting in ("smoke", "safety_smoke"):
                continue  # crash-test settings, not scientific runs
            terms = parse_components(cols[11])
            detail = cols[12] if len(cols) > 12 else ""
            key = (network, setting, agent, method, path)
            run = runs.setdefault(
                key, {"network": network, "setting": setting, "method": method,
                      "rows": []}
            )
            run["rows"].append({"detail": detail, "terms": terms, "order": order})

    # Collapse to one run per (network, setting, method) -- keep the run with the most
    # eval rows (the full run) if a combo appears in several files.
    best = {}
    for key, run in runs.items():
        combo = (run["network"], run["setting"], run["method"])
        if combo not in best or len(run["rows"]) > len(best[combo]["rows"]):
            best[combo] = run

    # Which components actually carry weight anywhere (perturb only these).
    active = set()
    for run in best.values():
        for r in run["rows"]:
            active.update(k for k, v in r["terms"].items() if abs(v) > 1e-9)

    # Group by (network, setting).
    groups = defaultdict(dict)  # (network,setting) -> method -> run
    for (network, setting, method), run in best.items():
        groups[(network, setting)][method] = run

    f = args.factor
    perturbations = [("baseline", {})]
    for c in sorted(active):
        perturbations.append((f"{c}x{f:g}", {c: COMPONENT_NORM.get(c, 1.0) * f}))
        perturbations.append((f"{c}x{1 / f:g}", {c: COMPONENT_NORM.get(c, 1.0) / f}))

    print(f"# rescore_norms: {len(files)} files, {len(best)} runs, "
          f"factor +/-{f:g}, active components {sorted(active)}\n")

    any_affected = False
    for (network, setting), methods in sorted(groups.items()):
        base_scores = {
            m: deployed_score(m, run["rows"], {}) for m, run in methods.items()
        }
        base_scores = {m: s for m, s in base_scores.items() if s is not None}
        if not base_scores:
            continue
        base_rank = sorted(base_scores, key=lambda m: base_scores[m], reverse=True)
        naive_base = base_scores.get("naive")

        affected_labels = []
        for label, override in perturbations:
            if label == "baseline":
                continue
            scores = {
                m: deployed_score(m, run["rows"], override)
                for m, run in methods.items()
            }
            scores = {m: s for m, s in scores.items() if s is not None}
            rank = sorted(scores, key=lambda m: scores[m], reverse=True)
            rank_changed = rank != base_rank
            sign_changed = False
            if naive_base is not None and "naive" in scores:
                for m in scores:
                    if m == "naive":
                        continue
                    if (m in base_scores
                            and (base_scores[m] - naive_base > 0)
                            != (scores[m] - scores["naive"] > 0)):
                        sign_changed = True
                        break
            if rank_changed or sign_changed:
                affected_labels.append(label)

        status = "AFFECTED" if affected_labels else "STABLE"
        any_affected = any_affected or bool(affected_labels)
        print(f"## {network} / {setting}   [{status}]")
        print("   baseline ranking (best R_real first): "
              + ", ".join(f"{m}={base_scores[m]:.3f}" for m in base_rank))
        if affected_labels:
            print("   flips under: " + ", ".join(affected_labels))
        print()

    print("=" * 60)
    print("OVERALL:", "AFFECTED -> use data-driven norms (Task 8b)"
          if any_affected else "STABLE -> keep hand norms; this is the appendix table")
    if skipped:
        uniq = sorted(set(skipped))
        print(f"\nskipped {len(uniq)} unparseable exp_name(s) (first few): {uniq[:5]}")


if __name__ == "__main__":
    main()
