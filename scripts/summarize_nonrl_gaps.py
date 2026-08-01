"""Summarize the non-RL controller baselines across every gap setting.

Reads the DTLs written by scripts/run_nonrl_gap_evals.sh and reports, per gap axis and
agent, the real-side (sumo) travel time and the gap against that controller's own
cityflow run -- gap = REAL_TEST - cityflow_baseline, the same definition the RL tables
use. Rewards are reported as R_real (reward units), never as a gap.

Usage: python scripts/summarize_nonrl_gaps.py [--csv tables/nonrl_gaps.csv]
"""
import argparse
import statistics
from pathlib import Path

AGENTS = ["fixedtime", "maxpressure"]
NETWORKS = ["tempe_1x1", "bullhead_1", "cologne1", "ingolstadt1", "hz1x1",
            "tempe_16", "bullhead_3", "cologne3", "ingolstadt7", "hz4x4"]
PT_NETWORKS = ["tempe_1x1", "bullhead_1", "tempe_16", "bullhead_3"]

AXES = {
    "observations": (
        Path("logs/sim2real_observations"), NETWORKS,
        ["noise3", "noise5", "noise10", "noise20", "dz10", "dz20", "dz50", "dz100",
         "sensor5", "sensor20", "sensor50", "sensor70",
         "combine1", "combine2", "combine3", "combine4"],
        "baseline_v1_direct_transfer_{s}_s0",
    ),
    "transitions": (
        Path("logs/sim2real_transitions"), NETWORKS,
        ["setting1", "setting2", "setting3", "setting4"],
        "baseline_v1_direct_transfer_{s}_s0",
    ),
    "action_delay": (
        Path("logs/sim2real_actions"), NETWORKS,
        ["setting1", "setting2", "setting3", "setting4"],
        "action_delay_optimized_s0_direct_transfer_{s}_s0",
    ),
    "action_pt": (
        Path("logs/sim2real_actions"), PT_NETWORKS,
        ["cyclic", "flexible", "barrier_leading_fixed", "barrier_lagging_fixed",
         "barrier_leading_lagging_fixed"],
        "action_pt_s0_direct_transfer_{s}_s0",
    ),
}
REWARD_SETTINGS = ["efficiency_aligned", "emission_heavy", "fairness_heavy",
                   "physical_safety_heavy"]


def newest_dtl(directory):
    files = sorted(Path(directory).glob("*_DTL.log"))
    return files[-1] if files else None


def rows(path, mode):
    out = []
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) > 3 and parts[1] == mode:
            out.append(parts)
    return out


def real_test_tt(directory):
    """travel time of the single REAL_TEST row (None when the run is missing)."""
    row = real_test_row(directory)
    return row["travel_time"] if row else None


def real_test_row(directory, pt=False):
    """The single REAL_TEST row as a dict, in the same metric set the RL tables carry.
    Non-RL controllers are eval-only, so there is exactly one row and no checkpoint
    selection -- `selection_count` is 1 by construction, not a top-K mean.

    `pt` gates the two trailing compliance columns: only a phase-transition DTL puts
    violation/forceoff rates at 10/11. The rewards DTL is a WIDER schema whose column
    10 is true_reward, so reading them unconditionally silently files R_real as a
    force-off rate."""
    dtl = newest_dtl(directory)
    if dtl is None:
        return None
    got = rows(dtl, "REAL_TEST")
    if not got:
        return None
    p = got[-1]

    def num(i):
        try:
            return float(p[i])
        except (IndexError, ValueError):
            return ""

    out = {"real_mode": "REAL_TEST", "selection_count": 1, "travel_time": num(3),
           "reward": num(5), "queue": num(6), "delay": num(7), "throughput": num(8),
           "log_file": Path(dtl).name}
    # phase-transition runs carry two extra columns (per-decision compliance rates)
    if pt and len(p) > 10:
        out["violation_rate"], out["forceoff_rate"] = num(9), num(10)
    return out


def cityflow_baseline_row(agent, net):
    """FINAL_TEST metrics of the controller's own cityflow run -- the sim side of
    gap = real - sim. Same reference role the pretrained tsc eval plays for the RL
    tables, except a non-learned controller needs no pretrained checkpoint."""
    d = Path("logs/engine_baselines") / agent / net / "cityflow_direct_transfer"
    dtl = newest_dtl(d)
    if dtl is None:
        return None
    got = rows(dtl, "FINAL_TEST")
    if not got:
        return None
    p = got[-1]

    def num(i):
        try:
            return float(p[i])
        except (IndexError, ValueError):
            return ""

    return {"cityflow_att": num(3), "cityflow_queue": num(6),
            "cityflow_delay": num(7), "cityflow_throughput": num(8)}


def cityflow_baseline(agent, net):
    row = cityflow_baseline_row(agent, net)
    return row["cityflow_att"] if row else None


def pt_safety(directory):
    """(violation_rate, forceoff_rate): cols 10/11 of a phase-transition DTL row."""
    dtl = newest_dtl(directory)
    if dtl is None:
        return None, None
    for parts in rows(dtl, "REAL_TEST"):
        if len(parts) > 10:
            return float(parts[9]), float(parts[10])
    return None, None


def reward_real(agent, net, setting):
    d = Path("logs/sim2real_rewards") / agent / net / "direct_transfer" / setting
    dtl = newest_dtl(d)
    if dtl is None:
        return None
    got = rows(dtl, "REAL_TEST")
    return float(got[-1][10]) if got and len(got[-1]) > 10 else None


def make_record(axis, agent, network, setting, logger_dir, r_real=None):
    """One appendix-table row, in the same long-form schema as the RL per-axis tables
    (tables/observation_table.csv &c): identity, real-side metrics, the cityflow
    reference, and gap = real - cityflow for each metric. `method` is always
    direct_transfer -- a weightless controller has no mitigation to apply."""
    rec = {"axis": axis, "agent": agent, "network": network, "setting": setting,
           "method": "direct_transfer"}
    real = (real_test_row(logger_dir, pt=(axis == "action_pt"))
            if logger_dir is not None else None)
    if real:
        rec.update(real)
    base = cityflow_baseline_row(agent, network)
    if base:
        rec.update(base)
    # positive gap is worse for att/queue/delay; NEGATIVE is worse for throughput
    for metric, real_key, base_key in (("att", "travel_time", "cityflow_att"),
                                       ("queue", "queue", "cityflow_queue"),
                                       ("delay", "delay", "cityflow_delay"),
                                       ("throughput", "throughput",
                                        "cityflow_throughput")):
        a, b = rec.get(real_key, ""), rec.get(base_key, "")
        key = "gap_vs_cityflow" if metric == "att" else f"gap_{metric}"
        rec[key] = round(a - b, 4) if a != "" and b != "" else ""
    if r_real is not None:
        rec["r_real"] = r_real
    return rec


CSV_COLUMNS = ["axis", "agent", "network", "setting", "method", "real_mode",
               "selection_count", "travel_time", "reward", "queue", "delay",
               "throughput", "violation_rate", "forceoff_rate", "r_real",
               "cityflow_att", "gap_vs_cityflow", "cityflow_queue", "gap_queue",
               "cityflow_delay", "gap_delay", "cityflow_throughput", "gap_throughput",
               "log_file"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    records = []
    base = {(a, n): cityflow_baseline(a, n) for a in AGENTS for n in NETWORKS}

    print("=" * 78)
    print("CITYFLOW BASELINE (sim side of gap = real - sim), avg travel time")
    print("=" * 78)
    print(f"{'network':<14}" + "".join(f"{a:>14}" for a in AGENTS))
    for n in NETWORKS:
        print(f"{n:<14}" + "".join(
            f"{base[(a, n)]:>14.1f}" if base[(a, n)] is not None else f"{'--':>14}"
            for a in AGENTS))

    for axis, (root, nets, settings, prefix) in AXES.items():
        print()
        print("=" * 78)
        print(f"{axis.upper()}  --  gap = sumo REAL_TEST - cityflow baseline (seconds)")
        print("=" * 78)
        print(f"{'setting':<32}" + "".join(f"{a:>22}" for a in AGENTS))
        print(f"{'':<32}" + "".join(f"{'median gap (n)':>22}" for a in AGENTS))
        for s in settings:
            cells = []
            for a in AGENTS:
                # Fixed-time is OPEN-LOOP: a real cabinet executes its plan on its
                # own clock, so actuation delay does not apply to it. Piping it
                # through the decision-time delay queue measures an interface
                # artifact instead (it re-proposes switches off the stale observed
                # phase while its command is in flight, and the queue's chained
                # releases compound the backlog) -- the resulting travel times are
                # huge and non-monotonic in delta. Excluded from the table; its
                # delay-gap row is its plain direct-transfer number by construction.
                if axis == "action_delay" and a == "fixedtime":
                    cells.append(f"{'n/a (open-loop)':>22}")
                    continue
                gaps = []
                for n in nets:
                    tt = real_test_tt(root / f"cityflow_{a}" / n / prefix.format(s=s)
                                      / "logger")
                    b = base[(a, n)]
                    if tt is None or b is None:
                        continue
                    gaps.append(tt - b)
                    records.append(make_record(axis, a, n, s,
                                               root / f"cityflow_{a}" / n
                                               / prefix.format(s=s) / "logger"))
                cells.append(f"{statistics.median(gaps):>16.1f} ({len(gaps):>2})"
                             if gaps else f"{'-- (0)':>22}")
            print(f"{s:<32}" + "".join(cells))

    # phase-transition safety: the number that actually characterises this axis
    print()
    print("=" * 78)
    print("ACTION_PT  --  controller compliance (per-decision rates, median over nets)")
    print("=" * 78)
    root, nets, settings, prefix = AXES["action_pt"]
    print(f"{'setting':<32}" + "".join(f"{a:>22}" for a in AGENTS))
    print(f"{'':<32}" + "".join(f"{'violation / forceoff':>22}" for a in AGENTS))
    for s in settings:
        cells = []
        for a in AGENTS:
            vrs, frs = [], []
            for n in nets:
                vr, fr = pt_safety(root / f"cityflow_{a}" / n / prefix.format(s=s)
                                   / "logger")
                if vr is not None:
                    vrs.append(vr)
                    frs.append(fr)
            cells.append(
                f"{statistics.median(vrs):>10.2f} /{statistics.median(frs):>10.2f}"
                if vrs else f"{'--':>22}")
        print(f"{s:<32}" + "".join(cells))

    print()
    print("=" * 78)
    print("REWARDS  --  R_real of a reward-agnostic controller (reward units, NOT a gap)")
    print("=" * 78)
    print(f"{'setting':<32}" + "".join(f"{a:>22}" for a in AGENTS))
    for s in REWARD_SETTINGS:
        cells = []
        for a in AGENTS:
            vals = [v for n in NETWORKS if (v := reward_real(a, n, s)) is not None]
            cells.append(f"{statistics.median(vals):>16.3f} ({len(vals):>2})"
                         if vals else f"{'-- (0)':>22}")
            for n in NETWORKS:
                v = reward_real(a, n, s)
                if v is not None:
                    records.append(make_record(
                        "rewards", a, n, s,
                        Path("logs/sim2real_rewards") / a / n / "direct_transfer" / s,
                        r_real=v))
        print(f"{s:<32}" + "".join(cells))

    # fixedtime is obs-blind by construction: assert the flatline rather than claim it
    print()
    print("=" * 78)
    print("INVARIANCE CHECK  --  distinct REAL_TEST values across the 16 obs settings")
    print("(fixedtime never reads a detector, so 1 distinct value per network is the")
    print(" expected structural result; maxpressure should vary)")
    print("=" * 78)
    root, nets, settings, prefix = AXES["observations"]
    print(f"{'network':<14}" + "".join(f"{a:>14}" for a in AGENTS))
    for n in nets:
        cells = []
        for a in AGENTS:
            vals = {real_test_tt(root / f"cityflow_{a}" / n / prefix.format(s=s)
                                 / "logger") for s in settings}
            vals.discard(None)
            cells.append(f"{len(vals):>14}")
        print(f"{n:<14}" + "".join(cells))

    if args.csv:
        import csv
        keys = CSV_COLUMNS
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in records:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"\nwrote {len(records)} rows to {args.csv}")


if __name__ == "__main__":
    main()
