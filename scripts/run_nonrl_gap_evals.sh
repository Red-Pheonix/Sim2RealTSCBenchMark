#!/bin/bash
# ============================================================================
# Non-RL controller baselines (fixedtime, maxpressure) across every gap setting.
#
# These controllers carry no weights, so every mitigation in the benchmark is
# inapplicable by construction -- they all either retrain a policy (dr / da / gat /
# ugat / jlgat, vae / darla / atc / lusr / recon, reward_inference / morl_grid /
# dynamic_reward_shaping) or need a Q-function (delayed_q / oblivious_q / prlight).
# The only meaningful run per cell is therefore the zero-adaptation one, which each
# task spells `direct_transfer`. Sim side = cityflow, real side = sumo + the setting.
#
# 33 gap settings x their network roster = 300 runs per agent, 600 total, plus 20
# cityflow engine baselines (the sim side of gap = real - sim):
#   observations  16 settings x 10 nets = 160
#   transitions    4 settings x 10 nets =  40
#   action delay   4 settings x 10 nets =  40
#   action PT      5 settings x  4 nets =  20   (only 4 nets ship transition tables)
#   rewards        4 settings x 10 nets =  40
#
# Run dirs and prefixes deliberately mirror the RL runs, so the analysis notebooks
# pick these up by adding 'fixedtime'/'maxpressure' to their AGENTS list -- nothing
# else about the parsing changes.
#
# Usage: bash scripts/run_nonrl_gap_evals.sh              # everything
#        AGENTS=maxpressure TASKS=obs bash scripts/run_nonrl_gap_evals.sh
#        JOBS=4 bash scripts/run_nonrl_gap_evals.sh       # cap parallelism
# ============================================================================
set -u
cd "$(dirname "$0")/.."

AGENTS=${AGENTS:-"fixedtime maxpressure"}
TASKS=${TASKS:-"engine obs trans delay pt rewards"}
JOBS=${JOBS:-8}

NETWORKS=${NETWORKS:-"tempe_1x1 bullhead_1 cologne1 ingolstadt1 hz1x1 tempe_16 bullhead_3 cologne3 ingolstadt7 hz4x4"}
PT_NETWORKS=${PT_NETWORKS:-"tempe_1x1 bullhead_1 tempe_16 bullhead_3"}

OBS_SETTINGS="noise3 noise5 noise10 noise20 dz10 dz20 dz50 dz100 sensor5 sensor20 sensor50 sensor70 combine1 combine2 combine3 combine4"
TRANS_SETTINGS="setting1 setting2 setting3 setting4"
DELAY_SETTINGS="setting1 setting2 setting3 setting4"
PT_SETTINGS="cyclic flexible barrier_leading_fixed barrier_lagging_fixed barrier_leading_lagging_fixed"
REWARD_SETTINGS="efficiency_aligned emission_heavy fairness_heavy physical_safety_heavy"

RUNLOG=${RUNLOG:-"logs/nonrl_runs"}
mkdir -p "$RUNLOG"

# --- one unit of work -------------------------------------------------------
# Each line fed to xargs is: <kind> <agent> <net> <setting>
run_one() {
  kind=$1; agent=$2; net=$3; setting=$4
  tag="${kind}_${agent}_${net}_${setting}"
  out="$RUNLOG/$tag.out"

  case $kind in
    engine)
      # cityflow eval of the controller = the sim side of gap = real - sim
      prefix="cityflow_baseline"
      cmd="python run.py -t tsc -a $agent -w cityflow -n $net --prefix $prefix"
      src="data/output_data/tsc/cityflow_${agent}/${net}/${prefix}/logger"
      dst="logs/engine_baselines/${agent}/${net}/cityflow_direct_transfer"
      ;;
    obs)
      prefix="baseline_v1_direct_transfer_${setting}_s0"
      cmd="python run_s2r_observations.py -a $agent -n $net --real_world sumo --obs_model direct_transfer --real_setting $setting --prefix $prefix"
      src="data/output_data/sim2real_observations/cityflow_${agent}/${net}/${prefix}/logger"
      dst="logs/sim2real_observations/cityflow_${agent}/${net}/${prefix}/logger"
      ;;
    trans)
      prefix="baseline_v1_direct_transfer_${setting}_s0"
      cmd="python run_s2r.py -a $agent -n $net -gt direct_transfer --real_setting $setting --prefix $prefix"
      src="data/output_data/sim2real_transitions/cityflow_${agent}/${net}/${prefix}/logger"
      dst="logs/sim2real_transitions/cityflow_${agent}/${net}/${prefix}/logger"
      ;;
    delay)
      prefix="action_delay_optimized_s0_direct_transfer_${setting}_s0"
      cmd="python run_s2r_actions.py -a $agent -n $net --act_model direct_transfer --real_setting $setting --prefix $prefix"
      src="data/output_data/sim2real_actions/cityflow_${agent}/${net}/${prefix}/logger"
      dst="logs/sim2real_actions/cityflow_${agent}/${net}/${prefix}/logger"
      ;;
    pt)
      prefix="action_pt_s0_direct_transfer_${setting}_s0"
      cmd="python run_s2r_actions.py -a $agent -n $net --act_model direct_transfer --real_setting $setting --prefix $prefix"
      src="data/output_data/sim2real_actions/cityflow_${agent}/${net}/${prefix}/logger"
      dst="logs/sim2real_actions/cityflow_${agent}/${net}/${prefix}/logger"
      ;;
    rewards)
      prefix="baseline_rewards_direct_transfer_${setting}_s0"
      cmd="python run_s2r_rewards.py -a $agent -n $net --reward_model direct_transfer --real_setting $setting --prefix $prefix"
      src="data/output_data/sim2real_rewards/cityflow_${agent}/${net}/${prefix}/logger"
      # rewards notebook reads <agent>/<net>/<method>/<setting>/, not a prefix dir
      dst="logs/sim2real_rewards/${agent}/${net}/direct_transfer/${setting}"
      ;;
    *) echo "unknown kind: $kind" >&2; return 1 ;;
  esac

  if ! $cmd > "$out" 2>&1; then
    echo "FAILED $tag (see $out)"
    return 1
  fi
  if ! ls "$src"/*_DTL.log > /dev/null 2>&1; then
    echo "NO_DTL $tag (ran clean but produced no DTL; see $out)"
    return 1
  fi
  mkdir -p "$dst"
  cp "$src"/*.log "$dst/"
  echo "ok $tag"
}
export -f run_one
export RUNLOG

# --- build the work list ----------------------------------------------------
worklist=$(mktemp)
for agent in $AGENTS; do
  for task in $TASKS; do
    case $task in
      engine)  for n in $NETWORKS; do echo "engine $agent $n none"; done ;;
      obs)     for n in $NETWORKS;    do for s in $OBS_SETTINGS;    do echo "obs $agent $n $s"; done; done ;;
      trans)   for n in $NETWORKS;    do for s in $TRANS_SETTINGS;  do echo "trans $agent $n $s"; done; done ;;
      delay)   for n in $NETWORKS;    do for s in $DELAY_SETTINGS;  do echo "delay $agent $n $s"; done; done ;;
      pt)      for n in $PT_NETWORKS; do for s in $PT_SETTINGS;     do echo "pt $agent $n $s"; done; done ;;
      rewards) for n in $NETWORKS;    do for s in $REWARD_SETTINGS; do echo "rewards $agent $n $s"; done; done ;;
      *) echo "unknown task: $task" >&2; exit 1 ;;
    esac
  done
done > "$worklist"

total=$(wc -l < "$worklist")
echo "=== $total runs, $JOBS at a time; per-run stdout under $RUNLOG/ ==="
date

xargs -a "$worklist" -P "$JOBS" -L 1 bash -c 'run_one $0 $1 $2 $3' \
  | tee "$RUNLOG/summary.txt"

echo
date
echo "=== summary ==="
ok=$(grep -c '^ok ' "$RUNLOG/summary.txt")
bad=$(grep -cE '^(FAILED|NO_DTL) ' "$RUNLOG/summary.txt")
echo "ok=$ok failed=$bad of $total"
grep -E '^(FAILED|NO_DTL) ' "$RUNLOG/summary.txt"
rm -f "$worklist"
[ "$bad" -eq 0 ]
