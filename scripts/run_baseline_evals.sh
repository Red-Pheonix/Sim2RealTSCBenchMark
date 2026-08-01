#!/bin/bash
# ============================================================================
# Baseline evals of the committed pretrained tsc policies (pretrained/tsc/*),
# in BOTH engines, for the reward-gap (and any other gap) reference numbers:
#   cityflow -> the "sim" side of gap = real - sim   (ATT gap tables)
#   sumo     -> the zero-adaptation "real" reference; must byte-match the
#               reward-gap direct_transfer REAL_TEST rows (sumo is
#               deterministic; verified 2026-07-13: dqn/tempe_1x1 = 151.2112)
#
# Eval-only needs model.train_model=False + model.load_model=True. presslight.yml
# already ships that way; dqn.yml ships train-mode, so this script flips it and
# restores the original on exit (trap), even on Ctrl-C.
#
# DTLs land in data/output_data/tsc/<world>_<agent>/<net>/<PREFIX>/logger/ and
# are copied to logs/sim2real_rewards/<agent>/<net>/<world>_direct_transfer/.
#
# Usage: bash scripts/run_baseline_evals.sh            # both agents, both worlds
#        AGENTS=dqn WORLDS=sumo bash scripts/run_baseline_evals.sh
# ============================================================================
set -u
cd "$(dirname "$0")/.."

AGENTS=${AGENTS:-"dqn presslight"}
WORLDS=${WORLDS:-"cityflow sumo"}
NETWORKS=${NETWORKS:-"tempe_1x1 bullhead_1 cologne1 ingolstadt1 hz1x1 tempe_16 bullhead_3 cologne3 ingolstadt7 hz4x4"}
LOG_DEST=${LOG_DEST:-"logs/sim2real_rewards"}

DQN_YML=configs/tsc/dqn.yml
cp "$DQN_YML" "$DQN_YML.bak"
trap 'mv "$DQN_YML.bak" "$DQN_YML"' EXIT
python3 - <<'EOF'
import pathlib
p = pathlib.Path('configs/tsc/dqn.yml')
t = p.read_text()
if '  load_model: True' not in t:
    t = t.replace('  train_model: True\n  test_model: True',
                  '  train_model: False\n  test_model: True\n  load_model: True')
    p.write_text(t)
EOF

fail=0
for world in $WORLDS; do
  prefix="${world}_baseline"
  for agent in $AGENTS; do
    for net in $NETWORKS; do
      echo ">>> $world $agent $net"
      if ! python run.py -t tsc -a "$agent" -w "$world" -n "$net" --prefix "$prefix" > /dev/null 2>&1; then
        echo "FAILED: $world $agent $net"; fail=1; continue
      fi
      src="data/output_data/tsc/${world}_${agent}/${net}/${prefix}/logger"
      tgt="$LOG_DEST/$agent/$net/${world}_direct_transfer"
      mkdir -p "$tgt"
      cp "$src"/*.log "$tgt/"
    done
  done
done

echo "=== FINAL_TEST summary ==="
for world in $WORLDS; do for agent in $AGENTS; do for net in $NETWORKS; do
  f=$(ls -t "$LOG_DEST/$agent/$net/${world}_direct_transfer/"*_DTL.log 2>/dev/null | head -1)
  [ -n "$f" ] && awk -F'\t' -v w=$world -v a=$agent -v n=$net \
    '$2=="FINAL_TEST"{printf "%-8s %-10s %-12s ATT=%-7s queue=%-7s throughput=%s\n", w, a, n, $4, $7, $9}' "$f"
done; done; done
exit $fail
