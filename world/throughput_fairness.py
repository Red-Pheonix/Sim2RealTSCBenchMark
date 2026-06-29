"""Throughput-based fairness, per intersection (Raeis & Leon-Garcia, arXiv:2107.10146).

A controller is unfair if it keeps discharging one approach while another with waiting
vehicles is starved. We measure that as the spread of **cumulative served throughput**
across the intersection's approaches:

  * per approach `k`, accumulate `C_k` += vehicles that left its incoming lanes this step
    (served / crossed the stop bar), measured as `|prev_ids \\ current_ids|`;
  * an approach is **demand-active** iff it currently has waiting vehicles (`waiting > 0`);
  * per-decision cost = `max_active C_k - min_active C_k`, or 0 when fewer than 2
    approaches are active (can't be unfair to <=1 approach -- the generalized form of the
    paper's "both directions have demand" gate).

Uniform entitlement (no per-approach weights). The WORLD owns one instance and feeds it
per-step per-approach `(current vehicle-id set, waiting count)`; state resets with the
world (per episode), so the FeatureBank that reads `intersection_fairness` stays stateless.
"""


class ThroughputFairness:
    def __init__(self):
        self.reset()

    def reset(self):
        self._served = {}   # inter_id -> {approach: cumulative served count}
        self._prev = {}     # inter_id -> {approach: set(current vehicle ids)}
        self._cost = {}     # inter_id -> last computed max-min spread

    def update(self, inter_id, approaches):
        """`approaches`: {approach_key: (set_of_current_ids, waiting_count)}. Accumulate
        served = |prev \\ current| per approach and recompute the per-intersection cost."""
        served = self._served.setdefault(inter_id, {})
        prev = self._prev.setdefault(inter_id, {})
        active = []
        for k, (ids, waiting) in approaches.items():
            p = prev.get(k)
            if p is not None:
                served[k] = served.get(k, 0.0) + len(p - ids)
            else:
                served.setdefault(k, 0.0)
            prev[k] = ids
            if waiting > 0:
                active.append(served[k])
        self._cost[inter_id] = (
            float(max(active) - min(active)) if len(active) >= 2 else 0.0
        )

    def cost(self, inter_id):
        return self._cost.get(inter_id, 0.0)
