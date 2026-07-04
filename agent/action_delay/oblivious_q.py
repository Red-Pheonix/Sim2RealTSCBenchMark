"""Oblivious-Q baseline (Derman, Dalal & Mannor, ICLR 2021).

The paper's delay-unaware baseline: the agent is trained and evaluated *with* the
execution delay applied to its actions, but it chooses actions as if they execute
immediately -- it never compensates for the delay.

As a method model this is the trivial "do nothing" policy: act on the current
(delayed) observation, with no in-flight queue and no forward model to roll or
train. The delay itself is applied by the trainer's run loop (the action-delay
queue), independent of the model -- so wiring this model in keeps the env delay
while compensating for none of it, which is exactly *oblivious to a delay* rather
than *delay-free*.

Same model interface as :class:`DelayedQModel` (``select_action`` /
``train_predictor`` / ``reset_all``) so the Delayed-Q trainer can drive it
without any method-specific branching. The compensation horizon ``m`` is accepted
for that uniform signature but ignored -- an oblivious agent has no horizon.

Note this differs from the ``naive`` "direct transfer" baseline, which trains in
delay-free sim (sim_action_delay = 0). Oblivious-Q trains under the delay; it just
ignores it.
"""

from common.registry import Registry


@Registry.register_sim2real_model("oblivious_q")
class ObliviousQModel:
    def __init__(self, agents, m, lr=1e-3, train_iters=1):
        # Nothing to hold: no queue, no predictor. m/lr/train_iters are part of
        # the shared model signature but unused -- there is no compensation.
        self.m = 0

    def select_action(self, agent, idx, ob, phase, test, valid_mask_fn=None):
        # Act on the current (delayed) observation -- no look-ahead.
        kw = {"valid_mask_fn": valid_mask_fn} if valid_mask_fn is not None else {}
        return agent.get_action(ob, phase, test=test, **kw)

    def train_predictor(self, agents):
        pass

    def reset_all(self, init_phases):
        pass

    def save_aux(self, model_dir, e=None):
        pass  # no predictor -- nothing to persist

    def load_aux(self, model_dir, e=None):
        pass
