import numpy as np

from common.registry import Registry


@Registry.register_model('base')
class BaseAgent(object):
    '''
    BaseAgent Class is mainly used for creating a base agent and base methods.
    '''
    def __init__(self, world):
        # revise if it is multi-agents in one model
        self.world = world
        self.sub_agents = 1

    def get_ob(self):
        raise NotImplementedError()

    def get_reward(self):
        raise NotImplementedError()

    def get_action(self, ob, phase):
        raise NotImplementedError()

    def get_action_prob(self, ob, phase):
        return None


class NonRLAgent(BaseAgent):
    '''
    Base for the non-learned controllers (fixed-time, max-pressure). They carry no
    weights and no replay buffer, but the sim2real trainers drive every agent through
    the same RL lifecycle (load/save/remember/train/update_target_network), so those
    calls are no-ops here rather than a missing attribute. Keeping the signatures
    identical to the RL agents is what lets a non-RL controller be dropped into any
    gap trainer as an eval-only baseline.

    Subclasses must set `self.id` to their intersection id and must NOT cache the
    Intersection object -- see `inter_obj`.
    '''

    # TODO(post-submission): WORKAROUND, not a fix -- see notes/nonrl_cleanup_todo.md.
    # Delete this property (back to a plain attribute) once the observations trainer
    # resets in the right order; it only exists to survive world_sumo's rebuild.
    @property
    def inter_obj(self):
        '''
        The live Intersection, re-resolved from the world on every read.

        `world_sumo.reset()` REBUILDS its Intersection objects, and TSCEnv.reset()
        calls it after the trainers have already called `agent.reset()`. An
        Intersection captured at construction or in reset() is therefore a dead copy
        on sumo: its `current_phase` / `current_phase_time` never advance again. A
        timer-driven controller reading that copy sees phase_time stuck at 0, never
        clears its t_min / t_fixed threshold, and holds its start phase for the whole
        episode -- silently, with a plausible-looking travel time out the other end.
        (cityflow resets its intersections in place, which is why this only ever bit
        the sumo side.)

        :param: None
        :return: Intersection object for this agent's intersection
        '''
        return self.world.id2intersection[self.id]

    # TODO(post-submission): WORKAROUND -- see notes/nonrl_cleanup_todo.md. Subclasses
    # still BUILD a phase_generator that this override ignores (dead weight). Delete
    # this and go back to the generator once the reset order is fixed, or better, make
    # IntersectionPhaseGenerator resolve its intersection live for every agent.
    def get_phase(self):
        '''
        get_phase
        Current phase, read off the LIVE intersection. IntersectionPhaseGenerator
        caches the Intersection at construction, so on sumo its `generate()` returns
        the phase of the dead copy; bypassing it keeps this agent's phase honest.

        :param: None
        :return phase: current phase of the intersection
        '''
        return np.array([self.inter_obj.current_phase], dtype=np.int8)

    def load_model(self, model_dir, e=None):
        '''No weights to load. Also makes `load_pretrained` a no-op, which is
        correct: there is no pretrained/tsc checkpoint for a non-learned controller.'''
        return

    def save_model(self, model_dir="", e=None):
        '''No weights to save.'''
        return

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs,
                 cur_phase, done, key):
        '''No replay buffer.'''
        return

    def train(self):
        '''Nothing to train; reported as zero loss.'''
        return 0.0

    def update_target_network(self):
        '''No target network.'''
        return

    def sample(self):
        '''Random action, for the warm-up phases the RL trainers run before learning.'''
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def _valid_mask(self, valid_mask_fn, phase, n_phases):
        '''
        Resolve the phase-transition transform's decision-time mask into a flat
        boolean array over phases.

        :param valid_mask_fn: callable `phase -> mask`, or None when nothing masks
        :param phase: current phase, as handed to get_action
        :param n_phases: number of phases at this intersection
        :return: bool array of length n_phases, or None when unmasked
        '''
        if valid_mask_fn is None:
            return None
        return np.asarray(valid_mask_fn(phase)).reshape(-1).astype(bool)[:n_phases]
