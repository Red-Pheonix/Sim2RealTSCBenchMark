import logging

from .base import NonRLAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
import numpy as np
import gym

LOG = logging.getLogger(__name__)


@Registry.register_model('maxpressure')
class MaxPressureAgent(NonRLAgent):
    '''
    MaxPressureAgent using Max-Pressure method to control traffic light.
    '''
    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world
        self.rank = rank
        self.model = None

        # get generator for each MaxPressure
        self.id = self.world.intersection_ids[self.rank]
        self._build_generators()
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))

        # the minimum duration of time of one phase
        self.t_min = Registry.mapping['model_mapping']['setting'].param['t_min']

    def _build_generators(self):
        '''
        _build_generators
        (Re)build every generator this agent reads. Shared by __init__ and reset so
        the two cannot drift apart.

        :param: None
        :return: None
        '''
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                     in_only=True, average='all', negative=True)

        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)

        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True,
                                                     negative=False)

        # Pressure needs BOTH the incoming and the outgoing lane counts, so it cannot
        # reuse the in-only ob_generator. Reading it through a generator (rather than
        # world.get_info, as this agent used to) is what puts max-pressure's sensing
        # behind the observation-gap pipeline: noise and sensor dropout are applied in
        # LaneVehicleGenerator.generate, the detection zone by swapping the subscribed
        # fn. Straight off the world none of those fire, and every observation setting
        # silently reports the no-gap number.
        self.pressure_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                       in_only=False, average=None,
                                                       negative=False)
        # TODO(post-submission): measured 0 uncovered lanes on all 10 networks x both
        # engines, so this fallback is speculative generality -- delete it and assert
        # coverage instead. See notes/nonrl_cleanup_todo.md.
        # Lanes the phase lanelinks reference that this intersection's in/out roads do
        # not cover: no detector of ours reads them, so the setting defines no value
        # for them and they keep the raw engine count.
        covered = {lane for group in self.pressure_generator.lanes for lane in group}
        self._uncovered_lanes = sorted(
            {lane for links in self.inter_obj.phase_available_lanelinks
             for pair in links for lane in pair} - covered
        )
        if self._uncovered_lanes:
            LOG.warning(
                "maxpressure %s: %d lanelink lane(s) outside the in/out roads keep "
                "untransformed counts (e.g. %s)",
                self.id, len(self._uncovered_lanes), self._uncovered_lanes[:3],
            )

    def observed_lane_count(self):
        '''
        observed_lane_count
        Lane id -> vehicle count AS THE CONTROLLER SEES IT, i.e. after the
        observation-gap transforms. This is the only count max-pressure decides on.

        :param: None
        :return: dict of lane id -> count
        '''
        values = self.pressure_generator.generate()
        lanes = [lane for group in self.pressure_generator.lanes for lane in group]
        observed = dict(zip(lanes, values))
        if self._uncovered_lanes:
            raw = self.world.get_info("lane_count")
            for lane in self._uncovered_lanes:
                observed[lane] = raw[lane]
        return observed

    def reset(self):
        '''
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        '''
        # get generator for each MaxPressure
        self._build_generators()

    def __repr__(self):
        return 'Maxpressure Agent has no Network model'

    def get_ob(self):
        '''
        get_ob
        Get observation from environment.

        :param: None
        :return x_obs: observation generated by ob_generator
        '''
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        '''
        get_reward
        Get reward from environment.

        :param: None
        :return rewards: rewards generated by reward_generator
        '''
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards)) * 12
        return rewards
    
    # get_phase() is inherited from NonRLAgent: it reads the LIVE intersection
    # instead of phase_generator, whose cached Intersection goes stale on sumo.
    
    def get_action(self, ob, phase, test=True, valid_mask_fn=None):
        '''
        get_action
        Generate action.

        :param ob: observation, the shape is (1,12)
        :param phase: current phase, the shape is (1,)
        :param test: boolean, decide whether is test process
        :param valid_mask_fn: optional `phase -> bool mask` from the phase-transition
            transform. Under the gap (enforce) baseline this is all-permissive except
            at force-off, where holding past max-green is illegal; honoring it keeps
            this baseline's controller semantics identical to the RL agents'.
        :return action: action that has the highest score
        '''
        # get lane pressure, as the (possibly degraded) detectors report it
        lvc = self.observed_lane_count()
        cur = self.inter_obj.current_phase
        n_phases = len(self.inter_obj.phases)
        mask = self._valid_mask(valid_mask_fn, phase, n_phases)

        # Hold out the minimum green, unless the controller has masked holding out.
        if self.inter_obj.current_phase_time < self.t_min:
            if mask is None or mask[cur]:
                return cur

        max_pressure = None
        action = -1
        for phase_id in range(n_phases):
            if mask is not None and not mask[phase_id]:
                continue
            pressure = sum([lvc[start] - lvc[end] for start, end in self.inter_obj.phase_available_lanelinks[phase_id]])
            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure

        return action

    def get_queue(self):
        '''
        get_queue
        Get queue length of intersection.

        :param: None
        :return: total queue length
        '''
        queue = []
        queue.append(self.queue.generate())
        queue = np.sum(np.squeeze(np.array(queue)))
        return queue

    def get_delay(self):
        '''
        get_delay
        Get delay of intersection.

        :param: None
        :return: total delay
        '''
        delay = []
        delay.append(self.delay.generate())
        delay = np.sum(np.squeeze(np.array(delay)))
        return delay

