#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import Enum
from collections import deque
import random
import math
import numpy as np
import tensorflow as tf
import weakref
import pygame
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import cv2

import carla
from agents.navigation.controller_location import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints
from agents.navigation.rnn_cell_dev import BasicLSTMCell
from agents.navigation.QAQ import *
# import ./method
# import dqn_utils
from dqn_utils import ReplayBuffer_QAQ, ReplayBuffer_QAQ_old, huber_loss, minimize_and_clip
from utility import spawn_v_try_with_transforms, spawn_v_finally, xy2lms, lms2xy, raw_data2points
from agents.tools.misc import is_within_distance_ahead


# ================ install openai baseline first
import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

import gym
from gym.spaces import Discrete, Box, MultiDiscrete

from baselines.deepq.deepq import ActWrapper


import time
import functools
from baselines.common.policies import build_policy

from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from baselines.common import explained_variance
from baselines.common import tf_util

from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner
from baselines.ppo2.ppo2 import safemean

from tensorflow import losses
# ================================


def get_speed_ms(vehicle):
    """
    Compute speed of a vehicle in Kmh
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def draw_locations(world, locations, color=carla.Color(0, 255, 0), z=0.5):
    """
    locations: vector3D
    """

    for w in locations:
        #    begin = w + carla.Location(z=z)
        #    end = begin + carla.Location(z = 1.0)
        #    world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)
        if "0.9.11" in carla.__path__[0]:
            world.debug.draw_point(w, size=0.1, color=color, life_time=0.034)
        else:
            world.debug.draw_point(w, size=0.1, color=color, life_time=0.03)


def utility_pathlength(path):
    return len(path["reward"])


# sadly, the first lidar point has random yaw
def rescue_lms(points, points_per_frame, lms_range=50, return_points=False):
    # each channel must be in order
    ele = np.pi*2 / points_per_frame
    yaw, d = xy2lms(points)
    if "0.9.8" in carla.__path__[0]:
        # manually check!!! e.g. 1.8 is the z-location of lidar
        on_floor_mask = points[:, 2] > 1.7
    elif "0.9.11" in carla.__path__[0]:
        on_floor_mask = points[:, 2] < -1.7
    # this once output 360?!!!!, because of it go out of precision of float e.g. float(359.99999999999) -> 360.0
    idx = np.int32(np.mod(yaw, np.pi*2) / ele)
    idx = np.mod(idx, points_per_frame)  # this is to rescue the above issue
    d_rescue = np.zeros(points_per_frame) + lms_range

    # now we want to find lms from different channel
    if len(idx) > 0:
        segs = np.where(idx[1:] < idx[:-1])[0] + 1
        segs = list(segs)
        left = [0] + segs
        right = segs + [len(idx)]

        num_parts = len(left)
        d_rescue = np.tile(d_rescue, [num_parts, 1])
        for i in range(num_parts):
            d_rescue[i, idx[left[i]:right[i]]] = d[left[i]:right[i]] * \
                (1-on_floor_mask[left[i]:right[i]]) + \
                lms_range*on_floor_mask[left[i]:right[i]]

        d_rescue = np.min(d_rescue, axis=0)

    yaw_bias = np.mean(np.mod(yaw, ele))
    yaw_rescue = np.linspace(0, np.pi*2, points_per_frame+1)[:-1]+yaw_bias

    # debug
    if return_points:
        points_rescue = lms2xy(yaw_rescue, d_rescue)
        return yaw_rescue, d_rescue, points_rescue
    else:
        return yaw_rescue, d_rescue


# sadly, the first lidar point has random yaw
def rescue_lms_grid(points, points_per_frame, lms_range=50, return_points=False):
    # each channel must be in order
    ele = np.pi*2 / points_per_frame
    yaw, d = xy2lms(points)
    # manually check!!! e.g. 1.8 is the z-location of lidar
    on_floor_mask = points[:, 2] > 1.7
    # this once output 360?!!!!, because of it go out of precision of float e.g. float(359.99999999999) -> 360.0
    idx = np.int32(np.mod(yaw, np.pi*2) / ele)
    idx = np.mod(idx, points_per_frame)  # this is to rescue the above issue
    d_rescue = np.zeros(points_per_frame) + lms_range

    # now we want to find lms from different channel
    if len(idx) > 0:
        segs = np.where(idx[1:] < idx[:-1])[0] + 1
        segs = list(segs)
        left = [0] + segs
        right = segs + [len(idx)]

        num_parts = len(left)
        d_rescue = np.tile(d_rescue, [num_parts, 1])
        for i in range(num_parts):
            d_rescue[i, idx[left[i]:right[i]]] = d[left[i]:right[i]] * \
                (1-on_floor_mask[left[i]:right[i]]) + \
                lms_range*on_floor_mask[left[i]:right[i]]

        d_rescue = np.min(d_rescue, axis=0)

    yaw_bias = np.mean(np.mod(yaw, ele))
    yaw_rescue = np.linspace(0, np.pi*2, points_per_frame+1)[:-1]+yaw_bias

    # compute_grid_map
    def compute_grid_map_square(points, res_x, res_y, lms_range, not_on_floor_mask=None):

        points = points[:, :2]
        if not_on_floor_mask is not None:
            points = points[np.where(not_on_floor_mask)]
        points = points[:, :2]
        # debug
        # x = np.linspace(-50,50,100)
        # y = np.linspace(-50,50,100)
        x = points[:, 0]
        y = points[:, 1]

        # to ensure stability, we define the whole grid map size as (2*lms_range / res + 1,2 * lms_range / res + 1)
        res_x = float(res_x)
        res_y = float(res_y)
        x = np.where(x > 0, x + 0.5*res_x, x - 0.5*res_x)
        y = np.where(y > 0, y + 0.5*res_y, y - 0.5*res_y)
        center_x = (int)(lms_range / res_x + 0.5)
        center_y = (int)(lms_range / res_y + 0.5)
        x_idx = (x / res_x).astype(np.int32) + center_x
        y_idx = (y / res_y).astype(np.int32) + center_y  # 50 - > middle
        img = np.zeros([2*center_x + 1, 2*center_y + 1], dtype=np.float32)
        img[(x_idx, y_idx)] = 1
        cv2.imshow("grid", img)
    #    cv2.waitKey(0)
        return img

    def compute_grid_map_polar(yaw, d, not_on_floor_mask=None):
        pass

    points_rescue = lms2xy(yaw_rescue, d_rescue)
    img = compute_grid_map_square(points_rescue, 4., 2., 50)
    # debug
    if return_points:
        return yaw_rescue, d_rescue, img, points_rescue
    else:
        return yaw_rescue, d_rescue, img


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None, **kwargs):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self.world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        spawn_points = self._map.get_spawn_points()
        self.reborn_points = []
        self.reborn_points.append(self._map.get_waypoint(
            spawn_points[87].location).next(100)[0].transform)
        self.reborn_points.append(self._map.get_waypoint(
            spawn_points[47].location).next(1)[0].transform)
        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._max_distance = 50.
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 200
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        # this is not really waypoint, it is simple 2 or 3-d cordinate
        self._waypoint_bufferlist = []

        self.lms_sensor = None
        self.sess = None
        self.target_location = None
        self.lidar_frame = None
        self.lidar_transfrom = None
        self.lidar_points = None
        self.collision = None
        self.laneInvasion = None

        actor_list = self.world.get_actors()
        self.vehicle_list = []  # npc_vehicle_list
        vehicle_list = actor_list.filter("*vehicle*")
        for i in vehicle_list:
            if i.id != self._vehicle.id:
                self.vehicle_list.append(i)
        print(len(self.vehicle_list))

        # default control
        self.default_control = carla.VehicleControl()
        self.default_control.steer = 0.0
        self.default_control.throttle = 0.0
        self.default_control.brake = 1.0
        self.default_control.hand_brake = False
        self.default_control.manual_gear_shift = False

        self.steer_pre = 0.0

        """ initializing sensor """

        """ initializing controller """
        self._init_controller(opt_dict)

        """for training"""
        self.T = kwargs.pop('Planning_towards_future', 21)
        self.T_reward_accumulation_second = kwargs.pop(
            'rewards_accumulation_time', 1.5)
        self.T_reward_accumulation_second_for_ac = kwargs.pop(
            'rewards_accumulation_time', 0.5)
        self.save_dir = kwargs.pop('save_dir', './save/working/')
        self.tensorboard_dir = "/tmp/mylogs/session"
        self.replay_buffer = None

        """tmp"""
        self.tmp = 0

        """log"""
        self.r_history = []
        self.r1_history = []
        self.r2_histroy = []
        self.r3_history = []
        self.collision_log = []

    def set_client(self, client):
        self.client = client

    def initialize_training(self, sync_mode, **kwargs):

        try:
            os.rmdir(self.tensorboard_dir)
        except:
            print("error remove tensorboard dir")

    #    self.T = kwargs.pop('Planning_towards_future', 20)
        self.load = kwargs.pop('load', False)
        """ reward """
        self.same_traj_reward = 0

        """ initialize model """
        self.ebm = 0
        self.Q_learning = 0
        self.openai_Q = 0
        self.Q_with_trajectory = 0
        self.use_Value = 0
        self.openai_a2c = 0
        self.openai_ppo2 = 0
        self.openai_ddpg = 1

        # set T_reward_computing
        T_second = sync_mode.delta_seconds
        if self.openai_Q or self.openai_a2c or self.openai_ppo2 or self.openai_ddpg:
            self.T_reward_computing = 1
            print("T_reward_computing:", self.T_reward_computing)

        elif not self.use_Value and not self.openai_a2c and not self.openai_ppo2: 
            self.T_reward_computing = int(
                self.T_reward_accumulation_second / T_second)
            print("delta_seconds:", T_second,
                  ",T_reward_computing:", self.T_reward_computing)
        else:
            self.T_reward_computing = int(
                self.T_reward_accumulation_second_for_ac / T_second)
    #        self.T_reward_computing = 1
    #    self.T_reward_computing = 1#debug

        self.grid_size = (5, 7)
        self.D_s = self.lidar_points_per_frame + 2  # (lidar + v + steer)

        if self.ebm == 1:
            self.D_a = self.T
            if self.same_traj_reward == 1:
                self.D_a += 1
        elif self.openai_Q == 1:
            self.n_a = 3
        elif self.openai_a2c == 1:
            self.n_a = 3
        elif self.openai_ppo2 == 1:
            self.n_a = 3
        elif self.openai_ddpg == 1:
            self.n_a = 3
        else:
            self.D_a = 1

        """ training batch """
        self.s_episode = []
        self.all_a_episode = []
        # self.r_episode = []
        self.r_episode1 = []
        self.r_episode2 = []
        self.r_episode3 = []
        self.a_label_episode = []
        self.prob_episode = []
        self.num_traj_list = []
        self.a_episode = []
        self.value_episode = []
        self.steps = 0
        self.train_steps = 0
        self.episodes_step = 0

        self.batch_size = 400
        self.train_every = 10
        self.update_critic_every = 1
        self.buffer_every = 100
        self.buffer_every_maximum = 0
        self.target_update_counter = 0
        #    self.learning_rate = kwargs.pop('learning_rate', 1e-4)#for simple policy gradient
        self.learning_rate_critic = kwargs.pop(
            'learning_rate', 1e-3)  # for critic update,1e-4 is bad
        #    self.learning_rate = kwargs.pop('learning_rate', 5e-1)#old
        self.learning_rate = kwargs.pop('learning_rate', 5e-1)  # for ac
        # self.grad_norm_clipping = 0.1
        # self.grad_norm_clipping_critic = 0.1
        self.grad_norm_clipping = 0.01/self.learning_rate
        self.grad_norm_clipping_critic = 0.1

        if self.openai_Q or self.openai_a2c or self.openai_ppo2:
            self.gamma = 0.99
        else:
            self.gamma = 0.99  # previous O.98

        self.done = 1
        self.epsilon = 1
        self.EPSILON_DECAY = 0.99  # 0.9975 99975
        self.MIN_EPSILON = 0.001

        """traj"""
        self.lc_label = -1

        """"debug"""
    #    self.D_s = 5
    #    self.D_a = 3

        """ initializing graph """
        #    self.model = self.build_model()
        #    self.target_model = self.build_model()
        self.train_global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        if self.ebm == 1:
            if self.Q_learning == 1:
                self.solver_build_compuational_graph_Q()
            else:
                self.solver_build_compuational_graphAC()
        elif self.openai_Q == 1:
            # ============================== openai Q part
            # just copy the runtime params:
            #    self.alg_kwargs = {'network': 'mlp', 'lr': 0.0001, 'buffer_size': 10000, 'exploration_fraction': 0.1, 'exploration_final_eps': 0.01, 'train_freq': 4, 
            #                       'learning_starts': 10000, 'target_network_update_freq': 1000, 'gamma': 0.99, 'prioritized_replay': True, 'prioritized_replay_alpha': 0.6, 'checkpoint_freq': 10000, 'checkpoint_path': None, 'dueling': True}

            self.alg_kwargs = {'batch_size': self.batch_size, 'prioritized_replay_eps': 1e-6, 'network': 'mlp', 'lr': 0.0001, 'buffer_size': 10000, 'exploration_fraction': 0.1, 'exploration_final_eps': 0.01, 'train_freq': 4,
                               'learning_starts': 100, 'target_network_update_freq': 1000, 'gamma': 0.99, 'prioritized_replay': True, 'prioritized_replay_alpha': 0.6, 'checkpoint_freq': 10000, 'checkpoint_path': None, 'dueling': True}
            network = self.alg_kwargs['network']
            network_kwargs = {'dueling': True}
            exploration_fraction = self.alg_kwargs['exploration_fraction']
            gamma = 0.99
            param_noise = False
            prioritized_replay = self.alg_kwargs['prioritized_replay']
            buffer_size = self.alg_kwargs['buffer_size']
            prioritized_replay_beta_iters = None
            total_timesteps = 1e6
            prioritized_replay_alpha = 0.6
            prioritized_replay_beta0 = 0.4
            exploration_final_eps = 0.02
            # over

            # here it infact always build mlp
            q_func = build_q_func(network, **network_kwargs)

            # capture the shape outside the closure so that the env object is not serialized
            # by cloudpickle when serializing make_obs_ph

            fake_observation_space = Box(
                low=-999., high=999., shape=[self.D_s])
            observation_space = fake_observation_space

            def make_obs_ph(name):
                return ObservationInput(observation_space, name=name)

            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=make_obs_ph,
                q_func=q_func,
                num_actions=self.n_a,
                optimizer=tf.train.AdamOptimizer(
                    learning_rate=self.alg_kwargs['lr']),
                gamma=gamma,
                grad_norm_clipping=10,
                param_noise=param_noise
            )

            act_params = {
                'make_obs_ph': make_obs_ph,
                'q_func': q_func,
                'num_actions': self.n_a,
            }

            act = ActWrapper(act, act_params)

            # Create the replay buffer
            if prioritized_replay:
                self.openai_replay_buffer = PrioritizedReplayBuffer(
                    buffer_size, alpha=prioritized_replay_alpha)
                if prioritized_replay_beta_iters is None:
                    prioritized_replay_beta_iters = total_timesteps
                beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                               initial_p=prioritized_replay_beta0,
                                               final_p=1.0)
            else:
                self.openai_replay_buffer = ReplayBuffer(buffer_size)
                beta_schedule = None
            # Create the schedule for exploration starting from 1.
            exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                         initial_p=1.0,
                                         final_p=exploration_final_eps)

            # Initialize the parameters and copy them to the target network.
            U.initialize()
            update_target()

            saved_mean_reward = None
            reset = True

            self.openai_things = (act, train, update_target, debug, exploration, beta_schedule,
                                  gamma, param_noise, exploration_fraction, reset, saved_mean_reward)
        elif self.openai_a2c == 1:
            # ============================== openai a2c part
            # just copy the runtime params:
            alpha = 0.99
            ent_coef = 0.01
            epsilon = 1e-5
            gamma = self.gamma
            log_interval = 100
            lr = 0.0007
            lrschedule = 'linear'
            max_grad_norm = 0.5
            network = 'mlp'
            nsteps = self.batch_size
            seed = 1
            total_timesteps = 1000000
            vf_coef = 0.5

            load_path = None

            # some parameter setting
            n_envs = 1  # currently one simulator
            nbatch = self.batch_size

            # here we make a blank env to make codes work:
            env = gym.Env()
            env.num_envs = 1
            env.action_space = Discrete(self.n_a)
            env.observation_space = Box(low=-999., high=999., shape=[self.D_s])
            # the copy from .learn() file:
            set_global_seeds(seed)
            nenvs = env.num_envs
            policy = build_policy(env, network)

            from baselines.a2c.a2c import Model
            model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                          max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
            if load_path is not None:
                model.load(load_path)

            # Start total timer
            tstart = time.time()
            self.openai_things = model
        elif self.openai_ppo2 == 1:
            # ============================== openai a2c part
            # just set the runtime params:
            ent_coef = 0.01
            vf_coef = 0.5
            gamma = self.gamma
            lam = 0.95
            log_interval = 1
            max_grad_norm = 0.5
            mpi_rank_weight = 1
            network = 'mlp'
            network_kwargs = {}
            nminibatches = 1  # for our simulator
            noptepochs = 1  # for our simulator
            nsteps = self.batch_size
            save_interval = 0
            total_timesteps = 20000000
            seed = None
            load_path = None

            # here we make a blank env to make codes work:
            env = gym.Env()
            env.num_envs = 1
            env.action_space = Discrete(self.n_a)
            env.observation_space = Box(low=-999., high=999., shape=[self.D_s])
            # the copy from .learn() file:
            set_global_seeds(seed)
            def lr(f): return f * 2.5e-4
            cliprange = 0.1
            policy = build_policy(env, network, **network_kwargs)

            # Get the nb of env
            nenvs = env.num_envs

            # Get state_space and action_space
            ob_space = env.observation_space
            ac_space = env.action_space

            nbatch = nenvs * nsteps
            nbatch_train = nbatch // nminibatches

            is_mpi_root = True  # ???????????
            epinfobuf = deque(maxlen=100)

            from modified_openai.model_ppo2_without_adv_normalize import Model
            model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                          nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                          max_grad_norm=max_grad_norm, comm=None, mpi_rank_weight=mpi_rank_weight)

            if load_path is not None:
                model.load(load_path)

            # Start total timer
            tfirststart = time.perf_counter()
            nupdates = total_timesteps / nbatch
            self.openai_things = model, nminibatches, nupdates, lr, cliprange, noptepochs, nbatch_train, epinfobuf
        elif self.openai_ddpg == 1:
            self.alg_kwargs = {'batch_size': self.batch_size, 'prioritized_replay_eps': 1e-6, 'network': 'mlp', 
                                'lr': 0.0001, 'buffer_size': 10000, 'exploration_fraction': 0.1, 
                                'exploration_final_eps': 0.01, 'train_freq': 4,
                               'learning_starts': 100, 'target_network_update_freq': 1000, 'gamma': 0.99, 
                               'prioritized_replay': True, 'prioritized_replay_alpha': 0.6, 'checkpoint_freq': 10000, 
                               'checkpoint_path': None, 'dueling': True}

            # original ddpg
            seed=None
            total_timesteps=None
            nb_epochs = None  # with default settings, perform 1M steps total
            nb_epoch_cycles=20
            nb_rollout_steps=100
            reward_scale=1.0
            render=False
            render_eval=False
            noise_type = "adaptive-param_0.2" 
            normalize_returns=False
            normalize_observations=True
            critic_l2_reg=1e-2
            actor_lr=1e-4
            critic_lr=1e-3
            popart=False
            gamma=0.99
            clip_norm=None
            nb_train_steps=50 # per epoch cycle and MPI worker,
            nb_eval_steps=100
            batch_size=64 # per MPI worker
            tau=0.01
            eval_env=None
            param_noise_adaption_interval=50
            
            # modified ddpg
            network = self.alg_kwargs['network']
            network_kwargs = {}
            gamma = 0.99
            actor_lr = 0.0007
            critic_lr = 0.0006
            # lr = 0.0007
            max_grad_norm = 0.5
            nsteps = self.batch_size
            seed = 1
            total_timesteps = 1000000

            # some parameter setting
            n_envs = 1  # currently one simulator
            nbatch = self.batch_size

            # here we make a blank env to make codes work:
            env = gym.Env()
            env.num_envs = 1
            env.action_space = Box(low= -0.2,
                                    high= 0.2, shape=(1,),
                                    dtype=np.float32)
            env.observation_space = Box(low=-999., high=999., shape=[self.D_s])
            # the copy from .learn() file:
            set_global_seeds(seed)

            # not in original 
            nenvs = env.num_envs
            policy = build_policy(env, network)

            if total_timesteps is None:
                # assert nb_epochs is None # wrong ? 
                nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
            else:
                nb_epochs = 500

            rank = 0
            nb_actions = env.action_space.shape[-1]
            # assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

            memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
            critic = Critic(network=network, **network_kwargs)
            actor = Actor(nb_actions, network=network, **network_kwargs)

            action_noise = None
            param_noise = None
            if noise_type is not None:
                for current_noise_type in noise_type.split(','):
                    current_noise_type = current_noise_type.strip()
                    if current_noise_type == 'none':
                        pass
                    elif 'adaptive-param' in current_noise_type:
                        _, stddev = current_noise_type.split('_')
                        param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
                    elif 'normal' in current_noise_type:
                        _, stddev = current_noise_type.split('_')
                        action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                    elif 'ou' in current_noise_type:
                        _, stddev = current_noise_type.split('_')
                        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                    else:
                        raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

            max_action = env.action_space.high
            # logger.info('scaling actions by {} before executing in env'.format(max_action))

            self.agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
                batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                reward_scale=reward_scale)
            # logger.info('Using agent with the following configuration:')
            # logger.info(str(agent.__dict__.items()))

            eval_episode_rewards_history = deque(maxlen=100)
            episode_rewards_history = deque(maxlen=100)
            sess2 = U.get_session()

            self.agent.initialize(sess2)
            # sess2.graph.finalize()

            self.agent.reset()

            episode_reward = np.zeros(nenvs, dtype = np.float32) #vector
            episode_step = np.zeros(nenvs, dtype = int) # vector
            episodes = 0 #scalar
            t = 0 # scalar

            epoch = 0

        else:
            self.build_computation_graph()
        
        self.sess = tf.Session()
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.sess.__enter__()  # equivalent to `with self.sess:`

        """ initializing replay_buffer """
        if self.Q_learning == 1:
            self.replay_buffer = ReplayBuffer_QAQ(size=5000)
        else:
            self.replay_buffer = ReplayBuffer_QAQ(size=500)

        """ initializing variable """
        self.tf_saver = tf.train.Saver()

        if self.load == True:
            #            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            #            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if self.openai_Q:
                model_file = os.path.join(self.save_dir, "model")
                load_variables(model_file)
                print('Loaded model from', self.save_dir)
            elif self.openai_a2c:
                model_file = os.path.join(self.save_dir, "model")
                load_variables(model_file)
                print('Loaded model from', self.save_dir)
            elif self.openai_ppo2:  # not sure
                model_file = os.path.join(self.save_dir, "model")
                load_variables(model_file)
                print('Loaded model from', self.save_dir)
            elif self.openai_ddpg:
                model_file = os.path.join(self.save_dir, "model")
                load_variables(model_file)
                print('Loaded model from', self.save_dir)
            else:
                tf.train.Saver().restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
        else:
            tf.global_variables_initializer().run()  # pylint: disable=E1101
            os.system("cp %s %s" % ('~/CARLA_0.9.8/PythonAPI/carla/agents/navigation/local_planner.py',
                      '"%s"/local_planner.py' % self.save_dir))
        #     os.system ("cp %s %s" % ('../carla/agents/navigation/local_planner.py', '"%s"/local_planner.py' % self.save_dir))

        # self.writer.add

        self.gd_debug_history = []
        self.params_gd_history = []
        self.debug_r_history = []
        self.debug_prob_history = []
        self.debug_r2_history = []
        self.loss_history = []
        self.critic_loss_history = []

        self.reset_vehicle_rl()

        """AC_related"""
        self.local_step_in_critic = 0
        self.num_target_updates_every = 1
        self.num_grade_update_per_step = 2
        self.normalize_advantages = 0
        # if self.use_Value:
        #     self.normalize_advantages = 1
        # else:
        #     self.normalize_advantages = 0

        """debug"""
        # n = 100

        # s_episode = [np.arange(i,i+5) for i in np.arange(n)]
        # all_a_episode = [np.eye(3)]*n + np.expand_dims(np.expand_dims(np.arange(n),axis = 1),axis = 2)   #[](n,da)
        # r_episode = np.arange(n)
        # a_label_episode = np.ones(100,dtype=np.int32)
        # num_traj_list = np.ones(100,dtype=np.int32)*3

        # rb = ReplayBuffer_QAQ(100)
        # rb.store_frames(s_episode, all_a_episode, r_episode, a_label_episode, num_traj_list)

        # all_obs_batch, all_a_batch, rew_batch, grouping, a_label_batch = rb.sample(n-1)

        # feed_dict = {self.all_s_placeholder_nds:np.array(all_obs_batch),
        #                 self.all_a_placeholder_nda : np.array(all_a_batch),
        #                 self.gt_placeholder_n:np.array(a_label_batch),
        #                 self.adv_placeholder_n:np.array(rew_batch),
        #                 self.grouping_placeholder_nn:np.array(grouping),
        #                 }
        # _,global_step,r,Z = self.sess.run([self.update_op,self.train_global_step,self.r_est_n,self.debug_Z],feed_dict=feed_dict)
        # exp_n = np.exp(r)
        # cum_exp_1n = np.expand_dims(np.cumsum(exp_n),axis = 0)
        # debug_Z = np.matmul(cum_exp_1n , np.array(grouping))
        # pass

    def tick(self, sync_mode):
        """
        Thanks to Carla 0.96's synchronous_mode example.
        This is the function, that move the "tick()" inside the local planner classes

        - sync_mode: the synchronous_mode context

        """
        snapshot, self.lidar_frame, = sync_mode.tick(timeout=2.0)
        # print(len(self.lidar_frame))

    def draw_lidar_from_raw(self, DIM0, DIM1, blend=True):
        lidar_frame = self.lidar_frame
        points = raw_data2points(lidar_frame)
        # rescue_lms(points,12)
        lidar_data = np.array(points[:, :2])
        lidar_data *= DIM1 / 100.0
        lidar_data += (0.5 * DIM0, 0.5 * DIM1)
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (DIM0, DIM1, 3)
        lidar_img = np.zeros(lidar_img_size)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        image_surface = pygame.surfarray.make_surface(lidar_img)
        if blend:
            image_surface.set_alpha(100)

        return image_surface

    def build_model(self):
        # base_model=Xception(weights=None,include_top=False, input_shape=(120,120,1))
        # x=base_model.output
        # x=GlobalAveragePooling2D()(x)
        # predictions=Dense(4,activation="linear")(x)
        # model=Model(inputs=base_model.input,outputs=predictions)
        # model.compile(loss="mse",optimizer=Adam(lr=0.01),metrics=["accuracy"])

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                  activation='relu', input_shape=(120, 120, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5),
                  padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                  padding='Same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                  padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.25))
        model.add(Dense(output_dim=64, input_dim=256))
        model.add(Dense(output_dim=self.action_num, input_dim=64))
        model.compile(loss="mse", optimizer=Adam(
            lr=self.learning_rate), metrics=["accuracy"])

        # model = Sequential()
        # model.add(Dense(output_dim=64, input_dim=self.D_s))
        # model.add(Dense(output_dim=32, input_dim=64))
        # model.add(Dense(output_dim=self.action_num, input_dim=32))
        # model.compile(loss="mse",optimizer=Adam(lr=self.learning_rate),metrics=["accuracy"])
        return model

    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in policy gradient
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """
        sy_ob_no = tf.placeholder(
            shape=[None, self.D_s], name="ob", dtype=tf.float32)
        sy_ac_na = tf.placeholder(
            shape=[None, self.D_a], name="ac", dtype=tf.float32)
        # YOUR CODE HERE
        sy_adv_n = tf.placeholder(shape=[None], dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#

    def policy_forward_pass(self, sy_ob_no):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_logits_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case)
                and the mean (in the continuous case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """
        # YOUR_CODE_HERE
        sy_mean = build_mlp(sy_ob_no, self.D_a, 'policy_mean', 3, 128)
        # ---------why should std not a function of ob?
        sy_logstd = tf.get_variable('std', shape=(self.D_a,))
        return (sy_mean, sy_logstd)

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def sample_action(self, policy_parameters):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

            returns:
                sy_sampled_ac:
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)

            Hint: for the continuous case, use the reparameterization trick:
                 The output from a Gaussian distribution with mean 'mu' and std 'sigma' is

                      mu + sigma * z,         z ~ N(0, I)

                 This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
        """
        sy_mean, sy_logstd = policy_parameters
        # YOUR_CODE_HERE
        batch_size = tf.shape(sy_mean)[0]
        sy_sampled_ac = tf.exp(
            sy_logstd) * tf.random_normal((batch_size, self.D_a)) + sy_mean
        return sy_sampled_ac

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

                sy_ac_na:
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)

            returns:
                sy_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
                For the continuous case, use the log probability under a multivariate gaussian.
        """

        sy_mean, sy_logstd = policy_parameters
        # YOUR_CODE_HERE
        sy_mean.get_shape()[1]
        sy = (sy_ac_na - sy_mean) / tf.exp(sy_logstd)
        Z = - tf.reduce_sum(sy_logstd) - 0.5*self.D_a * tf.log(np.pi*2)
        sy_logprob_n = -0.5 * tf.reduce_sum(sy * sy, axis=1) + Z

        """
        sy_mean, sy_logstd = policy_parameters
        # YOUR_CODE_HERE
        batchsize = sy_mean.get_shape()[0]
        sy_logstd_tile = tf.reshape(sy_logstd,[1,-1])
        sy_logstd_tile = tf.tile(sy_logstd_tile,batchsize)
        tmp = tf.contrib.distributions.MultivariateNormalDiag(
            loc=sy_mean,scale_diag=sy_logstd_tile)
        sy_logprob_n = tmp.prob(sy_ac_na)
        """
        return sy_logprob_n


    def solver_build_compuational_graphAC(self):
        """
        use other samples
        """
        self.all_s_placeholder_nds = tf.placeholder(
            tf.float32, shape=(None, self.D_s))  # currently 'open-loop' RL
        self.all_a_placeholder_nda = tf.placeholder(
            tf.float32, shape=(None, self.D_a))
        self.gt_placeholder_n = tf.placeholder(tf.float32, shape=(None))
        self.prob_placeholder_n = tf.placeholder(tf.float32, shape=(None))
        """
        # self.gt_placeholder_n:
         - a 0-1 vector of, which one is the 'on-policy samples'
        """
        self.grouping_placeholder_nn = tf.placeholder(
            tf.float32, shape=(None, None))
        self.grouping2_placeholder_nn = tf.placeholder(
            tf.float32, shape=(None, None))
        """
        # explaination of grouping matrix nn:
        # -we need to compute the probability of action (from replay buffer) on updated policy
        # -thus we need to group sum: sum_j exp(R(s_i,a_j))
        # -thus we will firstly need to cumsum the exp(R(s,a))
        # -then multiply, e.g., following matrix (vacant == zero):
        [ , ,  ,  ,  ]
        [1,1,-1,  ,  ]
        [ , , 1,-1,-1]
        [ , ,  ,  ,  ]
        [ , ,  , 1, 1]
        this is sum([exp_r1,exp_r2]),sum([exp_r1,exp_r2]), sum(
            [exp_r3]), sum([exp_r4,exp_r5]) , sum([exp_r4,exp_r5])
        """

        x_input_nd = tf.concat(
            [self.all_s_placeholder_nds, self.all_a_placeholder_nda], axis=1)
        self.adv_placeholder_n = tf.placeholder(shape=[None], dtype=tf.float32)
        # self.all_adv_placeholder_n = tf.placeholder(shape=[None],dtype = tf.float32)        x1_input_nd = tf.concat([self.s_placeholder_nds,self.a_placeholder_nda],axis = 1)

        """#############################
            first part: compute loss for train
            PS: each block of 'wrong implementation' is found by a great deal of experiments
        """
        with tf.variable_scope('reward_est'):

            if 1:  # gradient check
                x_input_nd_variable = tf.get_variable("input_variable", shape=(
                    1, x_input_nd.shape[1]), dtype=tf.float32, initializer=tf.constant_initializer(value=0))
                tmpa, tmpb = x_input_nd[:1, :], x_input_nd[1:, :]
                tmpa += x_input_nd_variable
                x_input_nd = tf.concat(values=[tmpa, tmpb], axis=0)

            # self.r_est_n = tf.sigmoid(build_mlp(x_input_nd,1,'reward_est',3,128)[:,0] - 0.3)#slow
            self.r_est_n = build_mlp(x_input_nd, 1, 'reward_est', 3, 256)[:, 0]
            # self.r_est_n = build_1dcnn_model(x_input_nd,1,3,128)[:,0]

        """
        to solve the numerical problem of 'exp/sum_exp' , generally we need to deduct the min(or max) in advance


        thus we need a grouping matrix like:
        self.grouping2_placeholder_nn =
        grouping2 = np.array([[1., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 1.]])
        if so, we will use inner build 'softmax', and abondon the 'self.grouping_placeholder_nn'
        """


        # num_of_traj = tf.reduce_sum(self.grouping2_placeholder_nn,axis = 1)
        # uniform_prob = self.grouping2_placeholder_nn / tf.expand_dims(num_of_traj,axis = 1)
        # uniform_prob = tf.reduce_sum(uniform_prob, axis = 0)

        # gt_placeholder_nn = tf.zeros_like(self.grouping2_placeholder_nn)

        inf_mask = 1/self.grouping2_placeholder_nn - self.grouping2_placeholder_nn
        r_est_nn = self.r_est_n * self.grouping2_placeholder_nn

        """#abandoned
        r_sum_n = tf.reduce_sum(r_est_nn,axis = 1)
        r_gt_n = self.r_est_n * self.gt_placeholder_n
        tmp1 =  r_gt_n * self.grouping2_placeholder_nn
        r_gt_n = tf.reduce_sum(tmp1,axis = 1)
        tmp2_n = (r_sum_n - r_gt_n) / r_sum_n
        tmp2_n /= (num_of_traj-1)/num_of_traj
        gd_coef = self.grouping2_placeholder_nn * \
            tf.expand_dims(tmp2_n,axis = 1)

        # debug for gd_coef
        r_est_n = np.array([-6001.,-6002.,-6003.,-6004.,-6005])
        # r_est_n = np.array([-1.,-2.,-3.,-4.,-5])
        # r_est_n = np.array([1.,2.,3.,4.,5])
        # r_est_n = np.array([-6001.,6002.,6003.,6004.,6005])
        grouping2 = np.array([[1., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 1.]])
        gt = np.array([0,1,1,1,0])
        r_gt_n = r_est_n * gt
        tmp1 =  r_gt_n * grouping2
        r_gt_n = np.sum(tmp1,axis = 1)
        r_est_nn = r_est_n * grouping2
        r_sum_n = np.sum(r_est_nn,axis = 1)
        tmp2_n = (r_sum_n - r_gt_n) / r_sum_n
        gd_coef = grouping2 * np.expand_dims(tmp2_n,axis = 1)
        """

        r_est_nn = r_est_nn - inf_mask
        prob_nn = tf.nn.softmax(r_est_nn)
        prob_n = tf.reduce_sum(prob_nn, axis=0)
        prob_n = tf.stop_gradient(prob_n)



        ####################################
        # this part is also theoretically right, but has numerical problem
        #        exp_n = tf.exp(self.r_est_n)
        #        cum_exp_1n = tf.expand_dims(tf.cumsum(exp_n),axis = 0)
        #        self.debug_Z = tf.matmul(cum_exp_1n , self.grouping_placeholder_nn)
        #
        #        prob_n = tf.squeeze(exp_n / self.debug_Z,axis = 0)
        #        prob_n = tf.stop_gradient(prob_n)
        ####################################

        """#test for above three implementation
        # correct but has numerical problem
        # you will see, after some training, there will be inf or nan everywhere
        r_est_n = np.array([-6001.,-6002.,-6003.,-6004.,-6005])
        r_est_n = np.array([-1.,-2.,-3.,-4.,-5])
        r_est_n = np.array([1.,2.,3.,4.,5])
        r_est_n = np.array([6001.,6002.,6003.,6004.,6005])
        grouping =([[0 ,0 ,0 ,0 ,0],
                   [1,1,-1,0 ,0  ],
                   [0 ,0 ,1,-1,-1],
                   [0 ,0 ,0,0 ,0 ],
                   [0 ,0 ,0, 1, 1]])
        exp_n = np.exp(r_est_n)
        cum_exp_1n = np.expand_dims(np.cumsum(exp_n),axis = 0)
        debug_Z = np.matmul(cum_exp_1n , grouping)
        prob_n = np.squeeze(exp_n / debug_Z,axis = 0)

        # wrong
        grouping2 = np.array([[1., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 1.]])
        grouping2 = tf.constant(grouping2)
        r_est_nn = r_est_n * grouping2
        prob_nn = tf.nn.softmax(r_est_nn)
        prob_n2 = tf.reduce_sum(prob_nn,axis = 0)
        with tf.Session() as sess:
            prob_nn = sess.run(prob_nn)

        # wrong again for cases : r_est_nn < 0
        grouping2 = np.array([[1., 1., -np.inf, -np.inf, -np.inf],
                       [-np.inf, -np.inf, 1., -np.inf, -np.inf],
                       [-np.inf, -np.inf, -np.inf, 1., 1.]])
        grouping2 = tf.constant(grouping2)
        r_est_nn = r_est_n * grouping2
        prob_nn = tf.nn.softmax(r_est_nn)
        prob_n2 = tf.reduce_sum(prob_nn,axis = 0)
        with tf.Session() as sess:
            prob_nn,prob_n2 = sess.run([prob_nn,prob_n2])



        # right (correct usage of inf)
        grouping2 = np.array([[1., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 1.]])
        grouping2 = tf.constant(grouping2)
        inf_mask = 1/grouping2 - grouping2
        r_est_nn = r_est_n * grouping2
        r_est_nn = r_est_nn - inf_mask
        prob_nn = tf.nn.softmax(r_est_nn)
        prob_n2 = tf.reduce_sum(prob_nn,axis = 0)
        with tf.Session() as sess:
            prob_nn,prob_n2,r_est_nn = sess.run([prob_nn,prob_n2,r_est_nn])
        """

        # loss = - tf.reduce_mean(  self.sy_logprob_n * self.sy_adv_n )
        # It seems that, positive prob_n can have a "low-passing" effect and stablize the change of probability
        prob_n2 = self.prob_placeholder_n
        loss1 = - tf.reduce_mean(self.r_est_n *
                                 (self.gt_placeholder_n - prob_n2) * self.adv_placeholder_n)
        # loss2 = 0
        loss2 = - tf.reduce_mean(self.r_est_n * (prob_n2 - prob_n) * 2.)
        # loss = loss1 + loss2*2
        loss = loss1 + loss2
        # loss = loss1
        # loss = - tf.reduce_mean(self.r_est_n * (self.gt_placeholder_n - prob_n2) * self.adv_placeholder_n)
        # loss = - tf.reduce_mean(self.r_est_n * (self.gt_placeholder_n - prob_n2) * (-40))
        self.total_loss = loss
        self.debug_info = self.gt_placeholder_n - prob_n2
        # self.debug_info = prob_n2 - prob_n

        """#############################
            on-policy sampling
        """
        self.sample_prob_n = tf.nn.softmax(self.r_est_n)
        self.sample_a_idx = tf.random.categorical(
            tf.expand_dims(self.r_est_n, axis=0), 1)

        """
        # debug for tf.random.categorical
        sample_prob_n = tf.nn.softmax([[-10., -11.]])
        samples = tf.random.categorical([[-10., -11.]], 10000)
        with tf.Session() as sess:
            a,p = sess.run([samples,sample_prob_n])
        a = a[0]
        """

        # compute reward
        # rnn model
        # a_rnn = BasicLSTMCell(64,state_is_tuple = True)
        # b_rnn = BasicLSTMCell(16,state_is_tuple = True)
        # reward_model = MultiModel([a_rnn,b_rnn])
        # state_prev = reward_model.zero_state(N)
        # reward = reward_model(x_input_nd,state_prev)

        # self.update_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss,global_step = self.train_global_step)
        # self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss,global_step = self.train_global_step)# <- really bad performance, why?
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # opt = tf.train.AdamOptimizer(self.learning_rate)
        params = tf.trainable_variables()
        params.remove(x_input_nd_variable)
        self.gd = opt.compute_gradients(self.total_loss, params)
        gd = self.gd

        for i, (grad, var) in enumerate(gd):
            if grad is not None:
                gd[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)

        self.gd_debug = opt.compute_gradients(loss, x_input_nd_variable)
        self.update_op = opt.apply_gradients(
            gd, global_step=self.train_global_step)

        """#############################
            define the critic
        """
        self.sy_ob_no = tf.placeholder(
            shape=[None, self.D_s], name="ob", dtype=tf.float32)
        with tf.variable_scope('critic'):
            self.critic_prediction = tf.squeeze(build_mlp(
                self.sy_ob_no,
                1,
                "nn_critic",
                3,
                256))
        self.sy_target_n = tf.placeholder(
            shape=[None], name="critic_target", dtype=tf.float32)
        # self.critic_loss = tf.losses.mean_squared_error(self.sy_target_n, self.critic_prediction)
        self.critic_loss = tf.reduce_mean(huber_loss(
            self.sy_target_n - self.critic_prediction))
        # self.critic_update_op = tf.train.GradientDescentOptimizer(self.learning_rate_critic).minimize(self.critic_loss)
        # opt_critic = tf.train.AdamOptimizer(self.learning_rate_critic)
        # adam perform badly, but sgd will cause 'hard-to-converge' problem?
        opt_critic = tf.train.GradientDescentOptimizer(
            self.learning_rate_critic)
        params_critic = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "critic/")
        gd_critic = opt_critic.compute_gradients(
            self.critic_loss, params_critic)
        for i, (grad, var) in enumerate(gd_critic):
            if grad is not None:
                gd_critic[i] = (tf.clip_by_norm(
                    grad, self.grad_norm_clipping_critic), var)

        self.critic_update_op = opt_critic.apply_gradients(gd_critic)


    def solver_build_compuational_graph_Q(self):
        """
        use other samples
        """
        self.all_s_placeholder_nds = tf.placeholder(
            tf.float32, shape=(None, self.D_s))  # currently 'open-loop' RL
        self.all_a_placeholder_nda = tf.placeholder(
            tf.float32, shape=(None, self.D_a))
        self.gt_placeholder_n = tf.placeholder(tf.float32, shape=(None))
        self.prob_placeholder_n = tf.placeholder(tf.float32, shape=(None))
        self.all_s_placeholder_next_nds = tf.placeholder(
            tf.float32, shape=(None, self.D_s))  # currently 'open-loop' RL
        self.all_a_placeholder_next_nda = tf.placeholder(
            tf.float32, shape=(None, self.D_a))
        self.gt_placeholder_next_n = tf.placeholder(tf.float32, shape=(None))
        self.done_placeholder = tf.placeholder(tf.float32, shape=(None))

        """
        # self.gt_placeholder_n:
         - a 0-1 vector of, which one is the 'on-policy samples'
        """
        self.grouping_placeholder_nn = tf.placeholder(
            tf.float32, shape=(None, None))
        self.grouping2_placeholder_nn = tf.placeholder(
            tf.float32, shape=(None, None))
        self.grouping2_placeholder_next_nn = tf.placeholder(
            tf.float32, shape=(None, None))
        """
        # explaination of grouping matrix nn:
        # -we need to compute the probability of action (from replay buffer) on updated policy
        # -thus we need to group sum: sum_j exp(R(s_i,a_j))
        # -thus we will firstly need to cumsum the exp(R(s,a))
        # -then multiply, e.g., following matrix (vacant == zero):
        [ , ,  ,  ,  ]
        [1,1,-1,  ,  ]
        [ , , 1,-1,-1]
        [ , ,  ,  ,  ]
        [ , ,  , 1, 1]
        this is sum([exp_r1,exp_r2]),sum([exp_r1,exp_r2]), sum(
            [exp_r3]), sum([exp_r4,exp_r5]) , sum([exp_r4,exp_r5])
        """

        x_input_nd = tf.concat(
            [self.all_s_placeholder_nds, self.all_a_placeholder_nda], axis=1)
        x_input_next_nd = tf.concat(
            [self.all_s_placeholder_next_nds, self.all_a_placeholder_next_nda], axis=1)
        self.adv_placeholder_n = tf.placeholder(shape=[None], dtype=tf.float32)
        # self.all_adv_placeholder_n = tf.placeholder(shape=[None],dtype = tf.float32)        x1_input_nd = tf.concat([self.s_placeholder_nds,self.a_placeholder_nda],axis = 1)

        """#############################
            first part: compute loss for train
            PS: each block of 'wrong implementation' is found by a great deal of experiments
        """
        with tf.variable_scope('reward_est'):

            if 1:  # gradient check
                x_input_nd_variable = tf.get_variable("input_variable", shape=(
                    1, x_input_nd.shape[1]), dtype=tf.float32, initializer=tf.constant_initializer(value=0))
                tmpa, tmpb = x_input_nd[:1, :], x_input_nd[1:, :]
                tmpa += x_input_nd_variable
                x_input_nd = tf.concat(values=[tmpa, tmpb], axis=0)

            # self.r_est_n = tf.sigmoid(build_mlp(x_input_nd,1,'reward_est',3,128)[:,0] - 0.3)#slow
            self.r_est_n = build_mlp(x_input_nd, 1, 'q_func', 3, 256)[:, 0]
            # self.r_est_n = build_1dcnn_model(x_input_nd,1,3,128)[:,0]
            q_func_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
            self.r_est_next_n = build_mlp(
                x_input_next_nd, 1, 'target_q_func', 4, 512)[:, 0]
            target_q_func_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

        r_est_next_nn = self.r_est_next_n * self.grouping2_placeholder_next_nn
        inf_mask = 1/self.grouping2_placeholder_next_nn - \
            self.grouping2_placeholder_next_nn
        r_est_next_nn = r_est_next_nn - inf_mask

        y = self.adv_placeholder_n + self.gamma * \
            tf.reduce_max(r_est_next_nn, axis=1) * (1-self.done_placeholder)
        y_pred = tf.reduce_max(
            self.r_est_n * self.gt_placeholder_n * self.grouping2_placeholder_nn, axis=1)
        self.total_loss = tf.reduce_mean(huber_loss(y-y_pred))

        prob_n2 = self.prob_placeholder_n
        # loss = - tf.reduce_mean(self.r_est_n * (self.gt_placeholder_n - prob_n) * self.adv_placeholder_n)

        # self.debug_info = self.gt_placeholder_n - prob_n
        self.debug_info = prob_n2

        """#############################
            on-policy sampling
        """
        self.sample_prob_n = tf.nn.softmax(self.r_est_n)
        self.sample_a_idx = tf.random.categorical(
            tf.expand_dims(self.r_est_n, axis=0), 1)

        """
        # debug for tf.random.categorical
        sample_prob_n = tf.nn.softmax([[-10., -11.]])
        samples = tf.random.categorical([[-10., -11.]], 10000)
        with tf.Session() as sess:
            a,p = sess.run([samples,sample_prob_n])
        a = a[0]
        """


        # self.update_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss,global_step = self.train_global_step)
        # self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss,global_step = self.train_global_step)# <- really bad performance, why?
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # opt = tf.train.AdamOptimizer(self.learning_rate)
        params = tf.trainable_variables()
        params.remove(x_input_nd_variable)
        gd = opt.compute_gradients(self.total_loss, params)

        for i, (grad, var) in enumerate(gd):
            if grad is not None:
                gd[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)

        # clip
        for i, (grad, var) in enumerate(gd):
            if grad is not None:
                gd[i] = (tf.clip_by_norm(grad, 10), var)

        self.gd_debug = opt.compute_gradients(
            self.total_loss, x_input_nd_variable)
        self.update_op = opt.apply_gradients(
            gd, global_step=self.train_global_step)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

    def save(self):
        try:
            os.mkdir(self.save_dir)
        except:
            print('cannot_mkdir')
        if self.openai_Q:
            model_file = os.path.join(self.save_dir, "model")
            save_variables(model_file)
        elif self.openai_a2c:
            model_file = os.path.join(self.save_dir, "model")
            save_variables(model_file)
        elif self.openai_ppo2:  # not sure
            model_file = os.path.join(self.save_dir, "model")
            save_variables(model_file)
        elif self.openai_ddpg:  
            model_file = os.path.join(self.save_dir, "model")
            save_variables(model_file)
        else:
            self.tf_saver.save(self.sess, self.save_dir,
                               global_step=self.train_global_step)

    def __del__(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None
        if self.lms_sensor is not None:
            self.lms_sensor.destroy()
            self.lms_sensor = None
        if self.collision is not None:
            self.collision.destroy()
            self.collision = None
            self.collision_history = []
        if self.laneInvasion is not None:
            self.laneInvasion.destroy()
        if self._vehicle:
            self._vehicle.destroy()

        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _compute_carfollowing_distance(self, waypoint):
        """
        currently a naive one
        """

        ego_vehicle_location = waypoint.transform.location
        ego_vehicle_waypoint = waypoint
        vehicle_list = self.vehicle_list
        dis_list = []

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(
                target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = target_vehicle.get_location()
            dis = np.sqrt((loc.x - ego_vehicle_location.x)**2 +
                          (loc.y - ego_vehicle_location.y)**2)

            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self._vehicle.get_transform().rotation.yaw,
                                        200):
                dis_list.append(dis)

        if len(dis_list) > 0:
            return np.min(dis_list)
        else:
            return 1000

    def _check_location_safety(self, waypoint, threshhold=20.0):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_waypoint = waypoint
        ego_vehicle_location = waypoint.transform.location
        vehicle_list = self.vehicle_list
        print(waypoint.transform.location)
        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(
                target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = target_vehicle.get_location()

            dis = np.sqrt((loc.x - ego_vehicle_location.x)**2 +
                          (loc.y - ego_vehicle_location.y)**2)

            if dis < threshhold:  # too near
                print("too near")
                return False

        return True

    def reset_vehicle_rl(self):

        def set_vehicle_v(vehicle, v):
            vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            vehicle.set_target_velocity(v * tr.get_forward_vector())

        def set_vehicle(vehicle, tr, v):
            vehicle.set_transform(tr)
            vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            vehicle.set_target_velocity(v * tr.get_forward_vector())

        def reset_npc(tr, v, ego_v, worldmap, vehicle_list):
            """
            reset all vehicle to location arround tr, and arround velocity v
            """
            waypoint = worldmap.get_waypoint(tr.location)

            i = 0
            # front vehicle
            if vehicle_list[i].id == ego_v.id:
                i += 1
            set_vehicle(vehicle_list[i], waypoint.next(
                15 + 10*np.random.rand())[0].transform, v * 0.1 + v * np.random.rand())
            vehicle_list[i].set_autopilot(False)
            # vehicle_list[i].set_autopilot(True)
            i += 1

            tmp2 = np.random.rand()

            """find the num of lanes"""
            lane_change = waypoint.lane_change
            if str(lane_change) == 'Both' or str(lane_change) == 'Left':
                # Left side
                if vehicle_list[i].id == ego_v.id:
                    i += 1

                lc2 = waypoint.get_left_lane()
                # can use lc.transform
                lc1 = waypoint.previous(20)[0].get_left_lane()
                if tmp2 > 0.5:
                    set_vehicle(
                        vehicle_list[i], lc1.transform, v * 0.5 + v * np.random.rand())
                else:
                    set_vehicle(vehicle_list[i], lc2.transform, v)
                # set_vehicle(vehicle_list[i], lc.transform, 0)
                vehicle_list[i].set_autopilot(False)
                # vehicle_list[i].set_autopilot(True)
                i += 1
            if str(lane_change) == 'Both' or str(lane_change) == 'Right':
                # Right side
                if vehicle_list[i].id == ego_v.id:
                    i += 1

                lc2 = waypoint.get_right_lane()
                # can use lc.transform
                lc1 = waypoint.previous(20)[0].get_right_lane()
                if tmp2 <= 0.5:
                    set_vehicle(vehicle_list[i], lc1.transform, v)
                else:
                    set_vehicle(
                        vehicle_list[i], lc2.transform, v * 0.5 + v * np.random.rand())
                # set_vehicle(vehicle_list[i], lc.transform, 0)
                vehicle_list[i].set_autopilot(False)
                # vehicle_list[i].set_autopilot(True)
                i += 1

        def respawn_npc(tr, v, ego_v, worldmap, vehicle_list, client):
            """
            This is simply because the autopilot has no 'reset' function
            """
            # delete old vehicle
            n = len(vehicle_list)

            spawn_v_finally(client, vehicle_list)

            waypoint = worldmap.get_waypoint(tr.location)
            transforms = []
            vs = []

            transforms.append(waypoint.next(
                10 + 10*np.random.rand())[0].transform)
            vs.append(v * 0.2 + v * np.random.rand())
            # vehicle_list[i].set_autopilot(False)
            # vehicle_list[i].set_autopilot(True)

            """find the num of lanes"""
            lane_change = waypoint.lane_change
            if str(lane_change) == 'Both' or str(lane_change) == 'Left':
                # Left side
                lc = waypoint.get_left_lane()  # can use lc.transform
                transforms.append(lc.transform)
            vs.append(v * 0.5 + v * np.random.rand())
            #    vehicle_list[i].set_autopilot(False)
            #    vehicle_list[i].set_autopilot(True)

            if str(lane_change) == 'Both' or str(lane_change) == 'Right':
                # Right side
                lc = waypoint.get_right_lane()  # can use lc.transform
                transforms.append(lc.transform)
                vs.append(v * 0.5 + v * np.random.rand())
            #    vehicle_list[i].set_autopilot(False)
            #    vehicle_list[i].set_autopilot(True)
            vehicle_list = spawn_v_try_with_transforms(
                client, transforms, n, autopilot=True)
            [set_vehicle_v(vehicle_list[i], vs[i]) for i in range(len(vs))]
            return vehicle_list

        if 0:
            """ find a safe reborn point """
            waypoint = self._map.get_waypoint(self._vehicle.get_location())
            while True:
                if self._check_location_safety(waypoint):
                    self._vehicle.set_transform(waypoint.transform)
                    tr = waypoint.transform
                    break
                waypoint = random.choice(waypoint.next(1.5))

        else:
            """ or reborn from a designate location """
            """ but not the last one because the vehicle need time to delete """
            self.tmp += 1
            self._vehicle.set_transform(self.reborn_points[self.tmp % 2])
            tr = self.reborn_points[self.tmp % 2]

        # tr,reset_w,reset_v
        reset_w = carla.Vector3D(0.0, 0.0, 0.0)
        reset_v = 20
        self._vehicle.set_target_angular_velocity(reset_w)
        self._vehicle.set_target_velocity(reset_v * tr.get_forward_vector())

        """reset npc without autopilot, and then turn on autopilot at next instant"""
        reset_npc(tr, reset_v, self._vehicle, self._map, self.vehicle_list)

        self.lc_label = -1
        # self.vehicle_list = respawn_npc(tr,reset_v,self._vehicle,self._map,self.vehicle_list,self.client)

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 120.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                    opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._current_waypoint = self._map.get_waypoint(
            self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(
            self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        # self._compute_next_waypoints(k=200)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _random_choose_next_waypoint(self, last_waypoint, next_waypoints):
        if len(next_waypoints) == 1:
            # only one option available ==> lanefollowing
            next_waypoint = next_waypoints[0]
            road_option = RoadOption.LANEFOLLOW
        else:
            # random choice between the possible options
            road_options_list = _retrieve_options(
                next_waypoints, last_waypoint)
            road_option = random.choice(road_options_list)
            next_waypoint = next_waypoints[road_options_list.index(
                road_option)]
        return (next_waypoint, road_option)

    def choose_lanefollow_waypoint(self, last_waypoint, next_waypoints):
        if len(next_waypoints) == 1:
            # only one option available ==> lanefollowing
            next_waypoint = next_waypoints[0]
            road_option = RoadOption.LANEFOLLOW
        else:
            # random choice between the possible options
            road_options_list = _retrieve_options(
                next_waypoints, last_waypoint)
            road_option = RoadOption.LANEFOLLOW
            next_waypoint = next_waypoints[road_options_list.index(
                road_option)]
        return (next_waypoint, road_option)

    def set_sensors(self):
        #        bp.set_attribute('channels', '30')
        #        bp.set_attribute('points_per_second', '10800')
        #        bp.set_attribute('upper_fov', '-30')
        #        bp.set_attribute('lower_fov', '0')
        #        bp.set_attribute('range', '5000')

        # lidar
        self.lidar_points_per_frame = 360  # must manually check!!!!!
        blueprint_lidar = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint_lidar.set_attribute('channels', '8')
        blueprint_lidar.set_attribute('points_per_second', '108000')
        #  blueprint_lidar.set_attribute('points_per_second', '54000')
        blueprint_lidar.set_attribute('upper_fov', '-2')
        blueprint_lidar.set_attribute('lower_fov', '-18')
        blueprint_lidar.set_attribute('range', '50')
        # blueprint_lidar.set_attribute('lower_fov', '10')
        blueprint_lidar.set_attribute('rotation_frequency', '30')
        # blueprint_lidar.set_attribute('sensor_tick', '0.03')#if this is below frequency, there will be missing frame
        if "0.9.8" in carla.__path__[0]:
            lidar = self.world.spawn_actor(blueprint_lidar,
                                           carla.Transform(carla.Location(
                                               x=0, y=0, z=1.8), carla.Rotation(pitch=0, yaw=90)),
                                           attach_to=self._vehicle)  # when yaw = 0, the x-axis (zero degree) of local cordinates is vehicle's left. Anti-cloclwise. when yaw = 90, the x-axis will be vehicle's rear
        elif "0.9.11" in carla.__path__[0]:  # version update
            blueprint_lidar.set_attribute('atmosphere_attenuation_rate', '0.')
            blueprint_lidar.set_attribute('dropoff_general_rate', '0.')
            lidar = self.world.spawn_actor(blueprint_lidar,
                                           # yaw = 180 means vehicle's rear
                                           carla.Transform(carla.Location(
                                               x=-0.5, y=0, z=1.8), carla.Rotation(pitch=0, yaw=180)),
                                           attach_to=self._vehicle)
        #        lidar = self.world.spawn_actor(blueprint_lidar,
        #            carla.Transform(carla.Location(x=0, y = 1, z=0.8), carla.Rotation(pitch=0,yaw = 90)),
        #            attach_to=self._vehicle)
        #        lidar = self.world.spawn_actor(blueprint_lidar,
        #            carla.Transform(carla.Location(x=-2, z=0.8), carla.Rotation(pitch=0,yaw = -90)),
        #            attach_to=self._vehicle)#when yaw = 0, the x-axis (zero degree) of local cordinates is vehicle's left. Anti-cloclwise. when yaw = 90, the x-axis will be vehicle's rear
        #        lidar2 = self.world.spawn_actor(blueprint_lidar,
        #            carla.Transform(carla.Location(x=2, z=0.8), carla.Rotation(pitch=0,yaw = 90)),
        #            attach_to=self._vehicle)

        """will create self.collision, but this sensor must be non-syn """
        self.set_collision_sensor()

        """
        # set collision
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        collision = self.world.spawn_actor(bp, carla.Transform(), attach_to=self._vehicle)
        
        # set lane_invasion
        bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_invasion = self.world.spawn_actor(bp, carla.Transform(), attach_to=self._vehicle)
        """
        self.lidar = lidar

        return lidar, self.collision

    def set_collision_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision = self.world.spawn_actor(
            bp, carla.Transform(), attach_to=self._vehicle)
        weak_self = weakref.ref(self)
        # this is the mysterious reason for "memory fast loss"
        self.collision.listen(
            lambda event: self._on_collision(weak_self, event))
        self.collision_history = []
        return self.collision

    def _on_collision(self, weak_self, event):
        self = weak_self()
        if not self:
            return
        # actor_type = get_actor_display_name(event.other_actor)
        # self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) > 1:
            self.collision_history.pop(0)

    def solve_bezier(self, transform1, transform2, k):
        """
        X = x1-x2
        I = ii+i2
        d = I'X / I'I

        return: k+2 length location sequences, 
                !! with the [0] element as reference point prior than t=0 to calculate diff_yaw
                !! the [1] element is the start point ( current location ) of the trajectory!!!
        """
        def solve_bezier_np(x1, i1, x2, i2):
            X = x2-x1
            I = i1+i2
            d = np.dot(X, I)/np.dot(I, I)
            return d

        #        print(transform1.get_forward_vector())
        #        print(transform2.get_forward_vector())#return value is a base vector
        t1 = transform1.get_forward_vector()
        i1 = np.array([t1.x, t1.y, t1.z])
        t2 = transform2.get_forward_vector()
        i2 = np.array([t2.x, t2.y, t2.z])
        x1 = np.array(
            [transform1.location.x, transform1.location.y, transform1.location.z])
        x2 = np.array(
            [transform2.location.x, transform2.location.y, transform2.location.z])
        d = solve_bezier_np(x1, i1, x2, i2)
        control_point_1 = transform1.location + d*t1
        control_point_2 = transform2.location - d*t2

        t = np.linspace(0, 1, k+1)
        traj = transform1.location*(1-t)**3 + 3*control_point_1*(
            1-t)**2*t + 3*control_point_2*(1-t)*t**2 + transform2.location*t**3
        reference_point = transform1.location - t1
        traj = np.concatenate([[reference_point], traj])

        return traj, control_point_1, control_point_2

    def find_possible_waypoint_old(self, _vehicle, current_w, T, t_min=3.0, t_max=9.0):
        # not used
        return 0

    def find_possible_waypoint_11line(self, _vehicle, current_w, T, d_min=8.0, d_max=24.0, num_traj_per_side=5):

        # not used
        return 0

    def find_possible_waypoint_more_traj(self, _vehicle, current_w, T, d_min=8.0, d_max=24.0, num_traj_per_side=5):
        # did not used
        return 0

    def find_possible_waypoint(self, _vehicle, current_w, T, t_min=3.0, t_max=9.0):

        # lets say we are finding the lane_change info at at t_max
        v = get_speed_ms(_vehicle)

        way_point_min = current_w.next(t_min*v+0.1)[0]
        # way_point_max = current_w.next(t_max*v+0.1)[0]
        way_point_max = current_w.next(10.)[0]

        lane_change = way_point_max.lane_change
        lc_list = []
        label_list = []  # lets
        # print(lane_change)
        traj, c1, c2 = self.solve_bezier(
            _vehicle.get_transform(), way_point_max.transform, T)
        # draw_locations(self._vehicle.get_world(),traj)
        lc_list.append(traj)
        label_list.append(1)

        fw = way_point_max.transform.get_forward_vector()
        # ????this is left vector!!!!
        left_vector = carla.Vector3D(x=fw.y, y=-fw.x, z=fw.z)
        location = way_point_max.transform.location
        rotation = way_point_max.transform.rotation

        # if str(lane_change) == 'Both' or str(lane_change) == 'Left':
        if 1:
            # Left side
            lc = way_point_max.get_left_lane()
            if str(lane_change) == 'Both' or str(lane_change) == 'Left':
                left_lane_transfrom = lc.transform
            else:
                left_location = location + left_vector * \
                    4.5  # says she width of lane is 4meters
                left_lane_transfrom = carla.Transform(left_location, rotation)
            traj, c1, c2 = self.solve_bezier(
                _vehicle.get_transform(), left_lane_transfrom, T)
            # draw_locations(self._vehicle.get_world(),traj)
            draw_locations(self._vehicle.get_world(), [
                           way_point_min.transform.location, way_point_max.transform.location], color=carla.Color(0, 0, 255))
            lc_list.append(traj)
            label_list.append(0)
        if 1:
            # if str(lane_change) == 'Both' or str(lane_change) == 'Right':
            # Right side
            lc = way_point_max.get_right_lane()
            if str(lane_change) == 'Both' or str(lane_change) == 'Right':
                right_lane_transfrom = lc.transform
            else:
                right_location = location - left_vector * 4.5
                right_lane_transfrom = carla.Transform(
                    right_location, rotation)
            traj, c1, c2 = self.solve_bezier(
                _vehicle.get_transform(), right_lane_transfrom, T)
            # draw_locations(self._vehicle.get_world(),traj)
            draw_locations(self._vehicle.get_world(), [
                           way_point_min.transform.location, way_point_max.transform.location], color=carla.Color(0, 0, 255))
            lc_list.append(traj)
            label_list.append(2)
        return lc_list, label_list

        # tx,ty = method.batch_solve_parameter(human_y,human_y1,human_y2,ye,human_x,human_x1,human_x2,xe1,t)

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - \
            len(self._waypoints_queue)
        k = min(available_entries, k)

        current_w = self._map.get_waypoint(self._vehicle.get_location())
        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            # some plot
        #            lane_change = last_waypoint.lane_change
        #            print(lane_change)
        #            print(last_waypoint.get_right_lane())
        #            if last_waypoint.get_right_lane() != None:
        #                print(123)
        #                draw_waypoints(self._vehicle.get_world(),[last_waypoint.get_right_lane()])
        #        #        print(current_w.get_left_lane())
        #            if last_waypoint.get_left_lane() != None:
        #                draw_waypoints(self._vehicle.get_world(),[last_waypoint.get_left_lane()])
            self._waypoints_queue.append(
                self._random_choose_next_waypoint(last_waypoint, next_waypoints))

    def _compute_the_next_waypoint(self):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """

        current_w = self._map.get_waypoint(self._vehicle.get_location())

        next_waypoint = current_w.next(self._sampling_radius)
        return next_waypoint

    def set_global_plan(self, current_plan):
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW
        self._global_plan = True

    def purge_waypoint(self, vehicle_transform, _waypoint_queue, min_distance, max_distance):
        # modified ?
        # purge the buffer of obsolete waypoints, break when the next waypoint
        # is too far away
        max_index = -1
        tmpmin = max_distance

        for i, (waypoint, _) in enumerate(_waypoint_queue):
            dis = distance_vehicle(waypoint, vehicle_transform)
            if dis < tmpmin:
                max_index = i
                tmpmin = dis
            elif dis > max_distance:  # lost route
                break
        if max_index >= 0:
            for i in range(max_index):
                _waypoint_queue.popleft()
            if tmpmin < min_distance:
                _waypoint_queue.popleft()  # the nearest one is already arrived

    def purge_waypointlist(self, vehicle_transform, _waypoint_list, min_distance, max_distance):
        # _waypoint_list : [none](waypoint)

        def distance_vehicle2(cordinate, vehicle_transform):
            loc = vehicle_transform.location
            dx = cordinate.x - loc.x
            dy = cordinate.y - loc.y

            return math.sqrt(dx * dx + dy * dy)

        max_index = -1
        tmpmin = max_distance

        for i, waypoint in enumerate(_waypoint_list):
            dis = distance_vehicle2(waypoint, vehicle_transform)
            if dis < tmpmin:
                max_index = i
                tmpmin = dis
            elif dis > max_distance:  # lost route
                break
        if max_index >= 0:
            _waypoint_list = _waypoint_list[max_index:]

            if tmpmin < min_distance:
                _waypoint_list.pop(0)  # the nearest one is already arrived

    def train_one_step_QAQTATTATQAQ(self):
        # not used
        _, global_step = self.sess.run([self.update_op, self.train_global_step], feed_dict={self.s_placeholder_nds: np.array(
            self.s_batch), self.a_placeholder_nda: np.array(self.a_batch), self.adv_placeholder_n: np.array(self.r_batch)})
        print('global_step:', global_step+1)
        if global_step % 1000 == 0:
            self.save()

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        """
        # copied from cs294-112
            Estimates the advantage function value for each timestep.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        # First, estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # To get the advantage, subtract the V(s) to get A(s, a) = Q(s, a) - V(s)
        # This requires calling the critic twice --- to obtain V(s') when calculating Q(s, a),
        # and V(s) when subtracting the baseline
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing Q(s, a)
        # otherwise the values will grow without bound.
        # YOUR CODE HERE
        Vs_n = self.sess.run(self.critic_prediction,
                             feed_dict={self.sy_ob_no: ob_no})
        Vs_next_n = self.sess.run(self.critic_prediction, feed_dict={
                                  self.sy_ob_no: next_ob_no})
        Qsa_n = re_n + self.gamma * Vs_next_n * (1-terminal_n)
        adv_n = Qsa_n - Vs_n
        return adv_n

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                nothing
        """
        # Use a bootstrapped target values to update the critic
        # Compute the target values r(s, a) + gamma*V(s') by calling the critic to compute V(s')
        # In total, take n=self.num_grad_steps_per_target_update*self.num_target_updates gradient update steps
        # Every self.num_grad_steps_per_target_update steps, recompute the target values
        # by evaluating V(s') on the updated critic
        # Note: don't forget to use terminal_n to cut off the V(s') term when computing the target
        # otherwise the values will grow without bound.
        # YOUR CODE HERE

        if self.local_step_in_critic % self.num_target_updates_every == 0:
            Vs_next_n = self.sess.run(self.critic_prediction, feed_dict={
                                      self.sy_ob_no: next_ob_no})
            print('Value_min_max:', np.min(Vs_next_n), np.max(Vs_next_n))
            Qsa_n = re_n + self.gamma * Vs_next_n * (1-terminal_n)
            # Qsa_n = re_n#debug
            self.local_feed_for_loop = (ob_no, Qsa_n)
            self.local_step_in_critic = 0

        # now
        for i in range(self.num_grade_update_per_step):
            ob_no, Qsa_n = self.local_feed_for_loop
            critic_loss, _ = self.sess.run([self.critic_loss, self.critic_update_op], feed_dict={
                                           self.sy_ob_no: ob_no, self.sy_target_n: Qsa_n})
            self.critic_loss_history.append(critic_loss)
            self.writer.add_scalar(
                'critic_loss', critic_loss, global_step=self.episodes_step)
            if critic_loss > 100:
                print('update_critic:error_loss_too_large')

        self.local_step_in_critic += 1

    def train_one_step_DQN_EBM(self):

        if not self.replay_buffer.can_sample(self.batch_size):
            return 0

        all_obs_batch, all_a_batch, prob_batch, all_rew_batch, all_rew2_batch, grouping, a_label_batch, grouping2, act_batch, obs_batch, rew_batch, next_obs_batch, done_batch, num_a_batch, value_batch, next_value_batch, next_state_return = self.replay_buffer.sample(
            self.batch_size, sars=True)

        """
        # debug for ReplayBuffer
        n = 100
        s_episode = [np.arange(i,i+5) for i in np.arange(n)]
        all_a_episode = [np.eye(3)]*n   #[](n,da)
        prob_episode = [np.array([0,0,1])]*n
        r_episode = np.arange(n)
        a_label_episode = np.ones(100,dtype=np.int32)
        num_traj_list = np.ones(100,dtype=np.int32)*3
        done_array = np.zeros(n,dtype = np.int32)
        done_array[-1] = 1
        rb = ReplayBuffer_QAQ(100)
        rb.store_frames(s_episode, all_a_episode,prob_episode, r_episode, a_label_episode, num_traj_list,done_array)
        all_obs_batch, all_a_batch,prob_batch, all_rew_batch, grouping, a_label_batch,grouping2, act_batch, obs_batch, rew_batch, next_obs_batch, done_batch,num_a_batch = rb.sample(n-1)
        """

    def train_one_step_QAQTATTATQAQ2(self, only_update_critic=False):
        
        # update critic
        if not self.openai_Q:
            if only_update_critic == False or self.use_Value == True:
                if not self.replay_buffer.can_sample(self.batch_size):
                    return 0

                all_obs_batch, all_a_batch, prob_batch, all_rew_batch, all_rew2_batch, grouping, a_label_batch, grouping2, act_batch, obs_batch, rew_batch, next_obs_batch, done_batch, num_a_batch, value_batch, next_value_batch, next_state_things = self.replay_buffer.sample(
                    self.batch_size, sars=True)
                next_all_obs_batch, next_all_a_batch, _, _, _, _, next_a_label_batch, next_grouping2, next_act_batch, _, _, _, _, _, _, _ = next_state_things
                # tmp:only use all_rew_batch

            """
            # debug for ReplayBuffer
            n = 100
            s_episode = [np.arange(i,i+5) for i in np.arange(n)]
            all_a_episode = [np.eye(3)]*n   #[](n,da)
            prob_episode = [np.array([0,0,1])]*n
            r_episode = np.arange(n)
            a_label_episode = np.ones(100,dtype=np.int32)
            num_traj_list = np.ones(100,dtype=np.int32)*3
            done_array = np.zeros(n,dtype = np.int32)
            done_array[-1] = 1
            rb = ReplayBuffer_QAQ(100)
            rb.store_frames(s_episode, all_a_episode,prob_episode, r_episode, a_label_episode, num_traj_list,done_array)
            all_obs_batch, all_a_batch,prob_batch, all_rew_batch, grouping, a_label_batch,grouping2, act_batch, obs_batch, rew_batch, next_obs_batch, done_batch,num_a_batch = rb.sample(n-1)
            """

            if self.ebm == 1:
                if self.use_Value == 1:
                    self.update_critic(
                        obs_batch, next_obs_batch, rew_batch, done_batch)
                if only_update_critic == True:
                    return 0

            if only_update_critic == True:
                return 0

        # if not only update critic:
        print("""train""")
        if self.ebm:
            if self.use_Value == 1:

                # estimate advantage ===================
                adv_n = self.estimate_advantage(
                    obs_batch, next_obs_batch, rew_batch, done_batch)

                if self.normalize_advantages:
                    """they said emperically this reduce variance"""
                #    adv_n =  adv_n - np.mean(adv_n)
                    adv_n = (adv_n - np.mean(adv_n)) / \
                        np.std(adv_n)  # YOUR_HW2 CODE_HERE

                if np.isnan(adv_n).any():
                    print('errorrrrr')

                # here, need to convert the adv_n to all_adv_n
                all_adv_n = []

                def fun2(num_a_batch, obs_batch, all_obs_batch):
                    for (i, _) in enumerate(num_a_batch):
                        all_obs_batch += [obs_batch[i]] * num_a_batch[i]
                fun2(num_a_batch, adv_n, all_adv_n)

                feed_dict = {self.all_s_placeholder_nds: np.array(all_obs_batch),
                             self.all_a_placeholder_nda: np.array(all_a_batch),
                             self.gt_placeholder_n: np.array(a_label_batch),
                             self.adv_placeholder_n: np.array(all_adv_n),
                             self.grouping_placeholder_nn: np.array(grouping),
                             self.grouping2_placeholder_nn: np.array(grouping2),
                             self.prob_placeholder_n: np.array(prob_batch)
                             }
            elif self.Q_learning == 1:
                feed_dict = {self.all_s_placeholder_nds: np.array(all_obs_batch),
                             self.all_s_placeholder_next_nds: np.array(next_all_obs_batch),
                             self.all_a_placeholder_nda: np.array(all_a_batch),
                             self.all_a_placeholder_next_nda: np.array(next_all_a_batch),
                             self.gt_placeholder_n: np.array(a_label_batch),
                             self.gt_placeholder_next_n: np.array(next_a_label_batch),
                             self.adv_placeholder_n: np.array(rew_batch),
                             self.grouping2_placeholder_nn: np.array(grouping2),
                             self.grouping2_placeholder_next_nn: np.array(next_grouping2),
                             self.prob_placeholder_n: np.array(prob_batch),
                             self.done_placeholder: np.array(done_batch)
                             }
            else:

                if self.normalize_advantages:
                    """they said emperically this reduce variance"""
                    all_rew_batch = all_rew_batch - np.mean(rew_batch)
                #    all_rew_batch = ( all_rew_batch - np.mean(all_rew_batch) )/np.std(all_rew_batch) # YOUR_HW2 CODE_HERE

                # first objective
                feed_dict = {self.all_s_placeholder_nds: np.array(all_obs_batch),
                             self.all_a_placeholder_nda: np.array(all_a_batch),
                             self.gt_placeholder_n: np.array(a_label_batch),
                             self.adv_placeholder_n: np.array(all_rew_batch),
                             self.grouping_placeholder_nn: np.array(grouping),
                             self.grouping2_placeholder_nn: np.array(grouping2),
                             self.prob_placeholder_n: np.array(prob_batch)
                             }

            # update actor =================================
            for i in np.arange(1):
                params_gd, gd_debug, total_loss, _, global_step, r, prob = self.sess.run(
                    [self.gd, self.gd_debug, self.total_loss, self.update_op, self.train_global_step, self.r_est_n, self.debug_info], feed_dict=feed_dict)
                print('global_step:', global_step+1)

            #    self.params_gd_history.append(params_gd)
                self.gd_debug_history.append(gd_debug)
                self.debug_r_history.append(r)
                self.debug_r2_history.append(all_rew_batch)
                self.debug_prob_history.append(prob)
                self.writer.add_scalar(
                    'loss', total_loss, global_step=self.steps)
                self.loss_history.append(total_loss)
                self.target_update_counter += 1
            if self.Q_learning == 1:
                if self.target_update_counter > self.num_grad_steps_per_target_update:
                    self.sess.run(self.update_target_fn)
                    self.target_update_counter = 0

        elif self.openai_Q:
            act, train, update_target, debug, exploration, beta_schedule, gamma, param_noise, exploration_fraction, reset, saved_mean_reward = self.openai_things
            prioritized_replay = self.alg_kwargs['prioritized_replay']
            prioritized_replay_eps = self.alg_kwargs['prioritized_replay_eps']
            batch_size = self.alg_kwargs['batch_size']

            if self.steps % self.alg_kwargs['train_freq'] == 0:
                if prioritized_replay:
                    experience = self.openai_replay_buffer.sample(
                        batch_size, beta=beta_schedule.value(self.steps))
                    (obses_t, actions, rewards, obses_tp1,
                     dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = self.openai_replay_buffer.sample(
                        batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards,
                                  obses_tp1, dones, weights)
                self.writer.add_scalar('td_error', np.mean(
                    td_errors), global_step=self.steps)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    self.openai_replay_buffer.update_priorities(
                        batch_idxes, new_priorities)
            if self.steps % self.alg_kwargs['target_network_update_freq'] == 0:
                # Update target network periodically.
                update_target()

        elif self.openai_a2c:
            model = self.openai_things
            _, next_values, _, _ = model.train_model.step(
                next_obs_batch, M=done_batch)  # use new values
            #    next_values = model.value(next_obs_batch, M=done_batch)
            _, values, states, _ = model.train_model.step(
                obs_batch)  # use new values
            print(np.min(values), np.max(values))
            rew_batch = rew_batch + self.gamma * \
                next_values * (1-done_batch)  # in fact,
            ev = explained_variance(values, rew_batch)
            self.writer.add_scalar('explained_variance',
                                   ev, global_step=self.steps)

            policy_loss, value_loss, policy_entropy = model.train(
                obs_batch, None, rew_batch, None, act_batch, values)
            self.writer.add_scalar(
                'policy_loss', policy_loss, global_step=self.steps)
            self.writer.add_scalar(
                'value_loss', value_loss, global_step=self.steps)
            self.writer.add_scalar(
                'policy_entropy', policy_entropy, global_step=self.steps)
            # simulate the runner.run()
        # obs, states, rewards, masks, actions, values, epinfos = runner.run()
        elif self.openai_ppo2:
            model, nminibatches, nupdates, lr, cliprange, noptepochs, nbatch_train, epinfobuf = self.openai_things
            assert self.batch_size % nminibatches == 0
            # Start timer
            tstart = time.perf_counter()
            frac = 1.0 - (self.train_steps - 1.0) / nupdates
            # Calculate the learning rate
            lrnow = lr(frac)
            # Calculate the cliprange

            def constfn(val):
                def f(_):
                    return val
                return f
            if isinstance(cliprange, float):
                cliprange = constfn(cliprange)
            else:
                assert callable(cliprange)
            cliprangenow = cliprange(frac)

            # epinfobuf.extend(epinfos)
            # Here what we're going to do is for each minibatch calculate the loss and append it.
            mblossvals = []

            # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(self.batch_size)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, self.batch_size, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    _, values, states, neglogpacs = model.train_model.step(
                        obs_batch)  # use new values -- not very sure about this
                    print(np.min(values), np.max(values))
                    _, next_values, _, _ = model.train_model.step(
                        next_obs_batch, M=done_batch)  # use new values
                    # next_values = model.value(next_obs_batch, M=done_batch)

                    # the comment says rew is R+rV(S'), but the code in runner is different !!!!!!!!!!!!!!!!!!!??????????????!!!!!!!!!!!
                    rew_batch = rew_batch + self.gamma * \
                        next_values * (1-done_batch)
                    slices = (arr[mbinds] for arr in (obs_batch, rew_batch, np.zeros(
                        nbatch_train), act_batch, values, neglogpacs))
                    mblossvals.append(model.train(
                        lrnow, cliprangenow, *slices))
            # Feedforward --> get losses --> update
            lossvals = np.mean(mblossvals, axis=0)
            # End timer
            tnow = time.perf_counter()
            # Calculate the fps (frame per second)
            fps = int(self.batch_size / (tnow - tstart))
            ev = explained_variance(values, rew_batch)
            self.writer.add_scalar('explained_variance',
                                   ev, global_step=self.steps)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                self.writer.add_scalar(
                    'loss/' + lossname, lossval, global_step=self.steps)

        elif self.openai_ddpg:
            if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                distance = self.agent.adapt_param_noise()
                epoch_adaptive_distances.append(distance)

            cl, al = self.agent.train()
            epoch_critic_losses.append(cl)
            epoch_actor_losses.append(al)
            self.agent.update_target_net()

        else:
            ob_no = all_obs_batch
            ac_na = act_batch
            adv_n = rew_batch

            if self.normalize_advantages:
                """they said emperically this reduce variance"""
                adv_n = (adv_n - np.mean(adv_n)) / \
                    np.std(adv_n)  # YOUR_HW2 CODE_HERE

            feed_dict = {self.sy_ob_no: ob_no,
                         self.sy_ac_na: ac_na, self.sy_adv_n: adv_n}
            for i in np.arange(1):
                _, global_step = self.sess.run(
                    [self.update_op, self.train_global_step], feed_dict=feed_dict)
                print('global_step:', global_step+1)

            # self.debug_r_history.append(r)

        if self.openai_Q or self.openai_a2c or self.openai_ppo2 or self.openai_ddpg:
            if self.steps % 1000 == 0:
                self.save()
        else:
            if global_step % 1000 == 0:
                self.save()
        self.train_steps += 1

    def conclude_one_episode(self, r_final, done, collision):
        """
        flush to replay_buffer
        """
        r1_final, r2_final, r3_final = r_final

        def cumsum_reward(r_episode_tmp, T_reward_computing):
            n = len(r_episode_tmp)
            r_n = np.cumsum(r_episode_tmp[::-1])[::-1]
            if n > T_reward_computing:
                r_n[:-T_reward_computing] = r_n[:-T_reward_computing] - \
                    r_n[T_reward_computing:]
            return r_n

        def process_reward(self, r_episode, r_final, T_reward_computing, instant_reward=1):
            """
            about 'instant reward':
                for some reward like 'jerk' or 'high speed', it can be received as long as the action is yield. => instant_reward == 0
                Other reward like 'collision', it needs waiting for next timestep to receive => instant_rewarc == 0
            """
            r_episode_tmp = list(r_episode)
            if instant_reward == 0:
                # shift one-step to the future
                r_episode_tmp.append(r_final)
                r_episode_tmp = r_episode_tmp[1:]
            if not self.use_Value and not self.Q_learning:
                r_n = cumsum_reward(r_episode_tmp, T_reward_computing)
            else:
                r_n = cumsum_reward(r_episode_tmp, T_reward_computing)
                # r_n = np.array(r_episode_tmp)
            r_sum = np.sum(r_episode_tmp)
            return r_n, r_sum

        r1_n, sum1 = process_reward(
            self, self.r_episode1, r1_final, 1, instant_reward=1)
        r2_n, sum2 = process_reward(
            self, self.r_episode2, r2_final, 1, instant_reward=1)
        r3_n, sum3 = process_reward(
            self, self.r_episode3, r3_final, self.T_reward_computing, instant_reward=0)
        r_n = r1_n + r2_n + r3_n  # v,w,collision
        # currently for ebm, only r3_n is added to buffer
        average_r = (sum1+sum2+sum3)/len(r1_n)
        av1 = sum1/len(r1_n)
        av2 = sum2/len(r2_n)
        av3 = sum3/len(r3_n)
        self.writer.add_scalar('average_r_of_each_episode',
                               average_r, global_step=self.episodes_step)
        self.writer.add_scalar(
            'average_r1', av1, global_step=self.episodes_step)
        self.writer.add_scalar(
            'average_r2', av2, global_step=self.episodes_step)
        self.writer.add_scalar(
            'average_r3', av3, global_step=self.episodes_step)
        self.r_history.append(average_r)
        self.r1_history.append(av1)
        self.r2_histroy.append(av2)
        self.r3_history.append(av3)

        # ?????????????????/try!!!!!!!!!!!!!!!!
        if av2 > -0.02:
            r_n = r3_n

        # r_n = np.array(r2_n) + np.array(r3_n)
        # r_n = np.array(r3_n)
        n = len(r_n)

        if self.openai_Q:
            if collision:
                done_array = np.zeros(n, dtype=np.int32)
                done_array[-1] = 1
                for i in range(n):
                    self.openai_replay_buffer.add(
                        self.s_episode[i], self.a_label_episode[i], r_n[i], self.s_episode[(i+1) % n], done_array[i])
            else:
                done_array = np.zeros(n-1, dtype=np.int32)
                for i in range(n-1):
                    self.openai_replay_buffer.add(
                        self.s_episode[i], self.a_label_episode[i], r_n[i], self.s_episode[(i+1) % n], done_array[i])
            self.r_episode1 = []
            self.r_episode2 = []
            self.r_episode3 = []
            self.a_label_episode = []
            self.s_episode = []

        elif self.openai_a2c or self.openai_ppo2:
            if collision:
                done_array = np.zeros(n, dtype=np.int32)
                # for 'done' but not 'collision', it need special treatment
                collision_array = np.zeros(n, dtype=np.int32)
                done_array[-self.T_reward_computing:] = 1
                collision_array[-self.T_reward_computing:] = 1
                self.replay_buffer.store_frames(self.s_episode, self.all_a_episode, None, r_n, r2_n, self.a_label_episode,
                                                self.num_traj_list, done_array, collision_array, value_episode=self.value_episode)
            else:
                done_array = np.zeros(n, dtype=np.int32)
                done_array[-1] = 1
                # for 'done' but not 'collision', it need special treatment
                collision_array = np.zeros(n, dtype=np.int32)
                self.replay_buffer.store_frames(self.s_episode, self.all_a_episode, None, r_n, r2_n, self.a_label_episode,
                                                self.num_traj_list, done_array, collision_array, value_episode=self.value_episode)
            self.r_episode1 = []
            self.r_episode2 = []
            self.r_episode3 = []
            self.a_label_episode = []
            self.s_episode = []
            self.num_traj_list = []
            self.all_a_episode = []
            self.value_episode = []

        elif done:  # all flush to buffer
            done_array = np.zeros(n, dtype=np.int32)
            # for 'done' but not 'collision', it need special treatment
            collision_array = np.zeros(n, dtype=np.int32)
            if n > 0:
                if not self.use_Value and not self.Q_learning:
                    done_array[-1] = 1
                    if collision:
                        collision_array[-1] = 1
                else:
                    done_array[-1] = 1
                    if collision:
                        collision_array[-self.T_reward_computing:] = 1
                        done_array[-self.T_reward_computing:] = 1
            self.replay_buffer.store_frames(self.s_episode, self.all_a_episode, self.prob_episode,
                                            r_n, r2_n, self.a_label_episode, self.num_traj_list, done_array, collision_array)
            self.s_episode = []
            self.all_a_episode = []
            # self.r_episode = []
            self.r_episode1 = []
            self.r_episode2 = []
            self.r_episode3 = []
            self.a_label_episode = []
            self.prob_episode = []
            self.a_episode = []
            self.num_traj_list = []

        else:  # reserve the last 'self.T_reward_computing' frames
            done_array = np.zeros(n-self.T_reward_computing, dtype=np.int32)
            # for 'done' but not 'collision', it need special treatment
            collision_array = np.zeros(
                n-self.T_reward_computing, dtype=np.int32)
            self.replay_buffer.store_frames(self.s_episode[:-self.T_reward_computing], self.all_a_episode[:-self.T_reward_computing], self.prob_episode[:-self.T_reward_computing], r_n[:-
                                            self.T_reward_computing], r2_n[:-self.T_reward_computing], self.a_label_episode[:-self.T_reward_computing], self.num_traj_list[:-self.T_reward_computing], done_array, collision_array)
            self.s_episode = self.s_episode[-self.T_reward_computing:]
            self.all_a_episode = self.all_a_episode[-self.T_reward_computing:]
            # self.r_episode = self.r_episode[-self.T_reward_computing:]
            self.r_episode1 = self.r_episode1[-self.T_reward_computing:]
            self.r_episode2 = self.r_episode2[-self.T_reward_computing:]
            self.r_episode3 = self.r_episode3[-self.T_reward_computing:]
            self.a_label_episode = self.a_label_episode[-self.T_reward_computing:]
            self.prob_episode = self.prob_episode[-self.T_reward_computing:]
            self.num_traj_list = self.num_traj_list[-self.T_reward_computing:]

        self.episodes_step += 1

    def run_step(self, debug=True, training=False):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        current_w = self._map.get_waypoint(self._vehicle.get_location())
        lc_list, lc_label = self.find_possible_waypoint(
            self._vehicle, current_w, T=self.T)
        # each element is a 'traj', which is array of 3DVector
        # debug
        # lc_list = lc_list*5
        # lc_label = lc_label*5

        """^ the [0] is reference point!!! the [1] is the start point!!!"""

        # A new fix for 'long-traj': same_traj_reward
        if self.same_traj_reward == 1:
            is_same_traj = np.float32(np.array(lc_label) == self.lc_label)

        # update waypoint
        vehicle_transform = self._vehicle.get_transform()
        self.purge_waypointlist(
            vehicle_transform, self._waypoint_bufferlist, self._min_distance, self._max_distance)

        # compute action value
        # need window_size
        def x2yaw(x, y, dt):
            dy = np.diff(y, axis=-1)
            dx = np.diff(x, axis=-1)
            v = np.sqrt(dx**2+dy**2)/dt
            yaw = np.arctan2(dy, dx)
            return v, yaw

        def traj2action_old(traj):
            x = np.array([x.x for x in traj])  # length T+2
            y = np.array([x.y for x in traj])
            v, yaw = x2yaw(x, y, 1)  # length T+1
            v = v[1:]
            yaw_change = yaw[1:] - yaw[0]  # The difference is here
            return v, yaw_change

        def traj2action(traj):
            x = np.array([x.x for x in traj])  # length T+2
            y = np.array([x.y for x in traj])
            v, yaw = x2yaw(x, y, 1)  # length T+1
            v = v[1:]
            yaw_change = yaw[1:] - yaw[:-1]  # length 0
            return [yaw_change*8]

        """
        caution: the return of traj2action( 50-length ) is 49-length, because of 'np.diff'
        """
        # traj_list_action = [ np.concatenate(traj2action(traj))   for traj in lc_list]#(traj_num,)(Da)
        # traj_list_startpoint = np.cumsum([len(traj) for traj in lc_list])

        # a = np.concatenate(traj_list_action) #(2,sum_of_T)
        v = get_speed_ms(self._vehicle)
        c = self._vehicle.get_control()  # it seems this method cannot get last control!!!
        # steer_pre = c.steer

        # r1 = (v - 20)/16
        r1 = 0
        r2 = 0
        # r2 = -c.steer
        r3 = 0

        self.steps += 1
        self.buffer_every_maximum += 1

        [i.set_autopilot(True) for i in self.vehicle_list]
        
        if len(self.collision_history) > 0:  # collision
            print('collision !')
            self.collision_history.pop(0)
            self.collision_log.append(self.steps)
            self.writer.add_scalar('collision_history', len(
                self.collision_log), global_step=self.steps)

            # end this episode and compute r
            if not self.ebm:
                r3 = -10
            else:
                if self.Q_learning == True:
                    r3 = -200  # to ensure intial r is above this value
                else:
                    r3 = -10
            self.conclude_one_episode([r1, r2, r3], done=True, collision=True)
            self.reset_vehicle_rl()
            self.buffer_every_maximum = 0
            return self.default_contro
        elif self.buffer_every_maximum % self.buffer_every == 0:
            self.conclude_one_episode([r1, r2, r3], done=True, collision=False)
            self.reset_vehicle_rl()
            self.buffer_every_maximum = 0
            return self.default_control
            # self.conclude_one_episode(r,done = False)

        # if only update critic
        # if self.steps % self.train_every == 0 and self.steps < 1000: #debug for only update critic
        if self.steps % self.train_every == 0:
            only_update_critic = False
        else:
            only_update_critic = True

        # when to update
        if self.openai_Q:
            if self.openai_replay_buffer.__len__() > self.batch_size and self.steps > self.alg_kwargs['learning_starts']:
                self.train_one_step_QAQTATTATQAQ2(only_update_critic)
            else:
                print("""not enough samples, skipped iteration""")
        else:
            if self.replay_buffer.can_sample(self.batch_size):
                self.train_one_step_QAQTATTATQAQ2(only_update_critic)
            else:
                print("""not enough samples, skipped iteration""")

        # process lidar_frame
        lidar_frame = self.lidar_frame
        points = raw_data2points(lidar_frame)  # (n,3)

        #  self.lidar.transform.rotation  #rotation (carla.Rotation – degrees (pitch, yaw, roll))
        # x-axis is the rear side of the vehicle

        yaw, d, self.lidar_points = rescue_lms(
            points, self.lidar_points_per_frame, return_points=True)
        yaw2, d2, img = rescue_lms_grid(
            points, self.lidar_points_per_frame, return_points=False)
        lx, ly = np.shape(img)
        img = img[(int)(lx - self.grid_size[0])//2:(int)(lx + self.grid_size[0]) //
                  2, (int)(ly-self.grid_size[1])//2:(int)(ly+self.grid_size[1])//2]
        img = np.reshape(img, [-1])

        #   rl_s_ds = rl_s_ds2#to change
        #   rl_s_ds = np.concatenate([ d/25.,[v/20.],[self.steer_pre]])
        rl_s_ds = np.concatenate(
            [2 - d*2/(d+10), [v/20.], [self.steer_pre]])  # old
        rl_s_ds2 = np.concatenate([img, [v/20.], [self.steer_pre]])

        self.s_episode.append(rl_s_ds)
        # self.r_episode.append(r)#this is the reward because of last action

        # choose action
        if self.ebm == 1:  # sampling from policy
            # get observation
            n = len(lc_list)
            """caution : the return of traj2action( 51-length ) is 49-length, because of 'np.diff' and reference point"""
            if self.same_traj_reward == 1:
                rl_a_da_list = [np.concatenate(list(traj2action(
                    traj)) + [np.array([is_same_traj[i]])]) for (i, traj) in enumerate(lc_list)]
            else:
                # (num_of_traj)(2 * T))
                rl_a_da_list = [np.reshape(traj2action(traj), [-1])
                                for traj in lc_list]
            rl_a_nda = np.array(rl_a_da_list)

            # lets try a simple one
            # rl_a_nda = np.array(label_list)
            """ ^ a need normalization """
            rl_s_nds = np.repeat([rl_s_ds], n, axis=0)  # (n,ds)
            sample_a_idx, prob_n = self.sess.run([self.sample_a_idx, self.sample_prob_n], feed_dict={
                                                 self.all_s_placeholder_nds: rl_s_nds, self.all_a_placeholder_nda: rl_a_nda})
            self.writer.add_scalars(
                'prob', {'1': prob_n[0], '2': prob_n[1]}, global_step=self.steps)
            if self.Q_learning == 1:
                if np.random.random() > self.epsilon:
                    sample_a_idx = np.argmax(prob_n)
                else:
                    print('RRRRRRRRRRRRRandom:', sample_a_idx)
                    sample_a_idx = np.random.randint(0, n)

                if self.epsilon > self.MIN_EPSILON:
                    self.epsilon *= self.EPSILON_DECAY
                    self.epsilon = max(self.MIN_EPSILON, self.epsilon)

            else:
                sample_a_idx = sample_a_idx[0, 0]
            #   Buffering the waypoints
            self._waypoint_bufferlist += list(lc_list[sample_a_idx])

            """let's set target location on the selected trajectory,e.g. the '5'th coordinate """
            self.target_location = lc_list[sample_a_idx][5]

            # visulization
            if 1:
                for i, traj in enumerate(lc_list):
                    color = plt.cm.rainbow(float(prob_n[i]))[:3]
                    color = carla.Color(
                        int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    draw_locations(self._vehicle.get_world(),
                                   traj, color=color)

            # reward:
            self.a_label_episode.append(sample_a_idx)  # (n,)
            self.prob_episode.append(prob_n)
            self.all_a_episode.append(rl_a_nda)  # list of (n,da)
            self.num_traj_list.append(n)

        elif self.openai_Q == 1:  # openai Q learning
            obs = rl_s_ds

            # === restore the runtime parames
            act, train, update_target, debug, exploration, beta_schedule, gamma, param_noise, exploration_fraction, reset, saved_mean_reward = self.openai_things
            # === fake the running environment
            t = self.steps

            """#need work
            td = checkpoint_path or td
            
            
            model_file = os.path.join(td, "model")
            model_saved = False
            if tf.train.latest_checkpoint(td) is not None:
                load_variables(model_file)
                logger.log('Loaded model from {}'.format(model_file))
                model_saved = True
            elif load_path is not None:
                load_variables(load_path)
                logger.log('Loaded model from {}'.format(load_path))
            """
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = - \
                    np.log(1. - exploration.value(t) +
                           exploration.value(t) / float(self.n_a))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            action = act(np.array(obs)[None],
                         update_eps=update_eps, **kwargs)[0]

            if self.Q_with_trajectory == 0:
                # end-to-end Q, directly control the steering wheel
                # let's define several steer number:
                steer = [-0.2, 0, 0.2]

                tranform1 = self._vehicle.get_transform()
                forward = tranform1.get_forward_vector()
                left = carla.Vector3D(-forward.y, forward.x, forward.z)

                self.target_location = tranform1.location + \
                    forward * 10 + left*(-steer[action])

            else:
                # select trajectory
                self.target_location = lc_list[action][5]
            #    for i,traj in enumerate(lc_list):
            #        color = plt.cm.rainbow(float(prob_n[i]))[:3]
            #        color = carla.Color(int(color[0]*255),int(color[1]*255),int(color[2]*255))
            #        draw_locations(self._vehicle.get_world(),traj,color = color)

            # new_obs, rew, done, _ = env.step(env_action)#don't need this
            # Store transition in the replay buffer.

            self.a_label_episode.append(action)

        elif self.openai_a2c == 1 or self.openai_ppo2 == 1:
            # mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
            if self.openai_a2c:
                model = self.openai_things
            else:
                model = self.openai_things[0]
            actions, values, states, neglogpacs = model.step(rl_s_ds)
            action = actions[0]
            value = values[0]
            # let's define several steer number:
            steer = [-0.2, 0, 0.2]

            tranform1 = self._vehicle.get_transform()
            forward = tranform1.get_forward_vector()
            left = carla.Vector3D(-forward.y, forward.x, forward.z)

            self.target_location = tranform1.location + \
                forward * 10 + left*(-steer[action])

            self.a_label_episode.append(actions)
            self.value_episode.append(value)

            self.all_a_episode.append(np.array([0, 1, 2]))  # just for occupy
            self.num_traj_list.append(3)

        elif self.openai_ddpg == 1:
            obs = rl_s_ds
            action, q, _, _ = self.agent.step(obs, apply_noise=True, compute_Q=True)
            tranform1 = self._vehicle.get_transform()
            forward = tranform1.get_forward_vector()
            left = carla.Vector3D(-forward.y, forward.x, forward.z)

            self.target_location = tranform1.location + \
                forward * 10 + left*(-action)

            self.a_label_episode.append(action)
        
        else:  # sampling from policy but of pg
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={
                               self.sy_ob_no: rl_s_ds.reshape([1, -1])})  # YOUR CODE HERE
            ac = ac[0]/2
            tranform1 = self._vehicle.get_transform()
            forward = tranform1.get_forward_vector()
            left = carla.Vector3D(-forward.y, forward.x, forward.z)
            print('left:', ac[0])
            self.target_location = tranform1.location + \
                forward * 10 + left*(-ac[0])

            # visulization
            if 1:
                draw_locations(self._vehicle.get_world(),
                               [self.target_location])

            self.a_label_episode.append(0)  # (n,)
            self.all_a_episode.append(np.array([ac]))  # list of (n,da)
            self.num_traj_list.append(1)

        if 0:  # easy_simple_stupid_blind_drive
            cnt = 0
            if lc_list == []:
                cnt = 0
                if not self._global_plan:
                    next_waypoint = self._compute_the_next_waypoint()
                    self.target_location = next_waypoint[0].transform.location
            else:

                # compute the reward of each
                if len(lc_list) == 1:
                    self.target_location = lc_list[0][5]
                else:
                    if cnt % 2000 == 0:
                        rand = np.random.randint(2)
                    cnt += 1
                    self.target_location = lc_list[rand][5]

        #    if len(self._waypoints_queue) == 0 is None:
        #        return self.default_control

            #   Buffering the waypoints
        #    if not self._waypoint_buffer:
        #        for i in range(self._buffer_size):
        #            if self._waypoints_queue:
        #                self._waypoint_buffer.append(
        #                    self._waypoints_queue.popleft())
        #            else:
        #                break

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(
            self._vehicle.get_location())
        # target waypoint
        # self.target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # find possible waypoint from t_min to t_max

        # move using PID controllers

        # dis = self._compute_carfollowing_distance(self._current_waypoint)
        dis = min(d[self.lidar_points_per_frame//2],
                  d[(self.lidar_points_per_frame-1)//2])  # use the middle lms
        print(dis, " kkk ", self.target_location)

        control = self._vehicle_controller.run_step(
            min(max(dis*1.6-8, 0), 80), self.target_location)

        # action
        if (self.openai_Q and not self.Q_with_trajectory) or self.openai_a2c:
            
            control.steer = steer[action]
            print('action ', action, ' control steer: ', steer[action])
        elif self.openai_ddpg:
            control.steer = action
        

        if not self.ebm and not self.openai_Q and not self.openai_a2c and not self.openai_ppo2 \
                                        and not self.openai_ddpg:
            control.steer = float(ac[0]/2)
            r1 = 1
            r2 = 1-np.abs(control.steer)
        elif self.ebm:
            #    r1 = (v - 36)/16
            r1 = 0
            if self.same_traj_reward == 1:
                if self.lc_label == lc_label[sample_a_idx]:
                    r2 = 0
                else:
                    r2 = -0.2
                    # is the same traj with last timestep
                self.lc_label = lc_label[sample_a_idx]
            else:
                #    r2 = (-np.abs(control.steer))/2
                r2 = 0
        elif self.openai_Q or self.openai_a2c:

            #    r1 = (v - 36)/16
            r1 = 0
            # r2 = (-np.abs(control.steer))/2
            r2 = 0
            # control = self._vehicle_controller.run_step( 36 ,self.target_location)
            # r1 = (v - 10)/(2*self.T_reward_computing)
            # r2 = 1-np.abs(self.steer_pre - control.steer)**2
            # r2 = 1-np.abs(self.steer_pre - control.steer)*2
        elif self.openai_ddpg:
            r1 = 0
            r2 = 0

        self.r_episode1.append(r1)  # this is the reward because of last action
        self.r_episode2.append(r2)
        self.r_episode3.append(r3)

        self.steer_pre = control.steer

        # purge the buffer of obsolete waypoints
        # vehicle_transform = self._vehicle.get_transform()
        # self.purge_waypoint(vehicle_transform,self._waypoint_buffer,self._min_distance,self._max_distance)
        # self.purge_waypoint(vehicle_transform,self._waypoints_queue,self._min_distance,self._max_distance)

        if debug:
            draw_locations(self._vehicle.get_world(), [self.target_location])

        return control


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
