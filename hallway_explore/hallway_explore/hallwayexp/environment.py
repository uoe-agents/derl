from enum import Enum
import logging

import numpy as np
import gym
from gym.utils import seeding


class Action(Enum):
    LEFT = 0
    STAY = 1
    RIGHT = 2


class HallwayExplore(gym.Env):
    """
    A class that contains gym environment for Hallway Exploration Challenge

    1D exploration problem
    ###############################################################
    #         A                  G                                #
    ###############################################################
    A = Agent
    G = Goal

    Agent can move left, right or stay. Has to reach goal G to get reward
    and stay there further to get more reward until the episode ends.
    """
    def __init__(
        self,
        goal_left_length,
        goal_right_length,
        episode_length,
        randomise_start_location=False,
        left_reward=0.0,
        stay_reward=-0.01,
        right_reward=-0.01,
        goal_reward=1.0,
        goal_stay_duration=10,
    ):
        """
        Create environment
        :param goal_left_length (int): length of hallway to the left of goal (> 0)
        :param goal_right_length (int): length of hallway to the right of goal (>= 0)
            for = 0 very similar to DeepSea with N=goal_left_length
        :param episode_length (int): length of episode (>= goal_left_length)
            for = goal_left_length very similar to DeepSea
        :param randomise_start_location (bool): flag whether start location of agent
            at beginning of each episode should be random within the hallway (otherwise start on
            left)
        :param left_reward (float): reward for action to go left (default 0.0)
        :param stay_reward (float): reward for action to stay (default -0.01)
        :param right_reward (float): reward for action to go right (default -0.01)
        :param goal_reward (float): reward for reaching goal location (default 1.0)
        :param goal_stay_duration (int): number of steps agents can stay at goal location to get
            goal_reward again (default 10)
        """
        self.logger = logging.getLogger(__name__)
        self.seed()

        self.goal_left_length = goal_left_length
        self.goal_right_length = goal_right_length
        self.episode_length = episode_length
        self.randomise_start_location = randomise_start_location

        self.action_space = gym.spaces.Discrete(3)
        shape = (self.goal_left_length + 1 + self.goal_right_length, )
        low = np.full(shape, 0)
        high = np.full(shape, 1)
        self.observation_space = gym.spaces.Box(-low, high, shape, dtype=np.int32)

        self.left_reward = left_reward
        self.stay_reward = stay_reward
        self.right_reward = right_reward
        self.goal_reward = goal_reward

        self.goal_stay_duration = goal_stay_duration

        # current agent and goal location
        self.agent_location = None
        self.goal_location = self.goal_left_length

        # goal_reached
        self.goal_reached = False
        # number of steps agent stayed at goal
        self.goal_stayed = 0

        self.current_step = 0

    @property
    def left_border(self):
        """
        get leftmost location
        """
        return 0
    
    @property
    def right_border(self):
        """
        get rightmost location
        """
        return self.goal_left_length + self.goal_right_length

    def __get_obs(self):
        shape = (self.goal_left_length + 1 + self.goal_right_length, )
        obs = np.zeros(shape)
        obs[self.agent_location] = 1
        return obs

    def __valid_action(self, action):
        """
        Check if action is valid
        """
        if action < 0 or action > 2:
            return False
        return True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.randomise_start_location:
            # start at random location in hallway
            self.agent_location = np.random.randint(self.left_border, self.right_border + 1)
        else:
            # start at left side
            self.agent_location = 0

        self.goal_reached = False
        self.goal_stayed = 0
        self.current_step = 0
        return self.__get_obs()

    def step(self, action):
        self.current_step += 1

        # check if action is valid
        if not self.__valid_action(action):
            self.logger.warning(f"Invalid action {action} applied. Will be ignored.")
            action = Action.STAY
        else:
            action = Action(action)

        # move agent
        if action == Action.LEFT:
            self.agent_location = max(self.left_border, self.agent_location - 1)
        elif action == Action.RIGHT:
            self.agent_location = min(self.right_border, self.agent_location + 1)
        # else stay action --> location does not change

        obs = self.__get_obs()
        done = self.current_step >= self.episode_length

        # compute reward
        reward = 0.0
        # action reward
        if action == Action.LEFT:
            reward += self.left_reward
        elif action == Action.STAY:
            reward += self.stay_reward
        elif action == Action.RIGHT:
            reward += self.right_reward
        # goal reward for reaching goal location (only once)
        if self.agent_location == self.goal_location and not self.goal_reached:
            reward += self.goal_reward
            self.goal_reached = True
        # reward for staying at goal location for goal_stay_duration steps
        if action == Action.STAY and self.agent_location == self.goal_location:
            self.goal_stayed += 1
            if self.goal_stayed == self.goal_stay_duration:
                self.goal_stayed = 0
                reward += self.goal_reward

        return obs, reward, done, {}

    def render(self):
        arr = self.__get_obs()
        hallway = "#"
        for i, v in enumerate(arr):
            if v == 1:
                hallway += "A"
            elif i == self.goal_location:
                hallway += "G"
            elif v == 0:
                hallway += " "
        hallway += "#"
        print(hallway)

    def close(self):
        pass
