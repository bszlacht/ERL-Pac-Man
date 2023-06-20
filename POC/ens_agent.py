import math
import pickle
import random
from collections import defaultdict
from itertools import count
from typing import List, Tuple
from datetime import datetime
import os

import numpy as np
from pacman_state import PacmanState

from src.controller import Controller
from src.env.agent import Agent
import gym

from wrappers import SkipFrame
import dill


class EnsAgent(Agent):
    name = "trained"

    def __init__(self, **kwargs):
        super().__init__()
        self.layout = kwargs["layout"]
        self.filename_base = "trained_medium"

        # self.stats_filename = "stats_" + self.filename + ".pkl"
        self.q_tables = None
        self.stats = None

    def train(self, **kwargs):
        return

    def act(self, **kwargs):
        if self.q_tables is None:
            self.load_q_tables()

        state = EnsAgent.get_state(kwargs["player_pos"], kwargs["ghost_positions"])

        try:
            action_votes = dict()
            for q_table in self.q_tables:
                q_table_state = q_table[state]
                argmax = np.argmax(q_table_state)
                if argmax in action_votes:
                    action_votes[argmax] += 1
                else:
                    action_votes[argmax] = 1
            return max(action_votes, key=lambda k: action_votes[k])
        except KeyError:
            return random.randint(0, 3)

    def load_q_tables(self):
        directory = os.getcwd() + "/trained_agents"

        file_list = os.listdir(directory)
        dict_list = []

        for filename in file_list:
            if self.filename_base in filename:
                file_path = os.path.join(directory, filename)
                with open(file_path, "rb") as file:
                    dictionary = dill.load(file)
                    dict_list.append(dictionary)
        self.q_tables = dict_list
        # with open(self.filename, "rb") as handle:
        #     self.q_table = dill.load(handle)
        #     handle.close()

    # def load_stats(self):
    #     with open(self.stats_filename, "rb") as handle:
    #         self.stats = dill.load(handle)
    #         handle.close()

    def __del__(self):
        del self.q_tables

    @staticmethod
    def get_state(
        player_position: Tuple[int, int], ghosts_positions: List[Tuple[int, int]]
    ):
        return PacmanState(player_position, ghosts_positions).get_state()

    def run(self):
        env = gym.make("pacman-v0", layout=self.layout)
        env = SkipFrame(env, skip=1)
        info = env.reset(mode="info")
        for i in count():
            env.render()
            action = self.act(
                player_pos=info["player position"], ghost_positions=info["ghosts_pos"]
            )
            action = int(action)
            obs, rewards, done, info = env.step(action)
            if done:
                return


def run_agent(layout: str):
    agent = EnsAgent(layout=layout)
    agent.load_q_tables()
    # agent.run()
    controller = Controller(
        layout_name=layout, act_sound=True, act_state=False, ai_agent=agent
    )
    controller.load_menu()


if __name__ == "__main__":
    run_agent("medium")
