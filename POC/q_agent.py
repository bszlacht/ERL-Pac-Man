import math
import pickle
import random
from collections import defaultdict
from itertools import count
from typing import List, Tuple
from datetime import datetime

import numpy as np
from pacman_state import PacmanState

from src.controller import Controller
from src.env.agent import Agent
import gym

from wrappers import SkipFrame
import dill
import time

class QAgent(Agent):
    name = "q_agent"

    def __init__(self, **kwargs):
        super().__init__()
        self.layout = kwargs["layout"]
        self.version = kwargs["version"]
        if self.version is None:
            self.filename = "".join([self.name, "_", self.layout, ".pkl"])
        else:
            self.filename = "".join(
                [self.name, "_", self.layout, "_", self.version, ".pkl"]
            )
        self.stats_filename = "stats_" + self.filename + ".pkl"
        self.q_table = None
        self.stats = None

    def act(self, **kwargs):
        if self.q_table is None:
            self.load_q_table()
        # print(str(kwargs["ghost_pixel_positions"]) + " <-> " + str(kwargs["ghost_positions"]))
        state = QAgent.get_state(
            kwargs["player_pos"],
            kwargs["ghost_positions"],
            kwargs["number_of_scared_ghosts"],
        )

        try:
            q_table_state = self.q_table[state]
            argmax = np.argmax(q_table_state)
            return argmax
        except KeyError:
            return random.randint(0, 3)

    def load_q_table(self):
        with open(self.filename, "rb") as handle:
            self.q_table = dill.load(handle)
            handle.close()

    def load_stats(self):
        with open(self.stats_filename, "rb") as handle:
            self.stats = dill.load(handle)
            handle.close()

    def __del__(self):
        del self.q_table

    @staticmethod
    def get_state(
        player_position: Tuple[int, int],
        ghosts_positions: List[Tuple[int, int]],
        number_of_scared_ghosts: int,
    ):
        is_god_mode = None
        if number_of_scared_ghosts < 2:
            is_god_mode = 0
        else:
            is_god_mode = 1

        return PacmanState(player_position, ghosts_positions, is_god_mode).get_state()

    def run(self):
        env = gym.make("pacman-v0", layout=self.layout)
        env = SkipFrame(env, skip=1)
        info = env.reset(mode="info")
        for i in count():
            env.render()
            action = self.act(
                player_pos=info["player position"], ghost_positions=info["ghosts_pos"], 
            )
            action = int(action)
            obs, rewards, done, info = env.step(action)
            if done:
                return

    def train(self, episodes, start_episode=0, **kwargs):
        n_episodes = episodes
        discount = 0.99
        alpha = 0.6  # learning rate
        epsilon = 1.0
        epsilon_min = 0.05
        epsilon_decay_rate = 1000000
        env = gym.make("pacman-v0", layout=self.layout)
        env = SkipFrame(env, skip=1)
        if self.q_table is None:
            q_table = defaultdict(lambda: np.zeros(env.action_space.n))
            q_table.update()
        else:
            q_table = self.q_table
            q_table.setdefault(lambda: np.zeros(env.action_space.n))
        if self.stats is None:
            stats = []
        else:
            stats = self.stats

        def epsilon_by_frame(frame_idx):
            return epsilon_min + (epsilon - epsilon_min) * math.exp(
                -1.0 * frame_idx / epsilon_decay_rate
            )

        for episode in range(start_episode, n_episodes):
            info = env.reset(mode="info")
            state = QAgent.get_state(
                info["player position"],
                info["ghosts_pos"],
                info["number of scared ghosts"],
            ) 
            total_rewards = 0
            print("epsilon = " + str(int(10 * epsilon)))
            epsilon = epsilon_by_frame(episode)

            for i in count():
                #time.sleep(1)
                env.render()
                if random.uniform(0, 1) > epsilon:
                    action = int(np.argmax(q_table[state]))
                else:
                    action = env.action_space.sample()

                obs, rewards, done, info = env.step(action)
                next_state = QAgent.get_state(
                    info["player position"],
                    info["ghosts_pos"],
                    info["number of scared ghosts"],
                )
                if next_state != state:
                    q_table[state][action] += alpha * (
                        rewards
                        + discount * np.max(q_table[next_state])
                        - q_table[state][action]
                    )

                state = next_state
                total_rewards += rewards
                if done:
                    print(f"{episode} episode finished after {i} timesteps")
                    print(f"Total rewards: {total_rewards}")
                    print(f'win: {info["win"]}')
                    break
            stats.append((episode, total_rewards))
            if episode > 1000 and episode % 100 == 99:
                print(f"SAVING AT {episode} EPISODES")
                with open(self.filename, "wb") as handle:
                    pickle.dump(dict(q_table), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.close()
                with open(self.stats_filename, "wb") as handle:
                    pickle.dump(list(stats), handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.close()

        env.close()


def train_agent(layout: str, episodes: int = 5000):
    agent = QAgent(layout=layout, version="5")
    agent.train(episodes=episodes)


def run_agent(layout: str):
    agent = QAgent(layout=layout, version="5")
    agent.load_q_table()
    controller = Controller(
        layout_name=layout, act_sound=True, act_state=False, ai_agent=agent
    )
    controller.load_menu()


if __name__ == "__main__":
    run_agent("medium")
    #train_agent(layout="medium", episodes=50000)
    # boost_agent(layout="classic", episodes = 1000)

'''
1. QAgent -> -1 za krok, -10 za ściane, + 10 za pellet, + 5 za power, + 50 za ducha
2. QAgent -> -1 za krok, -10 za ściane, + 60 za pellet, + 40 za power, + 120 za ducha
3. QAgent -> kroki
'''
