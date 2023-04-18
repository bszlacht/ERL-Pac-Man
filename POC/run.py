import gym
import src.env
class PacmanGame:
    def __init__(self):
        self.env = gym.make("pacman-v0", layout="classic")
        self.action_space = self.env.action_space
    def make_action(self,state):
      return self.action_space.sample()

    def run(self):
      state = self.env.reset()
      while True:
        self.env.render()
        action = self.make_action(state)
        satte, rewards, done, info = self.env.step(action)
        if done:
          break
      self.env.close()

if __name__ == "__main__":
   PacmanGame().run()