"""
CS 441 Group Final Project

Thomas Pollard
Michael Kay
David Hawbaker
Gregory Hairfield
Andrew Ruskamp-White

"""
import gym
import numpy as np
import random

EPSILON = 0.1

class monteCarloLearningAgent():
    def __init__(self, actionSpaceSize, observationSpaceSize):
        self.actions = list(range(actionSpaceSize))
        self.QMatrix = np.zeros((observationSpaceSize, actionSpaceSize))
##        print(self.QMatrix)

    def chooseAction(self):
        # Epsilon greedy policy
        # Choose the best option available or choose random at probability of epsilon
        if random.random() <= EPSILON:
            actionIndex = random.choice(self.actions)
        else:
            actionIndex = np.argmax(self.QMatrix)
        return actionIndex


def customRender(self, mode='human'):
    out = self.desc.copy().tolist()
    out = [[c.decode('utf-8') for c in line] for line in out]
    taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

    def ul(x): return "_" if x == " " else x
    if pass_idx < 4:
        out[1 + taxi_row][2 * taxi_col + 1] = '○'

        pi, pj = self.locs[pass_idx]
        out[1 + pi][2 * pj + 1] = out[1 + pi][2 * pj + 1].lower()
    else:  # passenger in taxi
        out[1 + taxi_row][2 * taxi_col + 1] = '◙'

    di, dj = self.locs[dest_idx]
    out[1 + di][2 * dj + 1] = out[1 + di][2 * dj + 1].lower()
    print("\n".join(["".join(row) for row in out]) + "\n")
    if self.lastaction is not None:
        print(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction])
    else:
        print("\n")

env = gym.make('Taxi-v3')
env.render = customRender

monteCarlo = monteCarloLearningAgent(env.action_space.n, env.observation_space.n)

for i_episode in range(40):
    observation = env.reset()
    for t in range(100):
        env.render(env)
        print(observation)
##        action = env.action_space.sample()
        action = monteCarlo.chooseAction()
        observation, reward, done, info = env.step(action)
##        done = True
##        env.render(env)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
