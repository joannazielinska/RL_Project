from pong import Pong
import matplotlib.pyplot as plt
from random import randint
import pickle
import numpy as np
from simple_ai import PongAi


def plot(observation):
    plt.imshow(observation/255)
    plt.show()


env = Pong()
episodes = 1

player_id = 1
opponent_id = 3 - player_id
opponent = PongAi(env, opponent_id)
player = PongAi(env, player_id)

env.set_names(player.get_name(), opponent.get_name())

for i in range(0,episodes):
    done = False
    while not done:
        action1 = player.get_action()
        action2 = opponent.get_action()
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        #plot(ob1) # plot the observation of this frame
        env.render()
        if done:
            observation= env.reset()
            #plot(ob1) # plot the reset observation
            print("episode {} over".format(i))

# Needs to be called in the end to shut down pygame
env.end()



