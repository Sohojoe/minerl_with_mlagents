import minerl
import gym
import logging


def main():
    # do your main minerl code
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make('MineRLTreechop-v0')

    env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)    

if __name__ == '__main__':
    main()