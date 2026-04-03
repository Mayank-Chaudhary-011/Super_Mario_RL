import torch
import argparse
import os

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from wrappers import apply_wrappers
from agent import Agent
from utils import *


def make_env(env_name, display):
    if display:
        env = gym_super_mario_bros.make(
            env_name,
            render_mode="human",
            apply_api_compatibility=True
        )
    else:
        env = gym_super_mario_bros.make(
            env_name,
            apply_api_compatibility=True
        )

    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env)
    return env


def train(agent, env, model_path):
    CKPT_SAVE_INTERVAL = 500
    NUM_OF_EPISODES = 50_000

    for i in range(NUM_OF_EPISODES):
        print("Episode:", i)
        done = False
        state, _ = env.reset()
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)

            done = done or truncated
            total_reward += reward

            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()

            state = new_state

        print(f"Total_Reward --> {total_reward} | Epsilon --> {agent.epsilon} | Replay Buffer --> {len(agent.replay_buffer)} | Learn Step --> {agent.learn_step_counter}")

        if (i + 1) % CKPT_SAVE_INTERVAL == 0:
            agent.save_model(os.path.join(model_path, f"model_{i+1}.pt"))



def test(agent, env):
    agent.epsilon = 0.0  # no randomness

    for i in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            state, reward, done, truncated, _ = env.step(action)

            done = done or truncated
            total_reward += reward

        print("Episode reward:", total_reward)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario RL")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model', type=str, default=None)

    args = parser.parse_args()

    ENV_NAME = 'SuperMarioBros-1-1-v0'


    DISPLAY = not args.train   # train = no render, test = render

    env = make_env(ENV_NAME, DISPLAY)

    agent = Agent(
        input_dims=env.observation_space.shape,
        num_actions=env.action_space.n
    )

    if args.train:
        model_path = os.path.join("models", get_current_date_time_string())
        os.makedirs(model_path, exist_ok=True)

        train(agent, env, model_path)

    else:
        if args.model is None:
            raise ValueError("Provide --model path")

        agent.load_model(args.model)
        test(agent, env)