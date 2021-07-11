import copy
from typing import Tuple

import numpy as np
import os
import random
import signal

import gym
import torch
from torch import nn

env_name = 'Pong-ramNoFrameskip-v4'
env = gym.make(env_name)
render = False
observation = env.reset()
last_observation = copy.deepcopy(observation)
significant_sectors = [21, 49, 51, 54]
state_size = 2 * len(significant_sectors)
gamma = 0.99

device = "cuda" if torch.cuda.is_available() else "cpu"

n = state_size
m = 2
num_hidden_neurons = 512
num_output_neurons = 1

epsilon = 0.5


class Reinforcement(nn.Module):
    def __init__(self):
        super(Reinforcement, self).__init__()
        self.i = nn.Linear(m*n, num_hidden_neurons)
        torch.nn.init.xavier_uniform_(self.i.weight)
        torch.nn.init.zeros_(self.i.bias)
        self.l1 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)
        self.l2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.zeros_(self.l2.bias)
        self.o = nn.Linear(num_hidden_neurons, num_output_neurons)
        torch.nn.init.xavier_uniform_(self.o.weight)
        torch.nn.init.zeros_(self.o.bias)

    def forward(self, x):
        x = self.i(x)
        x = torch.relu(x)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.o(x)
        return x

    def __call__(self, *input, **kwargs) -> torch.Tensor:
        return super().__call__(*input, **kwargs)


model_a = Reinforcement()
model_b = Reinforcement()
if os.path.exists("state_a.torch"):
    model_a.load_state_dict(torch.load("state_a.torch"))
    model_b.load_state_dict(torch.load("state_b.torch"))
    print("loading model")
model = model_a
counter_model = model_b
model.to(device)
counter_model.to(device)


def get_q_estimates(obs: torch.Tensor) -> torch.Tensor:
    q_estimates = model(obs)
    return q_estimates


def get_counter_estimates(obs: torch.Tensor) -> torch.Tensor:
    counter_estimates = counter_model(obs)
    return counter_estimates


def stack_state(state: torch.Tensor, action: int) -> torch.Tensor:
    one_hot_encoded = torch.zeros(m).to(device)
    one_hot_encoded[action] = 1.0
    orthogonal_representation = torch.kron(one_hot_encoded, state[0])
    return orthogonal_representation


def wrap_state(state: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(state).float().unsqueeze(0).to(device)


def get_random_action(state) -> Tuple[torch.Tensor, int]:
    actions = [x for x in range(m)]
    sampled_action = random.choice(actions)
    stacked = stack_state(state, sampled_action)
    sample_value = get_q_estimates(stacked)
    return sample_value, sampled_action


def get_greedy_action(state) -> Tuple[torch.Tensor, int]:
    actions = [x for x in range(m)]
    best_q_value = -1e25
    best_action = 0
    first = True
    for action in actions:
        stacked = stack_state(state, action)
        q_value = get_q_estimates(stacked)
        if q_value > best_q_value or first:
            best_q_value = q_value
            best_action = action
            first = False
    return best_q_value, best_action


def get_action(obs) -> Tuple[torch.Tensor, int]:
    state = wrap_state(obs)
    action_selection = get_greedy_action
    if random.random() < epsilon:
        action_selection = get_random_action
    q_value, action = action_selection(state)
    return q_value, action


def get_greedy_action_double(state) -> Tuple[torch.Tensor, int]:
    actions = [x for x in range(m)]
    best_q_value = -1e25
    best_q_return = 0
    best_action = 0
    first = True
    for action in actions:
        stacked = stack_state(state, action)
        q_value = get_q_estimates(stacked)
        counter_value = get_counter_estimates(stacked)
        if (q_value + counter_value) / 2.0 > best_q_value or first:
            best_q_value = (q_value + counter_value) / 2.0
            best_q_return = counter_value
            best_action = action
            first = False
    return best_q_return, best_action


def get_random_action_double(state) -> Tuple[torch.Tensor, int]:
    actions = [x for x in range(m)]
    sampled_action = random.choice(actions)
    stacked = stack_state(state, sampled_action)
    counter_value = get_counter_estimates(stacked)
    return counter_value, sampled_action


def get_action_double(obs) -> Tuple[torch.Tensor, int]:
    state = wrap_state(obs)
    action_selection = get_greedy_action_double
    if random.random() < epsilon:
        action_selection = get_random_action_double
    q_value, action = action_selection(state)
    return q_value, action


def get_loss(reward: float, next_state_action_value: torch.Tensor, state_action_value: torch.Tensor) -> torch.Tensor:
    return torch.square(reward + gamma * next_state_action_value - state_action_value)


def get_loss_finishing(reward: float, state_action_value: torch.Tensor) -> torch.Tensor:
    return torch.square(reward - state_action_value)


def ball_player_penalty(state, action):
    ball_x = state[1]
    player_y = state[2]
    ball_y = state[3]
    player_y_diff = ball_y - player_y - 5.0
    rescaled_action = 2 * action - 1
    if ball_x < 180.0:
        return 0.0
    if ball_x >= 190.0:
        return 0.0
    if 180.0 <= ball_x <= 190.0:
        return 0.05 if rescaled_action * player_y_diff > 0 else -0.05
    return 0.0


def ball_computer_penalty(state, _action):
    cpu_y = state[0]
    ball_x = state[1]
    ball_y = state[3]
    player_y_diff = ball_y - cpu_y - 5.0
    if ball_x < 69.0:
        return 0.0
    if ball_x >= 75.0:
        return 0.0
    if 69.0 <= ball_x <= 75.0:
        return 0.05 if player_y_diff > 7 or player_y_diff < -7 else -0.05
    return 0.0


def compute_returns(rewards):
    returns = []
    acc = 0
    for value in rewards[::-1]:
        acc = value + gamma * acc
        returns.insert(0, acc)
    return returns


def signal_handler(_sig, _frame):
    print('saving state')
    torch.save(model.state_dict(), 'models/state_a-{}-{}-{}-{}-sarsa.torch'.format(n, m, num_hidden_neurons, env_name))
    torch.save(counter_model.state_dict(), 'models/state_b-{}-{}-{}-{}-sarsa.torch'.format(n, m, num_hidden_neurons, env_name))
    print('saved')


def encode_action(action: int) -> int:
    if action == 2:
        return 1
    return 3 if action >= 0.5 else 2


def encode_state(state, last_state):
    significant_state = [float(state[x]) for x in significant_sectors]
    significant_state.extend([float(state[x]) - float(last_state[x]) for x in significant_sectors])
    normalized_state = np.asarray(significant_state)
    return normalized_state


signal.signal(signal.SIGINT, signal_handler)

model_optim = torch.optim.SGD(model_a.parameters(), lr=1e-4, weight_decay=1e-6)
counter_optim = torch.optim.SGD(model_b.parameters(), lr=1e-4, weight_decay=1e-6)
torch.autograd.set_detect_anomaly(True)
continued_score = -1


def swap_models():
    delta = random.random()
    if delta < 0.5:
        global model
        global counter_model
        global model_optim
        global counter_optim
        temp = counter_model
        counter_model = model
        model = temp
        t = model_optim
        model_optim = counter_optim
        counter_optim = t


if __name__ == "__main__":
    for epoch in range(200000):
        batch_returns = []
        batch_rewards = []
        episode_count = 30
        observation = env.reset()
        rewards = []
        actions = []
        observations = []
        finished_rendering = False
        significant_state = encode_state(observation, last_observation)
        state_action_value, action = get_action(significant_state)
        for i in range(1, 1000000):
            if render or (epoch % 20 == 0 and not finished_rendering):
                env.render()
            last_observation = observation
            observations.append(significant_state)
            encoded_action = encode_action(action)
            observation, reward, done, info = env.step(encoded_action)
            model_optim.zero_grad()
            if reward > 0 or reward < 0 or done:
                batch_rewards.append(reward)
                future_state = encode_state(observation, last_observation)
                rewards.append(reward)
                stacked_state = stack_state(wrap_state(significant_state), action)
                state_action_value = get_q_estimates(stacked_state)
                loss = get_loss_finishing(reward, state_action_value)
                if isinstance(loss, torch.Tensor) is False:
                    print('finishing oops')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 500.0)
                model_optim.step()
                actions.append(action)
                episode_reward, episode_len = compute_returns(rewards), len(rewards)
                batch_returns.extend(episode_reward)
                finished_rendering = True
                done = False
                break
            else:
                future_state = encode_state(observation, last_observation)
                reward += ball_player_penalty(significant_state, action)
                reward += ball_computer_penalty(significant_state, action)
                with torch.no_grad():
                    next_state_action_value, next_action = get_action_double(future_state)
                stacked_state = stack_state(wrap_state(significant_state), action)
                state_action_value = get_q_estimates(stacked_state)
                loss = get_loss(reward, next_state_action_value, state_action_value)
                if isinstance(loss, torch.Tensor) is False:
                    print('continuing oops')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 500.0)
                model_optim.step()
                actions.append(action)
                rewards.append(reward)
                action = next_action
                significant_state = future_state
                swap_models()
        if epoch > 100:
            epsilon = 0.1
        if epoch > 5000:
            epsilon = 0.01
        if epoch > 30000:
            epsilon = 0.001
        continued_score = continued_score + 0.01 * (sum(batch_rewards) - continued_score)
        if epoch % 10 == 0:
            print('updating epoch {}'.format(epoch))
            print('episodes in batch: {}'.format(len(batch_rewards)))
            print('batch weights: {}'.format(len(actions)))
            print('reward: {}'.format(float(sum(batch_rewards))/len(batch_rewards)))
            print('returns: {}'.format(float(sum(batch_returns))/len(batch_returns)))
            print('continued score: {}'.format(continued_score))
            print('last loss: {}'.format(loss))
        if epoch % 200 == 0:
            signal_handler(None, None)
