import torch
import numpy as np
from ddqn import DDQN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer 
from torchrl.data.replay_buffers.storages import ListStorage

class Agent:
    def __init__(self , input_dims , num_actions):
        self.num_actions = num_actions
        self.learn_step_counter = 0


        #Hyperparameters
        self.lr = 0.00025
        self.gamma = 0.9
        self.epsilon = 1.0
        self.eps_decay = 0.99999975
        self.eps_min = 0.1
        self.batch_size = 32
        self.sync_netwrok_rate = 10_000

        #Networks

        self.online_network = DDQN(input_dims , num_actions)
        self.target_network = DDQN(input_dims,num_actions,freeze=True)

        #Optimizer and Loss

        self.optimizer = torch.optim.Adam(self.online_network.parameters(),lr = self.lr)
        self.loss = torch.nn.MSELoss()

        #Replay -Buffer
        replay_buffer_capacity = 100_000
        storage = ListStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)


    def choose_action(self , observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        observation = torch.tensor(np.array(observation),dtype=torch.float32).unsqueeze(0).to(self.online_network.device)
        return self.online_network(observation).argmax().item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay , self.eps_min)

    def store_in_memory(self , state , action , reward , next_state , done):
        self.replay_buffer.add(TensorDict({
            "state" : torch.tensor(np.array(state),dtype=torch.float32),
            "action" : torch.tensor(action),
            "reward" : torch.tensor(reward),
            "next_state" : torch.tensor(np.array(next_state),dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.float32)
        },
        batch_size=[]
        ))

    def sync_network(self):
        if self.learn_step_counter % self.sync_netwrok_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_network()

        self.optimizer.zero_grad()
        
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        # keys = ("state" ,"action" ,"reward","next_state","done")

        states = samples["state"]
        actions = samples["action"]
        rewards = samples["reward"]
        next_states = samples["next_state"]
        dones = samples["done"]

        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)

        target_q_values = self.target_network(next_states).max(dim=1)[0].detach()
        target_q_values = rewards + self.gamma * target_q_values *(1-dones.float())

        loss = self.loss(predicted_q_values , target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

    def save_model(self, path):
        torch.save({
            "online": self.online_network.state_dict(),
            "target": self.target_network.state_dict()
        }, path)


    def load_model(self, path):
        checkpoint = torch.load(path)
    
        self.online_network.load_state_dict(checkpoint["online"])
        self.target_network.load_state_dict(checkpoint["target"])
    
        self.online_network.eval()
        self.target_network.eval()