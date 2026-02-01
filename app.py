import random
import torch
import torch.nn as nn
import torch.optim as optim

GRID = 8
TARGET = (7,7)
THREATS = [(3,3),(4,4),(5,2)]
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]

class Field:
    def reset(self, n):
        self.pos = [[0,0] for _ in range(n)]
        return self.pos

    def step(self, actions):
        reward = 0
        for i,a in enumerate(actions):
            x,y = self.pos[i]
            dx,dy = ACTIONS[a]
            x = max(0, min(GRID-1, x+dx))
            y = max(0, min(GRID-1, y+dy))
            self.pos[i] = [x,y]

            if (x,y) in THREATS:
                reward -= 20
            if (x,y) == TARGET:
                reward += 80

        reward -= 1
        done = any(tuple(p)==TARGET for p in self.pos)
        return self.pos, reward, done


class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self,x):
        return self.net(x)


class Drone:
    def __init__(self, brain, optimizer):
        self.brain = brain
        self.optimizer = optimizer
        self.gamma = 0.9

    def act(self, state):
        with torch.no_grad():
            q = self.brain(torch.tensor([state],dtype=torch.float))
        return torch.argmax(q).item()

    def learn(self, s,a,r,ns):
        q = self.brain(torch.tensor([s],dtype=torch.float))
        nq = self.brain(torch.tensor([ns],dtype=torch.float))

        target = q.clone().detach()
        target[0][a] = r + self.gamma * torch.max(nq)

        loss = nn.MSELoss()(q,target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Swarm:
    def __init__(self, drones=4):
        self.env = Field()
        self.brain = Brain()
        self.opt = optim.Adam(self.brain.parameters(), lr=0.005)
        self.team = [Drone(self.brain,self.opt) for _ in range(drones)]
        self.n = drones

    def run_mission(self):
        states = self.env.reset(self.n)
        done=False
        total_reward=0

        while not done:
            actions = [self.team[i].act(states[i]) for i in range(self.n)]
            next_states, reward, done = self.env.step(actions)

            for i in range(self.n):
                self.team[i].learn(states[i], actions[i], reward, next_states[i])

            states = next_states
            total_reward += reward

        return self.env.pos, total_reward
