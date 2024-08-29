import random
import numpy as np
import time
import os

seed = time.time_ns() + os.getpid() + os.urandom(16).__hash__()
random.seed(seed)

class MonteCarloLearning:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.policy = {}
        self.value_table = {}
        self.returns = {}
        self.initialize_policy()

    def initialize_policy(self):
        x_lim = self.env.x_bounds[1]
        y_lim = self.env.y_bounds[1]
        
        left = 0
        right = 0
        straight = 0
        
        for x in range(-x_lim, x_lim + 1):
            for y in range(-y_lim, y_lim + 1):
                for orientation in self.env.directions:
                    self.policy[(x, y, orientation)] = (random.choice(self.env.actions), random.choice(self.env.velocities))
                    self.value_table[(x, y, orientation)] = 0
                    self.returns[(x, y, orientation)] = []
                    
                    if(self.policy[(x, y, orientation)][0] == 'left'):
                        left += 1
                    elif(self.policy[(x, y, orientation)][0] == 'right'):
                        right += 1
                    else:
                        straight += 1
                        
        print("Left: ", left)
        print("Right: ", right)
        print("Straight: ", straight)
    
    def is_valid_state(self, state):
        x, y, orientation = state
        if x < self.env.x_bounds[0] or x > self.env.x_bounds[1]:
            return False
        if y < self.env.y_bounds[0] or y > self.env.y_bounds[1]:
            return False
        return True

    def generate_episode(self, epsilon=0.5):
        state = (random.randint(-self.env.x_bounds[1], self.env.x_bounds[1]),
                random.randint(-self.env.y_bounds[1], self.env.y_bounds[1]),
                random.choice(self.env.directions))
        
        episode = []
        visited_states = set()
        
        self.env.reset()
        self.env.x, self.env.y, self.env.orientation = state
        visited_states.add(state)
        
        done = False
        while not done:
            state = (self.env.x, self.env.y, self.env.orientation)
            
            if state not in self.policy or self.policy[state] is None:
                break
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = (random.choice(self.env.actions), random.choice(self.env.velocities))
            else:
                action = self.policy[state]
            
            next_state, reward, done = self.env.step(*action)
            
            if not self.is_valid_state(next_state):
                break
            
            if next_state in visited_states:
                break
            
            visited_states.add(next_state)
            episode.append((state, action, reward))
            state = next_state  # Update state to the new state
        
        for sar in episode:
            print("\t", sar)
        
        return episode

    def update_value_function(self, episode):
        G = 0
        visited = {}
        for i in range(len(episode) - 1, -1, -1):
            state, _, _ = episode[i]
            visited[state] = i
            
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            G = self.gamma * G + reward
            if visited[state] == i:
                self.returns[state].append(G)
                self.value_table[state] = np.mean(self.returns[state])

    def improve_policy(self):
        for state in self.value_table.keys():
            best_action = None
            best_value = float('-inf')
            for action in self.env.actions:
                for velocity in self.env.velocities:
                    self.env.reset()
                    self.env.x, self.env.y, self.env.orientation = state
                    next_state, reward, done = self.env.step(action, velocity)
                    
                    if self.is_valid_state(next_state):
                        value = reward + self.gamma * self.value_table.get(next_state, 0)
                        
                        if value > best_value:
                            best_value = value
                            best_action = (action, velocity)
                            
            self.policy[state] = best_action

    def run_monte_carlo(self, episodes=100):
        ct = 0
        
        start_time = time.time()
        # for _ in range(episodes):
        while(ct < episodes):
            if(time.time() - start_time > 60):
                break
            
            itr = 1
            episode = self.generate_episode()
            while(episode is None):
                episode = self.generate_episode()
                itr += 1
                
                if(itr == 100):
                    break
                
            if(episode is None):
                continue
                
            self.update_value_function(episode)
            self.improve_policy()
            ct += 1
            
            print("Episode: ", ct)
            start_time = time.time()
            
        print("Total valid episodes: ", ct)

    def get_policy(self):
        return self.policy
