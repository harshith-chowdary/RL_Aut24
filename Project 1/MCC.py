import random
import numpy as np
import time
import os
import plotting

seed = time.time_ns() + os.getpid() + os.urandom(16).__hash__()
random.seed(seed)

class MonteCarloControl:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}  # Action-value function
        self.returns = {}  # Store returns for each state-action pair
        self.initialize_Q()
        self.start_states = set()

    def initialize_Q(self):
        x_lim = self.env.x_bounds[1]
        y_lim = self.env.y_bounds[1]
        
        for x in range(-x_lim, x_lim + 1):
            for y in range(-y_lim, y_lim + 1):
                for orientation in self.env.directions:
                    for action in self.env.actions:
                        for velocity in self.env.velocities:
                            state_action = ((x, y, orientation), (action, velocity))
                            self.Q[state_action] = 0
                            self.returns[state_action] = []

    def is_valid_state(self, state):
        x, y, orientation = state
        if x < self.env.x_bounds[0] or x > self.env.x_bounds[1]:
            return False
        if y < self.env.y_bounds[0] or y > self.env.y_bounds[1]:
            return False
        return True
    
    def print_q_values(self, states):
        for state in states:
            print(f"State: {state}")
            for action in self.env.actions:
                for velocity in self.env.velocities:
                    q_value = self.Q.get((state, (action, velocity)), 0)
                    print(f"  Action: {action}, Velocity: {velocity}, Q-Value: {q_value}")

    def generate_episode(self):
        if len(self.start_states) > 0:
            state = random.choice(list(self.start_states))
            self.start_states.remove(state)
        else:
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
            
            if random.random() < self.epsilon:
                action = (random.choice(self.env.actions), random.choice(self.env.velocities))
            else:
                action = self.get_best_action(state)
            
            next_state, reward, done = self.env.step(*action)
            
            if not self.is_valid_state(next_state):
                break
            
            if next_state in visited_states:
                break
            
            visited_states.add(next_state)
            episode.append((state, action, reward))
            state = next_state
        
        return episode

    def get_best_action(self, state):
        best_action = None
        best_value = float('-inf')
        for action in self.env.actions:
            for velocity in self.env.velocities:
                state_action = (state, (action, velocity))
                if self.Q[state_action] > best_value:
                    best_value = self.Q[state_action]
                    best_action = (action, velocity)
        return best_action

    def update_Q(self, episode):
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            state_action = (state, action)
            if state_action not in visited:
                self.returns[state_action].append(G)
                self.Q[state_action] = np.mean(self.returns[state_action])
                visited.add(state_action)

    def run_monte_carlo(self, episodes=1000):
        self.states = {(x, y, orientation) for x in range(-self.env.x_bounds[1], self.env.x_bounds[1] + 1)
                       for y in range(-self.env.y_bounds[1], self.env.y_bounds[1] + 1)
                       for orientation in self.env.directions}
        for i in range(episodes):
            episode = self.generate_episode()
            self.update_Q(episode)
            # print("Generated episode with length:", len(episode))
            
            if(i == 1):
                self.env.off_interactive()
                plotting.plot_value_function(self.get_q_values(), "Initial Q-Values of MCC", 1)
                self.env.on_interactive()
                
            if(i == episodes//2):
                self.env.off_interactive()
                plotting.plot_value_function(self.get_q_values(), "Half-way Q-Values of MCC", 1)
                self.env.on_interactive()

    def get_policy(self):
        policy = {}
        for state in set(sa[0] for sa in self.Q.keys()):
            policy[state] = self.get_best_action(state)
        return policy
    
    def get_q_values(self):
        q_values = {}
        for state in set(sa[0] for sa in self.Q.keys()):
            best_action = self.get_best_action(state)
            q_values[state] = self.Q[(state, best_action)]
        return q_values