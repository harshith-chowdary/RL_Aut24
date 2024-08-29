import random
import plotting

class PolicyIteration:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.policy = {}
        self.value_table = {}
        self.initialize_policy()

    def initialize_policy(self):
        x_lim = self.env.x_bounds[1]
        y_lim = self.env.y_bounds[1]
        
        for x in range(-x_lim, x_lim + 1):
            for y in range(-y_lim, y_lim + 1):
                for orientation in self.env.directions:
                    self.policy[(x, y, orientation)] = (random.choice(self.env.actions), random.choice(self.env.velocities))
                    self.value_table[(x, y, orientation)] = 0
            
    def is_valid_state(self, state):
        x, y, orientation = state
        if x < self.env.x_bounds[0] or x > self.env.x_bounds[1]:
            return False
        if y < self.env.y_bounds[0] or y > self.env.y_bounds[1]:
            return False
        return True

    def policy_evaluation(self):
        new_value_table = {}
        for state in self.value_table.keys():
            if(state == (self.env.target_position[0], self.env.target_position[1], self.env.target_orientation)):
                new_value_table[state] = 0
                continue
            
            self.env.x, self.env.y, self.env.orientation = state
            
            if(self.policy[state] == None):
                new_value_table[state] = self.value_table[state]
                continue
        
            action, velocity = self.policy[state]
            
            next_state, reward, done = self.env.step(action, velocity, False)
            
            if(self.is_valid_state(next_state) == False):
                new_value_table[state] = self.value_table[state]
                continue
                    
            new_value_table[state] = reward + self.gamma * self.value_table[next_state]
            
        self.value_table = new_value_table
        
    def policy_improvement(self):
        for state in self.value_table.keys():
            x, y, orientation = state
            best_action = None
            best_value = float('-inf')
                
            for action in self.env.actions:
                for velocity in self.env.velocities:
                    self.env.x, self.env.y, self.env.orientation = x, y, orientation
                    
                    next_state, reward, done = self.env.step(action, velocity, False)
                    
                    if(self.is_valid_state(next_state) == True):
                        value = reward + self.gamma * self.value_table[next_state]
                            
                        if value > best_value:
                            best_value = value
                            best_action = (action, velocity)
                            
            self.policy[state] = best_action

    def run_policy_iteration(self, iterations=100):
        for i in range(iterations):
            print("Policy Iteration: ", i)
            self.policy_evaluation()
            self.policy_improvement()
            
            if(i == 1):
                self.env.off_interactive()
                plotting.plot_value_function(self.value_table, "Initial Value Tablue of PI", 1)
                self.env.on_interactive()
                
            if(i == iterations//2):
                self.env.off_interactive()
                plotting.plot_value_function(self.value_table, "Half-way Value Tablue of PI", 1)
                self.env.on_interactive()

    def get_policy(self):
        return self.policy
    
    def get_value_table(self):
        return self.value_table
