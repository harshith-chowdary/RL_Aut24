import numpy as np
import matplotlib.pyplot as plt
import random
import time
import json
import os

from CarEnv import CarEnv
from PI import PolicyIteration
from MC import MonteCarloLearning
from MCC import MonteCarloControl
import plotting

def serialize_policy(policy):
    # Convert dictionary keys to strings
    return {str(k): v for k, v in policy.items()}

def save_policy(policy, filename):
    with open(filename, 'w') as f:
        json.dump(serialize_policy(policy), f, indent=4)
        
def deserialize_policy(policy):
    # Convert dictionary keys back to original type if needed
    # For now, assuming keys are strings that need to be tuples
    return {eval(k): v for k, v in policy.items()}

print("Enter target position (x, y): ", end= "")
x, y = list(map(int, input().split()))

target_position = (x, y)

print("Enter target orientation (N, NE, E, SE, S, SW, W, NW): ", end= "")
target_orientation = input()

# Test the policy
number_of_tests = int(input("Enter number of tests: "))
tests = []

for i in range(number_of_tests):
    x, y = list(map(int, input("Enter start position (x, y): ").split()))
    start_orientation = input("Enter start orientation (N, NE, E, SE, S, SW, W, NW): ")
    
    tests.append((x, y, start_orientation))

start_position = (0, 0)
start_orientation = 'N'

grid = 50

env = CarEnv(grid, grid, start_position, start_orientation, target_position, target_orientation)

target = (target_position[0], target_position[1], target_orientation)

policy_target_name = 'PI_policy_(' + str(target_position[0]) + ', ' + str(target_position[1]) + ', ' + str(target_orientation) + ').json'
values_target_name = 'PI_values_(' + str(target_position[0]) + ', ' + str(target_position[1]) + ', ' + str(target_orientation) + ').json'

# Policy Iteration
pi = PolicyIteration(env)

try:
    with open(policy_target_name, 'r') as f:
        policy = json.load(f)
        pi_policy = deserialize_policy(policy)
        
    print("\nPolicy loaded from file: ", policy_target_name)
    
    with open(values_target_name, 'r') as f:
        values = json.load(f)
        pi_values = deserialize_policy(values)
        
    print("\nValues loaded from file: ", values_target_name)
except:
    start_time = time.time()
    pi.run_policy_iteration(100)
    end_time = time.time()
    
    print("\nTime taken for Policy Iteration: ", end_time - start_time)
    
    pi_policy = pi.get_policy()

    save_policy(pi_policy, policy_target_name)
    
    print("\nPolicy saved to file: ", policy_target_name)
    
    pi_values = pi.get_value_table()
    
    save_policy(pi_values, values_target_name)
    
    print("\nValues saved to file: ", values_target_name)

# print("Policy Iteration Policy: ", pi_policy)

print()

results = []

for i in range(number_of_tests):
    env.reset()
    
    state = tests[i]
    
    if(state == target):
        print("\nTarget already reached!\n")
        print()
        continue
    
    env.orientation = state[2]
    env.x = state[0]
    env.y = state[1]
    
    print("Start position: ", state)
    
    vis = {state}
    
    done = False
    ct = 0
    while not done:
        action = pi_policy[state]
        
        if(action is None):
            print("No valid action found!")
            ct = 'INF'
            break
        
        _, _, done = env.step(*action, True)
        state = (env.x, env.y, env.orientation)
        
        if(state in vis):
            print("Stuck in loop!")
            ct = 'INF'
            break
        vis.add(state)
        ct += 1
        
    env.render()
    
    if(ct != 'INF'):
        print("\nTarget Reached in ", ct, " steps!\n")
    else:
        print("\nTarget not reached!\n")
        
    results.append([ct, -1])


env.off_interactive()

print('\nPlotting Values of PI:')
plotting.plot_value_function(pi_values, 'Final Value Tablue of PI', save = 0)

env.on_interactive()

#######################################################################

# Monte Carlo
# mc = MonteCarloLearning(env)
# policy_target_name = 'MC_policy_(' + str(target_position[0]) + ', ' + str(target_position[1]) + ', ' + str(target_orientation) + ').json'

mcc = MonteCarloControl(env)
policy_target_name = 'MCC_policy_(' + str(target_position[0]) + ', ' + str(target_position[1]) + ', ' + str(target_orientation) + ').json'
q_values_target_name = 'MCC_q_values_(' + str(target_position[0]) + ', ' + str(target_position[1]) + ', ' + str(target_orientation) + ').json'

try:
    with open(policy_target_name, 'r') as f:
        policy = json.load(f)
        # mc_policy = deserialize_policy(policy)
        mcc_policy = deserialize_policy(policy)
        
    print("\nPolicy loaded from file: ", policy_target_name)
    
    with open(q_values_target_name, 'r') as f:
        q_values = json.load(f)
        mcc_q_values = deserialize_policy(q_values)
        
    print("\nQ-Values loaded from file: ", q_values_target_name)
except:
    start_time = time.time()
    # mc.run_monte_carlo(200)
    mcc.run_monte_carlo(100000)
    end_time = time.time()
    
    print("\nTime taken for Monte Carlo Learning: ", end_time - start_time)
    
    # mc_policy = mc.get_policy()
    mcc_policy = mcc.get_policy()
    mcc_q_values = mcc.get_q_values()
    
    all_states = [(x, y, orientation) for x in range(-5, 6) for y in range(-5, 6) for orientation in env.directions]
    
    # mcc.print_q_values(all_states) # To print Q-values for all states
    
    # save_policy(mc_policy, policy_target_name)
    save_policy(mcc_policy, policy_target_name)
    
    print("\nPolicy saved to file: ", policy_target_name)
    
    save_policy(mcc_q_values, q_values_target_name)
    
    print("\nQ-Values saved to file: ", q_values_target_name)

# print("Policy Iteration Policy: ", mc_policy)

print()

for i in range(number_of_tests):
    env.reset()
    
    state = tests[i]
    
    if(state == target):
        print("\nTarget already reached!\n")
        print()
        continue
    
    env.orientation = state[2]
    env.x = state[0]
    env.y = state[1]
    
    print("Start position: ", (x, y, start_orientation))
    
    vis = {state}
    
    done = False
    ct = 0
    while not done:
        # action = mc_policy[state]
        action = mcc_policy[state]
        
        if(action is None):
            print("No valid action found!")
            ct = 'INF'
            break
        
        _, _, done = env.step(*action, True)
        state = (env.x, env.y, env.orientation)
        
        if state[0] < -grid or state[0] > grid or state[1] < -grid or state[1] > grid:
            ct = 'INF'
            break
        
        if(state in vis):
            print("Stuck in loop!")
            ct = 'INF'
            break
        vis.add(state)
        ct += 1
        
    env.render()
    
    if(ct != 'INF'):
        print("\nTarget Reached in ", ct, " steps!\n")
    else:
        print("\nTarget not reached!\n")
        
    results[i][1] = ct
    
print("\nResults - Target:", target)
print(f"{'Start Position':<25}{'Policy Iteration':<25}{'Monte Carlo'}")

for i in range(number_of_tests):
    print(f"{str(tests[i]):<25}{str(results[i][0]):<25}{str(results[i][1])}")

env.off_interactive()

print('\nPlotting Q-Values of MCC:')
plotting.plot_value_function(mcc_q_values, 'Final Q-Values of MCC', save = 0)

env.on_interactive()

env.reset()