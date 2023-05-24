import numpy as np
import random
import math
import time
import csv

def advance(action1, action2):
  newstate = 2*action1 + action2
  return newstate

rewards = [[3, 3],[0, 5],[5, 0],[1, 1],[0,0]]

#Learning rate
alpha1 = 0.01
alpha2 = 0.01

#Discount factor. 0 means future rewards not considered
gamma1 = 0.95
gamma2 = 0.95

#Exploration rate. Higher means more random exploration
epsilon1 = 0.01
epsilon2 = 0.01

def epsilongreedy(Q, state, e):
  randval = random.random()
  Qvals_for_state = Q[state,:]

  if randval > e and not Qvals_for_state[0] == Qvals_for_state[1]:
    
    action = np.argmax(Qvals_for_state)

  else:
    action = random.randrange(Qvals_for_state.size)

  return action

def stochastic_choose(Q, state, t):
  Qvals_for_state = Q[state,:].copy()
  randval = random.random()

  boundary = np.exp(Qvals_for_state[0]/t)/np.exp((Qvals_for_state[0]/t)+np.exp(Qvals_for_state[1]/t))

  if randval < boundary:
    return 0

  else:
    return 1

def transformstate(state):
  if state == 1:
    transformedstate = 2
  elif state == 2:
    transformedstate = 1
  else: transformedstate = state
  return transformedstate

def transformQ(Q):
  transformedQ = Q.copy()
  transformedQ[2] = Q[1].copy()
  transformedQ[1] = Q[2].copy()
  return transformedQ

def TFT_check(Q):
  if Q[0][0] > Q[0][1] and Q[1][0] < Q[1][1] and Q[2][0] > Q[2][1] and Q[3][0] < Q[3][1] and Q[4][0] > Q[4][1]:
    return 1
  else:
    return 0

def Co_op_check(Q):
  if Q[0][0] > Q[0][1] and Q[2][0] > Q[2][1] and Q[4][0] > Q[4][1]:
    return 1
  else:
    return 0

def Grim_check(Q):
  if Q[0][0] > Q[0][1] and Q[1][0] < Q[1][1] and Q[2][0] < Q[2][1] and Q[3][0] < Q[3][1] and Q[4][0] > Q[4][1]:
    return 1
  else:
    return 0

def strategy_classify(Q):
    out = ""
    for n in range(5):
        if Q[n][0] > Q[n][1] :
            out += "C"
        else:
            out += "D"
    return out

def two_strat_classify(Q2,Q1):
    out = 0

    if np.array_equal(Q2, np.zeros([5,2])):
        return 1024
    
    else:
        for i in range(5):
            if Q1[i][0] < Q1[i][1] :
                out += 2**i

        for i in range(5):
            if Q2[i][0] < Q2[i][1] :
                out += 2**(i+5)

        return out

#Grim Trigger:
grim_trigger = np.array([[29.99999, 0], [0, 29.9999], [0, 29.9999], [0, 29.9999], [29.9999, 0]])

#TFT
TFT1 = np.array([[100.0, 60.0], [60.0, 100.0], [100.0, 60.0], [60.0, 100.0], [100.0, 60.0]])
TFT2 = np.array([[100.09, 60.0], [100.0, 60.0], [60.0, 100.0], [60.0, 100.0], [100.0, 60.0]])

#Always Coop
always_CoOp = np.array([[100.0, 60.0], [100.0, 60.0], [100.0, 60.0], [100.0, 60.0], [100.0, 60.0]])

#Always Defect
always_defect = np.array([[19.4, 20.4], [19.4, 20.4], [19.4, 20.4], [19.4, 20.4], [19.4, 20.4]])

#Zero
Q_zero = np.zeros([5,2])

#Pavlov
Pavlov = np.array([[100.0, 60.0], [60.0, 100.0], [60.0, 100.0], [100.0, 60.0], [100.0, 60.0]])

def strategy_classify(Q):
    out = ""
    for n in range(5):
        if Q[n][0] > Q[n][1] :
            out += "C"
        else:
            out += "D"
    return out

def two_strat_classify(Q2,Q1):
    out = 0

    if np.array_equal(Q2, np.zeros([5,2])):
        return 1024
    
    else:
        for i in range(5):
            if Q1[i][0] < Q1[i][1] :
                out += 2**i

        for i in range(5):
            if Q2[i][0] < Q2[i][1] :
                out += 2**(i+5)

        return out

def Qlearning(rewards, num_episodes, num_steps, alpha1, alpha2, gamma1, gamma2, epsilon1, epsilon2, debug = False, Q_Player_A_preset = TFT1, Q_Player_B_preset = TFT2):
  Q_Player_A = Q_Player_A_preset.copy()
  Q_Player_B = Q_Player_B_preset.copy()
  episode_rewards_A = []
  episode_rewards_B = []


  for episode in range(num_episodes):
    episode_reward_A = 0
    episode_reward_B = 0
    state = 4 #set default initial state
    
    for step in range(num_steps):
      strat_transition_row = two_strat_classify(Q_Player_A,Q_Player_B)
      actionA = epsilongreedy(Q_Player_A, state, epsilon1)
      actionB = epsilongreedy(Q_Player_B, state, epsilon2)
      next_state = advance(actionA, actionB)

      reward_A = rewards[next_state][0]
      episode_reward_A += reward_A

      reward_B = rewards[next_state][1]
      episode_reward_B += reward_B
      
      #update Q for A
      best_next_actionA = np.argmax(Q_Player_A[next_state,:])

      td_targetA = reward_A + gamma1 * Q_Player_A[next_state, best_next_actionA]

      td_deltaA = td_targetA - Q_Player_A[state][actionA]

    
      # Update Q for B

      best_next_actionB = np.argmax(Q_Player_B[next_state,:])

      td_targetB = reward_B + gamma2 * Q_Player_B[next_state, best_next_actionB]


      td_deltaB = td_targetB - Q_Player_B[state][actionB]


      
      

        
      Q_Player_A[state][actionA] = Q_Player_A[state][actionA] + alpha1 * td_deltaA
      Q_Player_B[state][actionB] += alpha2 * td_deltaB
      
      
    episode_rewards_A.append(episode_reward_A)
    episode_rewards_B.append(episode_reward_B)
    
    stratA = strategy_classify(Q_Player_A)
    stratB = strategy_classify(Q_Player_B)
    
  return episode_rewards_A, stratA, stratB

ep_rewards_A, stratA, stratB = Qlearning(rewards, 2000000, 50, alpha1, alpha2, gamma1, gamma2, epsilon1,  epsilon2, False, Q_zero,Q_zero)
print(ep_rewards_A)
print(stratA)
print(stratB)
