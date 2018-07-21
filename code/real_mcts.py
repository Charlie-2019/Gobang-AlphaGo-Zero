#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:26:34 2018

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:32:01 2018

@author: root
"""
from chessboard import chessboard
import math
import copy
import time
import numpy as np

base = [1]
for i in range(1, 300):
    base.append(base[i - 1] * 3)
    
class EDGE(object):
    def __init__(self, prob):
        self.prob = prob
        self.visit_count = 0
        self.Q = 0
        self.W = 0
        
class NODE(object): #state of chessboard
    def __init__(self, cpuct = 5):
        self.child = {} #mapping position(action) to node
        self.edge = {} #mapping position(action) to edge(state, action)
        self.visit_count = 0
        self.cpuct = cpuct
    
    def select(self):
        tmp = None
        selected = None
        sqrt_visit_count = math.sqrt(self.visit_count)
        for position in self.edge:
            edge = self.edge[position]
            val = edge.Q + self.cpuct * edge.prob * sqrt_visit_count / (1 + edge.visit_count)
            if selected == None or val > tmp:
                tmp = val
                selected = position
        return selected
    
class real_mcts(object):
    def __init__(self, chess, policy, cpuct, simulation_times, temperature, num_history, is_selfplay):
        self.chess = chess
        self.policy = policy
        self.cpuct = cpuct
        self.simulation_times = simulation_times
        self.temperature = temperature
        self.num_history = num_history
        self.is_selfplay = is_selfplay
        
        self.states = {}
        self.root = NODE(self.cpuct)
        self.states[self.chess.hash_board] = self.root #mapping states of chessboard to the node
        self.random_steps = 6
        
    
    def simulation(self):
        for T in range(self.simulation_times):
# =============================================================================
#             if (T + 1) % 1000 == 0:
#                 print('real_mcts', T)
# =============================================================================
            gameboard = copy.deepcopy(self.chess)
            root = self.root
            actions = []
            val = None #our simulation result for the current first player     1 win 0 tie -1 lose
            end = False
            while True:#go to a leaf
                if root.child: #not a leaf
                    position = root.select()
                    actions.append(position)
                    root = root.child[position]
                    end, winner = gameboard.excute_move(position)
                    if end:
                        val = winner
                        break
                else: #leaf! simulation ending
                    end, winner = gameboard.end_winner()
                    if end == False:
# =============================================================================
#                         prob = 1 / len(gameboard.availables) #replcae it with policy network!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# =============================================================================
                        
                        prob, val = self.policy(gameboard.get_state(self.num_history))
                        if self.chess.player == gameboard.player:
                            val *= -1
                        prob = prob[0]
# =============================================================================
#                         print(prob)
# =============================================================================
                        HASH = gameboard.hash_board
                        for position in gameboard.availables:
                            HASH_tmp = HASH + gameboard.player * base[position]
                            if HASH_tmp in self.states:
                                newnode = self.states[HASH_tmp]
                            else:
                                newnode = NODE(self.cpuct)
                                self.states[HASH_tmp] = newnode
                            root.child[position] = newnode
                            root.edge[position] = EDGE(prob[position])
                        break
                    else:
                        val = winner
                        break
# =============================================================================
#             if val == None:
#                 val = evaluate(gameboard) #replace it with value network!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# =============================================================================
            
            if end == True:
                if val == self.chess.player:
                    val = 1
                else:
                    val = -1
            root = self.root
            for position in actions:
                root.visit_count = root.visit_count + 1
                edge = root.edge[position]
                edge.visit_count = edge.visit_count + 1
                edge.W = edge.W + val
                edge.Q = edge.W / edge.visit_count
                val = val * -1
                root = root.child[position]
    
    def trans_prob(self, prob, temperature):
        eps = 1e-10
        prob = np.power(prob, 1 / temperature) + eps
        tot = np.sum(prob)
        return prob / tot
        
        
    def get_action(self):
# =============================================================================
#         start = time.time()
# =============================================================================
        self.simulation()
        position = []
        prob = []
# =============================================================================
#         print('get_action', len(self.root.edge))
# =============================================================================
        for key in self.root.edge:
            position.append(key)
            prob.append(self.root.edge[key].visit_count)
# =============================================================================
#         print(prob)
# =============================================================================
        prob = self.trans_prob(prob, self.temperature)
# =============================================================================
#         print(prob)
# =============================================================================
# =============================================================================
#         end = time.time()
#         print(np.sum(prob))
#         print(len(self.states))
#         print((end - start) / self.simulation_times * 1000, 's/1000')
#         return position[np.argmax(prob)]
# =============================================================================
        move_probs = np.zeros(self.chess.length * self.chess.length)
        move_probs[position] = prob
        if self.is_selfplay and self.chess.excuted_step <= self.random_steps:
            move = np.random.choice(position, p = 0.75 * prob + 0.25 * np.random.dirichlet(0.3 * np.ones(len(prob))))#0.3?????
        else:
            move = position[np.argmax(prob)]
# =============================================================================
#         if self.is_selfplay:
#             move = np.random.choice(position, p = 0.75 * prob + 0.25 * np.random.dirichlet(0.3 * np.ones(len(prob))))#0.3?????
#         else:
#             move = position[np.argmax(prob)]
# =============================================================================
# =============================================================================
#         print('get_action', move, move_probs)
# =============================================================================
        return int(move), move_probs
    
    def update_action(self, position):
        if type(position) == int:
            index = position
        elif type(position) == tuple:
            index = self.chess.position_to_index(position)
        if index in self.root.child:
            self.root = self.root.child[index]
        else:
            self.root = NODE(self.cpuct)
# =============================================================================
#             print('index invalid')
# =============================================================================
        pop_list = []
        for state in self.states:
            if (state&self.chess.hash_board) != self.chess.hash_board:
                pop_list.append(state)
        
        for state in pop_list:
            self.states.pop(state)