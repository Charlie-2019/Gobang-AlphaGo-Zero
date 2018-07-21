#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:32:01 2018

@author: root
"""
from chessboard import chessboard
import math
import copy
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
    def __init__(self):
        self.child = {} #mapping position(action) to node
        self.edge = {} #mapping position(action) to edge(state, action)
        self.visit_count = 0
        self.cpuct = 5
    
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

def evaluate(gameboard):
    chess = copy.deepcopy(gameboard)
    while True:
        index = int(np.random.choice(list(chess.availables), 1)[0])
        end, winner = chess.excute_move(index)
        if end:
            break
    return winner
    
class pure_mcts(object):
    def __init__(self, chess, simulation_times):
        self.chess = chess
        self.states = {}
        self.root = NODE()
        self.states[self.chess.hash_board] = self.root #mapping states of chessboard to the node
        self.simulation_times = simulation_times
    
    def simulation(self):
        for T in range(self.simulation_times):
# =============================================================================
#             if (T + 1) % 1000 == 0:
#                 print(T)
# =============================================================================
            gameboard = copy.deepcopy(self.chess)
            root = self.root
            actions = []
            val = None #our simulation result for the current first player     1 win 0 tie -1 lose
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
                        prob = 1 / len(gameboard.availables) #replcae it with policy network!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        HASH = gameboard.hash_board
                        for position in gameboard.availables:
                            HASH_tmp = HASH + gameboard.player * base[position]
                            if HASH_tmp in self.states:
                                newnode = self.states[HASH_tmp]
                            else:
                                newnode = NODE()
                                self.states[HASH_tmp] = newnode
                            root.child[position] = newnode
                            root.edge[position] = EDGE(prob)
                        break
                    else:
                        val = winner
                        break
            if val == None:
                val = evaluate(gameboard) #replace it with value network!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            if val != 0:
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
            
    def get_action(self):
# =============================================================================
#         start = time.time()
# =============================================================================
        self.simulation()
        position = []
        prob = []
        for key in self.root.edge:
            position.append(key)
            prob.append(self.root.edge[key].visit_count)
# =============================================================================
#         end = time.time()
#         print(np.sum(prob))
#         print(len(self.states))
#         print((end - start) / self.simulation_times * 1000, 's/1000')
# =============================================================================
        return position[np.argmax(prob)], 0
    
    def update_action(self, position):
#        index = self.chess.position_to_index(position)
        index = position
        if index in self.root.child:
            self.root = self.root.child[index]
        else:
            self.root = NODE()
# =============================================================================
#             print('index invalid')
# =============================================================================
        pop_list = []
        for state in self.states:
            if (state & self.chess.hash_board) != self.chess.hash_board:
                pop_list.append(state)
        
        for state in pop_list:
            self.states.pop(state)
    
if __name__ == '__main__':
    pure_mcts(chessboard(6, n_in_rows=4))