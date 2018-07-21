# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:23:01 2018

@author: sjt
"""
import numpy as np

base = [1]
for i in range(1, 300):
    base.append(base[i - 1] * 3)

class chessboard(object):
    def __init__(self, length = 15, n_in_rows = 5):
        
        #chessboard size
        self.n_in_rows = n_in_rows
        self.length = length
        self.player = int(1)
        self.board = np.zeros((self.length, self.length)).astype(int)#0 empty 1 black 2 white
        
        #valid moves
        self.availables = set()
        limit = self.length * self.length
        for i in range(limit):
            self.availables.add(i)
        
        #direction for check
        self.direction_x = [1, -1, 1, -1, 0, 0, 1, -1]
        self.direction_y = [1, -1, -1, 1, 1, -1, 0, 0]
        
        #value for hashing
        self.hash_board = 0
        
        #record the history board
        self.excuted_step = 0
        self.history1 = [] #record the history of first player
        self.history2 = [] #record the history of second player both are in value 0 or 1
    
    def change_player(self):
        self.player = 3 - self.player
        
    def position_to_index(self, position):
        return self.length * position[0] + position[1]
    
    def index_to_position(self, index):
        return (index // self.length, index % self.length)
    
    def point_in_chessboard(self, x, y):
        return x >= 0 and x < self.length and y >= 0 and y < self.length
    
    def check_point(self, position):
        x = position[0]
        y = position[1]
        player = self.board[position]
        for i in range(4):
            cnt = 1
            for j in range(1, self.n_in_rows):
                dx = x + j * self.direction_x[i * 2]
                dy = y + j * self.direction_y[i * 2]
                #print(dx, dy)
                if (self.point_in_chessboard(dx, dy) and self.board[(dx, dy)] == player):
                    cnt = cnt + 1
                    #print(dx, dy)
                else:
                    break
            for j in range(1, self.n_in_rows):
                dx = x + j * self.direction_x[i * 2 + 1]
                dy = y + j * self.direction_y[i * 2 + 1]
                if (self.point_in_chessboard(dx, dy) and self.board[(dx, dy)] == player):
                    cnt = cnt + 1
                else:
                    break
            if cnt >= self.n_in_rows:
                return True
        return False
    
    def get_state(self, num_history):#need to see as current player
        if self.player == 1:
            state = [np.zeros_like(self.board)]
            for i in range(1, num_history + 1):
                #print('get_state', i)
                if i < self.excuted_step:
                    state = np.concatenate((state, [self.history1[-i]]))
                    state = np.concatenate((state, [self.history2[-i]]))
                else:
                    state = np.concatenate((state, [np.zeros_like(self.board)]))
                    state = np.concatenate((state, [np.zeros_like(self.board)]))
        elif self.player == 2:
            state = [np.ones_like(self.board)]
            for i in range(1, num_history + 1):
                if i < self.excuted_step:
                    state = np.concatenate((state, [self.history2[-i]]))
                    state = np.concatenate((state, [self.history1[-i]]))
                else:
                    state = np.concatenate((state, [np.zeros_like(self.board)]))
                    state = np.concatenate((state, [np.zeros_like(self.board)]))
        return state
            
            
                
            
    def excute_move(self, position): #return excute success or not, has a winner or not
        if type(position) == int:
            index = position
        elif type(position) == tuple:
            index = self.position_to_index(position)
        else:
            print('Input Type Error!')
            return
        
        if index not in self.availables:
            print('Invalid move!')
            return False, 0
        
        self.availables.remove(index)
        
        self.hash_board = self.hash_board + self.player * base[index]
        position = self.index_to_position(index)
        self.board[position] = self.player
        state1 = np.zeros_like(self.board)
        state1[self.board == 1] = 1
        self.history1.append(state1)
        state2 = np.zeros_like(self.board)
        state2[self.board == 2] = 1
        self.history2.append(state2)
        self.excuted_step += 1
        
        end = 0
        winner = 0
        if self.check_point(position) == True:
            winner = self.player
        if winner != 0:
            end = 1
        if len(self.availables) == 0:
            end = 1
        
        self.change_player()
        
        
        return end, winner
    
    def end_winner(self):
        length = self.length
        occupied = list(set(range(length * length)) - set(self.availables))
        for move in occupied:
            x = move % length 
            y = move // length 
            if(self.check_point((y, x)) == True):
                winner = self.board[y][x]
                return True, winner
        if len(occupied) == 0:
            return False, 0
        elif len(self.availables) == 0:
            return True, 0
        return False, 0