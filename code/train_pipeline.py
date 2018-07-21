#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 21:34:49 2018

@author: sjt
"""

import random
import numpy as np
import copy
import time
from collections import defaultdict, deque
from chessboard import chessboard
from real_mcts import real_mcts
from pure_mcts import pure_mcts
from policy_value_net import PolicyValueNet
from interface import interface

# =============================================================================
#             winner = start_play(current_real_mcts,
#                                 current_puer_mcts,
#                                 start_player=i % 2)
# =============================================================================
def start_play(player1, player2, start_player):#watch about the chessboard
    cnt = 0
    while True:
        if (cnt % 2 == start_player):
# =============================================================================
#             print('start_player', start_player, ' cnt', cnt)
# =============================================================================
            position, _ = player1.get_action()
            end, winner = player1.chess.excute_move(position)
            player1.update_action(position)
            player2.update_action(position)
        else:
# =============================================================================
#             print('start_player', start_player, ' cnt', cnt)
# =============================================================================
            position, _ = player2.get_action()
            end, winner = player2.chess.excute_move(position)
            player1.update_action(position)
            player2.update_action(position)
        cnt = cnt + 1
        if end:
            return winner

def start_self_play(player):#watch about the chessboard
    states, mcts_probs, current_players = [], [], []
    while True:
        current_players.append(player.chess.player)
        position, probs = player.get_action()
        end, winner = player.chess.excute_move(position)
        states.append(player.chess.get_state(player.num_history))
        mcts_probs.append(probs)
        player.update_action(position)
        
        if end:
            z = np.zeros(len(current_players))
            z[np.array(current_players) == winner] = 1
            z[np.array(current_players) != winner] = -1
            break
    return zip(states, mcts_probs, z)

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_length = 6
        self.n_in_row = 4
        self.num_history = 2
        self.chess = chessboard(self.board_length, self.n_in_row)
        # training params
        self.learn_rate = 5e-4
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temperature = 1.0  # the temperature param
        self.cpuct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 10
        self.kl_targ = 0.02
        self.check_freq = 50
        self.best_win_ratio = 0.0
        self.game_batch_num = 4000
        self.loss_dict = {}
        self.loss_hold = 50
        
        self.real_mcts_simulation_times = 400
        self.pure_mcts_simulation_times = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_length,
                                                   self.num_history,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_length,
                                                   self.num_history)
# =============================================================================
#         deepcopy self.chess or not???????????????????????????????????????????
# =============================================================================
        self.mcts_player = real_mcts(self.chess,
                            self.policy_value_net.policy_value,
                            self.cpuct,
                            self.real_mcts_simulation_times,
                            self.temperature,
                            self.num_history,
                            True)
# =============================================================================
#         self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
#                                       c_puct=self.c_puct,
#                                       n_playout=self.n_playout,
#                                       is_selfplay=1)
# =============================================================================

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_length, self.board_length)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games = 1):
        for i in range(n_games):
            inter = interface(self.board_length)
            current_board = copy.deepcopy(self.chess)
            current_real_mcts = real_mcts(current_board,
                                          self.policy_value_net.policy_value,
                                          self.cpuct,
                                          self.real_mcts_simulation_times,
                                          self.temperature,
                                          self.num_history,
                                          True)
            play_data = inter.start_self_play(player = current_real_mcts)
# =============================================================================
#             play_data = start_self_play(player = current_real_mcts)
# =============================================================================
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
# =============================================================================
#         mini_batch = self.data_buffer
# =============================================================================
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        first_loss = 0
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch,
                                                             mcts_probs_batch,
                                                             winner_batch,
                                                             self.learn_rate * self.lr_multiplier)
            if i == 0:
                first_loss = loss
# =============================================================================
#             if i % 10 == 0:
#                 print('loss: ', loss, ' entropy: ', entropy)
# =============================================================================
# =============================================================================
#             print('loss: ', loss, ' entropy: ', entropy)
# =============================================================================
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),axis=1))
# =============================================================================
#             if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
#                 break
#         # adaptively adjust the learning rate
#         if kl > self.kl_targ * 2 and self.lr_multiplier > 0.01:
#             self.lr_multiplier /= 1.5
#         elif kl < self.kl_targ / 2 and self.lr_multiplier < 100:
#             self.lr_multiplier *= 1.5
# =============================================================================

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "loss_change:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        first_loss - loss,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        win_cnt = defaultdict(int)
        
        for i in range(n_games):
            inter = interface(self.board_length)
            current_board = copy.deepcopy(self.chess)
            current_real_mcts = real_mcts(current_board,
                                self.policy_value_net.policy_value,
                                self.cpuct,
                                1000,
                                self.temperature,
                                self.num_history,
                                False)
            current_pure_mcts = pure_mcts(current_board,
                                          self.pure_mcts_simulation_times)
            winner = inter.start_play(current_real_mcts,
                                      current_pure_mcts,
                                      start_player=i % 2)
            win_cnt[winner] += 1
            print('winner', winner)
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[0]) / n_games
        print("num_simulation_times:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_simulation_times,win_cnt[1], win_cnt[2], win_cnt[0]))
        return win_ratio

    def run(self):
        total = 0
        for i in range(self.game_batch_num):
            if (i + 1) % 100 == 0:
                self.learn_rate = self.learn_rate * 0.85
# =============================================================================
#             start = time.time()
# =============================================================================
            self.collect_selfplay_data(self.play_batch_size)
            if len(self.data_buffer) >= self.batch_size:
                loss, entropy = self.policy_update()
                self.loss_dict[i] = loss
                total += loss
            if (i - self.loss_hold) in self.loss_dict:
                total -= self.loss_dict[i - self.loss_hold]
                self.loss_dict.pop(i - self.loss_hold)
            print("batch i:{}, episode_len:{}, loss_hist:{}".format(i + 1, self.episode_len, total / self.loss_hold))
            if (i + 1) % self.check_freq == 0:
                print("current self-play batch: {}".format(i+1))
                win_ratio = self.policy_evaluate()
                self.policy_value_net.save_model('./current_policy.model')
                if win_ratio > self.best_win_ratio:
                    print("New best policy!!!!!!!!")
                    self.best_win_ratio = win_ratio
                    self.policy_value_net.save_model('./best_policy.model')
                    if (self.best_win_ratio == 1.0 and self.pure_mcts_simulation_times < 10000):
                        self.pure_mcts_simulation_times += 1000
                        self.best_win_ratio = 0.0
# =============================================================================
#             end = time.time()
#             print(end - start)
# =============================================================================


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
# =============================================================================
#     training_pipeline = TrainPipeline(init_model = './current_policy.model')
# =============================================================================
    training_pipeline.run()