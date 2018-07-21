import pygame
import numpy as np
from chessboard import chessboard
from pure_mcts import pure_mcts
from real_mcts import real_mcts
from policy_value_net import PolicyValueNet

class interface(object):
    def __init__(self, LENGTH):
        self.LENGTH = LENGTH - 1
        self.pixel = 50
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        self.WIDTH = (self.LENGTH + 2) * self.pixel
        self.GRID_WIDTH = self.WIDTH // (self.LENGTH + 2)
        self.HEIGHT = (self.LENGTH + 2) * self.pixel
        self.new_x = []
        self.new_y = []
        self.FPS = 30
        self.clock = pygame.time.Clock()
        self.chess = chessboard(length=self.LENGTH + 1, n_in_rows=4)
    
    def draw_background(self, surf):
        # 加载背景图片
        surf.fill(self.YELLOW)
        rect_lines = [
            ((self.GRID_WIDTH, self.GRID_WIDTH), (self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH)),
            ((self.GRID_WIDTH, self.GRID_WIDTH), (self.WIDTH - self.GRID_WIDTH, self.GRID_WIDTH)),
            ((self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH),
             (self.WIDTH - self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH)),
            ((self.WIDTH - self.GRID_WIDTH, self.GRID_WIDTH),
             (self.WIDTH - self.GRID_WIDTH, self.HEIGHT - self.GRID_WIDTH)),
        ]
        for line in rect_lines:
            pygame.draw.line(surf, self.BLACK, line[0], line[1], 2)

        # 画出中间的网格线
        for i in range(self.LENGTH - 1):
            pygame.draw.line(surf, self.BLACK,
                             (self.GRID_WIDTH * (2 + i), self.GRID_WIDTH),
                             (self.GRID_WIDTH * (2 + i), self.HEIGHT - self.GRID_WIDTH))
            pygame.draw.line(surf, self.BLACK,
                             (self.GRID_WIDTH, self.GRID_WIDTH * (2 + i)),
                             (self.HEIGHT - self.GRID_WIDTH, self.GRID_WIDTH * (2 + i)))
        for i in range(len(self.new_x)):
            if i % 2 == 0:
                pygame.draw.circle(surf, self.BLACK, (self.new_x[i], self.new_y[i]), 16)
            else:
                pygame.draw.circle(surf, self.WHITE, (self.new_x[i], self.new_y[i]), 16)

    def get_last_position(self):
        if len(self.new_x):
            return (self.new_y[len(self.new_y) - 1] // self.pixel - 1, \
                    self.new_x[len(self.new_x) - 1] // self.pixel - 1)
            
    def mcts_position(self, pos):
        i = pos[0]
        j = pos[1]
        #if i in range(0, self.LENGTH + 1) and j in range(0, self.LENGTH + 1) and (i * (self.LENGTH + 1) + j) in self.chess.availables:
        if self.chess.position_to_index(pos) in self.chess.availables:
            nx = self.pixel + j * self.pixel
            ny = self.pixel + i * self.pixel
            self.new_x.append(nx)
            self.new_y.append(ny)
            return True
        else:
            print('false')
            return False

    def mouse_click(self, pos):
        pressed_x, pressed_y = pos[0], pos[1]
        mouse_chessboard_x, mouse_chessboard_y = pressed_x - self.pixel, pressed_y - self.pixel
        i, j = round(mouse_chessboard_y / self.pixel), round(mouse_chessboard_x / self.pixel)
        if i in range(0, self.LENGTH + 1) and j in range(0, self.LENGTH + 1) and (i * (self.LENGTH + 1) + j) in self.chess.availables:
            chessboard_x = self.pixel + (j) * self.pixel
            chessboard_y = self.pixel + (i) * self.pixel
            self.new_x.append(chessboard_x)
            self.new_y.append(chessboard_y)
            return True
        else:
            return False

    def player1(self, pos):
        flag = self.mouse_click(pos)
        return flag, self.get_last_position()

    def player2(self, pos):
        flag = self.mcts_position(pos)
        return flag, self.get_last_position()
    
    def start_play(self, player1, player2, start_player):#watch about the chessboard
        pygame.init()
        screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("五子棋")
        cnt = 0
        while True:
            self.clock.tick(self.FPS)
            if (cnt % 2 == start_player):
                position, _ = player1.get_action()
                flag, position = self.player2(self.chess.index_to_position(position))
                end, winner = player1.chess.excute_move(position)
                player1.update_action(position)
                player2.update_action(position)
            else:
                position, _ = player2.get_action()
                flag, position = self.player2(self.chess.index_to_position(position))
                end, winner = player2.chess.excute_move(position)
                player1.update_action(position)
                player2.update_action(position)
            self.draw_background(screen)
            pygame.display.flip()
            cnt = cnt + 1
            if end:
                if winner == 0:
                    return winner
                elif start_player == 0:
                    return winner
                else:
                    return 3 - winner

    def start_self_play(self, player):#watch about the chessboard
        pygame.init()
        screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("五子棋")
        states, mcts_probs, current_players = [], [], []
        while True:
            current_players.append(player.chess.player)
            position, probs = player.get_action()
            flag, position = self.player2(self.chess.index_to_position(position))
            end, winner = player.chess.excute_move(position)
            states.append(player.chess.get_state(player.num_history))
            mcts_probs.append(probs)
            player.update_action(position)
            self.draw_background(screen)
            pygame.display.flip()
            
            if end:
                z = np.zeros(len(current_players))
                z[np.array(current_players) == winner] = 1
                z[np.array(current_players) != winner] = -1
                pygame.quit()
                break
        return zip(states, mcts_probs, z)

    def run1(self):
        pygame.init()
        screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        policy_value_net = PolicyValueNet(self.chess.length, num_history = 2)
# =============================================================================
#         policy_value_net = PolicyValueNet(self.chess.length, num_history = 2, model_file='./best_policy.model')
# =============================================================================
        puremcts = pure_mcts(self.chess, 5000)
# =============================================================================
#         puremcts = real_mcts(self.chess,
#                              policy_value_net.policy_value,
#                              5,
#                              1000,
#                              1,
#                              2,
#                              False)
# =============================================================================
        pygame.display.set_caption("五子棋")
        while True:
            self.clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
# =============================================================================
#                     prob, val = puremcts.policy(self.chess.get_state(2))
# =============================================================================
# =============================================================================
#                     print(prob, val)
# =============================================================================
                    if self.chess.player == 1:
                        position, _ = puremcts.get_action()
                        flag, position = self.player2(self.chess.index_to_position(position))
                        #flag, position = self.player1(event.pos)
                    if flag:
                        end, winner = self.chess.excute_move(position)
# =============================================================================
#                         print('1:////////////////////')
#                         print(self.chess.get_state(2))
# =============================================================================
                        if end:
                            print("winner:", winner)
                            pygame.quit()
                            exit()
                        puremcts.update_action(position)
                        self.draw_background(screen)
                        pygame.display.flip()
                        position, _ = puremcts.get_action()
                        flag, position = self.player2(self.chess.index_to_position(position))
                        if flag:
                            end, winner = self.chess.excute_move(position)
# =============================================================================
#                             print('2:////////////////////')
#                             print(self.chess.get_state(2))
# =============================================================================
                            puremcts.update_action(position)
                            if end:
                                print("winner:", winner)
                                pygame.quit()
                                exit()
            self.draw_background(screen)

            pygame.display.flip()
            
    def run2(self):
        policy_value_net = PolicyValueNet(self.chess.length, num_history = 2, model_file='current_policy.model')
        
# =============================================================================
#         policy_value_net = PolicyValueNet(self.chess.length, num_history = 2)
# =============================================================================
# =============================================================================
#         deepcopy self.chess or not???????????????????????????????????????????
# =============================================================================
        mcts_player = real_mcts(self.chess,
                                     policy_value_net.policy_value,
                                     5,
                                     1000,
                                     1,
                                     2,
                                     False)
# =============================================================================
#         mcts_player = pure_mcts(self.chess, 10000)
# =============================================================================
        pygame.init()
        screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("五子棋")
        
        position, _ = mcts_player.get_action()
        flag, position = self.player2(self.chess.index_to_position(position))
        if flag:
            end, winner = self.chess.excute_move(position)
            mcts_player.update_action(position)
            if end:
                print("winner:", winner)
                pygame.quit()
                print(self.chess.board)
                exit()
        self.draw_background(screen)
        pygame.display.flip()
        
        while True:
            self.clock.tick(self.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
# =============================================================================
#                     if self.chess.player == 1:
# =============================================================================
                        #position, _ = self.mcts.get_action()
                        #flag, position = self.player1(self.chess.index_to_position(position))
                    flag, position = self.player1(event.pos)
                    if flag:
                        end, winner = self.chess.excute_move(position)
                        if end:
                            print("winner:", winner)
                            pygame.quit()
                            exit()
                        mcts_player.update_action(position)
                        self.draw_background(screen)
                        pygame.display.flip()
                        position, _ = mcts_player.get_action()
                        flag, position = self.player2(self.chess.index_to_position(position))
                        if flag:
                            end, winner = self.chess.excute_move(position)
                            mcts_player.update_action(position)
                            if end:
                                print("winner:", winner)
                                pygame.quit()
                                print(self.chess.board)
                                exit()
            self.draw_background(screen)

            pygame.display.flip()

if __name__ == '__main__':
    chess = interface(6)
    chess.run2()