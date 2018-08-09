#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import loggers as lg
import sys
sys.path.insert(0, './games/xq2')

from boardHelper import Position
from functools import reduce

class Game:
    syActionsMap = np.loadtxt('SymmetricalActionMap.txt',dtype=int)
    def __init__(self):
        # init_state = '8979695949392919097717866646260600102030405060708012720323436383'
        # http://bp.rifle.im/api/utils/converter

        # 使用4兵的简单形式看看
        # dongping = '9999999949999999999999314299999999999999409999999999993757999999'
        dongping = '8979695949392919097717866646260600102030405060708012720323436383'

        init_state = Position.dongping2strboard(dongping)
        self.gameState = GameState(init_state, 1, 0, []) # 最后一个参数是为了调试

        self.history = [] # 走子的历史

        # TODO
        self.grid_shape = (10,9)  # 10 row, 9 columns
        self.input_shape = (16, 10 , 9) # 14 kinds of pieces, each piece has a 9* 10 position squre array( 1 is 60 ply limits)
        # last one for playerTurn(For position only consider RED forever)
        self.name = 'xiangqi'
        self.state_size = len(self.gameState.binary)
        self.action_size = 2062

    def reset(self):
        # init_state = '8999999949999999999999999999999900999999309999999999999999999999'# http://bp.rifle.im/api/utils/converter
        # 使用4兵的简单形式看看
        # dongping = '9999999949999999999999314299999999999999409999999999993757999999'
        dongping = '8979695949392919097717866646260600102030405060708012720323436383'

        init_state = Position.dongping2strboard(dongping)
        self.gameState = GameState(init_state, 1, 0, []) # 最后一个参数是为了调试
        self.history = [] # 走子的历史
        return self.gameState

    def setState(self,dongping, player): # 此方法仅用于调试之用
        board = Position.dongping2strboard(dongping)
        self.gameState = GameState(board, player, 0, [])
        return self.gameState

    def step(self, action):
        self.history.append(action)
        next_state, value, done, taken = self.gameState.takeAction(action)

        self.gameState = next_state
        info = None

        return ((next_state, value, done, info))

    def identities(self, state, actionValues):
        # 这个是用来计入内存，用来训练的数据
        identities = [(state,actionValues)]

        # 对于每一个局面，同时生成对称局面，以加倍增加训练数据。
        # 注意，空格在左右是不对称的，所以，反过来后，要在左面加一空格，右部减一空格。最后还要减一空格？
        symmetricalBoard = Position.getSymmetricalBoard(state.board)
        if symmetricalBoard == state.board:
            return identities # 如果对称后相同，则不需要再加了。
        symmetricalAV = actionValues[Game.syActionsMap]
        # lg.logger_main.info("Board and symmetricalBoard is: \n%s\n%s",state.board, symmetricalBoard)
        identities.append((GameState(symmetricalBoard, state.playerTurn, state.steps, state.actionsTrack), symmetricalAV))
        return identities


class GameState():
    pieces = {1:'红', -1:'黑'}

    def _getAllActions(): # 这个考虑要做成静态的，以免每次都计算
        # ref: /Users/holibut/vcs/github/chess-alpha-zero/src/chess_zero/config.py

        moves_array = []
        rows = ['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
        columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        # First column, then row
        for l1 in range(10): # l is row
            for n1 in range(9): # n is column
                destinations = [(n1, t) for t in range(10)] + \
                               [(t, l1) for t in range(9)] + \
                               [(n1 + b, l1 + a) for (a, b) in
                                [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
                for (n2, l2) in destinations:
                    if (n1, l1) != (n2, l2) and l2 in range(10) and n2 in range(9):
                        move = columns[n1] + rows[l1] + columns[n2] + rows[l2]
                        moves_array.append(move)
        # end outer for
        # add advisor and Bishop's actions
        # 注：这个和东萍是倒过来的；即东萍的y，从是左上到左下为0至9；而这里，是按坐标系来的，从下到上为0到9.
        moves_advisor_red = ['3041', '5041', '3241', '5241', '4152', '4132', '4150', '4130']
        moves_bishop_red = ['2042', '6042', '2442', '6442', '4264', '4224', '4260', '4220','2002','0220', '0224', '2402',
                             '6082', '8264', '6482', '8260']

        moves_array += moves_advisor_red
        moves_array += moves_bishop_red
        return moves_array # total 2062

    allActions = _getAllActions()

    def __init__(self, board, playerTurn, steps, actionsTrack):

        self.board = board # 棋局的格式显示字符串
        self.steps = steps # 60步限招
        self.playerTurn = playerTurn
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.actionsTrack = actionsTrack  # 此属性仅用于调试之用
        self.allowedActions = self._allowedActions()
        self.value = (0, 0, 0) # 只有需要计算时再计算
        self.score = self._getScore()


    def _allowedActions(self):
        moves = list(Position.gen_moves(self.board))
        translatedMoves = list(map((lambda item: Position.translateMove(item[0], item[1])), moves))

        # 将这样的招法：(106, 93) ->（[[0, 3], [0, 4]]），变为这样：0304; 106是在　14行　×　13列　字符串中的位置。
        indices = np.where(np.in1d(GameState.allActions, translatedMoves))[0]
        assert len(indices) == len(moves)
        return indices


    # 这个用于生成给NN model的输入数据。目前采用15*10行*9列的样式
    # '          车 马相士帅士相马车炮炮兵兵兵兵兵车马象士将士象马车炮炮卒卒卒卒卒'
    # 上面是32个子，重复子的顺序，即第11位是黑将。第4位是红帅
    # ref http://58.241.217.181:15111/notebooks/work/alpha-xerox/step-06.ipynb for detail
    def _binary(self):
        PIECE_POS_INDEX = {'R':0, 'N':1, 'B':2, 'A':3, 'K':4, 'C':5, 'P':6,
                  'r':7, 'n':8, 'b':9, 'a':10, 'k':11, 'c':12, 'p':13}


        sixty_move = np.full((10, 9), self.steps, dtype=np.int)
        player_turn = np.full((10, 9), self.playerTurn, dtype=np.int)
        positions  = np.full((16, 10, 9), 0, dtype=np.int)
        positions[14] = sixty_move
        positions[15] = player_turn

        for i in range(Position.INDEX09, Position.INDEX80):
            if self.board[i] == '.' or self.board[i] == '\n'  or self.board[i] == ' ' : continue
            indexPos = Position.getColumnRow( i)
            piece_position = positions[PIECE_POS_INDEX[self.board[i]]]
            piece_position[indexPos[1]][indexPos[0]] = 1

        # end for
        positions = np.array(positions)
        # assert positions.shape == (16, 10, 9)
        return positions


    # 在agent的   evaluateLeaf中，有：if newState.id not in self.mcts.tree:
    # 即这个id用于判断此局面是否在树中已存在，是局面的唯一标识。使用东萍串很合适。
    def _convertStateToId(self):
        return self.board + str(self.steps)

    def checkForEndGame(self):
        redKingIndex = self.board.find('K')
        # 自己老将没了，则游戏结束
        if redKingIndex == -1:
            return 1

        redKingColumn, redKingRow = Position.getColumnRow(redKingIndex)
        blackKingIndex = self.board.find('k')
        blackKingColumn, blackKingRow = Position.getColumnRow(blackKingIndex)
        if redKingColumn == blackKingColumn:
            causeKingsTalked = True
            newPos = blackKingIndex + 13
            while newPos < redKingIndex:
                if self.board[newPos] != '.':
                    causeKingsTalked = False
                    break
                newPos += 13
            if causeKingsTalked: # 难得有种情况，对方走后，自己胜了：哈哈哈,你个笨蛋，我不走都能赢！！
                return 1

        # list(Position.gen_moves(self.board)) moves为空，则结束。TODO

        # 如果都没有长牙的子，则游戏结束
        if Position.noSoldier(self.board):
            return 1

        if self.steps >= 40: # 为了性能，暂时定为20回合就和棋。120:
            return 1

        # 0，游戏继续
        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        # 这个如果当前玩家的状态为输了，则返回：(-1, -1, 1)；否则返回：(0, 0, 0)
        # 在哪儿考虑平局呢？感觉是在外面处理的：结束了没赢，就是平？
        # 红为1，黑为-1.
        # 自己老将没了，输；否则，0

        redKingIndex = self.board.find('K')
        # 自己老将没了，则游戏结束,　负
        if redKingIndex == -1:
            self.value = (-1, -1, 1)
            return (-1, -1, 1)

        # 明将，胜
        redKingColumn, redKingRow = Position.getColumnRow(redKingIndex)
        blackKingIndex = self.board.find('k')
        blackKingColumn, blackKingRow = Position.getColumnRow(blackKingIndex)
        if redKingColumn == blackKingColumn:
            causeKingsTalked = True
            newPos = blackKingIndex + 13
            while newPos < redKingIndex:
                if self.board[newPos] != '.':
                    causeKingsTalked = False
                    break
                newPos += 13
            if causeKingsTalked: # 难得有种情况，对方走后，自己胜了：哈哈哈,你个笨蛋，我不走都能赢！！
                self.value = (1, 1, -1)
                return (1, 1, -1)

        self.value = (0, 0, 0)
        return (0, 0, 0)


    def _getScore(self):
        # 爲什麼有value，还有score: 为了兼容各种游戏，有的有权重
        tmp = self.value
        # value有三个位，第一个：0,和；1胜；－1负；
        # 第2个，已方score，无权重时，与第1位相同
        # 第3个，对方score，无权重时，与第1位相反
        return (tmp[1], tmp[2])


    def takeAction(self, actionIndex):
        # 注意，MCTS直接调用了这个方法，用于模拟；所以在这里，要考虑限招问题
        # 这里的action, 是action index
        actionStr = GameState.allActions[actionIndex]

        # debug only
        newTrack = self.actionsTrack + [actionStr] # 不修改原来的list
        # deng ends.

        action = Position.move2indexFormat(actionStr)
        newBoard, taken = Position.move(self.board, action)

        # taken是有吃子的意思
        newSteps = self.steps
        if taken : # taken is True or False
            newSteps =0 # 有吃子的话，则限招重新计数
        else:
            newSteps += 1 # 否则，限招计数加1
        newState = GameState(newBoard, -self.playerTurn,  newSteps, newTrack) # newTrack 是为了调试

        value = 0
        done = 0

        isEndGame = newState.checkForEndGame()
        if isEndGame:
            value = newState._getValue()[0]
            done = 1

        return (newState, value, done, taken)


    def render(self, logger):
        return
