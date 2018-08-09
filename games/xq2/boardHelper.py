#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

# Modified from: https://github.com/thomasahle/sunfish/blob/master/sunfish.py
from __future__ import print_function
import re, sys, time, math
from itertools import count
from collections import OrderedDict, namedtuple

###############################################################################
# Global constants
###############################################################################

# Our board is represented as a 14(rows) * 13(columns) character string. The padding allows for



# Lists of possible moves for each piece type.

N, E, S, W = -13, 1, 13, -1 # -13 是上一行的意思。
directions = {
    'P': (N, W, E),
    'N': (N+N+E, E+E+N, E+E+S, S+S+E, S+S+W, W+W+S, W+W+N, N+N+W),
    'B': (N+N+E+E, S+S+E+E, S+S+W+W, N+N+W+W),
    'R': (N, E, S, W),
    'C': (N, E, S, W),
    'A': (N+E, S+E, S+W, N+W),
    'K': (N, E, S, W)
}

RIVER_BOUND = 13*7 # 这个是index
PALACE_BOUND=((3,5), (0,2))
K_BLOCK=(N, E, E, S, S, W, W, N)
B_BLOCK=(N+E, S+E, S+W, N+W)

class Position(namedtuple('Position', 'board')):
    INDEX09 = 26 # VALID_START_INDEX = 26
    INDEX80 = 151 # VALID_END_INDEX 
    
    """ A state of a chess game
    board -- a 14*13 char representation of the board
    score -- the board evaluation
    """

    initial = (
    '            \n'  #   0 -  12
    '            \n'  #  13 - 25
    '  rnbakabnr \n'  #  26 - 38
    '  ......... \n'  #  39 - 51
    '  .c.....c. \n'  #  52 - 64
    '  p.p.p.p.p \n'  #  65 - 77
    '  ......... \n'  #  78 - 90
    '  ......... \n'  #  91 - 103
    '  P.P.P.P.P \n'  #  104 - 116
    '  .C.....C. \n'  #  117 - 129
    '  ......... \n'  #  130 - 142
    '  RNBAKABNR \n'  #  143 - 155
    '            \n'  #  156 - 168
    '            \n'  # 169 -181
    )

    def getColumnRow(index):
        row = 12 - math.ceil(index/13)
        column = index %13 - 2 

        return column, row # (x,y)

    # e.g.  index 110 is (4,3)
    # print (getColumnRow(110))
    
    def getIndex(logicColumn, logicRow):
        logicRow = 11 - logicRow
        return logicRow * 13 + logicColumn
    
    def move2indexFormat(actionStr):
        # from 0001 -> 
        fromColumn = int(actionStr[0:1])
        fromRow = int(actionStr[1:2])
        toColumn = int(actionStr[2:3])
        toRow = int(actionStr[3:4])
        
        fromPos = fromColumn + 2 + (11 - fromRow ) * 13
        toPos = toColumn + 2 + (11 - toRow ) * 13
        return [fromPos, toPos]
        

    def translateMove(fromIndex, toIndex):
        # fromPos = divmod(fromIndex, 13) # row, column
        
        fromColumn, fromRow = Position.getColumnRow(fromIndex)
        toColumn, toRow  = Position.getColumnRow(toIndex)

        # fromPos = [fromColumn , fromRow] # column, row; or, (x,y)
        # toPos = [ toColumn, toRow]

        return str(fromColumn)+str(fromRow) + str(toColumn) + str(toRow)

    #''' 为了性能，直接在game.py中写了
    #def causeKingsTalk(board):
    #    # 目前暫不使用查表法，感覺性能差別可能不大
    #    redKingIndex = getColumnRow(
    #'''
    
    def gen_moves(board):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        flag = 0
        for i, p in enumerate(board):
            flag +=1
                
            if not p.isupper(): continue
            for index, d in enumerate(directions[p]):
                cannon_jump=False
                for j in count(i+d, d):
                    q = board[j]
                    if q.isspace(): break; # 出盘即退
                        
                    # 遇子时
                    if q != '.':
                        if p != 'C' and  q.isupper(): break # 不是炮时，遇已方子则退
                        if p =='C' and not cannon_jump:  cannon_jump = True; continue # 是炮，开跳
                        if p =='C' and cannon_jump and q.isupper(): break # 是炮时，准备打时，遇已方子则退
                    
                    # Pawn move
                    if p == 'P' and i > RIVER_BOUND and d != N : break # 未过河，只能直走
                        
                    # KING or Adviser
                    if  p  in 'KA':
                        targetColumn, targetRow = Position.getColumnRow(j)

                        if targetColumn< PALACE_BOUND[0][0] or targetColumn > PALACE_BOUND[0][1] or \
                          targetRow < PALACE_BOUND[1][0] or targetRow > PALACE_BOUND[1][1]: break; # 出宫了
                    
                    # Knight move
                    if p== 'N':
                        footPos = i + K_BLOCK[index]
                        if board[footPos] != '.':
                            break # 如果脚处有子，则此方向不能走
                            
                    # Elephant move
                    if p== 'B':
                        waistPos = i + B_BLOCK[index]
                        if board[waistPos] != '.':
                            break # 如果腰处有子，则此方向不能飞
                        if j < RIVER_BOUND:
                            break # 不让过河
                    
                    # Generate this move
                    if (not cannon_jump) or (cannon_jump and q.islower()): yield (i, j)
                        
                    # Stop crawlers from sliding, and sliding after captures
                    if not (p  in 'CR') or (p == 'C' and cannon_jump and q.islower()) or \
                      (p!='C' and q.islower()): break
        
    def rotate(board):
        ''' Rotates the board '''
        return board[::-1].swapcase()
    
            
    ''' 爲了性能，這個方法改爲findRedKingIndex      
    def redKingDie(board):
        if board.find('K') == -1:
            return True
        else:
            return False
    '''  
    def findRedKingIndex(board):
        return board.find('K')
    
    # 爲了性能，這個方法在game中直接使用代碼，則不是使用方法
    '''
    def findBlackKingIndex(board):
        return board.find('k')
    '''
        
    def noSoldier(board):
        if board.find('R') ==-1 and  board.find('N') ==-1 and  board.find('C') ==-1 and  board.find('P') ==-1 and \
          board.find('r') ==-1 and  board.find('n') ==-1 and  board.find('c') ==-1 and  board.find('p') ==-1:
            return True
        else:
            return False
        

    def move(board, move):
        taken = False
        i, j = move
        p, q = board[i], board[j]
        if q !='.': taken = True # 有吃子
        put = lambda board, i, p: board[:i] + p + board[i+1:]
       
        # Actual move
        newBoard = put(board, j, board[i]) # 没有修改board
        newBoard = put(newBoard, i, '.')

        # We rotate the returned position, so it's ready for the next player
        return Position.rotate(newBoard), taken

    def render(board):
        print()
        uni_pieces = {'R':'车', 'N':'马', 'B':'相', 'A':'仕', 'K':'帅', 'P':'兵','C':'炮',
                      'r':'车', 'n':'马', 'b':'象', 'a':'士', 'k':'将', 'p':'卒', 'c':'炮','.':'・'}
        for i, row in enumerate(board.split()):
            print(' ', 9-i, ' '.join(uni_pieces.get(p, p) for p in row))
        print('    ０ １ ２ ３ ４ ５ ６ ７ ８ \n\n')
        # https://zh.wikipedia.org/wiki/%E5%85%A8%E5%BD%A2%E5%92%8C%E5%8D%8A%E5%BD%A2
        # 全形和半形
        
    def getSymmetricalBoard(board):
        # 注意，空格在左右是不对称的，所以，反过来后，要在左面加一空格，右部减一空格。最后还要减一空格
        # 如果是翻转棋，则第一个字符不是空格，最后一个字符也不是回车。这两个需要掉换一下。
        newBoard = board
        symmetricalBoard = ''
        if newBoard[0:1] == '\n': 
            symmetricalBoard = '\n'.join( map(lambda row: row[::-1] ,newBoard.split('\n')))
            symmetricalBoard =  symmetricalBoard[1:] + '\n'
        else:
            symmetricalBoard = '\n'.join( map(lambda row: ' '+ row[::-1][:-1] ,newBoard.split('\n')))[:-1]
            
        return symmetricalBoard
    
    def dongping2strboard(dongping):
        CHESS_MEN = '车马相士帅士相马车炮炮兵兵兵兵兵车马象士将士象马车炮炮卒卒卒卒卒';
        CHESS_MEN_FEN = 'RNBAKABNRCCPPPPPrnbakabnrccppppp';

        board =  (
        '            \n'  #   0 -  12
        '            \n'  #  13 - 25
        )


        for row in range (10):
            board += '  ' # two spaces
            for column in range(9):
                hasChessmen = False
                for i in count(0, 2):
                    if i>=64: 
                        break;

                    if int(dongping[i: i + 1]) == column and  int(dongping[i + 1: i + 2]) == row:
                        # 此位置有子
                        chessmenFen = CHESS_MEN_FEN[int(i / 2)];
                        board += chessmenFen;
                        hasChessmen = True

                if  not hasChessmen:
                    board += "."   
             # when new line
            board += ' \n'  # one space, on new line

        # 2 more lines 
        board += '            \n'  #  156 - 168
        board += '            \n'  # 169 -181
        return board;
