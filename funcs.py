import numpy as np
import random

import loggers as lg

from game import Game, GameState
from model import Residual_CNN

from agent import Agent, User

import config


def playMatchesBetweenVersions(env, run_version, player1version, player2version, EPISODES, logger, turns_until_tau0, goes_first=0):
    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE,
                                  env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

        if player1version > 0:
            player1_network = player1_NN.read(
                env.name, run_version, player1version)
            player1_NN.model.set_weights(player1_network.get_weights())
        player1 = Agent('player1', env.state_size, env.action_size,
                        config.MCTS_SIMS, config.CPUCT, player1_NN)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE,
                                  env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

        if player2version > 0:
            player2_network = player2_NN.read(
                env.name, run_version, player2version)
            player2_NN.model.set_weights(player2_network.get_weights())
        player2 = Agent('player2', env.state_size, env.action_size,
                        config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(
        player1, player2, EPISODES, logger, turns_until_tau0, None, goes_first)

    return (scores, memory, points, sp_scores)


def playMatches(player1, player2, EPISODES, logger, turns_until_tau0, memory=None, goes_first=0):

    env = Game()

    scores = {player1.name: 0, "drawn": 0, player2.name: 0}
    sp_scores = {'sp': 0, "drawn": 0, 'nsp': 0}
    points = {player1.name: [], player2.name: []}

    for e in range(EPISODES):

        logger.info('====================')
        logger.info('EPISODE %d OF %d', e+1, EPISODES)
        logger.info('====================')

        print(str(e+1) + ' ', end='')

        state = env.reset()

        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        # 这个参数一直未传，即在对战时，第一玩家的先后手目前是随机的。
        if goes_first == 0:
            player1Starts = random.randint(0, 1) * 2 - 1
        else:
            player1Starts = goes_first

        if player1Starts == 1:
            players = {1: {"agent": player1, "name": player1.name}, -1: {"agent": player2, "name": player2.name}
                       }
            logger.info(player1.name + ' plays as 红')
        else:
            players = {1: {"agent": player2, "name": player2.name}, -1: {"agent": player1, "name": player1.name}
                       }


        # 不结束的话，这局就一直下下去
        while done == 0:
            turn = turn + 1

            # Run the MCTS algo and return an action
            # 规定步数之前，策略有不同。什么不同？？？目前是10步
            # 下面，开始找到当前状态的最佳着法；使用NN了么？？？
            if turn < turns_until_tau0:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(
                    state, 1)
            else:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(
                    state, 0)

            # 每一步，都记入内存。注意，这里是记入短程内存：short term memory
            if memory != None:
                # Commit the move to memory
                memory.commit_stmemory(env.identities, state, pi)

            # 执行相应的招法
            # Do the action
            # the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move
            state, value, done, _ = env.step(action)
 
            if done == 1:
                # 调试之用，打印当前局所有招法
                if memory != None:
                    # If the game is finished, assign the values correctly to the game moves
                    # 如果结束了，就更新短程内存中的步数的分值 ，然后移入长程内存。
                    for move in memory.stmemory:
                        if move['playerTurn'] == state.playerTurn:
                            move['value'] = value
                        else:
                            move['value'] = -value

                    memory.commit_ltmemory()

                if value == 1:
                    logger.info('%s(%s) WINS! History: %s, value=%d.', players[state.playerTurn]['name'], GameState.pieces[state.playerTurn],str(env.history),   value)
                    scores[players[state.playerTurn]['name']
                           ] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1:
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                elif value == -1:
                    logger.info('%s(%s) WINS! History: %s, value=%d.', players[-state.playerTurn]['name'], GameState.pieces[-state.playerTurn], str(env.history),   value)
                    scores[players[-state.playerTurn]['name']
                           ] = scores[players[-state.playerTurn]['name']] + 1

                    if state.playerTurn == 1:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1

                else:
                    logger.info('DRAW... History: %s, value=%d.', str(env.history), value)
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[-state.playerTurn]['name']].append(pts[1])

    return (scores, memory, points, sp_scores)
