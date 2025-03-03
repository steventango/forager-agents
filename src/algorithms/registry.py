from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.GreedyAgent import GreedyAgent
from algorithms.RandomAgent import RandomAgent
from algorithms.nn.DQN import DQN
from algorithms.nn.EQRC import EQRC
from algorithms.nn.SAC import SAC

def getAgent(name) -> Type[BaseAgent]:
    if name.startswith('DQN'):
        return DQN

    if name.startswith('SAC'):
        return SAC

    if name == 'EQRC':
        return EQRC

    if name.startswith('Random'):
        return RandomAgent

    if name.startswith('Greedy'):
        return GreedyAgent

    raise Exception('Unknown algorithm')
