from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.GreedyAgent import GreedyAgent
from algorithms.RandomAgent import RandomAgent
from algorithms.nn.DQN import DQN
from algorithms.nn.EQRC import EQRC

def getAgent(name) -> Type[BaseAgent]:
    if name.startswith('DQN'):
        return DQN

    if name == 'EQRC':
        return EQRC

    if name == 'Random':
        return RandomAgent

    if name == 'Greedy':
        return GreedyAgent

    raise Exception('Unknown algorithm')
