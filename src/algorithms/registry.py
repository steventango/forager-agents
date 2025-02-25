from typing import Type
from algorithms.BaseAgent import BaseAgent

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

    if name == 'Random':
        return RandomAgent

    raise Exception('Unknown algorithm')
