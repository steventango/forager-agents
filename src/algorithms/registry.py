from typing import Type

from algorithms.BaseAgent import BaseAgent
from algorithms.GreedyAgent import GreedyAgent
from algorithms.nn.DQN import DQN
from algorithms.nn.PT_DQN import PT_DQN
from algorithms.nn.DRQN import DRQN
from algorithms.nn.EQRC import EQRC
from algorithms.nn.SAC import SAC
from algorithms.RandomAgent import RandomAgent


def getAgent(name) -> Type[BaseAgent]:
    if name.startswith("DQN") or name.startswith("W0-DQN"):
        return DQN

    if name.startswith("PT_DQN"):
        return PT_DQN

    if name.startswith("DRQN"):
        return DRQN

    if name.startswith("SAC"):
        return SAC

    if name == "EQRC":
        return EQRC

    if name.startswith("Random"):
        return RandomAgent

    if name.startswith("Greedy"):
        return GreedyAgent

    raise Exception("Unknown algorithm")
