"""
Opponent Strategy System for Multi-Round Games

Provides configurable opponent strategies for games like Prisoner's Dilemma
and Stag Hunt, where the LLM plays against a simulated opponent.
"""

import random
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class OpponentStrategyType(Enum):
    ALWAYS_COOPERATE = "always_cooperate"
    ALWAYS_DEFECT = "always_defect"
    RANDOM = "random"
    TIT_FOR_TAT = "tit_for_tat"
    CUSTOM = "custom"


@dataclass
class OpponentStrategy:
    """
    Configurable opponent strategy for multi-round games.

    Args:
        strategy_type: Which strategy the opponent follows
        p_cooperate: Probability of cooperate-equivalent action (for RANDOM)
        custom_sequence: Fixed action sequence (for CUSTOM)
        cooperate_label: Game-specific label for the cooperative action
        defect_label: Game-specific label for the defecting action
    """
    strategy_type: OpponentStrategyType
    p_cooperate: float = 0.5
    custom_sequence: Optional[List[str]] = None
    cooperate_label: str = "Push"
    defect_label: str = "Pull"

    def get_action(self, round_num: int, opponent_history: List[str]) -> str:
        """
        Get opponent's action for this round.

        Args:
            round_num: 1-indexed round number
            opponent_history: The LLM's previous actions (what the opponent observes)

        Returns:
            Action string (cooperate_label or defect_label)
        """
        if self.strategy_type == OpponentStrategyType.ALWAYS_COOPERATE:
            return self.cooperate_label
        elif self.strategy_type == OpponentStrategyType.ALWAYS_DEFECT:
            return self.defect_label
        elif self.strategy_type == OpponentStrategyType.RANDOM:
            return (self.cooperate_label
                    if random.random() < self.p_cooperate
                    else self.defect_label)
        elif self.strategy_type == OpponentStrategyType.TIT_FOR_TAT:
            if round_num == 1 or not opponent_history:
                return self.cooperate_label
            return opponent_history[-1]
        elif self.strategy_type == OpponentStrategyType.CUSTOM:
            if self.custom_sequence and (round_num - 1) < len(self.custom_sequence):
                return self.custom_sequence[round_num - 1]
            return self.cooperate_label
        raise ValueError(f"Unknown strategy type: {self.strategy_type}")
