"""
Game-Specific Belief Elicitation Configurations

Pre-configured belief elicitation setups for games where beliefs
have genuine strategic relevance (no dominant strategy equilibrium):

- Prisoner's Dilemma (repeated): beliefs about opponent's Push/Pull
- Stag Hunt (repeated): beliefs about partner's Stag/Hare
- Ultimatum Game (Proposer): beliefs about responder acceptance
- p-Beauty Contest: higher-order beliefs about others' choices
- First-Price Sealed-Bid Auction: beliefs about opponent's bid
"""

from typing import Dict, List, Any, Optional
from .core import (
    BeliefQuestion, BeliefType, ScoringRule,
    ElicitationPoint, GameBeliefConfig,
    create_simple_probability_question,
    create_expectation_question
)


# =============================================================================
# PRISONER'S DILEMMA
# =============================================================================

def get_prisoners_dilemma_config(
    n_rounds: int = 5,
    delta: float = 0.9,
    incentivized: bool = True
) -> GameBeliefConfig:
    """
    Belief configuration for infinitely repeated Prisoner's Dilemma
    with restricted type space {cooperative (TFT), selfish (always Pull)}.

    Elicits two beliefs per round (pre-decision):
    1. P(opponent is cooperative type) — the theoretically relevant belief
    2. P(Push) vs P(Pull) this round — the action-level belief

    Args:
        n_rounds: Number of rounds to play (game stops internally; LLM
                  is told continuation probability delta, not total rounds).
        delta: Continuation probability per round (default 0.9).
    """
    elicitation_points = []

    for r in range(1, n_rounds + 1):
        preamble = (
            "Before you make your decision, I'd like to understand your expectations."
            if r == 1 else
            f"Before round {r}, given what you've observed so far:"
        )

        # Q1: Opponent type belief
        q_type = BeliefQuestion(
            question_id=f"opponent_type_r{r}",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="Opponent's type",
            outcomes=["Cooperative", "Selfish"],
            prompt_template=f"""
{preamble}

{{incentive_explanation}}

Recall that the other player is one of two types:
- Cooperative type: starts with Push and mirrors your previous choice (tit-for-tat)
- Selfish type: always plays Pull

What probability (0-100%) do you assign to each type?

- Cooperative: ___%
- Selfish: ___%

(Must sum to 100%. Highlight each probability, e.g., [60]% and [40]%)
""",
            include_incentive_explanation=(r == 1),
            scoring_rule=ScoringRule.QUADRATIC
        )

        # Q2: Action belief (per-round)
        q_action = BeliefQuestion(
            question_id=f"opponent_action_r{r}",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="Opponent's action this round",
            outcomes=["Push", "Pull"],
            prompt_template=f"""
What probability (0-100%) do you assign to the other player's
action in this round?

- Push: ___%
- Pull: ___%

(Must sum to 100%. Highlight each probability, e.g., [60]% and [40]%)
""",
            include_incentive_explanation=False,
            scoring_rule=ScoringRule.QUADRATIC
        )

        elicitation_points.append(ElicitationPoint(
            point_id=f"round_{r}_beliefs",
            at_prompt_index=r,
            questions=[q_type, q_action]
        ))

    return GameBeliefConfig(
        game_name="prisoners_dilemma",
        elicitation_points=elicitation_points,
        incentivized=incentivized
    )


# =============================================================================
# STAG HUNT
# =============================================================================

def get_stag_hunt_config(
    n_rounds: int = 5,
    incentivized: bool = True
) -> GameBeliefConfig:
    """
    Belief configuration for repeated Stag Hunt.

    Two pure NE: (Stag,Stag) payoff-dominant, (Hare,Hare) risk-dominant.
    Beliefs about partner's choice determine equilibrium selection.
    """
    elicitation_points = []

    for r in range(1, n_rounds + 1):
        preamble = (
            "Before you make your decision, I'd like to understand your expectations "
            "about the other player's likely choice."
            if r == 1 else
            f"For round {r}, before you decide:"
        )
        q = BeliefQuestion(
            question_id=f"partner_action_r{r}",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="Partner's action",
            outcomes=["Stag", "Hare"],
            prompt_template=f"""
{preamble}

{{incentive_explanation}}

What probability (0-100%) do you assign to each of the other player's
possible actions in this round?

- Stag: ___%
- Hare: ___%

(Must sum to 100%. Highlight each probability, e.g., [60]% and [40]%)
""",
            include_incentive_explanation=(r == 1),
            scoring_rule=ScoringRule.QUADRATIC
        )
        elicitation_points.append(ElicitationPoint(
            point_id=f"round_{r}_beliefs",
            at_prompt_index=r,
            questions=[q]
        ))

    return GameBeliefConfig(
        game_name="stag_hunt",
        elicitation_points=elicitation_points,
        incentivized=incentivized
    )


# =============================================================================
# ULTIMATUM GAME (PROPOSER)
# =============================================================================

def get_ultimatum_proposer_config(
    incentivized: bool = True
) -> GameBeliefConfig:
    """
    Belief configuration for Ultimatum Game - Proposer role.

    Elicits beliefs about responder acceptance probability at various offer levels.
    No dominant strategy: optimal offer depends on beliefs about acceptance.
    """
    acceptance_questions = []
    for offer in [10, 20, 30, 40, 50]:
        acceptance_questions.append(
            BeliefQuestion(
                question_id=f"accept_prob_{offer}",
                belief_type=BeliefType.PROBABILITY,
                target_variable=f"P(Responder accepts | offer = ${offer})",
                outcomes=["Accept"],
                prompt_template=f"""
Suppose you were to offer ${offer} to the Responder.
What probability (0-100%) do you assign to the Responder accepting this offer?

{{incentive_explanation}}

Please highlight your probability in brackets, e.g., [75]%.
""",
                include_incentive_explanation=(offer == 10),
                scoring_rule=ScoringRule.QUADRATIC
            )
        )

    elicitation_point = ElicitationPoint(
        point_id="pre_decision_beliefs",
        at_prompt_index=1,
        questions=acceptance_questions
    )

    return GameBeliefConfig(
        game_name="ultimatum_proposer",
        elicitation_points=[elicitation_point],
        incentivized=incentivized
    )


# =============================================================================
# p-BEAUTY CONTEST
# =============================================================================

def get_beauty_contest_config(
    p: float = 2 / 3,
    n_players: int = 3,
    n_rounds: int = 3,
    incentivized: bool = True
) -> GameBeliefConfig:
    """
    Belief configuration for p-Beauty Contest.

    Tests higher-order beliefs: optimal choice requires beliefs about
    others' beliefs about others' beliefs, etc.

    Nash equilibrium: all choose 0. But depth of reasoning varies.
    """
    elicitation_points = []

    for r in range(1, n_rounds + 1):
        preamble = (
            "Before you choose your number, I'd like to understand "
            "your expectations about the other players."
            if r == 1 else
            f"For round {r}, before you choose:"
        )
        q = BeliefQuestion(
            question_id=f"expected_others_avg_r{r}",
            belief_type=BeliefType.EXPECTATION,
            target_variable="Average of other players' chosen numbers",
            value_range=(0, 100),
            prompt_template=f"""
{preamble}

{{incentive_explanation}}

What do you expect the average of the other {n_players - 1} players' chosen
numbers to be? (From 0 to 100)

Please highlight your answer in brackets, e.g., [50].
""",
            include_incentive_explanation=(r == 1),
            scoring_rule=ScoringRule.QUADRATIC
        )
        elicitation_points.append(ElicitationPoint(
            point_id=f"round_{r}_beliefs",
            at_prompt_index=r,
            questions=[q]
        ))

    return GameBeliefConfig(
        game_name="beauty_contest",
        elicitation_points=elicitation_points,
        incentivized=incentivized
    )


# =============================================================================
# FIRST-PRICE SEALED-BID AUCTION
# =============================================================================

def get_first_price_auction_config(
    n_bidders: int = 2,
    incentivized: bool = True
) -> GameBeliefConfig:
    """
    Belief configuration for First-Price Sealed-Bid Auction.

    Bayesian game: beliefs about opponent's valuation/bid determine
    optimal bid shading. BNE for 2 players with Uniform[0,100]: bid = v/2.
    """
    q = BeliefQuestion(
        question_id="expected_opponent_bid",
        belief_type=BeliefType.EXPECTATION,
        target_variable="Opponent's bid",
        value_range=(0, 100),
        prompt_template="""
Before you submit your bid, I'd like to understand your expectations.

{incentive_explanation}

What do you expect the other bidder's bid to be? (From $0 to $100)

Please highlight your answer in brackets, e.g., [$30].
""",
        include_incentive_explanation=True,
        scoring_rule=ScoringRule.QUADRATIC
    )

    elicitation_point = ElicitationPoint(
        point_id="pre_bid_beliefs",
        at_prompt_index=1,
        questions=[q]
    )

    return GameBeliefConfig(
        game_name="first_price_auction",
        elicitation_points=[elicitation_point],
        incentivized=incentivized
    )


# =============================================================================
# CONFIGURATION REGISTRY
# =============================================================================

GAME_CONFIG_REGISTRY = {
    "prisoners_dilemma": get_prisoners_dilemma_config,
    "stag_hunt": get_stag_hunt_config,
    "ultimatum_proposer": get_ultimatum_proposer_config,
    "beauty_contest": get_beauty_contest_config,
    "first_price_auction": get_first_price_auction_config,
}


def get_config(game_name: str, **kwargs) -> GameBeliefConfig:
    """Get belief configuration for a game by name."""
    if game_name not in GAME_CONFIG_REGISTRY:
        raise KeyError(
            f"Unknown game: {game_name}. "
            f"Available: {list(GAME_CONFIG_REGISTRY.keys())}"
        )
    return GAME_CONFIG_REGISTRY[game_name](**kwargs)


def list_available_games() -> List[str]:
    """Return list of games with available configurations."""
    return list(GAME_CONFIG_REGISTRY.keys())
