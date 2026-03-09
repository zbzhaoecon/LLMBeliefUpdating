"""
Belief Elicitation Framework for LLM Behavioral Economics Experiments

A flexible, game-agnostic framework for eliciting beliefs from LLMs
during economic games using proper scoring rules (PSR).

Key Features:
- PRE-DECISION belief elicitation in forked conversation branches
- Supports probabilities, expectations, and full distributions
- Incentivized (PSR) and direct-ask modes
- Pre-configured for games with strategic uncertainty
- Configurable opponent strategies for multi-round games
"""

__version__ = "0.2.0"

# Core classes and functions
from .core import (
    # Enums
    BeliefType,
    ScoringRule,

    # Configuration classes
    BeliefQuestion,
    ElicitationPoint,
    GameBeliefConfig,

    # Prompt building
    PSRPromptBuilder,
    BDMPromptBuilder,  # backward-compatible alias

    # Extraction
    BeliefExtractor,

    # Core engine
    BeliefElicitationEngine,

    # Convenience functions
    create_simple_probability_question,
    create_expectation_question,
)

# Opponent strategies
from .opponents import (
    OpponentStrategy,
    OpponentStrategyType,
)

# Game configurations
from .game_configs import (
    get_prisoners_dilemma_config,
    get_stag_hunt_config,
    get_ultimatum_proposer_config,
    get_beauty_contest_config,
    get_first_price_auction_config,
    get_config,
    list_available_games,
    GAME_CONFIG_REGISTRY,
)

# Integration with game runners
from .integration import (
    BeliefAwareRunner,
    create_belief_aware_runner,
    run_one_session_with_beliefs,
    run_pd_with_beliefs,
    run_stag_hunt_with_beliefs,
    run_beauty_contest_with_beliefs,
    run_first_price_auction_with_beliefs,
    run_ultimatum_proposer_with_beliefs,
    extract_choices_from_records,
    extract_beliefs_summary,
)

__all__ = [
    # Version
    '__version__',

    # Enums
    'BeliefType',
    'ScoringRule',

    # Configuration
    'BeliefQuestion',
    'ElicitationPoint',
    'GameBeliefConfig',

    # Building and extraction
    'PSRPromptBuilder',
    'BDMPromptBuilder',
    'BeliefExtractor',

    # Core engine
    'BeliefElicitationEngine',

    # Convenience
    'create_simple_probability_question',
    'create_expectation_question',

    # Opponents
    'OpponentStrategy',
    'OpponentStrategyType',

    # Game configs
    'get_prisoners_dilemma_config',
    'get_stag_hunt_config',
    'get_ultimatum_proposer_config',
    'get_beauty_contest_config',
    'get_first_price_auction_config',
    'get_config',
    'list_available_games',
    'GAME_CONFIG_REGISTRY',

    # Integration
    'BeliefAwareRunner',
    'create_belief_aware_runner',
    'run_one_session_with_beliefs',
    'run_pd_with_beliefs',
    'run_stag_hunt_with_beliefs',
    'run_beauty_contest_with_beliefs',
    'run_first_price_auction_with_beliefs',
    'run_ultimatum_proposer_with_beliefs',
    'extract_choices_from_records',
    'extract_beliefs_summary',
]
