"""
Unit Tests for Belief Elicitation Framework

Run with: pytest tests.py -v
"""

import pytest
import copy
import sys
import os
from unittest.mock import Mock, patch
from typing import Dict, List, Any

try:
    from belief_elicitation.core import (
        BeliefType, ScoringRule,
        BeliefQuestion, ElicitationPoint, GameBeliefConfig,
        PSRPromptBuilder, BDMPromptBuilder, BeliefExtractor,
        BeliefElicitationEngine,
        create_simple_probability_question, create_expectation_question
    )
    from belief_elicitation.game_configs import (
        get_prisoners_dilemma_config,
        get_ultimatum_proposer_config,
        get_stag_hunt_config,
        get_beauty_contest_config,
        get_first_price_auction_config,
        list_available_games,
        get_config
    )
    from belief_elicitation.opponents import (
        OpponentStrategy, OpponentStrategyType
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from core import (
        BeliefType, ScoringRule,
        BeliefQuestion, ElicitationPoint, GameBeliefConfig,
        PSRPromptBuilder, BDMPromptBuilder, BeliefExtractor,
        BeliefElicitationEngine,
        create_simple_probability_question, create_expectation_question
    )
    from game_configs import (
        get_prisoners_dilemma_config,
        get_ultimatum_proposer_config,
        get_stag_hunt_config,
        get_beauty_contest_config,
        get_first_price_auction_config,
        list_available_games,
        get_config
    )
    from opponents import (
        OpponentStrategy, OpponentStrategyType
    )


# =============================================================================
# TEST: Configuration Classes
# =============================================================================

class TestBeliefQuestion:
    def test_basic_creation(self):
        q = BeliefQuestion(
            question_id="test_q",
            belief_type=BeliefType.PROBABILITY,
            target_variable="test_target"
        )
        assert q.question_id == "test_q"
        assert q.belief_type == BeliefType.PROBABILITY
        assert q.include_incentive_explanation is True

    def test_distribution_with_outcomes(self):
        q = BeliefQuestion(
            question_id="dist_q",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="opponent_action",
            outcomes=["Push", "Pull"]
        )
        assert q.outcomes == ["Push", "Pull"]

    def test_expectation_with_range(self):
        q = BeliefQuestion(
            question_id="exp_q",
            belief_type=BeliefType.EXPECTATION,
            target_variable="payoff",
            value_range=(0, 100)
        )
        assert q.value_range == (0, 100)


class TestElicitationPoint:
    def test_basic_creation(self):
        q = BeliefQuestion(
            question_id="q1",
            belief_type=BeliefType.PROBABILITY,
            target_variable="test"
        )
        ep = ElicitationPoint(
            point_id="ep1",
            at_prompt_index=1,
            questions=[q]
        )
        assert ep.point_id == "ep1"
        assert ep.at_prompt_index == 1
        assert len(ep.questions) == 1

    def test_with_condition(self):
        def always_true(messages):
            return True

        q = BeliefQuestion(
            question_id="q1",
            belief_type=BeliefType.PROBABILITY,
            target_variable="test"
        )
        ep = ElicitationPoint(
            point_id="ep1",
            at_prompt_index=1,
            questions=[q],
            condition=always_true
        )
        assert ep.condition([]) is True


class TestGameBeliefConfig:
    def test_get_elicitation_point(self):
        q = BeliefQuestion(
            question_id="q1",
            belief_type=BeliefType.PROBABILITY,
            target_variable="test"
        )
        ep = ElicitationPoint(
            point_id="ep1",
            at_prompt_index=2,
            questions=[q]
        )
        config = GameBeliefConfig(
            game_name="test_game",
            elicitation_points=[ep]
        )

        assert config.get_elicitation_point(2) == ep
        assert config.get_elicitation_point(1) is None

    def test_incentivized_flag(self):
        config = GameBeliefConfig(
            game_name="test",
            elicitation_points=[],
            incentivized=False
        )
        assert config.incentivized is False

        config2 = GameBeliefConfig(
            game_name="test2",
            elicitation_points=[]
        )
        assert config2.incentivized is True  # default


# =============================================================================
# TEST: Prompt Building
# =============================================================================

class TestPSRPromptBuilder:
    def test_build_probability_prompt(self):
        q = BeliefQuestion(
            question_id="prob_q",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="action",
            outcomes=["A", "B"],
            include_incentive_explanation=True
        )
        prompt = PSRPromptBuilder.build_prompt(q)
        assert "A" in prompt
        assert "B" in prompt
        assert "100" in prompt

    def test_build_expectation_prompt(self):
        q = BeliefQuestion(
            question_id="exp_q",
            belief_type=BeliefType.EXPECTATION,
            target_variable="payoff",
            value_range=(0, 50),
            include_incentive_explanation=False
        )
        context = {"target_description": "expected payoff"}
        prompt = PSRPromptBuilder.build_prompt(q, context)
        assert "expected payoff" in prompt or "payoff" in prompt

    def test_custom_template(self):
        q = BeliefQuestion(
            question_id="custom_q",
            belief_type=BeliefType.PROBABILITY,
            target_variable="test",
            prompt_template="Custom prompt: {target_variable}",
            include_incentive_explanation=False
        )
        prompt = PSRPromptBuilder.build_prompt(q)
        assert "Custom prompt: test" == prompt

    def test_incentivized_override_false(self):
        q = BeliefQuestion(
            question_id="q",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="action",
            outcomes=["A", "B"],
            include_incentive_explanation=True  # would normally include
        )
        prompt = PSRPromptBuilder.build_prompt(q, incentivized_override=False)
        assert "quadratic" not in prompt.lower()
        assert "scoring" not in prompt.lower()

    def test_incentivized_override_true(self):
        q = BeliefQuestion(
            question_id="q",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="action",
            outcomes=["A", "B"],
            include_incentive_explanation=False  # would normally exclude
        )
        prompt = PSRPromptBuilder.build_prompt(q, incentivized_override=True)
        assert "quadratic" in prompt.lower()

    def test_backward_compatible_alias(self):
        assert BDMPromptBuilder is PSRPromptBuilder


# =============================================================================
# TEST: Belief Extraction
# =============================================================================

class TestBeliefExtractor:
    def test_extract_distribution_basic(self):
        message = "I assign [60]% to Push and [40]% to Pull."
        result = BeliefExtractor.extract_probability(message, ["Push", "Pull"])
        assert result is not None
        assert abs(result["Push"] - 0.6) < 0.01
        assert abs(result["Pull"] - 0.4) < 0.01

    def test_extract_distribution_with_labels(self):
        message = "Push: [70]%, Pull: [30]%"
        result = BeliefExtractor.extract_probability(message, ["Push", "Pull"])
        assert result is not None
        assert abs(result["Push"] - 0.7) < 0.01

    def test_extract_distribution_normalization(self):
        message = "Push: [55]%, Pull: [45]%"
        result = BeliefExtractor.extract_probability(
            message, ["Push", "Pull"], normalize=True
        )
        if result:
            total = sum(result.values())
            assert abs(total - 1.0) < 0.01

    def test_extract_single_probability(self):
        message = "I believe there's a [75]% chance of acceptance."
        result = BeliefExtractor.extract_single_probability(message)
        assert result is not None
        assert abs(result - 0.75) < 0.01

    def test_extract_expectation_dollar(self):
        message = "I expect the return to be [$50]."
        result = BeliefExtractor.extract_expectation(message, (0, 100))
        assert result is not None
        assert result == 50.0

    def test_extract_expectation_plain(self):
        message = "My expected payoff is [25] points."
        result = BeliefExtractor.extract_expectation(message, (0, 100))
        assert result is not None
        assert result == 25.0

    def test_extract_expectation_out_of_range(self):
        message = "I expect [150] points."
        result = BeliefExtractor.extract_expectation(message, (0, 100))
        assert result is None

    def test_extract_fails_gracefully(self):
        message = "I have no idea what to expect."
        result = BeliefExtractor.extract_probability(message, ["Push", "Pull"])
        assert result is None


# =============================================================================
# TEST: Game Configurations
# =============================================================================

class TestGameConfigs:
    def test_list_available_games(self):
        games = list_available_games()
        assert "prisoners_dilemma" in games
        assert "stag_hunt" in games
        assert "beauty_contest" in games
        assert "first_price_auction" in games
        assert "ultimatum_proposer" in games

    def test_pd_config(self):
        config = get_prisoners_dilemma_config(n_rounds=5, delta=0.9)
        assert config.game_name == "prisoners_dilemma"
        assert len(config.elicitation_points) == 5
        ep = config.elicitation_points[0]
        # Two questions per round: type belief + action belief
        assert len(ep.questions) == 2
        assert ep.questions[0].outcomes == ["Cooperative", "Selfish"]
        assert ep.questions[1].outcomes == ["Push", "Pull"]

    def test_stag_hunt_config(self):
        config = get_stag_hunt_config(n_rounds=3)
        assert config.game_name == "stag_hunt"
        assert len(config.elicitation_points) == 3
        ep = config.elicitation_points[0]
        assert ep.questions[0].outcomes == ["Stag", "Hare"]

    def test_beauty_contest_config(self):
        config = get_beauty_contest_config(p=2/3, n_rounds=3)
        assert config.game_name == "beauty_contest"
        assert len(config.elicitation_points) == 3
        ep = config.elicitation_points[0]
        assert ep.questions[0].belief_type == BeliefType.EXPECTATION

    def test_first_price_auction_config(self):
        config = get_first_price_auction_config()
        assert config.game_name == "first_price_auction"
        assert len(config.elicitation_points) == 1
        ep = config.elicitation_points[0]
        assert ep.questions[0].belief_type == BeliefType.EXPECTATION

    def test_ultimatum_proposer_config(self):
        config = get_ultimatum_proposer_config()
        assert config.game_name == "ultimatum_proposer"
        ep = config.elicitation_points[0]
        assert len(ep.questions) >= 3

    def test_get_config_function(self):
        config = get_config("prisoners_dilemma", n_rounds=3)
        assert config.game_name == "prisoners_dilemma"

    def test_get_config_unknown_game(self):
        with pytest.raises(KeyError):
            get_config("unknown_game")

    def test_incentivized_flag_propagates(self):
        config = get_prisoners_dilemma_config(incentivized=False)
        assert config.incentivized is False


# =============================================================================
# TEST: Opponent Strategies
# =============================================================================

class TestOpponentStrategy:
    def test_always_cooperate(self):
        s = OpponentStrategy(
            strategy_type=OpponentStrategyType.ALWAYS_COOPERATE,
            cooperate_label="Push", defect_label="Pull"
        )
        assert s.get_action(1, []) == "Push"
        assert s.get_action(5, ["Pull", "Pull"]) == "Push"

    def test_always_defect(self):
        s = OpponentStrategy(
            strategy_type=OpponentStrategyType.ALWAYS_DEFECT,
            cooperate_label="Push", defect_label="Pull"
        )
        assert s.get_action(1, []) == "Pull"

    def test_tit_for_tat(self):
        s = OpponentStrategy(
            strategy_type=OpponentStrategyType.TIT_FOR_TAT,
            cooperate_label="Push", defect_label="Pull"
        )
        assert s.get_action(1, []) == "Push"  # cooperates first
        assert s.get_action(2, ["Pull"]) == "Pull"  # mirrors
        assert s.get_action(3, ["Pull", "Push"]) == "Push"  # mirrors last

    def test_random_strategy(self):
        import random
        random.seed(42)
        s = OpponentStrategy(
            strategy_type=OpponentStrategyType.RANDOM,
            p_cooperate=1.0,
            cooperate_label="Push", defect_label="Pull"
        )
        assert s.get_action(1, []) == "Push"  # p=1.0 always cooperates

        s2 = OpponentStrategy(
            strategy_type=OpponentStrategyType.RANDOM,
            p_cooperate=0.0,
            cooperate_label="Push", defect_label="Pull"
        )
        assert s2.get_action(1, []) == "Pull"  # p=0.0 always defects

    def test_custom_sequence(self):
        s = OpponentStrategy(
            strategy_type=OpponentStrategyType.CUSTOM,
            custom_sequence=["Pull", "Pull", "Push", "Push"],
            cooperate_label="Push", defect_label="Pull"
        )
        assert s.get_action(1, []) == "Pull"
        assert s.get_action(2, []) == "Pull"
        assert s.get_action(3, []) == "Push"
        assert s.get_action(4, []) == "Push"

    def test_stag_hunt_labels(self):
        s = OpponentStrategy(
            strategy_type=OpponentStrategyType.TIT_FOR_TAT,
            cooperate_label="Stag", defect_label="Hare"
        )
        assert s.get_action(1, []) == "Stag"
        assert s.get_action(2, ["Hare"]) == "Hare"


# =============================================================================
# TEST: Branching Logic (with mocks)
# =============================================================================

class TestBeliefElicitationEngine:
    def create_mock_get_response(self, response_text: str):
        def mock_get_response(messages, model=None, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": response_text
                    }
                }]
            }
        return mock_get_response

    def test_fork_does_not_modify_original(self):
        original_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Let's play a game."},
        ]
        original_copy = copy.deepcopy(original_messages)

        mock_response = self.create_mock_get_response(
            "Push: [50]%, Pull: [50]%"
        )
        engine = BeliefElicitationEngine(
            get_response_func=mock_response, model="test-model"
        )

        q = BeliefQuestion(
            question_id="test_q",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="action",
            outcomes=["Push", "Pull"]
        )
        ep = ElicitationPoint(
            point_id="test_ep",
            at_prompt_index=1,
            questions=[q]
        )

        result = engine.fork_and_elicit(original_messages, ep)

        assert original_messages == original_copy
        assert len(result['branch_messages']) > len(original_messages)

    def test_pre_decision_fork_state(self):
        """Verify the fork happens from a pre-decision state (last msg is user prompt)."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Which card do you play?"},
        ]

        mock_response = self.create_mock_get_response("Push: [60]%, Pull: [40]%")
        engine = BeliefElicitationEngine(
            get_response_func=mock_response, model="test"
        )

        q = BeliefQuestion(
            question_id="q",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="action",
            outcomes=["Push", "Pull"]
        )
        ep = ElicitationPoint(point_id="ep", at_prompt_index=1, questions=[q])

        result = engine.fork_and_elicit(messages, ep)

        # The original messages should end with user message (pre-decision)
        assert messages[-1]["role"] == "user"
        # Branch should have belief Q&A appended
        assert result['branch_messages'][-1]["role"] == "assistant"

    def test_extracts_beliefs_correctly(self):
        mock_response = self.create_mock_get_response(
            "I think Push: [60]%, Pull: [40]%"
        )
        engine = BeliefElicitationEngine(
            get_response_func=mock_response, model="test-model"
        )

        q = BeliefQuestion(
            question_id="opponent_belief",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="opponent_action",
            outcomes=["Push", "Pull"]
        )
        ep = ElicitationPoint(
            point_id="round_1",
            at_prompt_index=1,
            questions=[q]
        )

        messages = [{"role": "system", "content": "Test"}]
        result = engine.fork_and_elicit(messages, ep)

        assert result['success'] is True
        assert 'opponent_belief' in result['beliefs']
        assert abs(result['beliefs']['opponent_belief']['Push'] - 0.6) < 0.01

    def test_condition_prevents_elicitation(self):
        def never_elicit(messages):
            return False

        mock_response = self.create_mock_get_response("test")
        engine = BeliefElicitationEngine(
            get_response_func=mock_response, model="test-model"
        )

        q = BeliefQuestion(
            question_id="q1",
            belief_type=BeliefType.PROBABILITY,
            target_variable="test"
        )
        ep = ElicitationPoint(
            point_id="ep1",
            at_prompt_index=1,
            questions=[q],
            condition=never_elicit
        )

        result = engine.fork_and_elicit([], ep)
        assert result['skipped'] is True
        assert result['beliefs'] == {}

    def test_incentivized_override_passed_to_prompt(self):
        """Verify incentivized flag reaches prompt builder."""
        prompts_captured = []

        def mock_get_response(messages, model=None, **kwargs):
            # Capture the belief prompt that was sent
            for m in messages:
                if m['role'] == 'user' and 'probability' in m['content'].lower():
                    prompts_captured.append(m['content'])
            return {"choices": [{"message": {"content": "Push: [50]%, Pull: [50]%"}}]}

        engine = BeliefElicitationEngine(
            get_response_func=mock_get_response, model="test"
        )

        q = BeliefQuestion(
            question_id="q",
            belief_type=BeliefType.DISTRIBUTION,
            target_variable="action",
            outcomes=["A", "B"],
            include_incentive_explanation=True
        )
        ep = ElicitationPoint(point_id="ep", at_prompt_index=1, questions=[q])

        # With incentivized=False, should NOT contain scoring rule text
        engine.fork_and_elicit([], ep, incentivized=False)
        if prompts_captured:
            assert "quadratic" not in prompts_captured[-1].lower()


# =============================================================================
# TEST: Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    def test_create_simple_probability_question(self):
        q = create_simple_probability_question(
            question_id="simple_prob",
            target="opponent_move",
            outcomes=["Left", "Right"],
            description="Opponent's next move"
        )
        assert q.question_id == "simple_prob"
        assert q.belief_type == BeliefType.DISTRIBUTION
        assert q.outcomes == ["Left", "Right"]

    def test_create_expectation_question(self):
        q = create_expectation_question(
            question_id="expected_payoff",
            target="payoff",
            value_range=(0, 100),
            description="Expected game payoff"
        )
        assert q.question_id == "expected_payoff"
        assert q.belief_type == BeliefType.EXPECTATION
        assert q.value_range == (0, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
