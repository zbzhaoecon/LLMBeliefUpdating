"""
Belief Elicitation Framework for LLM Behavioral Economics Experiments

This module provides a flexible, game-agnostic framework for eliciting beliefs
from LLMs during economic games using proper scoring rules (PSR).

Key Design Principles:
1. Belief elicitation occurs in FORKED conversation branches to avoid contamination
2. Supports multiple belief types: probabilities, expectations, distributions
3. Game-agnostic: works with any game via configuration
4. Proper scoring rule framing for incentive compatibility (even for LLMs)

Author: Belief Elicitation Framework
"""

import copy
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    List, Dict, Any, Optional, Callable, Tuple, Union
)
from abc import ABC, abstractmethod


# =============================================================================
# ENUMS AND TYPE DEFINITIONS
# =============================================================================

class BeliefType(Enum):
    """Types of beliefs that can be elicited."""
    PROBABILITY = "probability"           # P(event) ∈ [0,1]
    DISTRIBUTION = "distribution"         # Full distribution over discrete outcomes
    EXPECTATION = "expectation"           # E[X] for some random variable X
    QUANTILE = "quantile"                 # Median or other quantiles
    INTERVAL = "interval"                 # Confidence interval


class ScoringRule(Enum):
    """Proper scoring rules for incentive compatibility."""
    QUADRATIC = "quadratic"     # Brier score: 100 - 100*(p - I)^2
    LOGARITHMIC = "logarithmic" # Log score: log(p) if event, log(1-p) otherwise
    SPHERICAL = "spherical"     # Spherical score


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class BeliefQuestion:
    """
    Represents a single belief question to be asked.
    
    Attributes:
        question_id: Unique identifier for this question
        belief_type: Type of belief being elicited
        target_variable: What the belief is about (e.g., "opponent_action")
        outcomes: For discrete beliefs, list of possible outcomes
        prompt_template: Template string for the question (use {placeholders})
        extraction_hint: Regex pattern or instruction for extraction
        scoring_rule: Which proper scoring rule to use in framing
        include_incentive_explanation: Whether to explain the scoring rule
    """
    question_id: str
    belief_type: BeliefType
    target_variable: str
    outcomes: Optional[List[str]] = None
    prompt_template: str = ""
    extraction_hint: str = ""
    scoring_rule: ScoringRule = ScoringRule.QUADRATIC
    include_incentive_explanation: bool = True
    value_range: Optional[Tuple[float, float]] = None  # For continuous beliefs


@dataclass
class ElicitationPoint:
    """
    Defines when and what beliefs to elicit during a game.

    Beliefs are elicited PRE-DECISION: after the game prompt is shown
    but before the LLM responds with its action.

    Attributes:
        point_id: Identifier for this elicitation point
        at_prompt_index: Elicit beliefs at this prompt index (pre-decision)
        questions: List of BeliefQuestion objects to ask
        condition: Optional callable to determine if elicitation should occur
        context_builder: Optional callable to build context for prompts
    """
    point_id: str
    at_prompt_index: int
    questions: List[BeliefQuestion]
    condition: Optional[Callable[[List[Dict]], bool]] = None
    context_builder: Optional[Callable[[List[Dict]], Dict[str, Any]]] = None


@dataclass
class GameBeliefConfig:
    """
    Complete belief elicitation configuration for a game.

    Attributes:
        game_name: Name of the game
        elicitation_points: List of points where beliefs are elicited
        incentivized: If True, include PSR framing; if False, direct ask
        preamble: Optional preamble before belief questions
        postamble: Optional closing after belief questions
    """
    game_name: str
    elicitation_points: List[ElicitationPoint]
    incentivized: bool = True
    preamble: str = ""
    postamble: str = ""

    def get_elicitation_point(self, prompt_index: int) -> Optional[ElicitationPoint]:
        """Get elicitation point for a given prompt index, if any."""
        for ep in self.elicitation_points:
            if ep.at_prompt_index == prompt_index:
                return ep
        return None


# =============================================================================
# BDM PROMPT TEMPLATES
# =============================================================================

class PSRPromptBuilder:
    """
    Builds Proper Scoring Rule (PSR) prompts for belief elicitation.

    Uses proper scoring rules (quadratic/Brier, logarithmic, spherical)
    to frame questions in an incentive-compatible way.
    """
    
    INCENTIVE_EXPLANATIONS = {
        ScoringRule.QUADRATIC: """
Your payment depends on accuracy via a quadratic scoring rule:
- You earn 100 - 100×(p_reported - I_realized)² points
where I_realized = 1 if the event occurs, 0 otherwise.
Reporting your true belief maximizes your expected score.""",
        
        ScoringRule.LOGARITHMIC: """
Your payment depends on accuracy via a logarithmic scoring rule:
- You earn log(p_reported) if the event occurs
- You earn log(1 - p_reported) if the event does not occur
Reporting your true belief maximizes your expected score.""",

        ScoringRule.SPHERICAL: """
Your payment depends on accuracy via a spherical scoring rule.
Reporting your true belief maximizes your expected score."""
    }
    
    PROBABILITY_TEMPLATE = """
{preamble}

I would like to understand your belief about: {target_description}

{incentive_explanation}

What probability (0-100%) do you assign to the following:
{outcome_list}

Please ensure your probabilities sum to 100%.
Format your answer by highlighting each probability in brackets, e.g., [50]%.
"""

    EXPECTATION_TEMPLATE = """
{preamble}

I would like to understand your expectations about: {target_description}

{incentive_explanation}

What is your expected value for {target_variable}?
(The value should be between {min_val} and {max_val})

Please highlight your answer in brackets, e.g., [{example_val}].
"""

    DISTRIBUTION_TEMPLATE = """
{preamble}

I would like to understand your full belief distribution over: {target_description}

{incentive_explanation}

For each possible outcome, please assign a probability (0-100%):
{outcome_list}

Ensure probabilities sum to 100%. 
Format: [Outcome]: [probability]%
"""

    @classmethod
    def build_prompt(
        cls,
        question: BeliefQuestion,
        context: Optional[Dict[str, Any]] = None,
        incentivized_override: Optional[bool] = None
    ) -> str:
        """
        Build a PSR prompt for a belief question.

        Args:
            question: The BeliefQuestion to build a prompt for
            context: Optional context dictionary for template substitution
            incentivized_override: If set, overrides question.include_incentive_explanation

        Returns:
            Complete prompt string
        """
        context = context or {}

        # Determine whether to include incentive explanation
        include_incentive = (
            incentivized_override
            if incentivized_override is not None
            else question.include_incentive_explanation
        )

        incentive_explanation = ""
        if include_incentive:
            incentive_explanation = cls.INCENTIVE_EXPLANATIONS.get(
                question.scoring_rule, ""
            )
        
        # Use custom template if provided
        if question.prompt_template:
            template = question.prompt_template
        else:
            # Select default template based on belief type
            template = cls._get_default_template(question.belief_type)
        
        # Build outcome list for discrete beliefs
        outcome_list = ""
        if question.outcomes:
            outcome_list = "\n".join(
                f"- {outcome}: ___%" for outcome in question.outcomes
            )
        
        # Build the prompt - filter out keys that conflict with template params
        safe_context = {k: v for k, v in context.items() 
                        if k not in ['preamble', 'target_description', 'target_variable',
                                     'incentive_explanation', 'outcome_list', 
                                     'min_val', 'max_val', 'example_val']}
        prompt = template.format(
            preamble=context.get("preamble", ""),
            target_description=context.get(
                "target_description", 
                question.target_variable
            ),
            target_variable=question.target_variable,
            incentive_explanation=incentive_explanation,
            outcome_list=outcome_list,
            min_val=question.value_range[0] if question.value_range else 0,
            max_val=question.value_range[1] if question.value_range else 100,
            example_val=context.get("example_val", 50),
            **safe_context
        )
        
        return prompt.strip()
    
    @classmethod
    def _get_default_template(cls, belief_type: BeliefType) -> str:
        """Get default template for a belief type."""
        templates = {
            BeliefType.PROBABILITY: cls.PROBABILITY_TEMPLATE,
            BeliefType.DISTRIBUTION: cls.DISTRIBUTION_TEMPLATE,
            BeliefType.EXPECTATION: cls.EXPECTATION_TEMPLATE,
        }
        return templates.get(belief_type, cls.PROBABILITY_TEMPLATE)


# Backward-compatible alias
BDMPromptBuilder = PSRPromptBuilder


# =============================================================================
# BELIEF EXTRACTION
# =============================================================================

class BeliefExtractor:
    """
    Extracts belief responses from LLM outputs.
    
    Supports multiple extraction strategies:
    1. Regex-based extraction for well-formatted responses
    2. GPT-based extraction for complex responses (fallback)
    """
    
    @staticmethod
    def extract_probability(
        message: str,
        outcomes: List[str],
        normalize: bool = True
    ) -> Optional[Dict[str, float]]:
        """
        Extract probability distribution from response.
        
        Args:
            message: LLM response text
            outcomes: List of outcome labels to look for
            normalize: Whether to normalize to sum to 1
            
        Returns:
            Dictionary mapping outcomes to probabilities, or None if extraction fails
        """
        probs = {}
        
        for outcome in outcomes:
            # Try multiple patterns
            patterns = [
                # [Outcome]: [50]% or [Outcome]: [50]
                rf'\[?{re.escape(outcome)}\]?\s*:\s*\[(\d+(?:\.\d+)?)\]%?',
                # Outcome: 50% or [50]%
                rf'{re.escape(outcome)}\s*:\s*\[?(\d+(?:\.\d+)?)\]?%',
                # [50]% for Outcome
                rf'\[(\d+(?:\.\d+)?)\]%?\s*(?:for|to)\s*\[?{re.escape(outcome)}\]?',
                # Just [Outcome] with number nearby
                rf'\[{re.escape(outcome)}\][^\d]*(\d+(?:\.\d+)?)%?',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    probs[outcome] = float(match.group(1))
                    break
        
        if not probs:
            return None
            
        # Convert percentages to probabilities if needed
        if any(v > 1 for v in probs.values()):
            probs = {k: v / 100.0 for k, v in probs.items()}
        
        # Validate and optionally normalize
        total = sum(probs.values())
        if normalize and len(probs) == len(outcomes):
            if abs(total - 1.0) > 0.15:  # Allow some tolerance
                # Normalize
                probs = {k: v / total for k, v in probs.items()}
            return probs
        elif abs(total - 1.0) < 0.15:
            return probs
        
        return None if len(probs) != len(outcomes) else probs
    
    @staticmethod
    def extract_expectation(
        message: str,
        value_range: Optional[Tuple[float, float]] = None
    ) -> Optional[float]:
        """
        Extract expected value from response.
        
        Args:
            message: LLM response text
            value_range: Optional (min, max) for validation
            
        Returns:
            Extracted expectation value, or None if extraction fails
        """
        # Try patterns for bracketed numbers
        patterns = [
            r'\[(\$?\d+(?:\.\d+)?)\]',
            r'\[\$?(\d+(?:\.\d+)?)\]',
            r'expect(?:ed|ation)?[^\d]*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                value_str = match.group(1).replace('$', '')
                value = float(value_str)
                
                # Validate range if provided
                if value_range:
                    if value_range[0] <= value <= value_range[1]:
                        return value
                else:
                    return value
        
        return None
    
    @staticmethod
    def extract_single_probability(message: str) -> Optional[float]:
        """
        Extract a single probability value from response.
        
        Returns:
            Probability as float in [0,1], or None if extraction fails
        """
        patterns = [
            r'\[(\d+(?:\.\d+)?)\]%',
            r'\[(\d+(?:\.\d+)?)\]',
            r'(\d+(?:\.\d+)?)%',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                value = float(match.group(1))
                if value > 1:
                    value /= 100.0
                if 0 <= value <= 1:
                    return value
        
        return None


# =============================================================================
# CORE BRANCHING LOGIC
# =============================================================================

class BeliefElicitationEngine:
    """
    Core engine for running belief elicitation in forked conversation branches.
    
    This class manages the forking process and coordinates prompt building
    and response extraction.
    """
    
    def __init__(
        self,
        get_response_func: Callable,
        model: str = None,
        time_limit: int = 60,
        print_exceptions: bool = True
    ):
        """
        Initialize the belief elicitation engine.
        
        Args:
            get_response_func: Function to call the LLM API
            model: Model identifier
            time_limit: Timeout for API calls
            print_exceptions: Whether to print exceptions
        """
        self.get_response = get_response_func
        self.model = model
        self.time_limit = time_limit
        self.print_exceptions = print_exceptions
        self.prompt_builder = PSRPromptBuilder()
        self.extractor = BeliefExtractor()
    
    def fork_and_elicit(
        self,
        messages: List[Dict],
        elicitation_point: ElicitationPoint,
        context: Optional[Dict[str, Any]] = None,
        incentivized: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Fork the conversation and elicit beliefs in the forked branch.
        
        This is the core method that ensures belief elicitation doesn't
        contaminate the main conversation.
        
        Args:
            messages: Current conversation history (will NOT be modified)
            elicitation_point: Configuration for what beliefs to elicit
            context: Optional context for prompt building
            
        Returns:
            Dictionary containing:
                - 'point_id': Identifier for this elicitation point
                - 'beliefs': Dict mapping question_id to extracted beliefs
                - 'raw_responses': Raw LLM responses
                - 'branch_messages': Full forked conversation
                - 'success': Whether all extractions succeeded
        """
        # CRITICAL: Deep copy to create isolated branch
        branch_messages = copy.deepcopy(messages)
        
        # Build context if builder provided
        if elicitation_point.context_builder:
            context = elicitation_point.context_builder(messages)
        context = context or {}
        
        # Check condition if provided
        if elicitation_point.condition:
            if not elicitation_point.condition(messages):
                return {
                    'point_id': elicitation_point.point_id,
                    'beliefs': {},
                    'raw_responses': [],
                    'branch_messages': branch_messages,
                    'success': True,
                    'skipped': True
                }
        
        beliefs = {}
        raw_responses = []
        all_success = True
        
        for question in elicitation_point.questions:
            try:
                # Build the prompt
                prompt = self.prompt_builder.build_prompt(
                    question, context, incentivized_override=incentivized
                )
                
                # Add to branch and get response
                branch_messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                response = self.get_response(
                    branch_messages,
                    model=self.model
                )
                
                response_content = response["choices"][0]["message"]["content"]
                raw_responses.append({
                    'question_id': question.question_id,
                    'prompt': prompt,
                    'response': response_content,
                    'full_response': response
                })
                
                branch_messages.append({
                    "role": "assistant",
                    "content": response_content
                })
                
                # Extract belief based on type
                extracted = self._extract_belief(question, response_content)
                beliefs[question.question_id] = extracted
                
                if extracted is None:
                    all_success = False
                    
            except Exception as e:
                if self.print_exceptions:
                    print(f"Belief elicitation error for {question.question_id}: {e}")
                beliefs[question.question_id] = None
                all_success = False
        
        return {
            'point_id': elicitation_point.point_id,
            'beliefs': beliefs,
            'raw_responses': raw_responses,
            'branch_messages': branch_messages,
            'success': all_success,
            'skipped': False
        }
    
    def _extract_belief(
        self,
        question: BeliefQuestion,
        response: str
    ) -> Optional[Any]:
        """Extract belief from response based on question type."""
        
        if question.belief_type == BeliefType.PROBABILITY:
            if question.outcomes and len(question.outcomes) == 1:
                return self.extractor.extract_single_probability(response)
            elif question.outcomes:
                return self.extractor.extract_probability(
                    response, question.outcomes
                )
            else:
                return self.extractor.extract_single_probability(response)
                
        elif question.belief_type == BeliefType.DISTRIBUTION:
            return self.extractor.extract_probability(
                response, question.outcomes or []
            )
            
        elif question.belief_type == BeliefType.EXPECTATION:
            return self.extractor.extract_expectation(
                response, question.value_range
            )

        return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_simple_probability_question(
    question_id: str,
    target: str,
    outcomes: List[str],
    description: str = ""
) -> BeliefQuestion:
    """
    Convenience function to create a simple probability distribution question.
    
    Args:
        question_id: Unique identifier
        target: What the belief is about
        outcomes: List of possible outcomes
        description: Human-readable description
        
    Returns:
        Configured BeliefQuestion
    """
    return BeliefQuestion(
        question_id=question_id,
        belief_type=BeliefType.DISTRIBUTION,
        target_variable=target,
        outcomes=outcomes,
        include_incentive_explanation=True,
        scoring_rule=ScoringRule.QUADRATIC
    )


def create_expectation_question(
    question_id: str,
    target: str,
    value_range: Tuple[float, float],
    description: str = ""
) -> BeliefQuestion:
    """
    Convenience function to create an expectation question.
    
    Args:
        question_id: Unique identifier
        target: What the belief is about
        value_range: (min, max) for the expected value
        description: Human-readable description
        
    Returns:
        Configured BeliefQuestion
    """
    return BeliefQuestion(
        question_id=question_id,
        belief_type=BeliefType.EXPECTATION,
        target_variable=target,
        value_range=value_range,
        include_incentive_explanation=True,
        scoring_rule=ScoringRule.QUADRATIC
    )
