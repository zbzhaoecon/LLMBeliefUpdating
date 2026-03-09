# Belief Elicitation Framework for LLM Behavioral Economics

A flexible, game-agnostic framework for eliciting beliefs from Large Language Models during economic games, using proper scoring rules (BDM mechanism) to ensure incentive compatibility.

## Key Features

- **Non-contaminating belief elicitation**: Beliefs are elicited in forked conversation branches, leaving the main game conversation unaffected
- **Proper scoring rules**: Uses quadratic, logarithmic, or spherical scoring rules for incentive-compatible elicitation
- **Pre-configured games**: Ready-to-use configurations for Ultimatum, Dictator, Trust, Public Goods, Prisoner's Dilemma, and Bomb Risk games
- **Extensible design**: Easy to add new games via configuration
- **Type-safe**: Full type hints and dataclass-based configuration

## Installation

```bash
# Clone or copy the belief_elicitation directory to your project
# No external dependencies beyond what's in the original notebook
```

## Quick Start

```python
import openai
from belief_elicitation import (
    create_belief_aware_runner,
    run_pd_with_beliefs,
    get_prisoners_dilemma_config,
    extract_beliefs_summary
)

# Setup client
client = openai.OpenAI(api_key="your-key", base_url="your-url")

# Create belief-aware runner
runner, get_response, update_messages = create_belief_aware_runner(
    client=client,
    model='gpt-4'
)

# Run Prisoner's Dilemma with belief elicitation
records = run_pd_with_beliefs(
    runner=runner,
    update_messages_func=update_messages,
    get_response_func=get_response,
    n_instances=30
)

# Analyze beliefs
summary = extract_beliefs_summary(records)
print(summary)
```

## Architecture

### The Branching Principle

The key insight is that belief elicitation must occur in a **forked conversation branch** to avoid contaminating the agent's subsequent behavior:

```
Main Game Timeline:
[System] → [Prompt 1] → [Action 1] → [Prompt 2] → [Action 2] → ...
                             │                          │
                        (fork)                     (fork)
                             ↓                          ↓
Belief Branch 1:      [Belief Q1] → [Answer]    [Belief Q1] → [Answer]
(deep copy,           [Belief Q2] → [Answer]    [Belief Q2] → [Answer]
 discarded after      (extracted, discarded)    (extracted, discarded)
 extraction)
```

The main conversation continues without any trace of the belief elicitation.

### Module Structure

```
belief_elicitation/
├── __init__.py          # Package exports
├── core.py              # Core classes: BeliefQuestion, Engine, etc.
├── game_configs.py      # Pre-built game configurations
├── integration.py       # Integration with original notebook
├── tests.py             # Unit tests
└── demo_notebook.ipynb  # Usage examples
```

## Configuration Classes

### BeliefQuestion

Defines a single belief to elicit:

```python
from belief_elicitation import BeliefQuestion, BeliefType, ScoringRule

q = BeliefQuestion(
    question_id="opponent_action",
    belief_type=BeliefType.DISTRIBUTION,      # or PROBABILITY, EXPECTATION
    target_variable="Opponent's action",
    outcomes=["Cooperate", "Defect"],         # For discrete beliefs
    value_range=(0, 100),                     # For continuous beliefs
    scoring_rule=ScoringRule.QUADRATIC,       # Proper scoring rule
    include_incentive_explanation=True,       # Explain the scoring rule?
    prompt_template="..."                     # Custom prompt (optional)
)
```

### ElicitationPoint

Defines when to elicit beliefs:

```python
from belief_elicitation import ElicitationPoint

ep = ElicitationPoint(
    point_id="round_1_beliefs",
    after_prompt_index=1,     # Elicit after prompt at index 1
    questions=[q1, q2],       # List of BeliefQuestions
    condition=lambda msgs: len(msgs) > 2,  # Optional condition
    context_builder=lambda msgs: {...}     # Optional context
)
```

### GameBeliefConfig

Complete configuration for a game:

```python
from belief_elicitation import GameBeliefConfig

config = GameBeliefConfig(
    game_name="my_game",
    elicitation_points=[ep1, ep2, ep3],
    preamble="Before we continue...",
    postamble=""
)
```

## Pre-Built Game Configurations

| Game | Config Function | Beliefs Elicited |
|------|-----------------|------------------|
| Ultimatum (Proposer) | `get_ultimatum_proposer_config()` | P(Accept \| offer=$x) for various x |
| Ultimatum (Responder) | `get_ultimatum_responder_config()` | E[Offer], P(Offer ≥ $x) |
| Dictator | `get_dictator_config()` | E[What others would give] |
| Trust (Investor) | `get_trust_investor_config()` | E[Return rate], E[Return \| invest=$x] |
| Trust (Banker) | `get_trust_banker_config()` | Fairness beliefs |
| Public Goods | `get_public_goods_config(n_rounds)` | E[Others' contribution] per round |
| Prisoner's Dilemma | `get_prisoners_dilemma_config(n_rounds)` | P(Push), P(Pull) per round |
| Bomb Risk | `get_bomb_risk_config()` | Understanding check, risk perception |

## Belief Types

- **PROBABILITY**: Single probability P(event) ∈ [0,1]
- **DISTRIBUTION**: Full distribution over discrete outcomes (must sum to 1)
- **EXPECTATION**: Expected value E[X] within a range
- **QUANTILE**: Median or other quantiles (future)
- **INTERVAL**: Confidence intervals (future)

## Scoring Rules

Proper scoring rules incentivize truthful belief reporting:

- **QUADRATIC (Brier)**: Score = 100 - 100×(p - I)²
- **LOGARITHMIC**: Score = log(p) if event, log(1-p) otherwise
- **SPHERICAL**: Normalized quadratic score

## Extraction

The framework extracts beliefs from LLM responses using pattern matching:

```python
from belief_elicitation import BeliefExtractor

# Extract probability distribution
result = BeliefExtractor.extract_probability(
    "Push: [60]%, Pull: [40]%",
    outcomes=["Push", "Pull"]
)
# Returns: {"Push": 0.6, "Pull": 0.4}

# Extract expectation
value = BeliefExtractor.extract_expectation(
    "I expect [$50] return.",
    value_range=(0, 100)
)
# Returns: 50.0
```

## Creating Custom Games

```python
from belief_elicitation import (
    BeliefQuestion, BeliefType, ScoringRule,
    ElicitationPoint, GameBeliefConfig,
    run_one_session_with_beliefs
)

# 1. Define belief questions
q1 = BeliefQuestion(
    question_id="my_belief",
    belief_type=BeliefType.EXPECTATION,
    target_variable="payoff",
    value_range=(0, 100)
)

# 2. Define elicitation points
ep = ElicitationPoint(
    point_id="pre_decision",
    after_prompt_index=1,
    questions=[q1]
)

# 3. Create game config
config = GameBeliefConfig(
    game_name="my_custom_game",
    elicitation_points=[ep]
)

# 4. Run with beliefs
records = run_one_session_with_beliefs(
    prompts=my_prompts,
    n_instances=30,
    belief_config=config,
    runner=runner,
    update_messages_func=update_messages
)
```

## Output Format

```python
records = {
    'messages': [...],      # Main conversation histories
    'responses': [...],     # Raw API responses
    'beliefs': [            # Belief elicitation results per instance
        [                   # Instance 1
            {
                'point_id': 'round_1_beliefs',
                'prompt_index': 1,
                'action_response': "I'll play Push...",
                'beliefs': {
                    'opponent_action': {'Push': 0.6, 'Pull': 0.4}
                },
                'raw_responses': [...],
                'success': True
            },
            ...
        ],
        ...
    ],
    'config': 'prisoners_dilemma'
}
```

## Analysis Utilities

```python
from belief_elicitation import extract_beliefs_summary

summary = extract_beliefs_summary(records)
# Returns:
# {
#     'opponent_action': {
#         'Push': {'mean': 0.55, 'std': 0.12, 'n': 30},
#         'Pull': {'mean': 0.45, 'std': 0.12, 'n': 30}
#     }
# }
```

## Testing

```bash
cd belief_elicitation
pytest tests.py -v
```

## Theoretical Background

This framework implements belief elicitation following:

1. **Savage's Subjective Expected Utility**: Beliefs are points in the probability simplex
2. **Bayesian Updating**: Beliefs update via conditioning on observations
3. **Proper Scoring Rules**: Incentive-compatible elicitation mechanisms

See the accompanying `AI_Belief.pdf` for formal treatment.

## Citation

If you use this framework in research, please cite the accompanying paper on belief updating in AI models.

## License

MIT License
