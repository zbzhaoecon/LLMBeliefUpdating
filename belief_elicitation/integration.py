"""
Integration Module: Game Runners with Belief Elicitation

Provides game-specific run functions that:
1. Present game prompts to the LLM
2. Elicit beliefs PRE-DECISION in forked conversation branches
3. Collect the LLM's action
4. Record both beliefs and actions for analysis
"""

import copy
import re
import random
from typing import List, Dict, Any, Optional, Callable, Tuple
from tqdm import tqdm

from .core import (
    BeliefElicitationEngine, GameBeliefConfig,
    ElicitationPoint, BeliefQuestion, BeliefType
)
from .game_configs import (
    get_prisoners_dilemma_config,
    get_ultimatum_proposer_config,
    get_stag_hunt_config,
    get_beauty_contest_config,
    get_first_price_auction_config,
)
from .opponents import OpponentStrategy, OpponentStrategyType


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def extract_brackets(text: str, brackets: str = '[]') -> List[str]:
    """Extract content from bracketed text."""
    assert len(brackets) == 2
    pattern = re.escape(brackets[0]) + r'(.*?)' + re.escape(brackets[1])
    return re.findall(pattern, text)


def extract_amount(
    message: str,
    prefix: str = '',
    print_except: bool = True,
    value_type: type = float,
    brackets: str = '[]'
) -> Optional[Any]:
    """Extract numerical amount from message."""
    try:
        matches = extract_brackets(message, brackets=brackets)
        matches = [s.replace(' ', '') for s in matches]
        matches = [s[len(prefix):] if s.startswith(prefix) else s for s in matches]

        if len(matches) == 0:
            raise ValueError('No bracketed answer found: %s' % message[:200])
        for i in range(len(matches)):
            if matches[i] != matches[0]:
                raise ValueError('Ambiguous answer: %s' % message[:200])
        return value_type(matches[0])
    except Exception as e:
        if print_except:
            print(e)
        return None


# =============================================================================
# BELIEF-AWARE RUNNER
# =============================================================================

class BeliefAwareRunner:
    """
    Base class for running games with belief elicitation.

    Provides the core logic for forking conversations and eliciting
    beliefs pre-decision.
    """

    def __init__(
        self,
        get_response_func: Callable,
        update_messages_func: Callable,
        model: str = None,
        system_message: str = "You are a helpful assistant.",
        time_limit: int = 60,
        print_exceptions: bool = True
    ):
        self.get_response = get_response_func
        self.update_messages = update_messages_func
        self.model = model
        self.system_message = system_message
        self.time_limit = time_limit
        self.print_exceptions = print_exceptions

        self.belief_engine = BeliefElicitationEngine(
            get_response_func=get_response_func,
            model=model,
            print_exceptions=print_exceptions
        )

    def elicit_beliefs_in_branch(
        self,
        messages: List[Dict],
        elicitation_point: ElicitationPoint,
        context: Optional[Dict[str, Any]] = None,
        incentivized: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Elicit beliefs in a forked conversation branch.

        CRITICAL: This does NOT modify the original messages list.
        """
        return self.belief_engine.fork_and_elicit(
            messages, elicitation_point, context, incentivized=incentivized
        )


def create_belief_aware_runner(
    client,
    model: str,
    system_message: str = "You are a helpful assistant.",
    time_limit: int = 60
) -> Tuple[BeliefAwareRunner, Callable, Callable]:
    """
    Factory function to create a belief-aware runner with proper closures.

    Returns:
        Tuple of (runner, get_response_func, update_messages_func)
    """
    def get_response(messages: List[Dict], model: str = model, **kwargs) -> Dict:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=time_limit,
            **kwargs
        )
        return response.model_dump()

    def update_messages(
        messages: List[Dict],
        responses: List[Dict],
        prompt: str,
        **kwargs
    ) -> None:
        messages.append({"role": "user", "content": prompt})
        response = get_response(messages, **kwargs)
        responses.append(response)
        messages.append({
            "role": "assistant",
            "content": response["choices"][0]["message"]["content"]
        })

    runner = BeliefAwareRunner(
        get_response_func=get_response,
        update_messages_func=update_messages,
        model=model,
        system_message=system_message,
        time_limit=time_limit
    )

    return runner, get_response, update_messages


# =============================================================================
# HELPER: PRE-DECISION BELIEF ELICITATION + ACTION
# =============================================================================

def _add_prompt_elicit_then_respond(
    messages: List[Dict],
    responses: List[Dict],
    prompt: str,
    prompt_idx: int,
    belief_config: Optional[GameBeliefConfig],
    runner: BeliefAwareRunner,
    get_response_func: Callable,
    instance_beliefs: List[Dict],
    **kwargs
) -> str:
    """
    Core pattern: add prompt → elicit beliefs (pre-decision) → get LLM response.

    Returns the LLM's response content string.
    """
    messages.append({"role": "user", "content": prompt})

    # PRE-DECISION belief elicitation (fork before LLM responds)
    if belief_config:
        ep = belief_config.get_elicitation_point(prompt_idx)
        if ep:
            belief_result = runner.elicit_beliefs_in_branch(
                messages, ep, incentivized=belief_config.incentivized
            )
            belief_result['prompt_index'] = prompt_idx
            instance_beliefs.append(belief_result)

    # NOW get the LLM's action
    response = get_response_func(messages, **kwargs)
    responses.append(response)
    content = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": content})
    return content


# =============================================================================
# GENERAL-PURPOSE SESSION RUNNER
# =============================================================================

def run_one_session_with_beliefs(
    prompts: List[str],
    n_instances: int,
    belief_config: GameBeliefConfig,
    runner: BeliefAwareRunner,
    get_response_func: Callable,
    orders: Optional[List[List[int]]] = None,
    print_except: bool = True,
    tqdm_silent: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a standard prompt-based game session with pre-decision belief elicitation.
    """
    records = {
        'messages': [],
        'responses': [],
        'beliefs': [],
        'config': belief_config.game_name
    }

    if orders is None:
        orders = [list(range(len(prompts)))] * n_instances
    elif len(orders) < n_instances:
        raise ValueError(
            f"orders has {len(orders)} entries but n_instances={n_instances}"
        )

    with tqdm(total=n_instances, disable=tqdm_silent) as pbar:
        i = 0
        while i < n_instances:
            try:
                responses = []
                messages = [
                    {"role": "system", "content": runner.system_message}
                ]
                instance_beliefs = []

                for prompt_idx in orders[i]:
                    prompt = prompts[prompt_idx]
                    _add_prompt_elicit_then_respond(
                        messages, responses, prompt, prompt_idx,
                        belief_config, runner, runner.get_response,
                        instance_beliefs, **kwargs
                    )

                records['messages'].append(messages)
                records['responses'].append(responses)
                records['beliefs'].append(instance_beliefs)

                i += 1
                pbar.update(1)

            except Exception as e:
                if print_except:
                    print(e)
                continue

    return records


# =============================================================================
# PRISONER'S DILEMMA
# =============================================================================

def run_pd_with_beliefs(
    runner: BeliefAwareRunner,
    get_response_func: Callable,
    n_instances: int = 30,
    n_rounds: int = 10,
    delta: float = 0.9,
    opponent_strategy: Optional[OpponentStrategy] = None,
    belief_config: Optional[GameBeliefConfig] = None,
    print_except: bool = True,
    extractor_model: str = 'gpt-4.1-mini',
    prompt_prefix: str = '',
) -> Dict[str, Any]:
    """
    Run infinitely repeated Prisoner's Dilemma with pre-decision belief elicitation.

    Uses continuation probability framing (delta) instead of fixed rounds.
    Opponent type space is restricted to {cooperative (TFT), selfish (always Pull)}.

    Args:
        n_rounds: Number of rounds to actually play (LLM is not told this).
        delta: Continuation probability per round (told to LLM).
        opponent_strategy: How the simulated opponent plays. Defaults to RANDOM(p=0.5).
        prompt_prefix: Text prepended to the game prompt (e.g., info treatment).
    """
    if belief_config is None:
        belief_config = get_prisoners_dilemma_config(n_rounds=n_rounds, delta=delta)
    if opponent_strategy is None:
        opponent_strategy = OpponentStrategy(
            strategy_type=OpponentStrategyType.RANDOM,
            p_cooperate=0.5,
            cooperate_label="Push",
            defect_label="Pull"
        )

    prompt_init = "Hi, let's play a fun card game."

    prompt_1 = f'''{prompt_prefix}
You're paired with another player. Each round, you both choose a card: 'Push' or 'Pull'. Your payoff depends on both players' choices:

* Both play 'Push': Each earns $400
* You play 'Push', the other player plays 'Pull': You earn $0, the other player earns $700
* You play 'Pull', the other player plays 'Push': You earn $700, the other player earns $0
* Both play 'Pull': Each earns $300

After each round, there is a {delta*100:.0f}% chance the game continues to the next round. The game ends when the random draw says stop.

Your opponent is one of two types (you don't know which):
- Cooperative type: starts with Push and then mirrors your previous choice (tit-for-tat)
- Selfish type: always plays Pull

Which card would you play in the first round? Please give a concrete choice and highlight it with [] (i.e., [Push] or [Pull]).
'''

    round_labels = ['first', 'second', 'third', 'fourth', 'fifth',
                    'sixth', 'seventh', 'eighth', 'ninth', 'tenth']

    def make_feedback_prompt(label, my_card, opp_card, my_payoff, opp_payoff):
        return (
            f"In the {label} round, you played '{my_card}' and the other "
            f"player played '{opp_card}'. You earned ${my_payoff} and the "
            f"other player earned ${opp_payoff}. The game continues. "
            f"Which card would you play in the next round? Please give a "
            f"concrete choice and highlight it with [] (e.g., [Push] or [Pull])."
        )

    def extract_card_chatgpt(message: str) -> Optional[str]:
        prompt = f'''Below is a response of a game player. What's the card (Push or Pull) this player proposed to play? Output only the card. If unclear, output None.

Input: "{message}"

Output: '''
        try:
            temp_messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ]
            response = get_response_func(temp_messages, model=extractor_model)
            extracted = response["choices"][0]["message"]["content"].strip()
            if extracted in ['Push', 'Pull']:
                return extracted
            return None
        except Exception:
            return None

    payoff_dict = {
        ('Push', 'Push'): 400,
        ('Push', 'Pull'): 0,
        ('Pull', 'Push'): 700,
        ('Pull', 'Pull'): 300
    }

    records = {
        'messages': [],
        'responses': [],
        'choices': [],
        'beliefs': [],
        'opponent_actions': [],
        'config': belief_config.game_name
    }

    with tqdm(total=n_instances) as pbar:
        while pbar.n < n_instances:
            choices_tmp = []
            opp_actions_tmp = []
            instance_beliefs = []
            try:
                responses = []
                messages = [
                    {"role": "system", "content": runner.system_message}
                ]

                # Greeting (prompt index 0, no beliefs)
                runner.update_messages(messages, responses, prompt_init)

                # Round 1 (prompt index 1): elicit beliefs pre-decision, then get action
                content = _add_prompt_elicit_then_respond(
                    messages, responses, prompt_1, 1,
                    belief_config, runner, get_response_func,
                    instance_beliefs
                )

                card = extract_card_chatgpt(content)
                if card not in ['Push', 'Pull']:
                    raise ValueError(f'Invalid answer:\n{content[:200]}')
                choices_tmp.append(card)

                # Rounds 2..n_rounds
                for r in range(2, n_rounds + 1):
                    opp_card = opponent_strategy.get_action(r - 1, choices_tmp[:-1])
                    opp_actions_tmp.append(opp_card)

                    my_payoff = payoff_dict[(card, opp_card)]
                    opp_payoff = payoff_dict[(opp_card, card)]
                    label = round_labels[r - 2] if (r - 2) < len(round_labels) else f"{r-1}th"

                    feedback = make_feedback_prompt(
                        label, card, opp_card, my_payoff, opp_payoff
                    )

                    content = _add_prompt_elicit_then_respond(
                        messages, responses, feedback, r,
                        belief_config, runner, get_response_func,
                        instance_beliefs
                    )

                    card = extract_card_chatgpt(content)
                    if card not in ['Push', 'Pull']:
                        raise ValueError(f'Invalid answer:\n{content[:200]}')
                    choices_tmp.append(card)

                # Final round opponent action (for the last round played)
                final_opp = opponent_strategy.get_action(n_rounds, choices_tmp[:-1])
                opp_actions_tmp.append(final_opp)

                records['messages'].append(messages)
                records['responses'].append(responses)
                records['choices'].append(choices_tmp)
                records['opponent_actions'].append(opp_actions_tmp)
                records['beliefs'].append(instance_beliefs)

                pbar.update(1)

            except Exception as e:
                if print_except:
                    print(e)
                continue

    return records


# =============================================================================
# STAG HUNT
# =============================================================================

def run_stag_hunt_with_beliefs(
    runner: BeliefAwareRunner,
    get_response_func: Callable,
    n_instances: int = 30,
    n_rounds: int = 5,
    opponent_strategy: Optional[OpponentStrategy] = None,
    belief_config: Optional[GameBeliefConfig] = None,
    print_except: bool = True,
    extractor_model: str = 'gpt-4.1-mini',
    prompt_prefix: str = '',
    payoffs: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run repeated Stag Hunt with pre-decision belief elicitation.

    Args:
        prompt_prefix: Text prepended to the game prompt (e.g., info treatment).
        payoffs: Custom payoff dict. Default: (S,S)=4, (S,H)=0, (H,S)=3, (H,H)=2.
    """
    if belief_config is None:
        belief_config = get_stag_hunt_config(n_rounds=n_rounds)
    if opponent_strategy is None:
        opponent_strategy = OpponentStrategy(
            strategy_type=OpponentStrategyType.RANDOM,
            p_cooperate=0.5,
            cooperate_label="Stag",
            defect_label="Hare"
        )

    prompt_init = "Hi, let's play a coordination game."

    payoff_dict = payoffs if payoffs is not None else {
        ('Stag', 'Stag'): 4,
        ('Stag', 'Hare'): 0,
        ('Hare', 'Stag'): 3,
        ('Hare', 'Hare'): 2
    }

    ss = payoff_dict[('Stag', 'Stag')]
    sh = payoff_dict[('Stag', 'Hare')]
    hs = payoff_dict[('Hare', 'Stag')]
    hh = payoff_dict[('Hare', 'Hare')]

    prompt_1 = f'''{prompt_prefix}
You're paired with another player. Each round, you both simultaneously choose either 'Stag' or 'Hare'. Your payoffs depend on both choices:

* Both choose 'Stag': Each earns {ss} points
* Both choose 'Hare': Each earns {hh} points
* You choose 'Stag', the other chooses 'Hare': You earn {sh} points, the other earns {hs} points
* You choose 'Hare', the other chooses 'Stag': You earn {hs} points, the other earns {sh} points

Hunting the stag requires both players to coordinate, but hunting hare is safe regardless.

In {n_rounds} total rounds, which would you choose in the first round? Please highlight your choice with [] (i.e., [Stag] or [Hare]).
'''

    round_labels = ['first', 'second', 'third', 'fourth', 'fifth',
                    'sixth', 'seventh', 'eighth', 'ninth', 'tenth']

    def extract_choice_chatgpt(message: str) -> Optional[str]:
        prompt = f'''Below is a response of a game player. What's their choice (Stag or Hare)? Output only the choice. If unclear, output None.

Input: "{message}"

Output: '''
        try:
            temp_messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ]
            response = get_response_func(temp_messages, model=extractor_model)
            extracted = response["choices"][0]["message"]["content"].strip()
            if extracted in ['Stag', 'Hare']:
                return extracted
            return None
        except Exception:
            return None

    records = {
        'messages': [],
        'responses': [],
        'choices': [],
        'beliefs': [],
        'opponent_actions': [],
        'config': belief_config.game_name
    }

    with tqdm(total=n_instances) as pbar:
        while pbar.n < n_instances:
            choices_tmp = []
            opp_actions_tmp = []
            instance_beliefs = []
            try:
                responses = []
                messages = [
                    {"role": "system", "content": runner.system_message}
                ]

                runner.update_messages(messages, responses, prompt_init)

                content = _add_prompt_elicit_then_respond(
                    messages, responses, prompt_1, 1,
                    belief_config, runner, get_response_func,
                    instance_beliefs
                )

                choice = extract_choice_chatgpt(content)
                if choice not in ['Stag', 'Hare']:
                    raise ValueError(f'Invalid answer:\n{content[:200]}')
                choices_tmp.append(choice)

                for r in range(2, n_rounds + 1):
                    opp_choice = opponent_strategy.get_action(r - 1, choices_tmp[:-1])
                    opp_actions_tmp.append(opp_choice)

                    my_payoff = payoff_dict[(choice, opp_choice)]
                    opp_payoff = payoff_dict[(opp_choice, choice)]
                    label = round_labels[r - 2] if (r - 2) < len(round_labels) else f"{r-1}th"

                    feedback = (
                        f"In the {label} round, you chose '{choice}' and the other "
                        f"player chose '{opp_choice}'. You earned {my_payoff} points and the "
                        f"other player earned {opp_payoff} points. Which would you choose in "
                        f"the next round? Please highlight your choice with [] "
                        f"(i.e., [Stag] or [Hare])."
                    )

                    content = _add_prompt_elicit_then_respond(
                        messages, responses, feedback, r,
                        belief_config, runner, get_response_func,
                        instance_beliefs
                    )

                    choice = extract_choice_chatgpt(content)
                    if choice not in ['Stag', 'Hare']:
                        raise ValueError(f'Invalid answer:\n{content[:200]}')
                    choices_tmp.append(choice)

                final_opp = opponent_strategy.get_action(n_rounds, choices_tmp[:-1])
                opp_actions_tmp.append(final_opp)

                records['messages'].append(messages)
                records['responses'].append(responses)
                records['choices'].append(choices_tmp)
                records['opponent_actions'].append(opp_actions_tmp)
                records['beliefs'].append(instance_beliefs)

                pbar.update(1)

            except Exception as e:
                if print_except:
                    print(e)
                continue

    return records


# =============================================================================
# p-BEAUTY CONTEST
# =============================================================================

def run_beauty_contest_with_beliefs(
    runner: BeliefAwareRunner,
    get_response_func: Callable,
    n_instances: int = 30,
    n_rounds: int = 3,
    p: float = 2 / 3,
    n_players: int = 3,
    simulated_opponent_fn: Optional[Callable] = None,
    belief_config: Optional[GameBeliefConfig] = None,
    print_except: bool = True,
) -> Dict[str, Any]:
    """
    Run p-Beauty Contest with pre-decision belief elicitation.

    Args:
        p: Target multiplier (default 2/3)
        n_players: Total number of players including the LLM
        simulated_opponent_fn: Function(round, prev_results) -> list of other players' numbers.
            Defaults to uniform random in [0,100].
    """
    if belief_config is None:
        belief_config = get_beauty_contest_config(
            p=p, n_players=n_players, n_rounds=n_rounds
        )
    if simulated_opponent_fn is None:
        def simulated_opponent_fn(round_num, prev_results):
            return [random.uniform(0, 100) for _ in range(n_players - 1)]

    prompt_init = "Hi, let's play a number guessing game."

    prompt_1 = f"""
You are playing with {n_players - 1} other players. Each player simultaneously chooses a number from 0 to 100 (inclusive). The winner is the player whose number is closest to {p:.4g} times the average of ALL players' chosen numbers.

The winner earns a bonus. In case of a tie, the bonus is split equally.

There will be {n_rounds} round(s). What number do you choose for round 1? Please highlight your choice in brackets, e.g., [33].
"""

    records = {
        'messages': [],
        'responses': [],
        'choices': [],
        'beliefs': [],
        'round_results': [],
        'config': belief_config.game_name
    }

    with tqdm(total=n_instances) as pbar:
        while pbar.n < n_instances:
            choices_tmp = []
            round_results_tmp = []
            instance_beliefs = []
            try:
                responses = []
                messages = [
                    {"role": "system", "content": runner.system_message}
                ]

                runner.update_messages(messages, responses, prompt_init)

                # Round 1
                content = _add_prompt_elicit_then_respond(
                    messages, responses, prompt_1, 1,
                    belief_config, runner, get_response_func,
                    instance_beliefs
                )

                my_number = extract_amount(content, print_except=print_except)
                if my_number is None:
                    raise ValueError(f'Cannot extract number:\n{content[:200]}')
                my_number = float(my_number)
                choices_tmp.append(my_number)

                others = simulated_opponent_fn(1, [])
                all_numbers = [my_number] + others
                avg = sum(all_numbers) / len(all_numbers)
                target = p * avg
                round_results_tmp.append({
                    'round': 1,
                    'my_number': my_number,
                    'others': others,
                    'average': avg,
                    'target': target
                })

                # Subsequent rounds
                for r in range(2, n_rounds + 1):
                    feedback = (
                        f"Round {r-1} results: The average of all players' numbers was "
                        f"{avg:.1f}, so the target was {p:.4g} × {avg:.1f} = {target:.1f}. "
                        f"What number do you choose for round {r}? "
                        f"Please highlight your choice in brackets, e.g., [33]."
                    )

                    content = _add_prompt_elicit_then_respond(
                        messages, responses, feedback, r,
                        belief_config, runner, get_response_func,
                        instance_beliefs
                    )

                    my_number = extract_amount(content, print_except=print_except)
                    if my_number is None:
                        raise ValueError(f'Cannot extract number:\n{content[:200]}')
                    my_number = float(my_number)
                    choices_tmp.append(my_number)

                    others = simulated_opponent_fn(r, round_results_tmp)
                    all_numbers = [my_number] + others
                    avg = sum(all_numbers) / len(all_numbers)
                    target = p * avg
                    round_results_tmp.append({
                        'round': r,
                        'my_number': my_number,
                        'others': others,
                        'average': avg,
                        'target': target
                    })

                records['messages'].append(messages)
                records['responses'].append(responses)
                records['choices'].append(choices_tmp)
                records['round_results'].append(round_results_tmp)
                records['beliefs'].append(instance_beliefs)

                pbar.update(1)

            except Exception as e:
                if print_except:
                    print(e)
                continue

    return records


# =============================================================================
# FIRST-PRICE SEALED-BID AUCTION
# =============================================================================

def run_first_price_auction_with_beliefs(
    runner: BeliefAwareRunner,
    get_response_func: Callable,
    n_instances: int = 30,
    n_bidders: int = 2,
    valuation_range: Tuple[float, float] = (0, 100),
    opponent_bid_fn: Optional[Callable] = None,
    belief_config: Optional[GameBeliefConfig] = None,
    print_except: bool = True,
) -> Dict[str, Any]:
    """
    Run First-Price Sealed-Bid Auction with pre-decision belief elicitation.

    Each instance: LLM gets a random private valuation, submits a bid.
    BNE for 2 players with Uniform[0,100]: bid = v/2.

    Args:
        opponent_bid_fn: Function(opponent_valuation) -> bid. Defaults to BNE: v/2.
    """
    if belief_config is None:
        belief_config = get_first_price_auction_config(n_bidders=n_bidders)
    if opponent_bid_fn is None:
        def opponent_bid_fn(v):
            return v / 2.0  # BNE strategy

    prompt_init = "Hi, let's play an auction game."

    records = {
        'messages': [],
        'responses': [],
        'bids': [],
        'valuations': [],
        'beliefs': [],
        'outcomes': [],
        'config': belief_config.game_name
    }

    with tqdm(total=n_instances) as pbar:
        while pbar.n < n_instances:
            instance_beliefs = []
            try:
                responses = []
                messages = [
                    {"role": "system", "content": runner.system_message}
                ]

                # Draw private valuation
                v = random.uniform(*valuation_range)
                opp_v = random.uniform(*valuation_range)
                opp_bid = opponent_bid_fn(opp_v)

                runner.update_messages(messages, responses, prompt_init)

                auction_prompt = f"""
This is a first-price sealed-bid auction with {n_bidders} bidders. Each bidder has a private valuation drawn uniformly from ${valuation_range[0]:.0f} to ${valuation_range[1]:.0f}.

Your private valuation for the item is ${v:.0f}. You submit a bid, and the highest bidder wins the item and pays their own bid. If you win, your profit is your valuation minus your bid. If you lose, your profit is $0.

How much would you like to bid? Please highlight your bid in brackets, e.g., [$30].
"""

                content = _add_prompt_elicit_then_respond(
                    messages, responses, auction_prompt, 1,
                    belief_config, runner, get_response_func,
                    instance_beliefs
                )

                bid = extract_amount(content, prefix='$', print_except=print_except)
                if bid is None:
                    raise ValueError(f'Cannot extract bid:\n{content[:200]}')
                bid = float(bid)

                won = bid > opp_bid
                profit = (v - bid) if won else 0

                records['messages'].append(messages)
                records['responses'].append(responses)
                records['bids'].append(bid)
                records['valuations'].append(v)
                records['beliefs'].append(instance_beliefs)
                records['outcomes'].append({
                    'won': won,
                    'my_bid': bid,
                    'my_valuation': v,
                    'opponent_bid': opp_bid,
                    'opponent_valuation': opp_v,
                    'profit': profit
                })

                pbar.update(1)

            except Exception as e:
                if print_except:
                    print(e)
                continue

    return records


# =============================================================================
# ULTIMATUM GAME (PROPOSER)
# =============================================================================

def run_ultimatum_proposer_with_beliefs(
    runner: BeliefAwareRunner,
    get_response_func: Callable,
    n_instances: int = 30,
    belief_config: Optional[GameBeliefConfig] = None,
    print_except: bool = True
) -> Dict[str, Any]:
    """
    Run Ultimatum Game (Proposer) with pre-decision belief elicitation.
    """
    if belief_config is None:
        belief_config = get_ultimatum_proposer_config()

    prompts = [
        "Hi, let's play a game.",
        ('This is a two-player game. You are the Proposer, and the other '
         'player is the Responder. As the proposer, you propose how to divide '
         '$100 and the Responder chooses either Accept or Reject. If accepted, '
         'the two of you will earn as described by the accepted proposal '
         'accordingly. If rejected, then both of you will earn $0.\n'
         'How much would you like to propose to give to the Responder? '
         'Please give only one concrete choice, and highlight the amount '
         'with [] (such as [$x]).'),
    ]

    return run_one_session_with_beliefs(
        prompts=prompts,
        n_instances=n_instances,
        belief_config=belief_config,
        runner=runner,
        get_response_func=runner.get_response,
        print_except=print_except
    )


# =============================================================================
# EXTRACTION HELPERS
# =============================================================================

def extract_choices_from_records(
    records: Dict[str, Any],
    prefix: str = '$',
    brackets: str = '[]'
) -> List[Any]:
    """Extract choices from records (backward compatible)."""
    choices = [
        extract_amount(
            messages[-1]['content'],
            prefix=prefix,
            print_except=True,
            value_type=float,
            brackets=brackets
        )
        for messages in records['messages']
    ]
    return [x for x in choices if x is not None]


def extract_beliefs_summary(records: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary statistics of elicited beliefs."""
    import numpy as np

    summary = {}

    all_beliefs = records.get('beliefs', [])
    if not all_beliefs:
        return summary

    beliefs_by_question = {}
    for instance_beliefs in all_beliefs:
        for belief_record in instance_beliefs:
            for q_id, belief_value in belief_record.get('beliefs', {}).items():
                if q_id not in beliefs_by_question:
                    beliefs_by_question[q_id] = []
                if belief_value is not None:
                    beliefs_by_question[q_id].append(belief_value)

    for q_id, values in beliefs_by_question.items():
        if not values:
            continue

        if isinstance(values[0], dict):
            keys = values[0].keys()
            summary[q_id] = {
                k: {
                    'mean': float(np.mean([v[k] for v in values if k in v])),
                    'std': float(np.std([v[k] for v in values if k in v])),
                    'n': len([v for v in values if k in v])
                }
                for k in keys
            }
        else:
            summary[q_id] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'n': len(values),
                'values': values
            }

    return summary
