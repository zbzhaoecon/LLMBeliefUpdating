"""
Experiment 2: Causal Intervention
=================================

Three sub-experiments testing whether manipulating beliefs causally changes LLM strategy:
  2A: Information Treatment (partner history shifts beliefs -> actions)
  2B: Direct Belief Injection (state belief, observe EU-consistent action)
  2C: Fake History Priming (fabricated rounds, elicit beliefs at round 6)

Usage:
    # Dry run (2 instances per condition)
    python experiments/run_experiment2.py --dry_run --sub 2a

    # Run specific sub-experiment
    python experiments/run_experiment2.py --sub 2a --n_instances 30

    # Full experiment (all sub-experiments)
    python experiments/run_experiment2.py --n_instances 30

Run from project root: /Users/zbzhao/Desktop/Research/LLMBeliefUpdating/
"""

import os
import sys
import json
import copy
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
from tqdm import tqdm
from belief_elicitation import (
    create_belief_aware_runner,
    run_stag_hunt_with_beliefs,
    run_pd_with_beliefs,
    OpponentStrategy,
    OpponentStrategyType,
    get_stag_hunt_config,
    get_prisoners_dilemma_config,
    BeliefElicitationEngine,
    ElicitationPoint,
    BeliefQuestion,
    BeliefType,
    ScoringRule,
    GameBeliefConfig,
)
from belief_elicitation.integration import (
    _add_prompt_elicit_then_respond,
    extract_amount,
)

BASE_URL = "url here"
API_KEY = "your key here"
MODEL = "gpt-4.1-mini"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def save_records(sub_exp, game_name, records, condition=None):
    """Save experiment records to JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cond_str = f"_{condition}" if condition is not None else ""
    filename = f"exp2{sub_exp}_{game_name}{cond_str}_{MODEL}_{timestamp}.json"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Saved {filepath}")
    return filepath


# =============================================================================
# EXPERIMENT 2A: Information Treatment
# =============================================================================

def run_exp2a(runner, get_response, n_instances):
    """
    Run Experiment 2A: Information Treatment.

    For each treatment X in {1, 3, 5, 7, 9}, prepend info about opponent's
    past cooperation rate to the game prompt. Run Stag Hunt and PD.

    Returns dict of {game_treatment: filepath}.
    """
    treatments = [1, 3, 5, 7, 9]
    results = {}

    for X in treatments:
        # --- Stag Hunt ---
        info_text = (
            f"Before the game begins, you learn that in this partner's "
            f"10 most recent games with other players, they chose 'Stag' "
            f"in {X} out of 10 games.\n"
        )
        print(f"\n{'='*60}")
        print(f"EXP 2A - STAG HUNT: X={X}/10, {n_instances} instances")
        print(f"{'='*60}")

        records = run_stag_hunt_with_beliefs(
            runner=runner,
            get_response_func=get_response,
            n_instances=n_instances,
            n_rounds=5,
            opponent_strategy=OpponentStrategy(
                strategy_type=OpponentStrategyType.RANDOM,
                p_cooperate=0.5,
                cooperate_label="Stag",
                defect_label="Hare",
            ),
            prompt_prefix=info_text,
            print_except=True,
        )
        records["treatment_X"] = X
        fp = save_records("a", "stag_hunt", records, condition=f"X{X}")
        results[f"stag_hunt_X{X}"] = fp

        # --- PD ---
        info_text_pd = (
            f"Before the game begins, you learn that in this partner's "
            f"10 most recent games with other players, they chose 'Push' "
            f"in {X} out of 10 games.\n"
        )
        print(f"\n{'='*60}")
        print(f"EXP 2A - PD: X={X}/10, {n_instances} instances")
        print(f"{'='*60}")

        records = run_pd_with_beliefs(
            runner=runner,
            get_response_func=get_response,
            n_instances=n_instances,
            n_rounds=10,
            delta=0.9,
            opponent_strategy=OpponentStrategy(
                strategy_type=OpponentStrategyType.RANDOM,
                p_cooperate=0.5,
                cooperate_label="Push",
                defect_label="Pull",
            ),
            prompt_prefix=info_text_pd,
            print_except=True,
        )
        records["treatment_X"] = X
        fp = save_records("a", "pd", records, condition=f"X{X}")
        results[f"pd_X{X}"] = fp

    return results


# =============================================================================
# EXPERIMENT 2B: Direct Belief Injection
# =============================================================================

def _run_single_round_injection(
    runner, get_response, n_instances, game, injected_belief,
    payoffs=None, print_except=True, extractor_model='gpt-4.1-mini',
):
    """
    Run a single-round game with an injected belief statement.
    No belief elicitation (belief is set by design).

    Args:
        game: "stag_hunt" or "pd"
        injected_belief: Integer 0-100, the stated P(cooperate) %.
        payoffs: Custom payoff dict for Stag Hunt variants.
    """
    if game == "stag_hunt":
        if payoffs is None:
            payoffs = {
                ('Stag', 'Stag'): 4, ('Stag', 'Hare'): 0,
                ('Hare', 'Stag'): 3, ('Hare', 'Hare'): 2,
            }
        ss = payoffs[('Stag', 'Stag')]
        sh = payoffs[('Stag', 'Hare')]
        hs = payoffs[('Hare', 'Stag')]
        hh = payoffs[('Hare', 'Hare')]
        prompt = f"""You're paired with another player. You both simultaneously choose either 'Stag' or 'Hare'. Your payoffs depend on both choices:

* Both choose 'Stag': Each earns {ss} points
* Both choose 'Hare': Each earns {hh} points
* You choose 'Stag', the other chooses 'Hare': You earn {sh} points, the other earns {hs} points
* You choose 'Hare', the other chooses 'Stag': You earn {hs} points, the other earns {sh} points

Hunting the stag requires both players to coordinate, but hunting hare is safe regardless.

For this round, suppose you believe there is a {injected_belief}% chance the other player will choose Stag.

Which would you choose? Please highlight your choice with [] (i.e., [Stag] or [Hare])."""
        valid_actions = ["Stag", "Hare"]
        extract_prompt_template = 'Below is a response of a game player. What\'s their choice (Stag or Hare)? Output only the choice. If unclear, output None.\n\nInput: "{message}"\n\nOutput: '

    elif game == "pd":
        prompt = f"""You're paired with another player. Each round, you both choose a card: 'Push' or 'Pull'. Your payoff depends on both players' choices:

* Both play 'Push': Each earns $400
* You play 'Push', the other player plays 'Pull': You earn $0, the other player earns $700
* You play 'Pull', the other player plays 'Push': You earn $700, the other player earns $0
* Both play 'Pull': Each earns $300

After each round, there is a 90% chance the game continues to the next round. The game ends when the random draw says stop.

Your opponent is one of two types (you don't know which):
- Cooperative type: starts with Push and then mirrors your previous choice (tit-for-tat)
- Selfish type: always plays Pull

For this game, suppose you believe there is a {injected_belief}% chance the other player is the cooperative type (tit-for-tat).

Which card would you play in the first round? Please give a concrete choice and highlight it with [] (i.e., [Push] or [Pull])."""
        valid_actions = ["Push", "Pull"]
        extract_prompt_template = 'Below is a response of a game player. What\'s the card (Push or Pull) this player proposed to play? Output only the card. If unclear, output None.\n\nInput: "{message}"\n\nOutput: '
    else:
        raise ValueError(f"Unknown game: {game}")

    def extract_action(message):
        try:
            temp_messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": extract_prompt_template.format(message=message)}
            ]
            response = get_response(temp_messages, model=extractor_model)
            extracted = response["choices"][0]["message"]["content"].strip()
            if extracted in valid_actions:
                return extracted
            return None
        except Exception:
            return None

    records = {
        "messages": [],
        "responses": [],
        "actions": [],
        "injected_belief": injected_belief,
        "game": game,
        "config": f"exp2b_{game}",
    }

    with tqdm(total=n_instances) as pbar:
        while pbar.n < n_instances:
            try:
                messages = [{"role": "system", "content": runner.system_message}]
                responses = []

                # Greeting
                runner.update_messages(messages, responses, f"Hi, let's play a game.")

                # Game prompt (no belief elicitation)
                messages.append({"role": "user", "content": prompt})
                response = get_response(messages)
                responses.append(response)
                content = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": content})

                action = extract_action(content)
                if action not in valid_actions:
                    raise ValueError(f"Invalid action:\n{content[:200]}")

                records["messages"].append(messages)
                records["responses"].append(responses)
                records["actions"].append(action)

                pbar.update(1)

            except Exception as e:
                if print_except:
                    print(e)
                continue

    return records


def run_exp2b(runner, get_response, n_instances):
    """
    Run Experiment 2B: Direct Belief Injection.

    For each injected belief X% in {10, 30, 50, 70, 90}:
    - Stag Hunt (standard payoffs, threshold p*=2/3)
    - PD (single round)
    - Stag Hunt (modified payoffs, threshold p*=1/2) as control

    Returns dict of {variant_condition: filepath}.
    """
    injected_beliefs = [10, 30, 50, 70, 90]
    results = {}

    # Standard Stag Hunt: (S,S)=4, (S,H)=0, (H,S)=3, (H,H)=2, threshold p*=2/3
    standard_payoffs = {
        ('Stag', 'Stag'): 4, ('Stag', 'Hare'): 0,
        ('Hare', 'Stag'): 3, ('Hare', 'Hare'): 2,
    }

    # Modified Stag Hunt: (S,S)=4, (S,H)=0, (H,S)=2, (H,H)=2, threshold p*=1/2
    # V(Stag) = 4p, V(Hare) = 2, threshold: 4p = 2 => p* = 1/2
    modified_payoffs = {
        ('Stag', 'Stag'): 4, ('Stag', 'Hare'): 0,
        ('Hare', 'Stag'): 2, ('Hare', 'Hare'): 2,
    }

    for X in injected_beliefs:
        # Standard Stag Hunt
        print(f"\n{'='*60}")
        print(f"EXP 2B - STAG HUNT (standard): injected={X}%, {n_instances} inst")
        print(f"{'='*60}")
        records = _run_single_round_injection(
            runner, get_response, n_instances,
            game="stag_hunt", injected_belief=X,
            payoffs=standard_payoffs,
        )
        fp = save_records("b", "stag_hunt_standard", records, condition=f"B{X}")
        results[f"stag_standard_B{X}"] = fp

        # Modified Stag Hunt (control)
        print(f"\n{'='*60}")
        print(f"EXP 2B - STAG HUNT (modified, p*=1/2): injected={X}%, {n_instances} inst")
        print(f"{'='*60}")
        records = _run_single_round_injection(
            runner, get_response, n_instances,
            game="stag_hunt", injected_belief=X,
            payoffs=modified_payoffs,
        )
        records["payoff_variant"] = "modified_threshold_half"
        fp = save_records("b", "stag_hunt_modified", records, condition=f"B{X}")
        results[f"stag_modified_B{X}"] = fp

        # PD
        print(f"\n{'='*60}")
        print(f"EXP 2B - PD: injected={X}%, {n_instances} inst")
        print(f"{'='*60}")
        records = _run_single_round_injection(
            runner, get_response, n_instances,
            game="pd", injected_belief=X,
        )
        fp = save_records("b", "pd", records, condition=f"B{X}")
        results[f"pd_B{X}"] = fp

    return results


# =============================================================================
# EXPERIMENT 2C: Fake History Priming
# =============================================================================

def _construct_fake_stag_hunt_history(k, n_history_rounds=5, my_actions=None):
    """
    Construct a fake message history simulating n_history_rounds of Stag Hunt.

    Args:
        k: Number of rounds where opponent chose Stag (out of n_history_rounds).
        n_history_rounds: Number of fake rounds to construct.
        my_actions: LLM's actions in the fake history. If None, defaults to
            all Stag (to keep the LLM "in character" as a cooperator).

    Returns:
        List of message dicts ready to be extended with round n+1 prompt.
    """
    if my_actions is None:
        my_actions = ["Stag"] * n_history_rounds

    # Construct opponent action sequence: first k are Stag, rest are Hare
    # Shuffle to avoid order effects
    import random as _rng
    opp_actions = ["Stag"] * k + ["Hare"] * (n_history_rounds - k)
    _rng.shuffle(opp_actions)

    payoff_dict = {
        ('Stag', 'Stag'): (4, 4),
        ('Stag', 'Hare'): (0, 3),
        ('Hare', 'Stag'): (3, 0),
        ('Hare', 'Hare'): (2, 2),
    }

    round_labels = ['first', 'second', 'third', 'fourth', 'fifth',
                    'sixth', 'seventh', 'eighth', 'ninth', 'tenth']

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi, let's play a coordination game."},
        {"role": "assistant", "content": "Sure, I'd love to play a coordination game! What are the rules?"},
    ]

    # Round 1 prompt
    prompt_1 = f"""You're paired with another player. Each round, you both simultaneously choose either 'Stag' or 'Hare'. Your payoffs depend on both choices:

* Both choose 'Stag': Each earns 4 points
* Both choose 'Hare': Each earns 2 points
* You choose 'Stag', the other chooses 'Hare': You earn 0 points, the other earns 3 points
* You choose 'Hare', the other chooses 'Stag': You earn 3 points, the other earns 0 points

Hunting the stag requires both players to coordinate, but hunting hare is safe regardless.

In {n_history_rounds + 1} total rounds, which would you choose in the first round? Please highlight your choice with [] (i.e., [Stag] or [Hare])."""

    messages.append({"role": "user", "content": prompt_1})

    # Fake round 1 response
    my_choice_1 = my_actions[0]
    messages.append({
        "role": "assistant",
        "content": f"I'll choose [{my_choice_1}] for the first round."
    })

    # Rounds 2..n_history_rounds: feedback + fake response
    for r in range(2, n_history_rounds + 1):
        prev_my = my_actions[r - 2]
        prev_opp = opp_actions[r - 2]
        my_pay, opp_pay = payoff_dict[(prev_my, prev_opp)]
        label = round_labels[r - 2] if (r - 2) < len(round_labels) else f"{r-1}th"

        feedback = (
            f"In the {label} round, you chose '{prev_my}' and the other "
            f"player chose '{prev_opp}'. You earned {my_pay} points and the "
            f"other player earned {opp_pay} points. Which would you choose in "
            f"the next round? Please highlight your choice with [] "
            f"(i.e., [Stag] or [Hare])."
        )
        messages.append({"role": "user", "content": feedback})

        my_choice = my_actions[r - 1]
        messages.append({
            "role": "assistant",
            "content": f"I'll go with [{my_choice}] this round."
        })

    # Final round feedback (for round n_history_rounds)
    prev_my = my_actions[-1]
    prev_opp = opp_actions[-1]
    my_pay, opp_pay = payoff_dict[(prev_my, prev_opp)]
    label = round_labels[n_history_rounds - 1] if (n_history_rounds - 1) < len(round_labels) else f"{n_history_rounds}th"

    final_feedback = (
        f"In the {label} round, you chose '{prev_my}' and the other "
        f"player chose '{prev_opp}'. You earned {my_pay} points and the "
        f"other player earned {opp_pay} points. Which would you choose in "
        f"the next round? Please highlight your choice with [] "
        f"(i.e., [Stag] or [Hare])."
    )

    return messages, final_feedback, opp_actions


def run_exp2c(runner, get_response, n_instances):
    """
    Run Experiment 2C: Fake History Priming.

    For k in {0, 1, 2, 3, 4, 5}: construct a fake 5-round Stag Hunt history
    where the opponent chose Stag k times. Then elicit beliefs at round 6
    and get the LLM's action.
    """
    treatments = [0, 1, 2, 3, 4, 5]
    n_history = 5
    results = {}

    # Belief config for round 6 only
    belief_q = BeliefQuestion(
        question_id="partner_action_r6",
        belief_type=BeliefType.DISTRIBUTION,
        target_variable="Partner's action",
        outcomes=["Stag", "Hare"],
        prompt_template="""For round 6, before you decide:

{incentive_explanation}

What probability (0-100%) do you assign to each of the other player's
possible actions in this round?

- Stag: ___%
- Hare: ___%

(Must sum to 100%. Highlight each probability, e.g., [60]% and [40]%)
""",
        include_incentive_explanation=True,
        scoring_rule=ScoringRule.QUADRATIC,
    )
    belief_ep = ElicitationPoint(
        point_id="round_6_beliefs",
        at_prompt_index=6,  # round 6
        questions=[belief_q],
    )
    belief_config = GameBeliefConfig(
        game_name="stag_hunt_fake_history",
        elicitation_points=[belief_ep],
        incentivized=True,
    )

    def extract_choice(message):
        try:
            temp_messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": f'Below is a response of a game player. What\'s their choice (Stag or Hare)? Output only the choice. If unclear, output None.\n\nInput: "{message}"\n\nOutput: '}
            ]
            response = get_response(temp_messages, model='gpt-4.1-mini')
            extracted = response["choices"][0]["message"]["content"].strip()
            if extracted in ["Stag", "Hare"]:
                return extracted
            return None
        except Exception:
            return None

    for k in treatments:
        print(f"\n{'='*60}")
        print(f"EXP 2C - FAKE HISTORY: k={k}/{n_history} Stag, {n_instances} instances")
        print(f"{'='*60}")

        records = {
            "messages": [],
            "responses": [],
            "actions": [],
            "beliefs": [],
            "fake_opponent_actions": [],
            "treatment_k": k,
            "config": "exp2c_fake_history",
        }

        with tqdm(total=n_instances) as pbar:
            while pbar.n < n_instances:
                try:
                    # Construct fake history
                    messages, round6_prompt, opp_actions = (
                        _construct_fake_stag_hunt_history(k, n_history)
                    )
                    responses = []
                    instance_beliefs = []

                    # Round 6: elicit beliefs pre-decision, then get action
                    content = _add_prompt_elicit_then_respond(
                        messages, responses, round6_prompt, 6,
                        belief_config, runner, get_response,
                        instance_beliefs,
                    )

                    action = extract_choice(content)
                    if action not in ["Stag", "Hare"]:
                        raise ValueError(f"Invalid action:\n{content[:200]}")

                    records["messages"].append(messages)
                    records["responses"].append(responses)
                    records["actions"].append(action)
                    records["beliefs"].append(instance_beliefs)
                    records["fake_opponent_actions"].append(opp_actions)

                    pbar.update(1)

                except Exception as e:
                    if True:
                        print(e)
                    continue

        fp = save_records("c", "stag_hunt_fake", records, condition=f"k{k}")
        results[f"k{k}"] = fp

    return results


# =============================================================================
# MAIN
# =============================================================================

SUB_EXPERIMENTS = {
    "2a": run_exp2a,
    "2b": run_exp2b,
    "2c": run_exp2c,
}


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: Causal Intervention")
    parser.add_argument(
        "--n_instances", type=int, default=30,
        help="Instances per condition (default: 30)",
    )
    parser.add_argument(
        "--sub", nargs="+", default=list(SUB_EXPERIMENTS.keys()),
        choices=list(SUB_EXPERIMENTS.keys()),
        help="Which sub-experiments to run (default: all)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Quick test with 2 instances per condition",
    )
    args = parser.parse_args()

    n = 2 if args.dry_run else args.n_instances

    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    runner, get_response, _ = create_belief_aware_runner(
        client=client, model=MODEL,
        system_message="You are a helpful assistant.",
        time_limit=60,
    )

    print(f"Experiment 2: Causal Intervention")
    print(f"Model: {MODEL}")
    print(f"Instances per condition: {n}")
    print(f"Sub-experiments: {args.sub}")

    all_results = {}
    for sub in args.sub:
        print(f"\n{'#'*60}")
        print(f"SUB-EXPERIMENT {sub.upper()}")
        print(f"{'#'*60}")
        all_results[sub] = SUB_EXPERIMENTS[sub](runner, get_response, n)

    print(f"\n{'='*60}")
    print("EXPERIMENT 2 COMPLETE")
    print(f"{'='*60}")
    for sub, results in all_results.items():
        print(f"\n  {sub.upper()}:")
        for key, filepath in results.items():
            print(f"    {key}: {filepath}")


if __name__ == "__main__":
    main()
