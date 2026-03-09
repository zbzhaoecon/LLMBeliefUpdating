"""
Experiment 1: Belief-Action Consistency
=======================================

Run all 5 games with belief elicitation and save records for analysis.

Usage:
    # Dry run (2 instances, quick test)
    python experiments/run_experiment1.py --dry_run

    # Run specific game
    python experiments/run_experiment1.py --games stag_hunt --n_instances 50

    # Full experiment
    python experiments/run_experiment1.py --n_instances 50

Run from project root: /Users/zbzhao/Desktop/Research/LLMBeliefUpdating/
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
from belief_elicitation import (
    create_belief_aware_runner,
    run_pd_with_beliefs,
    run_stag_hunt_with_beliefs,
    run_beauty_contest_with_beliefs,
    run_first_price_auction_with_beliefs,
    run_ultimatum_proposer_with_beliefs,
    OpponentStrategy,
    OpponentStrategyType,
)

# API Configuration
BASE_URL = "url here"
API_KEY = "your key here"
MODEL = "gpt-4.1-mini"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def save_records(game_name, records):
    """Save experiment records to JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"exp1_{game_name}_{MODEL}_{timestamp}.json"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Saved {filepath}")
    return filepath


def run_stag_hunt(runner, get_response, n_instances):
    """Stag Hunt: 5 rounds, RANDOM(0.5) opponent."""
    print(f"\n{'='*60}")
    print(f"STAG HUNT: {n_instances} instances, 5 rounds, RANDOM(0.5)")
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
        print_except=True,
    )
    return save_records("stag_hunt", records), records


def run_pd(runner, get_response, n_instances):
    """PD: infinite horizon (delta=0.9), 10 rounds, RANDOM(0.5) opponent."""
    print(f"\n{'='*60}")
    print(f"PRISONER'S DILEMMA: {n_instances} instances, 10 rounds, delta=0.9")
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
        print_except=True,
    )
    return save_records("pd_infinite", records), records


def run_beauty(runner, get_response, n_instances):
    """Beauty Contest: 3 rounds, 3 players, p=2/3."""
    print(f"\n{'='*60}")
    print(f"BEAUTY CONTEST: {n_instances} instances, 3 rounds, p=2/3")
    print(f"{'='*60}")
    records = run_beauty_contest_with_beliefs(
        runner=runner,
        get_response_func=get_response,
        n_instances=n_instances,
        n_rounds=3,
        p=2 / 3,
        n_players=3,
        print_except=True,
    )
    return save_records("beauty_contest", records), records


def run_auction(runner, get_response, n_instances):
    """First-Price Auction: BNE opponent."""
    print(f"\n{'='*60}")
    print(f"FIRST-PRICE AUCTION: {n_instances} instances")
    print(f"{'='*60}")
    records = run_first_price_auction_with_beliefs(
        runner=runner,
        get_response_func=get_response,
        n_instances=n_instances,
        print_except=True,
    )
    return save_records("auction", records), records


def run_ultimatum(runner, get_response, n_instances):
    """Ultimatum Game (Proposer)."""
    print(f"\n{'='*60}")
    print(f"ULTIMATUM PROPOSER: {n_instances} instances")
    print(f"{'='*60}")
    records = run_ultimatum_proposer_with_beliefs(
        runner=runner,
        get_response_func=get_response,
        n_instances=n_instances,
        print_except=True,
    )
    return save_records("ultimatum", records), records


GAME_RUNNERS = {
    "stag_hunt": run_stag_hunt,
    "pd": run_pd,
    "beauty_contest": run_beauty,
    "auction": run_auction,
    "ultimatum": run_ultimatum,
}


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1")
    parser.add_argument(
        "--n_instances", type=int, default=50, help="Instances per game (default: 50)"
    )
    parser.add_argument(
        "--games",
        nargs="+",
        default=list(GAME_RUNNERS.keys()),
        choices=list(GAME_RUNNERS.keys()),
        help="Which games to run",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Quick test with 2 instances"
    )
    args = parser.parse_args()

    n = 2 if args.dry_run else args.n_instances

    # Setup client
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    runner, get_response, _ = create_belief_aware_runner(
        client=client,
        model=MODEL,
        system_message="You are a helpful assistant.",
        time_limit=60,
    )

    print(f"Experiment 1: Belief-Action Consistency")
    print(f"Model: {MODEL}")
    print(f"Instances per game: {n}")
    print(f"Games: {args.games}")

    results = {}
    for game in args.games:
        filepath, records = GAME_RUNNERS[game](runner, get_response, n)
        results[game] = {"filepath": filepath, "n_instances": n}

    print(f"\n{'='*60}")
    print("EXPERIMENT 1 COMPLETE")
    print(f"{'='*60}")
    for game, info in results.items():
        print(f"  {game}: {info['filepath']}")


if __name__ == "__main__":
    main()
