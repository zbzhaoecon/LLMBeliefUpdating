"""
Experiment 1 Analysis: Belief-Action Consistency
=================================================

Analysis functions for testing whether elicited beliefs predict LLM actions.

Usage:
    python experiments/analysis.py experiments/data/exp1_stag_hunt_*.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# GAME-AGNOSTIC UTILITIES
# =============================================================================

def compute_bacr(
    actions: List[str],
    beliefs: List[float],
    optimal_action_fn,
) -> Dict[str, Any]:
    """
    Belief-Action Consistency Rate.

    Args:
        actions: List of observed actions (e.g., ['Stag', 'Hare', ...])
        beliefs: List of belief values (e.g., P(Stag) for each observation)
        optimal_action_fn: Function(belief) -> predicted optimal action string

    Returns:
        Dict with rate, n_consistent, n_total, details
    """
    n_consistent = 0
    n_total = 0
    details = []
    for a, b in zip(actions, beliefs):
        if b is None:
            continue
        predicted = optimal_action_fn(b)
        consistent = (a == predicted)
        n_consistent += int(consistent)
        n_total += 1
        details.append({
            "action": a, "belief": b,
            "predicted": predicted, "consistent": consistent,
        })
    rate = n_consistent / n_total if n_total > 0 else None
    return {"rate": rate, "n_consistent": n_consistent, "n_total": n_total, "details": details}


def fit_logistic(
    actions: List[str],
    beliefs: List[float],
    positive_label: str,
) -> Dict[str, Any]:
    """
    Logistic regression: P(action=positive_label) = logistic(beta0 + beta1*belief).

    Returns dict with beta0, beta1, threshold, p_value, lambda_qre.
    """
    from scipy.special import expit
    from scipy.optimize import minimize

    y = np.array([1.0 if a == positive_label else 0.0 for a in actions])
    x = np.array(beliefs, dtype=float)

    # Remove NaN
    mask = ~np.isnan(x)
    y, x = y[mask], x[mask]

    if len(y) < 5:
        return {"error": "Too few observations", "n": len(y)}

    # MLE for logistic regression
    def neg_log_likelihood(params):
        b0, b1 = params
        p = expit(b0 + b1 * x)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize(neg_log_likelihood, [0.0, 1.0], method="Nelder-Mead")
    beta0, beta1 = result.x

    # Threshold: -beta0/beta1 (where P = 0.5)
    threshold = -beta0 / beta1 if abs(beta1) > 1e-8 else None

    # Hessian for p-value (approximate via finite differences)
    from scipy.optimize import approx_fprime
    eps = 1e-5
    H = np.zeros((2, 2))
    for i in range(2):
        def grad_i(params):
            return approx_fprime(params, neg_log_likelihood, eps)[i]
        H[i] = approx_fprime(result.x, grad_i, eps)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
        z_stat = beta1 / se[1] if se[1] > 0 else 0
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    except np.linalg.LinAlgError:
        se = [None, None]
        p_value = None

    return {
        "beta0": float(beta0),
        "beta1": float(beta1),
        "threshold": float(threshold) if threshold is not None else None,
        "p_value": p_value,
        "n": len(y),
        "mean_y": float(np.mean(y)),
        "mean_x": float(np.mean(x)),
    }


def compute_calibration(beliefs: List[float], true_rate: float) -> Dict[str, float]:
    """Compare mean(beliefs) to known true rate."""
    vals = [b for b in beliefs if b is not None]
    if not vals:
        return {"error": "No valid beliefs"}
    mean_b = float(np.mean(vals))
    return {
        "mean_belief": mean_b,
        "true_rate": true_rate,
        "bias": mean_b - true_rate,
        "abs_bias": abs(mean_b - true_rate),
        "std": float(np.std(vals)),
        "n": len(vals),
    }


def binned_analysis(
    actions: List[str],
    beliefs: List[float],
    positive_label: str,
    bins: List[Tuple[float, float]] = None,
) -> List[Dict]:
    """Fraction choosing positive_label in each belief bin."""
    if bins is None:
        bins = [(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)]

    results = []
    for lo, hi in bins:
        in_bin = [
            (a, b) for a, b in zip(actions, beliefs)
            if b is not None and lo <= b < hi
        ]
        n = len(in_bin)
        n_pos = sum(1 for a, _ in in_bin if a == positive_label)
        results.append({
            "bin": f"[{lo:.2f}, {hi:.2f})",
            "n": n,
            "n_positive": n_pos,
            "rate": n_pos / n if n > 0 else None,
        })
    return results


# =============================================================================
# GAME-SPECIFIC ANALYSIS
# =============================================================================

def _extract_per_round_beliefs_and_actions(records, belief_key_template, outcome_key):
    """Extract (action, belief_value) pairs across all instances and rounds."""
    actions = []
    beliefs = []
    for i, (choices, belief_list) in enumerate(
        zip(records["choices"], records["beliefs"])
    ):
        for r, (choice, belief_rec) in enumerate(zip(choices, belief_list)):
            bd = belief_rec.get("beliefs", {})
            key = belief_key_template.format(r=r + 1)
            val = bd.get(key, {})
            if isinstance(val, dict):
                b = val.get(outcome_key)
            else:
                b = val
            actions.append(choice)
            beliefs.append(b)
    return actions, beliefs


def analyze_stag_hunt(records: Dict) -> Dict[str, Any]:
    """
    Stag Hunt analysis:
    1. BACR with threshold p*=2/3
    2. Logistic regression
    3. Binned analysis
    4. Calibration vs true opponent rate (0.5)
    """
    actions, beliefs = _extract_per_round_beliefs_and_actions(
        records, "partner_action_r{r}", "Stag"
    )

    # BACR: Stag if P(Stag) > 2/3
    bacr = compute_bacr(
        actions, beliefs,
        lambda b: "Stag" if b > 2 / 3 else "Hare",
    )

    # Logistic regression
    valid = [(a, b) for a, b in zip(actions, beliefs) if b is not None]
    logistic = fit_logistic(
        [a for a, _ in valid],
        [b for _, b in valid],
        "Stag",
    )

    # Binned analysis
    bins = binned_analysis(actions, beliefs, "Stag")

    # Calibration (opponent is RANDOM(0.5))
    cal = compute_calibration(beliefs, 0.5)

    return {
        "game": "stag_hunt",
        "bacr": bacr,
        "logistic": logistic,
        "binned": bins,
        "calibration": cal,
        "n_observations": len(actions),
    }


def analyze_pd_infinite(records: Dict, delta: float = 0.9) -> Dict[str, Any]:
    """
    Infinite PD analysis:
    1. Compute p* threshold from delta and payoffs
    2. BACR using P(cooperative type) vs Push/Pull
    3. Bayesian updating test
    4. Correlation P(Push) vs Push choice
    """
    # Theoretical threshold: p* = 3(1-delta) / delta for delta > 3/4
    # (cooperation optimal against TFT when delta > 3/4 regardless of p)
    if delta > 3 / 4:
        p_star = 0.0  # cooperation always optimal against TFT for any p>0
        note = f"delta={delta} > 3/4: cooperation optimal for any p>0 against TFT"
    else:
        p_star = 3 * (1 - delta) / delta
        note = f"delta={delta}: cooperation threshold p*={p_star:.3f}"

    # Extract type beliefs and action beliefs
    type_actions = []
    type_beliefs = []
    action_actions = []
    action_beliefs = []
    updating_data = []  # For Bayesian updating test

    for i, (choices, belief_list, opp_actions) in enumerate(
        zip(records["choices"], records["beliefs"], records["opponent_actions"])
    ):
        prev_type_belief = None
        for r, (choice, belief_rec) in enumerate(zip(choices, belief_list)):
            bd = belief_rec.get("beliefs", {})

            # Type belief
            type_key = f"opponent_type_r{r + 1}"
            type_val = bd.get(type_key, {})
            p_coop = type_val.get("Cooperative") if isinstance(type_val, dict) else None

            # Action belief
            action_key = f"opponent_action_r{r + 1}"
            action_val = bd.get(action_key, {})
            p_push = action_val.get("Push") if isinstance(action_val, dict) else None

            type_actions.append(choice)
            type_beliefs.append(p_coop)
            action_actions.append(choice)
            action_beliefs.append(p_push)

            # Bayesian updating: track belief change after opponent action
            if prev_type_belief is not None and r > 0 and r - 1 < len(opp_actions):
                opp_action = opp_actions[r - 1]
                updating_data.append({
                    "round": r + 1,
                    "prev_belief": prev_type_belief,
                    "curr_belief": p_coop,
                    "opponent_action": opp_action,
                })
            prev_type_belief = p_coop

    # BACR using type belief
    # With delta=0.9 > 3/4, cooperation is optimal for any p>0
    bacr_type = compute_bacr(
        type_actions, type_beliefs,
        lambda b: "Push" if b > p_star else "Pull",
    )

    # Logistic on type belief
    valid_type = [(a, b) for a, b in zip(type_actions, type_beliefs) if b is not None]
    logistic_type = fit_logistic(
        [a for a, _ in valid_type],
        [b for _, b in valid_type],
        "Push",
    ) if len(valid_type) >= 5 else {"error": "Too few observations"}

    # Logistic on action belief
    valid_action = [(a, b) for a, b in zip(action_actions, action_beliefs) if b is not None]
    logistic_action = fit_logistic(
        [a for a, _ in valid_action],
        [b for _, b in valid_action],
        "Push",
    ) if len(valid_action) >= 5 else {"error": "Too few observations"}

    # Bayesian updating test: does P(coop) increase after opponent Push?
    updating_result = {"n": 0}
    if updating_data:
        after_push = [d for d in updating_data if d["opponent_action"] == "Push"
                      and d["curr_belief"] is not None and d["prev_belief"] is not None]
        after_pull = [d for d in updating_data if d["opponent_action"] == "Pull"
                      and d["curr_belief"] is not None and d["prev_belief"] is not None]
        updating_result = {
            "after_push": {
                "n": len(after_push),
                "mean_change": float(np.mean([d["curr_belief"] - d["prev_belief"] for d in after_push])) if after_push else None,
            },
            "after_pull": {
                "n": len(after_pull),
                "mean_change": float(np.mean([d["curr_belief"] - d["prev_belief"] for d in after_pull])) if after_pull else None,
            },
        }

    # Calibration
    cal_type = compute_calibration(type_beliefs, 0.5)
    cal_action = compute_calibration(action_beliefs, 0.5)

    return {
        "game": "pd_infinite",
        "delta": delta,
        "p_star": p_star,
        "note": note,
        "bacr_type": bacr_type,
        "logistic_type": logistic_type,
        "logistic_action": logistic_action,
        "bayesian_updating": updating_result,
        "calibration_type": cal_type,
        "calibration_action": cal_action,
        "n_observations": len(type_actions),
    }


def analyze_beauty_contest(
    records: Dict, p: float = 2 / 3, n_players: int = 3
) -> Dict[str, Any]:
    """
    Beauty Contest analysis:
    1. Regression: c = alpha + beta*mu (theory: alpha~0, beta~4/7)
    2. Level-k classification
    3. Best-response consistency
    """
    choices_flat = []
    beliefs_flat = []
    for i, (choices, belief_list) in enumerate(
        zip(records["choices"], records["beliefs"])
    ):
        for r, (choice, belief_rec) in enumerate(zip(choices, belief_list)):
            bd = belief_rec.get("beliefs", {})
            key = f"expected_others_avg_r{r + 1}"
            mu = bd.get(key)
            if mu is not None:
                choices_flat.append(float(choice))
                beliefs_flat.append(float(mu))

    if len(choices_flat) < 3:
        return {"game": "beauty_contest", "error": "Too few observations"}

    c = np.array(choices_flat)
    mu = np.array(beliefs_flat)

    # OLS regression: c = alpha + beta*mu
    X = np.column_stack([np.ones_like(mu), mu])
    beta_hat, residuals, _, _ = np.linalg.lstsq(X, c, rcond=None)
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    ss_res = float(np.sum((c - X @ beta_hat) ** 2))
    ss_tot = float(np.sum((c - np.mean(c)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Theoretical best response coefficient
    beta_theory = p * (n_players - 1) / (n_players - p)

    # Best-response deviation
    c_star = beta_theory * mu  # best response to each belief
    mad = float(np.mean(np.abs(c - c_star)))

    # Level-k classification from beliefs
    level_k_beliefs = {
        1: 50.0,
        2: 50.0 * beta_theory,
        3: 50.0 * beta_theory ** 2,
        4: 50.0 * beta_theory ** 3,
    }
    level_counts = {k: 0 for k in level_k_beliefs}
    for m in mu:
        # Assign to closest level
        closest_level = min(
            level_k_beliefs.keys(),
            key=lambda k: abs(m - level_k_beliefs[k]),
        )
        level_counts[closest_level] += 1

    return {
        "game": "beauty_contest",
        "regression": {
            "alpha": alpha,
            "beta": beta,
            "beta_theory": float(beta_theory),
            "r_squared": r_squared,
            "n": len(c),
        },
        "best_response": {
            "mean_abs_deviation": mad,
        },
        "level_k": {
            "reference_beliefs": {k: float(v) for k, v in level_k_beliefs.items()},
            "counts": level_counts,
        },
    }


def analyze_auction(records: Dict) -> Dict[str, Any]:
    """
    Auction analysis:
    1. Regression: bid = alpha + beta1*valuation + beta2*E[opp bid]
    2. Bid shading ratio
    """
    bids = []
    valuations = []
    e_opp_bids = []

    for i, (outcome, belief_list) in enumerate(
        zip(records["outcomes"], records["beliefs"])
    ):
        v = outcome["my_valuation"]
        b = outcome["my_bid"]
        e_opp = None
        if belief_list:
            bd = belief_list[0].get("beliefs", {})
            e_opp = bd.get("expected_opponent_bid")
        if e_opp is not None:
            bids.append(b)
            valuations.append(v)
            e_opp_bids.append(e_opp)

    if len(bids) < 3:
        return {"game": "auction", "error": "Too few observations"}

    bids = np.array(bids)
    vals = np.array(valuations)
    e_opp = np.array(e_opp_bids)

    # OLS: bid = alpha + beta1*v + beta2*E[opp]
    X = np.column_stack([np.ones_like(vals), vals, e_opp])
    beta_hat, _, _, _ = np.linalg.lstsq(X, bids, rcond=None)
    alpha, beta1, beta2 = [float(x) for x in beta_hat]
    ss_res = float(np.sum((bids - X @ beta_hat) ** 2))
    ss_tot = float(np.sum((bids - np.mean(bids)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Bid shading
    shading = bids / vals
    shading = shading[vals > 5]  # exclude very low valuations

    return {
        "game": "auction",
        "regression": {
            "alpha": alpha,
            "beta_valuation": beta1,
            "beta_valuation_theory": 0.5,
            "beta_e_opp_bid": beta2,
            "r_squared": r_squared,
            "n": len(bids),
        },
        "bid_shading": {
            "mean_ratio": float(np.mean(shading)) if len(shading) > 0 else None,
            "std_ratio": float(np.std(shading)) if len(shading) > 0 else None,
            "theory_ratio": 0.5,
            "n": len(shading),
        },
    }


def analyze_ultimatum(records: Dict) -> Dict[str, Any]:
    """
    Ultimatum analysis:
    1. Compute V(x) = (100-x)*P(accept|x) from belief schedule
    2. Find implied optimal offer
    3. Compare to actual offer
    """
    offer_levels = [10, 20, 30, 40, 50]

    actual_offers = []
    optimal_offers = []
    belief_schedules = []

    for i, (msgs, belief_list) in enumerate(
        zip(records["messages"], records["beliefs"])
    ):
        # Extract actual offer from last assistant message
        last_msg = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else ""
        import re
        match = re.search(r'\$(\d+)', last_msg)
        if not match:
            continue
        actual_offer = float(match.group(1))

        # Extract acceptance beliefs
        schedule = {}
        for belief_rec in belief_list:
            bd = belief_rec.get("beliefs", {})
            for x in offer_levels:
                key = f"accept_prob_{x}"
                if key in bd and bd[key] is not None:
                    schedule[x] = bd[key]

        if len(schedule) < 3:
            continue

        # Compute V(x) = (100-x)*P(accept|x)
        best_v = -1
        best_x = None
        for x, prob in schedule.items():
            v = (100 - x) * prob
            if v > best_v:
                best_v = v
                best_x = x

        actual_offers.append(actual_offer)
        optimal_offers.append(best_x)
        belief_schedules.append(schedule)

    if not actual_offers:
        return {"game": "ultimatum", "error": "No valid observations"}

    actual = np.array(actual_offers)
    optimal = np.array(optimal_offers, dtype=float)

    # Correlation
    corr = float(np.corrcoef(actual, optimal)[0, 1]) if len(actual) > 2 else None

    # Mean absolute deviation
    mad = float(np.mean(np.abs(actual - optimal)))

    # Mean acceptance belief schedule
    mean_schedule = {}
    for x in offer_levels:
        vals = [s.get(x) for s in belief_schedules if x in s]
        if vals:
            mean_schedule[x] = float(np.mean(vals))

    return {
        "game": "ultimatum",
        "n": len(actual_offers),
        "actual_offers": {
            "mean": float(np.mean(actual)),
            "std": float(np.std(actual)),
        },
        "optimal_offers": {
            "mean": float(np.mean(optimal)),
            "std": float(np.std(optimal)),
        },
        "consistency": {
            "correlation": corr,
            "mean_abs_deviation": mad,
            "exact_match_rate": float(np.mean(actual == optimal)),
        },
        "mean_acceptance_schedule": mean_schedule,
    }


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_report(results: Dict[str, Dict]) -> str:
    """Generate a summary report across all games."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT 1: BELIEF-ACTION CONSISTENCY REPORT")
    lines.append("=" * 70)

    if "stag_hunt" in results:
        r = results["stag_hunt"]
        lines.append(f"\nSTAG HUNT (threshold p*=2/3)")
        lines.append(f"  BACR:        {r['bacr']['rate']:.3f} ({r['bacr']['n_consistent']}/{r['bacr']['n_total']})")
        lg = r["logistic"]
        if "error" not in lg:
            lines.append(f"  Logistic:    beta1={lg['beta1']:.3f}, threshold={lg['threshold']:.3f}, p={lg['p_value']:.4f}" if lg['p_value'] else f"  Logistic:    beta1={lg['beta1']:.3f}, threshold={lg['threshold']:.3f}")
        cal = r["calibration"]
        lines.append(f"  Calibration: mean_belief={cal['mean_belief']:.3f}, true=0.5, bias={cal['bias']:.3f}")
        lines.append(f"  Binned:")
        for b in r["binned"]:
            lines.append(f"    {b['bin']}: {b['rate']:.2f} Stag ({b['n']} obs)" if b["rate"] is not None else f"    {b['bin']}: no obs")

    if "pd_infinite" in results:
        r = results["pd_infinite"]
        lines.append(f"\nPRISONER'S DILEMMA (delta={r['delta']}, {r['note']})")
        lines.append(f"  BACR (type):  {r['bacr_type']['rate']:.3f} ({r['bacr_type']['n_consistent']}/{r['bacr_type']['n_total']})")
        bu = r.get("bayesian_updating", {})
        if "after_push" in bu and bu["after_push"]["n"] > 0:
            lines.append(f"  Updating after Push: mean_change={bu['after_push']['mean_change']:+.3f} (n={bu['after_push']['n']})")
        if "after_pull" in bu and bu["after_pull"]["n"] > 0:
            lines.append(f"  Updating after Pull: mean_change={bu['after_pull']['mean_change']:+.3f} (n={bu['after_pull']['n']})")

    if "beauty_contest" in results:
        r = results["beauty_contest"]
        lines.append(f"\nBEAUTY CONTEST (p=2/3, 3 players)")
        reg = r["regression"]
        lines.append(f"  Regression: c = {reg['alpha']:.2f} + {reg['beta']:.3f}*mu (theory: 0 + {reg['beta_theory']:.3f}*mu)")
        lines.append(f"  R-squared:  {reg['r_squared']:.3f}")
        lines.append(f"  Best-response MAD: {r['best_response']['mean_abs_deviation']:.2f}")
        lines.append(f"  Level-k counts: {r['level_k']['counts']}")

    if "auction" in results:
        r = results["auction"]
        lines.append(f"\nFIRST-PRICE AUCTION")
        if "error" in r:
            lines.append(f"  Error: {r['error']}")
        else:
            reg = r["regression"]
            lines.append(f"  Regression: bid = {reg['alpha']:.2f} + {reg['beta_valuation']:.3f}*v + {reg['beta_e_opp_bid']:.3f}*E[opp]")
            lines.append(f"  beta_valuation: {reg['beta_valuation']:.3f} (theory: 0.5)")
            bs = r["bid_shading"]
            lines.append(f"  Bid/Valuation: {bs['mean_ratio']:.3f} (theory: 0.5)" if bs["mean_ratio"] else "  Bid/Valuation: N/A")

    if "ultimatum" in results:
        r = results["ultimatum"]
        lines.append(f"\nULTIMATUM PROPOSER")
        lines.append(f"  Actual offers: mean=${r['actual_offers']['mean']:.1f}, std=${r['actual_offers']['std']:.1f}")
        lines.append(f"  Optimal offers: mean=${r['optimal_offers']['mean']:.1f}")
        c = r["consistency"]
        corr_str = f"{c['correlation']:.3f}" if c['correlation'] is not None else "N/A"
        lines.append(f"  Consistency: corr={corr_str}, MAD=${c['mean_abs_deviation']:.1f}, exact_match={c['exact_match_rate']:.2f}")
        lines.append(f"  Acceptance schedule: {r['mean_acceptance_schedule']}")

    lines.append(f"\n{'='*70}")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def load_records(filepath: str) -> Dict:
    """Load records from JSON file."""
    with open(filepath) as f:
        return json.load(f)


ANALYZERS = {
    "stag_hunt": analyze_stag_hunt,
    "pd_infinite": analyze_pd_infinite,
    "beauty_contest": analyze_beauty_contest,
    "auction": analyze_auction,
    "ultimatum": analyze_ultimatum,
}


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 1 results")
    parser.add_argument("files", nargs="+", help="JSON record files to analyze")
    args = parser.parse_args()

    results = {}
    for filepath in args.files:
        records = load_records(filepath)
        # Detect game from filename or config
        game = records.get("config", "")
        if "stag_hunt" in filepath or game == "stag_hunt":
            results["stag_hunt"] = analyze_stag_hunt(records)
        elif "pd" in filepath or game == "prisoners_dilemma":
            results["pd_infinite"] = analyze_pd_infinite(records)
        elif "beauty" in filepath or game == "beauty_contest":
            results["beauty_contest"] = analyze_beauty_contest(records)
        elif "auction" in filepath or game == "first_price_auction":
            results["auction"] = analyze_auction(records)
        elif "ultimatum" in filepath or game == "ultimatum_proposer":
            results["ultimatum"] = analyze_ultimatum(records)
        else:
            print(f"Unknown game in {filepath}, skipping")

    print(generate_report(results))

    # Save analysis results
    out_path = os.path.join(
        os.path.dirname(args.files[0]),
        f"exp1_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
