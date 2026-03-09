"""
Experiment 2 Analysis: Causal Intervention
==========================================

Analysis functions for Experiment 2 sub-experiments:
  2A: Information Treatment (does info shift beliefs, do shifted beliefs shift actions?)
  2B: Direct Belief Injection (does injected belief produce EU-consistent actions?)
  2C: Fake History Priming (does fabricated history produce beliefs that predict actions?)

Usage:
    python experiments/analysis_exp2.py 2a experiments/data/exp2a_*.json
    python experiments/analysis_exp2.py 2b experiments/data/exp2b_*.json
    python experiments/analysis_exp2.py 2c experiments/data/exp2c_*.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# EXPERIMENT 2A: Information Treatment Analysis
# =============================================================================

def _extract_round1_belief_stag(records):
    """Extract round 1 P(Stag) belief from Stag Hunt records."""
    beliefs_list = records.get("beliefs", [])
    out = []
    for instance_beliefs in beliefs_list:
        if not instance_beliefs:
            out.append(None)
            continue
        r1 = instance_beliefs[0]
        b = r1.get("beliefs", {})
        key = [k for k in b if "partner_action_r1" in k]
        if key:
            val = b[key[0]]
            if isinstance(val, dict) and "Stag" in val:
                out.append(val["Stag"])
            else:
                out.append(None)
        else:
            out.append(None)
    return out


def _extract_round1_belief_push(records):
    """Extract round 1 P(Push) belief from PD records."""
    beliefs_list = records.get("beliefs", [])
    out = []
    for instance_beliefs in beliefs_list:
        if not instance_beliefs:
            out.append(None)
            continue
        r1 = instance_beliefs[0]
        b = r1.get("beliefs", {})
        # Look for opponent_action_r1 (action belief) or opponent_type_r1
        action_key = [k for k in b if "opponent_action_r1" in k]
        if action_key:
            val = b[action_key[0]]
            if isinstance(val, dict) and "Push" in val:
                out.append(val["Push"])
            else:
                out.append(None)
        else:
            out.append(None)
    return out


def _extract_round1_action(records, game):
    """Extract round 1 action from records."""
    if "choices" in records:
        return [c[0] if c else None for c in records["choices"]]
    elif "actions" in records:
        return records["actions"]
    return []


def analyze_exp2a(all_records_by_X: Dict[int, Dict]) -> Dict:
    """
    Analyze Experiment 2A: Information Treatment.

    Args:
        all_records_by_X: Dict mapping treatment X -> records dict.
            Each records dict should contain 'treatment_X'.

    Returns analysis with:
    - first_stage: regression of belief on X
    - reduced_form: regression of action on X
    - mediation: IV estimate
    - dose_response: action rates by X
    """
    X_vals = []
    beliefs = []
    actions_binary = []

    # Detect game type from first record
    sample = list(all_records_by_X.values())[0]
    is_stag = "stag_hunt" in sample.get("config", "")

    for X, records in sorted(all_records_by_X.items()):
        n = len(records.get("choices", records.get("actions", [])))

        if is_stag:
            b_list = _extract_round1_belief_stag(records)
            a_list = _extract_round1_action(records, "stag_hunt")
            coop_label = "Stag"
        else:
            b_list = _extract_round1_belief_push(records)
            a_list = _extract_round1_action(records, "pd")
            coop_label = "Push"

        for i in range(n):
            X_vals.append(X)
            beliefs.append(b_list[i] if i < len(b_list) else None)
            a = a_list[i] if i < len(a_list) else None
            actions_binary.append(1 if a == coop_label else 0)

    # Filter valid observations
    valid = [(x, b, a) for x, b, a in zip(X_vals, beliefs, actions_binary)
             if b is not None]

    if len(valid) < 5:
        return {"game": "stag_hunt" if is_stag else "pd", "error": "Too few valid observations"}

    X_arr = np.array([v[0] for v in valid], dtype=float)
    B_arr = np.array([v[1] for v in valid], dtype=float)
    A_arr = np.array([v[2] for v in valid], dtype=float)

    result = {
        "game": "stag_hunt" if is_stag else "pd",
        "n_total": len(valid),
        "treatments": sorted(all_records_by_X.keys()),
    }

    # First stage: b = gamma_0 + gamma_1 * X
    slope_b, intercept_b, r_b, p_b, se_b = stats.linregress(X_arr, B_arr)
    result["first_stage"] = {
        "gamma_0": float(intercept_b),
        "gamma_1": float(slope_b),
        "r_squared": float(r_b ** 2),
        "p_value": float(p_b),
        "se": float(se_b),
    }

    # Reduced form: a = delta_0 + delta_1 * X
    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(X_arr, A_arr)
    result["reduced_form"] = {
        "delta_0": float(intercept_a),
        "delta_1": float(slope_a),
        "r_squared": float(r_a ** 2),
        "p_value": float(p_a),
        "se": float(se_a),
    }

    # IV estimate: beta_IV = delta_1 / gamma_1
    if abs(slope_b) > 1e-10:
        beta_iv = slope_a / slope_b
        result["iv_estimate"] = {
            "beta_iv": float(beta_iv),
            "note": "Causal effect of belief on action (IV: X instruments b)",
        }
    else:
        result["iv_estimate"] = {"error": "Weak first stage (gamma_1 ~ 0)"}

    # Dose-response: action rate by X
    dose_response = {}
    for X in sorted(all_records_by_X.keys()):
        mask = X_arr == X
        if mask.sum() > 0:
            dose_response[int(X)] = {
                "coop_rate": float(A_arr[mask].mean()),
                "mean_belief": float(B_arr[mask].mean()),
                "n": int(mask.sum()),
            }
    result["dose_response"] = dose_response

    return result


# =============================================================================
# EXPERIMENT 2B: Direct Belief Injection Analysis
# =============================================================================

def analyze_exp2b(all_records_by_X: Dict[int, Dict], variant="standard") -> Dict:
    """
    Analyze Experiment 2B: Direct Belief Injection.

    Args:
        all_records_by_X: Dict mapping injected belief X% -> records dict.
        variant: "standard" (p*=2/3), "modified" (p*=1/2), or "pd".

    Returns:
    - action rates by X
    - logistic fit: P(cooperate) = logistic(beta_0 + beta_1 * X)
    - estimated threshold
    """
    X_vals = []
    actions_binary = []

    game = list(all_records_by_X.values())[0].get("game", "stag_hunt")
    coop_label = "Push" if game == "pd" else "Stag"

    for X, records in sorted(all_records_by_X.items()):
        action_list = records.get("actions", [])
        for a in action_list:
            X_vals.append(X)
            actions_binary.append(1 if a == coop_label else 0)

    X_arr = np.array(X_vals, dtype=float)
    A_arr = np.array(actions_binary, dtype=float)

    result = {
        "game": game,
        "variant": variant,
        "n_total": len(X_arr),
    }

    # Action rates by X
    action_rates = {}
    for X in sorted(all_records_by_X.keys()):
        mask = X_arr == X
        if mask.sum() > 0:
            action_rates[int(X)] = {
                "coop_rate": float(A_arr[mask].mean()),
                "n": int(mask.sum()),
            }
    result["action_rates"] = action_rates

    # Logistic regression: logit(P(coop)) = beta_0 + beta_1 * X
    if len(np.unique(A_arr)) < 2:
        result["logistic"] = {"error": "No variation in actions"}
        return result

    try:
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            b0, b1 = params
            z = b0 + b1 * X_arr
            z = np.clip(z, -500, 500)
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.sum(A_arr * np.log(p) + (1 - A_arr) * np.log(1 - p))

        res = minimize(neg_log_likelihood, [0.0, 0.0], method='Nelder-Mead')
        b0, b1 = res.x

        threshold = -b0 / b1 if abs(b1) > 1e-10 else None

        result["logistic"] = {
            "beta_0": float(b0),
            "beta_1": float(b1),
            "threshold": float(threshold) if threshold is not None else None,
            "converged": bool(res.success),
        }
    except Exception as e:
        result["logistic"] = {"error": str(e)}

    return result


# =============================================================================
# EXPERIMENT 2C: Fake History Priming Analysis
# =============================================================================

def analyze_exp2c(all_records_by_k: Dict[int, Dict]) -> Dict:
    """
    Analyze Experiment 2C: Fake History Priming.

    Args:
        all_records_by_k: Dict mapping k (# Stag in history) -> records dict.

    Returns:
    - belief_tracking: mean belief as function of k
    - action_rates: P(Stag) as function of k
    - belief_action_consistency: does b->a mapping match Exp1 pattern?
    - slope: beta from logistic P(Stag) = logistic(beta_0 + beta_1 * b)
    """
    k_vals = []
    beliefs = []
    actions_binary = []

    for k, records in sorted(all_records_by_k.items()):
        action_list = records.get("actions", [])
        belief_list = records.get("beliefs", [])

        for i in range(len(action_list)):
            k_vals.append(k)
            actions_binary.append(1 if action_list[i] == "Stag" else 0)

            # Extract belief
            if i < len(belief_list) and belief_list[i]:
                b = belief_list[i][0]  # first (and only) elicitation point
                b_dict = b.get("beliefs", {})
                stag_key = [key for key in b_dict if "partner_action" in key]
                if stag_key:
                    val = b_dict[stag_key[0]]
                    if isinstance(val, dict) and "Stag" in val:
                        beliefs.append(val["Stag"])
                    else:
                        beliefs.append(None)
                else:
                    beliefs.append(None)
            else:
                beliefs.append(None)

    k_arr = np.array(k_vals, dtype=float)
    A_arr = np.array(actions_binary, dtype=float)

    result = {
        "game": "stag_hunt_fake_history",
        "n_total": len(k_arr),
    }

    # Belief tracking: mean belief by k
    valid_beliefs = [(k, b) for k, b in zip(k_vals, beliefs) if b is not None]
    belief_tracking = {}
    for k in sorted(all_records_by_k.keys()):
        k_beliefs = [b for kk, b in valid_beliefs if kk == k]
        if k_beliefs:
            belief_tracking[int(k)] = {
                "mean_belief": float(np.mean(k_beliefs)),
                "std_belief": float(np.std(k_beliefs)),
                "n": len(k_beliefs),
            }
    result["belief_tracking"] = belief_tracking

    # Monotonicity test
    means = [belief_tracking[k]["mean_belief"] for k in sorted(belief_tracking.keys())]
    is_monotone = all(means[i] <= means[i+1] for i in range(len(means)-1))
    result["belief_monotone"] = is_monotone

    # Belief-to-k regression
    if valid_beliefs:
        k_b = np.array([v[0] for v in valid_beliefs], dtype=float)
        b_b = np.array([v[1] for v in valid_beliefs], dtype=float)
        slope, intercept, r, p, se = stats.linregress(k_b, b_b)
        result["belief_vs_k_regression"] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r**2),
            "p_value": float(p),
        }

    # Action rates by k
    action_rates = {}
    for k in sorted(all_records_by_k.keys()):
        mask = k_arr == k
        if mask.sum() > 0:
            action_rates[int(k)] = {
                "stag_rate": float(A_arr[mask].mean()),
                "n": int(mask.sum()),
            }
    result["action_rates"] = action_rates

    # Belief -> action logistic (using elicited beliefs)
    valid_ba = [(b, a) for b, a in zip(beliefs, actions_binary) if b is not None]
    if len(valid_ba) >= 5 and len(set(a for _, a in valid_ba)) >= 2:
        B_arr = np.array([v[0] for v in valid_ba], dtype=float)
        A_ba = np.array([v[1] for v in valid_ba], dtype=float)

        try:
            from scipy.optimize import minimize

            def neg_ll(params):
                b0, b1 = params
                z = b0 + b1 * B_arr
                z = np.clip(z, -500, 500)
                p = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return -np.sum(A_ba * np.log(p) + (1 - A_ba) * np.log(1 - p))

            res = minimize(neg_ll, [0.0, 0.0], method='Nelder-Mead')
            b0, b1 = res.x
            threshold = -b0 / b1 if abs(b1) > 1e-10 else None
            result["belief_action_logistic"] = {
                "beta_0": float(b0),
                "beta_1": float(b1),
                "threshold": float(threshold) if threshold is not None else None,
            }
        except Exception as e:
            result["belief_action_logistic"] = {"error": str(e)}

    return result


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_exp2_report(results: Dict) -> str:
    """Generate human-readable report for Experiment 2."""
    lines = [
        f"{'='*70}",
        "EXPERIMENT 2: CAUSAL INTERVENTION REPORT",
        f"{'='*70}",
    ]

    # 2A
    for key in ["2a_stag_hunt", "2a_pd"]:
        if key not in results:
            continue
        r = results[key]
        lines.append(f"\n2A - {r['game'].upper()} (Information Treatment)")
        lines.append(f"  N={r['n_total']}, treatments={r['treatments']}")

        fs = r.get("first_stage", {})
        lines.append(f"  First stage:  b = {fs.get('gamma_0', 0):.3f} + {fs.get('gamma_1', 0):.4f}*X  (p={fs.get('p_value', 1):.4f}, R2={fs.get('r_squared', 0):.3f})")

        rf = r.get("reduced_form", {})
        lines.append(f"  Reduced form: a = {rf.get('delta_0', 0):.3f} + {rf.get('delta_1', 0):.4f}*X  (p={rf.get('p_value', 1):.4f}, R2={rf.get('r_squared', 0):.3f})")

        iv = r.get("iv_estimate", {})
        if "beta_iv" in iv:
            lines.append(f"  IV estimate:  beta_IV = {iv['beta_iv']:.3f}")
        elif "error" in iv:
            lines.append(f"  IV estimate:  {iv['error']}")

        lines.append("  Dose-response:")
        for X, dr in sorted(r.get("dose_response", {}).items()):
            lines.append(f"    X={X}: coop_rate={dr['coop_rate']:.2f}, mean_belief={dr['mean_belief']:.3f} (n={dr['n']})")

    # 2B
    for key in ["2b_stag_standard", "2b_stag_modified", "2b_pd"]:
        if key not in results:
            continue
        r = results[key]
        lines.append(f"\n2B - {r['game'].upper()} ({r.get('variant', 'unknown')}) (Direct Injection)")
        lines.append(f"  N={r['n_total']}")

        lines.append("  Action rates by injected belief:")
        for X, ar in sorted(r.get("action_rates", {}).items()):
            lines.append(f"    X={X}%: coop_rate={ar['coop_rate']:.2f} (n={ar['n']})")

        lg = r.get("logistic", {})
        if "beta_1" in lg:
            lines.append(f"  Logistic: beta0={lg['beta_0']:.3f}, beta1={lg['beta_1']:.4f}, threshold={lg.get('threshold', 'N/A')}")

    # 2C
    if "2c" in results:
        r = results["2c"]
        lines.append(f"\n2C - FAKE HISTORY PRIMING")
        lines.append(f"  N={r['n_total']}, monotone={r.get('belief_monotone', 'N/A')}")

        lines.append("  Belief tracking:")
        for k, bt in sorted(r.get("belief_tracking", {}).items()):
            lines.append(f"    k={k}: mean_belief={bt['mean_belief']:.3f} +/- {bt['std_belief']:.3f} (n={bt['n']})")

        reg = r.get("belief_vs_k_regression", {})
        if reg:
            lines.append(f"  Belief vs k: slope={reg.get('slope', 0):.4f}, R2={reg.get('r_squared', 0):.3f}, p={reg.get('p_value', 1):.4f}")

        lines.append("  Action rates:")
        for k, ar in sorted(r.get("action_rates", {}).items()):
            lines.append(f"    k={k}: stag_rate={ar['stag_rate']:.2f} (n={ar['n']})")

        ba = r.get("belief_action_logistic", {})
        if "beta_1" in ba:
            lines.append(f"  Belief->Action logistic: beta1={ba['beta_1']:.4f}, threshold={ba.get('threshold', 'N/A')}")

    lines.append(f"\n{'='*70}")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def load_records(filepath: str) -> Dict:
    with open(filepath) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 2 results")
    parser.add_argument("sub", choices=["2a", "2b", "2c", "all"],
                        help="Which sub-experiment to analyze")
    parser.add_argument("files", nargs="+", help="JSON record files")
    args = parser.parse_args()

    results = {}

    if args.sub in ("2a", "all"):
        # Group files by game and treatment X
        stag_by_X = {}
        pd_by_X = {}
        for f in args.files:
            if "exp2a" not in f:
                continue
            records = load_records(f)
            X = records.get("treatment_X")
            if X is None:
                continue
            if "stag_hunt" in f:
                stag_by_X[X] = records
            elif "pd" in f:
                pd_by_X[X] = records
        if stag_by_X:
            results["2a_stag_hunt"] = analyze_exp2a(stag_by_X)
        if pd_by_X:
            results["2a_pd"] = analyze_exp2a(pd_by_X)

    if args.sub in ("2b", "all"):
        stag_standard_by_X = {}
        stag_modified_by_X = {}
        pd_by_X = {}
        for f in args.files:
            if "exp2b" not in f:
                continue
            records = load_records(f)
            X = records.get("injected_belief")
            if X is None:
                continue
            if "stag_hunt_standard" in f:
                stag_standard_by_X[X] = records
            elif "stag_hunt_modified" in f:
                stag_modified_by_X[X] = records
            elif "pd" in f:
                pd_by_X[X] = records
        if stag_standard_by_X:
            results["2b_stag_standard"] = analyze_exp2b(stag_standard_by_X, "standard (p*=2/3)")
        if stag_modified_by_X:
            results["2b_stag_modified"] = analyze_exp2b(stag_modified_by_X, "modified (p*=1/2)")
        if pd_by_X:
            results["2b_pd"] = analyze_exp2b(pd_by_X, "pd")

    if args.sub in ("2c", "all"):
        records_by_k = {}
        for f in args.files:
            if "exp2c" not in f:
                continue
            records = load_records(f)
            k = records.get("treatment_k")
            if k is not None:
                records_by_k[k] = records
        if records_by_k:
            results["2c"] = analyze_exp2c(records_by_k)

    print(generate_exp2_report(results))

    out_path = os.path.join(
        os.path.dirname(args.files[0]),
        f"exp2_analysis_{args.sub}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
