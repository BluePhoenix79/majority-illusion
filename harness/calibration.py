"""Model-specific post-hoc confidence calibration.

The collector obtains one raw 0-100 probability after the modal answer has
been selected.  This module applies Platt scaling using only that probability;
self-consistency is intentionally not a calibration feature, so behavioral
stability remains a separate diagnostic as required by the study design.

For rows from a completed experiment, the reported ``calibrated_confidence``
is leave-one-entity-out (LOEO): every entity is calibrated by a model trained on
the other entities.  Holding out an entire entity prevents its other ratios
from leaking into its own calibration fold.
"""

from __future__ import annotations

import math
from collections import defaultdict


EPSILON = 0.005
L2_PENALTY = 0.01
MAX_ITERATIONS = 100


def _clip_probability(value):
    return min(max(float(value), EPSILON), 1.0 - EPSILON)


def _logit(value):
    p = _clip_probability(value)
    return math.log(p / (1.0 - p))


def _sigmoid(value):
    if value >= 0:
        exp_neg = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(max(value, -700.0))
    return exp_pos / (1.0 + exp_pos)


def _loss(scores, labels, slope, intercept):
    loss = 0.5 * L2_PENALTY * slope * slope
    for score, label in zip(scores, labels):
        probability = min(
            max(_sigmoid(slope * score + intercept), 1e-12),
            1.0 - 1e-12,
        )
        loss -= label * math.log(probability)
        loss -= (1 - label) * math.log(1.0 - probability)
    return loss


def fit_platt(probabilities, labels):
    """Fit ``sigmoid(slope * logit(p) + intercept)`` via damped Newton steps.

    Returns ``None`` when the training fold cannot identify a calibration
    curve (fewer than two observations or only one correctness class).
    """
    probabilities = [float(value) for value in probabilities]
    labels = [int(value) for value in labels]
    if len(labels) < 2 or len(set(labels)) < 2:
        return None

    scores = [_logit(value) for value in probabilities]
    positives = sum(labels)
    smoothed_rate = (positives + 1.0) / (len(labels) + 2.0)
    slope = 0.0
    intercept = _logit(smoothed_rate)

    for _ in range(MAX_ITERATIONS):
        grad_slope = L2_PENALTY * slope
        grad_intercept = 0.0
        h_ss = L2_PENALTY
        h_si = 0.0
        h_ii = 1e-12
        for score, label in zip(scores, labels):
            probability = _sigmoid(slope * score + intercept)
            residual = probability - label
            weight = max(probability * (1.0 - probability), 1e-9)
            grad_slope += residual * score
            grad_intercept += residual
            h_ss += weight * score * score
            h_si += weight * score
            h_ii += weight

        determinant = h_ss * h_ii - h_si * h_si
        if abs(determinant) < 1e-12:
            break
        delta_slope = (grad_slope * h_ii - grad_intercept * h_si) / determinant
        delta_intercept = (h_ss * grad_intercept - h_si * grad_slope) / determinant
        if max(abs(delta_slope), abs(delta_intercept)) < 1e-8:
            break

        current_loss = _loss(scores, labels, slope, intercept)
        step = 1.0
        accepted = False
        while step >= 1e-4:
            candidate_slope = slope - step * delta_slope
            candidate_intercept = intercept - step * delta_intercept
            if _loss(
                scores, labels, candidate_slope, candidate_intercept
            ) <= current_loss:
                slope = candidate_slope
                intercept = candidate_intercept
                accepted = True
                break
            step /= 2.0
        if not accepted:
            break

    return slope, intercept


def predict_platt(raw_probability, parameters):
    slope, intercept = parameters
    return _sigmoid(slope * _logit(raw_probability) + intercept)


def _as_probability(row):
    value = row.get("posthoc_probability", "")
    if value in (None, ""):
        return None
    number = float(value)
    if not 0.0 <= number <= 100.0:
        return None
    return number / 100.0


def _as_label(row):
    value = row.get("modal_correct", "")
    if str(value) not in {"0", "1"}:
        return None
    return int(value)


def calibrate_condition_rows(rows):
    """Mutate condition rows with model-specific LOEO Platt predictions.

    Returns a compact per-model manifest containing full-data parameters and
    cross-validated Brier scores.  The full-data parameters are useful for
    applying the learned mapping to a later held-out run; they are *not* used
    to score the same rows on which they were fit.
    """
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        row["calibrated_confidence"] = ""
        row["calibration_method"] = ""
        row["calibration_status"] = ""
        row["platt_slope_full"] = ""
        row["platt_intercept_full"] = ""
        probability = _as_probability(row)
        label = _as_label(row)
        if probability is None or label is None:
            row["calibration_status"] = "missing_probability_or_truth"
            continue
        grouped[str(row.get("model_id", ""))].append(
            (index, row, probability, label)
        )

    manifest = {}
    for model_id, items in grouped.items():
        all_probabilities = [item[2] for item in items]
        all_labels = [item[3] for item in items]
        full_parameters = fit_platt(all_probabilities, all_labels)
        unique_entities = {str(item[1].get("entity_id", "")) for item in items}

        if full_parameters is not None:
            full_slope, full_intercept = full_parameters
            for _, row, _, _ in items:
                row["platt_slope_full"] = round(full_slope, 8)
                row["platt_intercept_full"] = round(full_intercept, 8)

        calibrated_pairs = []
        for _, row, raw_probability, label in items:
            held_out = str(row.get("entity_id", ""))
            train = [
                item for item in items
                if str(item[1].get("entity_id", "")) != held_out
            ]
            parameters = fit_platt(
                [item[2] for item in train],
                [item[3] for item in train],
            )
            if len(unique_entities) < 3:
                row["calibration_status"] = "need_at_least_3_entities"
            elif parameters is None:
                row["calibration_status"] = "training_fold_has_one_class"
            else:
                calibrated = predict_platt(raw_probability, parameters)
                row["calibrated_confidence"] = round(calibrated * 100.0, 2)
                row["calibration_method"] = "model_specific_platt_loeo_entity"
                row["calibration_status"] = "ok"
                calibrated_pairs.append((raw_probability, calibrated, label))

        if calibrated_pairs:
            raw_brier = sum(
                (raw - label) ** 2 for raw, _, label in calibrated_pairs
            ) / len(calibrated_pairs)
            calibrated_brier = sum(
                (calibrated - label) ** 2
                for _, calibrated, label in calibrated_pairs
            ) / len(calibrated_pairs)
        else:
            raw_brier = calibrated_brier = None
        manifest[model_id] = {
            "n_conditions": len(items),
            "n_entities": len(unique_entities),
            "full_slope": full_parameters[0] if full_parameters else None,
            "full_intercept": full_parameters[1] if full_parameters else None,
            "loeo_scored": len(calibrated_pairs),
            "raw_brier": raw_brier,
            "calibrated_brier": calibrated_brier,
        }
    return manifest
