import math

import numpy as np

from provrep.eval import metrics as M


def test_ranking_metrics_hand_computed():
    order = np.array([3, 0, 2, 1, 4])        # ranking of 5 candidates
    pos = {0, 1}
    assert M.recall_at_k(order, pos, 2) == 0.5
    assert M.precision_at_k(order, pos, 2) == 0.5
    # hits at ranks 2 and 4 -> AP = (1/2 + 2/4) / 2
    assert math.isclose(M.average_precision(order, pos), 0.5)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(M.ndcg_at_k(order, pos, 5), dcg / idcg)


def test_ranking_report_excludes_known_positives():
    scores = {7: np.array([0.9, 0.8, 0.1])}
    rep = M.ranking_report(scores, {7: {1}}, ks=(1,), exclude_by_query={7: {0}})
    assert rep["recall@1"] == 1.0 and rep["map"] == 1.0


def test_calibration_metrics():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 0, 1, 1])
    assert M.expected_calibration_error(p, y) < 0.11
    assert math.isclose(M.brier_score(p, y), 0.01)
    assert M.expected_calibration_error(np.full(4, 0.9), np.array([0, 0, 0, 0])) > 0.85


def test_auroc_with_ties():
    assert M.auroc(np.array([0.2, 0.8]), np.array([0, 1])) == 1.0
    assert M.auroc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
