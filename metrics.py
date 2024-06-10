from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


def get_mertic_score(srcs, pres, labs):
    """
    :param srcs: 原query
    :param pres: 预测query
    :param labs: label
    :return: precision, recall, f1
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    for src, pre, lab in zip(srcs, pres, labs):
        if src == lab:
            if lab == pre:
                TN += 1
            else:
                FP += 1
        else:
            if lab == pre:
                TP += 1
            else:
                FN += 1
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(
        f' Level correction: precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    return precision, recall, f1