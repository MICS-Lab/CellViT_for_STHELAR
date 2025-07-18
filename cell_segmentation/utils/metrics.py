# -*- coding: utf-8 -*-
# Implemented Metrics for Cell detection
#
# This code is based on the following repository: https://github.com/TissueImageAnalytics/PanNuke-metrics
#
# Implemented metrics are:
#
# Instance Segmentation Metrics
# Binary PQ
# Multiclass PQ
# Neoplastic PQ
# Non-Neoplastic PQ
# Inflammatory PQ
# Dead PQ
# Inflammatory PQ
# Dead PQ
#
# Detection and Classification Metrics
# Precision, Recall, F1
#
# Other
# dice1, dice2, aji, aji_plus
#
# Binary PQ (bPQ): Assumes all nuclei belong to same class and reports the average PQ across tissue types.
# Multi-Class PQ (mPQ): Reports the average PQ across the classes and tissue types.
# Neoplastic PQ: Reports the PQ for the neoplastic class on all tissues.
# Non-Neoplastic PQ: Reports the PQ for the non-neoplastic class on all tissues.
# Inflammatory PQ: Reports the PQ for the inflammatory class on all tissues.
# Connective PQ: Reports the PQ for the connective class on all tissues.
# Dead PQ: Reports the PQ for the dead class on all tissues.
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import List
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from collections import defaultdict
import re


def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    # if there is no background, fixing by adding it
    if 0 not in pred_id_list:
        pred_id_list = [0] + pred_id_list

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)  # good practice?
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


#####


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


####


def binarize(x):
    """
    convert multichannel (multiclass) instance segmetation tensor
    to binary instance segmentation (bg and nuclei),

    :param x: B*B*C (for PanNuke 256*256*5 )
    :return: Instance segmentation
    """
    out = np.zeros([x.shape[0], x.shape[1]])
    count = 1
    for i in range(x.shape[2]):
        x_ch = x[:, :, i]
        unique_vals = np.unique(x_ch)
        unique_vals = unique_vals.tolist()
        unique_vals.remove(0)
        for j in unique_vals:
            x_tmp = x_ch == j
            x_tmp_c = 1 - x_tmp
            out *= x_tmp_c
            out += count * x_tmp
            count += 1
    out = out.astype("int32")
    return out


def get_tissue_idx(tissue_indices, idx):
    for i in range(len(tissue_indices)):
        if tissue_indices[i].count(idx) == 1:
            tiss_idx = i
    return tiss_idx


def cell_detection_scores(
    paired_true, paired_pred, unpaired_true, unpaired_pred, w: List = [1, 1]
):
    tp_d = paired_pred.shape[0]
    fp_d = unpaired_pred.shape[0]
    fn_d = unpaired_true.shape[0]

    # tp_tn_dt = (paired_pred == paired_true).sum()
    # fp_fn_dt = (paired_pred != paired_true).sum()
    prec_d = tp_d / (tp_d + fp_d)
    rec_d = tp_d / (tp_d + fn_d)

    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    return f1_d, prec_d, rec_d


def cell_type_detection_scores(
    paired_true,
    paired_pred,
    unpaired_true,
    unpaired_pred,
    type_id,
    w: List = [2, 2, 1, 1],
    exhaustive: bool = True,
):
    type_samples = (paired_true == type_id) | (paired_pred == type_id)

    paired_true = paired_true[type_samples]
    paired_pred = paired_pred[type_samples]

    tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
    tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
    fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
    fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

    if not exhaustive:
        ignore = (paired_true == -1).sum()
        fp_dt -= ignore

    fp_d = (unpaired_pred == type_id).sum()  #
    fn_d = (unpaired_true == type_id).sum()

    prec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[0] * fp_dt + w[2] * fp_d)
    rec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[1] * fn_dt + w[3] * fn_d)

    f1_type = (2 * (tp_dt + tn_dt)) / (
        2 * (tp_dt + tn_dt) + w[0] * fp_dt + w[1] * fn_dt + w[2] * fp_d + w[3] * fn_d
    )
    return f1_type, prec_type, rec_type











def calculate_confusion_matrix(true_labels, pred_labels, class_names):
    """
    Calculate the confusion matrix and normalize it.
    
    Args:
        true_labels (np.ndarray): Ground truth labels (1D array).
        pred_labels (np.ndarray): Predicted labels (1D array).
        class_names (List[str]): List of class names for each label.

    Returns:
        np.ndarray: Normalized confusion matrix.
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized[np.isnan(cm_normalized)] = 0  # Avoid NaN for zero-row sums.
    return cm, cm_normalized


def log_confusion_matrix(logger, cm, cm_normalized, class_names):
    """
    Log the confusion matrix to the logger.

    Args:
        logger: Logger instance.
        cm (np.ndarray): Raw confusion matrix.
        cm_normalized (np.ndarray): Normalized confusion matrix.
        class_names (List[str]): List of class names for each label.
    """
    logger.info(f"{20*'*'} Confusion Matrix {20*'*'}")
    headers = ["True \\ Pred"] + class_names

    # # Log raw confusion matrix
    # raw_table = [[class_names[i]] + row.tolist() for i, row in enumerate(cm)]
    # logger.info("Raw Confusion Matrix:")
    # logger.info(tabulate(raw_table, headers=headers))

    # Log normalized confusion matrix
    normalized_table = [[class_names[i]] + row.tolist() for i, row in enumerate(cm_normalized)]
    logger.info("Normalized Confusion Matrix:")
    logger.info(tabulate(normalized_table, headers=headers))



def extract_slide_id(image_name):
    """
    Extracts the slide ID from an image name.
    Example: 'breast_s1_0.png' -> 'breast_s1'
    """
    return re.match(r"(.+?)_\d+\.png", image_name).group(1)



def compute_f1_per_slide(
    paired_all,
    unpaired_true_all,
    unpaired_pred_all,
    true_inst_type_all,
    pred_inst_type_all,
    paired_image_names_all,
    true_unpaired_image_names_all,
    pred_unpaired_image_names_all,
    nuclei_types,
    logger
):
    """
    Compute F1, precision, and recall scores for each nuclei type per slide.
    
    Args:
        paired_all (np.ndarray): Global indices of paired nuclei.
        unpaired_true_all (np.ndarray): Global indices of unpaired true nuclei.
        unpaired_pred_all (np.ndarray): Global indices of unpaired predicted nuclei.
        true_inst_type_all (np.ndarray): Global true nuclei type labels.
        pred_inst_type_all (np.ndarray): Global predicted nuclei type labels.
        paired_image_names_all (np.ndarray[str]): Image names of paired nuclei.
        true_unpaired_image_names_all (np.ndarray[str]): Image names of unpaired true nuclei.
        pred_unpaired_image_names_all (np.ndarray[str]): Image names of unpaired predicted nuclei.
        nuclei_types (Dict[str, int]): Mapping of nuclei type names to their IDs.
        logger: Logger instance for logging the results.
    
    Returns:
        Dict: Nested dictionary with F1, precision, and recall scores per nuclei type for each slide.
    """

    # Initialize metrics storage
    slide_metrics = {}

    # Convert all image names to their corresponding slide IDs
    paired_slide_ids = np.array([extract_slide_id(name) for name in paired_image_names_all])
    true_unpaired_slide_ids = np.array([extract_slide_id(name) for name in true_unpaired_image_names_all])
    pred_unpaired_slide_ids = np.array([extract_slide_id(name) for name in pred_unpaired_image_names_all])

    # Get unique slide IDs
    slide_ids = np.unique(paired_slide_ids)

    for slide_id in slide_ids:

        # Extract data for the current slide
        paired_indices_slide = paired_all[paired_slide_ids == slide_id]
        unpaired_true_indices_slide = unpaired_true_all[true_unpaired_slide_ids == slide_id]
        unpaired_pred_indices_slide = unpaired_pred_all[pred_unpaired_slide_ids == slide_id]
        
        paired_true_type_slide = true_inst_type_all[paired_indices_slide[:, 0]]
        paired_pred_type_slide = pred_inst_type_all[paired_indices_slide[:, 1]]
        unpaired_true_type_slide = true_inst_type_all[unpaired_true_indices_slide]
        unpaired_pred_type_slide = pred_inst_type_all[unpaired_pred_indices_slide]
        
        # Compute F1, precision, and recall for each nuclei type
        slide_metrics[slide_id] = {}
        for nuc_name, nuc_type in nuclei_types.items():
            if nuc_name.lower() == "background":
                continue
            
            f1_cell, prec_cell, rec_cell = cell_type_detection_scores(
                paired_true=paired_true_type_slide,
                paired_pred=paired_pred_type_slide,
                unpaired_true=unpaired_true_type_slide,
                unpaired_pred=unpaired_pred_type_slide,
                type_id=nuc_type,
            )
            
            slide_metrics[slide_id][nuc_name] = {
                "f1_cell": f1_cell,
                "prec_cell": prec_cell,
                "rec_cell": rec_cell,
            }
    
    # Log the results
    logger.info(f"\n\n{20*'*'} Nuclei Detection Metrics Per Slide {20*'*'}")
    for slide_id, metrics in slide_metrics.items():
        logger.info(f"{'-'*15}")
        logger.info(f"Slide: {slide_id}:")
        logger.info(f"{'-'*5}")
        table = [[nuc_name, data["prec_cell"], data["rec_cell"], data["f1_cell"]]
                 for nuc_name, data in metrics.items()]
        logger.info(tabulate(table, headers=["Nuclei Type", "Precision", "Recall", "F1"]))

    return slide_metrics