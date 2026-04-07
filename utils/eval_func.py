
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.global_variables import EMOTION_MAPPING

def get_pred_result(em_logits, ca_logits, couples_pred, emo_caus_pos, uttr_mask):
    em_logits = np.asarray(em_logits.cpu().detach())
    em_probs = np.exp(em_logits) / np.sum(np.exp(em_logits), axis=-1, keepdims=True)
    ca_probs = torch.sigmoid(ca_logits)

    predicted_emotions = np.argmax(em_probs, axis=-1) + 1 
    threshold = 0.5
    predicted_causes = (ca_probs >threshold).float()



    pred_labels = (torch.sigmoid(couples_pred) > threshold).float()

    pred_ec_pairs = []
    batch_size = couples_pred.size(0)
    for idx in range(batch_size):
        pred_ec_pairs_i = []
        for i, (pred_label, emo_caus) in enumerate(zip(pred_labels[idx], emo_caus_pos)):
            emo_idx, caus_idx = emo_caus
            if emo_idx <= uttr_mask[idx].sum().item() and caus_idx <= uttr_mask[idx].sum().item() and pred_label:
                if predicted_emotions[idx][emo_idx -1] != 7: 
                    pred_ec_pairs_i.append(emo_caus)
        pred_ec_pairs.append(pred_ec_pairs_i)

    return predicted_emotions, predicted_causes, pred_ec_pairs

def calc_eval_result(emotion_pred_all, emotion_list_all, cause_pred_all, cause_list_all, couples_pred_all, ec_pair_all):
    a = 1
    eval_result = {}
    for i in range(len(emotion_pred_all)):
        extra_length = len(emotion_list_all[i]) - len(emotion_pred_all[i])
        if extra_length > 0:
            emotion_pred_all[i].extend([7] * extra_length)
    flat_pred_emo = [pred for sublist in emotion_pred_all for pred in sublist]
    flat_true_emo = [true for sublist in emotion_list_all for true in sublist]

    flat_pred_cau = [pred for sublist in cause_pred_all for pred in sublist]
    flat_true_cau = [true for sublist in cause_list_all for true in sublist]

    flat_pred_emo = np.array(flat_pred_emo)
    flat_true_emo = np.array(flat_true_emo)

    flat_pred_cau = np.array(flat_pred_cau)
    flat_true_cau = np.array(flat_true_cau)

    valid_mask = flat_true_emo != 0

    valid_pred_emo = flat_pred_emo[valid_mask]
    valid_true_emo = flat_true_emo[valid_mask]

    gold_emo_bin = np.array([0 if w == 6 else 1 for w in flat_true_emo])
    pred_emo_bin = np.array([0 if w == 6 else 1 for w in flat_pred_emo])

    gold_emo_bin = gold_emo_bin[valid_mask]
    pred_emo_bin = pred_emo_bin[valid_mask]

    precision_ee = precision_score(gold_emo_bin, pred_emo_bin, average='binary')
    recall_ee = recall_score(gold_emo_bin, pred_emo_bin, average='binary')
    f1_ee = f1_score(gold_emo_bin, pred_emo_bin, average='binary')

    eval_result['precision_ee'] = precision_ee
    eval_result['recall_ee'] = recall_ee
    eval_result['f1_ee'] = f1_ee

    precision_emo = precision_score(valid_true_emo, valid_pred_emo, average='weighted') # macro
    recall_emo = recall_score(valid_true_emo, valid_pred_emo, average='weighted')
    f1_emo = f1_score(valid_true_emo, valid_pred_emo, average='weighted')
    eval_result['precision_emo'] = precision_emo
    eval_result['recall_emo'] = recall_emo
    eval_result['f1_emo'] = f1_emo

    valid_pred_cau = flat_pred_cau[valid_mask]
    valid_true_cau = flat_true_cau[valid_mask]

    precision_cau = precision_score(valid_true_cau, valid_pred_cau, average='weighted')
    recall_cau = recall_score(valid_true_cau, valid_pred_cau, average='weighted')
    f1_cau = f1_score(valid_true_cau, valid_pred_cau, average='weighted')
    eval_result['precision_cau'] = precision_cau
    eval_result['recall_cau'] = recall_cau
    eval_result['f1_cau'] = f1_cau

    emotions_6 = list(set(EMOTION_MAPPING['ECF'].values()))[:-1]
    emotions_to_remove = {'disgust', 'fear', 'neutral'}
    filtered_emotion_mapping = {k: v for k, v in EMOTION_MAPPING['ECF'].items() if k not in emotions_to_remove}
    emotions_4 = list(set(filtered_emotion_mapping.values()))

    ec_pair_all = [arr.tolist() for arr in ec_pair_all]
    precision_ec, recall_ec, f1_ec = calc_f1_by_list(ec_pair_all, couples_pred_all)
    eval_result['precision_ec'] = precision_ec
    eval_result['recall_ec'] = recall_ec
    eval_result['f1_ec'] = f1_ec
    (
        precision_ec_6,
        recall_ec_6,
        f1_ec_6,
        weighted_avg_f1_6,
        weighted_avg_p_6,
        weighted_avg_r_6
    ) = calc_f1_and_weighted_avg_ec_pairs(
        emotion_list_all,
        ec_pair_all,
        couples_pred_all,
        emotions_6
    )

    eval_result['precision_ec_per_class_6'] = precision_ec_6
    eval_result['recall_ec_per_class_6'] = recall_ec_6
    eval_result['f1_ec_per_class_6'] = f1_ec_6
    eval_result['weighted_avg_f1_6_ec'] = weighted_avg_f1_6
    eval_result['weighted_avg_p_6_ec'] = weighted_avg_p_6
    eval_result['weighted_avg_r_6_ec'] = weighted_avg_r_6

    (
        _,
        _,
        _,
        weighted_avg_f1_4,
        weighted_avg_p_4,
        weighted_avg_r_4
    ) = calc_f1_and_weighted_avg_ec_pairs(
        emotion_list_all,
        ec_pair_all,
        couples_pred_all,
        emotions_4
    )

    eval_result['weighted_avg_f1_4_ec'] = weighted_avg_f1_4
    eval_result['weighted_avg_p_4_ec'] = weighted_avg_p_4
    eval_result['weighted_avg_r_4_ec'] = weighted_avg_r_4

    triples_all_true = get_triples(emotion_list_all, ec_pair_all)
    triples_all_pred = get_triples(emotion_pred_all, couples_pred_all)
    precision_triple, recall_triple, f1_triple = calc_f1_by_list(triples_all_true, triples_all_pred)
    eval_result['precision_triple'] = precision_triple
    eval_result['recall_triple'] = recall_triple
    eval_result['f1_triple'] = f1_triple

    filtered_pred_triple = [[triple for triple in triples if triple[0] != EMOTION_MAPPING['ECF']['neutral']] for triples in triples_all_pred]
    filtered_true_triple = [[triple for triple in triples if triple[0] != EMOTION_MAPPING['ECF']['neutral']] for triples in triples_all_true]

    precision_per_class, recall_per_class, f1_per_class, weighted_avg_f1_6, weighted_avg_p_6, weighted_avg_r_6 = calc_weighted_avg_f1_by_list \
        (filtered_true_triple, filtered_pred_triple, emotions_6)
    eval_result['precision_per_class'] = precision_per_class
    eval_result['recall_per_class'] = recall_per_class
    eval_result['f1_per_class'] = f1_per_class
    eval_result['weighted_avg_f1_6'] = weighted_avg_f1_6
    eval_result['weighted_avg_p_6'] = weighted_avg_p_6
    eval_result['weighted_avg_r_6'] = weighted_avg_r_6


    _, _, _, weighted_avg_f1_4, weighted_avg_p_4, weighted_avg_r_4 = calc_weighted_avg_f1_by_list(filtered_true_triple, filtered_pred_triple, emotions_4)
    eval_result['weighted_avg_f1_4'] = weighted_avg_f1_4
    eval_result['weighted_avg_p_4'] = weighted_avg_p_4
    eval_result['weighted_avg_r_4'] = weighted_avg_r_4

    return eval_result


def calc_weighted_avg_f1_by_list(triples_all_true, triples_all_pred, emotions):
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    support_per_class = []

    for emotion in emotions:
        emotion_tp = 0
        emotion_fp = 0
        emotion_fn = 0
        emotion_support = 0

        for true_triples, pred_triples in zip(triples_all_true, triples_all_pred):
            true_emo_triples = [triple for triple in true_triples if triple[0] == emotion]
            pred_emo_triples = [triple for triple in pred_triples if triple[0] == emotion]

            TP = len([triple for triple in pred_emo_triples if triple in true_emo_triples])
            FP = len([triple for triple in pred_emo_triples if triple not in true_emo_triples])
            FN = len([triple for triple in true_emo_triples if triple not in pred_emo_triples])

            emotion_tp += TP
            emotion_fp += FP
            emotion_fn += FN
            emotion_support += len(true_emo_triples)

        precision = emotion_tp / (emotion_tp + emotion_fp) if (emotion_tp + emotion_fp) > 0 else 0
        recall = emotion_tp / (emotion_fn + emotion_tp) if (emotion_fn + emotion_tp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        support_per_class.append(emotion_support)

    weighted_avg_f1 = np.sum(np.array(f1_per_class) * np.array(support_per_class)) / np.sum(support_per_class)
    weighted_avg_p = np.sum(np.array(precision_per_class) * np.array(support_per_class)) / np.sum(support_per_class)
    weighted_avg_r = np.sum(np.array(recall_per_class) * np.array(support_per_class)) / np.sum(support_per_class)

    return precision_per_class, recall_per_class, f1_per_class, weighted_avg_f1, weighted_avg_p, weighted_avg_r

import numpy as np

def calc_f1_and_weighted_avg_ec_pairs(emotion_list_all, true_ec_pairs, pred_ec_pairs, emotions):
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    support_per_class = []

    for emo in emotions:

        tp = fp = fn = support = 0

        for emo_list, true_pairs, pred_pairs in zip(emotion_list_all, true_ec_pairs, pred_ec_pairs):
            true_cls_pairs = [
                p for p in true_pairs
                if (emo_list[p[0] - 1] - 1) == emo
            ]

            pred_cls_pairs = [
                p for p in pred_pairs
                if (emo_list[p[0] - 1] - 1) == emo
            ]

            tp += len([p for p in pred_cls_pairs if p in true_cls_pairs])
            fp += len([p for p in pred_cls_pairs if p not in true_cls_pairs])
            fn += len([p for p in true_cls_pairs if p not in pred_cls_pairs])
            support += len(true_cls_pairs)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        support_per_class.append(support)

    total_support = np.sum(support_per_class) if np.sum(support_per_class) > 0 else 1

    weighted_avg_f1 = np.sum(np.array(f1_per_class) * np.array(support_per_class)) / total_support
    weighted_avg_p  = np.sum(np.array(precision_per_class) * np.array(support_per_class)) / total_support
    weighted_avg_r  = np.sum(np.array(recall_per_class) * np.array(support_per_class)) / total_support

    return (
        precision_per_class,
        recall_per_class,
        f1_per_class,
        weighted_avg_f1,
        weighted_avg_p,
        weighted_avg_r
    )



def calc_f1_by_list(true_list, pred_list):
    TP = 0
    FP = 0
    FN = 0

    for true_pairs, pred_pairs in zip(true_list, pred_list): 

        for pair in pred_pairs:
            if pair in true_pairs:
                TP += 1
            else:
                FP += 1
        for pair in true_pairs:
            if pair not in pred_pairs:
                FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def get_triples(emotion_list_all, ec_pair_all):
    uttr_len = len(emotion_list_all)
    triples_all = []
    for i in range(uttr_len):
        ec_pairs = ec_pair_all[i]
        triples = []
        for pair in ec_pairs:
            emo_idx, caus_idx = pair
            emotion = emotion_list_all[i][emo_idx -1 ] -1 
            triple = [emotion, emo_idx, caus_idx]
            triples.append(triple)
        triples_all.append(triples)
    return triples_all



