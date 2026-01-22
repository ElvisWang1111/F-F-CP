import pickle 
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse
import torch
import torch.nn.functional as F

import csv
from pathlib import Path

options = ["A", "B", "C", "D", "E", "F"]
ids_to_remove = [1, 3, 5, 7, 9] # remove data points that have been used as demonstration data

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_jacobian_and_norm(logits: torch.Tensor, T: float = 1.0):
    """
    Compute softmax Jacobian and gradient norms for each sample in a batch.

    Args:
        logits (Tensor): shape [B, C], raw logits.
        T (float): temperature scalar.

    Returns:
        jacobian (Tensor): shape [B, C, C]
        norms (Tensor): shape [B, C], L2 norm of each row of the Jacobian
    """
    # logits: [B, C]
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)  # [C] → [1, C]
    device = logits.device
    T = torch.tensor(T, dtype=logits.dtype, device=device)

    # softmax with temperature
    probs = F.softmax(logits / T, dim=1)  # [B, C]

    B, C = probs.shape

    # diag(s) - s s^T
    # expand for batch
    diag_s = torch.diag_embed(probs)  # [B, C, C]
    outer_s = probs.unsqueeze(2) @ probs.unsqueeze(1)  # [B, C, C]

    jacobian = (diag_s - outer_s) / T  # [B, C, C]

    # row-wise L2 norm: ||J[i, c, :]||_2
    norms = torch.norm(jacobian, dim=2)  # [B, C]
    jacobian = jacobian.detach().cpu().numpy()
    norms = norms.detach().cpu().numpy().squeeze(0)

    return jacobian, norms


def get_raw_data(raw_data_dir, data_name, cal_ratio):
    """
    Get raw data from the json file and split it into a calibration set and a test set.
    """
    raw_data = json.load(open(os.path.join(raw_data_dir, data_name+".json"), "r"))
    raw_data = [item for idx, item in enumerate(raw_data) if idx not in ids_to_remove]
    cal_raw_data, test_raw_data = train_test_split(raw_data, train_size=cal_ratio, random_state=42)
    print(len(raw_data), len(cal_raw_data), len(test_raw_data))
    return cal_raw_data, test_raw_data

def get_logits_data(model_name, data_name, cal_raw_data, test_raw_data, 
                    logits_data_dir, cal_ratio, prompt_methods, icl_methods):
    """
    Get logit scores of data instances and split these scores into a calibration set and a test set accordingly.
    """
    logits_data_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            logits_file = os.path.join(logits_data_dir, model_name+"_"+data_name+"_"+m+"_"+fs+".pkl")
            with open(logits_file, 'rb') as f:
                logits_data = pickle.load(f)
            logits_data = [item for idx, item in enumerate(logits_data) if idx not in ids_to_remove]
            cal_logits_data, test_logits_data = train_test_split(logits_data, train_size=cal_ratio, random_state=42)
            assert len(cal_logits_data) == len(cal_raw_data)
            assert len(test_logits_data) == len(test_raw_data)
            logits_data_all[m+"_"+fs] = {}
            logits_data_all[m+"_"+fs]["cal"] = cal_logits_data
            logits_data_all[m+"_"+fs]["test"] = test_logits_data
    return logits_data_all

def APS_FFCP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1, delta=0.03, random=True,allow_empty_sets=True):
    opt2idx = {opt: i for i, opt in enumerate(options)}
    ada_pred_sets_all = {}

    for m in prompt_methods:
        for fs in icl_methods:
            key = m + "_" + fs
            ada_pred_sets_all[key] = {}

            cal_scores = []
            cal_logits_data = logits_data_all[key]["cal"]

            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"])
                _, grad_norms = softmax_jacobian_and_norm(row["logits_options"])
                probs = probs + delta * grad_norms
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]

                pi = np.argsort(probs)[::-1]
                p_sorted = probs[pi]
                cumsum = np.cumsum(p_sorted)

                y = opt2idx[truth_answer]
                r = int(np.where(pi == y)[0][0])
                cal_scores.append((cumsum[r] - p_sorted[r]) + np.random.random() * p_sorted[r])

            n = len(cal_logits_data)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method="higher")

            pred_sets = {}
            test_logits_data = logits_data_all[key]["test"]

            for row in test_logits_data:
                probs = softmax(row["logits_options"])
                _, grad_norms = softmax_jacobian_and_norm(row["logits_options"])
                probs = probs + delta * grad_norms
                pi = np.argsort(probs)[::-1]
                p_sorted = probs[pi]
                cumsum = np.cumsum(p_sorted)

                L0 = int(np.searchsorted(cumsum, qhat, side="left"))
                L = min(L0 + 1, len(p_sorted))  # L ∈ {1,...,K}

                if random and L >= 1:
                    S_L = cumsum[L - 1]
                    s_L = p_sorted[L - 1]
                    if s_L > 0:
                        V = (S_L - qhat) / s_L
                        V = np.clip(V, 0.0, 1.0)
                        U = np.random.random() 
                        if U <= V:
                            L -= 1
                if not allow_empty_sets and L == 0:
                    L = 1

                ps = [options[j] for j in pi[:L]]
                pred_sets[str(row["id"])] = ps

            ada_pred_sets_all[key] = pred_sets

    return ada_pred_sets_all


def get_accuracy(logits_data, raw_data):
    res = []
    preds = []
    for idx, row in enumerate(raw_data):
        truth_answer = row["answer"]
        pred = logits_data[idx]
        assert pred["id"] == row["id"]
        pred_answer = options[np.argmax(pred["logits_options"])]
        preds.append(pred_answer)
        if pred_answer == truth_answer:
            res.append(1)
        else:
            res.append(0)
    return sum(res) / len(res), preds

def cal_acc(logits_data_all, test_raw_data, prompt_methods, icl_methods):
    results_acc = {}
    E_ratios = {}
    F_ratios = {}
    for m in prompt_methods:
        for fs in icl_methods:
            test_logits_data = logits_data_all[m+"_"+fs]["test"]
            acc, preds = get_accuracy(test_logits_data, test_raw_data)
            results_acc[m+"_"+fs] = acc
            counts = Counter(preds)
            E_ratio = counts["E"] / len(preds)
            F_ratio = counts["F"] / len(preds)
            E_ratios[m+"_"+fs] = E_ratio
            F_ratios[m+"_"+fs] = F_ratio
    return results_acc, E_ratios, F_ratios

def convert_id_to_ans(test_raw_data):
    test_id_to_answer = {}
    for row in test_raw_data:
        test_id_to_answer[str(row["id"])] = row["answer"]
    return test_id_to_answer

def cal_coverage(pred_sets_all, test_id_to_answer, prompt_methods, icl_methods):
    """
    Calculate the coverage rate of prediction sets.
    """""
    coverage_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            cover = []
            pred_sets = pred_sets_all[m+"_"+fs]
            for k, v in pred_sets.items():
                if test_id_to_answer[k] in v:
                    cover.append(1)
                else:
                    cover.append(0)
            coverage_all[m+"_"+fs] = sum(cover) / len(cover)
    return coverage_all

def cal_set_size(pred_sets_all, prompt_methods, icl_methods):
    set_sizes = {}
    for m in prompt_methods:
        for fs in icl_methods:
            sz = []
            pred_sets = pred_sets_all[m+"_"+fs]
            for k, v in pred_sets.items():
                sz.append(len(v))
            # print(f"{m}_{fs}: {min(sz)}, {max(sz)}")
            # average set size
            set_sizes[m+"_"+fs] = sum(sz) / len(sz)
    return set_sizes


def apply_conformal_prediction(args):
    all_data_results = {}
    for data_name in args.data_names:
        cal_raw_data, test_raw_data = get_raw_data(args.raw_data_dir, data_name, args.cal_ratio)
        logits_data_all = get_logits_data(args.model, data_name, cal_raw_data, test_raw_data, 
                                          args.logits_data_dir, args.cal_ratio,
                                          args.prompt_methods, args.icl_methods)
        results_acc, E_ratios, F_ratios = cal_acc(logits_data_all, test_raw_data,
                                                  args.prompt_methods, args.icl_methods)
        test_id_to_answer = convert_id_to_ans(test_raw_data)

        pred_sets_all_APS = APS_FFCP(logits_data_all, cal_raw_data,
                                   args.prompt_methods, args.icl_methods,
                                   alpha=args.alpha)
        coverage_all_APS = cal_coverage(pred_sets_all_APS, test_id_to_answer,
                                        args.prompt_methods, args.icl_methods)
        set_sizes_APS = cal_set_size(pred_sets_all_APS, args.prompt_methods, args.icl_methods)

        all_data_results[data_name] = {}
        all_data_results[data_name]["Acc"] = results_acc
        all_data_results[data_name]["E_rate"] = E_ratios
        all_data_results[data_name]["F_rate"] = F_ratios
        all_data_results[data_name]["APS_set_size"] = set_sizes_APS
        all_data_results[data_name]["APS_coverage"] = coverage_all_APS

    return all_data_results

def main(args):
    all_data_results = apply_conformal_prediction(args)
    save_path = Path(f"summary_results_ffaps_{args.model}.csv")

    rows = []
    acc_list = []
    
    for data_name in args.data_names:
        acc_value = 100 * np.mean(list(all_data_results[data_name]["Acc"].values()))
        acc_list.append(acc_value)
        print(f"{data_name}_Acc: {acc_value:.2f}")

    rows_acc = dict(zip(args.data_names, acc_list))
    print(f"Average acc: {np.mean(acc_list):.2f}")

    LAC_set_size, APS_set_size = [], []
    LAC_coverage, APS_coverage = [], []

    for data_name in args.data_names:
        APS_set_size.append(np.mean(list(all_data_results[data_name]["APS_set_size"].values())))
        APS_coverage.append(100 * np.mean(list(all_data_results[data_name]["APS_coverage"].values())))
    print(f"Average size: {APS_set_size}")
    print(f"Average cvg: {APS_coverage}")

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "Acc", "Coverage", "SetSize"])

        for i, data_name in enumerate(args.data_names):
            writer.writerow([
                data_name,
                acc_list[i],
                APS_coverage[i],
                APS_set_size[i],
            ])

        writer.writerow([
            "mean",
            np.mean(acc_list),
            np.mean(APS_coverage),
            np.mean(APS_set_size),
        ])

    print(f"\nCSV saved to: {save_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--raw_data_dir", type=str, default="data",
                        help="Directory where raw data are stored.")
    parser.add_argument("--logits_data_dir", type=str, default="outputs",
                        help="Directory where logits data are stored.")
    parser.add_argument("--data_names", nargs='*', 
                        default=['mmlu_10k', 'cosmosqa_10k', 'hellaswag_10k', 'halu_dialogue', 'halu_summarization'], 
                        help='List of datasets to be evaluated. If empty, all datasets are evaluated.')
    parser.add_argument("--prompt_methods", nargs='*', 
                        default=['base'], 
                        help='List of prompting methods. If empty, all methods are evaluated.')
    parser.add_argument("--icl_methods", nargs='*', 
                        default=['icl1'], 
                        help='Select from icl1, icl0, icl0_cot.')
    parser.add_argument("--cal_ratio", type=float, default=0.5,
                        help="The ratio of data to be used as the calibration data.")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="The error rate parameter.")
    args = parser.parse_args()

    main(args)