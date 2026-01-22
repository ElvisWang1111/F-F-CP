import os, json, pickle, argparse, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from collections import Counter
from contextlib import contextmanager

import csv
from pathlib import Path
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)


options = ["A","B","C","D","E","F"]
ids_to_remove = [1,3,5,7,9]


def softmax(x):
    e=np.exp(x-np.max(x));return e/e.sum()


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

# ---------------- LiRPA: softmax  ----------------
class SoftmaxWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return torch.softmax(z, dim=-1)

@contextmanager
def _no_grad_eval(model):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        yield
    if was_training:
        model.train()

_BM_CACHE = {}

def lirpa_softmax_bounds_batch(
    z: torch.Tensor,
    eps,
    method: str = "backward", 
    ob_iter: int = 20,
    device=None,
    chunk_size: int = 256,
):

    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.to(device)
    B, C = z.shape
    dtype = z.dtype

    key = (C, device.type, str(dtype), method, int(ob_iter))
    if key not in _BM_CACHE:
        head = SoftmaxWrapper().to(device)
        dummy = torch.zeros(1, C, device=device, dtype=dtype)
        bm = BoundedModule(head, dummy)
        if "Optimized" in method:
            bm.set_bound_opts({
                "optimize_bound_args": {"ob_iteration": ob_iter, "ob_lr": 0.1, "ob_verbose": 0}
            })
        _BM_CACHE[key] = bm
    else:
        bm = _BM_CACHE[key]

    if isinstance(eps, torch.Tensor):
        eps_full = eps.to(device=device, dtype=dtype)
        if eps_full.ndim == 0:
            eps_full = torch.full((B, C), float(eps_full), device=device, dtype=dtype)
        elif eps_full.shape == (C,):
            eps_full = eps_full.view(1, C).expand(B, C).contiguous()
        elif eps_full.shape == (B, C):
            pass
        else:
            raise ValueError(f"eps tensor shape {eps_full.shape} not supported")
    elif isinstance(eps, (float, int)):
        eps_full = torch.full((B, C), abs(float(eps)), device=device, dtype=dtype)
    else:
        eps_np = np.asarray(eps, dtype=np.float32)
        if eps_np.ndim == 0:
            eps_full = torch.full((B, C), float(eps_np), device=device, dtype=dtype)
        elif eps_np.shape == (C,):
            eps_full = torch.from_numpy(eps_np).to(device=device, dtype=dtype).view(1, C).expand(B, C).contiguous()
        elif eps_np.shape == (B, C):
            eps_full = torch.from_numpy(eps_np).to(device=device, dtype=dtype)
        else:
            raise ValueError(f"eps array shape {eps_np.shape} not supported")
    eps_full = eps_full.abs()

    lbs, ubs = [], []

    with _no_grad_eval(bm):
        for s in range(0, B, chunk_size):
            e = min(s + chunk_size, B)
            z_chunk = z[s:e]
            ptb = PerturbationLpNorm(norm=float("inf"), eps=eps_full[s:e])
            z_bound = BoundedTensor(z_chunk, ptb)
            try:
                lb, ub = bm.compute_bounds(x=(z_bound,), method=method)
            except (RuntimeError, NotImplementedError):
                lb, ub = bm.compute_bounds(x=(z_bound,), method="IBP")
            lbs.append(lb.detach().cpu())
            ubs.append(ub.detach().cpu())

    lb_all = torch.cat(lbs, dim=0).clamp_(0, 1)
    ub_all = torch.cat(ubs, dim=0).clamp_(0, 1)
    sums = ub_all.sum(dim=1, keepdim=True).clamp(min=1.0)
    ub_all = ub_all / sums
    return lb_all, ub_all


class LiRPASoftmaxUB:
    def __init__(self, delta, cmethod="backward", device=None, chunk_size=256, ob_iter=20):
        self.delta = float(delta)
        self.cmethod = cmethod
        self.device = device
        self.chunk_size = chunk_size
        self.ob_iter = ob_iter

    def forward(self, logits_t: torch.Tensor) -> np.ndarray:
        _, ub = lirpa_softmax_bounds_batch(
            logits_t, self.delta, method=self.cmethod, ob_iter=self.ob_iter,
            device=self.device, chunk_size=self.chunk_size
        )
        return ub.numpy()
    

def APS_CP_LiRPA(
    logits_all,
    cal_raw,
    prompt_methods,
    icl_methods,
    lirpa_bounder,
    alpha=0.1,
    random=True,
    allow_empty_sets=True,
    rng=None,
):

    if rng is None:
        rng = np.random.default_rng()

    id2raw_cal = {str(r["id"]): r for r in cal_raw}
    opt2idx = {opt: i for i, opt in enumerate(options)}

    ada_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            ada_all[key] = {}

            cal_logits = logits_all[key]["cal"]
            cal_np = np.stack([r["logits_options"] for r in cal_logits], 0).astype(
                np.float32
            )
            cal_ub_t = lirpa_bounder.forward(torch.from_numpy(cal_np))
            cal_ub = cal_ub_t

            cal_scores = []
            for i, r in enumerate(cal_logits):
                ub = np.clip(cal_ub[i].astype(np.float32), 0.0, 1.0)
                pi = np.argsort(ub)[::-1]
                ub_sorted = ub[pi]
                cumsum = np.cumsum(ub_sorted)

                truth = id2raw_cal[str(r["id"])]["answer"]
                y = opt2idx[truth]
                rank = int(np.where(pi == y)[0][0])

                if random:
                    s_r = float(ub_sorted[rank])
                    score = (float(cumsum[rank]) - s_r) + float(rng.random()) * s_r
                else:
                    score = float(cumsum[rank])

                cal_scores.append(float(np.clip(score, 0.0, 1.0)))

            n = len(cal_scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(np.asarray(cal_scores, np.float32), q_level, method="higher")

            test_logits = logits_all[key]["test"]
            test_np = np.stack([r["logits_options"] for r in test_logits], 0).astype(
                np.float32
            )
            test_ub_t = lirpa_bounder.forward(torch.from_numpy(test_np))
            test_ub = test_ub_t

            pred = {}
            for i, r in enumerate(test_logits):
                ub = np.clip(test_ub[i].astype(np.float32), 0.0, 1.0)
                pi = np.argsort(ub)[::-1]
                ub_sorted = ub[pi]
                cumsum = np.cumsum(ub_sorted)

                L0 = int(np.searchsorted(cumsum, qhat, side="left"))
                L = min(L0 + 1, len(ub_sorted)) 

                if random and L >= 1:
                    S_L = float(cumsum[L - 1])
                    s_L = float(ub_sorted[L - 1])

                    if s_L > 0.0:
                        V = (S_L - float(qhat)) / s_L
                        V = float(np.clip(V, 0.0, 1.0))
                        U = float(rng.random())
                        if U <= V:
                            L -= 1

                if not allow_empty_sets and L == 0:
                    L = 1

                ps = [options[j] for j in pi[:L]]
                pred[str(r["id"])] = ps

            ada_all[key] = pred

    return ada_all

# ---------------- delta tuning by grid ----------------
def _APS_qhat(cal_logits_rows, id2raw, lirpa, alpha):
    np_logits = np.stack([r["logits_options"] for r in cal_logits_rows], 0).astype(np.float32)
    ub = lirpa.forward(torch.from_numpy(np_logits))
    sc = []
    for i, r in enumerate(cal_logits_rows):
        u = ub[i]; pi = np.argsort(u)[::-1]; cs = np.take_along_axis(u, pi, 0).cumsum()
        csr = np.take_along_axis(cs, pi.argsort(), 0)
        t = id2raw[str(r["id"])]["answer"]; sc.append(float(np.clip(csr[options.index(t)], 0, 1)))
    n = len(sc); q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(np.array(sc, np.float32), q, method="higher"))


def _eval(eval_logits_rows, id2raw, lirpa, qhat):
    np_logits = np.stack([r["logits_options"] for r in eval_logits_rows], 0).astype(np.float32)
    ub = lirpa.forward(torch.from_numpy(np_logits))
    cover, sz = [], []
    for i, r in enumerate(eval_logits_rows):
        u = ub[i]; pi = np.argsort(u)[::-1]; cs = np.take_along_axis(u, pi, 0).cumsum()
        ps = []; ii = 0
        while ii < len(cs) and cs[ii] <= qhat:
            ps.append(options[pi[ii]]); ii += 1
        if not ps: ps.append(options[pi[0]])
        sz.append(len(ps)); truth = id2raw[str(r["id"])]["answer"]
        cover.append(1 if truth in ps else 0)
    return np.mean(cover), np.mean(sz)


def tune_delta_by_grid(logits_all, cal_raw, prompt_methods, icl_methods, alpha=0.1,
                       deltas=None, cmethod="backward", inner_ratio=0.5, coverage_tol=0.0,
                       lirpa_device=None, lirpa_chunk_size=256, ob_iter=20):
    if deltas is None:
        deltas = np.linspace(1e-6, 5e-2, 100).tolist()
    res = []
    id2raw_cal = {str(r["id"]): r for r in cal_raw}
    for d in deltas:
        covs, sizes = [], []
        for m in prompt_methods:
            for fs in icl_methods:
                key = f"{m}_{fs}"
                cal_logits = logits_all[key]["cal"]
                fit, eval_ = train_test_split(cal_logits, train_size=inner_ratio, random_state=0)
                lirpa = LiRPASoftmaxUB(delta=d, cmethod=cmethod, device=lirpa_device,
                                       chunk_size=lirpa_chunk_size, ob_iter=ob_iter)
                qhat = _APS_qhat(fit, id2raw_cal, lirpa, alpha)
                cov, sz = _eval(eval_, id2raw_cal, lirpa, qhat)
                covs.append(cov)
                sizes.append(sz)
        res.append({"delta": d, "cov": np.mean(covs), "size": np.mean(sizes)})

    tgt = 1 - alpha - coverage_tol
    feas = [r for r in res if r["cov"] >= tgt]
    best = (sorted(feas, key=lambda r: (r["size"], -r["cov"]))[0]
             if feas else sorted(res, key=lambda r: (-r["cov"], r["size"]))[0])
    print("[auto-tune] grid:", res, "\n[auto-tune] best:", best)
    return best["delta"]


# ---------------- evaluation helpers ----------------

def get_accuracy(logits, raw):
    res = []; preds = []
    id2raw = {str(r["id"]): r for r in raw}
    for r in logits:
        truth = id2raw[str(r["id"])]["answer"]
        p = r
        pred_ans = options[np.argmax(p["logits_options"])]
        preds.append(pred_ans); res.append(int(pred_ans == truth))
    return np.mean(res), preds


def cal_acc(logits_all, test_raw, prompt_methods, icl_methods):
    accs, E, F = {}, {}, {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            a, p = get_accuracy(logits_all[key]["test"], test_raw)
            accs[key] = a; c = Counter(p)
            E[key] = c.get("E", 0) / len(p); F[key] = c.get("F", 0) / len(p)
    return accs, E, F


def convert_id_to_ans(raw):
    return {str(r["id"]): r["answer"] for r in raw}


def cal_coverage(pred_sets_all, id2ans, prompt_methods, icl_methods):
    out = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            v = pred_sets_all[key]
            cov = [1 if id2ans[k] in v[k] else 0 for k in v]
            out[key] = np.mean(cov)
    return out


def cal_set_size(pred_sets_all, prompt_methods, icl_methods):
    out = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = f"{m}_{fs}"
            out[key] = np.mean([len(v) for v in pred_sets_all[key].values()])
    return out


def apply_conformal_prediction(args):
    all_res = {}
    for data_name in args.data_names:
        cal_raw_data, test_raw_data = get_raw_data(args.raw_data_dir, data_name, args.cal_ratio)
        logits_data_all = get_logits_data(args.model, data_name, cal_raw_data, test_raw_data, 
                                          args.logits_data_dir, args.cal_ratio,
                                          args.prompt_methods, args.icl_methods)
        results_acc, E_ratios, F_ratios = cal_acc(logits_data_all, test_raw_data,
                                                  args.prompt_methods, args.icl_methods)
        test_id_to_answer = convert_id_to_ans(test_raw_data)

        best_delta = args.delta
        print(f"[LiRPA-APS] using delta={best_delta}")

        lirpa = LiRPASoftmaxUB(delta=best_delta, cmethod=args.cmethod,
                               device=args.lirpa_device, chunk_size=args.lirpa_chunk_size,
                               ob_iter=args.lirpa_ob_iter)
        pred_APS = APS_CP_LiRPA(logits_data_all, cal_raw_data, args.prompt_methods, args.icl_methods, lirpa_bounder=lirpa, alpha=args.alpha)
        coverage_all_APS  = cal_coverage(pred_APS, test_id_to_answer, args.prompt_methods, args.icl_methods)
        set_sizes_APS   = cal_set_size(pred_APS, args.prompt_methods, args.icl_methods)

        all_res[data_name] = {}
        all_res[data_name]["Acc"] = results_acc
        all_res[data_name]["APS_set_size"] = set_sizes_APS
        all_res[data_name]["APS_coverage"] = coverage_all_APS

    return all_res


def main(args):
    all_data_results = apply_conformal_prediction(args)
    save_path = Path(f"summary_results_faps_{args.model}.csv")
    
    rows = []
    acc_list = []
    
    for data_name in args.data_names:
        acc_value = 100 * np.mean(list(all_data_results[data_name]["Acc"].values()))
        acc_list.append(acc_value)
        print(f"{data_name}_Acc: {acc_value:.2f}")

    rows_acc = dict(zip(args.data_names, acc_list))
    print(f"Average acc: {np.mean(acc_list):.2f}")

    APS_coverage, APS_set_size = [], []

    for data_name in args.data_names:
        APS_set_size.append(np.mean(list(all_data_results[data_name]["APS_set_size"].values())))
        APS_coverage.append(100 * np.mean(list(all_data_results[data_name]["APS_coverage"].values())))

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
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--raw_data_dir", type=str, default="data")
    p.add_argument("--logits_data_dir", type=str, default="outputs_base")
    p.add_argument("--data_names", nargs='*', 
                        default=['mmlu_10k', 'cosmosqa_10k', 'hellaswag_10k', 'halu_dialogue', 'halu_summarization'], 
                        help='List of datasets to be evaluated. If empty, all datasets are evaluated.')
    p.add_argument("--prompt_methods", nargs='*', 
                        # default=['base', 'shared', 'task'], 
                        default=['base'], 
                        help='List of prompting methods. If empty, all methods are evaluated.')
    p.add_argument("--icl_methods", nargs='*', 
                        default=['icl1'], 
                        help='Select from icl1, icl0, icl0_cot.')
    p.add_argument("--cal_ratio", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--cmethod", type=str, default="backward", help="IBP | backward | CROWN-Optimized")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--auto_tune_delta", action="store_true", help="Grid search delta")
    p.add_argument("--lirpa_device", type=str, default=None, help="e.g., cuda, cuda:0, cpu")
    p.add_argument("--lirpa_chunk_size", type=int, default=512)
    p.add_argument("--lirpa_ob_iter", type=int, default=20)
    args = p.parse_args(); main(args)
