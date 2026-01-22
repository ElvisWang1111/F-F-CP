import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

from datasets import datasets
from conformal import helper
from conformal.icp import (
    IcpRegressor,
    RegressorNc,
    FeatRegressorNc,
    FFCP_RegressorNc,
    AbsErrorErrFunc,
    FeatErrorErrFunc,
    LCP_RegressorNc
)
from conformal.utils import (
    compute_coverage,
    seed_torch,
    write_list_to_excel,
    FFCP_write_list_to_excel
)


def partition_test(x_test, y_test):
    if sum(y_test == 0) / len(y_test) > 0.5:
        zero_sample_x = x_test[y_test == 0]
        zero_sample_y = y_test[y_test == 0]
        non_zero_x = x_test[y_test != 0]
        non_zero_y = y_test[y_test != 0]
        non_zero_parts = partition_bins(non_zero_x, non_zero_y, num_bins=2)
        return [(zero_sample_x, zero_sample_y)] + non_zero_parts
    else:
        return partition_bins(x_test, y_test, num_bins=3)


def partition_bins(x_test, y_test, num_bins):
    percentails = [np.percentile(y_test, 100 * i / num_bins) for i in range(1, num_bins)]
    percentails = [-float("inf")] + percentails + [float("inf")]
    cut_bins = pd.cut(y_test, percentails, labels=False)
    output = [(x_test[cut_bins == i], y_test[cut_bins == i]) for i in range(num_bins)]
    return output


def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")


def main(x_train, y_train, x_test, y_test, idx_train, idx_cal, args, seed):
    dir = f"ckpt/{args.data}_{args.epochs}"

    if os.path.exists(os.path.join(dir, f"model{seed}.pt")) and not args.no_resume:
        model = helper.mse_model(in_shape=in_shape, out_shape=out_shape, hidden_size=args.hidden_size,
                                 dropout=args.dropout)
        print(f"==> Load model from {dir}")
        model.load_state_dict(torch.load(os.path.join(dir, f"model{seed}.pt"), map_location=device))
    else:
        model = None
        print("not")

    mean_estimator = helper.MSENet_RegressorAdapter(model=model, device=device, fit_params=None,
                                                    in_shape=in_shape, out_shape=out_shape,
                                                    hidden_size=args.hidden_size, learn_func=nn_learn_func, epochs=args.epochs,
                                                    batch_size=args.batch_size, dropout=args.dropout, lr=args.lr, wd=args.wd,
                                                    test_ratio=cv_test_ratio, random_state=cv_random_state, )

    if float(args.feat_norm) <= 0 or args.feat_norm == "inf":
        args.feat_norm = "inf"
        print("Use inf as feature norm")
    else:
        args.feat_norm = float(args.feat_norm)

    nc = FeatRegressorNc(mean_estimator, inv_lr=args.feat_lr, inv_step=args.feat_step,
                         feat_norm=args.feat_norm, certification_method=args.cert_method, kernel_weight=args.kernel_weight)
    icp = IcpRegressor(nc)

    if os.path.exists(os.path.join(dir, f"model{seed}.pt")) and not args.no_resume:
        pass
    else:
        icp.fit(x_train[idx_train, :], y_train[idx_train])
        makedirs(dir)
        print(f"==> Saving model at {dir}/model.pt")
        torch.save(mean_estimator.model.state_dict(), os.path.join(dir, f"model{seed}.pt"))

    sub_test_list = partition_test(x_test, y_test)

    icp_LCP = IcpRegressor(RegressorNc(mean_estimator))
    icp_LCP.calibrate(x_train[idx_cal, :], y_train[idx_cal])

    assert len(sub_test_list) == 3
    LCP_group_coverage = []
    nc_LCP = LCP_RegressorNc(mean_estimator)
    for (sub_x_test, sub_y_test) in sub_test_list:
        sub_predictions = nc_LCP.predict(mean_estimator.model, x_train[idx_cal, :], y_train[idx_cal], sub_x_test, sub_y_test, 4,
                                      significance=alpha)
        y_lower, y_upper = sub_predictions[..., 0], sub_predictions[..., 1]
        coverage_cp_qnet, _ = compute_coverage(sub_y_test, y_lower, y_upper,significance=alpha)
        LCP_group_coverage.append(coverage_cp_qnet)

    nc_FFCP = FFCP_RegressorNc(mean_estimator)
    FFLCP_group_coverage = []
    for layer_index in range(5):
        FFLCP_group_coverage_each_layer = []
        for (sub_x_test, sub_y_test) in sub_test_list:
            intervals_FFLCP = nc_FFCP.predict(mean_estimator.model, x_train[idx_cal, :], y_train[idx_cal], sub_x_test, sub_y_test, layer_index,
                                      significance=alpha)
            y_lo, y_up = intervals_FFLCP[..., 0], intervals_FFLCP[..., 1]
            FFCP_coverage_qnet, _ = compute_coverage(sub_y_test, y_lo, y_up, alpha, name="FFLCPNet RegressorNc", verbose=False)
            FFLCP_group_coverage_each_layer.append(FFCP_coverage_qnet)
        FFLCP_group_coverage.append(FFLCP_group_coverage_each_layer)

    x_cal = x_train[idx_cal, :]
    y_cal = y_train[idx_cal]
    nc_cal_scores = nc.score(x_cal.copy(), y_cal.copy())

    FLCP_group_coverage = []
    for (sub_x_test, sub_y_test) in sub_test_list:
        intervals_FLCP = nc.predict(
            x=sub_x_test,
            nc=nc_cal_scores,
            significance=alpha,
            x_cal=x_cal 
        )
        y_lo, y_up = intervals_FLCP[..., 0], intervals_FLCP[..., 1]
        in_coverage = icp.if_in_coverage(sub_x_test, sub_y_test, significance=alpha, nc_cal_scores = nc_cal_scores)
        coverage_fcp = np.sum(in_coverage) * 100 / len(in_coverage)
        FLCP_group_coverage.append(coverage_fcp)

    return LCP_group_coverage, FFLCP_group_coverage, FLCP_group_coverage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=-1, type=int)
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument("--data", type=str, default="com", help="meps20 fb1 fb2 blog")
    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage error")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", "--bs", type=int, default=64)
    parser.add_argument("--hidden_size", "--hs", type=int, default=64)
    parser.add_argument("--dropout", "--do", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--no-resume", action="store_true", default=False)

    parser.add_argument("--feat_opt", "--fo", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--feat_lr", "--fl", type=float, default=1e-3)
    parser.add_argument("--feat_step", "--fs", type=int, default=60)
    parser.add_argument("--feat_norm", "--fn", default=-1)
    parser.add_argument("--cert_method", "--cm", type=int, default=0, choices=[0, 1, 2, 3])

    parser.add_argument("--kernel_weight", type=float, default=0.05)
    parser.add_argument("--test", action="store_true", default=False)

    args = parser.parse_args()
    # The path where Group coverage data is stored
    LCP_path = f"output/{args.data}_LCP_group_coverage.xlsx"
    FFLCP_path = f"output/{args.data}_FFLCP_group_coverage.xlsx"
    FLCP_path = f"output/{args.data}_FLCP_group_coverage.xlsx"

    fcp_coverage_list, fcp_length_list, cp_coverage_list, cp_length_list = [], [], [], []
    FFCP_coverage_list, FFCP_length_list = [], []
    FFCP_seed_each_layer_coverage, FFCP_seed_each_layer_length = [], []
    
    for seed in tqdm(args.seed):
        seed_torch(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
        device = torch.device("cpu") if args.device < 0 else torch.device("cuda")

        nn_learn_func = torch.optim.Adam

        # ratio of held-out data, used in cross-validation
        cv_test_ratio = 0.05
        # desired miscoverage error
        # alpha = 0.1
        alpha = args.alpha
        # used to determine the size of test set
        test_ratio = 0.2
        # seed for splitting the data in cross-validation.
        cv_random_state = 1

        dataset_base_path = "./datasets/"
        dataset_name = args.data
        X, y = datasets.GetDataset(dataset_name, dataset_base_path)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        n_train = x_train.shape[0]
        in_shape = x_train.shape[1]
        out_shape = y_train.shape[1] if len(y_train.shape) > 1 else 1

        print(dataset_base_path)
        print("Dataset: %s" % (dataset_name))
        print("Dimensions: train set (n=%d, p=%d) ; test set (n=%d, p=%d)" %
              (x_train.shape[0], x_train.shape[1], x_test.shape[0], x_test.shape[1]))

        # divide the data into proper training set and calibration set
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train / 2))
        idx_train, idx_cal = idx[:n_half], idx[n_half:2 * n_half]

        # zero mean and unit variance scaling
        scalerX = StandardScaler()
        scalerX = scalerX.fit(x_train[idx_train])
        x_train = scalerX.transform(x_train)
        x_test = scalerX.transform(x_test)

        # scale the labels by dividing each by the mean absolute response
        mean_y_train = np.mean(np.abs(y_train[idx_train]))
        y_train = np.squeeze(y_train) / mean_y_train
        y_test = np.squeeze(y_test) / mean_y_train

        LCP_group_coverage, FFLCP_group_coverage, FLCP_group_coverage = \
            main(x_train, y_train, x_test, y_test, idx_train, idx_cal, args, seed)

        write_list_to_excel(LCP_group_coverage, LCP_path)
        write_list_to_excel(FLCP_group_coverage, FLCP_path)
        FFCP_write_list_to_excel(FFLCP_group_coverage, FFLCP_path)

print("Locally Adaptive Conformal Prediction process completed successfully.")
print("All output files have been saved in the 'Group_coverage_output' folder.")
