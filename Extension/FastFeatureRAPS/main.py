import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from utils import *
# from utils_ori import *
from conformal_raps import ConformalModel_ori
from conformal_fraps import ConformalModel_fraps
from conformal_ffraps import ConformalModel as Conformal_FFRAPS
import torch.backends.cudnn as cudnn
import random
from PIL import ImageFile
import copy
ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_torch(seed):
    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='Run RAPS / FFRAPS / FRAPS on ImageNet')
parser.add_argument('data', metavar='IMAGENETVALDIR', help='path to ImageNet val')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--num_calib', type=int, default=10000)
parser.add_argument('--seed', type=int, nargs='+', default=[0])
parser.add_argument('--method', type=str, choices=['RAPS', 'FFRAPS', 'FRAPS', 'ALL'],
                    default='ALL', help='Choose which method to run (default: ALL)')
parser.add_argument('--fraps-calib-method', type=str, default='IBP',
                    help='Calibration-time LiRPA method for FRAPS (IBP/backward/CROWN), default: IBP')
parser.add_argument('--fraps-calib-bsz', type=int, default=32,
                    help='Calibration batch size for FRAPS when using IBP, default: 32')
parser.add_argument('--fraps-cmethod', type=str, default='backward',
                    help='Inference-time LiRPA method for FRAPS (backward/CROWN/IBP), default: backward')

if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    results = {
        "RAPS":   {"top1": [], "top5": [], "cov": [], "size": []},
        "FFRAPS": {"top1": [], "top5": [], "cov": [], "size": []},
        "FRAPS":  {"top1": [], "top5": [], "cov": [], "size": []},
    }

    for seed in tqdm(args.seed):
        seed_torch(seed)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225])
        ])

        full_val = torchvision.datasets.ImageFolder(args.data, transform)
        calib_len = int(args.num_calib)
        val_len = len(full_val) - calib_len
        imagenet_calib_data, imagenet_val_data = torch.utils.data.random_split(full_val, [calib_len, val_len])

        calib_loader_shared = torch.utils.data.DataLoader(
            imagenet_calib_data, batch_size=args.batch_size, shuffle=True,
            pin_memory=True, num_workers=args.num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            imagenet_val_data, batch_size=args.batch_size, shuffle=False,
            pin_memory=True, num_workers=args.num_workers
        )

        model_base = torchvision.models.resnext101_32x8d(pretrained=True, progress=True).to(device)
        model_base = torch.nn.DataParallel(model_base) if torch.cuda.device_count() > 1 else model_base
        model_base.eval()

        allow_zero_sets = False
        randomized = True
        optional_T = None

        # ---------------- RAPS ----------------
        if args.method in ["RAPS", "ALL"]:
            print("\n=== Start RAPS ===")
            model_RAPS = copy.deepcopy(model_base)
            raps = ConformalModel_ori(model_RAPS, calib_loader_shared, alpha=0.1, lamda=None,
                                      randomized=randomized, allow_zero_sets=allow_zero_sets)
            r_top1, r_top5, r_cov, r_size = validate(val_loader, raps, print_bool=True)
            results["RAPS"]["top1"].append(r_top1)
            results["RAPS"]["top5"].append(r_top5)
            results["RAPS"]["cov"].append(r_cov)
            results["RAPS"]["size"].append(r_size)

#         # ---------------- FRAPS (LiRPA) ----------------
        if args.method in ["FRAPS", "ALL"]:
            print("\n=== Start FRAPS ===")
            model_FRAPS = copy.deepcopy(model_base)
            fraps = ConformalModel_fraps(
                model_FRAPS,
                calib_loader_shared,
                alpha=0.1, lamda=None,
                randomized=randomized, allow_zero_sets=allow_zero_sets,
                delta=0.1, T=optional_T,
                calib_method=args.fraps_calib_method,
                calib_bs=args.fraps_calib_bsz,
                cmethod=args.fraps_cmethod
            )
            f_top1, f_top5, f_cov, f_size = validate(val_loader, fraps, print_bool=True)
            results["FRAPS"]["top1"].append(f_top1)
            results["FRAPS"]["top5"].append(f_top5)
            results["FRAPS"]["cov"].append(f_cov)
            results["FRAPS"]["size"].append(f_size)

        # # ---------------- FFRAPS ----------------
        if args.method in ["FFRAPS", "ALL"]:
            print("\n=== Start FFRAPS ===")
            model_FFRAPS = copy.deepcopy(model_base)
            ffraps = Conformal_FFRAPS(model_FFRAPS, calib_loader_shared, alpha=0.1, lamda=None,
                                      randomized=randomized, allow_zero_sets=allow_zero_sets,
                                      delta=0.1, T=optional_T)
            ff_top1, ff_top5, ff_cov, ff_size = validate(val_loader, ffraps, print_bool=True)
            results["FFRAPS"]["top1"].append(ff_top1)
            results["FFRAPS"]["top5"].append(ff_top5)
            results["FFRAPS"]["cov"].append(ff_cov)
            results["FFRAPS"]["size"].append(ff_size)

    print("\n===== FINAL RESULTS (mean ± std) =====")
    for key in ["RAPS", "FFRAPS", "FRAPS"]:
        if len(results[key]["top1"]) > 0:
            print("[{}] top1: {:.4f} ± {:.4f} | top5: {:.4f} ± {:.4f} | cov: {:.4f} ± {:.4f} | size: {:.4f} ± {:.4f}".format(
                key,
                np.mean(results[key]["top1"]), np.std(results[key]["top1"]),
                np.mean(results[key]["top5"]), np.std(results[key]["top5"]),
                np.mean(results[key]["cov"]), np.std(results[key]["cov"]),
                np.mean(results[key]["size"]), np.std(results[key]["size"])
            ))
