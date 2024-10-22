import sys

sys.path.append("../")
import torch
import os
from config.read_configurations import config_hydro_temp as config
from core.utils.randomseed_config import randomseed_config
from core.utils.small_codes import create_output_dirs
from MODELS.Differentiable_models import diff_gDM
from MODELS import train_test


def main(args):
    # updating args. all settings are here
    # args = update_args(args,
    #                     frac_smoothening_mode=frac_smooth,
    #                     randomseed=seed
    # )
    randomseed_config(seed=args["randomseed"][0])
    # Creating output directories and adding it to args
    args = create_output_dirs(args)

    if "Train" in args["Action"]:  # training mode
        diff_model = diff_gDM(args)
        optim = torch.optim.Adadelta(diff_model.parameters())
        train_test.train_differentiable_model(
            args=args,
            diff_model=diff_model,
            optim=optim
        )
    if "Test" in args["Action"]:  # testing mode
        modelFile = os.path.join(args["out_dir"], "model_Ep" + str(args["EPOCH_testing"]) + ".pt")
        diff_model = torch.load(modelFile)
        train_test.test_differentiable_model(
            args=args,
            diff_model=diff_model
        )


if __name__ == "__main__":
    args = config
    main(args)
    print("END")
