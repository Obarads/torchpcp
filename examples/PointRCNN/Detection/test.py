import os, sys
CW_DIR = os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../../"))) # for package path

from tqdm import tqdm
import hydra
import torch

# tools
from torchpcp.utils import monitor, pytorch_tools
from torchpcp.utils.metrics import MultiAssessmentMeter, LossMeter

# env
from libs.model_env import processing
from libs.model_env import (get_model, get_losses, get_dataset, get_loader,
                            get_checkpoint)

# config
from libs import configs

@hydra.main(config_name="modelnet40_config")
def main(cfg: configs.ModelNet40Config) -> None:
    # ifx paths
    cfg = monitor.fix_path_in_configs(CW_DIR, cfg, 
        [
            ["dataset", "root"],
            ["model", "resume"]
        ]
    )

    # set a seed
    pytorch_tools.set_seed(
        cfg.general.seed,
        cfg.general.device,
        cfg.general.reproducibility
    )

    # set a device
    cfg.general.device = pytorch_tools.select_device(cfg.general.device)

    # get a trained model
    checkpoint, checkpoint_cfg = get_checkpoint(cfg.model.resume)

    # test env
    ## Get model.
    model = get_model(checkpoint_cfg)
    ## Get dataset and loader.
    dataset = get_dataset(checkpoint_cfg)
    datset_loader = get_loader(checkpoint_cfg, dataset)
    ## Get loss functions.
    criterion = get_losses(checkpoint_cfg)

    # set trained params
    model.load_state_dict(checkpoint["model"])

    # test start
    test_log = test(cfg, model, datset_loader["test"], criterion)

    # show results
    print(test_log)

    print("Finish test.")


def test(cfg, model, loader, criterion, publisher="test"):
    model.eval()

    # metrics
    acc_meter = MultiAssessmentMeter(
        num_classes=cfg.dataset.num_classes, 
        metrics=["class","overall","iou"]
    )
    batch_loss = LossMeter()
    meters = (acc_meter, batch_loss)

    with torch.no_grad():
        for _, data in enumerate(loader):
            _ = processing(model, criterion, data, meters, cfg.general.device)

    # get epoch loss and accuracy
    epoch_loss = batch_loss.compute()
    epoch_acc = acc_meter.compute()

    # save loss and acc to tensorboard
    log_dict = {
        "{}/loss".format(publisher): epoch_loss,
        "{}/mAcc".format(publisher): epoch_acc["class"],
        "{}/oAcc".format(publisher): epoch_acc["overall"],
        "{}/IoU".format(publisher): epoch_acc["iou"]
    }

    return log_dict

if __name__ == "__main__":
    main()



