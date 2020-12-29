def dict2tensorboard(log_dict, writer, step):
    for key in log_dict:
        if isinstance(log_dict[key], list):
            _log = log_dict[key][0]
            _step = log_dict[key][1]
        else:
            _log = log_dict[key]
            _step = step
        writer.add_scalar(key, _log, _step)

def dict2wandb(log_dict, writer, step):
    writer.log(log_dict, step=step)

def dict2logger(log_dict, writer, step, log_service):
    # assert log_service in ["wandb","tensorboardX"]
    if log_service == "wandb":
        dict2wandb(log_dict, writer, step)
    elif log_service == "tensorboardX":
        dict2tensorboard(log_dict, writer, step)
    else:
        raise NotImplementedError("Unknown log_service: {}".format(log_service))


