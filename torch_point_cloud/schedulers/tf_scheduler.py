import warnings

from torch.optim.lr_scheduler import _LRScheduler

class TFScheduler(_LRScheduler):
    """
    Tensorflow learning_rate for Pytorch
    """
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch=last_epoch)

    # from _LRScheduler.step() on lr_scheduler.py
    def step(self, epoch=None):
        """
        Parameters
        ----------
        epoch: int
            value to add to global step
        """

        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                # warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch += epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class ExponentialDecay(TFScheduler):
    """
    tf.train.exponential_decay (for ASIS)
    """
    def __init__(self, optimizer, decay_rate, decay_step, last_epoch=-1):
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        # self.min_lr = min_lr
        super(ExponentialDecay, self).__init__(optimizer,last_epoch=last_epoch)

    def get_lr(self):
        last_step = self.last_epoch
        lrs = [
            self.decay_rate ** (last_step // self.decay_step) * base_lr 
            for base_lr in self.base_lrs
        ]
        return lrs
