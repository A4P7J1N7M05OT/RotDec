from torch.optim.lr_scheduler import _LRScheduler
import math

class RotDec(_LRScheduler):
    """
    Custom scheduler that combines cosine decay with linear or exponential decay of the 
    base learning rate, ensuring that the learning rate reaches min_lr by the end of training.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        epochs (int): Total number of epochs.
        steps_per_epoch (int): Number of steps per epoch.
        min_lr (float): Minimum learning rate after applying decay. Default: 1e-6.
        lr_decay (float or None): The decay value. 
            - If None: Linear decay to min_lr is used.
            - If float: Represents the absolute decay value applied per epoch.
        exp_decay (bool): If True, apply exponential decay. If False, apply linear decay. Default: False.
        decay_func (callable or None): Custom decay function. 
            If None, uses cosine annealing (cosine_decay).
        last_epoch (int): The index of the last epoch. Default: -1.
    """
    def __init__(self, optimizer, epochs, steps_per_epoch, min_lr=1e-6, lr_decay=None, 
                 exp_decay=False, decay_func=None, last_epoch=-1):
        
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.min_lr = min_lr
        self.exp_decay = exp_decay
        self.decay_func = decay_func or self.cosine_decay

        # Calculate decay rates for each parameter group
        self.lr_decay = []
        for param_group in optimizer.param_groups:
            base_lr = param_group['lr']
            if lr_decay is None:
                # Default: Linear decay to min_lr
                decay = (base_lr - self.min_lr) / (epochs - 1) 
            else:
                # Absolute decay value
                decay = lr_decay 
            self.lr_decay.append(decay)

        super().__init__(optimizer, last_epoch)

    def cosine_decay(self, current_step, total_steps):
        """Cosine decay function."""
        return 0.5 * (1 + math.cos(math.pi * current_step / total_steps))

    def get_lr(self):
        """
        Compute learning rates for all parameter groups based on the current step,
        applying decay_func and base LR decay (linear or exponential).
        """
        current_step = self.last_epoch + 1  # Adjust for 0-indexing
        current_epoch = current_step // self.steps_per_epoch

        # Calculate learning rates for each parameter group
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            if self.exp_decay:
                current_initial_lr = base_lr * (self.lr_decay[i] ** current_epoch)
            else:
                current_initial_lr = base_lr - (self.lr_decay[i] * current_epoch)

            # Apply decay within the epoch
            decay_factor = self.decay_func(current_step % self.steps_per_epoch, self.steps_per_epoch)

            # Calculate the new LR, ensuring it's within bounds
            new_lr = max(current_initial_lr * decay_factor, self.min_lr)
            lrs.append(new_lr)

        return lrs