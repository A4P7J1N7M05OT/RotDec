from torch.optim.lr_scheduler import _LRScheduler
import math

class RotDec(_LRScheduler):
    def __init__(self, optimizer, epochs, steps_per_epoch, min_lr=1e-6, last_epoch=-1):
        """
        Custom scheduler that combines cosine decay with linear decay of the base learning rate,
        ensuring that the learning rate reaches min_lr by the end of training.
        
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            epochs (int): Total number of epochs.
            steps_per_epoch (int): Number of steps per epoch.
            min_lr (float): Minimum learning rate after applying decay.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.min_lr = min_lr
        self.total_steps = epochs * steps_per_epoch
        self.lr_decay = [(param_group['lr'] - self.min_lr) / (epochs - 1) for param_group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rates for all parameter groups based on current step, 
        applying cosine annealing and linear decay.
        """
        # Total steps taken so far (current step across all epochs)
        current_step = self.last_epoch + 1
       
        current_epoch = current_step // self.steps_per_epoch
        step_in_epoch = current_step % self.steps_per_epoch
        
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            # Linear decay of the base learning rate across all epochs
            current_initial_lr = base_lr - (current_epoch * self.lr_decay[i])
            
            # Apply cosine annealing within the epoch
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_epoch / self.steps_per_epoch))
            
            # Calculate the new learning rate, ensuring it stays within bounds
            new_lr = max(current_initial_lr * cosine_decay, self.min_lr)
            lrs.append(new_lr)
        
        return lrs