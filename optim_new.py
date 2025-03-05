import numpy as np
from torch.optim import lr_scheduler

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ScheduledOptim():

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)  # note:it‘s the d_model

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class ScheduledOptim_new():

    def __init__(self, optimizer, d_model,init_lr, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        # self.init_lr = np.power(d_model, -0.5)  # 1/ sqrt(840) = 0.03226
        self.init_lr = init_lr
        self.d_model = d_model

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        # 预热策略的效果是在训练初期，学习率较大以便更快地收敛，然后逐渐降低学习率，使模型更加稳定。这有助于避免训练初期出现梯度爆炸等问题。
        # return np.min([
        #     np.power(self.n_current_steps, -0.5),
        #     np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])  #np.power(10000, -1.5) =  0.00000001
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_current_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warm_up=0, T_max=10, start_ratio=0.1):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min

        Example:
            >>> epochs = 100
            >>> warm_up = 5
            >>> cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            >>> lrs = []
            >>> for epoch in range(epochs):
            >>>     optimizer.step()
            >>>     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            >>>     cosine_lr.step()
            >>> plt.plot(lrs, color='r')
            >>> plt.show()

        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.T_max = T_max
        self.start_ratio = start_ratio
        self.cur = 0  # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (
                np.cos((self.cur - self.warm_up) / (self.T_max - self.warm_up) * np.pi) + 1
            )

        self.cur += 1

        return [lr for base_lr in self.base_lrs]