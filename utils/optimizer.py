import pdb
import torch
import numpy as np
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                model,
                lr=self.optim_dict['base_lr'],
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            parameters = []
            base_lr = self.optim_dict['learning_rate'].pop('base_lr')
            for n, p in model.named_children():
                lr_ = base_lr
                for m, lr in self.optim_dict['learning_rate'].items():
                    if m in n:
                        lr_ = lr
                print('learning rate {}={}'.format(n, lr_))
                parameters.append({'params': p.parameters(), 'lr': lr_})

            self.optimizer = optim.Adam(
                model.parameters(),
                # parameters,
                lr=base_lr,
                weight_decay=self.optim_dict['weight_decay']
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'])

    def define_lr_scheduler(self, optimizer, milestones):
        if self.optim_dict['scheduler'] == "consine":
            print("using CosineAnnealingLR....")
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=self.optim_dict['start_epoch'],
                T_max=self.optim_dict['num_epoch'],
            )
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
            return lr_scheduler
        else:
            raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
