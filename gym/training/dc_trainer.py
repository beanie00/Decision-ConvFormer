import numpy as np
import torch

from training.trainer import Trainer


class DecisionConvFormerTrainer(Trainer):
        
    def train_step(self):
        
        states, actions, rewards, dones, rtg, timesteps, traj_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        action_preds = self.model.forward(states, actions, rtg[:,:-1], timesteps)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[traj_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[traj_mask.reshape(-1) > 0]

        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
