import os
import torch
from avalanche.core import BasePlugin

class CheckpointPlugin(BasePlugin):
    def __init__(self, folder="checkpoints", save_every_exp=True):
        self.folder = folder
        self.save_every_exp = save_every_exp
        os.makedirs(folder, exist_ok=True)

    def after_training_exp(self, strategy, **kwargs):
        if self.save_every_exp:
            path = f"{self.folder}/model_checkpoint.pth"
            torch.save({
                "model_state": strategy.model.state_dict(),
                "optimizer_state": strategy.optimizer.state_dict(),
            }, path)
            print(f"Checkpoint saved: {path}")
    

