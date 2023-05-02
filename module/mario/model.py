import copy
import torch
from torch import nn


class MarioNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Mini CNN structured model
        input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output

        Args:
            input_dim (int): image size
            output_dim (int): number of action

        Raises:
            ValueError: not matched height
            ValueError: not matched weight
        """
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input weight: 84, got: {w}")
        
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False
    
    def forward(self, input: torch.Tensor, model: str) -> nn.modules.container.Sequential:
        """
        Return predict/target model 

        Args:
            input (torch.Tensor): input image
            model (str): type of model

        Returns:
            nn.modules.container.Sequential: selected model
        """

        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
