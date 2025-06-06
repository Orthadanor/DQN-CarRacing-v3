import torch as torch 
import torch.nn as nn


import torch 
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size:int, action_size:int, hidden_size:int=256,non_linear:nn.Module=nn.ReLU):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        hidden_size: int
            The number of neurons in the hidden layer

        This is a seperate class because it may be useful for the bonus questions
        """
        super(MLP, self).__init__()
        # ========== YOUR CODE HERE ==========
        # TODO:
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_size)
        self.non_linear = non_linear()
        # ====================================
    


        # ========== YOUR CODE ENDS ==========

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # ========== YOUR CODE HERE ==========
        output = self.output(self.non_linear(self.linear1(x)))
        return output
    


        # ========== YOUR CODE ENDS ==========
        return x

class Nature_Paper_Conv(nn.Module):
    """
    A class that defines a neural network with the following architecture:
    - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
    - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
    - 1 convolutional layer with 64 3x3 kernels with a stride of 1x1 w/ ReLU activation
    - 1 fully connected layer with 512 neurons and ReLU activation. 
    Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
    """
    def __init__(self, input_size:tuple[int], action_size:int,**kwargs):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        **kwargs: dict
            additional kwargs to pass for stuff like dropout, etc if you would want to implement it
        """
        super(Nature_Paper_Conv, self).__init__()
        # ========== YOUR CODE HERE ==========

        self.CNN = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size[0], 
                out_channels=32, 
                kernel_size=8, 
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=4, 
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=64, 
                kernel_size=3, 
                stride=1
            ),
            nn.ReLU()
        )
        
        self.MLP = MLP(
            input_size=64*7*7, 
            action_size=action_size, 
            hidden_size=512,
            non_linear=nn.ReLU
        )


        # ========== YOUR CODE ENDS ==========

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # ========== YOUR CODE HERE ==========
        hidden_output = self.CNN(x)
        
        # Flatten the output of the last convolutional layer
        post_conv_output = hidden_output.view(hidden_output.size(0), -1)
        # fc_input_size = post_conv_output.size(1)
    
        output = self.MLP(post_conv_output)
    
        # ========== YOUR CODE ENDS ==========
        return output
