


import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Dueling Q-Network for Banana Picking."""

    def __init__(self, state_size, action_size, seed):
        """
        Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Number of possible actions
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # outputs V(s)
        )

        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)  # outputs A(s,a)
        )

    def forward(self, state):
        """
        Forward pass: outputs Q-values for all actions given a state.
        """
        x = self.shared(state)
        value = self.value_stream(x)        # shape: [batch_size, 1]
        adv = self.adv_stream(x)            # shape: [batch_size, action_size]

        # Combine value and advantage streams
        q_vals = value + (adv - adv.mean(dim=1, keepdim=True))
        return q_vals


























# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class QNetwork(nn.Module):
#     """Actor (Policy) Model."""

#     def __init__(self, state_size, action_size, seed):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#         """
#         super(QNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         "*** YOUR CODE HERE ***"
#         self.SeqLayer=nn.Sequential(
#             nn.Linear(state_size,128),
#             nn.ReLU(),
#             nn.Linear(128,64),
#             nn.ReLU(),
#             nn.Linear(64,action_size)
            
            
#         )

#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         return self.SeqLayer(state)



# import torch 
# import torch.nn as nn

# class QNetwork(nn.Module):
#     def __init__(self,action_size,state_size):
#         super().__init__()
        
#         self.Net=nn.Sequential(
#             nn.Linear(state_size,128),
#             nn.ReLU(),
#             nn.Linear(128,64),
#             nn.ReLU(),
#             nn.Linear(64,action_size)
#         )
        
#     def forward(self,state):
#         return self.Net(state)
        