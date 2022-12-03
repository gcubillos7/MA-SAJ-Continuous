import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.non_linearity = F.relu
        self.use_layer_norm = getattr(args, "use_layer_norm ", False)
        self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim) 
    
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, actions=None):
        x = self.fc1(inputs)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x