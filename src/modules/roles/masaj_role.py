import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20

class MASAJRole(nn.Module):
    def __init__(self, args):
        super(MASAJRole, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.use_latent_normal = getattr(args, 'use_latent_normal', False)
        self.squash = self.args.squash
        
        if self.use_latent_normal:
            output_dim = args.action_latent_dim
        else:
            output_dim = args.n_actions
        
        self.mu_layer = nn.Linear(args.rnn_hidden_dim, output_dim)
        self.mu_layer.bias.data.fill_(0.0)

        self.use_std_layer = getattr(args, 'use_std_layer', True)

        max_std = np.exp(LOG_STD_MAX)
        min_std = np.exp(LOG_STD_MIN)
        init_bias_std = 1.0
        assert max_std > min_std, "max std has to be greater than min std"

        self.param = getattr(args, "parametrization", "exponential") 

        if self.use_std_layer:
            self.log_std_layer = nn.Linear(args.rnn_hidden_dim, output_dim)
            
            if self.param in ["exponential","exp"]:
                # scale parameter for init so it is around init_std (assumes output of layer centered at 0)
                self.init_param = np.log(init_bias_std)
                self.log_std_layer.bias.data.fill_(self.init_param)
                # add min_std to avoid clampings
                self.parametrize = lambda x: th.exp(x)
                self.exp = True
            elif self.param in ["sigmoid", "sig"]:
                # scale parameter for init so it is around init_std (assumes output of layer centered at 0)
                self.init_param = np.log(init_bias_std-min_std) - np.log(max_std + min_std - init_bias_std)     
                self.log_std_layer.bias.data.fill_(self.init_param)        
                self.parametrize = lambda x: max_std * th.sigmoid(x) + min_std
                self.exp = False
            else:
                AttributeError(f"Parametrization {args.parametrization} not recognized")
        else:

            log_std = th.log(th.ones((args.n_agents, args.n_actions), device = args.device) )
            self.log_std = nn.parameter.Parameter(data = log_std, requires_grad = True)
            self.parametrize = lambda x: th.exp(x)
            self.exp = True

        self.prior = None

    def forward(self, hidden):

        hidden = F.relu(hidden)
        latent_mu = self.mu_layer(hidden) 

        if self.use_std_layer:
            latent_log_std = self.log_std_layer(hidden) 
            if self.exp:
                latent_log_std = th.clamp(latent_log_std, LOG_STD_MIN, LOG_STD_MAX)
        else:
            latent_log_std = self.log_std
            latent_log_std = th.clamp(latent_log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        latent_std = self.parametrize(latent_log_std)

        return latent_mu, latent_std
        
    def update_prior(self, prior):
        self.prior = prior


class MASAJRoleDiscrete(nn.Module):
    def __init__(self, args):
        super(MASAJRoleDiscrete, self).__init__()
        self.args = args
        self.n_actions = args.n_actions

        self.fc = nn.Linear(args.rnn_hidden_dim, args.action_latent_dim)
        self.action_space = th.ones(args.n_actions).to(args.device)

    def forward(self, h, action_latent):
        role_key = self.fc(h)  # [bs, action_latent] [n_actions, action_latent]
        role_key = role_key.unsqueeze(-1)
        action_latent_reshaped = action_latent.unsqueeze(0).repeat(role_key.shape[0], 1, 1)

        dot = th.bmm(action_latent_reshaped, role_key).squeeze()

        return dot

    def update_action_space(self, new_action_space):
        self.action_space = th.Tensor(new_action_space).to(self.args.device).float()

