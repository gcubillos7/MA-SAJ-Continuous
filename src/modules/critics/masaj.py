import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MASAJCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MASAJCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        # obs + n_agents + n_actions

        self.output_type = "q"
        obs_shape = self._get_input_shape(scheme)
        self.input_shape = obs_shape + self.n_actions if args.continuous_actions else obs_shape
        if getattr(args, "obs_role", False):
            self.input_shape += args.n_roles

        self.dim_out = 1 if args.continuous_actions else self.n_actions
        # Set up network layers

        self.use_layer_norm = getattr(args, "use_layer_norm ", False)
        self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim) 

        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, self.dim_out)

    def forward(self, inputs, actions=None):
        if actions is not None:
            inputs = th.cat([inputs, actions], dim=-1)

        x = self.fc1(inputs)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q  # bs, max_t, n_agents, (1 if args.continous_actions else self.n_actions)

    def _build_inputs(self, batch, bs, max_t):

        inputs = [batch["obs"],
        th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1)]

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)

        return inputs  # [bs, max_t, n_agents, n_agents + n_obs]

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["obs"]["vshape"]
        input_shape += self.n_agents
        return input_shape  # [n_agents + n_obs]


class MASAJRoleCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MASAJRoleCritic, self).__init__()

        self.args = args
        self.n_roles = args.n_roles
        self.n_agents = args.n_agents
        self.role_interval = args.role_interval

        use_role_value = getattr(args, "use_role_value", False)
        # obs + n_agents
        obs_shape = self._get_input_shape(scheme)
        self.input_shape = obs_shape + self.n_roles if use_role_value else obs_shape
        self.output_type = "q"

        self.dim_out = 1 if use_role_value else self.n_roles

        self.use_layer_norm = getattr(args, "use_layer_norm ", False)
        self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim) 

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, self.dim_out)

    def forward(self, inputs, roles=None):
        if roles is not None:
            inputs = th.cat([inputs, roles], dim=-1)

        x = self.fc1(inputs)
        if self.use_layer_norm:
                    x = self.layer_norm(x)
        x = F.relu(x) 
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        return q  # bs, role_t, n_agents, n_actions

    def _build_inputs(self, batch, bs, max_t):
        inputs = batch["obs"][:, :-1][:, ::self.role_interval]
        # t_role = np.ceil(max_t/self.role_interval)
        t_role = inputs.shape[1]
        inputs = [inputs,
                  th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, t_role, -1, -1)]

        inputs = th.cat([x.reshape(bs, t_role, self.n_agents, -1) for x in inputs], dim=-1)

        return inputs  # [bs, max_t, n_agents, n_agents + n_obs]

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["obs"]["vshape"]
        input_shape += self.n_agents
        return input_shape  # [n_agents + n_obs]
