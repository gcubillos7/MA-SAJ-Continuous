from gym import spaces
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
from modules.roles import REGISTRY as role_REGISTRY
from components.role_selectors import REGISTRY as role_selector_REGISTRY
from torch.distributions.normal import Normal
import torch as th
import numpy as np
from itertools import cycle
import copy
import torch.nn.functional as F

# This multi-agent controller shares parameters between agents
class ROLEMAC:
    def __init__(self, scheme, groups, args):
        
        self.deactivate_roles = False
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.continuous_actions = args.continuous_actions
        self.role_interval = args.role_interval
        self.n_roles = args.n_roles
        self.n_clusters = args.n_role_clusters
        self.agent_output_type = getattr(args, "agent_output_type", None)
        self._get_input_shape = self._get_input_shape_continous if self.continuous_actions else self._get_input_shape_discrete
        
        input_shape = self._get_input_shape(scheme)

        self._build_agents(input_shape)
        self._build_roles()

        # Selectors
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.role_selector = role_selector_REGISTRY[args.role_selector](input_shape, args)
        action_encoder = getattr(args, "action_encoder", None)
        self.action_encoder = action_encoder_REGISTRY[action_encoder](args) if action_encoder else action_encoder 

        # Temp variables 
        self.hidden_states = None
        self.role_hidden_states = None
        self.selected_roles = None

        # Role latent and actions representations
        if not self.continuous_actions:
            self._build_discrete(args)
        else:
            self._build_continous(args)
            for _aid in range(self.n_agents):
                for _actid in range(self.args.action_spaces[_aid].shape[0]):
                    print(_aid,_actid, (self.args.action_spaces[_aid].low[_actid]),
                                                            (self.args.action_spaces[_aid].high[_actid])) 

    def select_actions_continuous(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        with th.no_grad():
            agent_outputs, _, _, _ = self.forward(ep_batch, t=t_ep, test_mode=test_mode, t_env=t_env)
        roles = self.selected_roles.view(ep_batch.batch_size, self.n_agents, -1)[bs]
        # actions = agent_outputs[bs]
        actions = self.add_exploration(agent_outputs[bs], ep_batch.batch_size, t_env, test_mode)
        return actions, roles

    def select_actions_discrete(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):

        # Chose a role and then an action, (agent_outputs are masked)
        with th.no_grad():
            agent_outputs, _, _, _ = self.forward(ep_batch, t=t_ep, test_mode=test_mode, t_env=t_env)

            avail_actions = ep_batch["avail_actions"][:, t_ep]

            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                                test_mode=test_mode)

            selected_roles = self.selected_roles.view(ep_batch.batch_size, self.n_agents, -1)

        return chosen_actions, selected_roles[bs]

    def get_avail_actions_role(self, batch_size):
        # self.selected_roles [BS*n_agents]
        role_index = self.selected_roles.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.n_actions).long()
        role_avail_actions = th.gather(self.role_action_spaces.unsqueeze(0).expand(batch_size * self.n_agents, -1, -1),
                                       dim=1, index=role_index)
        role_avail_actions = role_avail_actions[:, 0]
        return role_avail_actions.view(batch_size, self.n_agents, -1)

    def discrete_forward(self, ep_batch, t, test_mode=False, t_env=None):

        # self.action_selector.logger = self.logger
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_inputs = self._build_inputs(ep_batch, t)
        batch_size = ep_batch.batch_size

        self.role_hidden_states = self.role_agent(agent_inputs, self.role_hidden_states)
        
        pi_role = None
        selected_roles = None
        # select a role every self.role_interval steps
        if t % self.role_interval == 0:
            role_logits = self.role_selector(self.role_hidden_states, self.role_latent)
            role_pis = self.softmax_roles(role_logits, batch_size, test_mode=test_mode)
            # Get Index of the role of each agent
            selected_roles = self.role_selector.select_role(role_logits, test_mode=test_mode,
                                                                        t_env=t_env)

            self.selected_roles = selected_roles

            self.role_avail_actions = self.get_avail_actions_role(batch_size)

            selected_roles = self.selected_roles.unsqueeze(-1).view(batch_size, self.n_agents, -1)
            
            pi_role = role_pis.view(batch_size, self.n_agents, -1)

        # compute individual hidden_states for each agent
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        pi_action, log_p_action = self.discrete_actions_forward(batch_size, avail_actions, t_env, test_mode)

        return pi_action, log_p_action, selected_roles, pi_role

    def continuous_forward(self, ep_batch, t, test_mode=False, t_env=None):
        # self.action_selector.logger = self.logger
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_inputs = self._build_inputs(ep_batch, t)
        batch_size = ep_batch.batch_size
        
        self.role_hidden_states = self.role_agent(agent_inputs, self.role_hidden_states)
        
        pi_role = None
        selected_roles = None
        # select a role every self.role_interval steps
        if t % self.role_interval == 0:
            role_logits = self.role_selector(self.role_hidden_states, self.role_latent)
            role_pis = self.softmax_roles(role_logits, batch_size, test_mode=test_mode)
            # Get Index of the role of each agent
            selected_roles = self.role_selector.select_role(role_pis, test_mode=test_mode,
                                                                        t_env=t_env)
            if self.deactivate_roles:
                selected_roles = th.zeros(self.n_agents, device = ep_batch.device, dtype = th.long).unsqueeze(0).expand(batch_size, self.n_agents).reshape(-1)
            
            self.selected_roles = selected_roles
            selected_roles = selected_roles.unsqueeze(-1).view(batch_size, self.n_agents, -1)
            
            pi_role = role_pis.view(batch_size, self.n_agents, -1)

        # compute individual hidden_states for each agent
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        pi_action, log_p_action = self.continuos_actions_forward(batch_size, avail_actions, t_env, test_mode)

        return pi_action, log_p_action, selected_roles, pi_role

    def add_exploration(self, chosen_actions, batch_size, t_env, test_mode):

        if not (t_env is None) and (not test_mode):
            exploration_mode = getattr(self.args, "exploration_mode", "gaussian")
            if exploration_mode == "gaussian":
                if t_env < self.start_steps:
                    if getattr(self.args.env_args, "scenario_name", None) is None or self.args.env_args[
                                "scenario_name"] in ["Humanoid-v2", "HumanoidStandup-v2"]:
                        chosen_actions = th.from_numpy(np.array(
                            [[self.args.action_spaces[0].sample() for i in range(self.n_agents)] for _ in
                             range(batch_size)])).float()
                    else:
                        chosen_actions = th.from_numpy(np.array(
                            [[self.args.action_spaces[i].sample() for i in range(self.n_agents)] for _ in
                             range(batch_size)])).float() 
                    chosen_actions = chosen_actions

                elif self.start_steps <= t_env < self.stop_steps:
                    x = chosen_actions.detach().clone().zero_()
                    chosen_actions += self.act_noise * x.normal_()
                
                    if all([isinstance(act_space, spaces.Box) for act_space in self.args.action_spaces]):
                        chosen_actions = chosen_actions.clamp(self.args.actions_min.to(chosen_actions.device), self.args.actions_max.to(chosen_actions.device))

        return chosen_actions       

    def continuos_actions_forward(self, batch_size, avail_actions, t_env, test_mode):

        actions, log_p_action, kl_loss = [], [], []
        hidden_states = self.hidden_states.view(batch_size, self.n_agents, -1)

        for role_i in range(self.n_roles):
            
            mu, sigma = self.roles[role_i](hidden_states) # [bs, n_agents, n_actions]
            prior = self.roles[role_i].prior

            pi_action, log_p_action_taken, dkl_loss = self.action_selector.select_action(mu, sigma, prior=prior,
                                                                                         test_mode=test_mode,
                                                                                         t_env=t_env)
            actions.append(pi_action)

            if not test_mode:
                log_p_action.append(log_p_action_taken)
                kl_loss.append(dkl_loss)

        if not test_mode:
            if self.use_latent_normal:
                kl_loss = th.stack(kl_loss, dim=-1)  # [bs*n_agents, n_roles]
                kl_loss = kl_loss.view(batch_size * self.n_agents, -1)
                kl_loss = kl_loss.gather(index = self.selected_roles.unsqueeze(-1).expand(-1, self.n_roles), dim=-1)
                kl_loss = kl_loss[:, 0]
                kl_loss = kl_loss.view(batch_size, self.n_agents)
                self.kl_loss = kl_loss
            log_p_action = th.stack(log_p_action, dim=-1)  # [bs, n_agents, n_roles]
            log_p_action = log_p_action.view(batch_size * self.n_agents, -1) # [bs*n_agents, n_roles]
            log_p_action = log_p_action.gather(index=self.selected_roles.unsqueeze(-1).expand(-1, self.n_roles), dim=-1)
            log_p_action = log_p_action[:, 0]
            log_p_action = log_p_action.view(batch_size, self.n_agents)  # [bs,n_agents]
        
        actions = th.stack(actions, dim=-1)  # [bs*n_agents, dim_actions, n_roles]
        # for each batch/agent select an action
        actions = actions.view(batch_size * self.n_agents, self.n_actions, -1)
        actions = actions.gather(
            index = self.selected_roles.unsqueeze(-1).unsqueeze(-1).expand(-1, self.n_actions, self.n_roles), dim=-1)
        actions = actions[..., 0]
        actions = actions.view(batch_size, self.n_agents, self.n_actions)

        return actions, log_p_action

    def discrete_actions_forward(self, batch_size, avail_actions, t_env, test_mode):

        pi = []
        for role_i in range(self.n_roles):
            pi_out = self.roles[role_i](self.hidden_states, self.action_repr)
            pi_out = pi_out.view(batch_size, self.n_agents, self.n_actions)
            pi.append(pi_out)

        pi = th.stack(pi, dim=-1)  # [batch_size, self.n_agents, self.n_actions, n_roles]
        pi = pi.view(batch_size * self.n_agents, self.n_actions,
                     -1)  # [batch_size*self.n_agents*self.n_actions, n_roles]
        pi = pi.gather(index=self.selected_roles.unsqueeze(-1).unsqueeze(-1).expand(-1, self.n_actions, self.n_roles),
                       dim=-1)
        pi = pi[..., 0]

        if self.agent_output_type == "pi_logits":
            pi = self.softmax_actions(pi, batch_size, avail_actions, test_mode)

        pi = pi.view(batch_size, self.n_agents, -1)

        return pi, None

    def softmax_roles(self, role_outs, batch_size, test_mode):

        role_outs = F.softmax(role_outs, dim=-1)

        if not test_mode:
            # Epsilon floor
            epsilon_action_num = role_outs.size(-1)

            role_outs = ((1 - self.role_selector.epsilon) * role_outs
                         + th.ones_like(role_outs) * self.role_selector.epsilon / epsilon_action_num)

        return role_outs

    def softmax_actions(self, agent_outs, batch_size, avail_actions, test_mode):

        # Apply role mask (is applied before softmax to avoid no action available)
        role_avail_actions = self.role_avail_actions.reshape(batch_size * self.n_agents, -1)
        agent_outs[role_avail_actions == 0] = -1e11

        # Apply mask and softmax
        if getattr(self.args, "mask_before_softmax", True):
            # Make the logits for unavailable actions very negative to minimize their affect on the softmax
            reshaped_avail_actions = avail_actions.reshape(batch_size * self.n_agents, -1)

            agent_outs[reshaped_avail_actions == 0] = -1e11

        agent_outs = F.softmax(agent_outs, dim=-1)

        if not test_mode:
            # Epsilon floor
            epsilon_action_num = agent_outs.size(-1)
            if getattr(self.args, "mask_before_softmax", True):
                # With probability epsilon, we will pick an available action uniformly
                epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

            agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                          + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

            if getattr(self.args, "mask_before_softmax", True):
                # Zero out the unavailable actions
                agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(batch_size, self.n_agents, -1)

    def _build_role_latent(self, role_latent = None):

        if role_latent is None:
            self.role_latent = th.nn.Parameter(th.eye(self.n_roles, device=self.args.device), requires_grad = False)
        elif getattr(self.args, 'add_role_id', False):
            I = th.eye(self.n_roles, device=self.args.device)
            self.role_latent = th.nn.Parameter(th.cat([role_latent, I], dim = -1), requires_grad = False)
        else:
            self.role_latent = th.nn.Parameter(role_latent, requires_grad = False)

    def update_prior(self, role_i, mu, sigma):
        prior = Normal(mu, sigma)
        self.roles[role_i].update_prior(prior)

    def update_decoder(self, decoder):
        self.action_selector.update_decoder(decoder)

    def get_kl_loss(self):
        return self.kl_loss

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.role_hidden_states = self.role_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav


    def parameters(self):
        params = list(self.agent.parameters())
        
        for role_i in range(self.n_roles):
            params += list(self.roles[role_i].parameters())
        if not self.deactivate_roles:
            params += list(self.role_agent.parameters())
            params += list(self.role_selector.parameters())
        params += list(self.role_latent)

        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.role_agent.load_state_dict(other_mac.role_agent.state_dict())
        if other_mac.n_roles > self.n_roles:
            self.n_roles = other_mac.n_roles
            self.roles = copy.deepcopy(other_mac.roles)
        else:
            for role_i in range(self.n_roles):
                self.roles[role_i].load_state_dict(other_mac.roles[role_i].state_dict())

        self.role_selector.load_state_dict(other_mac.role_selector.state_dict())
        self.action_encoder.load_state_dict(other_mac.action_encoder.state_dict())
        self.role_latent = copy.deepcopy(other_mac.role_latent)
        self.action_repr = copy.deepcopy(other_mac.action_repr)

    def cuda(self):
        self.agent.cuda()
        self.role_agent.cuda()
        for role_i in range(self.n_roles):
            self.roles[role_i].cuda()
        self.role_selector.cuda()
        if self.action_encoder:
            self.action_encoder.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), f"{path}/agent.th")
        th.save(self.role_agent.state_dict(), f"{path}/role_agent.th")
        for role_i in range(self.n_roles):
            th.save(self.roles[role_i].state_dict(), f"{path}/role_{role_i}.th")
        th.save(self.role_selector.state_dict(), f"{path}/role_selector.th")
        if self.action_encoder:
            th.save(self.action_encoder.state_dict(), f"{path}/action_encoder.th")
        th.save(self.role_action_spaces, f"{path}/role_action_spaces.pt")
        th.save(self.role_latent, f"{path}/role_latent.pt")
        th.save(self.action_repr, f"{path}/action_repr.pt")

    def load_models(self, path):
        self.n_roles = self.role_action_spaces.shape[0]
        self.agent.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))
        self.role_agent.load_state_dict(
            th.load(f"{path}/role_agent.th", map_location=lambda storage, loc: storage))
        for role_i in range(self.n_roles):
            try:
                self.roles[role_i].load_state_dict(th.load(f"{path}/role_{role_i}.th",
                                                           map_location=lambda storage, loc: storage))
            except:
                self.roles.append(role_REGISTRY[self.args.role](self.args))
            if self.args.use_cuda:
                self.roles[role_i].cuda()

        self.role_selector.load_state_dict(th.load(f"{path}/role_selector.th",
                                                   map_location=lambda storage, loc: storage))
        if self.action_encoder:
            self.action_encoder.load_state_dict(th.load(f"{path}/action_encoder.th",
                                                    map_location=lambda storage, loc: storage))

        self.role_latent = th.load(f"{path}/role_latent.pt",
                                   map_location=lambda storage, loc: storage).to(self.args.device)
        self.action_repr = th.load(f"{path}/action_repr.pt",
                                   map_location=lambda storage, loc: storage).to(self.args.device)

    def _build_agents(self, input_shape):
        # agent w/o roles policy 
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        # rol selector policy
        self.role_agent = agent_REGISTRY[self.args.role_agent](input_shape, self.args)

    def _build_roles(self):
        self.roles = [role_REGISTRY[self.args.role](self.args) for _ in range(self.n_roles)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if self.continuous_actions:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t - 1])
            else:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
                else:
                    inputs.append(batch["actions_onehot"][:, t - 1])

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        
        return inputs

    def _get_input_shape_continous(self, scheme):  # Continuous
        input_shape = scheme["obs"]["vshape"]

        # Add agent ID to input
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        if self.args.obs_last_action:
            if self.continuous_actions:
                input_shape += scheme["actions"]["vshape"][0]
            else:
                input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape

    def _get_input_shape_discrete(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def update_roles(self):
        action_repr = self.action_encoder()
        raise NotImplementedError

    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])

    def _build_continous(self, args):
        self.start_steps = getattr(args, "start_steps", 0)
        self.stop_steps = getattr(args, "stop_steps", 0)
        self.act_noise = getattr(args, "act_noise", 0.0)

        self.use_latent_normal = getattr(args, "use_latent_normal", False)

        if self.use_latent_normal:
            role_latent = th.ones(self.n_roles, self.args.action_latent_dim).to(args.device)
        else:
            role_latent = None

        self._build_role_latent(role_latent)
        self.mask_before_softmax = args.mask_before_softmax
        self.forward = self.continuous_forward
        self.select_actions = self.select_actions_continuous
        self.kl_loss = None

    def _build_discrete(self, args):
        from sklearn.cluster import KMeans
        role_latent = th.ones(self.n_roles, self.args.action_latent_dim).to(args.device)
        self._build_role_latent(role_latent)
        self.forward = self.discrete_forward
        self.select_actions = self.select_actions_discrete
        np.random.seed(0)
        self.action_repr = th.from_numpy(np.random.rand(self.n_actions, self.args.action_latent_dim)).float().to(
            args.device)
        self.role_action_spaces = th.nn.Parameter(th.ones(self.n_roles, self.n_actions).to(args.device), requires_grad = False)

    def update_role_action_spaces(self):
        """
        (Discrete)
        Action spaces from rode 
        https://github.com/TonghanWang/RODE
        (SC2 Only)
        """

        action_repr = self.action_encoder()
        action_repr_array = action_repr.detach().cpu().numpy()  # [n_actions, action_latent_d]

        k_means = KMeans(n_clusters=self.n_clusters, random_state=0).fit(action_repr_array)

        spaces = []
        for cluster_i in range(self.n_clusters):
            spaces.append((k_means.labels_ == cluster_i).astype(np.float))

        o_spaces = copy.deepcopy(spaces)
        spaces = []

        for space_i, space in enumerate(o_spaces):
            _space = copy.deepcopy(space)
            _space[0] = 0.
            _space[1] = 0.

            if _space.sum() == 2.:
                spaces.append(o_spaces[space_i])
            if _space.sum() >= 3:
                _space[:6] = 1.
                spaces.append(_space)

        for space in spaces:
            space[0] = 1.

        spaces = np.unique(spaces, axis=0)
        spaces = list(spaces)
        n_roles = len(spaces)

        cyclic_spaces = cycle(spaces)

        while n_roles < self.n_roles:
            spaces.append(next(cyclic_spaces))
            n_roles += 1

        if n_roles > self.n_roles:
            for _ in range(self.n_roles, n_roles):
                self.roles.append(role_REGISTRY[self.args.role](self.args))
                if self.args.use_cuda:
                    self.roles[-1].cuda()
            self.n_roles = n_roles

        print('>>> Role Action Spaces', spaces)

        for role_i, space in enumerate(spaces):
            self.roles[role_i].update_action_space(space)

        self.role_action_spaces.data = th.Tensor(np.array(spaces)).to(self.args.device).float()  # [n_roles, n_actions]

        self.role_latent.data = th.matmul(self.role_action_spaces, action_repr) / self.role_action_spaces.sum(dim=-1,
                                                                                                         keepdim=True)
        self.role_latent.requires_grad = False
        self.action_repr.requires_grad = False
