from gym import spaces
import torch as th
import numpy as np

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
from modules.roles import REGISTRY as role_REGISTRY


# This multi-agent controller shares parameters between agents
class CMAC:
    def __init__(self, scheme, groups, args):
        
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.continuous_actions = args.continuous_actions
        self.agent_output_type = getattr(args, "agent_output_type", None)
        self._get_input_shape = self._get_input_shape_continous
        
        input_shape = self._get_input_shape(scheme)

        self._build_agents(input_shape)
        # Selectors
        self.action_selector = action_REGISTRY[args.action_selector](args)

        # Temp variables 
        self.hidden_states = None
        self.t_env = 0
        self._build_continous(args)

        for _aid in range(self.n_agents):
            for _actid in range(self.args.action_spaces[_aid].shape[0]):
                print("action space")
                print(_aid,_actid, (self.args.action_spaces[_aid].low[_actid]),
                                                        (self.args.action_spaces[_aid].high[_actid])) 

    def select_actions_continuous(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        with th.no_grad():
            (agent_outputs, _) = self.forward(ep_batch, t=t_ep, test_mode=test_mode, t_env=t_env)
        actions = self.add_exploration(agent_outputs[bs], ep_batch.batch_size, t_env, test_mode)
        return actions
  
    def continuous_forward(self, ep_batch, t, test_mode=False, t_env=None):
        # self.action_selector.logger = self.logger
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_inputs = self._build_inputs(ep_batch, t)
        batch_size = ep_batch.batch_size

        # compute individual hidden_states for each agent
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        (chosen_actions, log_p_action) = self.continuos_actions_forward(batch_size, avail_actions, t_env, test_mode)

        return (chosen_actions, log_p_action)

    def add_exploration(self, chosen_actions, batch_size, t_env, test_mode):
        if (not test_mode):
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
                    chosen_actions = chosen_actions/0.4

                elif self.start_steps <= t_env < self.stop_steps:
                    x = chosen_actions.detach().clone().zero_()
                    chosen_actions += self.act_noise * x.normal_()
                
                    if all([isinstance(act_space, spaces.Box) for act_space in self.args.action_spaces]):
                        chosen_actions = chosen_actions.clamp(self.args.actions_min.to(chosen_actions.device)/0.4, self.args.actions_max.to(chosen_actions.device)/0.4)
                        #/0.4 to fix humanoid action lim
        return chosen_actions       

    def continuos_actions_forward(self, batch_size, avail_actions, t_env, test_mode):

        hidden_states = self.hidden_states.view(batch_size, self.n_agents, -1)
        mu, sigma = self.policy(hidden_states) # [bs, n_agents, n_actions]
       
        chosen_actions, log_p_action_taken, _ = self.action_selector.select_action(mu, sigma, prior=None,
                                                                                         test_mode=test_mode,
                                                                                         t_env=t_env)
        chosen_actions = chosen_actions.view(batch_size, self.n_agents, self.n_actions)

        if log_p_action_taken is not None:
            log_p_action_taken =  log_p_action_taken.view(batch_size, self.n_agents)

        return chosen_actions, log_p_action_taken

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        params = list(self.agent.parameters())
        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.policy.load_state_dict(other_mac.policy.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.policy.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), f"{path}/agent.th")
        th.save(self.policy.state_dict(), f"{path}/policy.th")

    def load_models(self, path):
        self.agent.load_state_dict(th.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(th.load(f"{path}/policy.th", map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.policy = role_REGISTRY[self.args.role](self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if t == 0:
            inputs.append(th.zeros_like(batch["actions"][:, t]))
        else:
            inputs.append(batch["actions"][:, t - 1])

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


    def _build_continous(self, args):
        self.start_steps = getattr(args, "start_steps", 0)
        self.stop_steps = getattr(args, "stop_steps", 0)
        self.act_noise = getattr(args, "act_noise", 0.0)
        self.forward = self.continuous_forward
        self.select_actions = self.select_actions_continuous