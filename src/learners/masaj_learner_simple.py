import copy
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.masaj import MASAJCritic
from modules.mixers.qmix import QMixer
from modules.mixers.fop import FOPMixer
from utils.rl_utils import build_td_lambda_targets, polyak_update
from torch.optim import RMSprop, Adam, AdamW
from modules.critics.value import ValueNet
from components.epsilon_schedules import DecayThenFlatSchedule


class MASAJ_Simple:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.continuous_actions = args.continuous_actions
        self.logger = logger

        self.mac = mac
        self.mac.logger = logger

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.double_value = getattr(args, "double_value", False)
        self.use_target_actor = getattr(args, "use_target_actor", False)
        self.use_local_entropy = getattr(args, "use_local_entropy", True)
        
        self.critic1 = MASAJCritic(scheme, args)
        self.critic2 = MASAJCritic(scheme, args)

        self.mixer1 = FOPMixer(args) #FOPMixer(args)
        self.mixer2 = FOPMixer(args)

        if self.use_target_actor:
            self.target_mac = copy.deepcopy(self.mac)

        self.target_mixer1 = copy.deepcopy(self.mixer1)
        self.target_mixer2 = copy.deepcopy(self.mixer2)

        if self.use_local_entropy:
            assert self.use_local_entropy == hasattr(self.mixer1, "toggle_alpha_train"), "Mixer should implement local alpha"
            self.mixer1.toggle_alpha_train(self.use_local_entropy)
            self.mixer2.toggle_alpha_train(self.use_local_entropy)

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.critic_params1 = list(self.critic1.parameters()) + list(self.mixer1.parameters())
        self.critic_params2 = list(self.critic2.parameters()) + list(self.mixer2.parameters())

        # self.scope = self.args.runner_scope

        self.value_params = []
        if self.double_value:
            self.value_params2 = []

        if self.continuous_actions:
            self._get_policy = self._get_policy_continuous
            self.value = ValueNet(scheme, args)
            self.value_params += list(self.value.parameters())
            self.target_value = copy.deepcopy(self.value)
            if self.double_value:
                self.value2 = ValueNet(scheme, args)
                self.value_params2 += list(self.value2.parameters())
                self.target_value2 = copy.deepcopy(self.value2)
            self.use_latent_normal = getattr(args, "use_latent_normal", False)
        else:
            self._get_policy = self._get_policy_discrete

        self.agent_params = list(mac.parameters())

        self.device = args.device

        if not hasattr(self.args, "critic_grad_norm_clip"):
            self.args.critic_grad_norm_clip = self.args.grad_norm_clip

        self.eps = 1e-10

        self._setup_optimizers(args)
        self._build_ent_coefficient(args)

        self.debug_log = False


    def _setup_optimizers(self, args):

        if getattr(self.args, "optimizer", "rmsprop") in ["rmsprop", "rms"]:

            self.p_optimizer = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha,
                                       eps=args.optim_eps)

            self.c_optimizer1 = RMSprop(params=self.critic_params1, lr=args.c_lr,
                                        alpha=args.optim_alpha, eps=args.optim_eps)

            self.c_optimizer2 = RMSprop(params=self.critic_params2, lr=args.c_lr,
                                        alpha=args.optim_alpha, eps=args.optim_eps)

            if self.continuous_actions:
                self.val_optimizer = RMSprop(params=self.value_params, lr=args.v_lr, alpha=args.optim_alpha,
                                             eps=args.optim_eps)
                if self.double_value:
                    self.val_optimizer2 = RMSprop(params=self.value_params2, lr=args.v_lr, alpha=args.optim_alpha,
                                                  eps=args.optim_eps)

        elif getattr(self.args, "optimizer", "rmsprop")  in ["adam", "adamw"]:
            # added small weight_decay to improve stability
            self.p_optimizer = AdamW(params=self.agent_params, lr=args.lr,
                                    eps=getattr(args, "optim_eps", 1e-7),
                                    weight_decay=getattr(args, "weight_decay", 1e-5))

            self.c_optimizer1 = AdamW(params=self.critic_params1, lr=args.c_lr,
                                     # alpha=args.optim_alpha,
                                     eps=getattr(args, "optim_eps", 1e-7),
                                     weight_decay=getattr(args, "weight_decay", 1e-5))

            self.c_optimizer2 = AdamW(params=self.critic_params2, lr=args.c_lr,
                                     eps=getattr(args, "optim_eps", 1e-7),
                                     weight_decay=getattr(args, "weight_decay", 1e-5))

            if self.continuous_actions:
                self.val_optimizer = AdamW(params=self.value_params, lr=args.v_lr,
                                          eps=getattr(args, "optimizer_epsilon", 1e-7),
                                          weight_decay=getattr(args, "weight_decay", 1e-5))
                if self.double_value:
                    self.val_optimizer2 = AdamW(params=self.value_params2, lr=args.v_lr,
                                               eps=getattr(args, "optimizer_epsilon", 1e-7),
                                               weight_decay=getattr(args, "weight_decay", 1e-5))


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        
        # update critic and local entropy coef
        self.train_critic(batch, t_env)
        # update policy and global entropy coef
        self.train_actor(batch, t_env)

        if hasattr(self.mixer1, "update_alpha"):
            with th.no_grad():
                [mixer.update_alpha(alpha = self.get_alpha(t_env)) for mixer in [self.mixer1, self.mixer2]]

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _get_policy_discrete(self, batch, avail_actions, test_mode=False, use_target_actor = False):
        """
        Returns
        mac_out: returns distribution of actions .log_p(actions)
        """

        if use_target_actor:
            mac = self.target_mac
        else:
            mac = self.mac

        mac_out = []
        mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t, test_mode=test_mode)
            mac_out.append(agent_outs[0])

        pi_act = th.stack(mac_out, dim=1)
        
        # output is the full policy
        pi_act[avail_actions == 0] = self.eps
        pi_act = pi_act / pi_act.sum(dim=-1, keepdim=True)
        pi_act[avail_actions == 0] = self.eps

        pi = pi_act.clone()
        log_p_out = th.log(pi)
        mac_out = (pi_act, log_p_out)  # [..., n_actions], [..., n_actions]

        return mac_out

    def _get_policy_continuous(self, batch, avail_actions, test_mode=False, use_target_actor = False):
        """
        Returns
        mac_out: returns distribution of actions .log_p(actions)
        """
        if use_target_actor:
            mac = self.target_mac
        else:
            mac = self.mac

        # Get  mac policy
        mac_out = []
        log_p_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t, test_mode=test_mode)
            mac_out.append(agent_outs[0])
            log_p_out.append(agent_outs[1])

        # Outputs is action, log_p
        action_taken, log_p_action = (th.stack(mac_out, dim=1), th.stack(log_p_out, dim=1))
        mac_out = (action_taken, log_p_action)  # [BS, T, AGENTS], [...]
        
        return mac_out

    def _get_joint_q_target(self, target_inputs, states, next_action, alpha):
        """
        Get Q Joint Target
        # Input Shape shape [Bs, T,...] [Bs, TRole,...] (for TD lambda) 
        # Output Shape [Bs, T,...] [Bs, TRole,...] (for TD lambda) 
        """


        with th.no_grad():
            if self.continuous_actions:
                next_action_input = next_action.detach()
                q_vals_taken1 = self.target_critic1.forward(target_inputs, next_action_input)  # [...]
                q_vals_taken2 = self.target_critic2.forward(target_inputs, next_action_input)  # [...]
                vs1, vs2 = self._get_target_value(target_inputs) 
                # vs1, vs2 = self._get_value(target_inputs)  # [...]
                vs1 = vs1.detach()
                vs2 = vs2.detach()
            else:
                next_action_input = F.one_hot(next_action, num_classes=self.n_actions)
                q_vals1 = self.target_critic1.forward(target_inputs)  # [..., n_actions]
                q_vals2 = self.target_critic2.forward(target_inputs)  # [..., n_actions]

                q_vals_taken1 = th.gather(q_vals1, dim=3, index=next_action).squeeze(3)  # [...]
                q_vals_taken2 = th.gather(q_vals2, dim=3, index=next_action).squeeze(3)  # [...]

                vs1 = th.logsumexp(q_vals1 / (alpha + self.eps), dim=-1) * (alpha + self.eps)  # [...]
                vs2 = th.logsumexp(q_vals2 / (alpha + self.eps), dim=-1) * (alpha + self.eps)  # [...]

        # Get Q joint for actions (using individual Qs and Vs)
        q_vals1 = self.target_mixer1(q_vals_taken1, states, actions=next_action_input, vs=vs1)  # reduces n_agents
        q_vals2 = self.target_mixer2(q_vals_taken2, states, actions=next_action_input, vs=vs2)  # reduces n_agents
        target_q_vals = th.min(q_vals1, q_vals2)

        return target_q_vals.detach()

    def _get_joint_q(self, inputs, states, action, action_onehot, alpha, mask = None):
        """
        Get joint q
        # Input shape shape [Bs, T,...] [Bs, TRole,...]
        # Output shape [Bs*T]
        """

        # Get Q and V values for actions
        if self.continuous_actions:
            action_input = action
            q_vals_taken1 = self.critic1.forward(inputs, action_input)  # last q value isn't used
            q_vals_taken2 = self.critic2.forward(inputs, action_input)  # [...]
            with th.no_grad():
                vs1, vs2 = self._get_value(inputs)
                vs1 = vs1.detach()
                vs2 = vs2.detach()
        else:
            action_input = action_onehot
            q_vals1 = self.critic1.forward(inputs)  # [..., n_actions]
            q_vals2 = self.critic2.forward(inputs)

            q_vals_taken1 = th.gather(q_vals1, dim=3, index=action).squeeze(3)
            q_vals_taken2 = th.gather(q_vals2, dim=3, index=action).squeeze(3)

            vs1 = th.logsumexp(q_vals1 / (alpha + self.eps), dim=-1) * (alpha + self.eps)
            vs2 = th.logsumexp(q_vals2 / (alpha + self.eps), dim=-1) * (alpha + self.eps)

        # Get Q joint for actions (using individual Qs and Vs)
        q_vals1 = self.mixer1(q_vals_taken1, states, actions=action_input, vs=vs1, mask = mask)
        q_vals2 = self.mixer2(q_vals_taken2, states, actions=action_input, vs=vs2, mask = mask)

        return (q_vals1, q_vals2)

    def _get_value(self, inputs):
        vs1 = self.value(inputs)
        if self.double_value:
            vs2 = self.value2(inputs)
        else:
            vs2 = vs1
        return vs1, vs2

    def _get_target_value(self, inputs):
        # return self._get_value(inputs)
        vs1 = self.target_value(inputs)
        if self.double_value:
            vs2 = self.target_value2(inputs)
        else:
            vs2 = vs1
        return vs1, vs2

    def _get_q_values(self, inputs, action):
        """
        Get flattened individual Q values
        """
        # Save compute by deactivating grad of critic params
        for p1,p2 in zip(self.critic1.parameters(), self.critic2.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

        # Get Q values
        if self.continuous_actions:
            action_input = action
            q_vals1 = self.critic1.forward(inputs, action_input).view(-1, self.n_agents)
            q_vals2 = self.critic2.forward(inputs, action_input).view(-1, self.n_agents)
        else:
            q_vals1 = self.critic1.forward(inputs).view(-1, self.n_agents, self.n_actions)
            q_vals2 = self.critic2.forward(inputs).view(-1, self.n_agents, self.n_actions)

        # reactivate grad of critic params
        for p1,p2 in zip(self.critic1.parameters(), self.critic2.parameters()):
            p1.requires_grad = True
            p2.requires_grad = True

        return q_vals1, q_vals2

    def train_actor(self, batch, t_env,):
        """
        Update actor and value nets as in SAC (Haarjona)
        https://github.com/haarnoja/sac/blob/master/sac/algos/sac.py  
        Add regularization term for implicit constraints 
        Mixer isn't used during policy improvement
        """
        bs = batch.batch_size
        max_t = batch.max_seq_length
        mask = batch["filled"][:, :-1].float()
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents)
        mask = mask.reshape(-1, self.n_agents)
        avail_actions = batch["avail_actions"]

        # [ep_batch.batch_size, max_t, self.n_agents, -1]
        mac_out = self._get_policy(batch, avail_actions=avail_actions, use_target_actor= False)

        # [batch.batch_size, max_t, self.n_agents]
        action_out, log_p_action = mac_out
        action_out = action_out[:, :-1]
        log_p_action = log_p_action[:, :-1]  # remove last step 

        ent_coef_loss = self._update_ent_coefficient(log_p_action, mask)

        # crude estimation for weigth for the entropy of each agent (moving average)
        alpha = self.get_current_alphas(t_env)

        # if self.continuous_actions:
        #     log_p_action = log_p_action.reshape(-1, self.n_agents)
        #     entropies = - (th.exp(log_p_action) * log_p_action)
        #     entropies = ((entropies * mask).sum()/mask.sum()).item()
        # else:
        #     log_p_action = log_p_action.reshape(-1, self.n_agents, self.n_actions)
        #     pi = action_out.reshape(-1, self.n_actions)
        #     entropies = - ( ((pi * log_p_action).sum(dim=-1) * mask).sum() / mask.sum()).item()

        # inputs are shared between v's and q's
        inputs = self.critic1._build_inputs(batch, bs, max_t)

        inputs = inputs[:, :-1]

        # Get Q values with no grad and flattened
        q1, q2 = self._get_q_values(inputs, action_out) # 2x([-1, n_agents])

        if self.continuous_actions:
            vs1, vs2 = self._get_value(inputs)
            vs1, vs2 = vs1.reshape(-1, self.n_agents), vs2.reshape(-1, self.n_agents)

            # q_joint_1 = self.mixer1(q1.detach(), batch["state"][:, -1], actions=action_out, vs=vs1.detach())
            # q_joint_2 = self.mixer2(q2.detach(), batch["state"][:, -1], actions=action_out, vs=vs2.detach())
            # q_joint_min = th.min(q_joint_1, q_joint_2).reshape(-1, 1).expand(-1, self.n_agents)/self.n_agents

            q_min = th.min(q1, q2)

            if (not self.double_value):
                q1 = q_min

            v_act_target = ((q1 - alpha * log_p_action).detach() - vs1)**2 
            v_act_loss = (v_act_target * mask).sum() / mask.sum()

            if self.double_value:
                v_act_target2 = ((q2 - alpha * log_p_action).detach() - vs2)**2
                v_act_loss2 = (v_act_target2 * mask).sum() / mask.sum()

            
            act_target = (alpha * log_p_action - q_min)

            # # # regularize by joint q 
            # act_target -= (0.5 * q_joint_min)

            # regularize by action quad sum 
            if (self.args.norm_reg and self.args.continuous_actions):
                act_target += 1e-3 * (((action_out**2).sum(dim = -1)).reshape(-1, self.n_agents)) * self.reward_scale
            
        else:
            act_target = (pi * (alpha * log_p_action - q_min.detach())).sum(dim=-1)
            v_act_loss = 0

        # act_loss
        act_loss = (act_target).sum() / mask.sum() 
        
        loss_policy = act_loss

        # Optimize policy
        self.p_optimizer.zero_grad()
        loss_policy.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.p_optimizer.step()

        # If a value net exists, then optimize it
        if self.continuous_actions:
            loss_value = v_act_loss
            loss_value = loss_value 

            self.val_optimizer.zero_grad()
            loss_value.backward()
            value_grad_norm = th.nn.utils.clip_grad_norm_(self.value_params, self.args.grad_norm_clip)
            self.val_optimizer.step()

            if self.double_value:
                loss_value2 = v_act_loss2

                self.val_optimizer2.zero_grad()
                loss_value2.backward()
                th.nn.utils.clip_grad_norm_(self.value_params2, self.args.grad_norm_clip)
                self.val_optimizer2.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("act_loss", act_loss.item(), t_env)
            self.logger.log_stat("act_target_mean", act_target.mean().item(), t_env)
            
            if self.continuous_actions:

                self.logger.log_stat("v_act_loss", v_act_loss.item(), t_env)
                self.logger.log_stat("value_grad_norm", value_grad_norm.item(), t_env)

                if self.double_value:
                    self.logger.log_stat("v_act_loss2", v_act_loss2.item(), t_env)

            self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
            # self.logger.log_stat("act_entropy", entropies, t_env)

            if (ent_coef_loss is not None):
                self.logger.log_stat("ent_coef_loss", ent_coef_loss.item(), t_env)

            if self.use_local_entropy:
                for i, alpha_i in enumerate(alpha):
                    self.logger.log_stat(f"alpha_{i}", alpha_i.detach().cpu().item(), t_env) 

            if self.debug_log:
                mean_log_p = ((log_p_action * mask).sum()/mask.sum()).detach().cpu().item()
                self.logger.log_stat("mean_log_p_act", mean_log_p , t_env)  
                self.logger.log_stat("max_log_p_act", (log_p_action * mask).max().detach().cpu().item(), t_env) 
                self.logger.log_stat("min_log_p_act", (log_p_action * mask).min().detach().cpu().item(), t_env)  
        

            self.log_stats_t = t_env
            

    def train_critic(self, batch, t_env):

        alpha = self.get_alpha(t_env)
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards_tmp = batch["reward"][:, :-1]
        states = batch["state"]
        actions_taken = batch["actions"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        avail_actions = batch["avail_actions"]
        rewards = rewards_tmp*self.reward_scale

        
        with th.no_grad():
            mac_out = self._get_policy(batch, avail_actions=avail_actions, use_target_actor = self.use_target_actor)
            # [batch.batch_size, max_t, self.n_agents]
            next_action_out, log_p_action = mac_out[0].detach(), mac_out[1].detach()

        if self.continuous_actions:
            buff_action_one_hot = None
            next_action = next_action_out
            log_p_action_taken = log_p_action
        else:
            # in dicrete case the output is the full distribution
            buff_action_one_hot = batch["actions_onehot"][:, :-1].float()  # buffer actions are pre-processed
            next_action = Categorical(next_action_out).sample().long().unsqueeze(3)
            log_p_action_taken = th.gather(log_p_action, dim=3, index=next_action).squeeze(3)

        inputs = self.critic1._build_inputs(batch, bs, max_t)

        # Find Q values of actions and  according to current policy 
        with th.no_grad():
            target_act_joint_q = self._get_joint_q_target(inputs[:, 1:], states[:, 1:], next_action[:, 1:], alpha)

        # q(s_t+1)  - alpha * log_p_t+1
        target_act_joint_q = target_act_joint_q.detach() - alpha * log_p_action_taken[:, 1:].sum(dim=-1)
        # r + gamma*(q(s_t+1)  - alpha * log_p_t+1)
        target_v_act = rewards + self.args.gamma * (1.0 - terminated) * target_act_joint_q

        targets_act = target_v_act 

        # Find Q values of actions taken in batch
        q_act_taken = self._get_joint_q(inputs[:, :-1], states[:, :-1], actions_taken[:, :-1],
                                                      buff_action_one_hot, alpha, mask = mask[:, :-1])
        
        q1_act_taken, q2_act_taken = q_act_taken  # double q
        td_error1_act =  targets_act.detach() - q1_act_taken 
        td_error2_act =  targets_act.detach() - q2_act_taken

        # mask out
        mask = mask.expand_as(td_error1_act)
        masked_td_error1 = td_error1_act * mask

        critic_loss1 = (masked_td_error1 ** 2).sum() / mask.sum()
        masked_td_error2 = td_error2_act * mask
        critic_loss2 = (masked_td_error2 ** 2).sum() / mask.sum()

        # Optimize
        self.c_optimizer1.zero_grad()
        critic_loss1.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params1, self.args.critic_grad_norm_clip)
        self.c_optimizer1.step()

        self.c_optimizer2.zero_grad()
        critic_loss2.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params2, self.args.critic_grad_norm_clip)
        self.c_optimizer2.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:

            self.logger.log_stat("alpha", alpha, t_env)
            self.logger.log_stat("critic_loss", critic_loss1.item(), t_env)
            self.logger.log_stat("critic_grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error1.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (q1_act_taken * mask).sum().item() /mask_elems, t_env)

            self.logger.log_stat("target_mean", (targets_act).sum().item() /mask_elems,
                                 t_env)

            self.logger.log_stat("action_abs mean", (actions_taken).abs().mean().detach().cpu().item(), t_env)
            

            if self.debug_log:
                self.logger.log_stat("action_out_max", actions_taken.max().detach().cpu().item(), t_env)   
                self.logger.log_stat("action_out_min", actions_taken.min().detach().cpu().item(), t_env)   

    def _update_targets(self):
        
        # if self.use_local_entropy:
        #     self.current_alphas1 = self.target_mixer1.current_alphas.data.clone()
        #     self.current_alphas2 = self.target_mixer2.current_alphas.data.clone()

        if getattr(self.args, "polyak_update", False):

            tau = getattr(self.args, "tau", 0.001)
            polyak_update(self.critic1.parameters(), self.target_critic1.parameters(), tau)
            polyak_update(self.critic2.parameters(), self.target_critic2.parameters(), tau)

            polyak_update(self.mixer1.parameters(), self.target_mixer1.parameters(), tau)
            polyak_update(self.mixer2.parameters(), self.target_mixer2.parameters(), tau)

            if self.continuous_actions:
                polyak_update(self.value.parameters(), self.target_value.parameters(), tau)
                if self.double_value:
                    polyak_update(self.value2.parameters(), self.target_value2.parameters(), tau)
            if self.use_target_actor:
                polyak_update(self.mac.parameters(), self.target_mac.parameters(), tau)
                
        else:
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())

            self.target_mixer1.load_state_dict(self.mixer1.state_dict())
            self.target_mixer2.load_state_dict(self.mixer2.state_dict())

            if self.continuous_actions:
                self.target_value.load_state_dict(self.value.state_dict())
                if self.double_value:
                    self.target_value2.load_state_dict(self.value2.state_dict())
            if self.use_target_actor:
                self.target_mac.load_state_dict(self.mac.state_dict())

        # if self.use_local_entropy:
        #     self.target_mixer1.current_alphas.data = self.current_alphas1
        #     self.target_mixer1.current_alphas.data = self.current_alphas2

        if self.args.target_update_interval > 10:
            self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        if self.use_target_actor:
            self.target_mac.cuda()

        self.critic1.cuda()
        self.critic2.cuda()

        self.mixer1.cuda()
        self.mixer2.cuda()

        self.target_critic1.cuda()
        self.target_critic2.cuda()

        self.target_mixer1.cuda()
        self.target_mixer2.cuda()

        if self.continuous_actions:
            self.value.cuda()
            self.target_value.cuda()
            if self.double_value:
                self.value2.cuda()
                self.target_value2.cuda()


    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic1.state_dict(), "{}/critic1.th".format(path))
        th.save(self.mixer1.state_dict(), "{}/mixer1.th".format(path))
        th.save(self.critic2.state_dict(), "{}/critic2.th".format(path))
        th.save(self.mixer2.state_dict(), "{}/mixer2.th".format(path))
        th.save(self.p_optimizer.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.c_optimizer1.state_dict(), "{}/critic_opt1.th".format(path))
        th.save(self.c_optimizer2.state_dict(), "{}/critic_opt2.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        if self.use_target_actor:
            self.target_mac.load_models(path)
        self.critic1.load_state_dict(th.load("{}/critic1.th".format(path), map_location=lambda storage, loc: storage))
        self.critic2.load_state_dict(th.load("{}/critic2.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right, but I don't want to save target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.mixer1.load_state_dict(th.load("{}/mixer1.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer2.load_state_dict(th.load("{}/mixer2.th".format(path), map_location=lambda storage, loc: storage))

        self.p_optimizer.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.c_optimizer1.load_state_dict(
            th.load("{}/critic_opt1.th".format(path), map_location=lambda storage, loc: storage))
        self.c_optimizer2.load_state_dict(
            th.load("{}/critic_opt2.th".format(path), map_location=lambda storage, loc: storage))
        
        
    def _build_ent_coefficient(self, args):
        self.reward_scale = getattr(args, "reward_scale", 1.0)
        self.target_entropy = getattr(args, "target_entropy", None)

        if self.target_entropy is not None:
            init_value = getattr(args, "alpha_start", 1.0)
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            self.log_alpha = (th.log(th.ones(1, device=self.device) * init_value)).requires_grad_(True)
            if getattr(self.args, "optimizer", "rmsprop") in ["rmsprop", "rms"]:
                self.alpha_optimizer = RMSprop(params=[self.log_alpha], lr=args.lr * 1e-2, alpha=args.optim_alpha,
                                        eps=args.optim_eps)

            elif getattr(self.args, "optimizer", "rmsprop")  in ["adam", "adamw"]:
                self.alpha_optimizer = Adam(params=[self.log_alpha], lr=args.lr * 1e-2,
                                        eps=getattr(args, "optim_eps", 1e-7))
            # get alpha without backprop
            self.get_alpha = lambda t_env: th.exp(self.log_alpha).detach().item()
            # Learn a global alpha
            if self.target_entropy == 'auto':
                avail_actions = ((args.actions_max.to(args.device) - args.actions_min.to(args.device))/2.) > 0
                self.target_entropy = -avail_actions.sum().float()
                print("target_entropy", self.target_entropy)
            else:
                self.target_entropy = float(self.target_entropy)

        else:
            alpha_anneal_time = getattr(args, "alpha_anneal_time", 200000)
            alpha_start = getattr(args, "alpha_start", 0.5)
            alpha_finish = getattr(args, "alpha_finish", 0.05)
            alpha_decay = getattr(args, "alpha_decay", "linear")
            role_action_spaces_update_start = getattr(args, "role_action_spaces_update_start", 0)
            self.alpha_schedule = DecayThenFlatSchedule(alpha_start, alpha_finish, alpha_anneal_time,
                                                        time_length_exp=alpha_anneal_time,
                                                        role_action_spaces_update_start=role_action_spaces_update_start
                                                        , decay=alpha_decay)
            
            self.get_alpha = self.alpha_schedule.eval
                
    def _update_ent_coefficient(self, log_p, mask):
        if (self.target_entropy is not None):
            loss = - th.exp(self.log_alpha) * (log_p.sum(axis = -1).reshape(-1) + self.target_entropy).detach()
            mask_c = mask.sum(axis = -1).reshape(-1)/self.n_agents
            loss = (loss* mask_c).sum()/mask_c.sum()
            self.alpha_optimizer.zero_grad()
            loss.backward()
            self.alpha_optimizer.step() 
            return loss
        else:
            pass

    def get_current_alphas(self, t_env):
        if self.use_local_entropy:
            current_alphas = (self.target_mixer1.get_current_alphas() + self.target_mixer2.get_current_alphas())/2.0
            alpha = current_alphas.detach()
        else:
            alpha = self.get_alpha(t_env)

        return alpha     