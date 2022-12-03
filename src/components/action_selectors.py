import torch as th
from torch.distributions.kl import kl_divergence
from torch.distributions import Categorical
from torch.distributions.normal import Normal 
from torch.distributions.independent import Independent
from torch.distributions import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from .epsilon_schedules import DecayThenFlatSchedule
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
LOG_STD_MAX = 1.0
LOG_STD_MIN = -10.0

REGISTRY = {}

class GumbelSoftmax():
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.eps = 1e-10

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            U = th.rand(masked_policies.size()).cuda()
            y = masked_policies - th.log(-th.log(U + self.eps) + self.eps)
            y = F.softmax(y / 1, dim=-1)
            y[avail_actions == 0.0] = 0.0
            picked_actions = y.max(dim=2)[1]

        return picked_actions


REGISTRY["gumbel"] = GumbelSoftmax


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start,
                                              args.epsilon_finish, args.epsilon_anneal_time,
                                              time_length_exp=args.epsilon_anneal_time_exp,
                                              role_action_spaces_update_start=args.role_action_spaces_update_start,
                                              decay="linear")

        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0] = 0.0

        if t_env is not None:
            self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1] 
        else:
            # picked_actions = masked_policies.max(dim=2)[1] 
            
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              args.epsilon_anneal_time_exp,
                                              args.role_action_spaces_update_start,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]

        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class GaussianActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)
        self.decoder = None
        self.prior = None
        if not isinstance(self.prior, Normal) and (self.prior is not None):
            self.dkl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        else:
            self.dkl = kl_divergence
        self.unit2actions = args.actions2unit_coef
        self.squash = self.args.squash
        self.eps = 1e-10
        
        self.action_scale = (self.args.actions_max.to(args.device) - self.args.actions_min.to(args.device))/2.
        self.action_scale = self.action_scale/(self.action_scale+self.eps)

        self.action_bias =  (self.args.actions_max.to(args.device) + self.args.actions_min.to(args.device))/2.
        self.action_bias = self.action_bias * 0
        self.unvailable_mask =  (self.action_scale/(self.action_scale + self.eps) ) == 0
        self.max = 0

    def select_action(self, mu, sigma, t_env, prior, test_mode=False):
        
        dkl_loss = None
        if test_mode and self.test_greedy:
            picked_actions = mu 
            log_p_pi = None
            if self.squash :
                picked_actions = th.tanh(picked_actions) * self.action_scale + self.action_bias
        else:
            normal = th.distributions.Normal(mu, sigma)
            picked_actions = normal.rsample().view(mu.shape)

            log_p_pi = normal.log_prob(picked_actions)
            if self.squash :
                log_p_pi -= 2.0* (np.log(2.0) - picked_actions - F.softplus(-2.0*picked_actions)) # alternative
                picked_actions = th.tanh(picked_actions) * self.action_scale + self.action_bias 
                log_p_pi[self.unvailable_mask.expand(log_p_pi.size(0), -1, -1)] = 0.0 # log_p of padded actions
            log_p_pi = log_p_pi.sum(dim = -1) # sum over action dim
        # M = sigma
        # M= M.max()
        # if t_env is not None:
        #     if (t_env % 100) == 0:
        #         if (M >= self.max):
        #             self.max = M
        #             print("sigma", sigma)
        #             print("mu", mu)
        #             # print(M, th.argwhere(sigma == M))
        #             # print("sigma.mean()", sigma.mean())

        return picked_actions, log_p_pi, dkl_loss

REGISTRY["gaussian"] = GaussianActionSelector


class GaussianLatentActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)
        self.decoder = None
        self.use_latent_normal = getattr(args, "use_latent_normal", True)
        if not self.use_latent_normal:
            self.dkl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        else:
            self.dkl = kl_divergence
        self.with_logprob = getattr(args, "with_logprob", True)
        self.threshold = nn.parameter.Parameter(th.tensor(0.3181), requires_grad=False)  # 2 times the variance same mu
        self.unit2actions = args.actions2unit_coef
        self.actions_min = args.actions_min

    def update_decoder(self, decoder):
        self.decoder = decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

    def select_action(self, mu, sigma, t_env, prior, test_mode=False):
        dkl_loss = None
        latent_dist = Normal(mu, sigma)
        latent_action = mu if test_mode else latent_dist.sample()
        log_p_latent = latent_dist.log_prob(latent_action).sum(dim=-1)
        if not test_mode and prior:
            if self.use_latent_normal:  # dkl distributions
                # [bs, action_latent] [n_actions, action_latent]
                dkl_loss = self.dkl(latent_dist, prior)
            else:
                sample = prior.sample()
                log_p_prior = prior.log_prob(sample).sum(dim=-1)
                dkl_loss = self.dkl(log_p_latent, log_p_prior)
            dkl_loss = th.max(dkl_loss, self.threshold)  # don't enforce the dkl inside the threshold

        if self.with_logprob:
            pi_action, log_p_pi = self.decoder(latent_action)
            log_p_pi += log_p_latent  # p_latent * p_action

            log_p_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            pi_action = self.decoder(latent_action)
            log_p_pi = None

        pi_action = th.tanh(pi_action)
        pi_action = self.unit2actions * pi_action + self.actions_min

        return pi_action, log_p_pi, dkl_loss


REGISTRY["gaussian_latent"] = GaussianLatentActionSelector

