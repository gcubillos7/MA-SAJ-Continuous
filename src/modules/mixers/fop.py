import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import RMSprop, Adam, AdamW

class FOPMixer(nn.Module):
    def __init__(self, args, n_actions = None):
        super(FOPMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        if n_actions is not None:
            self.n_actions = n_actions   
        else:
            self.n_actions = args.n_actions

        self.action_dim = args.n_agents * self.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.state_action_dim = self.state_dim + self.action_dim
        self.n_head = args.n_head  
        self.embed_dim = args.mixing_embed_dim

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()
        
        for _ in range(self.n_head):  # multi-head attention
            if hasattr(self.args, "hypernet_embed"):
                print("key extractor has embed dim")
                # n_head * [( (self.state_dim + 1) * hypernet_embed + (hypernet_embed)) +
                #  ((self.state_dim + 1) * hypernet_embed + (hypernet_embed + 1) * self.n_agents)
                # + (self.state_action_dim + 1) * hypernet_embed + (hypernet_embed + 1) * self.n_agents] 

                hypernet_embed = self.args.hypernet_embed
                # (self.state_dim + 1) * hypernet_embed + (hypernet_embed+1)
                self.key_extractors.append(nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, 1, bias = False))) 
                # (self.state_dim + 1) * hypernet_embed + (hypernet_embed + 1) * self.n_agents
                self.agents_extractors.append(nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents, bias = False)))
                # (self.state_action_dim + 1) * hypernet_embed + (hypernet_embed + 1) * self.n_agents
                self.action_extractors.append(nn.Sequential(nn.Linear(self.state_action_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents, bias = False)))            
            else:    
                # n_head * [ (self.state_dim + 1) + (self.state_dim + 1) * self.n_agents+ ((self.state_dim + self.action_dim) +1 )*self.n_agents ]
                self.key_extractors.append(nn.Linear(self.state_dim, 1, bias = False)) 
                self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents, bias = False))  
                self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents, bias = False)) 

        # [(self.state_dim + 1) * self.embed_dim + (self.embed_dim +1)]
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        self.use_layer_norm = getattr(args, "use_layer_norm ", False)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.state_dim) 

        self.train_alpha = False # by default
        self.update_type = getattr(args, "local_alpha_update", "auto")
        
        init_alphas = self.args.alpha_start * th.ones(self.n_agents, device = args.device)
        self.alpha = th.tensor(self.args.alpha_start) 

        if self.update_type in ["soft", "moving_average"]:
            self.use_soft_update = True
            self.current_alphas = nn.parameter.Parameter(data = init_alphas, requires_grad = False)
            self.tau = 1e-4
        elif self.update_type in ["auto"]:
            self.use_soft_update = False
            init_alphas = th.log(init_alphas) # log_alpha
            self.current_alphas = nn.parameter.Parameter(data = init_alphas, requires_grad = True)
            self._setup_optimizer(args)
        else:
            raise Exception(f"Local alpha update type '{self.update_type}' is not supported.")

        self.eps = 1e-10

    def forward(self, agent_qs, states, actions=None, vs=None, mask = None):
        bs = agent_qs.size(0)

        if self.use_layer_norm:
            states = self.layer_norm(states)          

        v = self.V(states).reshape(-1, 1).repeat(1, self.n_agents) / self.n_agents

        agent_qs = agent_qs.reshape(-1, self.n_agents)
        vs = vs.reshape(-1, self.n_agents).detach()
         
        adv_q = (agent_qs - vs).detach()
        lambda_weight = self.lambda_weight(states, actions)
        
        adv_tot = th.sum(adv_q * lambda_weight, dim=1).reshape(bs, -1, 1) # = (lambda - 1) * A 
        
        v_tot = th.sum(agent_qs + v , dim=-1).reshape(bs, -1, 1) 

        if self.train_alpha:
            if self.use_soft_update:
                self._soft_update(lambda_weight + 1, adv_q, mask = None) 
            else:
                self._auto_update(lambda_weight + 1, mask = None)
        else:
            if self.use_soft_update:
                self.current_alphas.data.fill_(self.alpha)
            else:
                self.current_alphas.data.fill_(np.log(self.alpha))
        return adv_tot + v_tot 

    def lambda_weight(self, states, actions):
        # estimation of alpha_i/alpha
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        state_actions = th.cat([states, actions], dim=1)

        head_keys = [k_ext(states) for k_ext in self.key_extractors]
        head_agents = [k_ext(states) for k_ext in self.agents_extractors]
        head_actions = [sel_ext(state_actions) for sel_ext in self.action_extractors]

        lambda_weights = []
        
        for head_key, head_agents, head_action in zip(head_keys, head_agents, head_actions):
            key = th.abs(head_key).repeat(1, self.n_agents) + self.eps
            agents = th.sigmoid(head_agents)
            action = th.sigmoid(head_action)
            weights = key * agents * action
            lambda_weights.append(weights)
            
        lambdas = th.stack(lambda_weights, dim=1)
        lambdas = lambdas.reshape(-1, self.n_head, self.n_agents).sum(dim=1)
        
        return lambdas.reshape(-1, self.n_agents)

    def _soft_update(self, lambda_weight, adv_q , mask = None):
        with th.no_grad():
            if mask is not None:
                mask = mask.unsqueeze(-1).expand(-1, -1, self.n_agents).reshape(-1, self.n_agents)
                lambda_adv  = (((lambda_weight.detach()) * adv_q)*mask).sum(dim = 0)/mask.sum(dim = 0)
                adv_mean = (adv_q*mask).sum(dim = 0)/mask.sum(dim = 0)
            else:
                lambda_adv  =(lambda_weight.detach() * adv_q).mean(dim = 0)
                adv_mean = (adv_q).mean(dim = 0)
            
            weight_estimation = (adv_mean/lambda_adv)
            self.current_alphas += self.tau * (weight_estimation * self.alpha - self.current_alphas)

    def _auto_update(self, lambda_weight, mask = None):
        self.alpha_i_optimizer.zero_grad()
        if mask is not None:
            mask = mask.unsqueeze(-1).expand(-1, -1, self.n_agents).reshape(-1, self.n_agents)
            target  = ((self.alpha/(lambda_weight + 1e-8)).detach()*mask).sum(dim=0)/mask.sum(dim=0) # mean(alpha/lambda_i) # dim 0 <=> bs*T
        else:
            target  = (self.alpha/(lambda_weight + 1e-8)).detach().mean(dim=0)
    
        loss = ((th.exp(self.current_alphas) -  target)**2).sum()         
        loss.backward()
        self.alpha_i_optimizer.step()

    def _setup_optimizer(self, args):
        if getattr(args, "optimizer", "rmsprop") in ["rmsprop", "rms"]:
            # lr is lower
            self.alpha_i_optimizer = RMSprop(params=[self.current_alphas], lr=args.lr, alpha=args.optim_alpha,
                                       eps=args.optim_eps)

        elif getattr(self.args, "optimizer", "rmsprop")  in ["adam", "adamw"]:
            self.alpha_i_optimizer = Adam(params=[self.current_alphas], lr=args.lr,
                                    eps=getattr(args, "optim_eps", 1e-7)) 

    def update_alpha(self, alpha):
        # sync alphas of each agent with global
        self.alpha = alpha

    def toggle_alpha_train(self, train = True):
        self.train_alpha = train

    def get_current_alphas(self):

        if (self.use_soft_update):
                return F.relu(self.current_alphas) + 1e-8 # avoid 0 values
        else:
            return th.exp(self.current_alphas) + 1e-8 # avoid 0 values



        


