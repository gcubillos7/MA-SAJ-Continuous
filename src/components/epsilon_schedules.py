import numpy as np

class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 time_length_exp,
                 role_action_spaces_update_start,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay
        self.role_action_spaces_update_start = role_action_spaces_update_start
        self.reset = True
        self.time_length_exp = time_length_exp
        self.start_t = 0

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish/self.start) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * (T-self.start_t))
        elif self.decay in ["exp"]:
            return max(self.finish, self.start * np.exp(- T / self.exp_scaling))
