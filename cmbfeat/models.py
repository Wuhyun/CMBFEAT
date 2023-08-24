import numpy as np
from cobaya.theory import Theory
from cobaya.run import run
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file
import sys

class LinEnvOscPrimordialPk(Theory):
    # PPS with linear oscillations with an envelope

    def initialize(self):
        # need to provide valid results at wide k range, any that might be used
        self.ks = np.logspace(-5.5, 2, 5000)

    def calculate(self, state, want_derived=True, **params_values_dict):
        pivot_scalar = 0.05
        As, ns, A_osc, B_osc, omega_osc, kp_osc = [params_values_dict[key]
                        for key in ["As", "ns", "A_osc", "B_osc", "omega_osc", "kp_osc"]]

        base_pk = (self.ks / pivot_scalar) ** (ns - 1) * As
        osc_pk = A_osc * np.cos(omega_osc * (self.ks - kp_osc)) * np.exp(-(B_osc * ((self.ks - kp_osc) ** 2) / 2))
        pk = base_pk * (1 + osc_pk)

        state['primordial_scalar_pk'] = {'kmin': self.ks[0], 'kmax': self.ks[-1], 'Pk': pk, 'log_regular': True}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

    def get_can_support_params(self):
        return ["As", "ns", "A_osc", "B_osc", "omega_osc", "kp_osc"]


class LinOscPrimordialPk(Theory):

    def initialize(self):
        # need to provide valid results at wide k range, any that might be used
        self.ks = np.logspace(-5.5, 2, 2000)

    def calculate(self, state, want_derived=True, **params_values_dict):
        pivot_scalar = 0.05
        As, ns, A_osc, omega_osc, phi_osc = [params_values_dict[key]
                        for key in ["As", "ns", "A_osc", "omega_osc", "phi_osc"]]

        base_pk = (self.ks / pivot_scalar) ** (ns - 1) * As
        osc_pk = A_osc * np.sin(omega_osc * self.ks + phi_osc)
        pk = base_pk * (1 + osc_pk)

        state['primordial_scalar_pk'] = {'kmin': self.ks[0], 'kmax': self.ks[-1], 'Pk': pk, 'log_regular': True}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

    def get_can_support_params(self):
        return ["As", "ns", "A_osc", "omega_osc", "phi_osc"]
