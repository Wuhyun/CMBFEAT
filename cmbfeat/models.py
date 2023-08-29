import numpy as np
from cobaya.theory import Theory
import cmbbest as best

BASE_DELTA_PHI = (2 * (np.pi ** 2) * ((3 / 5) ** 2)
                    * (best.BASE_K_PIVOT ** (1 - best.BASE_N_SCALAR))
                    * best.BASE_A_S)
BASE_NORMALISATION = 6 * BASE_DELTA_PHI ** 2

PK_GRID_SIZE = 10000

def recursive_merge(a, b, path=[]):
    # Recursively merge two dictionaries. Modifies 'a'
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                recursive_merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                raise Exception('Dictionary merge conflict at path' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def merge_dicts(info_list):
    # Utility function for joining multiple Cobaya InfoDicts
    merged = {}
    for info in info_list:
        merged = recursive_merge(merged, info)

    return merged


class PowerLawPrimordialPk(Theory):
    # LCDM PPS following a power law

    def initialize(self):
        # need to provide valid results at wide k range, any that might be used
        self.ks = np.logspace(-5.5, 2, PK_GRID_SIZE)

    def calculate(self, state, want_derived=True, **params_values_dict):
        pivot_scalar = 0.05
        As, ns = [params_values_dict[key] for key in ["As", "ns"]]

        pk = (self.ks / pivot_scalar) ** (ns - 1) * As
        state['primordial_scalar_pk'] = {'kmin': self.ks[0], 'kmax': self.ks[-1], 'Pk': pk, 'log_regular': True}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

    def get_can_support_params(self):
        return ["As", "ns"]


class LinEnvOscPrimordialPk(Theory):
    # PPS with linear oscillations with an envelope

    def initialize(self):
        # need to provide valid results at wide k range, any that might be used
        self.ks = np.logspace(-5.5, 2, PK_GRID_SIZE)

    def calculate(self, state, want_derived=True, **params_values_dict):
        pivot_scalar = 0.05
        As, ns, A_osc, sigma_osc, omega_osc, kp_osc = [params_values_dict[key]
                        for key in ["As", "ns", "A_osc", "sigma_osc", "omega_osc", "kp_osc"]]

        B_osc = 1 / (sigma_osc * kp_osc) ** 2

        base_pk = (self.ks / pivot_scalar) ** (ns - 1) * As
        osc_pk = A_osc * np.cos(omega_osc * (self.ks - kp_osc)) * np.exp(-(B_osc * ((self.ks - kp_osc) ** 2) / 2))
        pk = base_pk * (1 + osc_pk)

        state['primordial_scalar_pk'] = {'kmin': self.ks[0], 'kmax': self.ks[-1], 'Pk': pk, 'log_regular': True}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

    def get_can_support_params(self):
        return ["As", "ns", "A_osc", "sigma_osc", "omega_osc", "kp_osc"]


class LinOscPrimordialPk(Theory):

    def initialize(self):
        # need to provide valid results at wide k range, any that might be used
        self.ks = np.logspace(-5.5, 2, PK_GRID_SIZE)

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


    def get_info(omega_min=10, omega_max=300):
        # Get base info for cobaya runs
        info = {}
        info["theory"] = {"cmbfeat.models.LinOscPrimordialPk": None}
        info["params"] = {
                "A_osc": {
                    "prior": {
                        "min": -0.3,
                        "max": 0.3
                    },
                    "latex": r"A_\mathrm{osc}"
                },
                "omega_osc": {
                    "prior": {
                        "min": omega_min,
                        "max": omega_max
                    },
                    "latex": r"\omega_\mathrm{osc}"
                },
                "phi_osc": {
                    "prior": {
                        "min": 0,
                        "max": np.pi
                    },
                    "latex": r"\phi_\mathrm{osc}"
                }
        }

        return info


class LinOscPrimordialB(Theory):

    def initialize(self):
        pass

    def initialize_with_provider(self, provider):
        self.provider = provider

    def get_requirements(self):
        return ["omega_osc", "phi_osc"]
    
    def get_can_provide(self):
        return ["cmbbest_model"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        omega_osc = self.provider.get_param("omega_osc")
        phi_osc = self.provider.get_param("phi_osc")

        model = best.Model("custom", shape_function=lambda k1, k2, k3: BASE_NORMALISATION * np.sin(omega_osc*(k1+k2+k3) + phi_osc))
        state["cmbbest_model"] = model

    def get_cmbbest_model(self):
        return self.current_state["cmbbest_model"]

    def get_info(omega_min=10, omega_max=300):
        # Get base info for cobaya runs
        info = {}
        info["theory"] = {"cmbfeat.models.LinOscPrimordialB": None}
        info["params"] = {
                "omega_osc": {
                    "prior": {
                        "min": omega_min,
                        "max": omega_max
                    },
                    "latex": r"\omega_\mathrm{osc}"
                },
                "phi_osc": {
                    "prior": {
                        "min": 0,
                        "max": np.pi
                    },
                    "latex": r"\phi_\mathrm{osc}"
                }
        }

        return info