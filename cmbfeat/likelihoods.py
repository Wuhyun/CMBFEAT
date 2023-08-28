import numpy as np
import cmbbest as best
from cobaya.theory import Theory
from cobaya.likelihood import Likelihood

class BispectrumLikelihood(Likelihood):

    def initialize(self, mode_p_max=30):
        # Initialise arrays and parameters necessary for likelihood computation
        pass

    def get_requirements(self):
        return ["fnl", "fnl_MLE", "fnl_sigma"]

    def logp(self, _derived=None, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        fnl = self.provider.get_param("fnl")
        fnl_MLE = self.provider.get_param("fnl_MLE")
        fnl_sigma = self.provider.get_param("fnl_sigma")

        dlnL = (fnl * fnl_MLE - (1/2) * fnl ** 2) / (fnl_sigma ** 2)

        return dlnL

    def get_info(unit_fnl=False):
        # Get base info for cobaya runs

        info = {}
        info["likelihood"] = {"bestlike": BispectrumLikelihood}

        if unit_fnl:
            # fnl is fixed to 1
            info["params"] = {
                        "fnl": {
                            "value": 1,
                            "latex": r"f_\mathrm{NL}"
                        }
                    }
        else:
            # Sample fnl directly
            info["params"] = {
                        "fnl": {
                            "prior": {
                                "min": -100,
                                "max": 100
                            },
                            "latex": r"f_\mathrm{NL}"
                        }
            }
        
        return info



class BispectrumDecomp(Theory):
    """ A class for basis decomposition of a given cmbbest.Model instance """

    def initialize(self, mode_p_max=30):
        """called from __init__ to initialize"""
        self.basis = best.Basis(mode_p_max=mode_p_max)
        self.basis.precompute_pseudoinv()

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return ["cmbbest_model"]

    def get_can_provide_params(self):
        return ["fnl_MLE", "fnl_sigma", "fnl_sample_sigma", "fnl_LISW_bias", "decomp_conv_corr", "decomp_conv_MSE"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        best_model = self.provider.get_result("cmbbest_model")
        constraint = self.basis.constrain_models([best_model], silent=True, use_pseudoinverse=True)

        if want_derived:
            state["derived"] = {"fnl_MLE": constraint.single_f_NL[0,0],
                                "fnl_sigma": constraint.single_fisher_sigma[0],
                                "fnl_sample_sigma": constraint.single_sample_sigma[0],
                                "fnl_LISW_bias": constraint.single_LISW_bias[0],
                                "decomp_conv_corr": constraint.convergence_correlation,
                                "decomp_conv_MSE": constraint.convergence_MSE}

    def get_info():
        # Get base info for cobaya runs

        info = {}
        info["theory"] = {"decomp": BispectrumDecomp}
        info["params"] = {
                    "fnl_MLE": {
                        "latex": r"\widehat{f_\mathrm{NL}}"
                    },
                    "fnl_sigma": {
                        "min": 0,
                        "latex": r"\sigma(f_\mathrm{NL})"
                    },
                    "fnl_sample_sigma": {
                        "min": 0,
                        "latex": r"\widehat{\sigma(f_\mathrm{NL})}"
                    },
                    "fnl_LISW_bias": {
                        "latex": r"\widehat{f_\mathrm{NL}^\mathrm{LISW}}"
                    },
                    "decomp_conv_corr": {
                        "min": -1,
                        "max": 1,
                        "latex": r"R_\mathrm{conv}"
                    },
                    "decomp_conv_MSE": {
                        "latex": r"\epsilon_\mathrm{conv}"
                    }
                }
        
        return info



class BispectrumFromPkGSR(Theory):
    """ A class for computing the bispectrum expected from given primordial power spectrum,
        under Generalized Slow Roll (GSR). Expression given in 1512.08977 """

    def initialize(self):
        """called from __init__ to initialize"""
        pass

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return ["primordial_scalar_pk"]

    def get_can_provide(self):
        return ["cmbbest_model"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Compute the bispectrum corresponding to P(k) under GSR

        prim_pk = self.provider.get_result("primordial_scalar_pk")
        assert(prim_pk["log_regular"])  # Currently only implemented for log-spaced grids

        Pk = prim_pk["Pk"]
        logPk = np.log(Pk)
        kvec = np.logspace(prim_pk["kmin"], prim_pk["kmax"], len(Pk))
        logk = np.log(kvec)
        dlogk = logk[1] - logk[0]   # Equally spaced 

        ns = 1 + np.gradient(logPk) / dlogk 
        alphas = np.gradient(ns) / dlogk

        integrand = np.exp(-logk) * (ns - 1)

        term1 = (2 * np.pi) ** 4 * (Pk ** 2) * (np.sum(integrand) - np.cumsum(integrand)) * dlogk
        term2 = (2 * np.pi) ** 4 * (Pk ** 2) * (1 - ns)
        term3 = (2 * np.pi) ** 4 * (Pk ** 2) * alphas

        def shape_function(k1, k2, k3):
            meanK = (k1 + k2 + k3) / 2
            logmeanK = np.log(meanK)
            deltasq = meanK * (meanK - 2*k1) * (meanK - 2*k2) * (meanK - 2*k3)

            t1 = np.interp(logmeanK, logk, term1) * deltasq
            t2 = np.interp(logmeanK, logk, term2) * (((k1**2 + k2**2 + k3**2) * (k1*k2 + k2*k3 + k3*k1) / (16 * meanK))
                                                    + ((k1*k2)**2 + (k2*k3)**2 + (k3*k1)**2) / (8 * meanK)
                                                    - (k1*k2*k3) / 8)
            t3 = np.interp(logmeanK, logk, term3) * (k1*k2*k3) / 8

            return (k1*k2*k3)**(-3) * (t1 + t2 + t3)
        
        cmbbest_model = best.Model("custom", shape_function=shape_function, shape_name="Bispectrum from Pk")
        
        state["cmbbest_model"] = cmbbest_model

        def get_cmbbest_model(self):
            return self.current_state["cmbbest_model"]


    def get_info():
        # Get base info for cobaya runs

        info = {}
        info["theory"] = {"bispectrumGSR": BispectrumFromPkGSR}
        
        return info


