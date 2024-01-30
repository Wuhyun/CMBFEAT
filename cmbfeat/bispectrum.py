import numpy as np
import cmbbest as best
from cobaya.theory import Theory
from cobaya.likelihood import Likelihood

MODE_P_MAX = 30

class BispectrumLikelihood(Likelihood):

    def initialize(self):
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

    def get_info(fnl_type="unit"):
        # Get base info for cobaya runs

        info = {}
        info["likelihood"] = {"cmbfeat.bispectrum.BispectrumLikelihood": None}

        if fnl_type == "unit":
            # fnl is fixed to 1
            info["params"] = {
                        "fnl": {
                            "value": 1,
                            "latex": r"f_\mathrm{NL}"
                        }
                    }
        elif fnl_type == "sampled":
            # Directly sample fnl
            info["params"] = {
                        "fnl": {
                            "prior": {
                                "min": -100,
                                "max": 100
                            },
                            "latex": r"f_\mathrm{NL}"
                        }
            }
        elif fnl_type == "derived":
            # Indirectly sample fnl, e.g. through fnl_SNR
            info["params"] = {
                        "fnl": {
                            "latex": r"f_\mathrm{NL}"
                        }
            }
        
        return info


class BispectrumDecomp(Theory):
    """ A class for getting bispectrum constraints on a given cmbbest.Model instance """

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
        info["theory"] = {"cmbfeat.bispectrum.BispectrumDecomp": None}
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


class BispectrumLikelihoodFromAlpha(Likelihood):

    def initialize(self, mode_p_max=MODE_P_MAX):
        basis = best.Basis(mode_p_max=mode_p_max)
        f_sky = basis.parameter_f_sky
        self._beta = (1/6) * (basis.beta[0,:] - f_sky * basis.beta_LISW)
        self._gamma = (f_sky/6) * basis.gamma

    def get_requirements(self):
        return ["fnl", "cmbbest_alpha"]

    def logp(self, _derived=None, **params_values):
        fnl = self.provider.get_param("fnl")
        alpha = self.provider.get_result("cmbbest_alpha")

        dlnL = fnl * np.dot(self._beta, alpha) - (1/2) * (fnl ** 2) * np.dot(alpha, np.matmul(self._gamma, alpha))

        return dlnL

    def get_info(fnl_type="unit"):
        # Get base info for cobaya runs

        info = {}
        info["likelihood"] = {"cmbfeat.bispectrum.BispectrumLikelihoodFromAlpha": None}

        if fnl_type == "unit":
            # fnl is fixed to 1
            info["params"] = {
                        "fnl": {
                            "value": 1,
                            "latex": r"f_\mathrm{NL}"
                        }
                    }
        elif fnl_type == "sampled":
            # Directly sample fnl
            info["params"] = {
                        "fnl": {
                            "prior": {
                                "min": -100,
                                "max": 100
                            },
                            "latex": r"f_\mathrm{NL}"
                        }
            }
        elif fnl_type == "derived":
            # Indirectly sample fnl, e.g. through fnl_SNR
            info["params"] = {
                        "fnl": {
                            "latex": r"f_\mathrm{NL}"
                        }
            }
        
        return info


class BispectrumDecompAlpha(Theory):
    """ A class for basis decomposition alpha of a given cmbbest.Model instance """

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

    def get_can_provide(self):
        return ["cmbbest_alpha"]

    def get_can_provide_params(self):
        return ["decomp_conv_corr", "decomp_conv_MSE"]

    def calculate(self, state, want_derived=True, **params_values_dict):
        best_model = self.provider.get_result("cmbbest_model")

        alpha, shape_cov, conv_corr, conv_MSE = self.basis.pseudoinv_basis_expansion([best_model], silent=True)
        state["cmbbest_alpha"] = alpha.flatten()

        if want_derived:
            state["derived"] = {"decomp_conv_corr": conv_corr,
                                "decomp_conv_MSE": conv_MSE} 
    
    def get_cmbbest_alpha(self):
        return self.current_state["cmbbest_alpha"]

    def get_info():
        # Get base info for cobaya runs

        info = {}
        info["theory"] = {"cmbfeat.bispectrum.BispectrumDecompAlpha": None}
        info["params"] = {
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


class fnlSNR2fnl(Theory):
    """ A simple helper class for sampling fnl_SNR instead of fnl """

    def get_requirements(self):
        return ["fnl_SNR", "fnl_sigma"]
    
    def get_can_provide_params(self):
        return ["fnl"]
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        fnl_SNR = self.provider.get_param("fnl_SNR")
        fnl_sigma = self.provider.get_param("fnl_sigma")
        fnl = fnl_SNR * fnl_sigma
        if want_derived:
            state["derived"] = {"fnl": fnl}
    
    def get_info(sigma_bound=4):
        # Get base info for cobaya runs

        info = {}
        info["theory"] = {"cmbfeat.bispectrum.fnlSNR2fnl": None}

        # Directly sample fnl
        info["params"] = {
                    "fnl_SNR": {
                        "prior": {
                            "min": -sigma_bound,
                            "max": sigma_bound
                        },
                        "latex": r"f_\mathrm{NL}/\sigma(f_\mathrm{NL})"
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

    
    def shape_function_from_Pk(k_grid, primordial_pk, k_pivot=0.05, As_base=None, ns_base=None):
        # Static function for computing the bispectrum shape function
        # that corresponds to a given P(k) under GSR
        # ASSUMES that k_grid is uniformly spaced in log space

        Pk = primordial_pk
        logPk = np.log(Pk)
        logk = np.log(k_grid)
        dlogk = logk[1] - logk[0]   # Equally spaced 

        ns = 1 + np.gradient(logPk) / dlogk 
        alphas = np.gradient(ns) / dlogk

        # Parameters for the featureless PPS
        if As_base and ns_base:
            As_0, ns_0 = As_base, ns_base

        else:
            # Simple linear regression to fit As and ns for the featureless PPS
            # logP_featless = logAs_0 + (ns_0-1) log(k/k_pivot)
            ns_0 = 1 + np.dot((logPk - np.mean(logPk)), (logk - np.mean(logk))) / np.sum((logk - np.mean(logk)) ** 2)
            As_0 = np.exp(np.mean(logPk) - (ns_0 - 1) * (np.mean(logk) - np.log(k_pivot)))
            print(f"Featureless PPS parameters estimated through linear regression: ns_0={ns_0}, As_0={As_0}")

        Pk_0 = As_0 * ((k_grid / k_pivot) ** (ns_0 - 1))

        integrand = np.exp(-logk) * (ns - 1)

        # (3/5)**3 comes from the conversion from R to zeta
        fact = (3/5)**3 * (2 * np.pi)**4 * (Pk_0 ** 2)
        term1 = fact * (np.sum(integrand) - np.cumsum(integrand)) * dlogk
        term2 = fact * (1 - ns)
        term3 = fact * alphas

        def shape_function(k1, k2, k3):
            meanK = (k1 + k2 + k3) / 2
            logmeanK = np.log(meanK)
            deltasq = meanK * (meanK - k1) * (meanK - k2) * (meanK - k3)

            t1 = np.interp(logmeanK, logk, term1) * deltasq
            t2 = np.interp(logmeanK, logk, term2) * (((k1**2 + k2**2 + k3**2) * (k1*k2 + k2*k3 + k3*k1) / (16 * meanK))
                                                    + ((k1*k2)**2 + (k2*k3)**2 + (k3*k1)**2) / (8 * meanK)
                                                    - (k1*k2*k3) / 8)
            t3 = np.interp(logmeanK, logk, term3) * (k1*k2*k3) / 8

            return (k1*k2*k3)**(-1) * (t1 + t2 + t3)
        
        return shape_function


    def calculate(self, state, want_derived=True, **params_values_dict):
        # Compute the bispectrum corresponding to P(k) under GSR

        prim_pk = self.provider.get_result("primordial_scalar_pk")
        assert(prim_pk["log_regular"])  # Currently only implemented for log-spaced grids

        Pk = prim_pk["Pk"]
        kvec = np.geomspace(prim_pk["kmin"], prim_pk["kmax"], len(Pk))

        shape_function = BispectrumFromPkGSR.shape_function_from_Pk(kvec, Pk)
        cmbbest_model = best.Model("custom", shape_function=shape_function, shape_name="Bispectrum from Pk")
        
        state["cmbbest_model"] = cmbbest_model


    def get_cmbbest_model(self):
        return self.current_state["cmbbest_model"]


    def get_info():
        # Get base info for cobaya runs

        info = {}
        info["theory"] = {"cmbfeat.bispectrum.BispectrumFromPkGSR": None}
        
        return info


