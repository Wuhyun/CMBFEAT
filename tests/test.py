import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cmbfeat
import cmbfeat.models
import cmbfeat.samplers
import cmbfeat.bispectrum
from cobaya.yaml import yaml_load_file
from cobaya.run import run
from cobaya.model import get_model

def test_import():
    assert hasattr(cmbfeat.models, "LinOscPrimordialPk")
    assert hasattr(cmbfeat.models, "LinEnvOscPrimordialPk")


class TestBaseLCDM():
    info = yaml_load_file("yaml/lcdm_planck_base.yaml")
    info["output"] = "tests/chains/test_base_lcdm/"
    info["force"] = True

    '''
    def test_mcmc_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_mcmc_sampler_info())
        info["sampler"]["mcmc"].update({"max_samples": 2})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''

    '''
    def test_polychord_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_polychord_sampler_info())
        info.update({"params": {"tau": {"prior": {"max": 0.15}}}})
        #info.update({"sampler": {"polychord": {"max_ndead": 2}}})

        model = get_model(info)
        updated_info, sampler = run(info)
    '''

    '''
    def test_minimizer(self):
        info = self.info
        info.update({"sampler": {"minimize": None}})

        model = get_model(info)
        updated_info, sampler = run(info)
    '''

class TestPowerSpectrum():

    #info1 = yaml_load_file("yaml/lcdm_planck_base.yaml")
    info1 = yaml_load_file("yaml/lcdm_planck_bestfit.yaml")
    info2 = cmbfeat.models.LinOscPrimordialPk.get_info()

    info = cmbfeat.models.merge_dicts([info1, info2])
    #print("####################", info)

    info["output"] = "tests/chains/test_power_spectrum/"
    info["force"] = True

    '''
    def test_mcmc_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_mcmc_sampler_info())
        info["sampler"]["mcmc"].update({"max_samples": 2})
        #info.update({"sampler": {"mcmc": {"Rminus1_stop": 0.1}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''

    '''
    def test_polychord_sampling(self):
        # Polychord doesn't seem to work well with pytest...

        info = self.info

        info.update(cmbfeat.samplers.base_polychord_sampler_info())
        info.update({"sampler": {"polychord": {"max_ndead": 2}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''



class TestBispectrum():

    info1 = cmbfeat.bispectrum.BispectrumLikelihood.get_info()
    info2 = cmbfeat.bispectrum.BispectrumDecomp.get_info()
    info3 = cmbfeat.models.LinOscPrimordialB.get_info()

    info = cmbfeat.models.merge_dicts([info1, info2, info3])
    #print("####################", info)

    info["output"] = "tests/chains/test_bispectrum/"
    info["force"] = True

    '''
    def test_mcmc_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_mcmc_sampler_info())
        info["sampler"]["mcmc"].update({"max_samples": 2})
        #info.update({"sampler": {"mcmc": {"Rminus1_stop": 0.1}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''

    '''
    def test_polychord_sampling(self):
        # Polychord doesn't seem to work well with pytest...

        info = self.info

        info.update(cmbfeat.samplers.base_polychord_sampler_info())
        info.update({"sampler": {"polychord": {"max_ndead": 2}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''

class TestBispectrumAlpha():

    info1 = cmbfeat.bispectrum.BispectrumLikelihoodFromAlpha.get_info()
    info2 = cmbfeat.bispectrum.BispectrumDecompAlpha.get_info()
    info3 = cmbfeat.models.LinOscPrimordialB.get_info()

    info = cmbfeat.models.merge_dicts([info1, info2, info3])
    #print("####################", info)

    info["output"] = "tests/chains/test_bispectrum_alpha/"
    info["force"] = True

    '''
    def test_mcmc_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_mcmc_sampler_info())
        info["sampler"]["mcmc"].update({"max_samples": 2})
        #info.update({"sampler": {"mcmc": {"Rminus1_stop": 0.1}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''

    '''
    def test_polychord_sampling(self):
        # Polychord doesn't seem to work well with pytest...

        info = self.info

        info.update(cmbfeat.samplers.base_polychord_sampler_info())
        info.update({"sampler": {"polychord": {"max_ndead": 2}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''


class TestBispectrumSNR():

    info1 = cmbfeat.bispectrum.BispectrumLikelihood.get_info(fnl_type="derived")
    info2 = cmbfeat.bispectrum.fnlSNR2fnl.get_info(sigma_bound=4.5)
    info3 = cmbfeat.bispectrum.BispectrumDecomp.get_info()
    info4 = cmbfeat.models.LinOscPrimordialB.get_info()

    info = cmbfeat.models.merge_dicts([info1, info2, info3, info4])
    #print("####################", info)

    info["output"] = "tests/chains/test_bispectrum_SNR/"
    info["force"] = True

    '''
    def test_mcmc_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_mcmc_sampler_info())
        info["sampler"]["mcmc"].update({"max_samples": 2})
        #info.update({"sampler": {"mcmc": {"Rminus1_stop": 0.1}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''

    '''
    def test_polychord_sampling(self):
        # Polychord doesn't seem to work well with pytest...

        info = self.info

        info.update(cmbfeat.samplers.base_polychord_sampler_info())
        info.update({"sampler": {"polychord": {"max_ndead": 2}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''



class TestBispectrumFromPk():

    info1 = cmbfeat.bispectrum.BispectrumLikelihood.get_info()
    info2 = cmbfeat.bispectrum.BispectrumDecomp.get_info()
    info3 = cmbfeat.bispectrum.BispectrumFromPkGSR.get_info()
    info4 = cmbfeat.models.LinOscPrimordialPk.get_info()
    #info4 = {"theory": {"cmbfeat.models.PowerLawPrimordialPk": None}}

    info = cmbfeat.models.merge_dicts([info1, info2, info3, info4])
    #print("####################", info)

    info["params"].update({"As": {"value": 2.1e-9, "latex": r"A_\mathrm{s}"},
                           "ns": {"value": 0.9649, "latex": r"n_\mathrm{s}"}})

    info["output"] = "tests/chains/test_bispectrum_from_pk/"
    info["force"] = True

    '''
    def test_mcmc_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_mcmc_sampler_info())
        info["sampler"]["mcmc"].update({"max_samples": 2})
        #info.update({"sampler": {"mcmc": {"Rminus1_stop": 0.05}}})
        info["sampler"] = {"evaluate": {"override": {"A_osc": 0.3, "omega_osc": 300}}}
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''

    '''
    def test_polychord_sampling(self):
        # Polychord doesn't seem to work well with pytest...

        info = self.info

        info.update(cmbfeat.samplers.base_polychord_sampler_info())
        info.update({"sampler": {"polychord": {"max_ndead": 2}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''

class TestYAML():

    #filenames = ["lcdm_planck_base.yaml", "linosc_B_cmbbest.yaml",
    #            "lcdm_planck_bestfit.yaml",	"linosc_Pk_planck_bestfit.yaml"]
    filenames = ["linosc_PkGSR_cmbbest.yaml", "linosc_Pk_planck_bestfit.yaml", "linosc_PkGSR_joint.yaml",
                "linenvosc_PkGSR_cmbbest.yaml", "linenvosc_Pk_planck_bestfit.yaml", "linenvosc_PkGSR_joint.yaml"]
    #filenames = ["linosc_B_cmbbest.yaml"]
                

    '''
    def test_load_yaml(self):
        print("##########!!")
        for filename in self.filenames:
            info = yaml_load_file("yaml/" + filename)
            print("##########", info)
            model = get_model(info)
    '''
    #'''
    def test_run_yaml(self):
        for filename in self.filenames:
            info = yaml_load_file("yaml/" + filename)
            #info["sampler"]["mcmc"].update({"max_samples": 2})
            info["sampler"] = {"evaluate": None}
            updated_info, sampler = run(info)
    #'''