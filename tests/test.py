import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cmbfeat
import cmbfeat.models
import cmbfeat.samplers
import cmbfeat.likelihoods
from cobaya.yaml import yaml_load_file
from cobaya.run import run
from cobaya.model import get_model

def test_import():
    assert hasattr(cmbfeat.models, "LinOscPrimordialPk")
    assert hasattr(cmbfeat.models, "LinEnvOscPrimordialPk")


class TestBaseLCDM():
    info = yaml_load_file("tests/yaml/planck_lcdm_base.yaml")
    info["output"] = "tests/chains/test_base_lcdm/"
    info["force"] = True

    '''
    def test_mcmc_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_mcmc_sampler_info())
        info.update({"sampler": {"mcmc": {"max_samples": 2}}})
        
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


class TestBispectrum():

    info = {} 
    info1 = cmbfeat.likelihoods.BispectrumLikelihood.get_info()
    info2 = cmbfeat.likelihoods.BispectrumDecomp.get_info()
    info3 = cmbfeat.models.LinOscPrimordialB.get_info()

    for key in ["theory", "likelihood", "params"]:
        info[key] = {}
        for sub in [info1, info2, info3]:
            if key in sub.keys():
                info[key] = info[key] | sub[key]

    print("####################", info)

    info["output"] = "tests/chains/test_bispectrum/"
    info["force"] = True

    '''
    def test_mcmc_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_mcmc_sampler_info())
        #info.update({"sampler": {"mcmc": {"max_samples": 2}}})
        info.update({"sampler": {"mcmc": {"Rminus1_stop": 0.1}}})
        
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

    info = {} 
    info1 = cmbfeat.likelihoods.BispectrumLikelihood.get_info(unit_fnl=True)
    info2 = cmbfeat.likelihoods.BispectrumDecomp.get_info()
    info3 = cmbfeat.likelihoods.BispectrumFromPkGSR.get_info()
    info4 = cmbfeat.models.LinOscPrimordialPk.get_info()

    for key in ["theory", "likelihood", "params"]:
        info[key] = {}
        for sub in [info1, info2, info3, info4]:
            if key in sub.keys():
                info[key] = info[key] | sub[key]

    print("####################", info)

    info["params"].update({"As": {"value": 2.1e-9, "latex": r"A_\mathrm{s}"},
                           "ns": {"value": 0.9649, "latex": r"n_\mathrm{s}"}})

    info["output"] = "tests/chains/test_bispectrum_from_pk/"
    info["force"] = True

    #'''
    def test_mcmc_sampling(self):
        info = self.info

        info.update(cmbfeat.samplers.base_mcmc_sampler_info())
        #info.update({"sampler": {"mcmc": {"max_samples": 2}}})
        info.update({"sampler": {"mcmc": {"Rminus1_stop": 0.05}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    #'''

    '''
    def test_polychord_sampling(self):
        # Polychord doesn't seem to work well with pytest...

        info = self.info

        info.update(cmbfeat.samplers.base_polychord_sampler_info())
        info.update({"sampler": {"polychord": {"max_ndead": 2}}})
        
        model = get_model(info)
        updated_info, sampler = run(info)
    '''
