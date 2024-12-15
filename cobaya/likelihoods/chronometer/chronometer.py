r"""
module: chronometer
Synopsis: Cosmic Chronometer class for Hubble constant measurements 
Author: Nisha Kamble
"""

from cobaya.likelihoods.base_classes import H0
import numpy as np
import pandas as pd

class chronometer(H0):
    type = "chronometer"
    
    def initialize(self):
        # raise error if data_file is not provided
        if not self.data_file:
            raise ValueError("data_file is not provided")
        
        self.data = pd.read_csv(self.data_file, delim_whitespace=True, comment='#', header=None)
        if len(self.data.columns) == 4:
            self.data.columns = ['[z]', '[H0 at z]', '[error]', '[dataset]']
        else:
            raise ValueError("data_file should have 4 columns: [z], [H0 at z], [error], [dataset]")
        
    def get_requirements(self):
        return {'H0': None}

    def logp(self, **params_values):
        """
        calculate the log-likelihood for all datasets
        """
        H0_theory = params_values['H0']
        chi2=0
        for _, row in self.data.iterrows():
            H0_mean = row['[H0 at z]']
            H0_std = row['[error]']
            chi2 += -0.5*((H0_mean - H0_theory)**2 / H0_std)**2
        return chi2

class chronometer_all(chronometer):
    r"""
    Cosmic Chronometers data for 35 measurements
    """