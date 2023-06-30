"""Dalecv2 model class takes a data class and then uses functions to run the
dalecv2 model.
"""
import numpy as np
import algopy

from RTM_Optical import rtm_o, rtm_o_mds, BRF_hemi_dif_func
from Ebal import Ebal, Ebal_single
from PhotoSynth import PhotoSynth

xrange = range
class DalecModel():

    def __init__(self, dataclass, time_step=0, startrun=0):
        """ Model class for running DALEC2
        :param dataclass: DALEC2 data class containing data to run model
        :param time_step: time step of model to begin with
        :param strtrun: where to begin model runs within data
        :return:
        """
        self.dC = dataclass
        self.x = time_step
        self.lenrun = self.dC.len_run
        self.startrun = startrun
        self.endrun = self.lenrun

# ------------------------------------------------------------------------------
# Model functions (See Bloom and Williams 2015 for more details)
# ------------------------------------------------------------------------------
    @staticmethod
    def fit_polynomial(ep, mult_fac):
        """ Polynomial used to find phi_f and phi (offset terms used in
        phi_onset and phi_fall), given an evaluation point for the polynomial
        and a multiplication term.
        :param ep: evaluation point
        :param mult_fac: multiplication term
        :return: fitted polynomial value
        """
        cf = [2.359978471e-05, 0.000332730053021, 0.000901865258885,
              -0.005437736864888, -0.020836027517787, 0.126972018064287,
              -0.188459767342504]
        poly_val = cf[0]*ep**6 + cf[1]*ep**5 + cf[2]*ep**4 + cf[3]*ep**3 + cf[4]*ep**2 + \
            cf[5]*ep**1 + cf[6]*ep**0
        phi = poly_val*mult_fac
        return phi

    def temp_term(self, Theta, temperature):
        """ Calculates the temperature exponent factor for carbon pool
        respiration's given a value for Theta parameter.
        :param Theta: temperature dependence exponent factor
        :return: temperature exponent respiration
        """
        temp_term = np.exp(Theta*temperature)
        return temp_term

    def phi_onset(self, d_onset, cronset):

        """Leaf onset function (controls labile to foliar carbon transfer)
        takes d_onset value, cronset value and returns a value for phi_onset.
        """
        release_coeff = np.sqrt(2.)*cronset / 2.
        mag_coeff = (np.log(1.+1e-3) - np.log(1e-3)) / 2.
        offset = self.fit_polynomial(1+1e-3, release_coeff)
        phi_onset = (2. / np.sqrt(np.pi))*(mag_coeff / release_coeff) * \
            np.exp(-(np.sin((self.dC.D[self.x*24] - d_onset + offset) /
                     self.dC.radconv)*(self.dC.radconv / release_coeff))**2)
        return phi_onset

    def phi_fall(self, d_fall, crfall, clspan):
        """Leaf fall function (controls foliar to litter carbon transfer) takes
        d_fall value, crfall value, clspan value and returns a value for
        phi_fall.
        """
        release_coeff = np.sqrt(2.)*crfall / 2.
        mag_coeff = (np.log(clspan) - np.log(clspan - 1.)) / 2.
        offset = self.fit_polynomial(clspan, release_coeff)
        phi_fall = (2. / np.sqrt(np.pi))*(mag_coeff / release_coeff) * \
            np.exp(-(np.sin((self.dC.D[self.x*24] - d_fall + offset) /
                   self.dC.radconv)*self.dC.radconv / release_coeff)**2)
        return phi_fall
    
    def dalecv2(self, lai, sai):
        refl_mds = []
        if self.x in self.dC.brf_data['index'].values:
            loc = self.dC.brf_data[self.dC.brf_data['index']==self.x].index.values[0]
            refl_mds = rtm_o_mds(self.dC, loc, lai+sai)
 
        return refl_mds

    def mod_list(self, lai, sai):
        refls = []
        lais = []
        self.x = self.startrun
        for t in xrange(self.endrun-self.startrun):
            refl_d = self.dalecv2(lai[t], sai[t])

            if len(refl_d) > 1:
                refls.append(refl_d)
                lais.append(lai[t]+sai[t])

            self.x += 1

        self.x -= self.endrun
        return np.array(refls), np.array(lais), #np.array(fPAR_y), np.array(refls)
