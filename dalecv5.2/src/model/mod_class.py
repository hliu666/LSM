"""Dalecv2 model class takes a data class and then uses functions to run the
dalecv2 model.
"""
import numpy as np
import algopy

from RTM_Optical import rtm_o, rtm_o_mds, BRF_hemi_dif_func
from Ebal import Ebal
from PhotoSynth import PhotoSynth, calc_resp
from photo_pars import Rd25, Ear, T2K
         
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
    
    def dalecv2(self, p):
        """DALECV2 carbon balance model
        -------------------------------
        evolves carbon pools to the next time step, taking the 6 carbon pool
        values and 17 parameters at time t and evolving them to time t+1.
        Outputs both the 6 evolved C pool values and the 17 constant parameter
        values.
        
        phi_on = phi_onset(d_onset, cronset)
        phi_off = phi_fall(d_fall, crfall, clspan)
        gpp = acm(cf, clma, ceff)
        temp = temp_term(Theta)
        
        clab2 = (1 - phi_on)*clab + (1-f_auto)*(1-f_fol)*f_lab*gpp
        cf2 = (1 - phi_off)*cf + phi_on*clab + (1-f_auto)*f_fol*gpp
        cr2 = (1 - theta_roo)*cr + (1-f_auto)*(1-f_fol)*(1-f_lab)*f_roo*gpp
        cw2 = (1 - theta_woo)*cw + (1-f_auto)*(1-f_fol)*(1-f_lab)*(1-f_roo)*gpp
        cl2 = (1-(theta_lit+theta_min)*temp)*cl + theta_roo*cr + phi_off*cf
        cs2 = (1 - theta_som*temp)*cs + theta_woo*cw + theta_min*temp*cl
        """
        out = algopy.zeros(24, dtype=p)
      
        if self.x%365 == 150:
            self.dC.lai_const = max(p[17]/p[15], 1e-16)
            self.dC.sai_const = max(p[19]/10000, 1e-16)
        
        lai = max(p[17]/p[15], 1e-16)
        sai = max(p[19]/10000, 1e-16)
        
        if (self.x%365 > 150) and (self.x%365 < 280):
            if lai - self.dC.lai_const >0.0:
                lai = self.dC.lai_const
                sai = self.dC.sai_const

        refl_mds = []
        if self.x in self.dC.brf_data['index'].values:
            loc = self.dC.brf_data[self.dC.brf_data['index']==self.x].index.values[0]
            refl_mds = rtm_o_mds(self.dC, loc, lai+sai)
            
        nee_h, lst_h = [], []
        An_daily = 0 
        hemi_dif_brf = BRF_hemi_dif_func(self.dC.hemi_dif_pars, lai)

        #print(self.dC.D[self.x*24])
        #if self.dC.D[self.x*24] == 202:
        #    print("OKay")
            
        for h in range(0, 24):
            xh = self.x*24+h
            ebal_pars, k_pars = rtm_o(self.dC, xh, lai, hemi_dif_brf)
            #Ccu, Cch, Tcu, Tch, ecu, ech, APARu, APARh, Esolar, lst, Fc = Ebal_single(self.dC, xh, lai, ebal_pars, k_pars)
            Ccu, Cch, Tcu, Tch, ecu, ech, APARu, APARh, Esolar, lst, Fc = Ebal(self.dC, xh, lai, ebal_pars, k_pars)
            if (self.dC.tts[xh] < 75) and (lai > 0.5) and np.sum(Esolar[0][0:350])>0: 
                #----------------------two leaf model---------------------
                APARu = max(APARu, 1e-16) 
                APARh = max(APARh, 1e-16)  
                
                APARu_leaf, APARh_leaf = APARu/(lai*Fc), APARh/(lai*(1-Fc))
                meteo_u = [APARu_leaf, Ccu, Tcu, ecu]
                meteo_h = [APARh_leaf, Cch, Tch, ech]
                _,_,Anu = PhotoSynth(meteo_u)
                _,_,Anh = PhotoSynth(meteo_h)
                An = (Anu*Fc + Anh*(1-Fc))*lai

            else:
                Rdu = -calc_resp(Rd25, Ear, Tcu+T2K)
                Rdh = -calc_resp(Rd25, Ear, Tch+T2K)
                An = (Rdu*Fc + Rdh*(1-Fc))*lai
                
            An_daily += An
            #print(round(An,2), round(self.dC.sw[xh],2), round(APARu,2), round(APARh,2),round(lst,2))
            #nee = -(1. - p[1])*An + (p[7]*p[20] + p[8]*p[21])*self.temp_term(p[9], self.dC.t_mean[xh])
            nee = -An + (p[7]*p[20] + p[8]*p[21])*self.temp_term(p[9], self.dC.t_mean[xh])
            
            #record time series data 
            nee_h.append(nee)
            lst_h.append(lst)
        
        #1 umol CO2/m2/s = 1.03775 g C/day
        gpp = An_daily*1.03775/24
        t_mean_daily = np.mean(self.dC.t_mean[self.x*24:(self.x+1)*24])
        temp = self.temp_term(p[9], t_mean_daily)
                
        phi_on  = self.phi_onset(p[10], p[12])
        #phi_off = self.phi_fall(p[13], p[14], p[4])
        if (lai <= 0.5) and (self.dC.D[self.x*24] > 180): 
            phi_off = 1.0-1e-16  #only be used in the DBF!
        else:
            phi_off = self.phi_fall(p[13], p[14], p[4])
            
        out[16] = (1 - phi_on)*p[16] + (1-p[1])*(1-p[2])*p[11]*gpp
        out[17] = (1 - phi_off)*p[17] + phi_on*p[16] + (1-p[1])*p[2]*gpp
        out[18] = (1 - p[6])*p[18] + (1-p[1])*(1-p[2])*(1-p[11])*p[3]*gpp
        out[19] = (1 - p[5])*p[19] + (1-p[1])*(1-p[2])*(1-p[11])*(1-p[3])*gpp
        out[20] = (1 - (p[7]+p[0])*temp)*p[20] + p[6]*p[18] + phi_off*p[17]
        out[21] = (1 - p[8]*temp)*p[21] + p[5]*p[19] + p[0]*temp*p[20]
        out[22] = gpp 
        out[23] = lai
        
        out[0:16] = p[0:16]

        return out, nee_h, lst_h, refl_mds

    def mod_list(self, pvals):
        """Creates an array of evolving model values using dalecv2 function.
        Takes a list of initial param values.
        """
        mod_list = np.concatenate((np.array([pvals]),
                                  np.ones((self.endrun-self.startrun, len(pvals)))*-9999.))
        nee_y, lst_y, refls = [], [], []
        self.x = self.startrun
        for t in xrange(self.endrun-self.startrun):
            mod_list[(t+1)], nee_d, lst_d, refl_d = self.dalecv2(mod_list[t])
            nee_y  += nee_d
            lst_y  += lst_d
            if len(refl_d) > 1:
                refls.append(refl_d)

            self.x += 1

        self.x -= self.endrun
        return mod_list, np.array(nee_y), np.array(lst_y), np.array(refls)
