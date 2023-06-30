"""
TBM model class takes a data class and then uses functions to run the TBM model.
"""
import numpy as np
from constants import T2K

from RTM_Optical import rtm_o, BRF_hemi_dif_func
from Ebal import Ebal
from PhotoSynth_Jen import PhotoSynth_Jen, calc_resp
from SIF import cal_sif_leaf, cal_canopy_sif
from hydraulics import calc_hy_f

xrange = range


class TBM_Model():

    def __init__(self, dataclass, pramclass, time_step=0, startrun=0):
        """ Model class for running DALEC2
        :param dataclass: TBM data class containing data to run model
        :param time_step: time step of model to begin with
        :param strtrun: where to begin model runs within data
        :return:
        """
        self.d = dataclass
        self.p = pramclass
        self.x = time_step
        self.lenrun = self.d.len_run
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
        poly_val = cf[0] * ep ** 6 + cf[1] * ep ** 5 + cf[2] * ep ** 4 + cf[3] * ep ** 3 + cf[4] * ep ** 2 + \
                   cf[5] * ep ** 1 + cf[6] * ep ** 0
        phi = poly_val * mult_fac
        return phi

    def temp_term(self, Theta, temperature):
        """ Calculates the temperature exponent factor for carbon pool
        respiration's given a value for Theta parameter.
        :param Theta: temperature dependence exponent factor
        :return: temperature exponent respiration
        """
        temp_term = np.exp(Theta * temperature)
        return temp_term

    def phi_onset(self, d_onset, cronset):
        """Leaf onset function (controls labile to foliar carbon transfer)
        takes d_onset value, cronset value and returns a value for phi_onset.
        """
        release_coeff = np.sqrt(2.) * cronset / 2.
        mag_coeff = (np.log(1. + 1e-3) - np.log(1e-3)) / 2.
        offset = self.fit_polynomial(1 + 1e-3, release_coeff)
        phi_onset = (2. / np.sqrt(np.pi)) * (mag_coeff / release_coeff) * \
                    np.exp(-(np.sin((self.d.D[self.x * 24] - d_onset + offset) /
                                    self.p.radconv) * (self.p.radconv / release_coeff)) ** 2)
        return phi_onset

    def phi_fall(self, d_fall, crfall, clspan):
        """Leaf fall function (controls foliar to litter carbon transfer) takes
        d_fall value, crfall value, clspan value and returns a value for phi_fall.
        """
        release_coeff = np.sqrt(2.) * crfall / 2.
        mag_coeff = (np.log(clspan) - np.log(clspan - 1.)) / 2.
        offset = self.fit_polynomial(clspan, release_coeff)
        phi_fall = (2. / np.sqrt(np.pi)) * (mag_coeff / release_coeff) * \
                   np.exp(-(np.sin((self.d.D[self.x * 24] - d_fall + offset) /
                                   self.p.radconv) * self.p.radconv / release_coeff) ** 2)
        return phi_fall

    def tbm(self, pd, ph):
        """TBM carbon balance model
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

        An_daily = 0
        pd_out = np.zeros_like(pd)
        ph_out = np.zeros_like(ph)
        clab, cf, cr, cw, cl, cs = pd[0], pd[1], pd[2], pd[3], pd[4], pd[5]
        lai = max(cf / self.p.lma, 1e-16)
        hemi_dif_brf = BRF_hemi_dif_func(self.d.hemi_dif_pars, lai)

        for h in range(0, 24):
            xh = self.x * 24 + h
            rtm_o_dict = rtm_o(self.d, self.p, xh, lai, hemi_dif_brf)
            Ebal_dict = Ebal(self.d, self.p, xh, lai, rtm_o_dict)

            # ----------------------canopy intercepted wator and soil moisture factor---------------------
            self.d.w_can[xh + 1], fwet, self.d.sm_top[xh + 1], sf = calc_hy_f(self.d, self.p, xh, lai, Ebal_dict['Ev'], Ebal_dict['ET'])

            if (self.d.tts[xh] < 75) and np.sum(Ebal_dict['Esolars'][0][0:350]) > 0 and (lai > 0.5):
                # ----------------------two leaf model---------------------
                APARu = max(Ebal_dict['APARu'], 1e-16)
                APARh = max(Ebal_dict['APARh'], 1e-16)

                APARu_leaf, APARh_leaf = APARu / (lai * Ebal_dict['Fc']), APARh / (lai * (1 - Ebal_dict['Fc']))

                meteo_u = [APARu_leaf, Ebal_dict['Ccu'], Ebal_dict['Tcu'], Ebal_dict['ecu'], sf]
                meteo_h = [APARh_leaf, Ebal_dict['Cch'], Ebal_dict['Tch'], Ebal_dict['ech'], sf]

                rcw_u, _, Anu, fqe2u, fqe1u = PhotoSynth_Jen(meteo_u, self.p)
                rcw_h, _, Anh, fqe2h, fqe1h = PhotoSynth_Jen(meteo_h, self.p)

                An = (Anu * Ebal_dict['Fc'] + Anh * (1 - Ebal_dict['Fc'])) * lai

                fqe_pars = [fqe2u, fqe1u, fqe2h, fqe1h]
                Mu_pars, Mh_pars = cal_sif_leaf(self.d, fqe_pars)

                sif_u = cal_canopy_sif(self.d, xh, Ebal_dict['Esolars'], Mu_pars, rtm_o_dict, hemi_dif_brf)
                sif_h = cal_canopy_sif(self.d, xh, Ebal_dict['Esolars'] , Mh_pars, rtm_o_dict, hemi_dif_brf)

            else:
                Rdu = -calc_resp(self.p.Rd25, self.p.Ear, Ebal_dict['Tcu'] + T2K)
                Rdh = -calc_resp(self.p.Rd25, self.p.Ear, Ebal_dict['Tch'] + T2K)
                An = (Rdu * Ebal_dict['Fc'] + Rdh * (1 - Ebal_dict['Fc'])) * lai

                fqe_pars = [0.0, 0.0, 0.0, 0.0]

                sif_u = [0.0] * 9
                sif_h = [0.0] * 9

            An_daily += An
            nee = -An + (self.p.theta_lit * cl + self.p.theta_som * cs) * self.temp_term(self.p.Theta,
                                                                                         self.d.t_mean[xh])
            ph_out[h] = [nee, An, rtm_o_dict['fPAR'], Ebal_dict['LST'],
                         Ebal_dict['ERnuc'], Ebal_dict['ELnuc'], Ebal_dict['ERnhc'], Ebal_dict['ELnhc'],
                         sif_u[0], sif_u[1], sif_h[0], sif_h[1], Ebal_dict['Tcu'], Ebal_dict['Tch'], Ebal_dict['Tsu'], Ebal_dict['Tsh']]

        # 1 umol CO2/m2/s = 1.03775 g C/day
        gpp = An_daily * 1.03775 / 24
        t_mean_daily = np.mean(self.d.t_mean[self.x * 24:(self.x + 1) * 24])
        temp = self.temp_term(self.p.Theta, t_mean_daily)
        phi_on = self.phi_onset(self.p.d_onset, self.p.cronset)
        phi_off = self.phi_fall(self.p.d_fall, self.p.crfall, self.p.clspan)

        clab2 = (1 - phi_on) * clab + (1 - self.p.f_auto) * (1 - self.p.f_fol) * self.p.f_lab * gpp
        cf2 = (1 - phi_off) * cf + phi_on * clab + (1 - self.p.f_auto) * self.p.f_fol * gpp
        cr2 = (1 - self.p.theta_roo) * cr + (1 - self.p.f_auto) * (1 - self.p.f_fol) * (
                    1 - self.p.f_lab) * self.p.f_roo * gpp
        cw2 = (1 - self.p.theta_woo) * cw + (1 - self.p.f_auto) * (1 - self.p.f_fol) * (1 - self.p.f_lab) * (
                    1 - self.p.f_roo) * gpp
        cl2 = (1 - (self.p.theta_lit + self.p.theta_min) * temp) * cl + self.p.theta_roo * cr + phi_off * cf
        cs2 = (1 - self.p.theta_som * temp) * cs + self.p.theta_woo * cw + self.p.theta_min * temp * cl

        pd_out[:] = [clab2, cf2, cr2, cw2, cl2, cs2, gpp, lai]

        return pd_out, ph_out

    def mod_list(self, output_dim1, output_dim2):
        """Creates an array of evolving model values using dalecv2 function.
        Takes a list of initial param values.
        """
        mod_list_daily = np.full((self.endrun - self.startrun + 1, output_dim1), 0.0)
        mod_list_daily[0, 0:6] = [self.p.clab, self.p.cf, self.p.cr, self.p.cw, self.p.cl, self.p.cs]
        mod_list_hourly = np.full((self.endrun - self.startrun + 1, 24, output_dim2), 0.0)

        self.x = self.startrun
        for t in xrange(self.endrun - self.startrun):
            mod_list_daily[(t + 1)], mod_list_hourly[(t + 1)] = self.tbm(mod_list_daily[t], mod_list_hourly[t])
            self.x += 1

        self.x -= self.endrun
        return mod_list_daily, mod_list_hourly
