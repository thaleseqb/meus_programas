from typing import Any
import pyaccel as _pyaccel
import pymodels as _pymodels
import numpy as _np
import matplotlib.pyplot as _plt
import machine_functions as mchn_func

class Params:
    scraper_name_h = 'SHVC'
    scraper_name_v = 'SVVC'
    
    # create model
    fitm = _pymodels.si.create_accelerator()
    fitm = _pymodels.si.fitted_models.vertical_dispersion_and_coupling(fitm)
    fitm.vchamber_on = True
    fitm.radiation_on = True
    fitm.cavity_on = True
    print(fitm)
    
    booster = _pymodels.bo.create_accelerator()
    booster.energy = 3e9
    booster[187].voltage = 1e6
    booster.radiation_on=True
    booster.cavity_on=True
    
    # scraper data
    scraper_indices_h = _pyaccel.lattice.find_indices(fitm, 'fam_name', scraper_name_h)
    scraper_indices_v = _pyaccel.lattice.find_indices(fitm, 'fam_name', scraper_name_v)
    scraper_width0 = _pyaccel.lattice.get_attribute(fitm, 'hmax', indices=scraper_indices_h)[0] # first element because the vchamber's height is equal for the two indices 
    scraper_height0 = _pyaccel.lattice.get_attribute(fitm, 'vmax', indices=scraper_indices_v)[0]

    # calc optics
    spos = _pyaccel.lattice.find_spos(fitm, indices='closed')
    twiss,*_ = _pyaccel.optics.calc_twiss(fitm, indices='closed')
    
    # equilibrium parameters
    eqparams = _pyaccel.optics.beam_envelope.EqParamsFromBeamEnvelope(booster)
    coup = 1/100
    emitt0 = eqparams.emit1  # nno-coupling model
    h_emitt = 1/(1 + coup) * emitt0
    v_emitt = coup/(1 + coup) * emitt0
    print('hemitt [pm.rad]:', h_emitt * 1e12)
    print('vemitt [pm.rad]:', v_emitt * 1e12)
    sigmae = eqparams.espread0
    bun_len = eqparams.bunlen

    # defining the index 
    fam_name_dict = _pyaccel.lattice.find_dict(fitm, 'fam_name')
    nlk_index = fam_name_dict['InjNLKckr'][0] + 1

class Machine_study(Params):

    def __init__(self, bunch=None, nturns=10, coord_idx=0,
                 coord_amp_p=0.02, coord_amp_n=-0.02, coord_amp_nrpts=200, ):
        
        self._nr_particles = 10
        if bunch is None:
            bunch = _pyaccel.tracking.generate_bunch(n_part=self._nr_particles, envelope=None,
                                                     emit1=Params.h_emitt, emit2=Params.v_emitt,
                                                     sigmae=Params.sigmae,sigmas=Params.bun_len,
                                                     optics=Params.twiss[Params.nlk_index])
        
        self._bunch = bunch
        self._nturns = nturns
        self._coord_idx = coord_idx
        self._coord_amp_p = coord_amp_p
        self._coord_amp_n = coord_amp_n
        self._coord_amp_nrpts = coord_amp_nrpts

    @property
    def nr_particles(self):
        return self._nr_particles
    
    @nr_particles.setter
    def nr_particles(self, new_particles):
        self._nr_particles = new_particles

    @property
    def bunch(self):
        return self._bunch
    
    @bunch.setter
    def bunch(self, new_particles):
        self._bunch = _pyaccel.tracking.generate_bunch(n_part=new_particles, envelope=None,
                                                     emit1=Params.h_emitt, emit2=Params.v_emitt,
                                                     sigmae=Params.sigmae,sigmas=Params.bun_len,
                                                     optics=Params.twiss[Params.nlk_index])
    
    @property
    def nturns(self):
        return self._nturns
    
    @nturns.setter
    def nturns(self, new_turns):
        self._nturns = new_turns

    @property
    def coord_idx(self):
        return self._coord_idx
    
    @coord_idx.setter
    def coord_idx(self, new_coord):
        self._coord_idx = new_coord

    @property
    def coord_amp_p(self):
        return self._coord_amp_p
    
    @coord_amp_p.setter
    def coord_amp_p(self, new_amp):
        self._coord_amp_p = new_amp

    @property
    def coord_amp_n(self):
        return self._coord_amp_n
    
    @coord_amp_n.setter
    def coord_amp_n(self, new_amp):
        self._coord_amp_n = new_amp
    
    @property
    def coord_amp_nrpts(self):
        return self._coord_amp_nrpts
    
    @coord_amp_nrpts.setter
    def coord_amp_nrpts(self, new_amp_nrpts):
        self._coord_amp_nrpts = new_amp_nrpts


    def personalized_plot(self, mult_factor, coord_idx, scp_p_h, scp_n_h, scp_p_v, scp_n_v):

        ''' 
        mult_factor: Multiplicative factor that will fix the units of the graphic.
        coor_idx: Indicates in which coordinate the variation will be performed.
        label_y: Indicates the y label of the graphic, in all graphics shown in this plot, the y axis is shared.
        scp_p_h: Horizontal positive scraper width
        scp_n_h: Horizontal negative scraper width
        scp_p_v: Vertical positive scraper width
        scp_n_v: Vertical negative scraper width
        '''

        res_pos = mchn_func.varying_incmnts(params=Params, bunch=self._bunch,
                                        nturn=self._nturns, coord_idx=self._coord_idx,
                                        coord_amp=self._coord_amp_p, coord_amp_nrpts=self._coord_amp_nrpts,
                                        scp_wid_posh=scp_p_h, scp_wid_negh=scp_n_h,
                                        scp_wid_posv=scp_p_v, scp_wid_negv=scp_n_v, m_fact=mult_factor)
        
        res_neg = mchn_func.varying_incmnts(params=Params, bunch=self._bunch,
                                        nturn=self._nturns, coord_idx=self._coord_idx,
                                        coord_amp=self._coord_amp_n, coord_amp_nrpts=self._coord_amp_nrpts,
                                        scp_wid_posh=scp_p_h, scp_wid_negh=scp_n_h,
                                        scp_wid_posv=scp_p_v, scp_wid_negv=scp_n_v, m_fact=mult_factor)
        
        lostpos_np,lostpos_mp, losttur_np,losttur_mp, inc_qlst_np, inc_qlst_mp = res_pos
        lostpos_nn,lostpos_mn, losttur_nn,losttur_mn, inc_qlst_nn, inc_qlst_mn = res_neg

        rlostposn = _np.r_[_np.array(lostpos_nn, dtype=object),_np.array(lostpos_np, dtype=object)]
        rlostturn = _np.r_[_np.array(losttur_nn),_np.array(losttur_np)]
        rincn = _np.r_[_np.array(inc_qlst_np),_np.array(inc_qlst_nn)]

        rlostposm = _np.r_[_np.array(lostpos_mn, dtype=object),_np.array(lostpos_mp, dtype=object)]
        rlostturm = _np.r_[_np.array(losttur_mn),_np.array(losttur_mp)]
        rincm = _np.r_[_np.array(inc_qlst_mp),_np.array(inc_qlst_mn)]

        mchn_func.p_sim(Params,rlostposn, rlostposm, rlostturn, rlostturm, rincn, rincm, scp_p_h, scp_n_h, scp_p_v, scp_n_v, coord_idx=coord_idx)

        return


