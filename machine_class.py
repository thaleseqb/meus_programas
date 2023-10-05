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

    def __init__(self, bunch=None, nturns=10, coord_idx=0,  coord_amp=0.02, coord_amp_nrpts=200):
        
        if bunch is None:
            bunch = _pyaccel.tracking.generate_bunch(n_part=self._nr_particles, envelope=None,
                                                     emit1=Params.h_emitt, emit2=Params.v_emitt,
                                                     sigmae=Params.sigmae,sigmas=Params.bun_len,
                                                     optics=Params.twiss[Params.nlk_index])
        self._nr_particles = 10
        self._bunch = bunch
        self._nturns = nturns
        self._coord_idx = coord_idx
        self._coord_amp = coord_amp
        self._coord_amp_nrpts = coord_amp_nrpts

    @property
    def nr_particles(self):
        return self._nr_particles
    
    @nr_particles.setter
    def nr_particles(self, new_particles):
        self._nr_particles = new_particles
    
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
    def coord_amp(self):
        return self._coord_amp
    
    @coord_amp.setter
    def coord_amp(self, new_amp):
        self._coord_amp = new_amp
    
    @property
    def coord_amp_nrpts(self):
        return self._coord_amp_nrpts
    
    @coord_amp_nrpts.setter
    def coord_amp_nrpts(self, new_amp_nrpts):
        self._coord_amp_nrpts = new_amp_nrpts


    def personalized_plot(self, mult_factor, coord_idx, label_y, scp_p_h, scp_n_h, scp_p_v, scp_n_v):

        ''' 
        mult_factor: Multiplicative factor that will fix the units of the graphic.
        coor_idx: Indicates in which coordinate the variation will be performed.
        label_y: Indicates the y label of the graphic, in all graphics shown in this plot, the y axis is shared.
        scp_p_h: Horizontal positive scraper width
        scp_n_h: Horizontal negative scraper width
        scp_p_v: Vertical positive scraper width
        scp_n_v: Vertical negative scraper width
        '''

        res = mchn_func.varying_incmnts(params=Params, bunch=self._bunch,
                                        nturn=self._nturns, coord_idx=self._coord_idx,
                                        coord_amp=self._coord_amp, coord_amp_nrpts=self._coord_amp_nrpts,
                                        scp_wid_posh=scp_p_h, scp_wid_negh=scp_n_h,
                                        scp_wid_posv=scp_p_v, scp_wid_negv=scp_n_v, m_fact=mult_factor)
        
        lostpos_n, lostpos_m, losttur_n, losttur_m, inc_qlst_n, inc_qlst_m = res

        mchn_func.p_sim(lostpos_n, lostpos_m, losttur_n, losttur_m, inc_qlst_n, inc_qlst_m, scp_p_h, scp_n_h, scp_p_v, scp_n_v, coord_idx)

        # se a coordenada for 0 o que interessa ser mostrado nos plots é a o scraper horizontal
        # o fator multiplicativo vai ser de 1e3 para corrigir para milimetros

        # se a coordenada for 1 o que interessa ser mostrado nos plots com certeza devo mostrar o o scraperhorizontal, devo perguntar ainda se devo mostrar também o scraper vertical
        # o fator multiplicativo ainda deve se3r definido

        # se a coordenada for 2 o que interessa ser mostrar nos plots é o scraper vertical
        # o fator multiplicativo provavelmente também 1e3 porque a unidade que saem os returns são dadas em metros 
        
        # se a coordenada for 3 o que interessa ser mostrar nos plots provavelmente são os dois, bom vamos ver
        # 

        return 

    
    # eu preciso definir alguns parametros que serão uteis para esta classe 
    # por exemplo preciso indicar quando cada parametro será variado 
    # quando eu definir cada parametro que sera variado eu também preciso 
    
    # eu também preciso modificar a função que vai realizar os plots dos gráficos para o estudo de maquina

