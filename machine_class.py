from typing import Any
import pyaccel as _pyaccel
import pymodels as _pymodels
import numpy as _np
import matplotlib.pyplot as _plt

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

    def __init__(self, accelerator):
        self.model = accelerator

    def function(self):
        return 
    
    # eu preciso definir alguns parametros que serão uteis para esta classe 
    # por exemplo preciso indicar quando cada parametro será variado 
    # quando eu definir cada parametro que sera variado eu também preciso 
    
    # fatores que multiplicarao devidamente meu gráfico para que tudo esteja visualmente compreensível
    # eu também preciso modificar a função que vai realizar os plots dos gráficos para o estudo de maquina
