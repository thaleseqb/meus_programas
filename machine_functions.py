#!/usr/bin/env python-sirius

import pyaccel as _pyaccel
import pymodels
import numpy as _np
import matplotlib.pyplot as _plt



def set_vchamber_h(params, scp_wid_posh, scp_wid_negh, scp_wid_posv, scp_wid_negv):
    for iten in params.scraper_indices_h:
        params.fitm[iten].hmax = scp_wid_posh
        params.fitm[iten].hmin = -scp_wid_negh
    for iten in params.scraper_indices_v:
        params.fitm[iten].vmax = scp_wid_posv
        params.fitm[iten].vmin = -scp_wid_negv
    
    return None


def track_mchn_stdy(params, bunch, nturn, coord_idx, increment):
    # setting param to be the index of the array [x,x',y,y',delta,z]

    bunch[coord_idx] += increment
    tracked = _pyaccel.tracking.ring_pass(params.fitm, particles=bunch,
                                         nr_turns=nturn, turn_by_turn= True,
                                         element_offset=params.nlk_index, parallel=False)
    
    part_out, lost_flag, turn_lost, index_lost, plane_lost = tracked
    # print(lost_flag)
    
    turnl_element = []

    for i,iten in enumerate(turn_lost):
        if iten == nturn and index_lost[i] == params.nlk_index: # ignora elétrons que não foram perdidos
            pass
        else:
            turnl_element.append((iten, index_lost[i]))
    
    return turnl_element

def varying_incmnts(params, bunch, nturn, coord_idx, coord_amp, coord_amp_nrpts, scp_wid_posh, scp_wid_negh, scp_wid_posv, scp_wid_negv, m_fact):

    ''' 
        params: parameters used in simulation like the accelerator model 
        bunch: particle initial conditions
        nturn: number of turns
        coord_idx: index of [x,x',y,y',delta,z] -> [0,1,2,3,4,5]
        coord_amp_nrpts: range of the increment
        coord_amp: variable that goes in numpy linspace, the value till the variation must be performed
        scraper_width1 and scraper_width2: is the value that the user desires for setting the height of vchamber
        there are two widths the explanation for this is let the user choice when using simetric vacuum chamber widths or not 
    '''

    lostpos_n = []
    lostpos_m = []
    losttur_n = []
    losttur_m = []

    inc = _np.linspace(0, coord_amp, coord_amp_nrpts)
    incdif = _np.diff(inc)

    inc_qlst_n = []
    inc_qlst_m = []

    # nominal model tracking simulation
    bunchi = bunch.copy()
    for j, iten in enumerate(incdif):
        track = track_mchn_stdy(params, bunchi, nturn=nturn, coord_idx=coord_idx, increment=iten)
        length = len(track)
        if len(track) == 0:
            pass
        else:
            mean = _np.mean(bunchi[coord_idx]) * m_fact
            inc_qlst_n.append([mean,length]) # this will be the y axis of the graphic
            for lst in track:
                lost_turn, lost_pos = lst
                losttur_n.append([lost_turn, mean])
                lostpos_n.append([lost_pos, mean])

    # # changed scraper width within model tracking simulation
    set_vchamber_h(params, scp_wid_posh, scp_wid_negh, scp_wid_posv, scp_wid_negv) # after calling this function, vchamber's width will be changed
    print(params.fitm[params.scraper_indices_h[0]])
    
    bunchi = bunch.copy()
    for j, iten in enumerate(incdif):
        track = track_mchn_stdy(params, bunchi, nturn=nturn, coord_idx=coord_idx, increment=iten)
        length = len(track)
        if len(track) == 0:
            pass
        else:
            mean = _np.mean(bunchi[coord_idx])* m_fact
            inc_qlst_m.append([mean,length]) 
            for lst in track:
                lost_turn, lost_pos = lst
                losttur_m.append([lost_turn, mean])
                lostpos_m.append([lost_pos, mean])
    set_vchamber_h(params, params.scraper_width0, params.scraper_width0, params.scraper_height0, params.scraper_height0) # after calling this function, vchamber's height will be restored

    return lostpos_n, lostpos_m, losttur_n, losttur_m, inc_qlst_n, inc_qlst_m


def p_sim(params,lostpos_n, lostpos_m, losttur_n, losttur_m, inc_qlst_n, inc_qlst_m, scp_w1h, scp_w2h, scp_w1v, scp_w2v, coord_idx):
    fig1, (a1n,a2n,a3n) = _plt.subplots(nrows=1,ncols=3, sharey=True, figsize=(10,5))
    h = _np.mean(_np.array(lostpos_n)[:,1])
    
    #defining the title of the graphics
    if scp_w1h == scp_w2h and scp_w1v == scp_w2v:
        a3n.plot(params.spos[lostpos_n[0][0]], lostpos_n[0][1], label='(scp_h {} [mm]), (scp_v {} [mm])'.format(params.scraper_width0*1e3,params.scraper_height0*1e3), color='blue')
        a3n.plot(params.spos[lostpos_m[0][0]], lostpos_m[0][1], label='(scp_h {} [mm]), (scp_v {} [mm])'.format(scp_w1h*1e3, scp_w1v*1e3), color='red')
    else:
        a3n.plot(params.spos[lostpos_n[0][0]], lostpos_n[0][1], label='(scp_h {} [mm]), (scp_v {} [mm])'.format(params.scraper_width0*1e3,params.scraper_height0*1e3), color='blue')
        a3n.plot(params.spos[lostpos_m[0][0]], lostpos_m[0][1], label='(scp_h {}, {} [mm]), (scp_v {}, {} [mm])'.format(scp_w1h*1e3, scp_w2h*1e3, scp_w1v*1e3, scp_w2v*1e3), color='red')

    # if scp_w1h == scp_w2h and scp_w1v == scp_w2v:
    #     a3n.plot(params.spos[lostpos_n[0][0]], lostpos_n[0][1], label='{}, {}'.format(params.scraper_width0*1e3,params.scraper_height0*1e3), color='blue')
    #     a3n.plot(params.spos[lostpos_m[0][0]], lostpos_m[0][1], label='{}, {}'.format(scp_w1h*1e3, scp_w1v*1e3), color='red')
    # else:
    #     a3n.plot(params.spos[lostpos_n[0][0]], lostpos_n[0][1], label='{}, {}'.format(params.scraper_width0*1e3,params.scraper_height0*1e3), color='blue')
    #     a3n.plot(params.spos[lostpos_m[0][0]], lostpos_m[0][1], label='{}, {}, {}, {}'.format(scp_w1h*1e3, scp_w2h*1e3, scp_w1v*1e3, scp_w2v*1e3), color='red')

    a3n.legend(fontsize=12)
    
    fig1.subplots_adjust(hspace=0.1)


    for iten in inc_qlst_n:
        a1n.plot(iten[1], iten[0] , '.', color='blue', alpha = 0.4)
    a1n.set_xlabel(r'Number of electrons lost', fontsize=16)
    
    if coord_idx == 0:
        a1n.set_ylabel(r'horizontal position mean [mm]', fontsize=16)
    elif coord_idx == 1:
        a1n.set_ylabel(r'$x^{\prime}$ mean [mrad]', fontsize=16)
    elif coord_idx == 2:
        a1n.set_ylabel(r'vertical position mean [mm]', fontsize=16)
    elif coord_idx == 3:
        a1n.set_ylabel(r'$y^{\prime}$ mean [mrad]', fontsize=16)
    
    a1n.grid(True, alpha=0.5, ls='--', color='k')
    a2n.grid(True, alpha=0.5, ls='--', color='k')
    a3n.grid(True, alpha=0.5, ls='--', color='k')
    a1n.tick_params(axis='both', labelsize=12)
    a2n.tick_params(axis='both', labelsize=12)
    a3n.tick_params(axis='both', labelsize=12)
    a1n.xaxis.grid(False)
    a2n.xaxis.grid(False)
    a3n.xaxis.grid(False)
    
    for iten in losttur_n:
        a2n.plot(iten[0], iten[1], '.', color='blue', alpha = 0.4)
    a2n.set_xlabel(r'nturns', fontsize=16)
    for idx,iten in enumerate(lostpos_n):
        a3n.plot(params.spos[iten[0]], iten[1], '.', color='blue', alpha = 0.4)
    a3n.set_xlabel(r'spos [m]', fontsize=16)
    

    for iten in inc_qlst_m:
        a1n.plot(iten[1], iten[0] , '.', color='red', alpha=0.3)
    a1n.set_xlabel(r'Number of electrons lost', fontsize=16)

    for iten in losttur_m:
        a2n.plot(iten[0], iten[1], '.', color='red', alpha=0.3)
    a2n.set_xlabel(r'nturns', fontsize=16)
    for iten in lostpos_m:
        a3n.plot(params.spos[iten[0]], iten[1], '.', color='red', alpha=0.3)
    a3n.set_xlabel(r'spos [m]', fontsize=16)
    
    _pyaccel.graphics.draw_lattice(params.fitm, height=h, offset=0, gca=True)

    _plt.tight_layout()
    _plt.show()
    
    return 
