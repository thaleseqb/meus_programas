"""."""
import sys as _sys

import numpy as _np
import matplotlib.pyplot as _plt

from pymodels import si as _si, bo as _bo
import pyaccel as _pa


class Params:
    """."""

    SCRAPER_H = 'SHVC'
    SCRAPER_V = 'SVVC'

    # create model
    def __init__(self):
        """."""
        fitm = _si.create_accelerator()
        fitm = _si.fitted_models.vertical_dispersion_and_coupling(fitm)
        fitm.vchamber_on = True
        fitm.radiation_on = True
        fitm.cavity_on = True
        self.model = fitm
        print(fitm)

        # scraper data
        self.scraper_indices_h = _pa.lattice.find_indices(
            fitm, 'fam_name', self.SCRAPER_H)
        self.scraper_indices_v = _pa.lattice.find_indices(
            fitm, 'fam_name', self.SCRAPER_V)

        # calc optics
        self.spos = _pa.lattice.find_spos(fitm, indices='closed')
        self.twiss, *_ = _pa.optics.calc_twiss(fitm, indices='closed')

        # equilibrium parameters
        booster = _bo.create_accelerator()
        booster.energy = 3e9
        booster[187].voltage = 1e6
        booster.radiation_on = True
        booster.cavity_on = True
        eqparams = _pa.optics.beam_envelope.EqParamsFromBeamEnvelope(booster)
        coup = 1/100
        emitt0 = eqparams.emit1  # nno-coupling model
        self.h_emitt = 1/(1 + coup) * emitt0
        self.v_emitt = coup/(1 + coup) * emitt0
        print('hemitt [pm.rad]:', self.h_emitt * 1e12)
        print('vemitt [pm.rad]:', self.v_emitt * 1e12)
        self.sigmae = eqparams.espread0
        self.bun_len = eqparams.bunlen

        # defining the index
        fam_name_dict = _pa.lattice.find_dict(fitm, 'fam_name')
        self.nlk_index = fam_name_dict['InjNLKckr'][0] + 1


class Machine_study(Params):
    """."""

    def __init__(
            self, nturns=10, coord_idx=0, coord_min=-0.02,
            coord_max=0.02, coord_nrpts=200):
        """."""
        super().__init__()

        self._nr_part = 10
        self.injection_position = _np.array([-0.008, 0, 0, 0, 0, 0])
        self._bunch = self._create_bunch()

        self.nturns = nturns
        self.coord_idx = coord_idx
        self.coord_max = coord_max
        self.coord_min = coord_min
        self.coord_nrpts = coord_nrpts

    @property
    def nr_part(self):
        """."""
        return self._nr_part

    @nr_part.setter
    def nr_part(self, new_particles):
        self._nr_part = new_particles
        self._bunch = self._create_bunch()

    def _create_bunch(self):
        bun = _pa.tracking.generate_bunch(
            n_part=self._nr_part, envelope=None, emit1=self.h_emitt,
            emit2=self.v_emitt, sigmae=self.sigmae, sigmas=self.bun_len,
            optics=self.twiss[self.nlk_index])
        bun += self.injection_position[:, None]
        return bun

    @property
    def bunch(self):
        """."""
        return self._bunch

    def simulate_scraper_effect(self, vchamber):
        """Vary increments.

        Args:
            vchamber (list): four positions of the blades.

        Returns:
            dict: results of the simulation.

        """
        incs = _np.linspace(self.coord_min, self.coord_max, self.coord_nrpts)

        # changed scraper width within model tracking simulation
        # after calling this function, vchamber's width will be changed
        chamb0 = self.get_vchamber_scraper()
        self.set_vchamber_scraper(vchamber)

        res = dict()
        res['scrap_chamb'] = _np.array(self.get_vchamber_scraper())
        res['increments'] = incs
        res['bunch_mean'] = []
        res['nr_plost'] = []
        res['turn_lost'] = []
        res['idx_lost'] = []
        for inc in incs:
            bun = self.bunch.copy()
            turn_lost, index_lost = self.track_mchn_stdy(bun, inc)
            res['turn_lost'].append(_np.array(turn_lost))
            res['idx_lost'].append(_np.array(index_lost))
            res['nr_plost'].append(turn_lost.size)
            res['bunch_mean'].append(_np.mean(bun[self.coord_idx]))
        res['nr_plost'] = _np.array(res['nr_plost'])
        res['bunch_mean'] = _np.array(res['bunch_mean'])

        # after calling this function, vchamber's height will be restored
        self.set_vchamber_scraper(chamb0)
        return res

    def set_vchamber_scraper(self, vchamber):
        """."""
        for iten in self.scraper_indices_h:
            self.model[iten].hmin = vchamber[0]
            self.model[iten].hmax = vchamber[1]
        for iten in self.scraper_indices_v:
            self.model[iten].vmin = vchamber[2]
            self.model[iten].vmax = vchamber[3]

    def get_vchamber_scraper(self):
        """."""
        chamb = []
        chamb.append(_pa.lattice.get_attribute(
            self.model, 'hmin', indices=self.scraper_indices_h)[0])
        chamb.append(_pa.lattice.get_attribute(
            self.model, 'hmax', indices=self.scraper_indices_h)[0])
        chamb.append(_pa.lattice.get_attribute(
            self.model, 'vmin', indices=self.scraper_indices_v)[0])
        chamb.append(_pa.lattice.get_attribute(
            self.model, 'vmax', indices=self.scraper_indices_v)[0])
        return chamb

    def track_mchn_stdy(self, bunch, increment):
        """."""
        # setting param to be the index of the array [x,x',y,y',delta,z]
        bunch[self.coord_idx] += increment
        parallel = self.nr_part > 50
        tracked = _pa.tracking.ring_pass(
            self.model, particles=bunch, nr_turns=self.nturns,
            turn_by_turn=False, element_offset=self.nlk_index,
            parallel=parallel, )

        turn_lost, index_lost = _np.array(tracked[2]), _np.array(tracked[3])
        lost_flag = _np.logical_not(
            (turn_lost == self.nturns) & (index_lost == self.nlk_index))
        return turn_lost[lost_flag], index_lost[lost_flag]

    # ----------- Plotting methods ---------------

    def plot_simulation_results(self, res):
        """."""
        units = {0: 1e3, 1: 1e3, 2: 1e3, 3: 1e3, 4: 1e2, 5: 1e3}
        ylabels = {
            0: r'horizontal position mean [mm]',
            1: r'$x^{\prime}$ mean [mrad]',
            2: r'vertical position mean [mm]',
            3: r'$y^{\prime}$ mean [mrad]',
            4: r'$\delta [%]$',
            5: r'$\delta l [mm]',
            }

        fig, (a1n, a2n, a3n) = _plt.subplots(
            1, 3, sharey=True, figsize=(10, 5))

        if isinstance(res, dict):
            res = [res, ]

        min_ = _sys.maxsize
        max_ = -_sys.maxsize
        for j, re_ in enumerate(res):
            cor = _plt.cm.jet(j/len(res))
            chmb = re_['scrap_chamb'] * 1e3
            lab = 'h=({:.1f}, {:.1f}) v=({:.1f}, {:.1f}) [mm]'.format(*chmb)

            idx_lost = re_['idx_lost']
            turn_lost = re_['turn_lost']
            bun_m = re_['bunch_mean']
            set_lab = False
            for i, (idx, trn) in enumerate(zip(idx_lost, turn_lost)):
                if not idx.size:
                    continue
                bmean = bun_m[i] * units[self.coord_idx]
                min_ = min(bmean, min_)
                max_ = max(bmean, max_)
                a2n.plot(trn, _np.full(trn.shape, bmean), '.', color=cor)
                a3n.plot(
                    self.spos[idx], _np.full(idx.shape, bmean), '.', color=cor)
                lin = a1n.plot(re_['nr_plost'][i], bmean, '.', color=cor)[0]
                if not set_lab:
                    lin.set_label(lab)
                    set_lab = True

        a1n.grid(True, alpha=0.5, ls='--', color='k', lw=1)
        a2n.grid(True, alpha=0.5, ls='--', color='k', lw=1)
        a3n.grid(True, alpha=0.5, ls='--', color='k', lw=1)

        a1n.tick_params(axis='both', labelsize=12)
        a2n.tick_params(axis='both', labelsize=12)
        a3n.tick_params(axis='both', labelsize=12)

        a1n.xaxis.grid(False)
        a2n.xaxis.grid(False)
        a3n.xaxis.grid(False)

        a1n.set_xlabel('# of Electrons Lost', fontsize=16)
        a2n.set_xlabel('Lost Turn', fontsize=16)
        a3n.set_xlabel('Lost Position [m]', fontsize=16)
        a1n.set_ylabel(ylabels[self.coord_idx], fontsize=16)

        a1n.legend(loc='best', fontsize=12)

        off = min_*0.8 if min_*max_ > 0 else 0.0
        hei = abs(max_-min_) * 0.15
        _pa.graphics.draw_lattice(self.model, height=hei, offset=off, gca=a3n)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        fig.show()

        return fig, (a1n, a2n, a3n)
