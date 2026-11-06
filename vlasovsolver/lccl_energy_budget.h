#pragma once
/*
 * This file is part of Vlasiator.
 *
 * Energy accounting of the LCCL (CSL-RME) cycle (fieldsolver.energyAccounting). Header only, so
 * that the main loop, the Vlasov acceleration and the field solver share one instance without a
 * new object file.
 *
 * Per step and species s, with K_s the kinetic energy sum_cells 1/2 m_s sum_v f v^2 dv^3 dV over
 * the interior (NOT_SYSBOUNDARY) cells:
 *   dKacc_s   K_s after minus before the acceleration;
 *   Wmap_s    the change an exact remap would give: the acceleration applies to every velocity the
 *             affine map v -> A v + b of its subcycles (rotation about B, kick by E), so
 *             Wmap = 1/2 m N (2 b.(A u) + |b|^2) with N the number of particles and u their mean
 *             velocity before the acceleration; dKacc - Wmap is the error of the remap and of the
 *             sparse velocity mesh during the acceleration;
 *   dKtrans_s the rest of the change over the step: both translations, the adjustment of the
 *             sparse mesh after them and the boundary conditions (particles leaving or entering).
 * Field side, per step, over the interior cells, W = sum eps0 |E|^2/2 + |B0 + dB|^2/(2 mu0) dV:
 *   Ppred     dt sum J^{k+theta}.E^{k+theta} dV with the current the solve predicts,
 *             J^{k+theta} = J-hat + theta dt mu E^{k+theta} (Eq. 37), and the raw E^{k+theta};
 *   Dtheta    (theta - 1/2) sum [eps0 |E^{k+1} - E^k|^2 + |B^{k+1} - B^k|^2/mu0] dV of the raw
 *             fields: the energy the theta scheme removes (zero for theta = 1/2);
 *   dWcorr    W after minus before the Gauss correction and the low-pass filter of B;
 *   RF        W_raw - W^k + Ppred + Dtheta: zero for the discrete Poynting theorem of the shared
 *             centred curl in a periodic domain without the E filter; otherwise the Poynting flux
 *             through the boundaries and the effect of the E filter on the Faraday step.
 * Then, over a step, the change of the total energy W + sum_s K_s is exactly
 *   (RF - Dtheta + dWcorr) + (sum_s Wmap_s - Ppred) + sum_s (dKacc_s - Wmap_s) + sum_s dKtrans_s,
 * where sum_s Wmap_s - Ppred is the particle-field coupling: the work the push does on the actual
 * distribution minus the work the field solve assumed (Ji et al. 2023).
 * All sums are cumulative since the start of this run (a restart starts them again at zero).
 */
#include <vector>

struct LcclEnergyBudget {
   bool active = false;   // fieldsolver.energyAccounting with the LCCL solver
   bool haveK = false;    // K holds the kinetic energies at the start of the current step
   std::vector<double> K;                      // per species, at the start of the current step [J]
   std::vector<double> dKacc, Wmap, dKtrans;   // per species, cumulative [J]
   std::vector<double> stepKaccBefore, stepKaccAfter;  // per species, current step [J]
   double Ppred = 0.0, Dtheta = 0.0, dWcorr = 0.0, RF = 0.0;  // field side, cumulative [J]

   void resize(size_t nSpecies) {
      K.assign(nSpecies, 0.0);
      dKacc.assign(nSpecies, 0.0);
      Wmap.assign(nSpecies, 0.0);
      dKtrans.assign(nSpecies, 0.0);
      stepKaccBefore.assign(nSpecies, 0.0);
      stepKaccAfter.assign(nSpecies, 0.0);
   }
};

inline LcclEnergyBudget& lcclEnergyBudget() {
   static LcclEnergyBudget budget;
   return budget;
}
