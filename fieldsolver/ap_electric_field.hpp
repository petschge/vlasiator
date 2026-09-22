#pragma once
/*
 * This file is part of Vlasiator.
 *
 * AP (CSL-RME) field solver -- Liu, Cai, Cao, Lapenta (2025), J. Comput.
 * Phys. 528, 113840. Implements Eqs. (23), (35)-(41), (45): the
 * reformulated-Maxwell electric field solve (Eq. 38) replacing the old
 * explicit RK2/Ohm's-law propagateFields, keeping Faraday's law (Eq. 23)
 * and adding an optional Gauss's-law correction (Eq. 45).
 *
 *  - Periodic boundaries only for now
 *  - ehall/egradpe/egradpedt2/dmoments/dmomentsdt2/bgb's non-volume
 *    components are accepted but NOT used.
 *  - Only bgb's *volume* components (BGBXVOL/BGBYVOL/BGBZVOL) are read, to combine
 *    with perb's volume components into a cell-centered total B for the tensor
 *    build.
 */

#include "fs_common.h"
#include "derivatives.hpp"
#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_struct_ls.h>
#include <array>
#include <vector>

/*! Per-species rotation tensor alpha_s (Eq. 37) and its accumulation into
 *  mu (Eq. 36) and J-hat (Eq. 35), evaluated at cell centers
 *
 * \param speciesRhoQ  rho_s^k per population
 * \param speciesJ     J_s^{k*} per population
 * \param perb         perturbed B, face-centered/direct (PERBX/Y/Z).
 * \param bgb          background B, face-centered/direct (BGBX/Y/Z).
 * \param theta        the AP splitting parameter
 * \param dt           current timestep.
 * \param outMu        output as 3x3 tensor per cell
 * \param outJhat      output as 3 vector per cell
 */
void ap_BuildSpeciesTensors(
   std::vector<fsgrids::speciesrhoqspan>& speciesRhoQ,
   std::vector<fsgrids::speciesjspan>& speciesJ,
   fsgrids::constperbspan perb,
   fsgrids::constbgbspan bgb,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real theta,
   Real dt,
   std::vector<std::array<Real,9>>& outMu,
   std::vector<std::array<Real,3>>& outJhat
);

/*! Assemble and solve Eq. (38) for E^{k+theta} via HYPRE IJ + BoomerAMG,
 *  then compute E^{k+1} (Eq. 40) and write both into e/edt2. */
bool ap_SolveElectricField(
   fsgrids::efieldspan e,
   fsgrids::efieldspan edt2,
   fsgrids::constperbspan perb,
   fsgrids::constbgbspan bgb,
   fsgrids::constdperbspan dperb,
   const std::vector<std::array<Real,9>>& mu,
   const std::vector<std::array<Real,3>>& Jhat,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real c,
   Real theta,
   Real dt
);

/*! Faraday update: B^{k+1} = B^k - dt curl(E^{k+theta})  (Eq. 23), then
 *  B^{k+theta} = theta B^{k+1} + (1-theta) B^k  (Eq. 39).  */
void ap_UpdateMagneticField(
   fsgrids::perbspan perb,
   fsgrids::perbspan perbdt2,
   fsgrids::constefieldspan e,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real theta,
   Real dt
);

/*! Gauss's-law (Boris) correction (Eq. 45 then Eq. 41).  */
void ap_GaussLawCorrection(
   fsgrids::efieldspan e,
   fsgrids::efieldspan edt2,
   fsgrids::constefieldspan eOld,
   fsgrids::momentsspan moments,
   const std::vector<std::array<Real,9>>& mu,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real dt
);

/*! Top-level AP entry point, called from propagateFields */
bool ap_propagateFields(fsgrids::perbspan perb,
                     fsgrids::perbspan perbdt2,
                     fsgrids::efieldspan e,
                     fsgrids::efieldspan edt2,
                     fsgrids::momentsspan moments,
                     std::vector<fsgrids::speciesrhoqspan>& speciesRhoQ,
                     std::vector<fsgrids::speciesjspan>& speciesJ,
                     fsgrids::dperbspan dperb,
                     fsgrids::dmomentsspan dmoments,
                     fsgrids::bgbspan bgb,
                     fsgrids::volspan vol,
                     fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                     creal& dt, cuint subcycles);
