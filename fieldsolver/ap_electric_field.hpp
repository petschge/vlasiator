#pragma once
/*
 * This file is part of Vlasiator.
 *
 * AP (CSL-RME) field solver -- Liu, Cai, Cao, Lapenta (2025), J. Comput.
 * Phys. 528, 113840. Implements Eqs. (23), (35)-(41), (45): the
 * reformulated-Maxwell electric field solve (Eq. 38) replacing the old
 * explicit RK2/Ohm's-law propagateFields, keeping Faraday's law (Eq. 23) and
 * adding an optional Gauss's-law correction (Eq. 45) and optional low-pass
 * filtering.
 *
 * E, B, rho_s, J_s, mu and J-hat are all collocated in the fieldsolver. They
 * live at the centres of the Vlasov cells, exactly where f_s and its moments
 * are stored. The solver therefore keeps its own persistent "node-centred"
 * copy of E^k and B^k and never has to interpolate between the field solve and
 * the particle push.
 *
 * To couple with the rest of vlasiator he node-centred B^{k+theta} and
 * E^{k+theta} fields are written to the `vol` fields (PERB?VOL, E?VOL). For
 * output (that expects quantities on a Yee lattice) we interpolate
 * node-centered fields to produce face center B fields (perb at time level k+1
 * and perbdt2) and edge centered E fields (e at time k+1 and edt2). B_c sits
 * on the face between cell i-1 and i along c; E_c sits on the edge shared by
 * the 4 cells at offsets {0,-1} along the two axes other than c. For
 * initialization and restart we have to do to averaging to get node-centered
 * state from perb and e from the restart files.
 *
 * The solver is using second-order centered differences. The same centred curl
 * is used for Faraday (Eq. 23) and Ampere (curl B^k in Eq. 38), and curl-curl
 * is its composition, so that the discrete Poynting theorem holds exactly for
 * theta = 1/2 and div B is preserved exactly. The paper was using a
 * third-order finite-element discretisation instead.
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
 * \param Bnode        perturbed B, node-centered
 * \param bgb          background B, volume-averaged
 * \param theta        the AP splitting parameter
 * \param dt           current timestep.
 * \param outMu        output as 3x3 tensor per cell
 * \param outJhat      output as 3 vector per cell
 */
void ap_BuildSpeciesTensors(
   std::vector<fsgrids::speciesrhoqspan>& speciesRhoQ,
   std::vector<fsgrids::speciesjspan>& speciesJ,
   std::span<const std::array<Real,3>> Bnode,
   fsgrids::constbgbspan bgb,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real theta,
   Real dt,
   fsgrid::FsData<std::array<Real,9>>& outMu,
   fsgrid::FsData<std::array<Real,3>>& outJhat
);

/*! Assemble and solve Eq. (38) for E^{k+theta} via HYPRE IJ + BoomerAMG,
 *  then compute E^{k+1} (Eq. 40). */
bool ap_SolveElectricField(
   std::span<const std::array<Real,3>> Ek,
   std::span<const std::array<Real,3>> Bk,
   fsgrids::constbgbspan bgb,
   std::span<const std::array<Real,9>> mu,
   std::span<const std::array<Real,3>> Jhat,
   std::span<std::array<Real,3>> Etheta,
   std::span<std::array<Real,3>> Ekp1,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   SysBoundary& sysBoundaries,
   Real c,
   Real theta,
   Real dt
);

/*! Faraday update: B^{k+1} = B^k - dt curl(E^{k+theta})  (Eq. 23), then
 *  B^{k+theta} = theta B^{k+1} + (1-theta) B^k  (Eq. 39).  */
void ap_UpdateMagneticField(
   std::span<const std::array<Real,3>> Bk,
   std::span<const std::array<Real,3>> Etheta,
   std::span<std::array<Real,3>> Bkp1,
   std::span<std::array<Real,3>> Btheta,
   fsgrids::constbgbspan bgb,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   SysBoundary& sysBoundaries,
   Real theta,
   Real dt
);

/*! Gauss's-law (Boris) correction (Eq. 45 then Eq. 41). */
void ap_GaussLawCorrection(
   std::span<std::array<Real,3>> Ekp1,
   std::span<const std::array<Real,3>> Ek,
   std::span<const std::array<Real,1>> rho,
   std::span<const std::array<Real,9>> mu,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real dt
);

/*! Top-level AP entry point, called from propagateFields.
 *
 * dt == 0: one-time setup call. Initialises the node-centred state from
 * perb/e and publishes it to vol; no field is advanced and the Yee arrays are
 * left exactly as the project set them.
 * dt  > 0: advances one step following Algorithm 3.4 of the paper, step (2).
 */
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
                     SysBoundary& sysBoundaries,
                     creal& dt, cuint subcycles);
