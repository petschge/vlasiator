/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <set>

#include "../../common.h"
#include "../../parameters.h"
#include "../../readparameters.h"
#include "../../backgroundfield/backgroundfield.h"
#include "../../object_wrapper.h"

#include "Harris.h"

using namespace std;
using namespace spatial_cell;

namespace projects {
   Harris::Harris(): TriAxisSearch() { }
   Harris::~Harris() { }
   
   bool Harris::initialize(void) {return Project::initialize();}
   
   void Harris::addParameters(){
      typedef Readparameters RP;
      RP::add<Real>("Harris.Scale_size", "Harris sheet scale size (m)", this->SCA_LAMBDA, 150000.0);
      RP::add<Real>("Harris.BX0", "Magnetic field at infinity (T)", this->BX0, 8.33061003094e-8);
      RP::add<Real>("Harris.BY0", "Magnetic field at infinity (T)", this->BY0, 8.33061003094e-8);
      RP::add<Real>("Harris.BZ0", "Magnetic field at infinity (T)", this->BZ0, 8.33061003094e-8);
      RP::add<Real>("Harris.perturbationAmplitude", "Amplitude phi0 of an optional magnetic island perturbation "
              "delta_B = grad(phi) x e_z, with phi(x,y) = phi0*cos(pi*(x-xmid)/Lx)*cos(pi*(y-ymid)/(2*Ly)) "
              "over the full domain extent (Lx, Ly are half the domain's x and y extent, centred on its "
              "midpoint) -- or, if Harris.doubleSheet=1, phi(x,y) = -phi0*cos(pi*(x-xmid)/Lx)*sin(pi*(y-ymid)/L) "
              "(L = Ly, the domain's y half-extent). Units T*m. Default 0 (no perturbation), either way.", this->PERT_AMPLITUDE, 0.0);
      RP::add<bool>("Harris.doubleSheet", "If 1, replace the single Harris sheet at y=0 with TWO sheets of "
              "opposite polarity at y=+-L/2 (L = domain half-height = 0.5*(ymax-ymin)), fully periodic in "
              "y -- no wall boundary needed. Each sheet uses the same Scale_size and the same per-species "
              "Temperature/rho/rhoBackground as the single-sheet case; sheet 1 (y=-L/2) drifts at the "
              "same sign as the single-sheet formula, sheet 2 (y=+L/2, opposite polarity) at the opposite "
              "sign -- see getPopulationsAt() and setProjectBField(). Default false (original single sheet).", this->DOUBLE_SHEET, false);
      RP::add<int>("Harris.currentLoading", "How the sheet drifts are loaded. 0: analytic Harris drifts. "
              "1: each sheet's drifts scaled so the loaded current equals the LCCL solver's centred discrete "
              "curl of the initial B/mu0 (removes the step-1 current impulse at the sheets). 2: as 1, and "
              "also load the in-sheet share n_sheet/n_total of the perturbation's discrete current "
              "(doubleSheet only; needs a background population). Default 0.", this->CURRENT_LOADING, 0);
      RP::add<std::string>("Harris.perturbationType", "Perturbation that starts reconnection. gem: the island of "
              "Harris.perturbationAmplitude (Sec. 5.4 of Liu et al. 2025). localized: instead a localized X-point "
              "at each sheet's X-line as in iPic3D's GEM setup (Lapenta et al. 2010, Phys. Plasmas 17, 082106), "
              "delta_B = grad(psi) x e_z with psi = s*A*BX0*w*exp(-(X^2+Y^2)/w^2)*cos(pi*X/(10w))*cos(pi*Y/(10w)) "
              "for |X|,|Y| < 5w and 0 outside (where it is below 1e-10 of its peak). X, Y: from the X-line; "
              "A = Harris.localizedAmplitude, w = Harris.localizedWidth*Scale_size; s = +1 for the single sheet "
              "and sheet 1, -1 for sheet 2. Its peak flux A*BX0*w is 0.2*B0*d_i with the defaults, the same as "
              "the GEM island's 2*phi0. Default gem.", this->PERTURBATION_TYPE, "gem");
      RP::add<Real>("Harris.localizedAmplitude", "Amplitude A of the localized perturbation, in units of BX0*w. "
              "Default 0.4 (iPic3D).", this->LOCAL_AMPLITUDE, 0.4);
      RP::add<Real>("Harris.localizedWidth", "Width w of the localized perturbation, in units of Scale_size. "
              "Default 1 (iPic3D: the sheet half-thickness).", this->LOCAL_WIDTH, 1.0);
      RP::add<Real>("Harris.symmetrySeed", "Amplitude, relative to the perturbation's flux (perturbationAmplitude, or "
              "localizedAmplitude*BX0*w if localized), of a localized term "
              "phi_s = seed*phi0*(X/w)*exp(-(X^2+(Y-Y_sheet)^2)/w^2) added at each X-line (X = x-xmid). It is "
              "odd in x, so it breaks the x -> -x mirror symmetry that otherwise pins a secondary island "
              "at x=0; with doubleSheet, sheet 2 gets the rotation-by-pi image of sheet 1's term, so the "
              "two sheets stay mirror images of each other. Default 0 (off).", this->SEED_AMPLITUDE, 0.0);
      RP::add<Real>("Harris.symmetrySeedWidth", "Width w of the symmetry seed, in units of Scale_size. "
              "Default 2.", this->SEED_WIDTH, 2.0);

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
         const std::string& pop = getObjectWrapper().particleSpecies[i].name;
         HarrisSpeciesParameters* sP=new HarrisSpeciesParameters();
         speciesParamsRead.push_back(sP);

         RP::add<Real>(pop + "_Harris.Temperature", "Temperature (K)", sP->TEMPERATURE, 2.0e6);
         RP::add<Real>(pop + "_Harris.rho", "Number density at infinity (m^-3)", sP->DENSITY, 1.0e7);
         RP::add<Real>(pop + "_Harris.rhoBackground", "Background (non-drifting) number density added to the "
                 "Harris-sheet population, same temperature (m^-3). Default 0.", sP->RHO_BACKGROUND, 0.0);
      }
   }
   
   void Harris::getParameters(){
      Project::getParameters();
      if (this->PERTURBATION_TYPE != "gem" && this->PERTURBATION_TYPE != "localized") {
         std::cerr << "Harris.perturbationType must be gem or localized, not " << this->PERTURBATION_TYPE << std::endl;
         abort();
      }
      this->LOCALIZED_PERTURBATION = (this->PERTURBATION_TYPE == "localized");
      this->LOCAL_WIDTH *= this->SCA_LAMBDA;
      if (this->LOCALIZED_PERTURBATION) {
         if (this->LOCAL_WIDTH <= 0.0) {
            std::cerr << "Harris.localizedWidth must be positive" << std::endl;
            abort();
         }
         if (this->PERT_AMPLITUDE != 0.0) {
            // the localized perturbation replaces the island, including its current for currentLoading=2
            std::cerr << "Harris.perturbationType=localized: Harris.perturbationAmplitude is ignored" << std::endl;
            this->PERT_AMPLITUDE = 0.0;
         }
      }
      this->SEED_REF = this->LOCALIZED_PERTURBATION ? this->LOCAL_AMPLITUDE * this->BX0 * this->LOCAL_WIDTH
                                                    : this->PERT_AMPLITUDE;

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
         HarrisSpeciesParameters sP = *this->speciesParamsRead.at(i);

         // Drift velocity from Ampere's law applied to the Harris equilibrium B(y) = BX0*tanh(y/lambda0)
         // (Eq. 55 of Liu et al. 2025, JCP 528, 113840): v_ds = -2*m_s*v_ts^2/(q_s*B0*lambda0). Substituting
         // the standard thermal speed v_ts^2 = k_B*T_s/m_s (matching the convention used by
         // MaxwellianPhaseSpaceDensity) cancels m_s, giving a mass-independent result in T_s and the signed
         // charge q_s -- which is why electrons and ions drift in opposite directions without any
         // species-specific sign logic here. This assumes the standard GEM geometry (field varying as BX0
         // along y, as set by setProjectBField when BY0=BZ0=0); if BX0 is zero the drift is left at zero
         // rather than dividing by zero, since there is then no Harris current sheet to balance.
         const Real charge = getObjectWrapper().particleSpecies[i].charge;
         if (this->BX0 != 0.0 && charge != 0.0) {
            sP.DRIFT_V = -2.0 * physicalconstants::K_B * sP.TEMPERATURE / (charge * this->BX0 * this->SCA_LAMBDA);
         } else {
            sP.DRIFT_V = 0.0;
         }

         speciesParams.push_back(sP);
      }

      this->SEED_WIDTH *= this->SCA_LAMBDA;
      if (this->SEED_AMPLITUDE != 0.0 && this->SEED_WIDTH <= 0.0) {
         std::cerr << "Harris.symmetrySeedWidth must be positive" << std::endl;
         abort();
      }
      if (this->CURRENT_LOADING < 0 || this->CURRENT_LOADING > 2) {
         std::cerr << "Harris.currentLoading must be 0, 1 or 2" << std::endl;
         abort();
      }
      if (this->CURRENT_LOADING == 2 && !this->DOUBLE_SHEET) {
         std::cerr << "Harris.currentLoading=2 is implemented for doubleSheet only; using 1" << std::endl;
         this->CURRENT_LOADING = 1;
      }
      // Species sums for currentLoading=2: sheet current per unit sech^2 and the densities that set
      // each sheet's share of the perturbation current.
      this->SHEET_CURRENT = 0.0;
      this->DENSITY_SUM = 0.0;
      this->BACKGROUND_SUM = 0.0;
      for (uint i = 0; i < speciesParams.size(); i++) {
         this->SHEET_CURRENT += getObjectWrapper().particleSpecies[i].charge * speciesParams[i].DENSITY
                                * speciesParams[i].DRIFT_V;
         this->DENSITY_SUM += speciesParams[i].DENSITY;
         this->BACKGROUND_SUM += speciesParams[i].RHO_BACKGROUND;
      }
   }

   /*! Ratio of the LCCL solver's centred difference of tanh(u/lambda0) (node Bx is exact at the cell
    * centre here, the faces sharing its y) to the analytic derivative, at offset u from a sheet:
    * [tanh(a+d)-tanh(a-d)]/(2d sech^2 a) = (sinh(2d)/2d)/(1 + sinh^2(d) sech^2(a)), a=u/lambda0,
    * d=dy/lambda0. Written this way it stays exact where tanh(a+-d) both round to +-1.
    */
   Real Harris::discreteCurlFactor(Real u) const {
      const Real d = Parameters::dy_ini / this->SCA_LAMBDA;
      const Real sech = 1.0 / cosh(u / this->SCA_LAMBDA);
      return (sinh(2.0 * d) / (2.0 * d)) / (1.0 + sinh(d) * sinh(d) * sech * sech);
   }

   /*! (curl dB)_z/mu0 of the doubleSheet island perturbation as the LCCL solver sees it: face values
    * from setProjectBField(), node values as the mean of each cell's two faces, centred curl.
    * Closed form for phi = -phi0*cos(a X)*sin(b Y), a=pi/Lx, b=pi/Ly; x, y are cell centres.
    */
   Real Harris::perturbationDiscreteCurrent(Real x, Real y) const {
      const Real Lx = 0.5 * (Parameters::xmax - Parameters::xmin);
      const Real Ly = 0.5 * (Parameters::ymax - Parameters::ymin);
      const Real X = x - 0.5 * (Parameters::xmax + Parameters::xmin);
      const Real Y = y - 0.5 * (Parameters::ymax + Parameters::ymin);
      const Real dx = Parameters::dx_ini, dy = Parameters::dy_ini;
      const Real a = M_PI / Lx, b = M_PI / Ly;
      const Real k2 = a * (sin(a * dx) / dx) * cos(0.5 * b * dy) + b * (sin(b * dy) / dy) * cos(0.5 * a * dx);
      return -this->PERT_AMPLITUDE * cos(a * X) * sin(b * Y) * k2 / physicalconstants::MU_0;
   }

   /*! See the HarrisPopDesc comment in Harris.h. x, y are the spatial cell centre, in the same
    * convention each caller already used (cell->parameters[CellParams::YCRD] + 0.5*DY etc., or the
    * x, y passed into getV0()).
    */
   std::vector<HarrisPopDesc> Harris::getPopulationsAt(Real x, Real y, const uint popID) const {
      const HarrisSpeciesParameters& sP = speciesParams[popID];
      std::vector<HarrisPopDesc> pops;
      if (this->DOUBLE_SHEET) {
         // Two sheets of opposite polarity at y=ymid+-L/2, L=domain half-height=Ly. Near either sheet
         // the OTHER sheet's tanh term is saturated (+-1, for L >> lambda0), so locally this reduces to
         // the same single-sheet Bx(y)=+-B0*tanh((y-y_sheet)/lambda0) and n(y)=n0*sech^2((y-y_sheet)/
         // lambda0) as the single-sheet case (Eq. 52-53 of Liu et al. 2025), just recentred -- see
         // setProjectBField() for the corresponding B-field formula and its own derivation comment.
         // Current (hence drift) sign flips with the local sign of dBx/dy, which is positive at
         // y=-L/2 (same as the original single-sheet profile) and negative at y=+L/2 (opposite) -- so
         // sheet 1 uses +DRIFT_V (as computed above, unchanged) and sheet 2 uses -DRIFT_V.
         const Real Lhalf = 0.5 * (Parameters::ymax - Parameters::ymin);
         const Real ymid  = 0.5 * (Parameters::ymax + Parameters::ymin);
         const Real yc = y - ymid;
         const Real y1 = yc + 0.5*Lhalf; // offset from sheet 1 at yc=-Lhalf/2
         const Real y2 = yc - 0.5*Lhalf; // offset from sheet 2 at yc=+Lhalf/2
         const Real sech1 = 1.0 / cosh(y1/this->SCA_LAMBDA);
         const Real sech2 = 1.0 / cosh(y2/this->SCA_LAMBDA);
         const Real rho1 = sP.DENSITY * sech1 * sech1;
         const Real rho2 = sP.DENSITY * sech2 * sech2;
         // Same factor for every species keeps the electron/ion share of each sheet's current.
         Real f1 = 1.0, f2 = 1.0;
         if (this->CURRENT_LOADING >= 1) {
            f1 = discreteCurlFactor(y1);
            f2 = discreteCurlFactor(y2);
         }
         if (this->CURRENT_LOADING == 2 && this->PERT_AMPLITUDE != 0.0 && this->SHEET_CURRENT != 0.0) {
            // Sheet k carries n_k/n_total of the perturbation current dJ: eps_k*f_k*J_k = (n_k/n_tot)*dJ
            // with J_k = s_k*SHEET_CURRENT*sech^2 (s_1=+1, s_2=-1), so eps_k stays bounded where the
            // sheet density vanishes; the rest of dJ (mostly in the lobes) is left to the field solver.
            const Real nTot = this->DENSITY_SUM * (sech1 * sech1 + sech2 * sech2) + this->BACKGROUND_SUM;
            if (nTot > 0.0) {
               const Real g = this->DENSITY_SUM * perturbationDiscreteCurrent(x, y) / (nTot * this->SHEET_CURRENT);
               f1 += g;
               f2 -= g;
            }
         }
         pops.push_back({ f1*sP.DRIFT_V, rho1, sP.TEMPERATURE});
         pops.push_back({-f2*sP.DRIFT_V, rho2, sP.TEMPERATURE});
      } else {
         const Real sech = 1.0 / cosh(y/this->SCA_LAMBDA);
         const Real f = (this->CURRENT_LOADING >= 1) ? discreteCurlFactor(y) : 1.0;
         pops.push_back({f*sP.DRIFT_V, sP.DENSITY * sech * sech, sP.TEMPERATURE});
      }
      if (sP.RHO_BACKGROUND > 0.0) {
         pops.push_back({0.0, sP.RHO_BACKGROUND, sP.TEMPERATURE});
      }
      return pops;
   }

   Realf Harris::fillPhaseSpace(spatial_cell::SpatialCell *cell,
                                       const uint popID,
                                       const uint nRequested
      ) const {
      // Fetch spatial cell center coordinates
      const Real x  = cell->parameters[CellParams::XCRD] + 0.5*cell->parameters[CellParams::DX];
      const Real y  = cell->parameters[CellParams::YCRD] + 0.5*cell->parameters[CellParams::DY];
      const Real mass = getObjectWrapper().particleSpecies[popID].mass;

      // Computed here on the host (getPopulationsAt() -- see Harris.h for why this is shared with
      // probePhaseSpace/getV0/findBlocksToInitialize rather than duplicated), then copied into a small
      // FIXED-SIZE array: the ARCH_LOOP_LAMBDA/ARCH_INNER_BODY kernel below must be capturable for GPU
      // execution, and a std::vector (heap-allocated) cannot be. At most 3 terms (sheet[+sheet2]+
      // background), per getPopulationsAt()'s own return size.
      const std::vector<HarrisPopDesc> popsVec = getPopulationsAt(x, y, popID);
      constexpr uint MAX_HARRIS_POPS = 3;
      Real popV0z[MAX_HARRIS_POPS] = {0,0,0}, popRho[MAX_HARRIS_POPS] = {0,0,0}, popT[MAX_HARRIS_POPS] = {0,0,0};
      const uint nPops = popsVec.size();
      for (uint p = 0; p < nPops && p < MAX_HARRIS_POPS; ++p) {
         popV0z[p] = popsVec[p].v0z; popRho[p] = popsVec[p].rho; popT[p] = popsVec[p].T;
      }

      #ifdef USE_GPU
      vmesh::VelocityMesh *vmesh = cell->dev_get_velocity_mesh(popID);
      vmesh::VelocityBlockContainer* VBC = cell->dev_get_velocity_blocks(popID);
      #else
      vmesh::VelocityMesh *vmesh = cell->get_velocity_mesh(popID);
      vmesh::VelocityBlockContainer* VBC = cell->get_velocity_blocks(popID);
      #endif
      // Loop over blocks
      Realf rhosum = 0;
      arch::parallel_reduce<arch::null>(
         {WID, WID, WID, nRequested},
         ARCH_LOOP_LAMBDA (const uint i, const uint j, const uint k, const uint initIndex, Realf *lsum ) {
            vmesh::GlobalID *GIDlist = vmesh->getGrid()->data();
            Realf* bufferData = VBC->getData();
            const vmesh::GlobalID blockGID = GIDlist[initIndex];
            // Calculate parameters for new block
            Real blockCoords[6];
            vmesh->getBlockInfo(blockGID,&blockCoords[0]);
            creal vxBlock = blockCoords[0];
            creal vyBlock = blockCoords[1];
            creal vzBlock = blockCoords[2];
            creal dvxCell = blockCoords[3];
            creal dvyCell = blockCoords[4];
            creal dvzCell = blockCoords[5];
            ARCH_INNER_BODY(i, j, k, initIndex, lsum) {
               creal vx = vxBlock + (i+0.5)*dvxCell;
               creal vy = vyBlock + (j+0.5)*dvyCell;
               creal vz = vzBlock + (k+0.5)*dvzCell;
               Realf value = 0;
               for (uint p = 0; p < nPops; ++p) {
                  value += MaxwellianPhaseSpaceDensity(vx, vy, vz - popV0z[p], popT[p], popRho[p], mass);
               }
               bufferData[initIndex*WID3 + k*WID2 + j*WID + i] = value;
               //lsum[0] += value;
            };
         }, rhosum);
      return rhosum;
   }

   /* Evaluates local SpatialCell properties for the project and population,
      then evaluates the phase-space density at the given coordinates.
      Used as a probe for projectTriAxisSearch.
   */
   Realf Harris::probePhaseSpace(spatial_cell::SpatialCell *cell,
                                        const uint popID,
                                        Real vx_in, Real vy_in, Real vz_in
      ) const {
      // Fetch spatial cell center coordinates
      const Real x  = cell->parameters[CellParams::XCRD] + 0.5*cell->parameters[CellParams::DX];
      const Real y  = cell->parameters[CellParams::YCRD] + 0.5*cell->parameters[CellParams::DY];
      const Real mass = getObjectWrapper().particleSpecies[popID].mass;

      Realf value = 0;
      for (const auto& pop : getPopulationsAt(x, y, popID)) {
         value += MaxwellianPhaseSpaceDensity(vx_in, vy_in, vz_in - pop.v0z, pop.T, pop.rho, mass);
      }
      return value;
   }

   void Harris::calcCellParameters(spatial_cell::SpatialCell* cell,creal& t) { }

   vector<std::array<Real, 3>> Harris::getV0(
      creal x,
      creal y,
      creal z,
      const uint popID
   ) const {
      vector<std::array<Real, 3>> V0;
      for (const auto& pop : getPopulationsAt(x, y, popID)) {
         V0.push_back({{0.0, 0.0, pop.v0z}});
      }
      return V0;
   }

   /*! Overrides TriAxisSearch's generic findBlocksToInitialize.
    *
    * The generic version walks outward from each getV0() peak along one axis at a time,
    * probing the SUMMED (sheet+background) phase-space density and stopping once it drops
    * below tolerance*minValue, on the assumption that this summed value decays monotonically
    * away from the peak being searched. That assumption breaks for two populations with
    * nearby peaks and very different local weight: far from the current sheet (y away from
    * 0), the sheet's own density has decayed to a tiny fraction of its peak, so the walk away
    * from the sheet's peak (at v_ds) toward v=0 runs straight into the *background's* own,
    * much larger peak there, which keeps the summed probe value above threshold all the way
    * past it -- so the search never terminates early and instead sweeps out nearly the whole
    * velocity-space domain, isotropically (since the resulting radius is shared across all
    * three axes), at every spatial cell where the background dominates. Confirmed numerically
    * against an actual failing cell before writing this fix.
    *
    * Fix: compute each population's own extent analytically and independently, rather than by
    * probing the summed distribution. Every term getPopulationsAt() can return (sheet,
    * [sheet2 for doubleSheet,] background) is an isotropic Maxwellian (MaxwellianPhaseSpaceDensity's
    * own definition), so the radius at which population p's OWN density alone drops to
    * tolerance*minValue has a closed form:
    *   f_p(r) = f_p,peak * exp(-r^2 / (2*v_ts^2)) = tolerance*minValue
    *   => r^2 = 2*v_ts^2 * ln(f_p,peak / (tolerance*minValue))   [only if f_p,peak exceeds it]
    * This exactly reproduces what the generic per-axis search would find for any one population
    * considered alone (verified by construction: the same single-axis slice through an
    * isotropic Gaussian gives the same threshold-crossing distance), while never letting one
    * population's own search be influenced by where another one's peak happens to sit -- this
    * argument is pairwise and so holds for however many terms getPopulationsAt() returns (2 for
    * the original single-sheet case, up to 3 for doubleSheet), not just exactly two.
    *
    * The block-inclusion test below (distance from this population's own v0 to each block's
    * centre, compared against the resulting radius) otherwise follows the same structure as
    * TriAxisSearch::findBlocksToInitialize.
    */
   uint Harris::findBlocksToInitialize(spatial_cell::SpatialCell* cell, const uint popID) const {
      vmesh::VelocityMesh *vmesh = cell->get_velocity_mesh(popID);
      vmesh::GlobalID *GIDbuffer;
      const vmesh::LocalID* vblocks_ini = cell->get_velocity_grid_length(popID);
      const uint blocksCount = vblocks_ini[0]*vblocks_ini[1]*vblocks_ini[2];
      #ifdef USE_GPU
      CHK_ERR( gpuMallocHost((void**)&GIDbuffer,blocksCount*sizeof(vmesh::GlobalID)) );
      #endif

      std::set<vmesh::GlobalID> singleSet;

      creal minValue = cell->getVelocityBlockMinValue(popID);
      uint buffer = 2;
      if (WID > 4 && blocksCount > 8) {
         buffer = 1;
      }
      creal tolerance = 0.1;

      creal x = cell->parameters[CellParams::XCRD];
      creal dx = cell->parameters[CellParams::DX];
      creal y = cell->parameters[CellParams::YCRD];
      creal dy = cell->parameters[CellParams::DY];
      creal dvxBlock = cell->get_velocity_grid_block_size(popID)[0];
      creal dvyBlock = cell->get_velocity_grid_block_size(popID)[1];
      creal dvzBlock = cell->get_velocity_grid_block_size(popID)[2];

      const size_t vxblocks_ini = cell->get_velocity_grid_length(popID)[0];
      const size_t vyblocks_ini = cell->get_velocity_grid_length(popID)[1];
      const size_t vzblocks_ini = cell->get_velocity_grid_length(popID)[2];

      const Real mass = getObjectWrapper().particleSpecies[popID].mass;
      creal xc = x + 0.5*dx;
      creal yc = y + 0.5*dy;
      // getPopulationsAt() (see Harris.h) returns up to 3 terms now (sheet1+sheet2+background for
      // doubleSheet, vs the 2 this override originally hard-coded for sheet+background) -- everything
      // below this point already iterates generically over however many it gets, so doubleSheet needs
      // no further changes here.
      const std::vector<HarrisPopDesc> pops = getPopulationsAt(xc, yc, popID);
      const bool singlePeak = (pops.size() == 1);

      vmesh::LocalID LID = 0;
      for (const auto& pop : pops) {
         const Realf fpeak = MaxwellianPhaseSpaceDensity(0.0, 0.0, 0.0, pop.T, pop.rho, mass);
         Real radius = 0.0; // m/s; stays 0 if this population is below threshold everywhere
         if (fpeak > tolerance*minValue) {
            const Real vts2 = physicalconstants::K_B * pop.T / mass;
            radius = std::sqrt(2.0 * vts2 * std::log(fpeak / (tolerance*minValue)));
         }
         const Real radiusX = (std::floor(radius/dvxBlock) + 1 + buffer) * dvxBlock;
         const Real radiusY = (std::floor(radius/dvyBlock) + 1 + buffer) * dvyBlock;
         const Real radiusZ = (std::floor(radius/dvzBlock) + 1 + buffer) * dvzBlock;
         const Real vRadiusSquared = std::max({radiusX*radiusX, radiusY*radiusY, radiusZ*radiusZ});
         const uint counterX = (uint)(radiusX/dvxBlock);
         const uint counterY = (uint)(radiusY/dvyBlock);
         const uint counterZ = (uint)(radiusZ/dvzBlock);

         #ifndef USE_GPU // non-GPU mesh resizing
         vmesh::LocalID currentMaxSize = LID + 5*counterX*counterY*counterZ;
         vmesh->setNewSize(currentMaxSize);
         GIDbuffer = vmesh->getGrid()->data();
         #endif

         Real V_crds[3];
         for (uint kv=0; kv<vzblocks_ini; ++kv) {
            for (uint jv=0; jv<vyblocks_ini; ++jv) {
               for (uint iv=0; iv<vxblocks_ini; ++iv) {
                  const vmesh::GlobalID GID = vmesh->getGlobalID(iv,jv,kv);
                  vmesh->getBlockCoordinates(GID,V_crds); // block's minimum corner

                  // v0x=v0y=0 always for this project (only the z-directed Harris drift is nonzero).
                  const Real cx = V_crds[0] + 0.5*dvxBlock;
                  const Real cy = V_crds[1] + 0.5*dvyBlock;
                  const Real cz = V_crds[2] + 0.5*dvzBlock - pop.v0z;
                  const Real R2 = cx*cx + cy*cy + cz*cz;

                  #ifndef USE_GPU // non-GPU mesh resizing
                  if (LID >= currentMaxSize) {
                     currentMaxSize = LID + counterX*counterY*counterZ;
                     vmesh->setNewSize(currentMaxSize);
                     GIDbuffer = vmesh->getGrid()->data();
                  }
                  #endif
                  if (singlePeak) {
                     if (R2 < vRadiusSquared) {
                        GIDbuffer[LID] = GID;
                        LID++;
                     }
                  } else {
                     if (R2 < vRadiusSquared && singleSet.count(GID)==0) {
                        singleSet.insert(GID);
                        GIDbuffer[LID] = GID;
                        LID++;
                     }
                  }
               } // vxblocks_ini
            } // vyblocks_ini
         } // vzblocks_ini
      } // iteration over populations

      cell->get_population(popID).N_blocks = LID;

      #ifdef USE_GPU
      cell->dev_resize_vmesh(popID,LID);
      vmesh::GlobalID *GIDtarget = vmesh->getGrid()->data();
      gpuStream_t stream = gpu_getStream();
      CHK_ERR( gpuMemcpyAsync(GIDtarget, GIDbuffer, LID*sizeof(vmesh::GlobalID), gpuMemcpyHostToDevice, stream));
      CHK_ERR( gpuStreamSynchronize(stream) );
      CHK_ERR( gpuFreeHost(GIDbuffer));
      #else
      vmesh->setNewSize(LID);
      #endif

      return LID;
   }

   void Harris::setProjectBField(fsgrids::perbspan perb,
                                 fsgrids::bgbspan bgb,
                                 fsgrids::technicalspan technical, FieldSolverGrid &fsgrid) {
      setBackgroundFieldToZero(fsgrid, technical, bgb);

      if(!P::isRestart) {
         // local copies for lambda capture
         const auto BX0_l = this->BX0;
         const auto BY0_l = this->BY0;
         const auto BZ0_l = this->BZ0;
         const auto SCA_LAMBDA_l = this->SCA_LAMBDA;
         const auto PHI0_l = this->PERT_AMPLITUDE;
         const auto DOUBLE_SHEET_l = this->DOUBLE_SHEET;
         // Island perturbation delta_B = grad(phi) x e_z = (dphi/dy, -dphi/dx, 0), with
         // phi(x,y) = phi0*cos(pi*(x-xmid)/Lx)*cos(pi*(y-ymid)/(2*Ly)) over the full domain extent, so that
         // it matches the domain's x-periodicity (Section 5.4 of Liu et al. 2025; Lx, Ly generalized here to
         // half the domain extent about its midpoint rather than assuming the domain is centred on the
         // origin, though the GEM setup itself uses a centred domain).
         const auto Lx_l = 0.5 * (Parameters::xmax - Parameters::xmin);
         const auto Ly_l = 0.5 * (Parameters::ymax - Parameters::ymin);
         const auto xmid_l = 0.5 * (Parameters::xmax + Parameters::xmin);
         const auto ymid_l = 0.5 * (Parameters::ymax + Parameters::ymin);
         const auto SEED_l = this->SEED_AMPLITUDE * this->SEED_REF;
         const auto SEEDW_l = this->SEED_WIDTH;
         const auto LOCAL_l = this->LOCALIZED_PERTURBATION;
         const auto LPSI_l = this->LOCAL_AMPLITUDE * this->BX0 * this->LOCAL_WIDTH;   // peak flux A*BX0*w
         const auto LW_l = this->LOCAL_WIDTH;

         fsgrid.parallel_for([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                             phiprof::initializeTimer("setProjectBField-loop"), technical,
                             [=](const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const std::array<Real, 3> xyz = coordinates.getPhysicalCoords(stencil.i, stencil.j, stencil.k);
            const std::array<Real, 3> gridSpacing = coordinates.physicalGridSpacing;
            auto& cell = perb[stencil.ooo()];

            if (DOUBLE_SHEET_l) {
               // Two Harris sheets of opposite polarity at yc=+-Ly_l/2 (yc=y-ymid, Ly_l=domain y
               // half-extent), fully periodic in y:
               //   Bx(y) = B0*[tanh((yc+Ly_l/2)/lambda0) - tanh((yc-Ly_l/2)/lambda0) - 1]
               // Check: at yc=0 (between the sheets) this -> B0*[2*tanh(Ly_l/2/lambda0)-1] -> +B0 for
               // Ly_l>>lambda0; at yc=+-Ly_l (the periodic wrap point, same physical point either side)
               // both expressions give the same value (-B0 for Ly_l>>lambda0), confirming periodicity;
               // near yc=-Ly_l/2 the first tanh term varies while the second saturates at -1, reducing
               // locally to the single original tanh((yc+Ly_l/2)/lambda0) profile recentred on sheet 1
               // -- and near yc=+Ly_l/2, to MINUS that profile, recentred on sheet 2 (opposite
               // polarity) -- matching getPopulationsAt()'s own local-density/drift reasoning for
               // each sheet. By0/Bz0 (BY0_l/BZ0_l, always 0 for the GEM setup) are untouched: this
               // generalisation is specific to the Bx0-driven Harris sheet(s) along y, not those.
               const Real ycD = xyz[1] + 0.5 * gridSpacing[1] - ymid_l;
               cell[fsgrids::bfield::PERBX] = BX0_l * ( tanh((ycD + 0.5*Ly_l) / SCA_LAMBDA_l)
                                                       - tanh((ycD - 0.5*Ly_l) / SCA_LAMBDA_l) - 1.0 );
            } else {
               cell[fsgrids::bfield::PERBX] = BX0_l * tanh((xyz[1] + 0.5 * gridSpacing[1]) / SCA_LAMBDA_l);
            }
            cell[fsgrids::bfield::PERBY] = BY0_l * tanh((xyz[2] + 0.5 * gridSpacing[2]) / SCA_LAMBDA_l);
            cell[fsgrids::bfield::PERBZ] = BZ0_l * tanh((xyz[0] + 0.5 * gridSpacing[0]) / SCA_LAMBDA_l);

            if (PHI0_l != 0.0) {
               if (DOUBLE_SHEET_l) {
                  // phi = -phi0*cos(pi*xc/Lx)*sin(pi*yc/Ly_l): one wavelength along x with the X-line at
                  // xc=0 on both sheets -- Sec. 5.4's island on sheet 1 and its mirror image on sheet 2.
                  // delta_B = grad(phi) x e_z = (dphi/dy, -dphi/dx, 0). Bx is evaluated on its x-face and
                  // By on its y-face, so the face average the LCCL solver uses for node values is 2nd order.
                  const Real xf  = xyz[0] - xmid_l;
                  const Real yf  = xyz[1] - ymid_l;
                  const Real xcD = xf + 0.5 * gridSpacing[0];
                  const Real ycD = yf + 0.5 * gridSpacing[1];
                  cell[fsgrids::bfield::PERBX] += -PHI0_l * (M_PI / Ly_l) * cos(M_PI * xf / Lx_l) * cos(M_PI * ycD / Ly_l);
                  cell[fsgrids::bfield::PERBY] += -PHI0_l * (M_PI / Lx_l) * sin(M_PI * xcD / Lx_l) * sin(M_PI * yf / Ly_l);
               } else {
                  // phi = phi0*cos(pi*X/Lx)*cos(pi*Y/(2*Ly)) (Sec. 5.4), Ly the domain's y half-extent, so
                  // By vanishes at the domain edge -- with conducting walls that is beyond their boundary
                  // layers, not on the walls. Bx on its x-face and By on its y-face, as for the double sheet.
                  const Real xf  = xyz[0] - xmid_l;
                  const Real yf  = xyz[1] - ymid_l;
                  const Real xcD = xf + 0.5 * gridSpacing[0];
                  const Real ycD = yf + 0.5 * gridSpacing[1];
                  cell[fsgrids::bfield::PERBX] += -PHI0_l * (M_PI / (2.0 * Ly_l)) * cos(M_PI * xf / Lx_l) * sin(M_PI * ycD / (2.0 * Ly_l));
                  cell[fsgrids::bfield::PERBY] +=  PHI0_l * (M_PI / Lx_l) * sin(M_PI * xcD / Lx_l) * cos(M_PI * yf / (2.0 * Ly_l));
               }
            }

            if (LOCAL_l) {
               // Localized X-point at each sheet's X-line, as in iPic3D's GEM setup:
               // psi = sum_k s_k*A*BX0*w*exp(-(X^2+(Y-Y_k)^2)/w^2)*cos(pi*X/(10w))*cos(pi*(Y-Y_k)/(10w)) for
               // |X|,|Y-Y_k| < 5w (both cosines positive), 0 outside; delta_B = grad(psi) x e_z = (dpsi/dy, -dpsi/dx, 0).
               // Each face value is the difference of psi between the face's two corners over its length,
               // i.e. the exact face average, so the discrete div B vanishes to round-off however sharp the
               // perturbation is on the grid. Sheet 2 (s=-1) is the rotation-by-pi image of sheet 1.
               // Distances use the nearest periodic image, as for the seed.
               const Real xf = xyz[0] - xmid_l, yf = xyz[1] - ymid_l;
               const Real dx = gridSpacing[0], dy = gridSpacing[1];
               const Real cx[3] = {xf, xf, xf + dx};          // corners: lower-left, upper-left, lower-right
               const Real cy[3] = {yf, yf + dy, yf};
               Real p[3] = {0.0, 0.0, 0.0};
               const int nSheets = DOUBLE_SHEET_l ? 2 : 1;
               for (int c = 0; c < 3; ++c) {
                  for (int k = 0; k < nSheets; ++k) {
                     const Real Yk = DOUBLE_SHEET_l ? (k == 0 ? -0.5 * Ly_l : 0.5 * Ly_l) : 0.0;
                     const Real sk = (k == 0) ? 1.0 : -1.0;
                     const Real X = cx[c] - 2.0 * Lx_l * std::floor((cx[c] + Lx_l) / (2.0 * Lx_l));
                     const Real Y = (cy[c] - Yk) - 2.0 * Ly_l * std::floor((cy[c] - Yk + Ly_l) / (2.0 * Ly_l));
                     if (std::fabs(X) < 5.0 * LW_l && std::fabs(Y) < 5.0 * LW_l) {
                        p[c] += sk * LPSI_l * std::exp(-(X * X + Y * Y) / (LW_l * LW_l))
                                * std::cos(M_PI * X / (10.0 * LW_l)) * std::cos(M_PI * Y / (10.0 * LW_l));
                     }
                  }
               }
               cell[fsgrids::bfield::PERBX] += (p[1] - p[0]) / dy;
               cell[fsgrids::bfield::PERBY] -= (p[2] - p[0]) / dx;
            }

            if (SEED_l != 0.0) {
               // phi_s = sum_k s_k*SEED*(X/w)*exp(-(X^2+(Y-Y_k)^2)/w^2): odd in X, so the X-line moves by
               // ~SEED/phi0*Lx^2/(pi^2 w) and a secondary island at x=0 is no longer held there by
               // symmetry. Sheet 2 (s=-1) is the rotation-by-pi image of sheet 1, phi_s(x,-Y) = -phi_s(x,Y).
               // Distances use the nearest periodic image; the Gaussian is negligible at the seams for
               // w well below Lx and Ly/2. Bx on its x-face, By on its y-face, as for the main term.
               const Real xf = xyz[0] - xmid_l, xc = xf + 0.5 * gridSpacing[0];
               const Real yf = xyz[1] - ymid_l, yc = yf + 0.5 * gridSpacing[1];
               const int nSheets = DOUBLE_SHEET_l ? 2 : 1;
               for (int k = 0; k < nSheets; ++k) {
                  const Real Yk = DOUBLE_SHEET_l ? (k == 0 ? -0.5 * Ly_l : 0.5 * Ly_l) : 0.0;
                  const Real sk = (k == 0) ? 1.0 : -1.0;
                  // dphi_s/dy at the x-face
                  const Real Xa = xf - 2.0 * Lx_l * std::floor((xf + Lx_l) / (2.0 * Lx_l));
                  const Real Ya = (yc - Yk) - 2.0 * Ly_l * std::floor((yc - Yk + Ly_l) / (2.0 * Ly_l));
                  const Real ga = std::exp(-(Xa * Xa + Ya * Ya) / (SEEDW_l * SEEDW_l));
                  cell[fsgrids::bfield::PERBX] += sk * SEED_l * (Xa / SEEDW_l) * (-2.0 * Ya / (SEEDW_l * SEEDW_l)) * ga;
                  // -dphi_s/dx at the y-face
                  const Real Xb = xc - 2.0 * Lx_l * std::floor((xc + Lx_l) / (2.0 * Lx_l));
                  const Real Yb = (yf - Yk) - 2.0 * Ly_l * std::floor((yf - Yk + Ly_l) / (2.0 * Ly_l));
                  const Real gb = std::exp(-(Xb * Xb + Yb * Yb) / (SEEDW_l * SEEDW_l));
                  cell[fsgrids::bfield::PERBY] += -sk * (SEED_l / SEEDW_l) * (1.0 - 2.0 * Xb * Xb / (SEEDW_l * SEEDW_l)) * gb;
               }
            }
         });
      }
   }

} // namespace projects
