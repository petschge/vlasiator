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
#include <numbers>

#include "../../common.h"
#include "../../readparameters.h"
#include "../../backgroundfield/backgroundfield.h"
#include "../../object_wrapper.h"

#include "LCCL_reconnection.h"

using namespace std;

namespace projects {
   LCCL_Reconnection::LCCL_Reconnection(): TriAxisSearch() { }
   LCCL_Reconnection::~LCCL_Reconnection() { }

   bool LCCL_Reconnection::initialize(void) {
      bool success = Project::initialize();

      return success;
   }

   void LCCL_Reconnection::addParameters() {
      typedef Readparameters RP;
      RP::add<Real>("LCCL_Reconnection.B0", "B0", this->B0,1.0e-10);
      RP::add<Real>("LCCL_Reconnection.ionInertialLength", "ionInertialLength", this->ionInertialLength,227694.0);
      RP::add<bool>("LCCL_Reconnection.useDoubleCS", "useDoubleCS", this->useDoubleCS,false);

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
         const std::string& pop = getObjectWrapper().particleSpecies[i].name;
         LCCL_ReconnectionSpeciesParameters* sP=new LCCL_ReconnectionSpeciesParameters();

         this->speciesParamsRead.push_back(sP);
         RP::add<Real>(pop + "_LCCL_Reconnection.rho", "Number density (m^-3)", sP->rho,10.e6);
         // RP::add<Real>(pop + "_LCCL_Reconnection.Temperature", "Temperature (K)", sP->T,0.86456498092);
         RP::add<Real>(pop + "_LCCL_Reconnection.thermalSpeed", "Thermal speed (m/s)", sP->thermalSpeed,1.0);
         RP::add<Real>(pop + "_LCCL_Reconnection.driftSpeed", "Drift speed (m/s)", sP->driftSpeed,1.0);         
      }
   }

   void LCCL_Reconnection::getParameters(){
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
        this->speciesParams.push_back(*this->speciesParamsRead.at(i));
      }
   }

   Realf LCCL_Reconnection::fillPhaseSpace(spatial_cell::SpatialCell *cell,
                                       const uint popID,
                                       const uint nRequested
      ) const {
      const LCCL_ReconnectionSpeciesParameters& sP = this->speciesParams[popID];

      // Fetch spatial cell center coordinates
      const Real x  = cell->parameters[CellParams::XCRD] + 0.5*cell->parameters[CellParams::DX];
      const Real y  = cell->parameters[CellParams::YCRD] + 0.5*cell->parameters[CellParams::DY];
      // const Real z  = cell->parameters[CellParams::ZCRD] + 0.5*cell->parameters[CellParams::DZ];
      //const Real Lx = 12.8 * this->ionInertialLength;
      //const Real Ly = 6.4 * this->ionInertialLength;
      const Real Lx = (P::xmax - P::xmin) * 0.5;
      const Real Ly = (P::ymax - P::ymin) * 0.5;

      creal mass = getObjectWrapper().particleSpecies[popID].mass;
      creal charge = getObjectWrapper().particleSpecies[popID].charge;
      creal mu0 = physicalconstants::MU_0;
      Real rho_drift;

      if (this->useDoubleCS) {
         // two current sheets at y = 0.5 * Ly and y = -0.5 * Ly
         rho_drift = sP.rho * (
            1.0 / pow(cosh((y + 0.5 * Ly) / (this->ionInertialLength * 0.5)), 2)
            + 1.0 / pow(cosh((y - 0.5 * Ly) / (this->ionInertialLength * 0.5)), 2)
         );
      } else {
         // one current sheet at y = 0
         rho_drift = sP.rho * 1.0 / pow(cosh(y / (this->ionInertialLength * 0.5)), 2);
      }

      creal rho_background = sP.rho * 0.2;

      creal v_ts = sP.thermalSpeed;
      Real v_ds = sP.driftSpeed;
      
      if (this->useDoubleCS && y>=0) {
         v_ds *= -1.0;
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
               //const Realf value = MaxwellianPhaseSpaceDensity(vx, vy, vz, sP.T, rho, mass);
               const Realf pert = (rho_drift / pow((sqrt(2.0 * numbers::pi) * v_ts), 3.0)) * exp(- (vx*vx + vy*vy + (vz-v_ds)*(vz-v_ds))/(2.0 * v_ts * v_ts)); 
               const Realf background =  (rho_background / pow((sqrt(2.0 * numbers::pi) * v_ts), 3.0)) * exp(- (vx*vx + vy*vy + vz*vz)/(2.0 * v_ts * v_ts));
               const Realf value = pert + background;

               bufferData[initIndex*WID3 + k*WID2 + j*WID + i] = value;
               //lsum[0] += value;
            };
         }, rhosum);
      return rhosum;
   }

   void LCCL_Reconnection::calcCellParameters(spatial_cell::SpatialCell* cell,creal& t) {}

   Realf LCCL_Reconnection::probePhaseSpace(spatial_cell::SpatialCell *cell,
                                        const uint popID,
                                        Real vx_in, Real vy_in, Real vz_in
      ) const {

      const LCCL_ReconnectionSpeciesParameters& sP = this->speciesParams[popID];

      // Fetch spatial cell center coordinates
      const Real x  = cell->parameters[CellParams::XCRD] + 0.5*cell->parameters[CellParams::DX];
      const Real y  = cell->parameters[CellParams::YCRD] + 0.5*cell->parameters[CellParams::DY];
      // const Real z  = cell->parameters[CellParams::ZCRD] + 0.5*cell->parameters[CellParams::DZ];
      //const Real Lx = 12.8 * this->ionInertialLength;
      //const Real Ly = 6.4 * this->ionInertialLength;
      const Real Lx = (P::xmax - P::xmin) * 0.5;
      const Real Ly = (P::ymax - P::ymin) * 0.5;

      creal mass = getObjectWrapper().particleSpecies[popID].mass;
      creal charge = getObjectWrapper().particleSpecies[popID].charge;
      creal mu0 = physicalconstants::MU_0;
      Real rho_drift;

      if (this->useDoubleCS) {
         // two current sheets at y = 0.5 * Ly and y = -0.5 * Ly
         rho_drift = sP.rho * (
            1.0 / pow(cosh((y + 0.5 * Ly) / (this->ionInertialLength * 0.5)), 2)
            + 1.0 / pow(cosh((y - 0.5 * Ly) / (this->ionInertialLength * 0.5)), 2)
         );
      } else {
         // one current sheet at y = 0
         rho_drift = sP.rho * 1.0 / pow(cosh(y / (this->ionInertialLength * 0.5)), 2);
      }

      creal rho_background = sP.rho * 0.2;

      creal v_ts = sP.thermalSpeed;
      Real v_ds = sP.driftSpeed;
      
      if (this->useDoubleCS && y>=0) {
         v_ds *= -1.0;
      }

      const Realf pert = (rho_drift / pow((sqrt(2.0 * numbers::pi) * v_ts), 3.0)) * exp(- (vx_in*vx_in + vy_in*vy_in + (vz_in-v_ds)*(vz_in-v_ds))/(2.0 * v_ts * v_ts)); 
      const Realf background =  (rho_background / pow((sqrt(2.0 * numbers::pi) * v_ts), 3.0)) * exp(- (vx_in*vx_in + vy_in*vy_in + vz_in*vz_in)/(2.0 * v_ts * v_ts));
      const Realf value = pert + background;

      return value;
   }

   std::vector<std::array<Real, 3> > LCCL_Reconnection::getV0(
                                                creal x,
                                                creal y,
                                                creal z,
                                                const uint popID
                                               ) const {
      const LCCL_ReconnectionSpeciesParameters& sP = speciesParams[popID];
      Real v_ds = sP.driftSpeed;
      
      if (this->useDoubleCS && y>=0) {
         v_ds *= -1.0;
      }
      // each cell has effectively two peaks: one at v_ds and one at 0

      vector<std::array<Real, 3> > centerPoints = {{0.0, 0.0, 0.0}, {0.0, 0.0, -v_ds}};

      return centerPoints;
   }

   void LCCL_Reconnection::setProjectBField(fsgrids::perbspan perb,
                                 fsgrids::bgbspan bgb,
                                 fsgrids::technicalspan technical, FieldSolverGrid &fsgrid) {
      setBackgroundFieldToZero(fsgrid, technical, bgb);

      if (!P::isRestart) {
         // local copies for lambda capture
         const Real ionInertialLength_lcl = this->ionInertialLength;
         const Real B0_lcl = this->B0;
         //const Real Lx = 12.8 * this->ionInertialLength;
         //const Real Ly = 6.4 * this->ionInertialLength;
         const Real Lx = (P::xmax - P::xmin) * 0.5;
         const Real Ly = (P::ymax - P::ymin) * 0.5;

         const Real phi0 = 0.1 * this->B0 * this->ionInertialLength;
         const bool useDoubleCS_lcl = this->useDoubleCS;

         fsgrid.parallel_for([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                             phiprof::initializeTimer("setProjectBField"), technical,
                             [=](const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const std::array<Real, 3> xyz = coordinates.getPhysicalCoords(stencil.i, stencil.j, stencil.k);
            const std::array<Real, 3> gridSpacing = coordinates.physicalGridSpacing;
            auto& cell = perb[stencil.ooo()];

            const Real dx = gridSpacing[0];
            const Real dy = gridSpacing[1];

            const Real x = xyz[0] + 0.5 * dx;
            const Real y = xyz[1] + 0.5 * dy;

            if (useDoubleCS_lcl) {
               // two current sheets at y = 0.5 * Ly and y = -0.5 * Ly
               cell[fsgrids::bfield::PERBX] = B0_lcl * (tanh((y + 0.5 * Ly) / (ionInertialLength_lcl*0.5)) - tanh((y - 0.5*Ly) / (ionInertialLength_lcl*0.5)) - 1.0);
               cell[fsgrids::bfield::PERBY] = 0.0;
               cell[fsgrids::bfield::PERBZ] = 0.0;

               cell[fsgrids::bfield::PERBX] += -1.0*(1.0 * phi0 * M_PI / Ly)* (cos(M_PI * (x) / Lx)*sin(1.0*M_PI * (y + 0.5*Ly) / Ly));
               cell[fsgrids::bfield::PERBY] += 1.0*(1.0 * phi0 * M_PI / Lx)* (sin(M_PI * (x) / Lx)*cos(1.0*M_PI * (y + 0.5*Ly) / Ly));

            } else {
               // single current sheet at y = 0
               cell[fsgrids::bfield::PERBX] = B0_lcl * tanh(y / (ionInertialLength_lcl*0.5));
               cell[fsgrids::bfield::PERBY] = 0.0;
               cell[fsgrids::bfield::PERBZ] = 0.0;

               cell[fsgrids::bfield::PERBX] += -1.0*(0.5 * phi0 * M_PI / Ly)* (cos(M_PI * (x) / Lx)*sin(0.5*M_PI * (y) / Ly));
               cell[fsgrids::bfield::PERBY] += 1.0*(1.0 * phi0 * M_PI / Lx)* (sin(M_PI * (x) / Lx)*cos(0.5*M_PI * (y) / Ly));
            }
         });
      }
   }

} // namespace projects