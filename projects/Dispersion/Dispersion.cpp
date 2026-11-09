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
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <random>

#include "../../common.h"
#include "../../readparameters.h"
#include "../../object_wrapper.h"
#include "../../velocity_mesh_parameters.h"
#include "../../backgroundfield/backgroundfield.h"
#include "../../backgroundfield/constantfield.hpp"
#include "../../fieldsolver/es_electric_field.hpp"

#include "Dispersion.h"

Real projects::Dispersion::rndRho, projects::Dispersion::rndVel[3];

using namespace std;
using namespace spatial_cell;

namespace projects {
   Dispersion::Dispersion(): Project() { }
   Dispersion::~Dispersion() { }

   bool Dispersion::initialize(void) {return Project::initialize();}

   void Dispersion::addParameters() {
      typedef Readparameters RP;
      RP::add<Real>("Dispersion.B0", "Guide magnetic field strength (T)", this->B0,1.0e-9);
      RP::add<Real>("Dispersion.magXPertAbsAmp", "Absolute amplitude of the magnetic perturbation along x (T)", this->magXPertAbsAmp,1.0e-9);
      RP::add<Real>("Dispersion.magYPertAbsAmp", "Absolute amplitude of the magnetic perturbation along y (T)", this->magYPertAbsAmp,1.0e-9);
      RP::add<Real>("Dispersion.magZPertAbsAmp", "Absolute amplitude of the magnetic perturbation along z (T)", this->magZPertAbsAmp,1.0e-9);
      RP::add<Real>("Dispersion.maxwCutoff", "Cutoff for the maxwellian distribution", this->maxwCutoff,1e-12);
      RP::add<Real>("Dispersion.angleXY", "Orientation of the guide magnetic field with respect to the x-axis in x-y plane (rad)", this->angleXY,0.001);
      RP::add<Real>("Dispersion.angleXZ", "Orientation of the guide magnetic field with respect to the x-axis in x-z plane (rad)", this->angleXZ,0.001);

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {

        DispersionSpeciesParameters* sP=new DispersionSpeciesParameters();
        
        speciesParamsRead.push_back(sP);
        const std::string& pop = getObjectWrapper().particleSpecies[i].name;
        RP::add<Real>(pop + "_Dispersion.VX0", "Bulk velocity (m/s)", sP->VX0,0.0);
        RP::add<Real>(pop + "_Dispersion.VY0", "Bulk velocity (m/s)",sP->VY0,0.0);
        RP::add<Real>(pop + "_Dispersion.VZ0", "Bulk velocity (m/s)",sP->VZ0,0.0);
        RP::add<Real>(pop + "_Dispersion.rho", "Number density (m^-3)", sP->DENSITY,1.0e7);
        RP::add<Real>(pop + "_Dispersion.Temperature", "Temperature (K)", sP->TEMPERATURE,2.0e6);
        RP::add<Real>(pop + "_Dispersion.densityPertRelAmp", "Relative amplitude of the density perturbation", sP->densityPertRelAmp,0.1);
        RP::add<Real>(pop + "_Dispersion.velocityPertAbsAmp", "Absolute amplitude of the velocity perturbation", sP->velocityPertAbsAmp,1.0e6);
        RP::add<Real>(pop + "_Dispersion.velocityNoiseRelAmp", "Relative amplitude of random noise in every velocity cell: f -> f (1 + a ((1/2 - r) - c.w)), r uniform in [0,1) and independent for every velocity cell, w = (1, (v - V0)/v_th), with c chosen so that the noise changes neither the density nor the bulk velocity of the cell. Unlike the density and velocity perturbations, the noise has structure in gyrophase at all harmonics, like the noise of particle codes, and so excites kinetic modes such as ion Bernstein modes; it adds no charge and no current. Reproducible: r depends only on Project_common.seed, the position of the cell, the population and the velocity cell. CPU builds only. 0: off", sP->velocityNoiseRelAmp,0.0);
      }
   }

   void Dispersion::getParameters() {
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
        this->speciesParams.push_back(*this->speciesParamsRead.at(i));
      }
   }

   void Dispersion::hook(
      cuint& stage,
      const dccrg::Dccrg<spatial_cell::SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
      fsgrids::perbspan perb,
      fsgrids::technicalspan technical, FieldSolverGrid &fsgrid
   ) const {
      /*
      if(hook::END_OF_TIME_STEP == stage) {
         int myRank;
         MPI_Comm_rank(MPI_COMM_WORLD,&myRank);

         vector<Real> localRhom(P::xcells_ini, 0.0), outputRhom(P::xcells_ini, 0.0);

         const vector<CellID>& cells = getLocalCells();

         for(uint i=0; i<cells.size(); i++) {
            if(cells[i] <= P::xcells_ini) {
               localRhom[cells[i] - 1] = mpiGrid[cells[i]]->parameters[CellParams::RHOM];
            }
         }

         MPI_Reduce(&(localRhom[0]), &(outputRhom[0]), P::xcells_ini, MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);

         vector<Real> localPerBx(P::xcells_ini, 0.0);
         vector<Real> localPerBy(P::xcells_ini, 0.0);
         vector<Real> localPerBz(P::xcells_ini, 0.0);
         vector<Real> outputPerBx(P::xcells_ini, 0.0);
         vector<Real> outputPerBy(P::xcells_ini, 0.0);
         vector<Real> outputPerBz(P::xcells_ini, 0.0);

         const auto* localSize = &fsgrid.getLocalSize()[0];
         const auto& localStart = fsgrid.getLocalStart();
         for (auto x = 0; x < localSize[0]; ++x) {
            const auto stencil = fsgrid.makeStencil(x, 0, 0);
            localPerBx[x + localStart[0]] = perb[stencil.ooo()][fsgrids::bfield::PERBX];
            localPerBy[x + localStart[0]] = perb[stencil.ooo()][fsgrids::bfield::PERBY];
            localPerBz[x + localStart[0]] = perb[stencil.ooo()][fsgrids::bfield::PERBZ];
         }

         MPI_Reduce(&(localPerBx[0]), &(outputPerBx[0]), P::xcells_ini, MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
         MPI_Reduce(&(localPerBy[0]), &(outputPerBy[0]), P::xcells_ini, MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
         MPI_Reduce(&(localPerBz[0]), &(outputPerBz[0]), P::xcells_ini, MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);

         if(myRank == MASTER_RANK) {
            FILE* outputFile = fopen("perBxt.bin", "ab");
            fwrite(&(outputPerBx[0]), sizeof(outputPerBx[0]), P::xcells_ini, outputFile);
            fclose(outputFile);
            outputFile = fopen("perByt.bin", "ab");
            fwrite(&(outputPerBy[0]), sizeof(outputPerBy[0]), P::xcells_ini, outputFile);
            fclose(outputFile);
            outputFile = fopen("perBzt.bin", "ab");
            fwrite(&(outputPerBz[0]), sizeof(outputPerBz[0]), P::xcells_ini, outputFile);
            fclose(outputFile);
            outputFile = fopen("rhomt.bin", "ab");
            fwrite(&(outputRhom[0]), sizeof(outputRhom[0]), P::xcells_ini, outputFile);
            fclose(outputFile);
         }
      }
      */
   }

   Realf Dispersion::fillPhaseSpace(spatial_cell::SpatialCell *cell,
                                       const uint popID,
                                       const uint nRequested
      ) const {
      const DispersionSpeciesParameters& sP = speciesParams[popID];
      const Real mass = getObjectWrapper().particleSpecies[popID].mass;
      Real initT = sP.TEMPERATURE;
      Real initRho = sP.DENSITY * (1.0 + sP.densityPertRelAmp * (0.5 - this->rndRho));
      const Real initV0X = sP.VX0 + sP.velocityPertAbsAmp * (0.5 - this->rndVel[0]);
      const Real initV0Y = sP.VY0 + sP.velocityPertAbsAmp * (0.5 - this->rndVel[1]);
      const Real initV0Z = sP.VZ0 + sP.velocityPertAbsAmp * (0.5 - this->rndVel[2]);

      // cerr << cell->parameters[CellParams::XCRD] << " " << this->rndRho << " " << this->rndVel[0] << " " << this->rndVel[1] << " " << this->rndVel[2] << endl;

      #ifdef USE_GPU
      vmesh::VelocityMesh *vmesh = cell->dev_get_velocity_mesh(popID);
      vmesh::VelocityBlockContainer* VBC = cell->dev_get_velocity_blocks(popID);
      #else
      vmesh::VelocityMesh *vmesh = cell->get_velocity_mesh(popID);
      vmesh::VelocityBlockContainer* VBC = cell->get_velocity_blocks(popID);
      #endif
      // Loop over blocks
      Realf rhosum = 0;
      #ifdef USE_GPU
      if (sP.velocityNoiseRelAmp != 0.0) {
         std::cerr << "Dispersion: " << getObjectWrapper().particleSpecies[popID].name
                   << "_Dispersion.velocityNoiseRelAmp is implemented for CPU builds only" << std::endl;
         abort();
      }
      #endif
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
               creal vx = vxBlock + (i+0.5)*dvxCell - initV0X;
               creal vy = vyBlock + (j+0.5)*dvyCell - initV0Y;
               creal vz = vzBlock + (k+0.5)*dvzCell - initV0Z;
               Realf value = MaxwellianPhaseSpaceDensity(vx,vy,vz,initT,initRho,mass);
               bufferData[initIndex*WID3 + k*WID2 + j*WID + i] = value;
               //lsum[0] += value;
            };
         }, rhosum);

      #ifndef USE_GPU
      if (sP.velocityNoiseRelAmp != 0.0) {
         addVelocityNoise(cell, popID, nRequested, initV0X, initV0Y, initV0Z, std::sqrt(physicalconstants::K_B*initT/mass),
                          sP.velocityNoiseRelAmp);
      }
      #endif
      return rhosum;
   }

   // Counter-based hash (splitmix64): a reproducible random number for every velocity cell, independent of the
   // number of processes and threads and of the order in which the blocks are filled.
   static inline uint64_t dispersionSplitmix64(uint64_t x) {
      x += 0x9E3779B97F4A7C15ULL;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      return x ^ (x >> 31);
   }

   /* Noise in velocity space (velocityNoiseRelAmp = a): f -> f (1 + a (eta - c.w)), eta = 1/2 - r uniform in
    * (-1/2, 1/2] per velocity cell, w = (1, u_x, u_y, u_z), u = (v - V0)/v_th. The four coefficients c solve
    * sum f w w^T c = sum f eta w over the cell's blocks, so that the noise changes neither sum f (the density) nor
    * sum f u (the bulk velocity); it changes the shape of the distribution, including its structure in gyrophase.
    * Values that would become negative (only possible for a > 1) are set to zero. */
   void Dispersion::addVelocityNoise(spatial_cell::SpatialCell* cell, const uint popID, const uint nBlocks,
                                     const Real V0X, const Real V0Y, const Real V0Z, const Real vth,
                                     const Real amplitude) const {
      vmesh::VelocityMesh* vmesh = cell->get_velocity_mesh(popID);
      vmesh::VelocityBlockContainer* VBC = cell->get_velocity_blocks(popID);
      const vmesh::GlobalID* GIDlist = vmesh->getGrid()->data();
      Realf* data = VBC->getData();
      std::knuth_b rnd;
      setRandomCellSeed(cell, rnd);
      const uint64_t key = (static_cast<uint64_t>(rnd()) << 32) ^ static_cast<uint64_t>(rnd())
                           ^ (static_cast<uint64_t>(popID) << 56);
      auto eta = [&](const vmesh::GlobalID gid, const uint idx) {
         const uint64_t h = dispersionSplitmix64(key ^ (static_cast<uint64_t>(gid)*WID3 + idx));
         return 0.5 - static_cast<Real>(h >> 11)*(1.0/9007199254740992.0);   // 2^-53
      };
      auto weights = [&](const Real* bc, const uint i, const uint j, const uint k, Real (&w)[4]) {
         w[0] = 1.0;
         w[1] = (bc[0] + (i + 0.5)*bc[3] - V0X)/vth;
         w[2] = (bc[1] + (j + 0.5)*bc[4] - V0Y)/vth;
         w[3] = (bc[2] + (k + 0.5)*bc[5] - V0Z)/vth;
      };
      // pass 1: M = sum f w w^T, S = sum f eta w
      Real M[4][4] = {{0}}, S[4] = {0};
      for (uint b = 0; b < nBlocks; ++b) {
         Real bc[6];
         vmesh->getBlockInfo(GIDlist[b], &bc[0]);
         for (uint k = 0; k < WID; ++k) for (uint j = 0; j < WID; ++j) for (uint i = 0; i < WID; ++i) {
            const uint idx = k*WID2 + j*WID + i;
            const Real f = data[b*WID3 + idx];
            if (f == 0.0) { continue; }
            Real w[4];
            weights(bc, i, j, k, w);
            const Real e = eta(GIDlist[b], idx);
            for (int p = 0; p < 4; ++p) {
               S[p] += f*e*w[p];
               for (int q = 0; q < 4; ++q) { M[p][q] += f*w[p]*w[q]; }
            }
         }
      }
      // solve M c = S (symmetric positive definite; Gaussian elimination with partial pivoting)
      Real c[4];
      {
         Real A[4][5];
         for (int p = 0; p < 4; ++p) { for (int q = 0; q < 4; ++q) { A[p][q] = M[p][q]; } A[p][4] = S[p]; }
         for (int col = 0; col < 4; ++col) {
            int piv = col;
            for (int r = col + 1; r < 4; ++r) { if (std::abs(A[r][col]) > std::abs(A[piv][col])) { piv = r; } }
            if (A[piv][col] == 0.0) { return; }   // empty cell: nothing to perturb
            for (int q = 0; q < 5; ++q) { std::swap(A[col][q], A[piv][q]); }
            for (int r = 0; r < 4; ++r) {
               if (r == col) { continue; }
               const Real fac = A[r][col]/A[col][col];
               for (int q = col; q < 5; ++q) { A[r][q] -= fac*A[col][q]; }
            }
         }
         for (int p = 0; p < 4; ++p) { c[p] = A[p][4]/A[p][p]; }
      }
      // pass 2: apply
      for (uint b = 0; b < nBlocks; ++b) {
         Real bc[6];
         vmesh->getBlockInfo(GIDlist[b], &bc[0]);
         for (uint k = 0; k < WID; ++k) for (uint j = 0; j < WID; ++j) for (uint i = 0; i < WID; ++i) {
            const uint idx = k*WID2 + j*WID + i;
            Realf& f = data[b*WID3 + idx];
            if (f == 0.0) { continue; }
            Real w[4];
            weights(bc, i, j, k, w);
            const Real factor = 1.0 + amplitude*(eta(GIDlist[b], idx) - (c[0]*w[0] + c[1]*w[1] + c[2]*w[2] + c[3]*w[3]));
            f = static_cast<Realf>(std::max(f*factor, 0.0));
         }
      }
   }

   void Dispersion::calcCellParameters(spatial_cell::SpatialCell* cell,creal& t) {
      std::knuth_b rndState;
      setRandomCellSeed(cell,rndState);

      this->rndRho=getRandomNumber(rndState);

      this->rndVel[0]=getRandomNumber(rndState);
      this->rndVel[1]=getRandomNumber(rndState);
      this->rndVel[2]=getRandomNumber(rndState);
   }

   void Dispersion::setProjectBField(
      fsgrids::perbspan perb,
      fsgrids::bgbspan bgb,
      fsgrids::technicalspan technical, FieldSolverGrid &fsgrid
   ) {
      ConstantField bgField;
      bgField.initialize(this->B0 * cos(this->angleXY) * cos(this->angleXZ),
                         this->B0 * sin(this->angleXY) * cos(this->angleXZ),
                         this->B0 * sin(this->angleXZ));

      setBackgroundField(bgField, bgb, technical, fsgrid);

      if(!P::isRestart) {
         // local copies for lambda capture
         const auto magXPertAbsAmp_l = this->magXPertAbsAmp;
         const auto magYPertAbsAmp_l = this->magYPertAbsAmp;
         const auto magZPertAbsAmp_l = this->magZPertAbsAmp;
         const auto seed = this->seed;

         fsgrid.parallel_for([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                             phiprof::initializeTimer("setProjectBField"), technical,
                             [=](const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            //const std::array<Real, 3> xyz = coordinates.getPhysicalCoords(stencil.i, stencil.j, stencil.k);
            auto& cell = perb[stencil.ooo()];

            const auto seedmodifier = coordinates.globalIDFromLocalCoordinates(stencil.i, stencil.j, stencil.k);
            std::knuth_b rndState_l;
            rndState_l.seed(seed+seedmodifier);
            Real rndBuffer[3];
            rndBuffer[0] = std::uniform_real_distribution<>(-0.5,0.5)(rndState_l);
            rndBuffer[1] = std::uniform_real_distribution<>(-0.5,0.5)(rndState_l);
            rndBuffer[2] = std::uniform_real_distribution<>(-0.5,0.5)(rndState_l);

            cell[fsgrids::bfield::PERBX] = magXPertAbsAmp_l * rndBuffer[0];
            cell[fsgrids::bfield::PERBY] = magYPertAbsAmp_l * rndBuffer[1];
            cell[fsgrids::bfield::PERBZ] = magZPertAbsAmp_l * rndBuffer[2];
         });
      }
   }

   /*! Solve -eps0*grad^2(Phi) = rho(x,0), E = -grad(Phi), and write the
    * result into e. Reuses the ES field solver's own Poisson solve
    * (es_ElectrostaticPotential, a general 3D 7-point-stencil solve --
    * no assumption that rho varies in only one dimension) and its
    * per-component gradient functions (es_calculateElectricFieldX/Y/Z).
    * Calls the plain per-component gradients directly rather than the
    * SysBoundary-aware calculateElectrostaticField wrapper: Dispersion
    * is a periodic-domain project, so there are no non-trivial boundary
    * cells needing that delegation, and this avoids pulling in a
    * SysBoundary dependency here for something that would never fire.
    *
    * moments must already hold rho(x,0) (fed from the initial f^0
    * before this is called -- see vlasiator.cpp's pre-loop
    * initialization sequence). Overwrites e rather than adding to it:
    * only meaningful as a one-time initialization at t=0, not a
    * per-step correction.
    */
   void Dispersion::setProjectEField(
      fsgrids::momentsspan moments,
      fsgrids::efieldspan e,
      fsgrids::technicalspan technical, FieldSolverGrid &fsgrid
   ) const {
      if (P::isRestart) {
         return; // restarts already have a properly-saved, self-consistent E
      }

      fsgrid::FsData<std::array<Real, fsgrids::potential::N_POTENTIAL>> Phi(fsgrid.getNumStorageCells());
      es_ElectrostaticPotential(Phi.view(), moments, technical, fsgrid);

      const auto dxyz = fsgrid.getGridSpacing();
      const auto PhiView = Phi.view();
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("setProjectEField"), technical,
         [=](const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            es_calculateElectricFieldX(e, PhiView, stencil, dxyz);
            es_calculateElectricFieldY(e, PhiView, stencil, dxyz);
            es_calculateElectricFieldZ(e, PhiView, stencil, dxyz);
         });
      fsgrid.updateGhostCells(e);
   }
} // namespace projects
