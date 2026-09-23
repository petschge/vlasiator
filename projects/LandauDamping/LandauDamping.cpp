/*
 * This file is part of Vlasiator.
 *
 * See LandauDamping.h for the physics setup this project reproduces
 * (Liu, Cai, Cao & Lapenta 2025, JCP 528:113840, Section 5.1).
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "../../common.h"
#include "../../readparameters.h"
#include "../../object_wrapper.h"
#include "../../velocity_mesh_parameters.h"
#include "../../backgroundfield/backgroundfield.h"
#include "../../backgroundfield/constantfield.hpp"
#include "../../fieldsolver/es_electric_field.hpp"

#include "LandauDamping.h"

using namespace std;
using namespace spatial_cell;

namespace projects {
   LandauDamping::LandauDamping(): Project() { }
   LandauDamping::~LandauDamping() { }

   bool LandauDamping::initialize(void) {return Project::initialize();}

   void LandauDamping::addParameters() {
      typedef Readparameters RP;
      RP::add("LandauDamping.k", "Wavenumber of the density perturbation (m^-1)", 1.0);

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
         const std::string& pop = getObjectWrapper().particleSpecies[i].name;
         RP::add(pop + "_LandauDamping.VX0", "Bulk velocity (m/s)", 0.0);
         RP::add(pop + "_LandauDamping.VY0", "Bulk velocity (m/s)", 0.0);
         RP::add(pop + "_LandauDamping.VZ0", "Bulk velocity (m/s)", 0.0);
         RP::add(pop + "_LandauDamping.rho", "Number density (m^-3)", 1.0e6);
         RP::add(pop + "_LandauDamping.Temperature", "Temperature (K)", 1.0e5);
         RP::add(pop + "_LandauDamping.densityPertRelAmp",
                 "Relative amplitude of the sin(k*x) density perturbation "
                 "(the paper's alpha0; set to 0 for a non-perturbed, e.g. "
                 "static-ion-background, species)", 0.0);
      }
   }

   void LandauDamping::getParameters() {
      Project::getParameters();
      typedef Readparameters RP;
      RP::get("LandauDamping.k", this->k);

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
        const std::string& pop = getObjectWrapper().particleSpecies[i].name;
        LandauDampingSpeciesParameters sP;
        RP::get(pop + "_LandauDamping.VX0", sP.VX0);
        RP::get(pop + "_LandauDamping.VY0", sP.VY0);
        RP::get(pop + "_LandauDamping.VZ0", sP.VZ0);
        RP::get(pop + "_LandauDamping.rho", sP.DENSITY);
        RP::get(pop + "_LandauDamping.Temperature", sP.TEMPERATURE);
        RP::get(pop + "_LandauDamping.densityPertRelAmp", sP.densityPertRelAmp);

         speciesParams.push_back(sP);
      }
   }

   Realf LandauDamping::fillPhaseSpace(spatial_cell::SpatialCell *cell,
                                       const uint popID,
                                       const uint nRequested
      ) const {
      const LandauDampingSpeciesParameters& sP = speciesParams[popID];
      const Real mass = getObjectWrapper().particleSpecies[popID].mass;

      // Spatial cell center coordinate -- same pattern as MultiPeak's own
      // fillPhaseSpace (x-dependent rhoFactor computed host-side, before
      // the device loop, then captured by value).
      const Real x = cell->parameters[CellParams::XCRD] + 0.5*cell->parameters[CellParams::DX];

      const Real initT = sP.TEMPERATURE;
      // f_e(x,vx,vy,0) = (1/(pi*beta))*exp[-(vx^2+vy^2)/beta]*(1+alpha0*sin(k*x))
      // -- the (1+alpha0*sin(k*x)) factor is exactly a sinusoidal
      // modulation of the DENSITY (the Maxwellian's own normalization),
      // not of vx/vy/T, so it is applied here to rho alone.
      const Real initRho = sP.DENSITY * (1.0 + sP.densityPertRelAmp * sin(this->k * x));
      const Real initV0X = sP.VX0;
      const Real initV0Y = sP.VY0;
      const Real initV0Z = sP.VZ0;

      // device-accessable variables
      const Real initRhoDev = initRho;
      const Real initTDev = initT;
      const Real initV0XDev = initV0X;
      const Real initV0YDev = initV0Y;
      const Real initV0ZDev = initV0Z;

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
               creal vx = vxBlock + (i+0.5)*dvxCell - initV0XDev;
               creal vy = vyBlock + (j+0.5)*dvyCell - initV0YDev;
               creal vz = vzBlock + (k+0.5)*dvzCell - initV0ZDev;
               Realf value = MaxwellianPhaseSpaceDensity(vx,vy,vz,initTDev,initRhoDev,mass);
               bufferData[initIndex*WID3 + k*WID2 + j*WID + i] = value;
            };
         }, rhosum);
      return rhosum;
   }

   // Base class default is a print-warning-and-exit(1) stub (confirmed
   // directly against projects/project.cpp), not a safe no-op -- every
   // project must supply its own override. This project needs no
   // per-cell setup (no random state, nothing precomputed outside
   // fillPhaseSpace itself); an empty body is sufficient, matching
   // Riemann1::calcCellParameters's own, identical, working pattern.
   void LandauDamping::calcCellParameters(spatial_cell::SpatialCell* cell, creal& t) { }

   void LandauDamping::setProjectBField(
      fsgrids::perbspan perb,
      fsgrids::bgbspan bgb,
      fsgrids::technicalspan technical, FieldSolverGrid &fsgrid
   ) {
      // No guide field, no perturbation -- Bx=By=Bz=0 everywhere,
      // matching the paper's own Bz(x,0)=0 (Section 5.1); Bx,By are not
      // dynamical variables in the paper's reduced 1d2v system at all
      // (Eq. 51 has only Ex, Ey, Bz), so they are held at exactly zero
      // throughout via the same mechanism.
      ConstantField bgField;
      bgField.initialize(0.0, 0.0, 0.0);
      setBackgroundField(bgField, bgb, technical, fsgrid);

      if(!P::isRestart) {
         fsgrid.parallel_for([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                             phiprof::initializeTimer("setProjectBField"), technical,
                             [=](const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            auto& cell = perb[stencil.ooo()];
            cell[fsgrids::bfield::PERBX] = 0.0;
            cell[fsgrids::bfield::PERBY] = 0.0;
            cell[fsgrids::bfield::PERBZ] = 0.0;
         });
      }
   }

   /*! Solve -eps0*grad^2(Phi) = rho(x,0), E = -grad(Phi), and write the
    * result into e. Identical approach to Dispersion::setProjectEField
    * (see that file's own, more detailed comment) -- reuses the ES field
    * solver's own Poisson solve and per-component gradient functions.
    * This gives exactly the paper's own initial-condition precondition
    * (Section 5.1: "The initial electric field Ex is solved by the
    * Poisson equation"), solved numerically here from the ACTUAL initial
    * (sinusoidally perturbed) charge density -- including the uniform,
    * static ion background's own contribution -- rather than from the
    * paper's own hand-derived, dimensionless lambda^2*Ex(x,0)=1-n_e(x)
    * (which is this same Poisson solve's dimensional equivalent for
    * their specific lambda=1 unit choice; solving it numerically here
    * avoids needing to hand-rederive that closed form in SI units, and
    * matches what the paper's own CSL-PE reference solution does).
    */
   void LandauDamping::setProjectEField(
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
