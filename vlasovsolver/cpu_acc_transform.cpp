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

#include "../object_wrapper.h"
#include "../sysboundary/ionosphere.h"

#include "cpu_acc_transform.hpp"

using namespace std;
using namespace spatial_cell;
using namespace Eigen;



/*!
  Compute max timestep for vlasov acceleration for the particular population
  in one spatial cell.

 * @param spatial_cell Spatial cell containing the accelerated population.
 * @param popID ID of the accelerated particle species.
*/
void updateAccelerationMaxdt(
   SpatialCell* spatial_cell,
   const uint popID)
{
   const Real Bx = spatial_cell->parameters[CellParams::BGBXVOL]+spatial_cell->parameters[CellParams::PERBXVOL];
   const Real By = spatial_cell->parameters[CellParams::BGBYVOL]+spatial_cell->parameters[CellParams::PERBYVOL];
   const Real Bz = spatial_cell->parameters[CellParams::BGBZVOL]+spatial_cell->parameters[CellParams::PERBZVOL];
   const Eigen::Matrix<Real,3,1> B(Bx,By,Bz);
   const Real B_mag = B.norm() + 1e-30;
   const Real gyro_period = 2 * M_PI * getObjectWrapper().particleSpecies[popID].mass
      / (getObjectWrapper().particleSpecies[popID].charge * B_mag);

   // Set maximum timestep limit for this cell, based on a maximum allowed rotation angle
   spatial_cell->set_max_v_dt(popID,fabs(gyro_period)*(P::maxSlAccelerationRotation/360.0));
}


/*!
 Compute the affine (rotation + translation) transform representing the
 velocity-space Lorentz-force push over one subcycle substep.

 the combined rotation+translation is built up from many small substeps (0.1
 deg of gyration each) rather than in one step, so that the E x B drift is
 captured to good accuracy even when dt corresponds to many degrees of
 gyration.

 * @param spatial_cell Spatial cell containing the accelerated population.
 * @param popID ID of the accelerated particle species.
 * @param dt Time step of one subcycle.
*/

Eigen::Transform<Real,3,Eigen::Affine> compute_acceleration_transformation(
        const SpatialCell* spatial_cell,
        const uint popID,
        const Real& dt) {
   // total field
   const Real Bx = spatial_cell->parameters[CellParams::BGBXVOL]+spatial_cell->parameters[CellParams::PERBXVOL];
   const Real By = spatial_cell->parameters[CellParams::BGBYVOL]+spatial_cell->parameters[CellParams::PERBYVOL];
   const Real Bz = spatial_cell->parameters[CellParams::BGBZVOL]+spatial_cell->parameters[CellParams::PERBZVOL];

   const Eigen::Matrix<Real,3,1> B(Bx,By,Bz);
   Eigen::Matrix<Real,3,1> unit_B(B.normalized());

   // If B equals zero then gyro_period and unit_B are NAN.
   // Guard against that by adding epsilons:
   const Real B_mag = B.norm() + 1e-30;
   if (B_mag < 1e-28) {
      unit_B(0,0) = 0; unit_B(1,0) = 0; unit_B(2,0) = 1;
   }

   const Real gyro_period
     = 2 * M_PI * getObjectWrapper().particleSpecies[popID].mass
     / (getObjectWrapper().particleSpecies[popID].charge * B_mag);

   // Only LDZ needs the Hall/bulk-velocity pivot reconstruction; ES and AP
   // hand us the complete E
   const bool fieldSolverNeedsHallReconstruction = (P::fieldSolverMethod == "LDZ" || P::fieldSolverMethod == "default_fieldsolver");

   // curl of B (only perturbed, curl of background field is always 0!) and
   // the Hall prefactor 1/(mu0*rho_q)
   Real hallPrefactor = 0.0;
   Real dBXdy = 0.0, dBXdz = 0.0, dBYdx = 0.0, dBYdz = 0.0, dBZdx = 0.0, dBZdy = 0.0;
   Eigen::Matrix<Real,3,1> bulk_velocity(0.0, 0.0, 0.0);
   if (fieldSolverNeedsHallReconstruction) {
      // scale rho for hall term, if user requests
      const Real EPSILON = 1e10 * numeric_limits<Real>::min();
      const Real rhoq = spatial_cell->parameters[CellParams::RHOQ_V] + EPSILON;
      const Real hallRhoq = (rhoq <= Parameters::hallMinimumRhoq) ? Parameters::hallMinimumRhoq : rhoq;
      hallPrefactor = 1.0 / (physicalconstants::MU_0 * hallRhoq);

      dBXdy = spatial_cell->derivativesBVOL[bvolderivatives::dPERBXVOLdy];
      dBXdz = spatial_cell->derivativesBVOL[bvolderivatives::dPERBXVOLdz];
      dBYdx = spatial_cell->derivativesBVOL[bvolderivatives::dPERBYVOLdx];
      dBYdz = spatial_cell->derivativesBVOL[bvolderivatives::dPERBYVOLdz];
      dBZdx = spatial_cell->derivativesBVOL[bvolderivatives::dPERBZVOLdx];
      dBZdy = spatial_cell->derivativesBVOL[bvolderivatives::dPERBZVOLdy];

      bulk_velocity = Eigen::Matrix<Real,3,1>(spatial_cell->parameters[CellParams::VX_V],
                                              spatial_cell->parameters[CellParams::VY_V],
                                              spatial_cell->parameters[CellParams::VZ_V]);
   }

   // The electric field from ES/AP or
   const Eigen::Matrix<Real,3,1> E(spatial_cell->parameters[CellParams::EXVOL],
                                   spatial_cell->parameters[CellParams::EYVOL],
                                   spatial_cell->parameters[CellParams::EZVOL]);

   // compute total transformation
   Transform<Real,3,Affine> total_transform(Matrix<Real, 4, 4>::Identity());

   unsigned int bulk_velocity_substeps; // in this many substeps we iterate forward velocity when the complete transformation is computed (0.1 deg per substep).
   bulk_velocity_substeps = fabs(dt) / fabs(gyro_period*(0.1/360.0));
   if (bulk_velocity_substeps < 1) {
      bulk_velocity_substeps=1;
   }

   const Real substeps_radians = -(2.0*M_PI*dt/gyro_period)/bulk_velocity_substeps; // how many radians each substep is.
   const Real substeps_dt=dt/bulk_velocity_substeps;                                // how many s each substep is
   Eigen::Matrix<Real,3,1> EgradPe(
      spatial_cell->parameters[CellParams::EXGRADPE],
      spatial_cell->parameters[CellParams::EYGRADPE],
      spatial_cell->parameters[CellParams::EZGRADPE]);

   for (uint i=0; i<bulk_velocity_substeps; ++i) {
      if (fieldSolverNeedsHallReconstruction) {
         // rotation origin is the point through which we place our rotation axis (direction of which is unitB).
         // first add bulk velocity (using the total transform computed this far.
         Eigen::Matrix<Real,3,1> rotation_pivot(total_transform*bulk_velocity);

         //include lorentzHallTerm (we should include, always)
         rotation_pivot[0]-= hallPrefactor*(dBZdy - dBYdz);
         rotation_pivot[1]-= hallPrefactor*(dBXdz - dBZdx);
         rotation_pivot[2]-= hallPrefactor*(dBYdx - dBXdy);

         // add to transform matrix the small rotation around  pivot
         // when added like this, and not using *= operator, the transformations
         // are in the correct order
         total_transform = Translation<Real,3>(-rotation_pivot)*total_transform;
         total_transform = AngleAxis<Real>(substeps_radians,unit_B)*total_transform;
         total_transform = Translation<Real,3>(rotation_pivot)*total_transform;
      } else {
         // v x B gyration: pure rotation around the B axis through the origin.
         total_transform = AngleAxis<Real>(substeps_radians,unit_B)*total_transform;
      }

      // Electron pressure gradient term, only for the old field solver
      if(Parameters::ohmGradPeTerm > 0) {
         total_transform=Translation<Real,3>( (fabs(getObjectWrapper().particleSpecies[popID].charge)/getObjectWrapper().particleSpecies[popID].mass) * EgradPe * substeps_dt) * total_transform;
      }

      // Electric field acceleration, only for ES and AP field solver
      if (!fieldSolverNeedsHallReconstruction) {
         total_transform=Translation<Real,3>( (getObjectWrapper().particleSpecies[popID].charge/getObjectWrapper().particleSpecies[popID].mass) * E * substeps_dt) * total_transform;
      }
   }

   return total_transform;
}
