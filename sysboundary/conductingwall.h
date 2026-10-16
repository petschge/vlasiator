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

#ifndef CONDUCTINGWALL_H
#define CONDUCTINGWALL_H

#include <vector>
#include "../definitions.h"
#include "../readparameters.h"
#include "../spatial_cells/spatial_cell_wrapper.hpp"
#include "sysboundarycondition.h"

namespace SBC {

   struct ConductingWallSpeciesParameters {
      /*! Array of bool telling which faces are going to be skipped by the Vlasov system boundary condition for
       * this species. */
      std::array<bool, 6> facesToSkipVlasov;
   };

   /*!\brief ConductingWall is a class applying a perfectly-conducting planar wall boundary condition.
    *
    * ConductingWall handles cells tagged as sysboundarytype::CONDUCTINGWALL. It models a stationary,
    * perfectly-conducting (PEC) wall at the given face(s):
    *
    * - Fields: the wall-normal component of B is set to zero (flux cannot penetrate a conductor); the
    *   tangential components of B are copied from the nearest interior (NOT_SYSBOUNDARY) cell found by
    *   searching strictly along the wall's own interior-facing direction (never across the wall or into
    *   another boundary layer). The tangential components of E are set to zero.
    *
    * - Particles: specular reflection. The wall-normal velocity component of the distribution function is
    *   mirrored: f_wall(v) = f_interior(v with the normal component negated), read directly off the velocity
    *   grid of the nearest interior cell. This is an exact operation (not a Maxwellian approximation) because
    *   the velocity mesh is uniform and, for the configurations this is intended for, symmetric about v=0 in
    *   the wall-normal direction, so mirroring is a block/cell index flip rather than an interpolation.
    *
    * Unlike Outflow, this boundary condition is only meaningful on a single face's normal axis at a time; a
    * cell sitting in a corner between two ConductingWall faces is not supported (and is not needed for the
    * planar GEM-type setups this was written for).
    */
   class ConductingWall: public OuterBoundaryCondition {
   public:
      ConductingWall();
      virtual ~ConductingWall();

      static void addParameters();
      virtual void getParameters() override;

      virtual void initSysBoundary(
         creal& t,
         Project &project
      ) override;
      virtual void applyInitialState(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                     fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                                     fsgrids::perbspan perb,
                                     fsgrids::bgbspan bgb, Project& project) override;
      virtual void updateState(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                               fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                               fsgrids::perbspan perb,
                               fsgrids::bgbspan bgb, creal t) override;
      virtual Real fieldSolverBoundaryCondMagneticField(fsgrids::perbspan b,
                                                        fsgrids::constbgbspan bgb,
                                                        fsgrids::consttechnicalspan technical,
                                                        const std::array<Real, 3>& gridSpacing,
                                                        const std::array<fsgrid::FsSize_t, 3>& globalCoordinates,
                                                        const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void fieldSolverBoundaryCondElectricField(fsgrids::efieldspan e,
                                                        const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void fieldSolverBoundaryCondHallElectricField(fsgrids::ehallspan ehall,
                                                            const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void
      fieldSolverBoundaryCondGradPeElectricField(fsgrids::egradpespan EGradPe,
                                                 const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void
      fieldSolverBoundaryCondDerivatives(fsgrids::dperbspan dperb,
                                         fsgrids::dmomentsspan dmoments,
                                         const fsgrid::FsStencil& stencil, cuint RKCase, cuint component) override;
      virtual void fieldSolverBoundaryCondBVOLDerivatives(fsgrids::volspan vols,
                                                          const fsgrid::FsStencil& stencil, cuint component) override;
      virtual void vlasovBoundaryCondition(
         dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
         const CellID& cellID,
         const uint popID,
         const bool calculate_V_moments
      ) override;

      virtual void getFaces(bool* faces) override;
      virtual std::string getName() const override;
      virtual uint getIndex() const override;

   protected:
      /*! Array of bool telling which faces are going to be processed by the fields system boundary condition.*/
      bool facesToSkipFields[6];
      /*! List of faces on which conducting-wall boundary conditions are to be applied ([xyz][+-]). */
      std::vector<std::string> faceList;
      /*! List of faces on which no fields conducting-wall boundary conditions are to be applied ([xyz][+-]). */
      std::vector<std::string> faceNoFieldsList;
      std::vector<ConductingWallSpeciesParameters> speciesParams;

      /*! Given a face index (0:x+,1:x-,2:y+,3:y-,4:z+,5:z-), return the axis it is normal to (0,1,2). */
      static uint faceNormalAxis(uint face) { return face / 2; }
      /*! Given a face index, return the single-cell offset (-1/0/0 etc.) pointing from the wall into the
       * interior of the domain along its normal axis. */
      static int faceInteriorOffset(uint face) { return (face % 2 == 0) ? -1 : 1; }

      /*! fsgrid-level: search strictly along (axis,dir) for the nearest NOT_SYSBOUNDARY cell's tangential B. */
      static Real fieldBoundaryCopyFromInteriorMagneticField(
         fsgrids::perbspan b, fsgrids::consttechnicalspan technical,
         const fsgrid::FsStencil& stencil, cuint component, cuint axis, cint dir);

      /*! dccrg-level: search strictly along (axis,dir) for the nearest NOT_SYSBOUNDARY cell. */
      static CellID findInteriorNeighborAlongAxis(
         const dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
         const CellID& cellID, uint axis, int dir);
   }; // class ConductingWall
} // namespace SBC

#endif
