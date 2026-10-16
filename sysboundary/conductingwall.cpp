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

/*!\file conductingwall.cpp
 * \brief Implementation of the class SysBoundaryCondition::ConductingWall to handle cells classified as
 * sysboundarytype::CONDUCTINGWALL.
 */

#include <cstdlib>
#include <iostream>

#include "../object_wrapper.h"
#include "conductingwall.h"
#include "../projects/projects_common.h"
#include "../vlasovsolver/vlasovmover.h"
#include "../grid.h"

using namespace std;

namespace SBC {
   ConductingWall::ConductingWall() : OuterBoundaryCondition() {}
   ConductingWall::~ConductingWall() {}

   void ConductingWall::addParameters() {
      Readparameters::addComposing("conductingwall.faceNoFields", "List of faces on which no field conducting-wall boundary conditions are to be applied ([xyz][+-]).");
      Readparameters::add("conductingwall.precedence", "Precedence value of the conducting-wall system boundary condition (integer), the higher the stronger.", 4);
      Readparameters::add("conductingwall.reapplyUponRestart", "If 0 (default), keep going with the state existing in the restart file. If 1, calls again applyInitialState. Can be used to change boundary condition behaviour during a run.", 0);

      // Per-population parameters
      for (uint i = 0; i < getObjectWrapper().particleSpecies.size(); i++) {
         const string& pop = getObjectWrapper().particleSpecies[i].name;

         Readparameters::addComposing(pop + "_conductingwall.reapplyFaceUponRestart", "List of faces on which conducting-wall boundary conditions are to be reapplied upon restart ([xyz][+-]).");
         Readparameters::addComposing(pop + "_conductingwall.face", "List of faces on which conducting-wall boundary conditions are to be applied ([xyz][+-]).");
      }
   }

   void ConductingWall::getParameters() {
      Readparameters::get("conductingwall.faceNoFields", this->faceNoFieldsList);
      Readparameters::get("conductingwall.precedence", precedence);
      uint reapply;
      Readparameters::get("conductingwall.reapplyUponRestart", reapply);
      this->applyUponRestart = false;
      if (reapply == 1) {
         this->applyUponRestart = true;
      }

      // Per-species parameters
      for (uint i = 0; i < getObjectWrapper().particleSpecies.size(); i++) {
         const string& pop = getObjectWrapper().particleSpecies[i].name;
         ConductingWallSpeciesParameters sP;

         // Unless we find out otherwise, we assume that this species will not be treated at any boundary
         for (int j = 0; j < 6; j++) {
            sP.facesToSkipVlasov[j] = true;
         }

         vector<string> thisSpeciesFaceList;
         Readparameters::get(pop + "_conductingwall.face", thisSpeciesFaceList);

         for (auto& face : thisSpeciesFaceList) {
            if (face == "x+") {
               facesToProcess[0] = true;
               sP.facesToSkipVlasov[0] = false;
            }
            if (face == "x-") {
               facesToProcess[1] = true;
               sP.facesToSkipVlasov[1] = false;
            }
            if (face == "y+") {
               facesToProcess[2] = true;
               sP.facesToSkipVlasov[2] = false;
            }
            if (face == "y-") {
               facesToProcess[3] = true;
               sP.facesToSkipVlasov[3] = false;
            }
            if (face == "z+") {
               facesToProcess[4] = true;
               sP.facesToSkipVlasov[4] = false;
            }
            if (face == "z-") {
               facesToProcess[5] = true;
               sP.facesToSkipVlasov[5] = false;
            }
         }

         speciesParams.push_back(sP);
      }
   }

   void ConductingWall::initSysBoundary(creal& t, Project& project) {
      for (uint i = 0; i < 6; i++) {
         facesToProcess[i] = false;
         facesToSkipFields[i] = false;
      }

      this->getParameters();

      dynamic = false;

      for (auto& face : faceNoFieldsList) {
         if (face == "x+") facesToSkipFields[0] = true;
         if (face == "x-") facesToSkipFields[1] = true;
         if (face == "y+") facesToSkipFields[2] = true;
         if (face == "y-") facesToSkipFields[3] = true;
         if (face == "z+") facesToSkipFields[4] = true;
         if (face == "z-") facesToSkipFields[5] = true;
      }
   }

   void ConductingWall::applyInitialState(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                          fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                                          fsgrids::perbspan perb,
                                          fsgrids::bgbspan bgb, Project& project) {
      // At t=0 the wall cell is given the same state as any other cell at that location (e.g. the Harris
      // equilibrium value there); the reflective vlasovBoundaryCondition only takes over from the next step
      // onward, once particles actually interact with the wall.
      const vector<CellID>& cells = getLocalCells();
      #pragma omp parallel for schedule(static)
      for (uint i = 0; i < cells.size(); ++i) {
         CellID id = cells[i];
         SpatialCell* cell = mpiGrid[id];
         if (cell->sysBoundaryFlag != this->getIndex()) {
            continue;
         }
         project.setCell(cell);
         cell->parameters[CellParams::RHOM_DT2] = cell->parameters[CellParams::RHOM];
         cell->parameters[CellParams::RHOQ_DT2] = cell->parameters[CellParams::RHOQ];
         cell->parameters[CellParams::VX_DT2] = cell->parameters[CellParams::VX];
         cell->parameters[CellParams::VY_DT2] = cell->parameters[CellParams::VY];
         cell->parameters[CellParams::VZ_DT2] = cell->parameters[CellParams::VZ];
         cell->parameters[CellParams::P_11_DT2] = cell->parameters[CellParams::P_11];
         cell->parameters[CellParams::P_22_DT2] = cell->parameters[CellParams::P_22];
         cell->parameters[CellParams::P_33_DT2] = cell->parameters[CellParams::P_33];
      }
   }

   void ConductingWall::updateState(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                    fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                                    fsgrids::perbspan perb,
                                    fsgrids::bgbspan bgb, creal t) {}

   /*! Search strictly along the wall's own interior-facing axis (never across the wall, never into another
    * boundary layer) for the nearest NOT_SYSBOUNDARY cell, and return its B value for the requested component.
    * This deliberately does not reuse fieldBoundaryCopyFromSolvingNbrMagneticField, which searches the full
    * surrounding neighborhood and could otherwise pick up another wall-layer cell or something past the wall.
    * \param axis Axis normal to the wall (0=x,1=y,2=z).
    * \param dir Offset direction (-1 or +1) pointing from the wall into the domain interior along that axis.
    */
   Real ConductingWall::fieldBoundaryCopyFromInteriorMagneticField(
       fsgrids::perbspan b, fsgrids::consttechnicalspan technical,
       const fsgrid::FsStencil& stencil, cuint component, cuint axis, cint dir) {
      const uint mask = 1 << component;
      for (int step = 1; step <= 2; ++step) {
         int offset[3] = {0, 0, 0};
         offset[axis] = dir * step;
         if (!stencil.cellExists(offset[0], offset[1], offset[2])) {
            continue;
         }
         const auto index = stencil.indexFromOffset(offset[0], offset[1], offset[2]);
         const auto& tech = technical[index];
         if ((tech.SOLVE & mask) == mask && tech.sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
            return b[index][fsgrids::bfield::PERBX + component];
         }
      }
      abort_mpi("ConductingWall: no interior cell found along the wall normal for tangential B copy", 1);
      return 0.0; // unreachable
   }

   Real ConductingWall::fieldSolverBoundaryCondMagneticField(fsgrids::perbspan b,
                                                             fsgrids::constbgbspan bgb,
                                                             fsgrids::consttechnicalspan technical,
                                                             const std::array<Real, 3>& gridSpacing,
                                                             const std::array<fsgrid::FsSize_t, 3>& globalCoordinates,
                                                             const fsgrid::FsStencil& stencil, cuint component) {
      creal dx = gridSpacing[0];
      creal dy = gridSpacing[1];
      creal dz = gridSpacing[2];
      creal x = (convert<Real>(globalCoordinates[0]) + 0.5) * dx + Parameters::xmin;
      creal y = (convert<Real>(globalCoordinates[1]) + 0.5) * dy + Parameters::ymin;
      creal z = (convert<Real>(globalCoordinates[2]) + 0.5) * dz + Parameters::zmin;

      std::array<bool, 6> isThisCellOnAFace;
      determineFace(&isThisCellOnAFace[0], x, y, z, dx, dy, dz, true);

      for (uint face = 0; face < 6; ++face) {
         if (!isThisCellOnAFace[face] || !facesToProcess[face] || facesToSkipFields[face]) {
            continue;
         }
         const uint axis = faceNormalAxis(face);
         if (component == axis) {
            return 0.0; // zero normal B at a perfectly-conducting wall
         }
         return fieldBoundaryCopyFromInteriorMagneticField(b, technical, stencil, component, axis, faceInteriorOffset(face));
      }
      // Cell is flagged CONDUCTINGWALL but not on a processed face with fields enabled: leave as-is.
      return b[stencil.ooo()][fsgrids::bfield::PERBX + component];
   }

   void ConductingWall::fieldSolverBoundaryCondElectricField(fsgrids::efieldspan e,
                                                             const fsgrid::FsStencil& stencil, cuint component) {
      e[stencil.ooo()][fsgrids::efield::EX + component] = 0.0;
   }

   void ConductingWall::fieldSolverBoundaryCondHallElectricField(fsgrids::ehallspan ehall,
                                                                  const fsgrid::FsStencil& stencil, cuint component) {
      // Hall term is not used by the AP field solver; zeroed for interface completeness only, matching Outflow.
      array<Real, fsgrids::ehall::N_EHALL>& cp = ehall[stencil.ooo()];
      switch (component) {
      case 0:
         cp[fsgrids::ehall::EXHALL_000_100] = 0.0;
         cp[fsgrids::ehall::EXHALL_010_110] = 0.0;
         cp[fsgrids::ehall::EXHALL_001_101] = 0.0;
         cp[fsgrids::ehall::EXHALL_011_111] = 0.0;
         break;
      case 1:
         cp[fsgrids::ehall::EYHALL_000_010] = 0.0;
         cp[fsgrids::ehall::EYHALL_100_110] = 0.0;
         cp[fsgrids::ehall::EYHALL_001_011] = 0.0;
         cp[fsgrids::ehall::EYHALL_101_111] = 0.0;
         break;
      case 2:
         cp[fsgrids::ehall::EZHALL_000_001] = 0.0;
         cp[fsgrids::ehall::EZHALL_100_101] = 0.0;
         cp[fsgrids::ehall::EZHALL_010_011] = 0.0;
         cp[fsgrids::ehall::EZHALL_110_111] = 0.0;
         break;
      default:
         cerr << __FILE__ << ":" << __LINE__ << ":" << " Invalid component" << endl;
      }
   }

   void ConductingWall::fieldSolverBoundaryCondGradPeElectricField(
       fsgrids::egradpespan EGradPe, const fsgrid::FsStencil& stencil,
       cuint component) {
      // Electron pressure gradient term is not used by the AP field solver; zeroed for interface completeness.
      EGradPe[stencil.ooo()][fsgrids::egradpe::EXGRADPE + component] = 0.0;
   }

   void ConductingWall::fieldSolverBoundaryCondDerivatives(fsgrids::dperbspan dperb,
                                                           fsgrids::dmomentsspan dmoments,
                                                           const fsgrid::FsStencil& stencil, cuint RKCase, cuint component) {
      this->setCellDerivativesToZero(dperb, dmoments, stencil, component);
   }

   void ConductingWall::fieldSolverBoundaryCondBVOLDerivatives(fsgrids::volspan vols,
                                                               const fsgrid::FsStencil& stencil, cuint component) {
      this->setCellBVOLDerivativesToZero(vols, stencil, component);
   }

   /*! Search strictly along (axis,dir) from cellID, up to two cells deep (matching the field solver's stencil
    * depth, and the maximum sysBoundaryLayer depth in normal use), for the nearest NOT_SYSBOUNDARY neighbor.
    * Written as a direct search rather than relying on the existing allClosestNonsysboundaryCells cache,
    * because that cache is only populated for sysBoundaryLayer==1 cells (see
    * updateSysBoundaryConditionsAfterLoadBalance), and because restricting the direction explicitly avoids
    * ever reading across the wall or from another boundary layer.
    */
   CellID ConductingWall::findInteriorNeighborAlongAxis(
       const dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
       const CellID& cellID, uint axis, int dir) {
      // Neighborhoods::SYSBOUNDARIES is only +-1 cell wide (grid.cpp:initializeStencils); a
      // layer-2 wall cell's step=2 interior neighbor is not reachable through it at all. Use
      // SYSBOUNDARIES_EXTENDED (+-2 wide), already kept up to date immediately before
      // vlasovBoundaryCondition is called (see SysBoundary::applySysBoundaryVlasovConditions).
      for (int step = 1; step <= 2; ++step) {
         for (const auto& nbrPair : *mpiGrid.get_neighbors_of(cellID, Neighborhoods::SYSBOUNDARIES_EXTENDED)) {
            if (nbrPair.first == INVALID_CELLID) {
               continue;
            }
            bool matchesOffset = (nbrPair.second[axis] == dir * step);
            for (uint otherAxis = 0; otherAxis < 3; ++otherAxis) {
               if (otherAxis != axis && nbrPair.second[otherAxis] != 0) {
                  matchesOffset = false;
               }
            }
            if (matchesOffset && mpiGrid[nbrPair.first]->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
               return nbrPair.first;
            }
         }
      }
      abort_mpi("ConductingWall: no interior cell found along the wall normal for VDF reflection", 1);
      return INVALID_CELLID; // unreachable
   }

   /*!
    * Specular reflection: mirror the wall-normal velocity component of the nearest interior cell's VDF
    * directly on the velocity grid. Because the velocity mesh is uniform and symmetric about v=0 along the
    * wall-normal axis (as required of any species used with this boundary), mirroring is an exact block and
    * within-block index flip, not an interpolation or an assumption about the distribution's shape.
    *
    * NOTE that this is called once for each particle species, and modifies only this cell's own block
    * structure (adding blocks that mirror the neighbor's populated ones), which is safe: vlasovBoundaryCondition
    * is only ever invoked for boundary-flagged cells, so the interior neighbor read here is never itself being
    * concurrently modified by another thread's call to this function.
    */
   void ConductingWall::vlasovBoundaryCondition(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                                const CellID& cellID, const uint popID, const bool calculate_V_moments) {
      SpatialCell* thisCell = mpiGrid[cellID];
      if (thisCell->sysBoundaryFlag != this->getIndex()) {
         return;
      }
      const ConductingWallSpeciesParameters& sP = this->speciesParams[popID];

      std::array<bool, 6> isThisCellOnAFace;
      determineFace(isThisCellOnAFace, mpiGrid, cellID, true);

      for (uint face = 0; face < 6; ++face) {
         if (!isThisCellOnAFace[face] || !facesToProcess[face] || sP.facesToSkipVlasov[face]) {
            continue;
         }
         const uint axis = faceNormalAxis(face);
         const int dir = faceInteriorOffset(face);

         const CellID neighborID = findInteriorNeighborAlongAxis(mpiGrid, cellID, axis, dir);
         SpatialCell* neighborCell = mpiGrid[neighborID];

         thisCell->clear(popID, false); // drop old content, keep allocated capacity

         vmesh::VelocityMesh* vmesh = neighborCell->get_velocity_mesh(popID); // mesh geometry is shared across cells
         const vmesh::LocalID* gridLength = vmesh->getGridLength();

         const vmesh::LocalID nBlocks = neighborCell->get_number_of_velocity_blocks(popID);
         for (vmesh::LocalID srcLID = 0; srcLID < nBlocks; ++srcLID) {
            const vmesh::GlobalID srcGID = neighborCell->get_velocity_block_global_id(srcLID, popID);
            vmesh::LocalID bi, bj, bk;
            vmesh->getIndices(srcGID, bi, bj, bk);
            vmesh::LocalID bIdx[3] = {bi, bj, bk};
            bIdx[axis] = gridLength[axis] - 1 - bIdx[axis];
            const vmesh::GlobalID dstGID = vmesh->getGlobalID(bIdx[0], bIdx[1], bIdx[2]);

            thisCell->add_velocity_block(dstGID, popID);
            const vmesh::LocalID dstLID = thisCell->get_velocity_block_local_id(dstGID, popID);

            const Realf* srcData = neighborCell->get_data(srcLID, popID);
            Realf* dstData = thisCell->get_data(dstLID, popID);

            for (uint k = 0; k < WID; ++k) {
               for (uint j = 0; j < WID; ++j) {
                  for (uint i = 0; i < WID; ++i) {
                     uint mi = i, mj = j, mk = k;
                     if (axis == 0) { mi = WID - 1 - i; }
                     else if (axis == 1) { mj = WID - 1 - j; }
                     else { mk = WID - 1 - k; }
                     dstData[k * WID2 + j * WID + i] = srcData[mk * WID2 + mj * WID + mi];
                  }
               }
            }
         }
         thisCell->adjustSingleCellVelocityBlocks(popID, true);
         // Moments are recomputed externally (calculateMoments_V/_R) after this function returns for every
         // boundary cell; no need to set them here. A single ConductingWall face is assumed per cell (see
         // class doc comment on corner cells), so stop after the first match.
         return;
      }
   }

   void ConductingWall::getFaces(bool* faces) {
      for (uint i = 0; i < 6; i++) {
         faces[i] = facesToProcess[i];
      }
   }

   string ConductingWall::getName() const { return "ConductingWall"; }
   uint ConductingWall::getIndex() const { return sysboundarytype::CONDUCTINGWALL; }

} // namespace SBC
