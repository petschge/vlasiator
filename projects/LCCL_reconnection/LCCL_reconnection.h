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

#ifndef LCCL_RECONNECTION_H
#define LCCL_RECONNECTION_H

#include "../../definitions.h"
#include "../projectTriAxisSearch.h"

namespace projects {

   struct LCCL_ReconnectionSpeciesParameters {
      Real rho;
      //Real T;
      Real thermalSpeed;
      Real driftSpeed;
   };

   class LCCL_Reconnection: public TriAxisSearch {
    public:
      LCCL_Reconnection();
      virtual ~LCCL_Reconnection();
      
      virtual bool initialize(void) override;
      virtual void addParameters(void) override;
      virtual void getParameters(void) override;
      virtual void setProjectBField(
         fsgrids::perbspan perb,
         fsgrids::bgbspan bgb,
         fsgrids::technicalspan technical, FieldSolverGrid &fsgrid
      ) override;

      virtual Realf fillPhaseSpace(spatial_cell::SpatialCell *cell,
                                  const uint popID,
                                  const uint nRequested) const override;
      virtual Realf probePhaseSpace(spatial_cell::SpatialCell* cell,
                                    const uint popID,
                                    Real vx_in, Real vy_in, Real vz_in
                                   ) const override;

      virtual void calcCellParameters(spatial_cell::SpatialCell* cell, creal& t) override;
      virtual std::vector<std::array<Real, 3>> getV0(creal x, creal y, creal z, const uint popID) const override;

      Real B0;
      Real ionInertialLength;
      bool useDoubleCS;
      std::vector<LCCL_ReconnectionSpeciesParameters> speciesParams;
      std::vector<LCCL_ReconnectionSpeciesParameters*> speciesParamsRead;
   } ; // class LCCL_Reconnection
} // namespace projects

#endif