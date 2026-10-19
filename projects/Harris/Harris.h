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

#ifndef HARRIS_H
#define HARRIS_H

#include "../../definitions.h"
#include "../projectTriAxisSearch.h"

namespace projects {

   struct HarrisSpeciesParameters {
      Real TEMPERATURE;
      Real DENSITY;
      Real RHO_BACKGROUND; /*!< Background (non-drifting) number density added to the Harris-sheet
                                 population(s), same temperature. Default 0 (no background). */
      Real DRIFT_V;        /*!< z-directed bulk drift magnitude of a single Harris sheet, computed from
                                 Ampere's law applied to the Harris equilibrium (see getParameters()).
                                 For doubleSheet, sheet 1 (at y=-L/2) uses +DRIFT_V and sheet 2 (at
                                 y=+L/2, opposite polarity) uses -DRIFT_V -- see getPopulationsAt(). */
   };

   /*! One drifting-Maxwellian term (v0 along z, density, temperature) contributing to the total VDF at
    * a given y. Shared by fillPhaseSpace, probePhaseSpace, getV0 and findBlocksToInitialize so the list
    * of populations summed is defined in exactly one place (see getPopulationsAt()) -- single-sheet
    * returns {sheet, [background]}, doubleSheet returns {sheet1, sheet2, [background]}. */
   struct HarrisPopDesc {
      Real v0z;
      Real rho;
      Real T;
   };

   class Harris: public TriAxisSearch {
   public:
      Harris();
      virtual ~Harris();

      virtual bool initialize(void) override;
      static void addParameters(void);
      virtual void getParameters(void) override;
      virtual void calcCellParameters(spatial_cell::SpatialCell* cell, creal& t) override;
      virtual void setProjectBField(fsgrids::perbspan perb,
                                    fsgrids::bgbspan bgb,
                                    fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) override;

      virtual Realf fillPhaseSpace(spatial_cell::SpatialCell* cell, const uint popID,
                                   const uint nRequested) const override;
      virtual Realf probePhaseSpace(spatial_cell::SpatialCell* cell, const uint popID, Real vx_in, Real vy_in,
                                    Real vz_in) const override;
      virtual std::vector<std::array<Real, 3>> getV0(creal x, creal y, creal z, const uint popID) const override;
      virtual uint findBlocksToInitialize(spatial_cell::SpatialCell* cell, const uint popID) const override;

      Real SCA_LAMBDA;
      Real BX0, BY0, BZ0;
      Real PERT_AMPLITUDE; /*!< Amplitude phi0 of an optional magnetic island perturbation. Single-sheet:
                                 delta_B = grad(phi) x e_z, phi=phi0*cos(pi*x/Lx)*cos(pi*y/2Ly). doubleSheet:
                                 phi=-phi0*cos(pi*x/Lx)*sin(pi*y/L), X-line at x=0 on both sheets -- see
                                 setProjectBField(). Default 0
                                 (no perturbation) either way. */
      bool DOUBLE_SHEET;   /*!< If true, two Harris sheets of opposite polarity at y=+-L/2 (L = domain
                                 half-height), fully periodic in y -- see getPopulationsAt() and
                                 setProjectBField(). If false (default), the original single sheet. */
      int CURRENT_LOADING; /*!< 0: analytic drifts; 1: drifts scaled so the loaded current equals the AP
                                 solver's discrete curl(B)/mu0; 2: also the in-sheet share of the
                                 perturbation's current (doubleSheet) -- see getPopulationsAt(). */
      Real WALL_HALF_HEIGHT; /*!< single sheet: sheet-to-wall distance used by the perturbation; 0 = domain half-height */
      Real SEED_AMPLITUDE; /*!< symmetry-breaking seed, relative to PERT_AMPLITUDE -- see setProjectBField() */
      Real SEED_WIDTH;     /*!< its Gaussian width, m */
      Real SHEET_CURRENT;  /*!< sum over species of q*DENSITY*DRIFT_V (A/m^2) */
      Real DENSITY_SUM;    /*!< sum over species of DENSITY */
      Real BACKGROUND_SUM; /*!< sum over species of RHO_BACKGROUND */
      std::vector<HarrisSpeciesParameters> speciesParams;

      /*! All (v0z, rho, T) Maxwellian terms contributing to population popID's VDF at the spatial cell
       * centre (x, y). See the HarrisPopDesc comment above for why this is centralised here rather than
       * duplicated per caller. */
      std::vector<HarrisPopDesc> getPopulationsAt(Real x, Real y, const uint popID) const;
      Real discreteCurlFactor(Real u) const;
      Real perturbationDiscreteCurrent(Real x, Real y) const;

   }; // class Harris
} // namespace Harris

#endif
