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

#ifndef FS_COMMON_H
#define FS_COMMON_H

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <list>
#include <set>
#include <stdint.h>

#include <fsgrid.hpp>
#include <phiprof.hpp>

#include "../definitions.h"
#include "../common.h"
#include "../parameters.h"
#include "../projects/project.h"
#include "../sysboundary/sysboundary.h"
#include "../sysboundary/sysboundarycondition.h"

#include "fs_constants.h"

using namespace std;

bool propagateFields(fsgrids::perbspan perb,
                     fsgrids::perbspan perbdt2,
                     fsgrids::efieldspan e,
                     fsgrids::efieldspan edt2,
                     fsgrids::ehallspan ehall,
                     fsgrids::egradpespan egradpe,
                     fsgrids::egradpespan egradpedt2,
#ifdef FS_ES
                     // electrostatic field and potential
                     fsgrids::efieldspan e_es,
                     fsgrids::potentialspan Phi,
#endif
                     fsgrids::momentsspan moments,
                     fsgrids::momentsspan momentsdt2,
#ifdef FS_AP
                     // per-species rho_s^k and J_s^{k*}, one span entry per
                     // population
                     std::vector<fsgrids::speciesrhoqspan>& speciesRhoQ,
                     std::vector<fsgrids::speciesjspan>& speciesJ,
                     // FIXME do we need a separate electric field here to, same as in the electrostatic case since the acceleration step does NOT take the regular efield span?
#endif
                     fsgrids::dperbspan dperb,
                     fsgrids::dmomentsspan dmoments,
                     fsgrids::dmomentsspan dmomentsdt2,
                     fsgrids::bgbspan bgb,
                     fsgrids::volspan vol,
                     fsgrids::technicalspan technical, FieldSolverGrid &fsgrid, SysBoundary& sysBoundaries,
                     creal& dt, cuint subcycles);

Real divideIfNonZero(creal rhoV, creal rho);

/*! Namespace encompassing the enum defining the list of reconstruction coefficients used in field component reconstructions.*/
namespace Rec {
   /*! Enum defining the list of reconstruction coefficients used in field component reconstructions.*/
   enum Rec {
      a_0, a_x, a_y, a_z, a_xx, a_yy, a_zz, a_xy, a_xz, a_yz, a_xxx, a_xxy, a_xyy, a_xxz, a_xzz, a_xyz,
      b_0, b_x, b_y, b_z, b_xx, b_yy, b_zz, b_xy, b_xz, b_yz, b_xxy, b_xyy, b_yyy, b_yyz, b_yzz, b_xyz,
      c_0, c_x, c_y, c_z, c_xx, c_yy, c_zz, c_xy, c_xz, c_yz, c_xxz, c_xzz, c_yyz, c_yzz, c_xyz, c_zzz,
      N_REC_COEFFICIENTS
   };
}

std::array<Real, Rec::N_REC_COEFFICIENTS>
reconstructionCoefficients(fsgrids::perbspan perb,
                           fsgrids::constdperbspan dperb,
                           const fsgrid::FsStencil& stencil, Real reconstructionOrder);

std::array<Real, 3> interpolatePerturbedB(
   fsgrids::perbspan perb,
   fsgrids::constdperbspan dperb,
   fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
   std::map<std::array<int, 3>, std::array<Real, Rec::N_REC_COEFFICIENTS>>& reconstructionCoefficientsCache,
   cint i,
   cint j,
   cint k,
   const std::array<Real, 3> x
);

std::array<Real, 3> interpolateCurlB(
   fsgrids::perbspan perb,
   fsgrids::constdperbspan dperb,
   fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
   std::map<std::array<int, 3>, std::array<Real, Rec::N_REC_COEFFICIENTS>>& reconstructionCoefficientsCache,
   cint i,
   cint j,
   cint k,
   const std::array<Real, 3> x
);

#endif
