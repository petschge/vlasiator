#ifndef ES_ELECTRIC_FIELD_HPP
#define ES_ELECTRIC_FIELD_HPP

#include "fs_common.h"

void es_calculateElectricFieldX(fsgrids::efieldspan e,
                                   fsgrids::potentialspan Phi,
                                   const fsgrid::FsStencil& stencil,
                                   const std::array<Real, 3>& gridSpacing);

void es_calculateElectricFieldY(fsgrids::efieldspan e,
                                   fsgrids::potentialspan Phi,
                                   const fsgrid::FsStencil& stencil,
                                   const std::array<Real, 3>& gridSpacing);

void es_calculateElectricFieldZ(fsgrids::efieldspan e,
                                   fsgrids::potentialspan Phi,
                                   const fsgrid::FsStencil& stencil,
                                   const std::array<Real, 3>& gridSpacing);

void es_ElectrostaticPotential(fsgrids::potentialspan Phi,
                               fsgrids::momentsspan moments,
                               fsgrids::technicalspan technical,
                               FieldSolverGrid &fsgrid);

void calculateElectrostaticField(fsgrids::efieldspan e,
                                 fsgrids::potentialspan Phi,
                                 fsgrids::momentsspan moments,
                                 fsgrids::technicalspan technical,
                                 FieldSolverGrid &fsgrid,
                                 SysBoundary& sysBoundaries);

#endif
