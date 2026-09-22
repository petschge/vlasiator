/*! \file es_main.cpp
 * \brief A simple and stupid, explicit, spectral, first-order, electrostatic solver
 *
 */

#include "fs_common.h"
#include "../logger.h"
#include "es_electric_field.hpp"
#include "ldz_magnetic_field.hpp"
#include "derivatives.hpp"
#include "volume_averages.hpp"
#include "../fieldtracing/fieldtracing.h"

/*! \brief Top-level field propagation function.
 *
 * \param e_es fsgrid holding electrostatic field
 * \param moments fsgrid holding moments
 * \param momentsdt2 fsgrid holding moments at Runge-Kutta half-step
 * \param technical fsgrid holding technical parameters
 * \param fsgrid fsgrids container
 * \param sysBoundaries Container of existing system boundaries
 * \param dt Length of the time step
 * \param subcycles Number of subcycles to compute.
 *
 */
bool es_propagateFields(fsgrids::efieldspan e_es,
#ifdef FS_ES
                        fsgrids::potentialspan Phi,
#endif
                        fsgrids::momentsspan moments,
                        fsgrids::momentsspan momentsdt2,
                        fsgrids::technicalspan technical,
                        FieldSolverGrid &fsgrid,
                        SysBoundary& sysBoundaries,
                        creal& dt,
                        cuint subcycles) {

   if (subcycles != 1) {
      cerr << "Elestrostatic field solver subcycles have to be 1." << endl;
      exit(1);
   }

   const auto* localSize = &fsgrid.getLocalSize()[0];

   fsgrid.parallel_for([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                       phiprof::initializeTimer("Initialize technical.maxFsDt"), technical,
                       [=](const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
                          // technical[stencil.ooo()].maxFsDt = std::numeric_limits<Real>::max();

                          // We would want to loop over all species, get their
                          // number densities, compute the per species plasma
                          // frequencies, add them quadratically and limit the
                          // timestep relative to that. But that requires per
                          // species number densities that we don't have on the
                          // fsgrid. So let's make a crute approximation. If we
                          // divide the total mass density by the electron mass
                          // we'll get a (not particularily tight) upper limit
                          // on the number density of electrons. Those are
                          // typically the lightest species and will dominate
                          // the overall plasma frequency. Since we're
                          // overestimating their number by quite a bit the
                          // real plasma frequency is likely well below the
                          // electron plasma frequency based on that
                          // overestimated electron number density.
                          Real n_approx = momentsdt2[stencil.ooo()][fsgrids::moments::RHOM] / physicalconstants::MASS_ELECTRON;

                          // compute plasma frequency
                          Real wp = sqrt(n_approx*physicalconstants::CHARGE*physicalconstants::CHARGE / (physicalconstants::MASS_ELECTRON * physicalconstants::EPS_0));

                          // limit timestep to wp*dt ~ 1
                          technical[stencil.ooo()].maxFsDt = 1./wp;
                       });

   calculateElectrostaticField(
      e_es,
      Phi,
      momentsdt2,
      technical,
      fsgrid,
      sysBoundaries
   );

   //calculateVolumeAveragedFieldsSimple(perb, e, dperb, vol, technical, fsgrid);
   //calculateVolumeAveragedFieldsSimple(perb, edt2, dperb, vol, technical, fsgrid);
   //calculateBVOLDerivativesSimple(vol, technical, fsgrid);
   //if (FieldTracing::fieldTracingParameters.doTraceFullBox || Parameters::computeCurvature) {
   //   fsgrid.updateGhostCells(vol);
   //   calculateCurvatureSimple(vol, bgb, technical, fsgrid);
   //}

   return true;
}
