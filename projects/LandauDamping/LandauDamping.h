/*
 * This file is part of Vlasiator.
 *
 * LandauDamping project: reproduces the 1D-space, 2D-velocity (1d2v)
 * linear/nonlinear Landau damping test of Liu, Cai, Cao & Lapenta,
 * "An asymptotic-preserving conservative semi-Lagrangian scheme for the
 * Vlasov-Maxwell system in the quasi-neutral limit", J. Comput. Phys.
 * 528 (2025) 113840, Section 5.1 / Eq. (51):
 *
 *   f_e(x,vx,vy,0) = (1/(pi*beta)) * exp[-(vx^2+vy^2)/beta] * (1 + alpha0*sin(k*x))
 *
 * with a uniform, static (very heavy, cold, unperturbed) ion background
 * providing overall charge neutrality -- the paper's own "ions form a
 * uniform static background, only electron dynamics considered" (Section
 * 5.1/5.2). Unlike Dispersion/MultiPeak/Fluctuations/Distributions, none
 * of which support a DETERMINISTIC single-mode sinusoidal density
 * perturbation (all use either random per-cell noise, or -- MultiPeak's
 * "testcase" -- a hardcoded rectangular patch), this project adds that
 * specific, minimal capability, parametrized by a single global
 * wavenumber k and a per-species relative perturbation amplitude
 * (densityPertRelAmp; the background species sets this to 0).
 *
 * setProjectEField reuses Dispersion's own approach verbatim: solve
 * -eps0*grad^2(Phi)=rho(x,0), E=-grad(Phi) via the ES field solver's own
 * Poisson machinery, giving the same "initial Ex solved from the Poisson
 * equation" precondition the paper's own Section 2.1/5.1 requires
 * (lambda^2*Ex(x,0)=1-n_e(x) is exactly this Poisson solve's dimensional
 * equivalent, solved numerically here rather than hand-derived
 * analytically -- matching what the paper's own reference "solve for Ex
 * using the Poisson equation" solution, CSL-PE, does).
 */

#ifndef LANDAUDAMPING_H
#define LANDAUDAMPING_H

#include <stdlib.h>

#include "../../definitions.h"
#include "../project.h"

namespace projects {

   struct LandauDampingSpeciesParameters {
      Real VX0;
      Real VY0;
      Real VZ0;
      Real DENSITY;
      Real TEMPERATURE;
      Real densityPertRelAmp;
   };

   class LandauDamping: public Project {
    public:
      LandauDamping();
      virtual ~LandauDamping();

      virtual bool initialize(void) override;
      static void addParameters(void);
      virtual void getParameters(void) override;
      virtual void setProjectBField(
         fsgrids::perbspan perb, fsgrids::bgbspan bgb,
         fsgrids::technicalspan technical, FieldSolverGrid& fsgrid
      ) override;
      virtual void setProjectEField(
         fsgrids::momentsspan moments, fsgrids::efieldspan e,
         fsgrids::technicalspan technical, FieldSolverGrid& fsgrid
      ) const override;
      virtual Realf fillPhaseSpace(
         spatial_cell::SpatialCell* cell, const uint popID,
         const uint nRequested
      ) const override;
      // Base class Project::calcCellParameters is NOT a safe no-op
      // default (it's a print-warning-and-exit(1) stub, confirmed
      // directly against projects/project.cpp -- every project must
      // provide its own override). This project needs no per-cell setup
      // at all (no random state, nothing precomputed outside
      // fillPhaseSpace, which already gets everything it needs directly
      // from cell->parameters itself) -- an empty override is both
      // sufficient and an established, working pattern elsewhere in the
      // codebase (Riemann1::calcCellParameters is likewise empty).
      virtual void calcCellParameters(spatial_cell::SpatialCell* cell, creal& t) override;

      // Global (not per-species): the density perturbation's wavenumber,
      // shared by whichever species has a nonzero densityPertRelAmp.
      Real k;

      std::vector<LandauDampingSpeciesParameters> speciesParams;
   } ; // class LandauDamping
} // namespace projects

#endif
