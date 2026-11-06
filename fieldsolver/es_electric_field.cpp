#include "fs_common.h"
#include "es_electric_field.hpp"
#include <HYPRE.h>
#include <HYPRE_sstruct_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_krylov.h>
#include <cmath>
#include <vector>
#include "es_poisson_periodic.hpp"

#ifdef DEBUG_VLASIATOR
#define DEBUG_FSOLVER
#endif

namespace pc = physicalconstants;
using namespace std;

/*! \brief Electric field computation function.
 *
 * Take a partial derivative of the potential in the x direction to get the
 * electric field component in the x direction at interior grid cells
 *
 * \param e_es fsGrid holding the electric field
 * \param Phi fsGrid holding the electric potential
 * \param stencil current cell's fsgrid stencil
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 */
void es_calculateElectricFieldX(fsgrids::efieldspan e_es, fsgrids::potentialspan Phi,
                                    const fsgrid::FsStencil& stencil,
                                    const std::array<Real, 3>& gridSpacing) {

   e_es[stencil.ooo()][fsgrids::efield::EX] = -(Phi[stencil.poo()][fsgrids::potential::PHI]-Phi[stencil.moo()][fsgrids::potential::PHI])/(2.*gridSpacing[0]);
}

/*! \brief Electric field computation function.
 *
 * Take a partial derivative of the potential in the y direction to get the
 * electric field component in the y direction at interior grid cells
 *
 * \param e_es fsGrid holding the electric field
 * \param Phi fsGrid holding the electric potential
 * \param stencil current cell's fsgrid stencil
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 */
void es_calculateElectricFieldY(fsgrids::efieldspan e_es, fsgrids::potentialspan Phi,
                                    const fsgrid::FsStencil& stencil,
                                    const std::array<Real, 3>& gridSpacing) {
   e_es[stencil.ooo()][fsgrids::efield::EY] = -(Phi[stencil.opo()][fsgrids::potential::PHI]-Phi[stencil.omo()][fsgrids::potential::PHI])/(2.*gridSpacing[1]);
}

/*! \brief Electric field computation function.
 *
 * Take a partial derivative of the potential in the z direction to get the
 * electric field component in the z direction at interior grid cells
 *
 * \param e_es fsGrid holding the electric field
 * \param Phi fsGrid holding the electric potential
 * \param stencil current cell's fsgrid stencil
 * \param gridSpacing fsgrid cell size in x,y,z
 *
 */
void es_calculateElectricFieldZ(fsgrids::efieldspan e_es, fsgrids::potentialspan Phi,
                                    const fsgrid::FsStencil& stencil,
                                    const std::array<Real, 3>& gridSpacing) {

   e_es[stencil.ooo()][fsgrids::efield::EZ] = -(Phi[stencil.oop()][fsgrids::potential::PHI]-Phi[stencil.oom()][fsgrids::potential::PHI])/(2.*gridSpacing[2]);
}

/*! \brief Electric field computation function.
 *
 * Calls the general or the system boundary electric field propagation functions.
 *
 * \param e fsGrid holding the electric field
 * \param Phi fsGrid holding the electric potential
 * \param technical fsGrid holding technical information (such as boundary types)
 * \param stencil current cell's fsgrid stencil
 * \param gridSpacing fsgrid cell size in x,y,z
 * \param sysBoundaries System boundary conditions existing
 *
 */
void es_calculateElectricField(fsgrids::efieldspan e_es, fsgrids::potentialspan Phi,
                               fsgrids::technicalspan technical,
                               const fsgrid::FsStencil& stencil,
                               const FieldSolverGrid &fsgrid,
                               const std::array<Real, 3>& gridSpacing, SysBoundary& sysBoundaries) {
   cuint cellSysBoundaryFlag = technical[stencil.ooo()].sysBoundaryFlag;
   cuint bitfield = technical[stencil.ooo()].SOLVE;

   if (cellSysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE ||
       cellSysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) {
      return;
   }

   if ((bitfield & compute::EX) == compute::EX) {
       es_calculateElectricFieldX(e_es, Phi, stencil, gridSpacing);
   } else {
       sysBoundaries.getSysBoundary(cellSysBoundaryFlag)->fieldSolverBoundaryCondElectricField(e_es, stencil, 0);
   }

   if ((bitfield & compute::EY) == compute::EY) {
       es_calculateElectricFieldY(e_es, Phi, stencil, gridSpacing);
   } else {
      sysBoundaries.getSysBoundary(cellSysBoundaryFlag)->fieldSolverBoundaryCondElectricField(e_es, stencil, 1);
   }

   if ((bitfield & compute::EZ) == compute::EZ) {
       es_calculateElectricFieldZ(e_es, Phi, stencil, gridSpacing);
   } else {
      sysBoundaries.getSysBoundary(cellSysBoundaryFlag)->fieldSolverBoundaryCondElectricField(e_es, stencil, 2);
   }
}

/*! \brief High-level electric potential computation function.
 *
 * Solves a Poisson equation to get the electric potential at corners of grid cells
 *
 * \param Phi fsgrid holding electric potential
 * \param moments fsgrid holding moments
 * \param technical fsGrid holding technical information (such as boundary types)
 * \param fsgrid fsgrids container
 */
void es_ElectrostaticPotential(fsgrids::potentialspan Phi,
                               fsgrids::momentsspan moments,
                               fsgrids::technicalspan technical,
                               FieldSolverGrid &fsgrid) {

   // Periodic Poisson problem -eps0 lap(Phi) = rho - mean(rho) with the compact 7-point Laplacian, solved by
   // es_SolvePoissonPeriodic (es_poisson_periodic.hpp): PCG with BoomerAMG, which unlike PFMG does not need
   // periodic extents that are powers of two. The null space (constants) is handled there.

   const auto& globalSize = fsgrid.getGlobalSize();
   const auto& localStart = fsgrid.getLocalStart();
   const auto& localSize  = fsgrid.getLocalSize();
   const auto  dxyz       = fsgrid.getGridSpacing();

   const int lx = localSize[0];
   const int ly = localSize[1];
   const int lz = localSize[2];
   const long long nlocal = (long long)lx * ly * lz;

   std::vector<double> rho(nlocal, 0.0);
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("Extract net charge"),
      technical,

      [&rho,moments,lx,ly]
      (const fsgrid::Coordinates &coordinates,
       const fsgrid::FsStencil& stencil,
       cuint sysBoundaryFlag,
       cuint sysBoundaryLayer
      ) {
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);
         rho[lidx] = moments[stencil.ooo()][fsgrids::moments::RHOQ];
      });

   // Monitor charge neutrality: the mean of rho is subtracted below (in a periodic domain only the
   // fluctuating part has a potential), but a large imbalance usually means a set-up error.
   double sumRho_local = 0.0, sumAbsRho_local = 0.0;
   for (long long li = 0; li < nlocal; ++li) {
      sumRho_local    += rho[li];
      sumAbsRho_local += std::abs(rho[li]);
   }
   double reduceLocal[2]  = { sumRho_local, sumAbsRho_local };
   double reduceGlobal[2] = { 0.0, 0.0 };
   MPI_Allreduce(reduceLocal, reduceGlobal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   constexpr double neutralityWarnThreshold = 1e-3;
   const double relativeImbalance = reduceGlobal[1] > 0.0 ? std::abs(reduceGlobal[0]) / reduceGlobal[1] : 0.0;
   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   if (relativeImbalance > neutralityWarnThreshold && myRank == MASTER_RANK) {
      fprintf(stderr, "es_ElectrostaticPotential: net charge: sum(rho)=%e, relative imbalance |sum(rho)|/sum(|rho|)=%e "
                      "(above %e); its mean is removed before the solve\n",
              reduceGlobal[0], relativeImbalance, neutralityWarnThreshold);
   }

   const std::array<long long,3> gsize = { (long long)globalSize[0], (long long)globalSize[1], (long long)globalSize[2] };
   const std::array<long long,3> lstart = { (long long)localStart[0], (long long)localStart[1], (long long)localStart[2] };
   const std::array<long long,3> lsize = { (long long)lx, (long long)ly, (long long)lz };
   const std::array<double,3> h = { (double)dxyz[0], (double)dxyz[1], (double)dxyz[2] };
   std::vector<double> phiLocal(nlocal, 0.0);
   double relres = 0.0;
   int its = 0;
   es_SolvePoissonPeriodic(gsize, lstart, lsize, h, rho, physicalconstants::EPS_0, phiLocal, relres, its);
   if (myRank == MASTER_RANK) {
      fprintf(stderr, "es_ElectrostaticPotential: PCG/BoomerAMG finished with relative residual %e after %d iterations%s\n",
              relres, its, relres > 1e-8 ? " -- NOT CONVERGED, the initial E is inaccurate" : "");
   }

   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("set local values of Phi"),
      technical,

      [&Phi,&phiLocal,lx,ly]
      (const fsgrid::Coordinates &coordinates,
       const fsgrid::FsStencil& stencil,
       cuint sysBoundaryFlag,
       cuint sysBoundaryLayer
      ) {
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);
         const auto lid = stencil.ooo();
         Phi[lid][fsgrids::PHI] = phiLocal[lidx];
      });

   fsgrid.updateGhostCells(Phi);
}

/*! \brief High-level electric field computation function.
 *
 * Computes the potential and the calculates the electric fields from finite differences
 *
 * \param e_es fsGrid holding the electric field quantities
 * \param Phi fsGrid holding the elctric potential
 * \param moments fsgrid holding moments
 * \param technical fsGrid holding technical information (such as boundary types)
 * \param fsgrid fsgrids container
 * \param sysBoundaries System boundary conditions existing
 *
 * \sa ldz_calculateElectricField
 */
void calculateElectrostaticField(fsgrids::efieldspan e_es, fsgrids::potentialspan Phi,
                                 fsgrids::momentsspan moments, fsgrids::technicalspan technical,
                                 FieldSolverGrid &fsgrid, SysBoundary& sysBoundaries) {
   const size_t numCells = fsgrid.getNumCells();

   phiprof::Timer ETimer{"Calculate electrostatic field"};

   phiprof::Timer mpiTimer{"Electric field ghost updates MPI", {"MPI"}};

   // compute electrostatic potential at the cell centers from the charge densitites in momemts
   es_ElectrostaticPotential(Phi, moments, technical, fsgrid);

   mpiTimer.stop();
   // Calculate electric field
   fsgrid.parallel_for([](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
                       phiprof::initializeTimer("Electric field compute cells"), technical,
                       [=, &fsgrid,&sysBoundaries](const fsgrid::Coordinates &coordinates, const fsgrid::FsStencil& stencil, cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
                          es_calculateElectricField(e_es, Phi, technical, stencil, fsgrid, coordinates.physicalGridSpacing, sysBoundaries);
                       });

   mpiTimer.start();
   // Exchange electric field with neighbouring processes
   fsgrid.updateGhostCells(e_es);
   mpiTimer.stop();

   ETimer.stop(numCells, "Spatial Cells");
}
