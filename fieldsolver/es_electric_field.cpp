#include "fs_common.h"
#include "es_electric_field.hpp"
#include <HYPRE.h>
#include <HYPRE_struct_ls.h>
#include <vector>

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

   // Note: PFMG convergence degrades for periodic dimensions whose size is far
   // from a power of two. Might have to switch to BoomerAMG

   // Deal with the null space of Poisson's equation by
   //   1. explicitly monitor sum(rho) over the whole domain via one MPI
   //      collective
   //   2. subtract mean(rho) from rho everywhere before solving, leaving
   //      the matrix itself untouched (still exactly singular, with the
   //      constant vector in its null space) and starting PCG from a zero
   //      initial guess.

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

   HYPRE_Int ilower[3] = { (HYPRE_Int)localStart[0], (HYPRE_Int)localStart[1], (HYPRE_Int)localStart[2] };
   HYPRE_Int iupper[3] = { (HYPRE_Int)(localStart[0]+lx-1), (HYPRE_Int)(localStart[1]+ly-1), (HYPRE_Int)(localStart[2]+lz-1) };

   HYPRE_StructGrid hgrid;
   HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &hgrid);
   HYPRE_StructGridSetExtents(hgrid, ilower, iupper);
   HYPRE_Int hypre_periodic[3] = { (HYPRE_Int)globalSize[0], (HYPRE_Int)globalSize[1], (HYPRE_Int)globalSize[2] };
   HYPRE_StructGridSetPeriodic(hgrid, hypre_periodic);
   HYPRE_StructGridAssemble(hgrid);

   // 7-point star stencil: 0=center, 1/2=-x/+x, 3/4=-y/+y, 5/6=-z/+z
   HYPRE_StructStencil hstencil;
   HYPRE_StructStencilCreate(3, 7, &hstencil);
   HYPRE_Int offsets[7][3] = {{0,0,0},{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
   for (int e = 0; e < 7; ++e) {
      HYPRE_StructStencilSetElement(hstencil, e, offsets[e]);
   }

   HYPRE_StructMatrix Amat;
   HYPRE_StructMatrixCreate(MPI_COMM_WORLD, hgrid, hstencil, &Amat);
   HYPRE_StructMatrixInitialize(Amat);

   HYPRE_StructVector bvec, xvec;
   HYPRE_StructVectorCreate(MPI_COMM_WORLD, hgrid, &bvec);
   HYPRE_StructVectorCreate(MPI_COMM_WORLD, hgrid, &xvec);
   HYPRE_StructVectorInitialize(bvec);
   HYPRE_StructVectorInitialize(xvec);

   // Monitor charge neutrality
   double sumRho_local = 0.0, sumAbsRho_local = 0.0;
   for (long long li = 0; li < nlocal; ++li) {
      sumRho_local    += rho[li];
      sumAbsRho_local += std::abs(rho[li]);
   }
   double reduceLocal[2]  = { sumRho_local, sumAbsRho_local };
   double reduceGlobal[2] = { 0.0, 0.0 };
   MPI_Allreduce(reduceLocal, reduceGlobal, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double sumRhoGlobal    = reduceGlobal[0];
   const double sumAbsRhoGlobal = reduceGlobal[1];

   const double totalPoints = (double)globalSize[0] * (double)globalSize[1] * (double)globalSize[2];
   const double meanRho = sumRhoGlobal / totalPoints;

   // Relative charge-imbalance threshold above which we log a warning.
   // sumAbsRhoGlobal normalizes this to the actual scale of rho, so the
   // threshold is meaningful regardless of units; 1e-3 is a starting point
   // and can be tightened/loosened based on what turns out to be typical
   // for a healthy run.
   constexpr double neutralityWarnThreshold = 1e-3;
   const double relativeImbalance = sumAbsRhoGlobal > 0.0 ? std::abs(sumRhoGlobal) / sumAbsRhoGlobal : 0.0;
   if (relativeImbalance > neutralityWarnThreshold) {
      int myRank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
      if (myRank == MASTER_RANK) {
         fprintf(stderr, "es_ElectrostaticPotential: charge neutrality violated: sum(rho)=%e, "
                         "relative imbalance |sum(rho)|/sum(|rho|)=%e (threshold %e)\n",
                 sumRhoGlobal, relativeImbalance, neutralityWarnThreshold);
      }
   }

   // Explicit per-point coefficients, for periodic boundaries we could
   // consider HYPRE_StructMatrixSetConstantEntries
   std::vector<double> mvals(7*nlocal);
   std::vector<double> bvals(nlocal);
   HYPRE_Int entries[7] = {0,1,2,3,4,5,6};

   long long idx = 0;
   for (int k = 0; k < lz; ++k) {
      for (int j = 0; j < ly; ++j) {
         for (int i = 0; i < lx; ++i) {
            double* mv = &mvals[7*idx];
            const long long lidx = i + lx*((long long)j + ly*k);
            mv[0] = 2.0/(dxyz[0]*dxyz[0]) + 2.0/(dxyz[1]*dxyz[1]) + 2.0/(dxyz[2]*dxyz[2]);
            mv[1] = mv[2] = -1.0/ (dxyz[0]*dxyz[0]);
            mv[3] = mv[4] = -1.0/(dxyz[1]*dxyz[1]);
            mv[5] = mv[6] = -1.0/(dxyz[2]*dxyz[2]);
            bvals[idx] = (rho[lidx] - meanRho) / physicalconstants::EPS_0;
            ++idx;
         }
      }
   }
   HYPRE_StructMatrixSetBoxValues(Amat, ilower, iupper, 7, entries, mvals.data());
   HYPRE_StructMatrixAssemble(Amat);
   HYPRE_StructVectorSetBoxValues(bvec, ilower, iupper, bvals.data());
   HYPRE_StructVectorAssemble(bvec);

   HYPRE_StructVectorSetConstantValues(xvec, 0.0);
   HYPRE_StructVectorAssemble(xvec);

   HYPRE_StructSolver pcgSolver, pfmgPrecond;
   HYPRE_StructPCGCreate(MPI_COMM_WORLD, &pcgSolver);
   HYPRE_StructPCGSetMaxIter(pcgSolver, 200);
   HYPRE_StructPCGSetTol(pcgSolver, 1e-10);
   HYPRE_StructPCGSetTwoNorm(pcgSolver, 1);
   HYPRE_StructPCGSetPrintLevel(pcgSolver, 0);

   HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &pfmgPrecond);
   HYPRE_StructPFMGSetMaxIter(pfmgPrecond, 1);
   HYPRE_StructPFMGSetTol(pfmgPrecond, 0.0);
   HYPRE_StructPFMGSetZeroGuess(pfmgPrecond);
   HYPRE_StructPFMGSetRelaxType(pfmgPrecond, 1);     // non-symmetric red/black Gauss-Seidel
   HYPRE_StructPFMGSetNumPreRelax(pfmgPrecond, 1);
   HYPRE_StructPFMGSetNumPostRelax(pfmgPrecond, 1);

   HYPRE_StructPCGSetPrecond(pcgSolver, HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, pfmgPrecond);

   HYPRE_StructPCGSetup(pcgSolver, Amat, bvec, xvec);
   HYPRE_StructPCGSolve(pcgSolver, Amat, bvec, xvec);

   HYPRE_Int  its    = 0;
   HYPRE_Real relres = 0.0;
   HYPRE_StructPCGGetNumIterations(pcgSolver, &its);
   HYPRE_StructPCGGetFinalRelativeResidualNorm(pcgSolver, &relres);
#ifndef DEBUG_FSOLVER
   // if (relres > 1e-8)
#endif
   {
      int myRank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
      if (myRank == MASTER_RANK) {
         fprintf(stderr, "es_ElectrostaticPotential: Hypre PCG/PFMG finished with relative residual %e after %d iterations "
                         "(consider BoomerAMG instead of PFMG)\n", (double)relres, (int)its);
      }
   }

   std::vector<double> phiLocal(nlocal);
   HYPRE_StructVectorGetBoxValues(xvec, ilower, iupper, phiLocal.data());

   HYPRE_StructPCGDestroy(pcgSolver);
   HYPRE_StructPFMGDestroy(pfmgPrecond);
   HYPRE_StructMatrixDestroy(Amat);
   HYPRE_StructVectorDestroy(bvec);
   HYPRE_StructVectorDestroy(xvec);
   HYPRE_StructGridDestroy(hgrid);
   HYPRE_StructStencilDestroy(hstencil);

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
