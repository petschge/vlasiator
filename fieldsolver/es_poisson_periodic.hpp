#pragma once
/*
 * This file is part of Vlasiator.
 *
 * Periodic Poisson solve for the initial electrostatic field (es_ElectrostaticPotential):
 *
 *     -eps0 lap(phi) = rho - mean(rho)
 *
 * on a structured grid that is periodic in every dimension, with the compact 7-point Laplacian. Axes with a
 * single cell carry no derivative. The grid is set up through HYPRE's semi-structured interface, so HYPRE
 * handles the periodic neighbours and the global numbering, and assembled as a ParCSR matrix, so that PCG can
 * use BoomerAMG. PFMG, used before, converges poorly for periodic extents far from a power of two (for
 * example a relative residual of 0.16 after 200 iterations on 5464 cells).
 *
 * The periodic Laplacian is singular (constants). The mean of rho is removed, so the system is consistent,
 * and a diagonal shift of 1e-8 times the smallest nonzero eigenvalue keeps it nonsingular for BoomerAMG's
 * coarse solve: it changes the solution by at most 1e-8 relative. The mean of phi is removed afterwards.
 *
 * Header only (static functions), so that the build needs no new object file; included by
 * es_electric_field.cpp only.
 */
#include <HYPRE.h>
#include <HYPRE_sstruct_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_krylov.h>
#include <mpi.h>
#include <array>
#include <cmath>
#include <vector>

/*! rho: local values (index i + lx*(j + ly*k)) of the charge density; phi: the potential, same layout.
 *  relres and its: final relative residual and iterations of PCG. */
static inline void es_SolvePoissonPeriodic(const std::array<long long,3>& globalSize,
                                           const std::array<long long,3>& localStart,
                                           const std::array<long long,3>& localSize,
                                           const std::array<double,3>& dxyz,
                                           const std::vector<double>& rho, double eps0,
                                           std::vector<double>& phi, double& relres, int& its) {
   const long long nlocal = localSize[0]*localSize[1]*localSize[2];
   const std::array<bool,3> active = {globalSize[0] > 1, globalSize[1] > 1, globalSize[2] > 1};

   // mean of rho over the whole grid
   double sums[2] = {0.0, (double)nlocal}, gsums[2];
   for (long long i = 0; i < nlocal; ++i) { sums[0] += rho[i]; }
   MPI_Allreduce(sums, gsums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double meanRho = gsums[1] > 0.0 ? gsums[0]/gsums[1] : 0.0;

   // smallest nonzero eigenvalue of the periodic Laplacian, for the shift
   double lambdaMin = 0.0;
   for (int d = 0; d < 3; ++d) {
      if (!active[d]) { continue; }
      const double l = std::pow(2.0*std::sin(M_PI/(double)globalSize[d])/dxyz[d], 2);
      lambdaMin = (lambdaMin == 0.0) ? l : std::min(lambdaMin, l);
   }
   const double shift = 1e-8*lambdaMin;

   HYPRE_Int ilower[3] = {(HYPRE_Int)localStart[0], (HYPRE_Int)localStart[1], (HYPRE_Int)localStart[2]};
   HYPRE_Int iupper[3] = {(HYPRE_Int)(localStart[0] + localSize[0] - 1), (HYPRE_Int)(localStart[1] + localSize[1] - 1),
                          (HYPRE_Int)(localStart[2] + localSize[2] - 1)};
   HYPRE_Int periodic[3] = {active[0] ? (HYPRE_Int)globalSize[0] : 0, active[1] ? (HYPRE_Int)globalSize[1] : 0,
                            active[2] ? (HYPRE_Int)globalSize[2] : 0};

   HYPRE_SStructGrid grid;
   HYPRE_SStructGridCreate(MPI_COMM_WORLD, 3, 1, &grid);
   HYPRE_SStructGridSetExtents(grid, 0, ilower, iupper);
   HYPRE_SStructVariable vartypes[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};
   HYPRE_SStructGridSetVariables(grid, 0, 1, vartypes);
   HYPRE_SStructGridSetPeriodic(grid, 0, periodic);
   HYPRE_SStructGridAssemble(grid);

   // stencil: centre, then -/+ neighbours of the active axes only
   std::vector<std::array<HYPRE_Int,3>> offsets = {{0, 0, 0}};
   for (int d = 0; d < 3; ++d) {
      if (!active[d]) { continue; }
      std::array<HYPRE_Int,3> m{0, 0, 0}, p{0, 0, 0};
      m[d] = -1; p[d] = 1;
      offsets.push_back(m);
      offsets.push_back(p);
   }
   const HYPRE_Int nEntries = (HYPRE_Int)offsets.size();
   HYPRE_SStructStencil stencil;
   HYPRE_SStructStencilCreate(3, nEntries, &stencil);
   for (HYPRE_Int e = 0; e < nEntries; ++e) { HYPRE_SStructStencilSetEntry(stencil, e, offsets[e].data(), 0); }

   HYPRE_SStructGraph graph;
   HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);
   HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);
   HYPRE_SStructGraphSetStencil(graph, 0, 0, stencil);
   HYPRE_SStructGraphAssemble(graph);

   // matrix -lap + shift (in units of 1/m^2), right-hand side (rho - mean)/eps0
   HYPRE_SStructMatrix A;
   HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
   HYPRE_SStructMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_SStructMatrixInitialize(A);
   std::vector<double> values(nEntries*nlocal);
   double diag = shift;
   std::vector<double> off;
   for (int d = 0; d < 3; ++d) {
      if (!active[d]) { continue; }
      diag += 2.0/(dxyz[d]*dxyz[d]);
      off.push_back(-1.0/(dxyz[d]*dxyz[d]));
      off.push_back(-1.0/(dxyz[d]*dxyz[d]));
   }
   for (long long i = 0; i < nlocal; ++i) {
      values[nEntries*i] = diag;
      for (HYPRE_Int e = 1; e < nEntries; ++e) { values[nEntries*i + e] = off[e - 1]; }
   }
   std::vector<HYPRE_Int> entries(nEntries);
   for (HYPRE_Int e = 0; e < nEntries; ++e) { entries[e] = e; }
   HYPRE_SStructMatrixSetBoxValues(A, 0, ilower, iupper, 0, nEntries, entries.data(), values.data());
   HYPRE_SStructMatrixAssemble(A);

   HYPRE_SStructVector b, x;
   HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
   HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);
   HYPRE_SStructVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_SStructVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_SStructVectorInitialize(b);
   HYPRE_SStructVectorInitialize(x);
   std::vector<double> bvals(nlocal), xvals(nlocal, 0.0);
   for (long long i = 0; i < nlocal; ++i) { bvals[i] = (rho[i] - meanRho)/eps0; }
   HYPRE_SStructVectorSetBoxValues(b, 0, ilower, iupper, 0, bvals.data());
   HYPRE_SStructVectorSetBoxValues(x, 0, ilower, iupper, 0, xvals.data());
   HYPRE_SStructVectorAssemble(b);
   HYPRE_SStructVectorAssemble(x);

   HYPRE_ParCSRMatrix parA;
   HYPRE_ParVector parb, parx;
   HYPRE_SStructMatrixGetObject(A, (void**)&parA);
   HYPRE_SStructVectorGetObject(b, (void**)&parb);
   HYPRE_SStructVectorGetObject(x, (void**)&parx);

   HYPRE_Solver amg, pcg;
   HYPRE_BoomerAMGCreate(&amg);
   HYPRE_BoomerAMGSetPrintLevel(amg, 0);
   HYPRE_BoomerAMGSetMaxIter(amg, 1);
   HYPRE_BoomerAMGSetTol(amg, 0.0);
   HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg);
   HYPRE_PCGSetMaxIter(pcg, 500);
   HYPRE_PCGSetTol(pcg, 1e-12);
   HYPRE_PCGSetTwoNorm(pcg, 1);
   HYPRE_PCGSetPrintLevel(pcg, 0);
   HYPRE_PCGSetPrecond(pcg, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, amg);
   HYPRE_ParCSRPCGSetup(pcg, parA, parb, parx);
   HYPRE_ParCSRPCGSolve(pcg, parA, parb, parx);
   HYPRE_Int hits = 0;
   HYPRE_Real hrel = 0.0;
   HYPRE_PCGGetNumIterations(pcg, &hits);
   HYPRE_PCGGetFinalRelativeResidualNorm(pcg, &hrel);
   its = (int)hits;
   relres = (double)hrel;

   HYPRE_SStructVectorGather(x);
   phi.assign(nlocal, 0.0);
   HYPRE_SStructVectorGetBoxValues(x, 0, ilower, iupper, 0, phi.data());

   // remove the mean of phi (the shift leaves a tiny constant; E does not depend on it)
   double ps[2] = {0.0, (double)nlocal}, gps[2];
   for (long long i = 0; i < nlocal; ++i) { ps[0] += phi[i]; }
   MPI_Allreduce(ps, gps, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   const double meanPhi = gps[1] > 0.0 ? gps[0]/gps[1] : 0.0;
   for (long long i = 0; i < nlocal; ++i) { phi[i] -= meanPhi; }

   HYPRE_ParCSRPCGDestroy(pcg);
   HYPRE_BoomerAMGDestroy(amg);
   HYPRE_SStructVectorDestroy(b);
   HYPRE_SStructVectorDestroy(x);
   HYPRE_SStructMatrixDestroy(A);
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructStencilDestroy(stencil);
   HYPRE_SStructGridDestroy(grid);
}
