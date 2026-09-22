#include "ap_electric_field.hpp"
#include "volume_averages.hpp"
#include <cmath>
#include <map>
#include <sstream>
#include <tuple>
#include "../object_wrapper.h"

using namespace std;

/* ============================================================
 * Section 1: alpha_s / mu / J-hat  (Eqs. 35-37)
 * ============================================================ */

void ap_BuildSpeciesTensors(
   std::vector<fsgrids::speciesrhoqspan>& speciesRhoQ,
   std::vector<fsgrids::speciesjspan>& speciesJ,
   fsgrids::constperbspan perb,
   fsgrids::constbgbspan bgb,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real theta,
   Real dt,
   std::vector<std::array<Real,9>>& outMu,
   std::vector<std::array<Real,3>>& outJhat
) {
   const auto& localSize = fsgrid.getLocalSize();
   const int lx = localSize[0], ly = localSize[1], lz = localSize[2];
   const long long nLocalCells = (long long)lx*ly*lz;
   outMu.assign(nLocalCells, std::array<Real,9>{0,0,0,0,0,0,0,0,0});
   outJhat.assign(nLocalCells, std::array<Real,3>{0,0,0});

   const uint numPops = speciesRhoQ.size();

   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: build alpha_s, mu, J-hat"),
      technical,
      [&, lx, ly](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {

         const size_t lid = stencil.ooo();
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);

         // Cell-centered total B = perb + bgb
         const Real Bx = perb[lid][fsgrids::bfield::PERBX] + bgb[lid][fsgrids::bgbfield::BGBXVOL];
         const Real By = perb[lid][fsgrids::bfield::PERBY] + bgb[lid][fsgrids::bgbfield::BGBYVOL];
         const Real Bz = perb[lid][fsgrids::bfield::PERBZ] + bgb[lid][fsgrids::bgbfield::BGBZVOL];

         std::array<Real,9> mu{0,0,0,0,0,0,0,0,0};
         std::array<Real,3> jhat{0,0,0};

         for (uint popID = 0; popID < numPops; ++popID) {
            const Real rhoQ = speciesRhoQ[popID][lid][fsgrids::speciesrhoq::SRHOQ];
            const Real Jx   = speciesJ[popID][lid][fsgrids::speciesj::SJX];
            const Real Jy   = speciesJ[popID][lid][fsgrids::speciesj::SJY];
            const Real Jz   = speciesJ[popID][lid][fsgrids::speciesj::SJZ];

            // eps_s = q_s * theta * dt / m_s  (Eq. 37)
            const Real charge = getObjectWrapper().particleSpecies[popID].charge;
            const Real mass   = getObjectWrapper().particleSpecies[popID].mass;
            const Real eps = charge * theta * dt / mass;

            const Real B2 = Bx*Bx + By*By + Bz*Bz;
            const Real denom = 1.0 + eps*eps*B2;

            // alpha_s = [I - eps * (I x B) + eps^2 * B B^T] / denom
            // (I x B) as a matrix is just
            //   [ 0   -Bz   By ]
            //   [ Bz   0   -Bx ]
            //   [-By   Bx   0  ]
            std::array<Real,9> alpha;
            alpha[0] = ( 1.0       + eps*eps*Bx*Bx) / denom;  // xx
            alpha[1] = (-eps*(-Bz) + eps*eps*Bx*By) / denom;  // xy
            alpha[2] = (-eps*( By) + eps*eps*Bx*Bz) / denom;  // xz
            alpha[3] = (-eps*( Bz) + eps*eps*By*Bx) / denom;  // yx
            alpha[4] = ( 1.0       + eps*eps*By*By) / denom;  // yy
            alpha[5] = (-eps*(-Bx) + eps*eps*By*Bz) / denom;  // yz
            alpha[6] = (-eps*(-By) + eps*eps*Bz*Bx) / denom;  // zx
            alpha[7] = (-eps*( Bx) + eps*eps*Bz*By) / denom;  // zy
            alpha[8] = ( 1.0       + eps*eps*Bz*Bz) / denom;  // zz

            // mu += (q_s/m_s) * rho_s^k * alpha_s   (Eq. 36)
            const Real qOverM_rho = (charge/mass) * rhoQ;
            for (int t = 0; t < 9; ++t) { mu[t] += qOverM_rho * alpha[t]; }

            // J-hat += alpha_s . J_s^{k*}   (Eq. 35)
            jhat[0] += alpha[0]*Jx + alpha[1]*Jy + alpha[2]*Jz;
            jhat[1] += alpha[3]*Jx + alpha[4]*Jy + alpha[5]*Jz;
            jhat[2] += alpha[6]*Jx + alpha[7]*Jy + alpha[8]*Jz;
         }

         outMu[lidx]   = mu;
         outJhat[lidx] = jhat;
      });
}

/* ============================================================
 * Section 2: discrete curl, composed for curl-curl matrix entries
 * ============================================================
 */

struct CurlTerm { int comp; int di, dj, dk; Real coeff; };

static std::array<CurlTerm,4> curlStencilCentered(int outComp, const std::array<Real,3>& dxyz) {
   const int p = (outComp+1)%3, q = (outComp+2)%3;
   std::array<int,3> plusP{0,0,0};  plusP[p]  = 1;
   std::array<int,3> minusP{0,0,0}; minusP[p] = -1;
   std::array<int,3> plusQ{0,0,0};  plusQ[q]  = 1;
   std::array<int,3> minusQ{0,0,0}; minusQ[q] = -1;
   return { CurlTerm{ q, plusP[0],plusP[1],plusP[2],     0.5/dxyz[p] },
            CurlTerm{ q, minusP[0],minusP[1],minusP[2], -0.5/dxyz[p] },
            CurlTerm{ p, minusQ[0],minusQ[1],minusQ[2],  0.5/dxyz[q] },
            CurlTerm{ p, plusQ[0],plusQ[1],plusQ[2],    -0.5/dxyz[q] } };
}

// Composed curl-curl stencil for output E-component `outComp`: a list of
// (inputComp, di,dj,dk, coefficient) accumulated by composing the
// centered curl stencil with itself
static std::vector<CurlTerm> curlCurlStencil(int outComp, const std::array<Real,3>& dxyz) {
   std::map<std::tuple<int,int,int,int>, Real> acc;
   for (const auto& outer : curlStencilCentered(outComp, dxyz)) {
      for (const auto& inner : curlStencilCentered(outer.comp, dxyz)) {
         acc[{inner.comp, outer.di+inner.di, outer.dj+inner.dj, outer.dk+inner.dk}]
            += outer.coeff * inner.coeff;
      }
   }
   std::vector<CurlTerm> result;
   result.reserve(acc.size());
   for (const auto& kv : acc) {
      const auto& [comp,di,dj,dk] = kv.first;
      if (kv.second != 0.0) { result.push_back({comp,di,dj,dk,kv.second}); }
   }
   return result;
}

/* ============================================================
 * Section 3: Eq. (38) -- HYPRE IJ assembly + BoomerAMG solve
 * ============================================================
 */

bool ap_SolveElectricField(
   fsgrids::efieldspan e,
   fsgrids::efieldspan edt2,
   fsgrids::constperbspan perb,
   fsgrids::constbgbspan bgb,
   fsgrids::constdperbspan dperb,
   const std::vector<std::array<Real,9>>& mu,
   const std::vector<std::array<Real,3>>& Jhat,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real c,
   Real theta,
   Real dt
) {
   const auto dxyz = fsgrid.getGridSpacing();
   const auto localSize = fsgrid.getLocalSize();
   const int lx = localSize[0], ly = localSize[1], lz = localSize[2];
   const long long nLocalCells = (long long)lx*ly*lz;
   const long long nLocalDofs  = 3*nLocalCells;

   // ---- Step 1: global DOF numbering (contiguous per rank by construction) ----
   long long dofOffset = 0;
   MPI_Exscan(&nLocalDofs, &dofOffset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
   int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   if (myRank == 0) {
      // MPI_Exscan leaves rank 0's result undefined
      dofOffset = 0;
   }

   // Local k-major linear index
   auto localLinear = [lx,ly](int i, int j, int k) -> long long {
      return i + lx*((long long)j + ly*k);
   };

   // DOF base (3x this cell's cell-index-within-numbering) per cell,
   // stored so it can be ghost-exchanged like any other fsgrid quantity.
   fsgrid::FsData<std::array<Real,1>> dofBase(fsgrid.getNumStorageCells());
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: assign DOF numbering"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const long long lidx = localLinear(stencil.i, stencil.j, stencil.k);
         dofBase[stencil.ooo()][0] = (Real)(dofOffset + 3*lidx);
      });
   fsgrid.updateGhostCells(dofBase.view());

   auto globalDof = [&](const fsgrid::FsStencil& stencil, size_t neighborLid, int comp) -> HYPRE_BigInt {
      return (HYPRE_BigInt)llround(dofBase[neighborLid][0]) + comp;
   };

   // ---- Step 2: assemble the IJ matrix + RHS ----
   HYPRE_IJMatrix Aij;
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, dofOffset, dofOffset+nLocalDofs-1,
                         dofOffset, dofOffset+nLocalDofs-1, &Aij);
   HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(Aij);

   HYPRE_IJVector bij, xij;
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, dofOffset, dofOffset+nLocalDofs-1, &bij);
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, dofOffset, dofOffset+nLocalDofs-1, &xij);
   HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(bij);
   HYPRE_IJVectorInitialize(xij);

   // In SI unlike the paper
   const Real reactionScale = 1.0/(dt*dt);
   const Real muScale       = theta*theta/physicalconstants::EPS_0;
   const Real curlScale     = c*c*theta*theta;

   fsgrid.serial_for( // serial: HYPRE_IJMatrixSetValues below is not thread-safe per-row without extra care
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: assemble Eq.38 system"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {

         if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) { return; }
         const size_t lid = stencil.ooo();
         const long long lidx = localLinear(stencil.i, stencil.j, stencil.k);
         const auto& m = mu[lidx];

         for (int comp = 0; comp < 3; ++comp) {
            const HYPRE_BigInt row = globalDof(stencil, lid, comp);

            // Reaction term: (1/dt^2) I + theta^2 mu/EPS_0, all at this cell.
            std::map<HYPRE_BigInt, Real> rowVals;
            for (int cc = 0; cc < 3; ++cc) {
               const Real diag = (cc==comp) ? reactionScale : 0.0;
               const Real val = diag + muScale*m[comp*3+cc];
               if (val != 0.0) { rowVals[globalDof(stencil, lid, cc)] += val; }
            }

            // curl-curl term, via the composed stencil (Section 2).
            for (const auto& term : curlCurlStencil(comp, dxyz)) {
               if (!stencil.cellExists(term.di, term.dj, term.dk)) { continue; } // Is this correct in a periodic box?
               const size_t nlid = stencil.indexFromOffset(term.di, term.dj, term.dk);
               rowVals[globalDof(stencil, nlid, term.comp)] += curlScale * term.coeff;
            }

            std::vector<HYPRE_BigInt> cols; cols.reserve(rowVals.size());
            std::vector<HYPRE_Real> vals; vals.reserve(rowVals.size());
            for (const auto& kv : rowVals) { cols.push_back(kv.first); vals.push_back(kv.second); }
            HYPRE_Int ncols = (HYPRE_Int)cols.size();
            HYPRE_IJMatrixSetValues(Aij, 1, &ncols, &row, cols.data(), vals.data());

            // RHS: (1/dt^2) E^k + (c^2 theta/dt) curl(B^k) - (theta/(dt*EPS_0)) J-hat
            const Real Ek = e[lid][comp]; // e holds E^k on entry, E^{k+theta} on exit -- read before overwrite

            // curl(B^k)
            // NOT via dperb, which uses a nonlinear TVD slope limiter
            std::array<Real,3> curlBtotalVec = {0.0, 0.0, 0.0};
            for (int outComp = 0; outComp < 3; ++outComp) {
               Real acc = 0.0;
               for (const auto& term : curlStencilCentered(outComp, dxyz)) {
                  if (!stencil.cellExists(term.di, term.dj, term.dk)) { continue; }
                  const size_t nlid = stencil.indexFromOffset(term.di, term.dj, term.dk);
                  acc += term.coeff * (perb[nlid][term.comp] + bgb[nlid][term.comp]);
               }
               curlBtotalVec[outComp] = acc;
            }
            const Real curlBtotal = curlBtotalVec[comp];

            const Real rhs = reactionScale*Ek + c*c*theta/dt*curlBtotal - theta/(dt*physicalconstants::EPS_0)*Jhat[lidx][comp];

            HYPRE_IJVectorSetValues(bij, 1, &row, &rhs);
            const Real x0 = Ek;
            HYPRE_IJVectorSetValues(xij, 1, &row, &x0); // initial guess = E^k
         }
      });

   HYPRE_IJMatrixAssemble(Aij);
   HYPRE_IJVectorAssemble(bij);
   HYPRE_IJVectorAssemble(xij);

   HYPRE_ParCSRMatrix Apar; HYPRE_IJMatrixGetObject(Aij, (void**)&Apar);
   HYPRE_ParVector bpar; HYPRE_IJVectorGetObject(bij, (void**)&bpar);
   HYPRE_ParVector xpar; HYPRE_IJVectorGetObject(xij, (void**)&xpar);

   // ---- Step 3: solve. BoomerAMG-preconditioned GMRES because  mu's I x B
   // term might make the reaction block antisymmetric
   HYPRE_Solver amg, gmres;
   HYPRE_BoomerAMGCreate(&amg);
   HYPRE_BoomerAMGSetPrintLevel(amg, 0);
   HYPRE_BoomerAMGSetMaxIter(amg, 1);
   HYPRE_BoomerAMGSetTol(amg, 0.0);

   HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &gmres);
   HYPRE_GMRESSetMaxIter(gmres, 200);
   HYPRE_GMRESSetTol(gmres, 1e-10);
   HYPRE_GMRESSetPrintLevel(gmres, 0);
   HYPRE_GMRESSetPrecond(gmres, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                          (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, amg);

   HYPRE_ParCSRGMRESSetup(gmres, Apar, bpar, xpar);
   HYPRE_ParCSRGMRESSolve(gmres, Apar, bpar, xpar);

   HYPRE_Int its = 0; HYPRE_Real relres = 0.0;
   HYPRE_GMRESGetNumIterations(gmres, &its);
   HYPRE_GMRESGetFinalRelativeResidualNorm(gmres, &relres);
   if (myRank == MASTER_RANK) {
      fprintf(stderr, "apSolveElectricField: GMRES/BoomerAMG finished, relres=%e after %d iters\n", (double)relres, (int)its);
   }

   // ---- Step 4: extract E^{k+theta}, compute E^{k+1} (Eq. 40) ----
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: extract E^(k+theta), compute E^(k+1)"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         for (int comp = 0; comp < 3; ++comp) {
            const HYPRE_BigInt row = globalDof(stencil, lid, comp);
            HYPRE_Real val;
            HYPRE_IJVectorGetValues(xij, 1, &row, &val);
            const Real Ektheta = (Real)val;
            const Real Ek = e[lid][comp];
            edt2[lid][comp] = Ektheta;                          // E^{k+theta}, staged in edt2
            e[lid][comp]    = (Ektheta - Ek)/theta + Ek;        // E^{k+1}  (Eq. 40)
         }
      });

   HYPRE_ParCSRGMRESDestroy(gmres);
   HYPRE_BoomerAMGDestroy(amg);
   HYPRE_IJMatrixDestroy(Aij);
   HYPRE_IJVectorDestroy(bij);
   HYPRE_IJVectorDestroy(xij);

   fsgrid.updateGhostCells(e);
   fsgrid.updateGhostCells(edt2);
   return (relres < 1e-6); // FIXME: what convergence threshold the rest of the solver treats as success
}

/* ============================================================
 * Section 3b: stage E for the Vlasov acceleration step (Eq. 20)
 * ============================================================
 * We have to get E into CellParams::EXVOL/EYVOL/EZVOL so that
 * getFieldsFromFsGrid gets it onto the Vlasov grid for us.
 */

void ap_StageElectricFieldForAcceleration(
   fsgrids::constefieldspan edt2,
   fsgrids::volspan vol,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid
) {
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: stage E into vol"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         vol[lid][fsgrids::volfields::EXVOL] = edt2[lid][0];
         vol[lid][fsgrids::volfields::EYVOL] = edt2[lid][1];
         vol[lid][fsgrids::volfields::EZVOL] = edt2[lid][2];
      });
   fsgrid.updateGhostCells(vol);
}

/* ============================================================
 * Section 3c: stage B for the Vlasov acceleration step (Eq. 20, row 8)
 * ============================================================
 */
void ap_StageMagneticFieldForAcceleration(
   fsgrids::constperbspan perbdt2,
   fsgrids::volspan vol,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid
) {
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: stage B^(k+theta) into vol"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         vol[lid][fsgrids::volfields::PERBXVOL] = perbdt2[lid][fsgrids::bfield::PERBX];
         vol[lid][fsgrids::volfields::PERBYVOL] = perbdt2[lid][fsgrids::bfield::PERBY];
         vol[lid][fsgrids::volfields::PERBZVOL] = perbdt2[lid][fsgrids::bfield::PERBZ];
      });
   fsgrid.updateGhostCells(vol);
}

/* ============================================================
 * Section 4: Faraday update (Eq. 23, 39)
 * ============================================================ */

void ap_UpdateMagneticField(
   fsgrids::perbspan perb,
   fsgrids::perbspan perbdt2,
   fsgrids::constefieldspan e, // holds E^{k+theta} here, per apSolveElectricField's edt2 output
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real theta,
   Real dt
) {
   const auto dxyz = fsgrid.getGridSpacing();
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: Faraday update"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         for (int comp = 0; comp < 3; ++comp) {
            Real curlE = 0.0;
            for (const auto& term : curlStencilCentered(comp, dxyz)) {
               if (!stencil.cellExists(term.di, term.dj, term.dk)) { continue; }
               const size_t nlid = stencil.indexFromOffset(term.di, term.dj, term.dk);
               curlE += term.coeff * e[nlid][term.comp];
            }

            const Real Bk = perb[lid][comp];
            const Real Bk1 = Bk - dt*curlE;                  // Eq. 23
            perbdt2[lid][comp] = theta*Bk1 + (1.0-theta)*Bk; // Eq. 39, B^{k+theta}, staged in perbdt2
            perb[lid][comp] = Bk1;                           // B^{k+1}
         }
      });
   fsgrid.updateGhostCells(perb);
   fsgrid.updateGhostCells(perbdt2);
}

/* ============================================================
 * Section 5: Gauss's-law correction (Eq. 45, 41)
 * ============================================================
 */

void ap_GaussLawCorrection(
   fsgrids::efieldspan e,
   fsgrids::efieldspan edt2,
   fsgrids::constefieldspan eOld,
   fsgrids::momentsspan moments,
   const std::vector<std::array<Real,9>>& mu,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real dt
) {

   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

   const auto& globalSize = fsgrid.getGlobalSize();
   const auto  localSize  = fsgrid.getLocalSize();
   const auto  dxyz       = fsgrid.getGridSpacing();
   const int lx=localSize[0], ly=localSize[1], lz=localSize[2];
   const long long nlocal = (long long)lx*ly*lz;

   long long dofOffset = 0;
   MPI_Exscan(&nlocal, &dofOffset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
   if (myRank == 0) {
      // MPI_Exscan leaves rank 0's result undefined
      dofOffset = 0;
   }

   auto localLinear = [lx,ly](int i, int j, int k) -> long long {
      return i + lx*((long long)j + ly*k);
   };

   fsgrid::FsData<std::array<Real,1>> dofBase(fsgrid.getNumStorageCells());
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: assign phi DOF numbering"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const long long lidx = localLinear(stencil.i, stencil.j, stencil.k);
         dofBase[stencil.ooo()][0] = (Real)(dofOffset + lidx);
      });
   fsgrid.updateGhostCells(dofBase.view());

   auto globalDof = [&](size_t neighborLid) -> HYPRE_BigInt {
      return (HYPRE_BigInt)llround(dofBase[neighborLid][0]);
   };

   const int nStencil = 19;
   HYPRE_Int offsets[19][3] = {
      {0,0,0},
      {-1,0,0},{1,0,0},
      {0,-1,0},{0,1,0},
      {0,0,-1},{0,0,1},
      {1,1,0},{-1,-1,0},{1,-1,0},{-1,1,0},
      {1,0,1},{-1,0,-1},{1,0,-1},{-1,0,1},
      {0,1,1},{0,-1,-1},{0,1,-1},{0,-1,1}
   };

   std::vector<double> mvals(nStencil*nlocal, 0.0), bvals(nlocal, 0.0);
   double meanRhoOverEps = 0.0;
   {
      double sumRho_local = 0.0;
      fsgrid.serial_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("AP: compute mean(rho)"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const size_t lid = stencil.ooo();
            sumRho_local += moments[lid][fsgrids::moments::RHOQ] / physicalconstants::EPS_0;
         });
      double sumRho_global = 0.0;
      MPI_Allreduce(&sumRho_local, &sumRho_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double totalPoints = (double)globalSize[0] * (double)globalSize[1] * (double)globalSize[2];
      meanRhoOverEps = sumRho_global / totalPoints;
   }

   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: assemble Gauss-correction system"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);
         const auto& m = mu[lidx];
         double* mv = &mvals[nStencil*lidx];

         // Isotropic part: -div[(I + (dt^2/EPS_0) diag(mu)) grad phi],
         // FACE-AVERAGED, not per-cell.  -- the +x entry computed here and the
         // -x entry the +x neighbor computes for the same shared face would
         // generally disagree. Averaging (this cell, neighbor)/2 at each face
         // fixes that: both cells compute the identical value for their shared
         // connection. Confirmed this reduces exactly to the old per-cell
         // formula when mu is spatially uniform (cxx_faceM=cxx_faceP=cxx_here
         // in that case).
         //
         // KNOWN LIMITATION: mu is local-only (no ghost cells), so at an
         // actual MPI rank boundary the true neighbor value isn't
         // available here and this falls back to this cell's own value --
         // NOT symmetric across rank boundaries specifically. A full fix
         // needs mu exchanged with fsgrid's updateGhostCells like every
         // other per-cell quantity in this file; not yet done. The cross
         // terms below have the identical per-cell-only issue and are
         // NOT fixed here, since they're currently negligible
         // (cross/diag ~1e-25 in the last diagnostic run) -- revisit if
         // that stops being true.
         const bool hasXm = (stencil.i > 0),    hasXp = (stencil.i < lx-1);
         const bool hasYm = (stencil.j > 0),    hasYp = (stencil.j < ly-1);
         const bool hasZm = (stencil.k > 0),    hasZp = (stencil.k < lz-1);
         const long long lidxXm = hasXm ? lidx-1     : lidx;
         const long long lidxXp = hasXp ? lidx+1     : lidx;
         const long long lidxYm = hasYm ? lidx-lx    : lidx;
         const long long lidxYp = hasYp ? lidx+lx    : lidx;
         const long long lidxZm = hasZm ? lidx-lx*ly : lidx;
         const long long lidxZp = hasZp ? lidx+lx*ly : lidx;

         const Real cxx_here = 1.0 + dt*dt*m[0]/physicalconstants::EPS_0;
         const Real cyy_here = 1.0 + dt*dt*m[4]/physicalconstants::EPS_0;
         const Real czz_here = 1.0 + dt*dt*m[8]/physicalconstants::EPS_0;
         const Real cxx_xm = 1.0 + dt*dt*mu[lidxXm][0]/physicalconstants::EPS_0;
         const Real cxx_xp = 1.0 + dt*dt*mu[lidxXp][0]/physicalconstants::EPS_0;
         const Real cyy_ym = 1.0 + dt*dt*mu[lidxYm][4]/physicalconstants::EPS_0;
         const Real cyy_yp = 1.0 + dt*dt*mu[lidxYp][4]/physicalconstants::EPS_0;
         const Real czz_zm = 1.0 + dt*dt*mu[lidxZm][8]/physicalconstants::EPS_0;
         const Real czz_zp = 1.0 + dt*dt*mu[lidxZp][8]/physicalconstants::EPS_0;

         const Real cxx_faceM = 0.5*(cxx_here+cxx_xm), cxx_faceP = 0.5*(cxx_here+cxx_xp);
         const Real cyy_faceM = 0.5*(cyy_here+cyy_ym), cyy_faceP = 0.5*(cyy_here+cyy_yp);
         const Real czz_faceM = 0.5*(czz_here+czz_zm), czz_faceP = 0.5*(czz_here+czz_zp);

         mv[0] = cxx_faceM/(dxyz[0]*dxyz[0]) + cxx_faceP/(dxyz[0]*dxyz[0])
               + cyy_faceM/(dxyz[1]*dxyz[1]) + cyy_faceP/(dxyz[1]*dxyz[1])
               + czz_faceM/(dxyz[2]*dxyz[2]) + czz_faceP/(dxyz[2]*dxyz[2]);
         mv[1] = -cxx_faceM/(dxyz[0]*dxyz[0]);
         mv[2] = -cxx_faceP/(dxyz[0]*dxyz[0]);
         mv[3] = -cyy_faceM/(dxyz[1]*dxyz[1]);
         mv[4] = -cyy_faceP/(dxyz[1]*dxyz[1]);
         mv[5] = -czz_faceM/(dxyz[2]*dxyz[2]);
         mv[6] = -czz_faceP/(dxyz[2]*dxyz[2]);

         // Cross terms from mu's off-diagonal
         const Real cxy = dt*dt*0.5*(m[1]+m[3])/physicalconstants::EPS_0; // symmetrized off-diagonal
         const Real cxz = dt*dt*0.5*(m[2]+m[6])/physicalconstants::EPS_0;
         const Real cyz = dt*dt*0.5*(m[5]+m[7])/physicalconstants::EPS_0;
         const Real kxy = -cxy/(2*dxyz[0]*dxyz[1]);
         const Real kxz = -cxz/(2*dxyz[0]*dxyz[2]);
         const Real kyz = -cyz/(2*dxyz[1]*dxyz[2]);
         mv[7]=mv[8] = kxy; mv[9]=mv[10] = -kxy;   // (+1,+1,0)/(-1,-1,0) vs (+1,-1,0)/(-1,+1,0)
         mv[11]=mv[12] = kxz; mv[13]=mv[14] = -kxz;
         mv[15]=mv[16] = kyz; mv[17]=mv[18] = -kyz;

         Real divE = 0.0;
         // div(E^k) via centered differences of the three E components at
         // this cell (E is edge-located; this samples eOld[] directly
         // rather than through curlStencil). Uses eOld, NOT e: e is
         // already E^{k+1} by this point (ap_SolveElectricField
         // overwrites it in place before returning), so reading e here
         // would compute div(E^{k+1}), not div(E^k) as the RHS formula
         // (csl_rme_si_units.md Section 6) actually requires.
         if (stencil.cellExists(1,0,0) && stencil.cellExists(-1,0,0)) {
            divE += (eOld[stencil.indexFromOffset(1,0,0)][0] - eOld[stencil.indexFromOffset(-1,0,0)][0])/(2*dxyz[0]);
         }
         if (stencil.cellExists(0,1,0) && stencil.cellExists(0,-1,0)) {
            divE += (eOld[stencil.indexFromOffset(0,1,0)][1] - eOld[stencil.indexFromOffset(0,-1,0)][1])/(2*dxyz[1]);
         }
         if (stencil.cellExists(0,0,1) && stencil.cellExists(0,0,-1)) {
            divE += (eOld[stencil.indexFromOffset(0,0,1)][2] - eOld[stencil.indexFromOffset(0,0,-1)][2])/(2*dxyz[2]);
         }

         const Real rhoOverEpsRaw = moments[lid][fsgrids::moments::RHOQ]/physicalconstants::EPS_0;
         bvals[lidx] = (rhoOverEpsRaw - meanRhoOverEps) - divE; // Eq. 45 RHS, SI, rescaled by 1/EPS_0, rho mean-subtracted
      });

   // Hand mvals/bvals to HYPRE via IJ, using dofBase for global
   // row/column indices
   HYPRE_IJMatrix Aij;
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, dofOffset, dofOffset+nlocal-1,
                         dofOffset, dofOffset+nlocal-1, &Aij);
   HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(Aij);

   HYPRE_IJVector bij, xij;
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, dofOffset, dofOffset+nlocal-1, &bij);
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, dofOffset, dofOffset+nlocal-1, &xij);
   HYPRE_IJVectorSetObjectType(bij, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(xij, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(bij);
   HYPRE_IJVectorInitialize(xij);

   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: hand Gauss-correction system to HYPRE IJ"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);
         const HYPRE_BigInt row = globalDof(stencil.ooo());
         const double* mv = &mvals[nStencil*lidx];

         std::map<HYPRE_BigInt, Real> rowVals; // accumulate in case any offsets coincide
         for (int s = 0; s < nStencil; ++s) {
            if (mv[s] == 0.0) { continue; }
            if (!stencil.cellExists(offsets[s][0], offsets[s][1], offsets[s][2])) { continue; }
            const size_t nlid = stencil.indexFromOffset(offsets[s][0], offsets[s][1], offsets[s][2]);
            rowVals[globalDof(nlid)] += mv[s];
         }
         std::vector<HYPRE_BigInt> cols; cols.reserve(rowVals.size());
         std::vector<HYPRE_Real> vals; vals.reserve(rowVals.size());
         for (const auto& kv : rowVals) { cols.push_back(kv.first); vals.push_back(kv.second); }
         HYPRE_Int ncols = (HYPRE_Int)cols.size();
         HYPRE_IJMatrixSetValues(Aij, 1, &ncols, &row, cols.data(), vals.data());

         const HYPRE_Real bval = bvals[lidx];
         HYPRE_IJVectorSetValues(bij, 1, &row, &bval);
         const HYPRE_Real x0 = 0.0;
         HYPRE_IJVectorSetValues(xij, 1, &row, &x0);
      });

   HYPRE_IJMatrixAssemble(Aij);
   HYPRE_IJVectorAssemble(bij);
   HYPRE_IJVectorAssemble(xij);

   HYPRE_ParCSRMatrix Apar; HYPRE_IJMatrixGetObject(Aij, (void**)&Apar);
   HYPRE_ParVector bpar; HYPRE_IJVectorGetObject(bij, (void**)&bpar);
   HYPRE_ParVector xpar; HYPRE_IJVectorGetObject(xij, (void**)&xpar);

   HYPRE_Solver amgPrecond, pcgSolver;
   HYPRE_BoomerAMGCreate(&amgPrecond);
   HYPRE_BoomerAMGSetPrintLevel(amgPrecond, 0);
   HYPRE_BoomerAMGSetMaxIter(amgPrecond, 1);
   HYPRE_BoomerAMGSetTol(amgPrecond, 0.0);

   HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcgSolver);
   HYPRE_PCGSetMaxIter(pcgSolver, 200);
   HYPRE_PCGSetTol(pcgSolver, 1e-10);
   HYPRE_PCGSetTwoNorm(pcgSolver, 1);
   HYPRE_PCGSetPrintLevel(pcgSolver, 0);
   HYPRE_PCGSetPrecond(pcgSolver, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                        (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, amgPrecond);

   HYPRE_ParCSRPCGSetup(pcgSolver, Apar, bpar, xpar);
   HYPRE_ParCSRPCGSolve(pcgSolver, Apar, bpar, xpar);

   HYPRE_Int pcgIts = 0; HYPRE_Real pcgRelres = 0.0;
   {
      HYPRE_PCGGetNumIterations(pcgSolver, &pcgIts);
      HYPRE_PCGGetFinalRelativeResidualNorm(pcgSolver, &pcgRelres);
      if (myRank == MASTER_RANK) {
         fprintf(stderr, "apGaussLawCorrection: Hypre PCG/BoomerAMG finished with relative residual %e "
                         "after %d iterations\n", (double)pcgRelres, (int)pcgIts);
      }
   }

   std::vector<double> phiLocal(nlocal);
   for (long long lidx = 0; lidx < nlocal; ++lidx) {
      const HYPRE_BigInt row = dofOffset + lidx;
      HYPRE_Real val;
      HYPRE_IJVectorGetValues(xij, 1, &row, &val);
      phiLocal[lidx] = (double)val;
   }

   {
      double maxAbsPhi = 0.0;
      bool phiNonFinite = false;
      for (long long c = 0; c < nlocal; ++c) {
         if (!std::isfinite(phiLocal[c])) { phiNonFinite = true; }
         if (std::abs(phiLocal[c]) > maxAbsPhi) { maxAbsPhi = std::abs(phiLocal[c]); }
      }
      double maxAbsPhiGlobal = 0.0;
      MPI_Allreduce(&maxAbsPhi, &maxAbsPhiGlobal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      int phiNonFiniteLocal = phiNonFinite ? 1 : 0, phiNonFiniteGlobal = 0;
      MPI_Allreduce(&phiNonFiniteLocal, &phiNonFiniteGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (myRank == MASTER_RANK) {
         fprintf(stderr, "apGaussLawCorrection: solved phi, max|phi|=%e%s\n", maxAbsPhiGlobal,
                 phiNonFiniteGlobal ? "  *** NaN/Inf IN SOLVED PHI ***" : "");
      }
   }

   HYPRE_ParCSRPCGDestroy(pcgSolver);
   HYPRE_BoomerAMGDestroy(amgPrecond);
   HYPRE_IJMatrixDestroy(Aij);
   HYPRE_IJVectorDestroy(bij);
   HYPRE_IJVectorDestroy(xij);

   // Belt-and-suspenders: subtract mean(phi) from the solution too
   {
      double sumPhi_local = 0.0;
      for (long long c = 0; c < nlocal; ++c) { sumPhi_local += phiLocal[c]; }
      double sumPhi_global = 0.0;
      MPI_Allreduce(&sumPhi_local, &sumPhi_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double totalPoints = (double)globalSize[0] * (double)globalSize[1] * (double)globalSize[2];
      const double meanPhi = sumPhi_global / totalPoints;
      for (long long c = 0; c < nlocal; ++c) { phiLocal[c] -= meanPhi; }
   }

   // Eq. 41: E~^{k+1} = E^{k+1} - grad(phi)
   fsgrid::FsData<std::array<Real,1>> phiGrid(fsgrid.getNumStorageCells());
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: stage phi"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);
         phiGrid[stencil.ooo()][0] = phiLocal[lidx];
      });
   fsgrid.updateGhostCells(phiGrid.view());

   // Eq. 41, only applied if actually converged
   if (pcgRelres <= 1.0) {
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("AP: apply Gauss correction to E"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const size_t lid = stencil.ooo();
            if (stencil.cellExists(1,0,0) && stencil.cellExists(-1,0,0)) {
               const Real dEx = -0.5*(phiGrid[stencil.indexFromOffset(1,0,0)][0]-phiGrid[stencil.indexFromOffset(-1,0,0)][0])/dxyz[0];
               e[lid][0] += dEx;
               edt2[lid][0] += dEx; // same phi, same correction, applied to E^{k+theta} too
            }
            if (stencil.cellExists(0,1,0) && stencil.cellExists(0,-1,0)) {
               const Real dEy = -0.5*(phiGrid[stencil.indexFromOffset(0,1,0)][0]-phiGrid[stencil.indexFromOffset(0,-1,0)][0])/dxyz[1];
               e[lid][1] += dEy;
               edt2[lid][1] += dEy;
            }
            if (stencil.cellExists(0,0,1) && stencil.cellExists(0,0,-1)) {
               const Real dEz = -0.5*(phiGrid[stencil.indexFromOffset(0,0,1)][0]-phiGrid[stencil.indexFromOffset(0,0,-1)][0])/dxyz[2];
               e[lid][2] += dEz;
               edt2[lid][2] += dEz;
            }
         });
      fsgrid.updateGhostCells(e);

   } else if (myRank == MASTER_RANK) {
      fprintf(stderr, "apGaussLawCorrection: relres=%e > 1.0 -- solve diverged, "
                      "SKIPPING correction this step, E left unmodified\n", (double)pcgRelres);
   }
}

/* ============================================================
 * Section 6: propagateFields entry point
 * ============================================================ */

// Reports max|E| (over all 3 components, all local cells)
static void ap_ReportFieldMagnitude(const char* label, fsgrids::constefieldspan field,
                                     fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   double maxAbsLocal[3] = {0.0, 0.0, 0.0};
   bool nonFiniteLocal = false;
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: report field magnitude"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         for (int comp = 0; comp < 3; ++comp) {
            const Real v = field[lid][comp];
            if (!std::isfinite(v)) { nonFiniteLocal = true; }
            if (std::abs(v) > maxAbsLocal[comp]) { maxAbsLocal[comp] = std::abs(v); }
         }
      });
   double maxAbsGlobal[3] = {0.0, 0.0, 0.0};
   MPI_Allreduce(maxAbsLocal, maxAbsGlobal, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   int nonFiniteLocalInt = nonFiniteLocal ? 1 : 0, nonFiniteGlobal = 0;
   MPI_Allreduce(&nonFiniteLocalInt, &nonFiniteGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   if (myRank == MASTER_RANK) {
      fprintf(stderr, "%s: max|Ex|=%e max|Ey|=%e max|Ez|=%e%s\n", label,
              maxAbsGlobal[0], maxAbsGlobal[1], maxAbsGlobal[2],
              nonFiniteGlobal ? "  *** NaN/Inf ***" : "");
   }
}

// Low-pass filter, 3-point binomial with alpha=1/2
// Transfer function T(k) = cos^2(k*dx/2)
// Two-pass: computes every filtered value into a local buffer reading
// neighbors from the original, then copies the buffer back.
template<typename SpanType>
static void ap_ApplyLowPassFilter1D(SpanType field, int axis, fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   const auto localSize = fsgrid.getLocalSize();
   const int lx = localSize[0], ly = localSize[1], lz = localSize[2];
   std::vector<std::array<Real,3>> filtered((size_t)lx*ly*lz);

   std::array<int,3> plusOffset{0,0,0};  plusOffset[axis]  =  1;
   std::array<int,3> minusOffset{0,0,0}; minusOffset[axis] = -1;

   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: low-pass filter, compute"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);
         const bool minusExists = stencil.cellExists(minusOffset[0], minusOffset[1], minusOffset[2]);
         const bool plusExists  = stencil.cellExists(plusOffset[0],  plusOffset[1],  plusOffset[2]);
         const size_t minusLid = minusExists ? stencil.indexFromOffset(minusOffset[0], minusOffset[1], minusOffset[2]) : lid;
         const size_t plusLid  = plusExists  ? stencil.indexFromOffset(plusOffset[0],  plusOffset[1],  plusOffset[2])  : lid;
         for (int comp = 0; comp < 3; ++comp) {
            Real val = 0.5 * field[lid][comp];
            val += 0.25 * field[minusLid][comp];
            val += 0.25 * field[plusLid][comp];
            filtered[lidx][comp] = val;
         }
      });

   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: low-pass filter, write back"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);
         for (int comp = 0; comp < 3; ++comp) {
            field[lid][comp] = filtered[lidx][comp];
         }
      });
   fsgrid.updateGhostCells(field);
}

// Separable filter in all three dimensions
template<typename SpanType>
static void ap_ApplyLowPassFilter3D(SpanType field, fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   ap_ApplyLowPassFilter1D(field, 0, technical, fsgrid); // x
   ap_ApplyLowPassFilter1D(field, 1, technical, fsgrid); // y
   ap_ApplyLowPassFilter1D(field, 2, technical, fsgrid); // z
}

bool ap_propagateFields(fsgrids::perbspan perb,
                     fsgrids::perbspan perbdt2,
                     fsgrids::efieldspan e,
                     fsgrids::efieldspan edt2,
                     fsgrids::momentsspan moments,
                     std::vector<fsgrids::speciesrhoqspan>& speciesRhoQ,
                     std::vector<fsgrids::speciesjspan>& speciesJ,
                     fsgrids::dperbspan dperb,
                     fsgrids::dmomentsspan dmoments,
                     fsgrids::bgbspan bgb,
                     fsgrids::volspan vol,
                     fsgrids::technicalspan technical, FieldSolverGrid &fsgrid,
                     creal& dt, cuint subcycles) {

   if (subcycles != 1) {
      stringstream s;
      s << "propagateFields (AP/CSL-RME): subcycles=" << subcycles << " requested, but the "
        << "implicit Eq. (38) solve doesn't subcycle. Callers must pass subcycles=1.";
      bailout(true, s.str(), __FILE__, __LINE__);
   }

   calculateDerivativesSimple(perb, moments, dperb, dmoments, technical, fsgrid, false);
   fsgrid.updateGhostCells(dperb);

   const Real theta = P::FieldSolverTheta;
   const bool apEnableLowPassFilter = true;
   const Real c = physicalconstants::LIGHT_SPEED;

   // dt==0 special case: vlasiator.cpp calls this once before the main loop
   // specifically for one-time setup. ap_SolveElectricField divides by dt and
   // dt^2 in multiple places, so calling it with dt=0 produces inf/NaN, which
   // then gets written into e and read back as the "initial guess" Ek on every
   // subsequent real solve
   // FIXME: This is done to get the timestep limit of the fieldsolver to the
   // rest of the code, but what ARE the timestep limits?
   bool converged = true;
   if (dt != 0.0) {

      std::vector<std::array<Real,9>> mu;
      std::vector<std::array<Real,3>> Jhat;
      ap_BuildSpeciesTensors(speciesRhoQ, speciesJ, perb, bgb, technical, fsgrid, theta, dt, mu, Jhat);

      // Snapshot E^k into its own storage BEFORE the solve overwrites e
      // in place with E^{k+1}.
      fsgrid::FsData<std::array<Real, fsgrids::efield::N_EFIELD>> eOldSnapshot(fsgrid.getNumStorageCells());
      fsgrid.serial_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("AP: snapshot E^k"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const size_t lid = stencil.ooo();
            eOldSnapshot[lid][0] = e[lid][0];
            eOldSnapshot[lid][1] = e[lid][1];
            eOldSnapshot[lid][2] = e[lid][2];
         });

      converged = ap_SolveElectricField(e, edt2, perb, bgb, dperb, mu, Jhat, technical, fsgrid, c, theta, dt);
      ap_ReportFieldMagnitude("ap_propagateFields: after raw solve (e)", e, technical, fsgrid);

      if (apEnableLowPassFilter) {
         ap_ApplyLowPassFilter3D(e, technical, fsgrid);
         ap_ApplyLowPassFilter3D(edt2, technical, fsgrid); // same filter, now also applied to E^{k+theta}
      }

      if (P::apEnforceGaussLaw) {
         ap_GaussLawCorrection(e, edt2, eOldSnapshot.view(), moments, mu, technical, fsgrid, dt);
      }

      ap_StageElectricFieldForAcceleration(edt2, vol, technical, fsgrid);
   }

   ap_UpdateMagneticField(perb, perbdt2, edt2 /* = E^{k+theta} */, technical, fsgrid, theta, dt);

   // Populate vol's dPERB?VOLd? and CURVATURE?  from perbdt2/edt2
   // (B^{k+theta}/E^{k+theta}).  This call ALSO writes vol's
   // PERBXVOL/YVOL/ZVOL and EXVOL/ YVOL/ZVOL so we rewrite those immediately
   // below.
   calculateVolumeAveragedFieldsSimple(perbdt2, edt2, dperb, vol, technical, fsgrid);
   ap_StageMagneticFieldForAcceleration(perbdt2, vol, technical, fsgrid);
   ap_StageElectricFieldForAcceleration(edt2, vol, technical, fsgrid);

   if (apEnableLowPassFilter) {
      ap_ApplyLowPassFilter3D(perb, technical, fsgrid);
   }

   return converged;
}
