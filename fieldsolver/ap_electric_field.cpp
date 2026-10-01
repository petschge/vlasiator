#include "ap_electric_field.hpp"
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include "../object_wrapper.h"

using namespace std;

// Everything below treats E and B as 3-vectors stored per cell.
static_assert(fsgrids::bfield::N_BFIELD == 3 && fsgrids::efield::N_EFIELD == 3, "AP solver assumes 3-component E and B");
static_assert(fsgrids::bfield::PERBX == 0 && fsgrids::bfield::PERBY == 1 && fsgrids::bfield::PERBZ == 2, "component order");
static_assert(fsgrids::efield::EX == 0 && fsgrids::efield::EY == 1 && fsgrids::efield::EZ == 2, "component order");

typedef std::array<Real,3> ApVec3;
typedef fsgrid::FsData<ApVec3> ApNodeField; // node-centred (= Vlasov cell-centre) vector field, with ghost cells

static const int kBgbVol[3] = {fsgrids::bgbfield::BGBXVOL, fsgrids::bgbfield::BGBYVOL, fsgrids::bgbfield::BGBZVOL};

/* ============================================================
 * Section 0: pure index/stencil helpers (no grid access).
 * Delimited so that they can be extracted verbatim by the standalone test.
 * ============================================================ */

// Yee <-> cell-centre incidence, as defined by volume_averages.cpp: cell i
// spans [x_i, x_{i+1}] with its centre (= the collocated node) at x_{i+1/2}.
//  * B_c(i,j,k) sits on the face between cell i-1 and i along axis c.
//  * E_c(i,j,k) sits on the edge shared by the 4 cells at offsets {0,-1}
//    along each of the two axes other than c (0 along c).

// Cells adjacent to the B_c face at a given Yee index, as offsets from that index.
static std::array<std::array<int,3>,2> faceCellOffsets(int c) {
   std::array<int,3> a{0,0,0}, b{0,0,0}; b[c] = -1;
   return {a, b};
}
// Cells adjacent to the E_c edge at a given Yee index.
static std::array<std::array<int,3>,4> edgeCellOffsets(int c) {
   const int p = (c+1)%3, q = (c+2)%3;
   std::array<std::array<int,3>,4> offs{};
   int n = 0;
   for (int dp = 0; dp >= -1; --dp) {
      for (int dq = 0; dq >= -1; --dq) {
         std::array<int,3> o{0,0,0}; o[p] = dp; o[q] = dq; offs[n++] = o;
      }
   }
   return offs;
}
// Yee B_c samples belonging to a cell (its two faces along c), as offsets from the cell index.
static std::array<std::array<int,3>,2> cellFaceOffsets(int c) {
   std::array<int,3> a{0,0,0}, b{0,0,0}; b[c] = 1;
   return {a, b};
}
// Yee E_c samples belonging to a cell (its 4 edges parallel to c), as offsets from the cell index.
static std::array<std::array<int,3>,4> cellEdgeOffsets(int c) {
   const int p = (c+1)%3, q = (c+2)%3;
   std::array<std::array<int,3>,4> offs{};
   int n = 0;
   for (int dp = 0; dp <= 1; ++dp) {
      for (int dq = 0; dq <= 1; ++dq) {
         std::array<int,3> o{0,0,0}; o[p] = dp; o[q] = dq; offs[n++] = o;
      }
   }
   return offs;
}

// Centred discrete curl on the collocated grid.
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

// Composed curl-curl stencil for output E-component `outComp`: the centred
// curl composed with itself. Using the identical operator for Faraday and
// Ampere is what makes the scheme satisfy the discrete Poynting theorem.
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

// One row of the Eq. (38) matrix for E-component `comp` at a node:
//   (1/dt^2) E + theta^2/eps0 mu.E + c^2 theta^2 curl curl E.
// All quantities are collocated, so the reaction term is purely local.
struct MatEntry { int comp; int di, dj, dk; Real val; };

static std::vector<MatEntry> ap_EquationRow(int comp, const std::array<Real,9>& mu,
                                            const std::array<Real,3>& dxyz,
                                            Real reactionScale, Real muScale, Real curlScale) {
   std::vector<MatEntry> row;
   for (int cc = 0; cc < 3; ++cc) {
      const Real val = (cc == comp ? reactionScale : 0.0) + muScale*mu[comp*3+cc];
      if (val != 0.0) { row.push_back({cc, 0,0,0, val}); }
   }
   for (const auto& t : curlCurlStencil(comp, dxyz)) {
      row.push_back({t.comp, t.di, t.dj, t.dk, curlScale*t.coeff});
   }
   return row;
}

/* ============================================================
 * Section 1: persistent node-centred state and Yee <-> node interpolation
 * ============================================================ */

namespace {
struct ApNodeState {
   ApNodeField B; // perturbed B^k at nodes (background field is added where needed)
   ApNodeField E; // E^k at nodes
   explicit ApNodeState(size_t n) : B(n), E(n) {}
};
std::unique_ptr<ApNodeState> apNodeState;
} // namespace

// Yee -> node: the inverse pairing of the node -> Yee map below. E is exactly
// the 4-edge mean that calculateVolumeAveragedFields uses for EXVOL/EYVOL/EZVOL;
// B is the mean of the two faces of the cell (the legacy PERB?VOL without its
// slope-limited curvature correction, so that no nonlinear limiter enters).
static void ap_YeeToNodes(fsgrids::perbspan perb, fsgrids::efieldspan e,
                          std::span<ApVec3> Bn, std::span<ApVec3> En,
                          fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   fsgrid.updateGhostCells(perb);
   fsgrid.updateGhostCells(e);
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: Yee -> node"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         for (int c = 0; c < 3; ++c) {
            Real acc = 0.0; int n = 0;
            for (const auto& o : cellFaceOffsets(c)) {
               if (!stencil.cellExists(o[0],o[1],o[2])) { continue; }
               acc += perb[stencil.indexFromOffset(o[0],o[1],o[2])][c]; ++n;
            }
            Bn[lid][c] = n > 0 ? acc/n : perb[lid][c];
            acc = 0.0; n = 0;
            for (const auto& o : cellEdgeOffsets(c)) {
               if (!stencil.cellExists(o[0],o[1],o[2])) { continue; }
               acc += e[stencil.indexFromOffset(o[0],o[1],o[2])][c]; ++n;
            }
            En[lid][c] = n > 0 ? acc/n : e[lid][c];
         }
      });
   fsgrid.updateGhostCells(Bn);
   fsgrid.updateGhostCells(En);
}

// Node -> Yee, for output. B_c: mean of the two cells sharing the face.
// E_c: mean of the 4 cells sharing the edge. (Transpose of the map above.)
static void ap_NodesToYee(std::span<const ApVec3> Bn, std::span<const ApVec3> En,
                          fsgrids::perbspan perbOut, fsgrids::efieldspan eOut,
                          fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: node -> Yee"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         for (int c = 0; c < 3; ++c) {
            Real acc = 0.0; int n = 0;
            for (const auto& o : faceCellOffsets(c)) {
               if (!stencil.cellExists(o[0],o[1],o[2])) { continue; }
               acc += Bn[stencil.indexFromOffset(o[0],o[1],o[2])][c]; ++n;
            }
            perbOut[lid][c] = n > 0 ? acc/n : Bn[lid][c];
            acc = 0.0; n = 0;
            for (const auto& o : edgeCellOffsets(c)) {
               if (!stencil.cellExists(o[0],o[1],o[2])) { continue; }
               acc += En[stencil.indexFromOffset(o[0],o[1],o[2])][c]; ++n;
            }
            eOut[lid][c] = n > 0 ? acc/n : En[lid][c];
         }
      });
   fsgrid.updateGhostCells(perbOut);
   fsgrid.updateGhostCells(eOut);
}

// The persistent state is created (and filled from the Yee arrays) on first use
// and whenever the dt==0 setup call is made, which covers fresh starts (project
// initial conditions) and restarts (first real step).
static ApNodeState& ap_GetNodeState(fsgrids::perbspan perb, fsgrids::efieldspan e,
                                    fsgrids::technicalspan technical, FieldSolverGrid& fsgrid,
                                    bool forceRebuild) {
   if (forceRebuild || !apNodeState || apNodeState->B.size() != (size_t)fsgrid.getNumStorageCells()) {
      apNodeState = std::make_unique<ApNodeState>(fsgrid.getNumStorageCells());
      ap_YeeToNodes(perb, e, apNodeState->B.view(), apNodeState->E.view(), technical, fsgrid);
   }
   return *apNodeState;
}

// Write the fields the Vlasov push will use straight into the volume-averaged
// slots of vol. No interpolation: the node values are the cell-centre values.
static void ap_PublishToVol(std::span<const ApVec3> B, std::span<const ApVec3> E, fsgrids::volspan vol,
                            fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: publish fields to vol"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         vol[lid][fsgrids::volfields::PERBXVOL] = B[lid][0];
         vol[lid][fsgrids::volfields::PERBYVOL] = B[lid][1];
         vol[lid][fsgrids::volfields::PERBZVOL] = B[lid][2];
         vol[lid][fsgrids::volfields::EXVOL]    = E[lid][0];
         vol[lid][fsgrids::volfields::EYVOL]    = E[lid][1];
         vol[lid][fsgrids::volfields::EZVOL]    = E[lid][2];
      });
   fsgrid.updateGhostCells(vol);
}

/* ============================================================
 * Section 2: alpha_s / mu / J-hat  (Eqs. 35-37), evaluated at the nodes
 * ============================================================ */

void ap_BuildSpeciesTensors(
   std::vector<fsgrids::speciesrhoqspan>& speciesRhoQ,
   std::vector<fsgrids::speciesjspan>& speciesJ,
   std::span<const ApVec3> Bnode,
   fsgrids::constbgbspan bgb,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real theta,
   Real dt,
   fsgrid::FsData<std::array<Real,9>>& outMu,
   ApNodeField& outJhat
) {
   const uint numPops = speciesRhoQ.size();

   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: build alpha_s, mu, J-hat"),
      technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {

         const size_t lid = stencil.ooo();

         // Total B at the node = perturbed (node-centred state) + background (volume average)
         const Real Bx = Bnode[lid][0] + bgb[lid][kBgbVol[0]];
         const Real By = Bnode[lid][1] + bgb[lid][kBgbVol[1]];
         const Real Bz = Bnode[lid][2] + bgb[lid][kBgbVol[2]];

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

         outMu[lid]   = mu;
         outJhat[lid] = jhat;
      });
   // mu is read at neighbouring nodes by the Gauss-correction operator
   fsgrid.updateGhostCells(outMu.view());
   fsgrid.updateGhostCells(outJhat.view());
}

/* ============================================================
 * Section 3: Eq. (38) -- HYPRE IJ assembly + BoomerAMG-preconditioned GMRES
 * ============================================================ */

bool ap_SolveElectricField(
   std::span<const ApVec3> Ek,
   std::span<const ApVec3> Bk,
   fsgrids::constbgbspan bgb,
   std::span<const std::array<Real,9>> mu,
   std::span<const ApVec3> Jhat,
   std::span<ApVec3> Etheta,   // out: E^{k+theta}
   std::span<ApVec3> Ekp1,     // out: E^{k+1}  (Eq. 40)
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

   auto localLinear = [lx,ly](int i, int j, int k) -> long long {
      return i + lx*((long long)j + ly*k);
   };

   // DOF base per cell, stored so it can be ghost-exchanged like any other fsgrid quantity.
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

   auto globalDof = [&](size_t neighborLid, int comp) -> HYPRE_BigInt {
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

         for (int comp = 0; comp < 3; ++comp) {
            const HYPRE_BigInt row = globalDof(lid, comp);

            std::map<HYPRE_BigInt, Real> rowVals;
            for (const auto& entry : ap_EquationRow(comp, mu[lid], dxyz, reactionScale, muScale, curlScale)) {
               if (!stencil.cellExists(entry.di, entry.dj, entry.dk)) { continue; }
               const size_t nlid = stencil.indexFromOffset(entry.di, entry.dj, entry.dk);
               rowVals[globalDof(nlid, entry.comp)] += entry.val;
            }

            std::vector<HYPRE_BigInt> cols; cols.reserve(rowVals.size());
            std::vector<HYPRE_Real> vals; vals.reserve(rowVals.size());
            for (const auto& kv : rowVals) { cols.push_back(kv.first); vals.push_back(kv.second); }
            HYPRE_Int ncols = (HYPRE_Int)cols.size();
            HYPRE_IJMatrixSetValues(Aij, 1, &ncols, &row, cols.data(), vals.data());

            // curl(B^k) with the same centred curl Faraday uses (NOT via dperb,
            // which uses a nonlinear TVD slope limiter). Total B = perturbed + background.
            Real curlB = 0.0;
            for (const auto& term : curlStencilCentered(comp, dxyz)) {
               if (!stencil.cellExists(term.di, term.dj, term.dk)) { continue; }
               const size_t nlid = stencil.indexFromOffset(term.di, term.dj, term.dk);
               curlB += term.coeff * (Bk[nlid][term.comp] + bgb[nlid][kBgbVol[term.comp]]);
            }

            // RHS of Eq. (38): (1/dt^2) E^k + (c^2 theta/dt) curl(B^k) - (theta/(dt*EPS_0)) J-hat
            const Real Ekc = Ek[lid][comp];
            const Real rhs = reactionScale*Ekc + c*c*theta/dt*curlB - theta/(dt*physicalconstants::EPS_0)*Jhat[lid][comp];

            HYPRE_IJVectorSetValues(bij, 1, &row, &rhs);
            const Real x0 = Ekc;
            HYPRE_IJVectorSetValues(xij, 1, &row, &x0); // initial guess = E^k
         }
      });

   HYPRE_IJMatrixAssemble(Aij);
   HYPRE_IJVectorAssemble(bij);
   HYPRE_IJVectorAssemble(xij);

   HYPRE_ParCSRMatrix Apar; HYPRE_IJMatrixGetObject(Aij, (void**)&Apar);
   HYPRE_ParVector bpar; HYPRE_IJVectorGetObject(bij, (void**)&bpar);
   HYPRE_ParVector xpar; HYPRE_IJVectorGetObject(xij, (void**)&xpar);

   // ---- Step 3: solve. BoomerAMG-preconditioned GMRES because mu's I x B
   // term makes the reaction block non-symmetric
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
            const HYPRE_BigInt row = globalDof(lid, comp);
            HYPRE_Real val;
            HYPRE_IJVectorGetValues(xij, 1, &row, &val);
            const Real Ektheta = (Real)val;
            const Real Ekc = Ek[lid][comp];
            Etheta[lid][comp] = Ektheta;
            Ekp1[lid][comp]   = (Ektheta - Ekc)/theta + Ekc;   // Eq. 40
         }
      });

   HYPRE_ParCSRGMRESDestroy(gmres);
   HYPRE_BoomerAMGDestroy(amg);
   HYPRE_IJMatrixDestroy(Aij);
   HYPRE_IJVectorDestroy(bij);
   HYPRE_IJVectorDestroy(xij);

   fsgrid.updateGhostCells(Etheta);
   fsgrid.updateGhostCells(Ekp1);
   return (relres < 1e-6); // FIXME: what convergence threshold the rest of the solver treats as success
}

/* ============================================================
 * Section 4: Faraday update (Eq. 23, 39), on the nodes
 * ============================================================ */

void ap_UpdateMagneticField(
   std::span<const ApVec3> Bk,
   std::span<const ApVec3> Etheta,
   std::span<ApVec3> Bkp1,
   std::span<ApVec3> Btheta,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   Real theta,
   Real dt
) {
   const auto dxyz = fsgrid.getGridSpacing();
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: Faraday update"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         for (int comp = 0; comp < 3; ++comp) {
            Real curlE = 0.0;
            for (const auto& term : curlStencilCentered(comp, dxyz)) {
               if (!stencil.cellExists(term.di, term.dj, term.dk)) { continue; }
               curlE += term.coeff * Etheta[stencil.indexFromOffset(term.di, term.dj, term.dk)][term.comp];
            }
            const Real Bkc = Bk[lid][comp];
            const Real Bk1 = Bkc - dt*curlE;                    // Eq. 23
            Bkp1[lid][comp]   = Bk1;                            // B^{k+1}
            Btheta[lid][comp] = theta*Bk1 + (1.0-theta)*Bkc;    // Eq. 39
         }
      });
   fsgrid.updateGhostCells(Bkp1);
   fsgrid.updateGhostCells(Btheta);
}

/* ============================================================
 * Section 5: Gauss's-law correction (Eq. 45, 41), on the nodes
 * ============================================================
 * As in Algorithm 3.4 step (2d), only E^{k+1} is corrected (it becomes the
 * next step's E^k); E^{k+theta}, which the Vlasov push already uses, is not.
 */

void ap_GaussLawCorrection(
   std::span<ApVec3> Ekp1,
   std::span<const ApVec3> Ek,
   std::span<const std::array<Real,1>> rho,   // rho^k at the nodes
   std::span<const std::array<Real,9>> mu,    // mu^k at the nodes (ghost-exchanged)
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
            sumRho_local += rho[stencil.ooo()][0] / physicalconstants::EPS_0;
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
         const auto& m = mu[lid];
         double* mv = &mvals[nStencil*lidx];

         // -div[(I + (dt^2/EPS_0) diag(mu)) grad phi], FACE-AVERAGED: the +x
         // entry of this node and the -x entry of its +x neighbour describe the
         // same face and must agree, so the coefficient at a face is the mean of
         // the two adjacent nodes. mu is ghost-exchanged, so this is also
         // symmetric across periodic seams and MPI rank boundaries.
         const bool hasXm = stencil.cellExists(-1,0,0), hasXp = stencil.cellExists(1,0,0);
         const bool hasYm = stencil.cellExists(0,-1,0), hasYp = stencil.cellExists(0,1,0);
         const bool hasZm = stencil.cellExists(0,0,-1), hasZp = stencil.cellExists(0,0,1);
         const size_t lidXm = hasXm ? stencil.indexFromOffset(-1,0,0) : lid;
         const size_t lidXp = hasXp ? stencil.indexFromOffset(1,0,0)  : lid;
         const size_t lidYm = hasYm ? stencil.indexFromOffset(0,-1,0) : lid;
         const size_t lidYp = hasYp ? stencil.indexFromOffset(0,1,0)  : lid;
         const size_t lidZm = hasZm ? stencil.indexFromOffset(0,0,-1) : lid;
         const size_t lidZp = hasZp ? stencil.indexFromOffset(0,0,1)  : lid;

         const Real cxx_here = 1.0 + dt*dt*m[0]/physicalconstants::EPS_0;
         const Real cyy_here = 1.0 + dt*dt*m[4]/physicalconstants::EPS_0;
         const Real czz_here = 1.0 + dt*dt*m[8]/physicalconstants::EPS_0;
         const Real cxx_xm = 1.0 + dt*dt*mu[lidXm][0]/physicalconstants::EPS_0;
         const Real cxx_xp = 1.0 + dt*dt*mu[lidXp][0]/physicalconstants::EPS_0;
         const Real cyy_ym = 1.0 + dt*dt*mu[lidYm][4]/physicalconstants::EPS_0;
         const Real cyy_yp = 1.0 + dt*dt*mu[lidYp][4]/physicalconstants::EPS_0;
         const Real czz_zm = 1.0 + dt*dt*mu[lidZm][8]/physicalconstants::EPS_0;
         const Real czz_zp = 1.0 + dt*dt*mu[lidZp][8]/physicalconstants::EPS_0;

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

         // Cross terms from mu's off-diagonal (symmetrised)
         const Real cxy = dt*dt*0.5*(m[1]+m[3])/physicalconstants::EPS_0;
         const Real cxz = dt*dt*0.5*(m[2]+m[6])/physicalconstants::EPS_0;
         const Real cyz = dt*dt*0.5*(m[5]+m[7])/physicalconstants::EPS_0;
         const Real kxy = -cxy/(2*dxyz[0]*dxyz[1]);
         const Real kxz = -cxz/(2*dxyz[0]*dxyz[2]);
         const Real kyz = -cyz/(2*dxyz[1]*dxyz[2]);
         mv[7]=mv[8] = kxy; mv[9]=mv[10] = -kxy;   // (+1,+1,0)/(-1,-1,0) vs (+1,-1,0)/(-1,+1,0)
         mv[11]=mv[12] = kxz; mv[13]=mv[14] = -kxz;
         mv[15]=mv[16] = kyz; mv[17]=mv[18] = -kyz;

         // div(E^k), centred differences of the node-centred E^k (ghost-exchanged).
         Real divE = 0.0;
         if (hasXp && hasXm) { divE += (Ek[lidXp][0] - Ek[lidXm][0])/(2*dxyz[0]); }
         if (hasYp && hasYm) { divE += (Ek[lidYp][1] - Ek[lidYm][1])/(2*dxyz[1]); }
         if (hasZp && hasZm) { divE += (Ek[lidZp][2] - Ek[lidZm][2])/(2*dxyz[2]); }

         const Real rhoOverEpsRaw = rho[lid][0]/physicalconstants::EPS_0;
         bvals[lidx] = (rhoOverEpsRaw - meanRhoOverEps) - divE; // Eq. 45 RHS, SI, rescaled by 1/EPS_0, rho mean-subtracted
      });

   // Hand mvals/bvals to HYPRE via IJ, using dofBase for global row/column indices
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

   // Eq. 41: E~^{k+1} = E^{k+1} - grad(phi). E^{k+theta} is not directly
   // corrected by Eq. 41 -- only E^{k+1} is -- so to stay consistent with
   // Eq. 40 (E^{k+theta} = theta*E^{k+1} + (1-theta)*E^k, with E^k left as
   // is here) the correction applied to edt2 must be theta*(-grad(phi)),
   // not the full -grad(phi) applied to e.
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
               Ekp1[lid][0] -= 0.5*(phiGrid[stencil.indexFromOffset(1,0,0)][0]-phiGrid[stencil.indexFromOffset(-1,0,0)][0])/dxyz[0];
            }
            if (stencil.cellExists(0,1,0) && stencil.cellExists(0,-1,0)) {
               Ekp1[lid][1] -= 0.5*(phiGrid[stencil.indexFromOffset(0,1,0)][0]-phiGrid[stencil.indexFromOffset(0,-1,0)][0])/dxyz[1];
            }
            if (stencil.cellExists(0,0,1) && stencil.cellExists(0,0,-1)) {
               Ekp1[lid][2] -= 0.5*(phiGrid[stencil.indexFromOffset(0,0,1)][0]-phiGrid[stencil.indexFromOffset(0,0,-1)][0])/dxyz[2];
            }
         });
      fsgrid.updateGhostCells(Ekp1);

   } else if (myRank == MASTER_RANK) {
      fprintf(stderr, "apGaussLawCorrection: relres=%e > 1.0 -- solve diverged, "
                      "SKIPPING correction this step, E left unmodified\n", (double)pcgRelres);
   }
}

/* ============================================================
 * Section 6: diagnostics and the optional low-pass filter
 * ============================================================ */

// Reports max|E| (over all 3 components, all local cells)
static void ap_ReportFieldMagnitude(const char* label, std::span<const ApVec3> field,
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

// Low-pass filter, 3-point binomial with alpha=1/2 (transfer function
// cos^2(k*dx/2)). Two-pass: every filtered value is computed from the
// original, then copied back. Not part of the paper; off unless
// fieldsolver.lowPassFilter is set.
static void ap_ApplyLowPassFilter1D(std::span<ApVec3> field, int axis, fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   const auto localSize = fsgrid.getLocalSize();
   const int lx = localSize[0], ly = localSize[1], lz = localSize[2];
   std::vector<ApVec3> filtered((size_t)lx*ly*lz);

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
            filtered[lidx][comp] = 0.5*field[lid][comp] + 0.25*field[minusLid][comp] + 0.25*field[plusLid][comp];
         }
      });

   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("AP: low-pass filter, write back"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);
         for (int comp = 0; comp < 3; ++comp) { field[lid][comp] = filtered[lidx][comp]; }
      });
   fsgrid.updateGhostCells(field);
}

// Separable filter in all three dimensions
static void ap_ApplyLowPassFilter3D(std::span<ApVec3> field, fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   ap_ApplyLowPassFilter1D(field, 0, technical, fsgrid);
   ap_ApplyLowPassFilter1D(field, 1, technical, fsgrid);
   ap_ApplyLowPassFilter1D(field, 2, technical, fsgrid);
}

/* ============================================================
 * Section 7: propagateFields entry point
 * ============================================================ */

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

   // Derivatives of the Yee-lattice perb and of the moments, plus vol's derivative
   // and curvature slots (computed from vol's own PERB?VOL, on the same one-step
   // lag as for the other field solvers). Nothing in the solver below reads dperb.
   calculateDerivativesSimple(perb, moments, dperb, dmoments, technical, fsgrid, false);
   fsgrid.updateGhostCells(dperb);

   const Real theta = P::FieldSolverTheta;
   const Real c = physicalconstants::LIGHT_SPEED;

   ApNodeState& state = ap_GetNodeState(perb, e, technical, fsgrid, dt==0.0);

   // dt==0 special case: vlasiator.cpp calls this once before the main loop
   // specifically for one-time setup. ap_SolveElectricField divides by dt and
   // dt^2, so nothing is advanced. The initial node-centred fields are
   // published to vol (the Vlasov solver reads them from there); the Yee arrays
   // are left exactly as the project set them.
   // FIXME: This is also how the timestep limit of the fieldsolver gets to the
   // rest of the code, but what ARE the timestep limits?
   if (dt == 0.0) {
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("AP: initial half-step Yee arrays"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const size_t lid = stencil.ooo();
            perbdt2[lid] = perb[lid];
            edt2[lid]    = e[lid];
         });
      fsgrid.updateGhostCells(perbdt2);
      fsgrid.updateGhostCells(edt2);
      ap_PublishToVol(state.B.view(), state.E.view(), vol, technical, fsgrid);
      return true;
   }

   const size_t nStorage = fsgrid.getNumStorageCells();
   fsgrid::FsData<std::array<Real,9>> mu(nStorage);
   ApNodeField Jhat(nStorage);
   ap_BuildSpeciesTensors(speciesRhoQ, speciesJ, state.B.view(), bgb, technical, fsgrid, theta, dt, mu, Jhat);

   // Step (2b): E^{k+theta} from Eq. (38), E^{k+1} from Eq. (40)
   ApNodeField Etheta(nStorage), Ekp1(nStorage), Btheta(nStorage), Bkp1(nStorage);
   const bool converged = ap_SolveElectricField(state.E.view(), state.B.view(), bgb, mu.view(), Jhat.view(),
                                                Etheta.view(), Ekp1.view(), technical, fsgrid, c, theta, dt);
   ap_ReportFieldMagnitude("ap_propagateFields: after raw solve (E^{k+theta})", Etheta.view(), technical, fsgrid);

   if (P::apLowPassFilter) {
      ap_ApplyLowPassFilter3D(Etheta.view(), technical, fsgrid);
      ap_ApplyLowPassFilter3D(Ekp1.view(), technical, fsgrid);
   }

   // Step (2c): B^{k+1} from Eq. (23), B^{k+theta} from Eq. (39)
   ap_UpdateMagneticField(state.B.view(), Etheta.view(), Bkp1.view(), Btheta.view(), technical, fsgrid, theta, dt);

   // Step (2d): Gauss correction of E^{k+1} only. rho^k is the total charge density
   // at the start of the step, the same speciesRhoQ that defines mu (Eq. 36).
   if (P::apEnforceGaussLaw) {
      fsgrid::FsData<std::array<Real,1>> rho(nStorage);
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("AP: total rho^k"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const size_t lid = stencil.ooo();
            Real sum = 0.0;
            for (size_t popID = 0; popID < speciesRhoQ.size(); ++popID) {
               sum += speciesRhoQ[popID][lid][fsgrids::speciesrhoq::SRHOQ];
            }
            rho[lid][0] = sum;
         });
      ap_GaussLawCorrection(Ekp1.view(), state.E.view(), rho.view(), mu.view(), technical, fsgrid, dt);
   }

   if (P::apLowPassFilter) {
      ap_ApplyLowPassFilter3D(Bkp1.view(), technical, fsgrid);
      ap_ApplyLowPassFilter3D(Btheta.view(), technical, fsgrid);
   }

   // Hand-off to the Vlasov solver (Eq. 20): E^{k+theta}, B^{k+theta} straight into vol.
   ap_PublishToVol(Btheta.view(), Etheta.view(), vol, technical, fsgrid);

   // Yee-lattice quantities for output and for everything else that expects them.
   ap_NodesToYee(Btheta.view(), Etheta.view(), perbdt2, edt2, technical, fsgrid); // time level k+theta
   ap_NodesToYee(Bkp1.view(),   Ekp1.view(),   perb,    e,    technical, fsgrid); // time level k+1

   // The corrected E^{k+1} and B^{k+1} become E^k and B^k of the next step.
   state.E.swap(Ekp1);
   state.B.swap(Bkp1);

   return converged;
}
