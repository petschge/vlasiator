#include "lccl_electric_field.hpp"
#include "lccl_restart.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include "../object_wrapper.h"

using namespace std;

// Everything below treats E and B as 3-vectors stored per cell.
static_assert(fsgrids::bfield::N_BFIELD == 3 && fsgrids::efield::N_EFIELD == 3, "LCCL solver assumes 3-component E and B");
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

static std::vector<MatEntry> lccl_EquationRow(int comp, const std::array<Real,9>& mu,
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
std::unique_ptr<ApNodeState> lcclNodeState;
} // namespace

// Yee -> node: the inverse pairing of the node -> Yee map below. E is exactly
// the 4-edge mean that calculateVolumeAveragedFields uses for EXVOL/EYVOL/EZVOL;
// B is the mean of the two faces of the cell (the legacy PERB?VOL without its
// slope-limited curvature correction, so that no nonlinear limiter enters).
static void lccl_YeeToNodes(fsgrids::perbspan perb, fsgrids::efieldspan e,
                          std::span<ApVec3> Bn, std::span<ApVec3> En,
                          fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   fsgrid.updateGhostCells(perb);
   fsgrid.updateGhostCells(e);
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: Yee -> node"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { return; }
         const size_t lid = stencil.ooo();
         for (int c = 0; c < 3; ++c) {
            Real acc = 0.0; int n = 0;
            for (const auto& o : cellFaceOffsets(c)) {
               if (!stencil.cellExists(o[0],o[1],o[2])) { continue; }
               const size_t nlid = stencil.indexFromOffset(o[0],o[1],o[2]);
               if (technical[nlid].sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || technical[nlid].sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { continue; }
               acc += perb[nlid][c]; ++n;
            }
            Bn[lid][c] = n > 0 ? acc/n : perb[lid][c];
            acc = 0.0; n = 0;
            for (const auto& o : cellEdgeOffsets(c)) {
               if (!stencil.cellExists(o[0],o[1],o[2])) { continue; }
               const size_t nlid = stencil.indexFromOffset(o[0],o[1],o[2]);
               if (technical[nlid].sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || technical[nlid].sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { continue; }
               acc += e[nlid][c]; ++n;
            }
            En[lid][c] = n > 0 ? acc/n : e[lid][c];
         }
      });
   fsgrid.updateGhostCells(Bn);
   fsgrid.updateGhostCells(En);
}

// Node -> Yee, for output. B_c: mean of the two cells sharing the face.
// E_c: mean of the 4 cells sharing the edge. (Transpose of the map above.)
static void lccl_NodesToYee(std::span<const ApVec3> Bn, std::span<const ApVec3> En,
                          fsgrids::perbspan perbOut, fsgrids::efieldspan eOut,
                          fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: node -> Yee"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { return; }
         const size_t lid = stencil.ooo();
         for (int c = 0; c < 3; ++c) {
            Real acc = 0.0; int n = 0;
            for (const auto& o : faceCellOffsets(c)) {
               if (!stencil.cellExists(o[0],o[1],o[2])) { continue; }
               const size_t nlid = stencil.indexFromOffset(o[0],o[1],o[2]);
               if (technical[nlid].sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || technical[nlid].sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { continue; }
               acc += Bn[nlid][c]; ++n;
            }
            perbOut[lid][c] = n > 0 ? acc/n : Bn[lid][c];
            acc = 0.0; n = 0;
            for (const auto& o : edgeCellOffsets(c)) {
               if (!stencil.cellExists(o[0],o[1],o[2])) { continue; }
               const size_t nlid = stencil.indexFromOffset(o[0],o[1],o[2]);
               if (technical[nlid].sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || technical[nlid].sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { continue; }
               acc += En[nlid][c]; ++n;
            }
            eOut[lid][c] = n > 0 ? acc/n : En[lid][c];
         }
      });
   fsgrid.updateGhostCells(perbOut);
   fsgrid.updateGhostCells(eOut);
}

// The persistent state is filled from the Yee arrays by the dt==0 setup call of a fresh start (the
// project's initial conditions are given on the Yee lattice). A restart installs it from the restart
// file instead (lccl_RestoreNodeState, called by readGrid), because rebuilding it from the averaged Yee
// arrays smooths it; only a restart file without the node-centred fields falls back to the Yee arrays,
// on the first real step.
static ApNodeState& lccl_GetNodeState(fsgrids::perbspan perb, fsgrids::efieldspan e,
                                    fsgrids::technicalspan technical, FieldSolverGrid& fsgrid,
                                    bool forceRebuild) {
   if (forceRebuild || !lcclNodeState || lcclNodeState->B.size() != (size_t)fsgrid.getNumStorageCells()) {
      lcclNodeState = std::make_unique<ApNodeState>(fsgrid.getNumStorageCells());
      lccl_YeeToNodes(perb, e, lcclNodeState->B.view(), lcclNodeState->E.view(), technical, fsgrid);
   }
   return *lcclNodeState;
}

// ---- restart support (lccl_restart.hpp) ----
bool lccl_HasNodeState() {
   return static_cast<bool>(lcclNodeState);
}

std::span<const std::array<Real,3>> lccl_NodeStateE() {
   if (!lcclNodeState) { return {}; }
   return lcclNodeState->E.view();
}

std::span<const std::array<Real,3>> lccl_NodeStateB() {
   if (!lcclNodeState) { return {}; }
   return lcclNodeState->B.view();
}

void lccl_RestoreNodeState(std::span<const std::array<Real,3>> B, std::span<const std::array<Real,3>> E,
                           FieldSolverGrid& fsgrid) {
   const size_t nStorage = fsgrid.getNumStorageCells();
   if (B.size() != nStorage || E.size() != nStorage) {
      bailout(true, "lccl_RestoreNodeState: arrays do not match the field solver grid", __FILE__, __LINE__);
      return;
   }
   lcclNodeState = std::make_unique<ApNodeState>(nStorage);
   auto Bv = lcclNodeState->B.view();
   auto Ev = lcclNodeState->E.view();
   std::copy(B.begin(), B.end(), Bv.begin());
   std::copy(E.begin(), E.end(), Ev.begin());
   fsgrid.updateGhostCells(Bv);
   fsgrid.updateGhostCells(Ev);
}

// Write the fields the Vlasov push will use straight into the volume-averaged
// slots of vol. No interpolation: the node values are the cell-centre values.
static void lccl_PublishToVol(std::span<const ApVec3> B, std::span<const ApVec3> E, fsgrids::volspan vol,
                            fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: publish fields to vol"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { return; }
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
 * Section 1b: incident-wave condition on an x- Maxwellian face
 *             (fieldsolver.inflowCharacteristic)
 * ============================================================
 * Imposes the incoming characteristics and takes the outgoing ones from the
 * interior, instead of pinning B to the template and E to 0:
 *    incoming  Ey + c Bz = S_y,  Ez - c By = S_z   with S_y = c Bz_tpl, S_z = -c By_tpl
 *    outgoing  Ey - c Bz,        Ez + c By         constant along x + c t
 * Pinning E = 0 drives only one of the two sub-lattices of the centred
 * collocated stencils; this condition sets both consistently.
 * The template keeps its meaning Bz = Ey_in/c, so the input files are unchanged.
 * Normal components (Ex, Bx) stay with the boundary condition's own dispatch.
 */
namespace {
struct ApInflowState {
   ApNodeField Etheta, Ekp1, Btheta, Bkp1;
   std::vector<char> active;  // per storage cell
   explicit ApInflowState(size_t n) : Etheta(n), Ekp1(n), Btheta(n), Bkp1(n), active(n, 0) {}
};
std::unique_ptr<ApInflowState> lcclInflow;  // reset each step; null when the condition is off
} // namespace

static void lccl_PrepareInflowCharacteristic(std::span<const ApVec3> Ek, std::span<const ApVec3> Bk,
                                           fsgrids::perbspan perbYee, fsgrids::constbgbspan bgb,
                                           fsgrids::technicalspan technical, FieldSolverGrid& fsgrid,
                                           SysBoundary& sysBoundaries, Real c, Real theta, Real dt) {
   // bailout() only flags; each failure returns with the condition off so nothing below runs
   lcclInflow.reset();
   const auto globalSize = fsgrid.getGlobalSize();
   if (globalSize[1] != 1 || globalSize[2] != 1) {  // the split below is for an x-normal face
      bailout(true, "fieldsolver.inflowCharacteristic requires a grid that is 1D in x", __FILE__, __LINE__);
      return;
   }
   const auto dxyz = fsgrid.getGridSpacing();
   const Real foot = c*theta*dt/dxyz[0];  // where the outgoing characteristic was at t^k, in cells
   if (foot >= FS_STENCIL_WIDTH) {  // interpolation reads offsets floor(foot) and floor(foot)+1
      bailout(true, "fieldsolver.inflowCharacteristic: c*theta*dt/dx >= FS_STENCIL_WIDTH", __FILE__, __LINE__);
      return;
   }
   const int o0 = (int)std::floor(foot);
   const Real f = foot - o0;

   lcclInflow = std::make_unique<ApInflowState>(fsgrid.getNumStorageCells());
   ApInflowState& st = *lcclInflow;
   int unsupported_local = 0;
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: x- incident-wave boundary"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         if (sysBoundaryFlag != sysboundarytype::MAXWELLIAN) { return; }
         const auto g = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         // x- face cells are the first two layers (SysBoundaryCondition::determineFace)
         if (g[0] >= 2 || !stencil.cellExists(o0 + 1, 0, 0)) { unsupported_local = 1; return; }
         const size_t lid = stencil.ooo();
         const size_t a = (o0 == 0) ? lid : stencil.indexFromOffset(o0, 0, 0);
         const size_t b = stencil.indexFromOffset(o0 + 1, 0, 0);

         SBC::SysBoundaryCondition* bc = sysBoundaries.getSysBoundary(sysBoundaryFlag);
         const Real BtplY = bc->fieldSolverBoundaryCondMagneticField(perbYee, bgb, technical, dxyz, g, stencil, 1);
         const Real BtplZ = bc->fieldSolverBoundaryCondMagneticField(perbYee, bgb, technical, dxyz, g, stencil, 2);
         const Real Sy = c*BtplZ, Sz = -c*BtplY;  // template is evaluated at t + theta*dt (vlasiator.cpp)

         const Real wm = (1.0-f)*(Ek[a][1] - c*Bk[a][2]) + f*(Ek[b][1] - c*Bk[b][2]);
         const Real um = (1.0-f)*(Ek[a][2] + c*Bk[a][1]) + f*(Ek[b][2] + c*Bk[b][1]);

         st.Etheta[lid] = {0.0, 0.5*(Sy + wm), 0.5*(Sz + um)};
         st.Btheta[lid] = {0.0, 0.5*(um - Sz)/c, 0.5*(Sy - wm)/c};
         for (int comp = 1; comp < 3; ++comp) {
            st.Ekp1[lid][comp] = (st.Etheta[lid][comp] - Ek[lid][comp])/theta + Ek[lid][comp];     // Eq. 40
            st.Bkp1[lid][comp] = (st.Btheta[lid][comp] - (1.0-theta)*Bk[lid][comp])/theta;        // Eq. 39
         }
         st.active[lid] = 1;
      });
   int unsupported = 0;
   MPI_Allreduce(&unsupported_local, &unsupported, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   if (unsupported != 0) {
      bailout(true, "fieldsolver.inflowCharacteristic supports a Maxwellian boundary on the x- face only",
              __FILE__, __LINE__);
      lcclInflow.reset();
   }
}

/* ============================================================
 * Section 2: alpha_s / mu / J-hat  (Eqs. 35-37), evaluated at the nodes
 * ============================================================ */

// Operators of the exact Lorentz push over dt with constant E and B, du/dt = (q/m)(E + u x B):
//   u(dt) = R u + Phi (q/m) E,  R = exp(W dt),  Phi = int_0^dt exp(W s) ds,  W u = u x omega,  omega = (q/m) B.
// This is what the Vlasov acceleration does (it composes 0.1-degree rotations with the E push).
// With W^3 = -|omega|^2 W:
//   R   = I + sin(psi)/|omega| W + (1 - cos psi)/|omega|^2 W^2,
//   Phi = dt I + (1 - cos psi)/|omega|^2 W + (psi - sin psi)/|omega|^3 W^2,   psi = |omega| dt.
// The coefficients are evaluated as dt^n f(psi) with series for small psi, so B = 0 is exact.
static void lccl_ExactPushOperators(Real qOverM, Real Bx, Real By, Real Bz, Real dt,
                                    std::array<Real,9>& R, std::array<Real,9>& Phi, Real& psi) {
   const Real wx = qOverM*Bx, wy = qOverM*By, wz = qOverM*Bz;
   const Real w2 = wx*wx + wy*wy + wz*wz;
   psi = std::sqrt(w2)*std::fabs(dt);
   Real sinc, c1, c2; // sin(psi)/psi, (1 - cos psi)/psi^2, (psi - sin psi)/psi^3
   if (psi < 1e-3) {
      const Real p2 = psi*psi;
      sinc = 1.0 - p2/6.0 + p2*p2/120.0;
      c1   = 0.5 - p2/24.0 + p2*p2/720.0;
      c2   = 1.0/6.0 - p2/120.0 + p2*p2/5040.0;
   } else {
      sinc = std::sin(psi)/psi;
      c1   = (1.0 - std::cos(psi))/(psi*psi);
      c2   = (psi - std::sin(psi))/(psi*psi*psi);
   }
   const Real a1 = dt*sinc, a2 = dt*dt*c1, b2 = dt*dt*dt*c2;
   const std::array<Real,9> W  = { 0.0,  wz, -wy,
                                  -wz, 0.0,  wx,
                                   wy, -wx, 0.0 };
   const std::array<Real,9> W2 = { wx*wx - w2, wx*wy,      wx*wz,
                                   wy*wx,      wy*wy - w2, wy*wz,
                                   wz*wx,      wz*wy,      wz*wz - w2 };
   for (int t = 0; t < 9; ++t) {
      const Real I = (t % 4 == 0) ? 1.0 : 0.0;
      R[t]   = I + a1*W[t] + a2*W2[t];
      Phi[t] = dt*I + a2*W[t] + b2*W2[t];
   }
}

void lccl_BuildSpeciesTensors(
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
   std::atomic<bool> gyrationBeyondPi{false};

   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: build alpha_s, mu, J-hat"),
      technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { return; }

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

            const Real charge = getObjectWrapper().particleSpecies[popID].charge;
            const Real mass   = getObjectWrapper().particleSpecies[popID].mass;
            const Real qOverM_rho = (charge/mass) * rhoQ;
            std::array<Real,9> alpha;

            if (P::lcclMatchedCurrent) {
               // Matched current prediction: the current the exact push over dt will produce from J*,
               // J^{k+theta} = (1-theta) J* + theta J^{k+1} with J^{k+1} = R J* + (q/m) rho Phi E^{k+theta},
               // i.e. J-hat with alpha_s = (1-theta) I + theta R_s and mu += (q_s/m_s) rho_s Phi_s / dt.
               // For B = 0 this is identical to Eqs. 35-37. The rotation uses B^k, as alpha_s does.
               std::array<Real,9> R, Phi;
               Real psi;
               lccl_ExactPushOperators(charge/mass, Bx, By, Bz, dt, R, Phi, psi);
               if (psi >= M_PI) { gyrationBeyondPi.store(true, std::memory_order_relaxed); }
               for (int t = 0; t < 9; ++t) {
                  const Real I = (t % 4 == 0) ? 1.0 : 0.0;
                  alpha[t] = (1.0 - theta)*I + theta*R[t];
                  mu[t] += qOverM_rho * Phi[t] / dt;
               }
            } else {
               // eps_s = q_s * theta * dt / m_s  (Eq. 37)
               const Real eps = charge * theta * dt / mass;

               const Real B2 = Bx*Bx + By*By + Bz*Bz;
               const Real denom = 1.0 + eps*eps*B2;

               // alpha_s = [I - eps * (I x B) + eps^2 * B B^T] / denom
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
               for (int t = 0; t < 9; ++t) { mu[t] += qOverM_rho * alpha[t]; }
            }

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

   if (P::lcclMatchedCurrent) {
      // Beyond Omega_c dt = pi the symmetric part of Phi/dt turns negative (sin(psi)/psi < 0 across B):
      // mu is no longer positive definite and the AP property is lost. Warn once.
      static bool warned = false;
      int local = gyrationBeyondPi.load() ? 1 : 0, global = 0;
      MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (global && !warned && rank == 0) {
         fprintf(stderr, "lcclBuildSpeciesTensors: WARNING: fieldsolver.matchedCurrent with |q B/m| dt >= pi for some "
                         "species; mu is no longer positive definite there. Reduce dt.\n");
      }
      if (global) { warned = true; }
   }
}

/* ============================================================
 * Section 3: Eq. (38) -- HYPRE IJ assembly + BoomerAMG-preconditioned GMRES
 * ============================================================ */

bool lccl_SolveElectricField(
   std::span<const ApVec3> Ek,
   std::span<const ApVec3> Bk,
   fsgrids::constbgbspan bgb,
   std::span<const std::array<Real,9>> mu,
   std::span<const ApVec3> Jhat,
   std::span<ApVec3> Etheta,   // out: E^{k+theta}
   std::span<ApVec3> Ekp1,     // out: E^{k+1}  (Eq. 40)
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   SysBoundary& sysBoundaries,
   Real c,
   Real theta,
   Real dt
) {
   const auto dxyz = fsgrid.getGridSpacing();
   const auto localSize = fsgrid.getLocalSize();
   const int lx = localSize[0], ly = localSize[1], lz = localSize[2];
   const long long nLocalCells = (long long)lx*ly*lz;
   const long long nLocalDofs  = 3*nLocalCells;

   // Safeguard for fieldsolver.outflowAbsorbing (see parameters.h): the
   // Ey=c*Bz, Ez=-c*By absorbing condition below is derived assuming an
   // x-normal face and is not valid as written for a y- or z-facing
   // Outflow boundary, so only ever apply it on a genuinely 1D-in-x grid.
   const auto globalSizeForAbsorbing = fsgrid.getGlobalSize();
   const bool outflowAbsorbingActive =
      P::lcclOutflowAbsorbing && globalSizeForAbsorbing[1] == 1 && globalSizeForAbsorbing[2] == 1;

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
      phiprof::initializeTimer("LCCL: assign DOF numbering"), technical,
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
      phiprof::initializeTimer("LCCL: assemble Eq.38 system"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {

         const size_t lid = stencil.ooo();

         // IDENTITY ROW: ionospheric-radius interior / fsgrid-only padding,
         // nothing physical to solve, and no SysBoundaryCondition object to
         // dispatch to either -- the legacy LDZ solver never calls
         // getSysBoundary() for these flags (ldz_calculateElectricField
         // returns before reaching its own dispatch), so this stays a
         // literal, undispatched 0. Direct-pin both E arrays here rather
         // than leaving it to Eq. 40 in the extraction loop below: that
         // formula gives (0-Ek)*(1-1/theta), which is only exactly 0 for
         // theta==1 (true for the POS cfg, but not in general) or Ek==0 --
         // pinning both arrays directly is correct for every theta.
         if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) {
            for (int comp = 0; comp < 3; ++comp) {
               Etheta[lid][comp] = 0.0;
               Ekp1[lid][comp]   = 0.0;
               const HYPRE_BigInt row = globalDof(lid, comp);
               HYPRE_Int one = 1;
               const HYPRE_Real diag = 1.0, zero = 0.0;
               HYPRE_IJMatrixSetValues(Aij, 1, &one, &row, &row, &diag);
               HYPRE_IJVectorSetValues(bij, 1, &row, &zero);
               HYPRE_IJVectorSetValues(xij, 1, &row, &zero);
            }
            return;
         }

         // IDENTITY ROW: genuine boundary cell (Maxwellian, Outflow, ...).
         // Ask the actual boundary condition what E should be here, exactly
         // as ldz_calculateElectricField does -- not a hard-coded 0, even
         // though every type this solver currently targets (Maxwellian,
         // Outflow) happens to always return 0 (see inflow.cpp, outflow.cpp)
         // -- so this stays correct if a type that doesn't is added later.
         // fieldSolverBoundaryCondElectricField mutates its span argument in
         // place at stencil.ooo(); called directly on Etheta/Ekp1 (both
         // std::span<std::array<Real,3>>, the same underlying type as
         // fsgrids::efieldspan), so both are pinned with no extra copying.
         // Direct pinning: Ekp1 is set this way too, not via Eq. 40.
         if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) {
            // fieldsolver.outflowAbsorbing (see parameters.h): for an
            // Outflow cell specifically, replace the dispatched E=0 with
            // the absorbing condition for a rightward-propagating vacuum
            // wave, Ey=c*Bz, Ez=-c*By -- derived directly from the vacuum
            // wave equation (dBz/dt=-dEy/dx, dEy/dt=-c^2 dBz/dx -> a
            // rightward travelling solution satisfies Ey=c*Bz exactly),
            // and consistent with the SAME relationship already used to
            // derive the driven x- boundary's own Bz(t)=Ey_in(t)/c. E=0
            // is correct for LAUNCHING a wave at a driving boundary
            // (paired with a time-dependent Bz(t)), but has no physical
            // justification at an undriven Outflow face: it is the
            // boundary condition for a perfectly-conducting wall, not an
            // absorbing one, and reflects whatever reaches it. Ex (the
            // face-normal component) is untouched -- this transverse-wave
            // relationship says nothing about the normal component, so it
            // keeps the existing (0) dispatch. Uses Bk (this step's B^k,
            // the only B available yet -- lccl_UpdateMagneticField, which
            // computes this step's B^{k+theta}/B^{k+1}, hasn't run yet at
            // this point in lccl_propagateFields): a one-step lag, same
            // spirit as 0008's own direct-pinning lag for Outflow's B.
            const bool useAbsorbing = outflowAbsorbingActive && (sysBoundaryFlag == sysboundarytype::OUTFLOW);
            const bool useInflow = lcclInflow && lcclInflow->active[lid];
            // bc is needed regardless of useAbsorbing: comp==0 (Ex) always
            // goes through the real dispatch below, even on an Outflow
            // cell with the absorbing condition active for comp 1 and 2.
            SBC::SysBoundaryCondition* bc = sysBoundaries.getSysBoundary(sysBoundaryFlag);
            for (int comp = 0; comp < 3; ++comp) {
               if (useAbsorbing && comp == 1) {
                  Etheta[lid][1] = c * Bk[lid][2];
                  Ekp1[lid][1]   = c * Bk[lid][2];
               } else if (useAbsorbing && comp == 2) {
                  Etheta[lid][2] = -c * Bk[lid][1];
                  Ekp1[lid][2]   = -c * Bk[lid][1];
               } else if (useInflow && comp != 0) {
                  Etheta[lid][comp] = lcclInflow->Etheta[lid][comp];
                  Ekp1[lid][comp]   = lcclInflow->Ekp1[lid][comp];
               } else {
                  bc->fieldSolverBoundaryCondElectricField(Etheta, stencil, comp);
                  bc->fieldSolverBoundaryCondElectricField(Ekp1, stencil, comp);
               }
               const HYPRE_BigInt row = globalDof(lid, comp);
               HYPRE_Int one = 1;
               const HYPRE_Real diag = 1.0;
               const HYPRE_Real bval = (HYPRE_Real)Etheta[lid][comp];
               HYPRE_IJMatrixSetValues(Aij, 1, &one, &row, &row, &diag);
               HYPRE_IJVectorSetValues(bij, 1, &row, &bval);
               HYPRE_IJVectorSetValues(xij, 1, &row, &bval);
            }
            return;
         }

         for (int comp = 0; comp < 3; ++comp) {
            const HYPRE_BigInt row = globalDof(lid, comp);

            std::map<HYPRE_BigInt, Real> rowVals;
            for (const auto& entry : lccl_EquationRow(comp, mu[lid], dxyz, reactionScale, muScale, curlScale)) {
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
      fprintf(stderr, "lcclSolveElectricField: GMRES/BoomerAMG finished, relres=%e after %d iters\n", (double)relres, (int)its);
   }

   // ---- Step 4: extract E^{k+theta}, compute E^{k+1} (Eq. 40) ----
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: extract E^(k+theta), compute E^(k+1)"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         // Boundary-flagged cells (uncomputed or genuine) were already fully
         // pinned, both arrays, during assembly above -- Eq. 40 is only
         // valid for an interior cell's own E^k, not a value sourced from a
         // boundary condition or held at 0 by convention.
         if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) { return; }
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

void lccl_UpdateMagneticField(
   std::span<const ApVec3> Bk,
   std::span<const ApVec3> Etheta,
   std::span<ApVec3> Bkp1,
   std::span<ApVec3> Btheta,
   fsgrids::constbgbspan bgb,
   fsgrids::technicalspan technical,
   FieldSolverGrid& fsgrid,
   SysBoundary& sysBoundaries,
   Real theta,
   Real dt
) {
   const auto dxyz = fsgrid.getGridSpacing();
   // Pass 1: interior (NOT_SYSBOUNDARY) cells only, via the centred curl
   // (Eq. 23/39). Boundary-flagged cells, uncomputed or genuine, are left
   // for pass 2 below.
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: Faraday update, interior"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) { return; }
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
   // Ghost-exchange before pass 2, so a boundary cell's "nearest solving
   // neighbour" lookup (Outflow) sees pass 1's freshly computed Bkp1 at
   // every interior cell, including ones owned by a neighbouring rank --
   // not the previous step's Bk. ldz_magnetic_field.cpp does the same
   // between its own L1 and L2 passes, for the same reason.
   fsgrid.updateGhostCells(Bkp1);

   // Pass 2: boundary-flagged cells. Uncomputed cells (no dispatchable
   // SysBoundaryCondition -- see the matching note in lccl_SolveElectricField)
   // are left untouched, same as pass 1 skipped them. Genuine boundary
   // cells are DIRECTLY PINNED (for now): both Bkp1 and Btheta take
   // fieldSolverBoundaryCondMagneticField's value, rather than deriving
   // Btheta from Bkp1 via the interior theta-blend formula. The lookup
   // array passed in is Bkp1 itself (perbspan demands non-const; Bk is
   // const and won't bind) -- correct for a single boundary layer thanks
   // to the ghost exchange above, but unlike the legacy solver's own
   // two-sysBoundaryLayer L1/L2 split, a layer-2 cell whose nearest
   // solving neighbour is itself another boundary cell is not specially
   // handled yet.
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: Faraday update, boundary"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         if (sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) { return; }
         if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { return; }
         const size_t lid = stencil.ooo();
         const auto globalCoordinates = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         SBC::SysBoundaryCondition* bc = sysBoundaries.getSysBoundary(sysBoundaryFlag);
         const bool useInflow = lcclInflow && lcclInflow->active[lid];
         for (int comp = 0; comp < 3; ++comp) {
            if (useInflow && comp != 0) {
               Bkp1[lid][comp]   = lcclInflow->Bkp1[lid][comp];
               Btheta[lid][comp] = lcclInflow->Btheta[lid][comp];
               continue;
            }
            const Real val = bc->fieldSolverBoundaryCondMagneticField(
               Bkp1, bgb, technical, dxyz, globalCoordinates, stencil, comp);
            Bkp1[lid][comp]   = val;
            Btheta[lid][comp] = val;  // direct pinning, not the theta-blend formula
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

// ---- Consistent Gauss correction (fieldsolver.gaussLawConsistent, the default) -------------------------
// The translation moves charge between cells with face fluxes of a high-order reconstruction, so for
// a mode exp(i k x) it changes rho with (very nearly) the exact wavenumber k. The centred divergence
// of Eq. 45 sees K = sin(k dx)/dx instead. With that mismatch the correction feeds back on the
// Langmuir oscillation and makes short waves grow (white paper Sec. 3.8; cold analysis: up to
// 0.5 omega_pe for omega_pe dt < 1 when Landau damping is absent). The consistent form:
//  - takes the residual rho/eps0 - D.E^k with the divergence D = K Q(delta^2) along each axis,
//    Q = sum_{j<n} c_j x^j, x = -delta^2/4, c_j = 4^j (j!)^2/(2j+1)!, the series of k/K in
//    sin^2(k dx/2) (error O(sin^(2n)(k dx/2)); n = fieldsolver.gaussLawOrder);
//  - solves the compact system of Eq. 45, -div[(I + dt^2 mu/eps0) grad phi] = residual;
//  - low-pass filters phi with m binomial passes per axis (transfer cos^(2m)(k dx/2);
//    m = fieldsolver.gaussLawFilterPasses): near k dx = pi no collocated operator matches the
//    translation (K -> 0), and without a correction the cold analysis is neutral there. In a uniform
//    plasma filtering phi is the same as filtering the residual; at a sharp plasma-vacuum edge it is
//    not: a residual filter spreads the edge charge from the dense side, where the weight
//    1 + dt^2 mu/eps0 makes the correction weak, into the vacuum, where it acts in full, and made the
//    high-density plasma opening switch fail within 14 steps;
//  - corrects E with the centred gradient of Eq. 45; with the filter in place this is as stable as a
//    matching gradient and needs no further passes.
// Two measures keep it away from sharp plasma edges, which the uniform analysis does not cover; in a
// uniform plasma neither changes anything:
//  - the filter treats a neighbour whose weight differs by more than fieldsolver.gaussLawFilterBarrier
//    from that of the cell as the cell itself, so it does not mix plasma and vacuum;
//  - cells whose neighbourhood of fieldsolver.gaussLawUniformityWidth cells contains plasma densities
//    differing by more than fieldsolver.gaussLawUniformityRatio, or no plasma, get neither a correction
//    source nor a correction: the series reaches 7 cells to either side and would distort sheaths
//    narrower than a cell. There Ampere's law alone acts, as without a correction.
// Communication: every sweep reads at most two cells away along an axis (the ghost width
// FS_STENCIL_WIDTH = 2) and treats all axes at once, so the extra ghost exchanges per step are
// ceil((n-1)/2) for the series, ceil(m/2) for the filter and about ceil(width/2)-1 for the mask,
// independent of the number of dimensions: 12 for the defaults n = 7, m = 12, width 8.
// Cold, linear analysis over all k and 0.005 <= omega_pe dt <= 20, in one and two dimensions:
// theta = 1 neutral or damped; theta = 1/2 at most 1e-4 omega_pe (3e-5 for omega_pe dt >= 0.02).
// fieldsolver.gaussLawConsistent = 0 restores Eq. 45 as written.
static const Real lcclSeriesCoeff[8] = {1.0, 2.0/3.0, 8.0/15.0, 16.0/35.0, 128.0/315.0, 256.0/693.0,
                                         1024.0/3003.0, 2048.0/6435.0};

static inline bool lccl_IsPhysicalCell(uint flag) {
   return flag != sysboundarytype::DO_NOT_COMPUTE && flag != sysboundarytype::OUTER_BOUNDARY_PADDING;
}

typedef fsgrid::FsData<std::array<Real,1>> ApScalarField;

// out[c] = Q(delta_c^2) E[c] for every active axis c (out[c] = E[c] for inactive axes), by Horner's
// scheme in x^2: Q = P_0 + x^2 (P_1 + x^2 (P_2 + ...)), P_i = c_2i + c_2i+1 x. Each level reads the
// previous one up to two cells away, so one ghost exchange per level; out is ghost-exchanged on return.
// Terms whose stencil would leave the physical cells are dropped (lower order next to boundaries).
static void lccl_SeriesQAllAxes(std::span<const ApVec3> E, ApNodeField& out, uint nTerms,
                                const std::array<bool,3>& active, fsgrids::technicalspan technical,
                                FieldSolverGrid& fsgrid) {
   const size_t nStorage = E.size();
   const int degree = (int)std::min<uint>(std::max<uint>(nTerms, 1u), 8u) - 1;
   auto coeff = [degree](int j) -> Real { return (j >= 0 && j <= degree) ? lcclSeriesCoeff[j] : 0.0; };
   const int levels = degree/2 + 1;                       // P_0 .. P_{levels-1}
   // the top level is a constant (c_degree) when degree is even: fold it into the next one
   const bool topIsConstant = (degree % 2 == 0) && levels > 1;
   const int firstLevel = topIsConstant ? levels - 2 : levels - 1;

   ApNodeField prev(nStorage), cur(nStorage);
   for (int level = firstLevel; level >= 0; --level) {
      const bool hasInner = level < levels - 1;           // + x^2 T_{level+1}
      const bool innerIsE = hasInner && topIsConstant && level == levels - 2;
      const Real innerScale = innerIsE ? coeff(degree) : 1.0;
      const auto pv = prev.view();
      auto cv = cur.view();
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("LCCL: consistent Gauss, series level"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const size_t lid = stencil.ooo();
            for (int c = 0; c < 3; ++c) {
               if (!active[c]) { cv[lid][c] = E[lid][c]; continue; }
               Real v = coeff(2*level)*E[lid][c];
               if (!lccl_IsPhysicalCell(sysBoundaryFlag)) { cv[lid][c] = v; continue; }
               std::array<int,3> o1{0,0,0}, o2{0,0,0};
               o1[c] = 1; o2[c] = 2;
               const bool has1 = stencil.cellExists(o1[0], o1[1], o1[2]) && stencil.cellExists(-o1[0], -o1[1], -o1[2]);
               size_t p1 = lid, m1 = lid;
               bool ok1 = false;
               if (has1) {
                  p1 = stencil.indexFromOffset(o1[0], o1[1], o1[2]);
                  m1 = stencil.indexFromOffset(-o1[0], -o1[1], -o1[2]);
                  ok1 = lccl_IsPhysicalCell(technical[p1].sysBoundaryFlag) && lccl_IsPhysicalCell(technical[m1].sysBoundaryFlag);
               }
               if (ok1) {
                  // x f = -(f(+1) - 2 f + f(-1))/4
                  v += coeff(2*level + 1)*(-0.25)*(E[p1][c] - 2.0*E[lid][c] + E[m1][c]);
               }
               if (hasInner && ok1) {
                  const bool has2 = stencil.cellExists(o2[0], o2[1], o2[2]) && stencil.cellExists(-o2[0], -o2[1], -o2[2]);
                  if (has2) {
                     const size_t p2 = stencil.indexFromOffset(o2[0], o2[1], o2[2]);
                     const size_t m2 = stencil.indexFromOffset(-o2[0], -o2[1], -o2[2]);
                     if (lccl_IsPhysicalCell(technical[p2].sysBoundaryFlag) && lccl_IsPhysicalCell(technical[m2].sysBoundaryFlag)) {
                        // x^2 f = (f(+2) - 4 f(+1) + 6 f - 4 f(-1) + f(-2))/16
                        const Real f2 = innerIsE ? E[p2][c] : pv[p2][c];
                        const Real f1 = innerIsE ? E[p1][c] : pv[p1][c];
                        const Real f0 = innerIsE ? E[lid][c] : pv[lid][c];
                        const Real g1 = innerIsE ? E[m1][c] : pv[m1][c];
                        const Real g2 = innerIsE ? E[m2][c] : pv[m2][c];
                        v += innerScale*(f2 - 4.0*f1 + 6.0*f0 - 4.0*g1 + g2)/16.0;
                     }
                  }
               }
               cv[lid][c] = v;
            }
         });
      fsgrid.updateGhostCells(cv);
      prev.swap(cur);
   }
   out.swap(prev);
}

// Uniformity mask for the consistent correction: 1 where the plasma density (trace of mu, which is
// proportional to sum_s q_s^2 n_s/m_s) varies by less than 'ratio' within 'width' cells along every
// active axis, 0 elsewhere and wherever there is no plasma. The neighbourhood max/min is built with
// sweeps two cells wide (one ghost exchange between sweeps), as the other operators here.
static void lccl_UniformityMask(std::span<const std::array<Real,9>> mu, Real ratio, uint width,
                                const std::array<bool,3>& active, fsgrids::technicalspan technical,
                                FieldSolverGrid& fsgrid, std::vector<Real>& mask) {
   const size_t nStorage = mu.size();
   ApNodeField a(nStorage), b(nStorage);   // components 0 and 1: neighbourhood max and min
   {
      auto av = a.view();
      for (size_t c = 0; c < nStorage; ++c) {
         const Real tr = mu[c][0] + mu[c][4] + mu[c][8];
         av[c][0] = tr; av[c][1] = tr; av[c][2] = 0.0;
      }
   }
   const int rx = active[0] ? 2 : 0, ry = active[1] ? 2 : 0, rz = active[2] ? 2 : 0;
   const uint sweeps = std::max(1u, (width + 1)/2);
   for (uint sw = 0; sw < sweeps; ++sw) {
      const auto av = a.view();
      auto bv = b.view();
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("LCCL: consistent Gauss, uniformity mask"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const size_t lid = stencil.ooo();
            Real mx = av[lid][0], mn = av[lid][1];
            for (int dz = -rz; dz <= rz; ++dz) {
               for (int dy = -ry; dy <= ry; ++dy) {
                  for (int dx = -rx; dx <= rx; ++dx) {
                     if (!(dx || dy || dz) || !stencil.cellExists(dx, dy, dz)) { continue; }
                     const size_t n = stencil.indexFromOffset(dx, dy, dz);
                     if (!lccl_IsPhysicalCell(technical[n].sysBoundaryFlag)) { continue; }
                     mx = std::max(mx, av[n][0]);
                     mn = std::min(mn, av[n][1]);
                  }
               }
            }
            bv[lid][0] = mx; bv[lid][1] = mn; bv[lid][2] = 0.0;
         });
      if (sw + 1 < sweeps) { fsgrid.updateGhostCells(bv); }
      a.swap(b);
   }
   mask.assign(nStorage, 0.0);
   const auto av = a.view();
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: consistent Gauss, uniformity mask"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         const Real mx = av[lid][0], mn = av[lid][1];
         mask[lid] = (mn > 0.0 && mx <= ratio*mn) ? 1.0 : 0.0;
      });
}

// One tensor-product binomial sweep over NOT_SYSBOUNDARY cells: weights (1,2,1)/4 (halfWidth 1) or
// (1,4,6,4,1)/16 (halfWidth 2) along each active axis, i.e. cos^2 or cos^4(k dx/2) per axis in a single
// sweep. A stencil point that is not an interior cell is replaced by the centre value (zero gradient,
// constants are preserved); boundary cells keep their value.
static void lccl_BinomialTensorSweep(ApScalarField& src, ApScalarField& dst, int halfWidth,
                                     const std::array<bool,3>& active, std::span<const Real> apWeight,
                                     Real barrier, fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   const bool useBarrier = barrier > 1.0 && apWeight.size() == src.size();
   static const Real w1[3] = {0.25, 0.5, 0.25};
   static const Real w2[5] = {1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0};
   const Real* w = (halfWidth == 2) ? w2 : w1;
   const int r = (halfWidth == 2) ? 2 : 1;
   const int rx = active[0] ? r : 0, ry = active[1] ? r : 0, rz = active[2] ? r : 0;
   const auto sv = src.view();
   auto dv = dst.view();
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: consistent Gauss, residual filter"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) { dv[lid][0] = sv[lid][0]; return; }
         const Real centre = sv[lid][0];
         Real acc = 0.0;
         for (int dz = -rz; dz <= rz; ++dz) {
            const Real wz = rz ? w[dz + r] : 1.0;
            for (int dy = -ry; dy <= ry; ++dy) {
               const Real wy = ry ? w[dy + r] : 1.0;
               for (int dx = -rx; dx <= rx; ++dx) {
                  const Real wx = rx ? w[dx + r] : 1.0;
                  Real v = centre;
                  if ((dx || dy || dz) && stencil.cellExists(dx, dy, dz)) {
                     const size_t n = stencil.indexFromOffset(dx, dy, dz);
                     bool use = technical[n].sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY;
                     if (use && useBarrier) {
                        const Real ratio = apWeight[n]/apWeight[lid];
                        use = ratio <= barrier && ratio*barrier >= 1.0;
                     }
                     if (use) { v = sv[n][0]; }
                  }
                  acc += wx*wy*wz*v;
               }
            }
         }
         dv[lid][0] = acc;
      });
}

// m binomial passes per active axis as ceil(m/2) tensor sweeps; f must be ghost-exchanged on entry and
// is ghost-exchanged on return.
static void lccl_FilterScalar(ApScalarField& f, uint passes, const std::array<bool,3>& active,
                              std::span<const Real> apWeight, Real barrier,
                              fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   ApScalarField tmp(f.size());
   uint done = 0;
   while (done < passes) {
      const int halfWidth = (passes - done >= 2) ? 2 : 1;
      lccl_BinomialTensorSweep(f, tmp, halfWidth, active, apWeight, barrier, technical, fsgrid);
      done += (uint)halfWidth;
      fsgrid.updateGhostCells(tmp.view());
      f.swap(tmp);
   }
}

void lccl_GaussLawCorrection(
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
      phiprof::initializeTimer("LCCL: assign phi DOF numbering"), technical,
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

   // Local-only (nlocal-sized, no ghosts) per-cell flag, 1 for NOT_SYSBOUNDARY,
   // populated during the assembly loop below. Needed so the mean-phi
   // subtraction further down -- a bare sequential loop over phiLocal, with
   // no sysBoundaryFlag of its own -- can also restrict to interior cells;
   // phiLocal is indexed by lidx (a *local-only* linear index), which is a
   // different index space from technical's (ghost-inclusive) stencil.ooo(),
   // so technical[] can't just be read directly at a lidx.
   std::vector<double> mvals(nStencil*nlocal, 0.0), bvals(nlocal, 0.0);
   double meanRhoOverEps = 0.0;
   bool useOutflowNeumann = false;
   double nInteriorGlobal = 0.0;
   bool hasAnchorGlobal = false;  // true if any boundary-flagged (Dirichlet-pinned) cell exists anywhere
   bool hasPinnedNonOutflow = false;
   {
      // Mean-subtracting rho (and, further down, phi) is NOT part of the
      // paper's Eq. (45) -- it is an addition needed only for a FULLY
      // PERIODIC domain (no boundary cells anywhere), where the Poisson
      // operator has a 1D null space (a constant shift of phi solves the
      // homogeneous problem) and the system is singular unless the RHS is
      // made to integrate to zero. A domain with at least one genuine
      // boundary-flagged cell already has a Dirichlet pin (that cell's own
      // phi=0 identity row, from the IDENTITY ROW branch below) breaking
      // that null space -- the system is already well-posed, and adding a
      // domain-wide mean-rho subtraction on top does not fix anything
      // there; it SUBTRACTS A SPURIOUS, GENERALLY LARGE CONSTANT (order
      // rho_slab/2 for a domain that's roughly half plasma, half vacuum,
      // like this test) from the RHS EVERYWHERE, including the vacuum
      // region where the true RHS is ~0 -- confirmed to be the actual
      // cause of a Plasma Opening Switch run bailing out on literally its
      // first step: the resulting phi (order rho*L_vacuum^2/eps0, an
      // order-of-magnitude match to the max|phi| actually observed) has
      // nothing to do with any real dynamics, since it only depends on the
      // (mean-subtracted) initial rho and reproduces with Ek=0 exactly.
      // So: check once whether ANY boundary-flagged cell exists anywhere
      // in the domain (an MPI_Allreduce over a local 0/1 flag -- global,
      // not per-rank, since a Dirichlet pin on one rank anchors the WHOLE
      // coupled system, not just that rank's local piece of it), and only
      // compute+apply the mean-subtraction when none do.
      double sumRho_local = 0.0, nInterior_local = 0.0;
      bool hasAnchor_local = false;
      bool hasPinnedNonOutflow_local = false;  // a phi = 0 cell that stays coupled when Outflow faces float
      // DIAGNOSTIC (debugging a POS bailout, not part of the fix above):
      // max|rho/eps0| and roughly where it occurs, over NOT_SYSBOUNDARY
      // cells. Rank-local x only (no cross-rank argmax reduction) -- exact
      // for the single-rank runs this is being debugged with; for a
      // multi-rank run, read it as "the worst cell on rank 0" specifically.
      double maxAbsRhoOverEps_local = 0.0, xAtMax_local = 0.0;
      fsgrid.serial_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("LCCL: compute mean(rho)"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) {
               hasAnchor_local = true;
               if (sysBoundaryFlag != sysboundarytype::OUTFLOW
                   && sysBoundaryFlag != sysboundarytype::DO_NOT_COMPUTE
                   && sysBoundaryFlag != sysboundarytype::OUTER_BOUNDARY_PADDING) { hasPinnedNonOutflow_local = true; }
               return;
            }
            const Real rhoOverEpsHere = rho[stencil.ooo()][0] / physicalconstants::EPS_0;
            sumRho_local += rhoOverEpsHere;
            nInterior_local += 1.0;
            if (std::fabs(rhoOverEpsHere) > std::fabs(maxAbsRhoOverEps_local)) {
               maxAbsRhoOverEps_local = rhoOverEpsHere;
               xAtMax_local = coordinates.getPhysicalCoords(stencil.i, stencil.j, stencil.k)[0];
            }
         });
      double sumRho_global = 0.0;
      MPI_Allreduce(&sumRho_local, &sumRho_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&nInterior_local, &nInteriorGlobal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&hasAnchor_local, &hasAnchorGlobal, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
      MPI_Allreduce(&hasPinnedNonOutflow_local, &hasPinnedNonOutflow, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
      meanRhoOverEps = hasAnchorGlobal ? 0.0 : (sumRho_global / nInteriorGlobal);
      // with every pinned cell behind a zero-flux face the system would be singular
      useOutflowNeumann = P::lcclGaussOutflowNeumann && hasPinnedNonOutflow;
      if (P::lcclGaussOutflowNeumann && !hasPinnedNonOutflow) {
         bailout(true, "fieldsolver.gaussLawOutflowNeumann needs a boundary other than Outflow to pin phi",
                 __FILE__, __LINE__);
      }
      int myRankForDiag; MPI_Comm_rank(MPI_COMM_WORLD, &myRankForDiag);
      fprintf(stderr, "lcclGaussLawCorrection: rank %d: max|rho/eps0|=%e at x=%e m (hasAnchor=%d, meanRhoOverEps=%e)\n",
              myRankForDiag, (double)maxAbsRhoOverEps_local, (double)xAtMax_local,
              hasAnchorGlobal ? 1 : 0, meanRhoOverEps);
   }

   // Consistent form: residual rho/eps0 - D.E^k with the translation-consistent divergence (see the
   // description above lccl_SeriesQAllAxes). Axes with a single cell carry no derivative.
   const size_t nStorageCells = fsgrid.getNumStorageCells();
   const auto globalSize = fsgrid.getGlobalSize();
   const std::array<bool,3> activeAxis = {globalSize[0] > 1, globalSize[1] > 1, globalSize[2] > 1};
   ApScalarField gaussResid(nStorageCells);
   double maskedResidualMean = 0.0;
   bool useMaskedMean = false;
   // Uniformity mask (fieldsolver.gaussLawUniformityRatio), kept for the application of the correction:
   // masked cells get neither a correction source nor a correction, so the potential of the residual in
   // the uniform regions cannot build up fields in the vacuum or the sheaths, where nothing removes them.
   std::vector<Real> gaussMask;
   // AP weight 1 + dt^2 tr(mu)/(3 eps0) per cell (mu is ghost-exchanged), for the filter barrier
   std::vector<Real> apWeight;
   if (P::lcclGaussConsistent && P::lcclGaussFilterBarrier > 1.0) {
      apWeight.resize(nStorageCells);
      for (size_t c = 0; c < nStorageCells; ++c) {
         apWeight[c] = 1.0 + dt*dt*(mu[c][0] + mu[c][4] + mu[c][8])/(3.0*physicalconstants::EPS_0);
      }
   }
   if (P::lcclGaussConsistent) {
      if (P::lcclGaussOrder < 1 || P::lcclGaussOrder > 8) {
         bailout(true, "fieldsolver.gaussLawOrder must be between 1 and 8", __FILE__, __LINE__);
      }
      ApNodeField QE(nStorageCells);
      lccl_SeriesQAllAxes(Ek, QE, P::lcclGaussOrder, activeAxis, technical, fsgrid);
      const auto qv = QE.view();
      auto rv = gaussResid.view();
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("LCCL: consistent Gauss, residual"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const size_t lid = stencil.ooo();
            if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) { rv[lid][0] = 0.0; return; }
            Real divQE = 0.0;
            for (int c = 0; c < 3; ++c) {
               if (!activeAxis[c]) { continue; }
               std::array<int,3> o{0,0,0}; o[c] = 1;
               if (!stencil.cellExists(o[0], o[1], o[2]) || !stencil.cellExists(-o[0], -o[1], -o[2])) { continue; }
               divQE += (qv[stencil.indexFromOffset(o[0], o[1], o[2])][c]
                         - qv[stencil.indexFromOffset(-o[0], -o[1], -o[2])][c])/(2.0*dxyz[c]);
            }
            rv[lid][0] = rho[lid][0]/physicalconstants::EPS_0 - divQE;
         });
      if (P::lcclGaussUniformityRatio > 1.0) {
         std::vector<Real>& mask = gaussMask;
         lccl_UniformityMask(mu, P::lcclGaussUniformityRatio, P::lcclGaussUniformityWidth, activeAxis, technical, fsgrid, mask);
         double sum_local = 0.0, sum_global = 0.0, masked_local = 0.0, masked_global = 0.0;
         fsgrid.serial_for(
            [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
            phiprof::initializeTimer("LCCL: consistent Gauss, apply mask"), technical,
            [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
                cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
               if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) { return; }
               const size_t lid = stencil.ooo();
               rv[lid][0] *= mask[lid];
               sum_local += rv[lid][0];
               masked_local += 1.0 - mask[lid];
            });
         MPI_Allreduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(&masked_local, &masked_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         // a periodic domain needs a right-hand side that integrates to zero; the masked residual
         // no longer has the mean of rho/eps0, so its own mean replaces it below
         maskedResidualMean = (nInteriorGlobal > 0.0) ? sum_global/nInteriorGlobal : 0.0;
         useMaskedMean = true;
         static int reported = 0;
         if (myRank == MASTER_RANK && reported < 5) {
            fprintf(stderr, "lcclGaussLawCorrection: uniformity mask removes the correction source in %.0f of %.0f cells\n",
                    masked_global, nInteriorGlobal);
            ++reported;
         }
      }
      static bool announced = false;
      if (!announced && myRank == MASTER_RANK) {
         fprintf(stderr, "lcclGaussLawCorrection: consistent form, %u series terms, %u filter passes per axis on phi, "
                 "barrier %g, uniformity mask %g over %u cells\n",
                 (unsigned)P::lcclGaussOrder, (unsigned)P::lcclGaussFilterPasses, (double)P::lcclGaussFilterBarrier,
                 (double)P::lcclGaussUniformityRatio, (unsigned)P::lcclGaussUniformityWidth);
      }
      announced = true;
   }
   const auto gaussResidView = gaussResid.view();

   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: assemble Gauss-correction system"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const size_t lid = stencil.ooo();
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);

         // IDENTITY ROW: phi = 0 at any cell this correction doesn't apply
         // to -- DO_NOT_COMPUTE/OUTER_BOUNDARY_PADDING (nothing physical
         // here at all) and genuine boundary cells alike (Maxwellian,
         // Outflow, ...: their E was already finalised by
         // lccl_SolveElectricField's own identity-row dispatch, so there's no
         // meaningful correction to solve for there either -- there's no
         // fieldSolverBoundaryCondPhi-style API to dispatch through even if
         // there were, since this correction is unique to this solver).
         // mvals/bvals already default-initialise to 0, so only the
         // diagonal needs setting.
         if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) {
            mvals[nStencil*lidx + 0] = 1.0;
            return;
         }

         // EXPLICIT REGULARISATION for a fully periodic domain (!hasAnchorGlobal,
         // checked above): no genuine boundary cell exists anywhere to supply the identity
         // row above, so without this the matrix is exactly singular (a uniform shift of phi
         // solves the homogeneous problem -- this operator is a divergence of (mu-dependent)
         // gradient, and grad(constant)=0 regardless of the coefficient multiplying it). The
         // existing mean-subtraction of rho (this function, above) and of phi (previously
         // applied after the solve) keep the compatibility condition satisfied and pick the
         // zero-mean representative of the solution family -- correct in exact arithmetic, but
         // relies on PCG's Krylov iterates never acquiring a null-space component, which in
         // turn relies on this matrix being exactly symmetric. It isn't: the cross terms a few
         // lines below (kxy/kxz/kyz, from mu's off-diagonal components) are built from mu at
         // THIS cell only, not face-averaged with the neighbour the way the diagonal cxx/cyy/
         // czz terms explicitly are just above -- so where mu varies between neighbouring
         // cells (it does, sharply, near either Harris sheet), the coefficient from cell A to
         // cell B generally differs from B back to A.
         if (!hasAnchorGlobal && dofOffset + lidx == 0) {
            mvals[nStencil*lidx + 0] = 1.0; // bvals[lidx] already 0.0 by default-init, same as above
            return;
         }

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

         Real cxx_faceM = 0.5*(cxx_here+cxx_xm), cxx_faceP = 0.5*(cxx_here+cxx_xp);
         Real cyy_faceM = 0.5*(cyy_here+cyy_ym), cyy_faceP = 0.5*(cyy_here+cyy_yp);
         Real czz_faceM = 0.5*(czz_here+czz_zm), czz_faceP = 0.5*(czz_here+czz_zp);
         if (useOutflowNeumann) {
            // no correction flux into an open end: the potential there floats instead of being
            // pinned to the other wall's, which would return any sheath potential step through
            // the vacuum gaps as uniform fields
            const auto outflow = [&](bool exists, size_t n) {
               return exists && technical[n].sysBoundaryFlag == sysboundarytype::OUTFLOW;
            };
            if (outflow(hasXm, lidXm)) { cxx_faceM = 0.0; }
            if (outflow(hasXp, lidXp)) { cxx_faceP = 0.0; }
            if (outflow(hasYm, lidYm)) { cyy_faceM = 0.0; }
            if (outflow(hasYp, lidYp)) { cyy_faceP = 0.0; }
            if (outflow(hasZm, lidZm)) { czz_faceM = 0.0; }
            if (outflow(hasZp, lidZp)) { czz_faceP = 0.0; }
         }

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
         if (P::lcclGaussConsistent) {
            // consistent residual; its mean is that of rho/eps0 (D.E sums to zero on a periodic grid), or that
            // of the masked residual when the uniformity mask is on
            bvals[lidx] = gaussResidView[lid][0] - (hasAnchorGlobal ? 0.0 : (useMaskedMean ? maskedResidualMean : meanRhoOverEps));
         }
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
      phiprof::initializeTimer("LCCL: hand Gauss-correction system to HYPRE IJ"), technical,
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
         fprintf(stderr, "lcclGaussLawCorrection: Hypre PCG/BoomerAMG finished with relative residual %e "
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
         fprintf(stderr, "lcclGaussLawCorrection: solved phi, max|phi|=%e%s\n", maxAbsPhiGlobal,
                 phiNonFiniteGlobal ? "  *** NaN/Inf IN SOLVED PHI ***" : "");
      }
   }

   HYPRE_ParCSRPCGDestroy(pcgSolver);
   HYPRE_BoomerAMGDestroy(amgPrecond);
   HYPRE_IJMatrixDestroy(Aij);
   HYPRE_IJVectorDestroy(bij);
   HYPRE_IJVectorDestroy(xij);

   // Eq. 41: E~^{k+1} = E^{k+1} - grad(phi). E^{k+theta} is not directly
   // corrected by Eq. 41 -- only E^{k+1} is -- so to stay consistent with
   // Eq. 40 (E^{k+theta} = theta*E^{k+1} + (1-theta)*E^k, with E^k left as
   // is here) the correction applied to edt2 must be theta*(-grad(phi)),
   // not the full -grad(phi) applied to e.
   fsgrid::FsData<std::array<Real,1>> phiGrid(fsgrid.getNumStorageCells());
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: stage phi"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         const long long lidx = stencil.i + lx*((long long)stencil.j + ly*stencil.k);
         phiGrid[stencil.ooo()][0] = phiLocal[lidx];
      });
   fsgrid.updateGhostCells(phiGrid.view());
   if (useOutflowNeumann) {
      // zero normal gradient: an Outflow cell takes phi of the nearest interior cell, so the
      // centred gradient at the last interior node sees no step into the boundary
      fsgrid.serial_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("LCCL: mirror phi into Outflow cells"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            if (sysBoundaryFlag != sysboundarytype::OUTFLOW) { return; }
            for (int dist = 1; dist <= FS_STENCIL_WIDTH; ++dist) {
               for (int axis = 0; axis < 3; ++axis) {
                  for (int sgn = -1; sgn <= 1; sgn += 2) {
                     std::array<int,3> o{0,0,0};
                     o[axis] = sgn*dist;
                     if (!stencil.cellExists(o[0], o[1], o[2])) { continue; }
                     const size_t n = stencil.indexFromOffset(o[0], o[1], o[2]);
                     if (technical[n].sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
                        phiGrid[stencil.ooo()][0] = phiGrid[n][0];
                        return;
                     }
                  }
               }
            }
         });
      fsgrid.updateGhostCells(phiGrid.view());
   }

   // Consistent form: low-pass filter phi (after the solve, so the charge of a sharp edge is weighted
   // where it sits before anything is smoothed). Ghost-exchanged for the centred gradient below.
   if (P::lcclGaussConsistent && pcgRelres <= 1.0) {
      lccl_FilterScalar(phiGrid, P::lcclGaussFilterPasses, activeAxis, apWeight, P::lcclGaussFilterBarrier,
                        technical, fsgrid);
   }

   // Eq. 41, only applied if actually converged
   if (pcgRelres <= 1.0) {
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("LCCL: apply Gauss correction to E"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            // Not just DO_NOT_COMPUTE/OUTER_BOUNDARY_PADDING: a genuine boundary
            // cell's Ekp1 was already finalised by lccl_SolveElectricField's own
            // identity-row dispatch (fieldSolverBoundaryCondElectricField),
            // before this function was even called. Subtracting a
            // phi-gradient here would silently overwrite that pinned value.
            if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) { return; }
            const size_t lid = stencil.ooo();
            if (!gaussMask.empty() && gaussMask[lid] < 0.5) { return; }   // masked: left to Ampere's law
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
      fprintf(stderr, "lcclGaussLawCorrection: relres=%e > 1.0 -- solve diverged, "
                      "SKIPPING correction this step, E left unmodified\n", (double)pcgRelres);
   }
}

/* ============================================================
 * Section 6: diagnostics and the optional low-pass filter
 * ============================================================ */

// Reports max|E| (over all 3 components, all local cells)
static void lccl_ReportFieldMagnitude(const char* label, std::span<const ApVec3> field,
                                     fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   int myRank; MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   double maxAbsLocal[3] = {0.0, 0.0, 0.0};
   bool nonFiniteLocal = false;
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: report field magnitude"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { return; }
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

// Low-pass filter along one axis: Shapiro (1970) filter of order 2n, transfer
// 1 - sin^(2n)(k*dx/2). n = 1 is the original 3-point binomial (cos^2(k*dx/2)).
// Every order removes the cell-to-cell mode that the centred stencils of this
// collocated scheme cannot see; higher orders leave longer features almost alone
// (order 8: -6% per application at 4 cells, -0.05% at 8 cells). Boundary cells keep
// their values but are read as neighbours. Not part of the paper; off unless
// fieldsolver.lowPassFilter is set.
static void lccl_ApplyLowPassFilter1D(std::span<ApVec3> field, int axis, int n, fsgrids::technicalspan technical,
                                    FieldSolverGrid& fsgrid) {
   std::array<int,3> plusOffset{0,0,0};  plusOffset[axis]  =  1;
   std::array<int,3> minusOffset{0,0,0}; minusOffset[axis] = -1;

   // (delta^2)^n of the field, one ghost exchange per pass so the result is rank-layout independent
   ApNodeField d(field.size()), dd(field.size());
   std::copy(field.begin(), field.end(), d.view().begin());
   for (int pass = 0; pass < n; ++pass) {
      const auto src = d.view();
      auto dst = dd.view();
      fsgrid.serial_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("LCCL: low-pass filter, second difference"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            const size_t lid = stencil.ooo();
            if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE
                || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING
                || !stencil.cellExists(minusOffset[0], minusOffset[1], minusOffset[2])
                || !stencil.cellExists(plusOffset[0], plusOffset[1], plusOffset[2])) {
               dst[lid] = {0.0, 0.0, 0.0};  // no full stencil (domain edge): no correction from here
               return;
            }
            const size_t m = stencil.indexFromOffset(minusOffset[0], minusOffset[1], minusOffset[2]);
            const size_t p = stencil.indexFromOffset(plusOffset[0], plusOffset[1], plusOffset[2]);
            for (int comp = 0; comp < 3; ++comp) {
               dst[lid][comp] = src[p][comp] - 2.0*src[lid][comp] + src[m][comp];
            }
         });
      fsgrid.updateGhostCells(dst);
      d.swap(dd);
   }

   const Real coeff = std::pow(-0.25, n);  // u_filtered = u - (-1/4)^n (delta^2)^n u
   const auto dn = d.view();
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("LCCL: low-pass filter, apply"), technical,
      [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
          cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
         // boundary values are set by their own conditions
         if (sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY) { return; }
         const size_t lid = stencil.ooo();
         for (int comp = 0; comp < 3; ++comp) { field[lid][comp] -= coeff*dn[lid][comp]; }
      });
   fsgrid.updateGhostCells(field);
}

// Separable filter in all three dimensions, of order fieldsolver.lowPassFilterOrder
static void lccl_ApplyLowPassFilter3D(std::span<ApVec3> field, fsgrids::technicalspan technical, FieldSolverGrid& fsgrid) {
   int n = (int)(P::lcclLowPassFilterOrder / 2);
   if (P::lcclLowPassFilterOrder % 2 != 0 || n < 1) {
      bailout(true, "fieldsolver.lowPassFilterOrder must be an even number >= 2", __FILE__, __LINE__);
      n = 1;  // bailout() only flags; finish this step with the original filter
   }
   lccl_ApplyLowPassFilter1D(field, 0, n, technical, fsgrid);
   lccl_ApplyLowPassFilter1D(field, 1, n, technical, fsgrid);
   lccl_ApplyLowPassFilter1D(field, 2, n, technical, fsgrid);
}

/* ============================================================
 * Section 7: propagateFields entry point
 * ============================================================ */

bool lccl_propagateFields(fsgrids::perbspan perb,
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
                     SysBoundary& sysBoundaries,
                     creal& dt, cuint subcycles) {

   if (subcycles != 1) {
      stringstream s;
      s << "propagateFields (LCCL/CSL-RME): subcycles=" << subcycles << " requested, but the "
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

   ApNodeState& state = lccl_GetNodeState(perb, e, technical, fsgrid, dt==0.0);

   // dt==0 special case: vlasiator.cpp calls this once before the main loop
   // specifically for one-time setup. lccl_SolveElectricField divides by dt and
   // dt^2, so nothing is advanced. The initial node-centred fields are
   // published to vol (the Vlasov solver reads them from there); the Yee arrays
   // are left exactly as the project set them.
   // FIXME: This is also how the timestep limit of the fieldsolver gets to the
   // rest of the code, but what ARE the timestep limits?
   if (dt == 0.0) {
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("LCCL: initial half-step Yee arrays"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { return; }
            const size_t lid = stencil.ooo();
            perbdt2[lid] = perb[lid];
            edt2[lid]    = e[lid];
         });
      fsgrid.updateGhostCells(perbdt2);
      fsgrid.updateGhostCells(edt2);
      lccl_PublishToVol(state.B.view(), state.E.view(), vol, technical, fsgrid);
      return true;
   }

   const size_t nStorage = fsgrid.getNumStorageCells();
   fsgrid::FsData<std::array<Real,9>> mu(nStorage);
   ApNodeField Jhat(nStorage);
   lccl_BuildSpeciesTensors(speciesRhoQ, speciesJ, state.B.view(), bgb, technical, fsgrid, theta, dt, mu, Jhat);

   if (P::lcclInflowCharacteristic) {
      lccl_PrepareInflowCharacteristic(state.E.view(), state.B.view(), perb, bgb, technical, fsgrid,
                                     sysBoundaries, c, theta, dt);
   } else {
      lcclInflow.reset();
   }

   // Step (2b): E^{k+theta} from Eq. (38), E^{k+1} from Eq. (40)
   ApNodeField Etheta(nStorage), Ekp1(nStorage), Btheta(nStorage), Bkp1(nStorage);
   const bool converged = lccl_SolveElectricField(state.E.view(), state.B.view(), bgb, mu.view(), Jhat.view(),
                                                Etheta.view(), Ekp1.view(), technical, fsgrid, sysBoundaries,
                                                c, theta, dt);
   lccl_ReportFieldMagnitude("lccl_propagateFields: after raw solve (E^{k+theta})", Etheta.view(), technical, fsgrid);

   if (P::lcclLowPassFilter) {
      lccl_ApplyLowPassFilter3D(Etheta.view(), technical, fsgrid);
      lccl_ApplyLowPassFilter3D(Ekp1.view(), technical, fsgrid);
   }

   // Step (2c): B^{k+1} from Eq. (23), B^{k+theta} from Eq. (39)
   lccl_UpdateMagneticField(state.B.view(), Etheta.view(), Bkp1.view(), Btheta.view(), bgb, technical, fsgrid,
                          sysBoundaries, theta, dt);

   // Step (2d): Gauss correction of E^{k+1} only. rho^k is the total charge density
   // at the start of the step, the same speciesRhoQ that defines mu (Eq. 36).
   if (P::lcclEnforceGaussLaw) {
      fsgrid::FsData<std::array<Real,1>> rho(nStorage);
      fsgrid.parallel_for(
         [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
         phiprof::initializeTimer("LCCL: total rho^k"), technical,
         [&](const fsgrid::Coordinates& coordinates, const fsgrid::FsStencil& stencil,
             cuint sysBoundaryFlag, cuint sysBoundaryLayer) {
            if (sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE || sysBoundaryFlag == sysboundarytype::OUTER_BOUNDARY_PADDING) { return; }
            const size_t lid = stencil.ooo();
            Real sum = 0.0;
            for (size_t popID = 0; popID < speciesRhoQ.size(); ++popID) {
               sum += speciesRhoQ[popID][lid][fsgrids::speciesrhoq::SRHOQ];
            }
            rho[lid][0] = sum;
         });
      lccl_GaussLawCorrection(Ekp1.view(), state.E.view(), rho.view(), mu.view(), technical, fsgrid, dt);
   }

   if (P::lcclLowPassFilter) {
      lccl_ApplyLowPassFilter3D(Bkp1.view(), technical, fsgrid);
      lccl_ApplyLowPassFilter3D(Btheta.view(), technical, fsgrid);
   }

   // Hand-off to the Vlasov solver (Eq. 20): E^{k+theta}, B^{k+theta} straight into vol.
   lccl_PublishToVol(Btheta.view(), Etheta.view(), vol, technical, fsgrid);

   // Yee-lattice quantities for output and for everything else that expects them.
   lccl_NodesToYee(Btheta.view(), Etheta.view(), perbdt2, edt2, technical, fsgrid); // time level k+theta
   lccl_NodesToYee(Bkp1.view(),   Ekp1.view(),   perb,    e,    technical, fsgrid); // time level k+1

   // The corrected E^{k+1} and B^{k+1} become E^k and B^k of the next step.
   state.E.swap(Ekp1);
   state.B.swap(Bkp1);

   return converged;
}
