#include "fs_common.h"
#include "es_electric_field.hpp"
#include <fftw3.h>
#include "array.hpp"

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

   int myRank;
   MPI_Comm_rank(MPI_COMM_WORLD,&myRank);

   const std::array<fsgrid::FsSize_t, 3>& globalSize = fsgrid.getGlobalSize();

   Array3D rho(globalSize[0],globalSize[1],globalSize[2], 0.0);
   Array3D phi(globalSize[0],globalSize[1],globalSize[2], 0.0);

   // Get the charge from all species at cell centers and combine to get ned charge density
   fsgrid.parallel_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("Collect net charge"),
      technical,

      [&rho,moments]
      (const fsgrid::Coordinates &coordinates,
       const fsgrid::FsStencil& stencil,
       cuint sysBoundaryFlag,
       cuint sysBoundaryLayer
      ) {
         const std::array<fsgrid::FsSize_t, 3> globalIndices = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);
         rho(globalIndices[0], globalIndices[1], globalIndices[2]) = moments[stencil.ooo()][fsgrids::moments::RHOQ];
      });

   // collect data from all ranks to MASTER_RANK using MPI_Reduce
   // MPI_IN_PLACE is only valid as the ROOT's sendbuf. Non-root ranks must pass their real sendbuf.
   if (myRank == MASTER_RANK) {
      MPI_Reduce(MPI_IN_PLACE, &rho.data[0], globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
   } else {
      MPI_Reduce(&rho.data[0], nullptr, globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);
   }

   // Serial for now. The field solver is sufficently cheaper than the Vlasov
   // solver this is coupled to.
   if(myRank == MASTER_RANK) {
      // We'll convert rho to complex numbers and perform a complex-to-complex
      // 3d DFT forward, operate in k space and then perform the inverse
      // transform. This wastes a factor 2 of memory (and some compute power)
      // since rho is real (and so will the resulting Phi be), but this is a lot
      // less confusing than the packed real-to-complex transforms that compute
      // only the non-redundant half of the Fourier space along one dimension.
      ComplexArray3D complex_buffer(globalSize[0],globalSize[1],globalSize[2]);

      // Create plans for the 3d FFT. Keeping the plans around isn't actually faster, likely because FFTW caches them internally.
      auto base_addr  = &complex_buffer.data[0];
      fftw_plan fwd_plan = fftw_plan_dft_3d(globalSize[0],globalSize[1],globalSize[2], base_addr, base_addr, FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_plan rwd_plan = fftw_plan_dft_3d(globalSize[0],globalSize[1],globalSize[2], base_addr, base_addr, FFTW_BACKWARD, FFTW_ESTIMATE);

      // Convert rho to an array of fftw_complex
      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               complex_buffer(i,j,k)[0] = rho(i,j,k);
               complex_buffer(i,j,k)[1] = 0.0;
            }
         }
      }

      // Execute the FTT
      fftw_execute(fwd_plan);

      // Perform point by point operation in k-space
      const auto dxyz = fsgrid.getGridSpacing();
      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         double kx = i / double(globalSize[0]);
                kx = kx <= 0.5 ? kx : kx-1.;
                kx *= 2. * M_PI / dxyz[0];
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            double ky = j / double(globalSize[1]);
                   ky = ky <= 0.5 ? ky : ky-1.;
                   ky *= 2. * M_PI / dxyz[1];
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               double kz = k / double(globalSize[2]);
                      kz = kz <= 0.5 ? kz : kz-1.;
                      kz *= 2. * M_PI / dxyz[2];

               double ksquared = kx*kx + ky*ky + kz*kz;

               if(ksquared > 0.) {
                  complex_buffer(i,j,k)[0] = complex_buffer(i,j,k)[0] / (physicalconstants::EPS_0 * ksquared);
                  complex_buffer(i,j,k)[1] = complex_buffer(i,j,k)[1] / (physicalconstants::EPS_0 * ksquared);
               } else {
                  // Charge neurality is enforced by the setup, so we are free to set the zero mode of the potential to zero
                  complex_buffer(i,j,k)[0] = 0.;
                  complex_buffer(i,j,k)[1] = 0.;
               }
            }
         }
      }

      // Perform inverse FFT on rho_complex
      fftw_execute(rwd_plan);

      // Copy out the real part of complex_buffer, which holds phi at that point
      double max_real_mag = 0.;
      double max_imag_mag = 0.;
      for (unsigned int i = 0; i < globalSize[0]; ++i) {
         for (unsigned int j = 0; j < globalSize[1]; ++j) {
            for (unsigned int k = 0; k < globalSize[2]; ++k) {
               max_real_mag = std::max(max_real_mag, std::abs(complex_buffer(i,j,k)[0]));
               max_imag_mag = std::max(max_imag_mag, std::abs(complex_buffer(i,j,k)[1]));
               phi(i,j,k) = complex_buffer(i,j,k)[0] / double(globalSize[0]*globalSize[1]*globalSize[2]); // Division comes from unnormalized nature of the FFTW transformations
            }
         }
      }

#ifdef DEBUG_FSOLVER
      // In debug mode we want to unconditionally print this message
#else
      // In production mode we only want to print it if things are going sideways
      if (max_imag_mag > 1e-10 * max_real_mag)
#endif
      {
         fprintf(stderr, "reverse FFT ended up with a phi of real magnitude %e and an imaginary amplitude %e\n", max_real_mag, max_imag_mag);
      }

      fftw_destroy_plan(fwd_plan);
      fftw_destroy_plan(rwd_plan);
   }

   // Use MPI_Broadcast from MASTER_RANK to update everyone on the value of phi
   MPI_Bcast(&phi.data[0], globalSize[0]*globalSize[1]*globalSize[2], MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);

   // Distribute local values of Phi
   fsgrid.serial_for(
      [](int timerId) -> phiprof::Timer { return phiprof::Timer{timerId}; },
      phiprof::initializeTimer("set local values of Phi"),
      technical,

      [&Phi,&phi,myRank]
      (const fsgrid::Coordinates &coordinates,
       const fsgrid::FsStencil& stencil,
       cuint sysBoundaryFlag,
       cuint sysBoundaryLayer
      ) {
         const std::array<fsgrid::FsSize_t, 3> globalIndices = coordinates.localToGlobal(stencil.i, stencil.j, stencil.k);

         const auto lid = stencil.ooo();
         Phi[lid][fsgrids::PHI] = phi(globalIndices[0], globalIndices[1], globalIndices[2]);
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
