#include "ldz_main.h"

#ifdef FS_ES
# include "es_main.h"
#endif

/*! \brief Top-level field propagation function.
 *
 * Check that the selected field solver has been compiled in an call it passend
 * on the run time selection
 *
 * \sa ldz_propagateFields
 *
 */
bool propagateFields(fsgrids::perbspan perb,
                    fsgrids::perbspan perbdt2,
                    fsgrids::efieldspan e,
                    fsgrids::efieldspan edt2,
                    fsgrids::ehallspan ehall,
                    fsgrids::egradpespan egradpe,
                    fsgrids::egradpespan egradpedt2,
#ifdef FS_ES
                    fsgrids::efieldspan e_es,
                    fsgrids::potentialspan Phi,
#endif
                    fsgrids::momentsspan moments,
                    fsgrids::momentsspan momentsdt2,
                    fsgrids::dperbspan dperb,
                    fsgrids::dmomentsspan dmoments,
                    fsgrids::dmomentsspan dmomentsdt2,
                    fsgrids::bgbspan bgb,
                    fsgrids::volspan vol,
                    fsgrids::technicalspan technical,
                    FieldSolverGrid &fsgrid,
                    SysBoundary& sysBoundaries,
                    creal& dt,
                    cuint subcycles) {

   if (P::fieldSolverMethod == "LDZ" || P::fieldSolverMethod == "default_fieldsolver") {
      // if (P::fieldSolverMethod == "LDZ") {
      //    fprintf(stderr, "LDZ field solver selected by P::fieldSolverMethod\n");
      // } else if (P::fieldSolverMethod == "default_fieldsolver") {
      //    fprintf(stderr, "LDZ field solver selected by default\n");
      // }
      return ldz_propagateFields(perb, perbdt2, e, edt2, ehall, egradpe, egradpedt2, moments, momentsdt2, dperb, dmoments, dmomentsdt2, bgb, vol, technical, fsgrid, sysBoundaries, dt, subcycles);

   } else if (P::fieldSolverMethod == "ES") {
      # ifdef FS_ES
         // fprintf(stderr, "Electrostatic field solver selected by P::fieldSolverMethod\n");
         return es_propagateFields(e_es, Phi, moments, momentsdt2, technical, fsgrid, sysBoundaries, dt, subcycles);
      # else
         fprintf(stderr, "Electrostatic field solver selected by P::fieldSolverMethod but FS_ES was not defined at compile time\n");
         abort();
      # endif
   } else {
      fprintf(stderr, "No viable field solver selected by P::fieldSolverMethod\n");
      abort();
   }
}
