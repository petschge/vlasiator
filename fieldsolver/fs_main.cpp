#include "ldz_main.h"

#ifdef FS_ES
# include "es_main.h"
#endif
#ifdef FS_AP
# include "ap_electric_field.hpp"
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
#ifdef FS_AP
                    std::vector<fsgrids::speciesrhoqspan>& speciesRhoQ,
                    std::vector<fsgrids::speciesjspan>& speciesJ,
#endif
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
      return ldz_propagateFields(perb, perbdt2, e, edt2, ehall, egradpe, egradpedt2, moments, momentsdt2, dperb, dmoments, dmomentsdt2, bgb, vol, technical, fsgrid, sysBoundaries, dt, subcycles);
   } else if (P::fieldSolverMethod == "ES") {
      # ifdef FS_ES
         return es_propagateFields(e_es, Phi, moments, momentsdt2, technical, fsgrid, sysBoundaries, dt, subcycles);
      # else
         fprintf(stderr, "Electrostatic field solver selected by P::fieldSolverMethod but FS_ES was not defined at compile time\n");
         abort();
      # endif
   } else if (P::fieldSolverMethod == "AP") {
      # ifdef FS_AP
         return ap_propagateFields(perb, perbdt2, e, edt2, moments, speciesRhoQ, speciesJ, dperb, dmoments, bgb, vol, technical, fsgrid, dt, subcycles);
      # else
         fprintf(stderr, "Implicit electromagnetic field solver selected by P::fieldSolverMethod but FS_AP was not defined at compile time\n");
         abort();
      # endif
   } else {
      fprintf(stderr, "No viable field solver selected by P::fieldSolverMethod\n");
      abort();
   }
}
