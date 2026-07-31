#ifndef FS_LDZ_H
#define FS_LDZ_H

#include "fs_common.h"

bool ldz_propagateFields(fsgrids::perbspan perb,
                         fsgrids::perbspan perbdt2,
                         fsgrids::efieldspan e,
                         fsgrids::efieldspan edt2,
                         fsgrids::ehallspan ehall,
                         fsgrids::egradpespan egradpe,
                         fsgrids::egradpespan egradpedt2,
                         fsgrids::momentsspan moments,
                         fsgrids::momentsspan momentsdt2,
                         fsgrids::dperbspan dperb,
                         fsgrids::dmomentsspan dmoments,
                         fsgrids::dmomentsspan dmomentsdt2,
                         fsgrids::bgbspan bgb,
                         fsgrids::volspan vol,
                         fsgrids::technicalspan technical, FieldSolverGrid &fsgrid, SysBoundary& sysBoundaries,
                         creal& dt, cuint subcycles);

#endif
