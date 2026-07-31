#ifndef FS_ES_H
#define FS_ES_H

#include "fs_common.h"

bool es_propagateFields(fsgrids::efieldspan e_es,
#ifdef FS_ES
                        fsgrids::potentialspan Phi,
#endif
                        fsgrids::momentsspan moments,
                        fsgrids::momentsspan momentsdt2,
                        fsgrids::technicalspan technical, FieldSolverGrid &fsgrid, SysBoundary& sysBoundaries,
                        creal& dt, cuint subcycles);

#endif
