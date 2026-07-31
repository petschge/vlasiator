#ifndef FS_CONSTANTS_H
#define FS_CONSTANTS_H

// Constants: not needed as such, but if field solver is implemented on GPUs
// these force CPU to use float accuracy, which in turn helps to compare
// CPU and GPU results.

const Real HALF    = 0.5;
const Real MINUS   = -1.0;
const Real PLUS    = +1.0;
const Real THIRD   = 1.0/3.0;
const Real FOURTH  = 1.0/4.0;
const Real SIXTH   = 1.0/6.0;
const Real EIGTH   = 1.0/8.0;
const Real TENTH   = 1.0/10.0;
const Real TWELWTH = 1.0/12.0;
const Real TWO     = 2.0;
const Real ZERO    = 0.0;

static creal EPS = 1.0e-30;

#endif
