#pragma once
/*
 * This file is part of Vlasiator.
 *
 * Restart support of the LCCL (CSL-RME) field solver, kept free of the HYPRE headers so that the I/O
 * code can include it.
 *
 * The solver carries the node-centred (= Vlasov cell-centre) E^k and perturbed B^k from one step to
 * the next. The Yee arrays fg_E and fg_PERB of a restart file are only averages of them onto edges
 * and faces; rebuilding the node values from those averages smooths them (a cell-to-cell pattern is
 * lost entirely) and breaks the discrete div B = 0 of the centred curl. Restart files therefore also
 * store the node values as fg_LCCL_E and fg_LCCL_PERB (iowrite.cpp), and readGrid (ioread.cpp) hands
 * them back with lccl_RestoreNodeState before the first step, so that a restarted run continues
 * exactly as an uninterrupted one. Restart files without these variables (written before they were
 * added, or by another field solver) fall back to rebuilding the state from fg_PERB and fg_E.
 */
#include <array>
#include <span>
#include "../definitions.h"

/*! True once the solver holds a node-centred state (after its setup call or a restore). */
bool lccl_HasNodeState();

/*! Node-centred E^k and perturbed B^k, indexed like any fsgrid array (storage index incl. ghosts);
 *  empty spans if there is no state. */
std::span<const std::array<Real,3>> lccl_NodeStateE();
std::span<const std::array<Real,3>> lccl_NodeStateB();

/*! Install the node-centred state read from a restart file (local cells filled, ghosts are exchanged
 *  here). The solver then uses it on its first step instead of rebuilding it from the Yee arrays. */
void lccl_RestoreNodeState(std::span<const std::array<Real,3>> B, std::span<const std::array<Real,3>> E,
                           FieldSolverGrid& fsgrid);
