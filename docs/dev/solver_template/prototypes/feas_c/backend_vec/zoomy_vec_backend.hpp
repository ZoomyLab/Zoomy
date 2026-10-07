// "dmplex-like" plain-array backend for the feasibility test: every typed
// handle kind is a DISTINCT std::vector wrapper, so a kind mix-up in the
// emitted orchestration is a C++ type error, not a silent aliasing bug.
#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "model_layout.hpp"   // EMITTED from the NSM

template <class Tag, class T>
struct Handle : std::vector<T> {
    using std::vector<T>::vector;
};
struct cell_state_tag {}; struct cell_aux_tag {}; struct params_tag {};
struct face_state_tag {}; struct face_scalar_tag {}; struct cell_scalar_tag {};
struct cell_mask_tag {}; struct cell_grad_tag {}; struct bc_values_tag {};
using CellState  = Handle<cell_state_tag, double>;   // N_VARS x n_cells (SoA)
using CellAux    = Handle<cell_aux_tag, double>;     // N_AUX  x n_cells
using Params     = Handle<params_tag, double>;       // N_PARAMS
using FaceState  = Handle<face_state_tag, double>;   // N_VARS x n_faces
using FaceScalar = Handle<face_scalar_tag, double>;  // n_faces
using CellScalar = Handle<cell_scalar_tag, double>;  // n_cells
using CellMask   = Handle<cell_mask_tag, int>;       // n_cells
using CellGrad   = Handle<cell_grad_tag, double>;    // N_VARS x n_cells (1-D)
using BCValues   = Handle<bc_values_tag, double>;    // N_VARS x 2 boundary faces

struct Ctx;

// F2 hook: the emitted whole-array Assign hands a generic lambda here.
template <class F, class Target, class... Ops>
inline void zoomy_elementwise(F&& f, Target& t, const Ops&... ops) {
    assert(((ops.size() == t.size()) && ...));
    for (std::size_t i = 0; i < t.size(); ++i) f(t[i], ops[i]...);
}

#include "march_decls.hpp"   // EMITTED: MarchWorkspace + external declarations

struct Mesh {
    int n_cells = 0, n_faces = 0;
    double length = 1.0, dx = 0.0;
    FaceScalar inradius_f;
    CellScalar cell_volumes;
};
struct State {
    CellState Q; CellAux Qaux; Params p;
    double time = 0.0; int iteration = 0; int i_snapshot = 0;
};
struct Ctx {
    Mesh mesh;
    MarchWorkspace ws;
    State state;
    int order = 1;
    long max_steps = 1000;
};

inline void zoomy_alloc_cell_state(Ctx& c, CellState& a) { a.assign(N_VARS * c.mesh.n_cells, 0.0); }
inline void zoomy_alloc_cell_aux(Ctx& c, CellAux& a) { a.assign(N_AUX * c.mesh.n_cells, 0.0); }
inline void zoomy_alloc_face_state(Ctx& c, FaceState& a) { a.assign(N_VARS * c.mesh.n_faces, 0.0); }
inline void zoomy_alloc_face_scalar(Ctx& c, FaceScalar& a) { a.assign(c.mesh.n_faces, 0.0); }
inline void zoomy_alloc_cell_mask(Ctx& c, CellMask& a) { a.assign(c.mesh.n_cells, 0); }
inline void zoomy_alloc_cell_grad(Ctx& c, CellGrad& a) { a.assign(N_VARS * c.mesh.n_cells, 0.0); }
inline void zoomy_alloc_bc_values(Ctx& c, BCValues& a) { (void)c; a.assign(N_VARS * 2, 0.0); }
