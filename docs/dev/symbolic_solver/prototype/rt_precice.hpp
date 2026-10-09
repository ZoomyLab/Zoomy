// C++ preCICE adapter: the couple_* effects.  API calls only.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <precice/precice.hpp>
#include "rt.hpp"

namespace zs {
struct Cpl {
    std::unique_ptr<precice::Participant> p;
    std::string mesh;
    std::vector<std::string> wn, rn;
    std::vector<precice::VertexID> ids{0};
};
inline Cpl& cpl() { static Cpl c; return c; }
inline void cpl_setup(const std::string& name, const std::string& cfg, const std::string& mesh,
                      std::vector<std::string> wn, std::vector<std::string> rn, double x) {
    auto& c = cpl();
    c.p = std::make_unique<precice::Participant>(name, cfg, 0, 1);
    c.mesh = mesh; c.wn = wn; c.rn = rn;
    std::vector<double> xy{x, 0.0};
    c.p->setMeshVertices(mesh, xy, c.ids);
}
inline void cpl_write(const Field& q) {
    auto& c = cpl();
    for (size_t i = 0; i < c.wn.size(); ++i) { std::vector<double> v{q(int(i), 0)}; c.p->writeData(c.mesh, c.wn[i], c.ids, v); }
}
inline std::tuple<long, bool> couple_init(long tok, const Field& q) {
    if (cpl().p->requiresInitialData()) cpl_write(q);
    cpl().p->initialize();
    return {tok + 1, cpl().p->isCouplingOngoing()};
}
inline std::tuple<long, double> couple_max_dt(long tok) { return {tok + 1, cpl().p->getMaxTimeStepSize()}; }
inline std::tuple<long, Field> couple_read(long tok, double dt) {
    auto& c = cpl();
    Field q(int(c.rn.size()), 1);
    for (size_t i = 0; i < c.rn.size(); ++i) { std::vector<double> v(1); c.p->readData(c.mesh, c.rn[i], c.ids, dt, v); q(int(i), 0) = v[0]; }
    return {tok + 1, q};
}
inline std::tuple<long> couple_write(long tok, const Field& q) { cpl_write(q); return {tok + 1}; }
inline std::tuple<long, bool> couple_advance(long tok, double dt) { cpl().p->advance(dt); return {tok + 1, cpl().p->isCouplingOngoing()}; }
}  // namespace zs
