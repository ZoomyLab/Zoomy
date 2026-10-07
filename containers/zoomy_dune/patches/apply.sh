#!/bin/sh
# dune-fem-dg 2.12.0.2: make the user-defined advection flux instantiable.
#
# dgpyoperator.hh builds its discrete model around
#     ModelType = ModelWrapper<Grid, AdvModel, DiffModel, Additional, Limiter, Problem>
# but binds the userdefined flux to the RAW advection model, and then
# discretemodelcommon.hh asks that flux for a ModelType& -- which no user class
# can supply, because Additional and the limiter function are chosen inside the
# operator and are not exposed. Upstream's own pydemo/camc-paper/sw_wb.py fails
# the same way; master 8b6dcc3 (2026-10-06) adds a static_assert naming the
# mismatch but does not repair it.
#
# Two edits, both in dgpyoperator.hh: bind the flux to ModelType like every
# other flux, and let the operator construct it instead of demanding one
# through the constructor.
set -e
F="${1:-/usr/local/include/dune/fem-dg/operator/dg/dgpyoperator.hh}"
O="${2:-/usr/local/lib/python3.12/dist-packages/dune/femdg/_operators.py}"

# femDGOperator has no way to add headers to the generated module, and
# advModel.cppIncludes hands back a copy, so appending to it is a no-op. Let
# the Model carry them.
python3 - "$O" <<'PYO'
import io, sys
path = sys.argv[1]
s = io.open(path).read()
if "ZOOMY_USER_INCLUDES_PATCH" not in s:
    old = """    includes += advModel.cppIncludes + diffModel.cppIncludes"""
    new = """    includes += advModel.cppIncludes + diffModel.cppIncludes
    # ZOOMY_USER_INCLUDES_PATCH: headers a user-defined flux needs
    includes += list(getattr(Model, "userFluxIncludes", []))"""
    assert s.count(old) == 1, "include block not found -- version drift?"
    io.open(path, "w").write(s.replace(old, new))
    print("patched " + path)
PYO

grep -q "ZOOMY_USERDEFINED_FLUX_PATCH" "$F" && exit 0

python3 - "$F" <<'PY'
import io, sys
path = sys.argv[1]
s = io.open(path).read()

old = """    typedef typename std::conditional< fluxIsUserDefined,
                                       DGAdvectionFlux< AdvectionModel, advFluxId >,
                                       DGAdvectionFlux< ModelType, advFluxId > > :: type  AdvectionFluxType;"""
new = """    // ZOOMY_USERDEFINED_FLUX_PATCH: bind the userdefined flux to ModelType as
    // every other flux is bound; the raw AdvectionModel can never answer the
    // ModelType& that DiscreteModelCommon asks it for.
    typedef DGAdvectionFlux< ModelType, advFluxId >  AdvectionFluxType;"""
assert s.count(old) == 1, "flux-type typedef not found -- version drift?"
s = s.replace(old, new)

old = """      DUNE_THROW(InvalidStateException,"DGOperator::DGOPerator: When advFluxId is userdefined, flux needs to be passed in constructor!");
      return *((AdvectionFluxType *) 0);"""
new = """      // ZOOMY_USERDEFINED_FLUX_PATCH: constructible from the wrapper now, so
      // build it here rather than requiring one through the constructor.
      advFluxPtr_.reset( new AdvectionFluxType( model_, parameter ) );
      return *advFluxPtr_;"""
assert s.count(old) == 1, "userdefined flux ctor guard not found -- version drift?"
s = s.replace(old, new)

io.open(path, "w").write(s)
print("patched " + path)
PY
