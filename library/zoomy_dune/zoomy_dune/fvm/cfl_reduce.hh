#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <dune/grid/common/partitionset.hh>
#include <dune/geometry/referenceelements.hh>
#include <dune/fem/function/localfunction/const.hh>

#include "Model.H"
#include "Numerics.H"
#include "ZoomyParameters.H"

/* The CFL reduction, compiled against the SAME printed kernels the flux uses.
 *
 *     max over owned cells of  lambda_max(Q_K) / ( |K| / perimeter(K) )
 *
 * Python divides CFL by the global maximum of this. Doing it instead by
 * interpolating the wave speed into a discrete function costs more per step
 * than the whole flux sweep -- measured at roughly 190 ms against 12 ms on
 * 70k cells over 16 ranks, which is why the dt had to be computed here.
 *
 * The cell size is taken from the grid rather than passed in. It could be read
 * from the cached sizes grid_cache.hh builds, but then the dt would depend on
 * that cache being rebuilt after every adapt, and a size indexed for the old
 * grid is silently a wrong dt rather than an error. Recomputing the perimeter
 * is a handful of flops against the state evaluation already done per cell.
 *
 * Non-finite is returned, not skipped. std::max( a, NaN ) is a < NaN ? NaN : a,
 * which is `a` -- so a NaN cell reduces to nothing and the dt comes back
 * SMALLER than the truth, the marker de-refines around the hole and the run
 * marches to time_end reporting success. That cost a 160 s AMR run that
 * announced it had survived with an all-NaN field. Python raises on the
 * reduced value, after its collective.
 *
 * Interior cells only: a ghost is some other rank's to limit, and its own rank
 * reduces it. The caller still has to comm.max across ranks.
 *
 * The spectrum is sampled along the two axis normals rather than per face. For
 * a system whose fastest wave is not axis-aligned that UNDER-estimates the
 * speed, so it is the same convention the UFL path uses and no weaker.
 */
template< class GridView, class DiscreteFunction >
double cflReduce ( const GridView &gridView, const DiscreteFunction &Q )
{
  typedef Model< double >    ZoomyModel;
  typedef Numerics< double > ZoomyNumerics;
  static constexpr int nvar = ZoomyNumerics::n_dof_q;
  static constexpr int naux = ZoomyModel::n_dof_qaux;

  Dune::Fem::ConstLocalFunction< DiscreteFunction > local( Q );
  typename DiscreteFunction::RangeType value;

  const double normals[ 2 ][ 2 ] = { { 1.0, 0.0 }, { 0.0, 1.0 } };
  const double zero[ naux ] = {};
  double worst = 0.0;

  for( const auto &element : elements( gridView, Dune::Partitions::interior ) )
  {
    const auto &geometry = element.geometry();
    const auto centre = Dune::ReferenceElements< double, GridView::dimension >
        ::general( element.type() ).position( 0, 0 );

    local.bind( element );
    local.evaluate( centre, value );
    local.unbind();

    double state[ nvar ];
    for( int i = 0; i < nvar; ++i )
      state[ i ] = value[ i ];

    const auto global = geometry.center();
    double X[ GridView::dimensionworld ];
    for( int d = 0; d < GridView::dimensionworld; ++d )
      X[ d ] = global[ d ];

    auto aux = ZoomyModel::update_aux_variables( state, zero,
                                                 zoomy_parameters, 0.0, X );
    double speed = 0.0;
    for( int axis = 0; axis < 2; ++axis )
    {
      auto s = ZoomyNumerics::local_max_abs_eigenvalue(
          state, aux.begin(), zoomy_parameters, normals[ axis ] );
      speed = std::max( speed, std::abs( s[ 0 ] ) );
    }

    double perimeter = 0.0;
    for( const auto &face : intersections( gridView, element ) )
      perimeter += face.geometry().volume();
    const double size = geometry.volume() / std::max( perimeter, 1e-300 );
    const double ratio = speed / std::max( size, 1e-12 );

    if( !std::isfinite( ratio ) )
      return std::numeric_limits< double >::infinity();
    worst = std::max( worst, ratio );
  }
  return worst;
}
