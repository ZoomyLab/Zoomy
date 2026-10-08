#pragma once

#include <tuple>
#include <vector>

#include <dune/common/fvector.hh>
#include <dune/grid/common/partitionset.hh>

/* Per-cell geometry and the interior face adjacency the AMR indicator walks.
 *
 * Every entry is rebuilt after each adaptation, so doing it from Python means
 * one interpreter callback per cell and per intersection. On Malpasset that
 * was the single largest cost in the run -- larger than the flux itself.
 * Here it is one compiled pass and a bulk copy back.
 *
 * The pass runs over Partitions::all, NOT the default interior iteration. The
 * vectors are indexed by the index set, which COUNTS GHOSTS: iterating the
 * interior leaves every ghost row at its initial value, and the Python side
 * then divides the wave speed by a zero cell size. Measured on 4 ranks of
 * Malpasset: indexSet.size(0) = 6732 against 6500 interior elements, so 232
 * zero rows per rank, and `compute_dt` returned nan on every rank. Ghost
 * entities carry valid geometry, so filling them is the fix rather than
 * masking them out.
 *
 *   size   |K| / perimeter, the length the CFL condition sees -- NOT sqrt(|K|),
 *          which overestimates it badly on a sliver
 *   level  refinement level, for comparing against the indicator's target
 *   a, b   the two cells of each interior face, each face listed once
 *   n_int  interior elements, i.e. the cells this rank OWNS. Summing
 *          gridView.size(0) over ranks double-counts every ghost.
 */
template< class GridView >
std::tuple< std::vector< double >, std::vector< int >,
            std::vector< int >, std::vector< int >, int >
gridCache ( const GridView &gridView )
{
  const auto &indexSet = gridView.indexSet();
  const std::size_t cells = indexSet.size( 0 );

  std::vector< double > size( cells, 0.0 );
  std::vector< int > level( cells, 0 );
  std::vector< int > a, b;
  a.reserve( 2 * cells );
  b.reserve( 2 * cells );
  int n_int = 0;

  for( const auto &element : elements( gridView, Dune::Partitions::all ) )
  {
    const auto inside = indexSet.index( element );
    const bool interior = ( element.partitionType() == Dune::InteriorEntity );
    if( interior )
      ++n_int;
    double perimeter = 0.0;
    for( const auto &face : intersections( gridView, element ) )
    {
      perimeter += face.geometry().volume();
      if( face.neighbor() && interior )
      {
        const auto outsideEntity = face.outside();
        const auto outside = indexSet.index( outsideEntity );
        /* `inside < outside` lists an interior-interior face once, from the
         * lower-indexed side. A face to a GHOST must be listed unconditionally:
         * the ghost side is never the `inside` of this loop, so the ordering
         * test would drop the face outright whenever inside > outside -- about
         * half of every partition boundary, invisible to the indicator. */
        const bool neighbourInterior =
          ( outsideEntity.partitionType() == Dune::InteriorEntity );
        if( !neighbourInterior || inside < outside )
        {
          a.push_back( int( inside ) );
          b.push_back( int( outside ) );
        }
      }
    }
    size[ inside ] = element.geometry().volume() / perimeter;
    level[ inside ] = element.level();
  }
  return std::make_tuple( size, level, a, b, n_int );
}
