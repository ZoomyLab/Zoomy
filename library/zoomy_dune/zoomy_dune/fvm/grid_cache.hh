#pragma once

#include <tuple>
#include <vector>

#include <dune/common/fvector.hh>

/* Per-cell geometry and the interior face adjacency the AMR indicator walks.
 *
 * Every entry is rebuilt after each adaptation, so doing it from Python means
 * one interpreter callback per cell and per intersection. On Malpasset that
 * was the single largest cost in the run -- larger than the flux itself.
 * Here it is one compiled pass and a bulk copy back.
 *
 *   size  |K| / perimeter, the length the CFL condition sees -- NOT sqrt(|K|),
 *         which overestimates it badly on a sliver
 *   level refinement level, for comparing against the indicator's target
 *   a, b  the two cells of each interior face, each face listed once
 */
template< class GridView >
std::tuple< std::vector< double >, std::vector< int >,
            std::vector< int >, std::vector< int > >
gridCache ( const GridView &gridView )
{
  const auto &indexSet = gridView.indexSet();
  const std::size_t cells = indexSet.size( 0 );

  std::vector< double > size( cells, 0.0 );
  std::vector< int > level( cells, 0 );
  std::vector< int > a, b;
  a.reserve( 2 * cells );
  b.reserve( 2 * cells );

  for( const auto &element : elements( gridView ) )
  {
    const auto inside = indexSet.index( element );
    double perimeter = 0.0;
    for( const auto &face : intersections( gridView, element ) )
    {
      perimeter += face.geometry().volume();
      if( face.neighbor() )
      {
        const auto outside = indexSet.index( face.outside() );
        if( inside < outside )
        {
          a.push_back( int( inside ) );
          b.push_back( int( outside ) );
        }
      }
    }
    size[ inside ] = element.geometry().volume() / perimeter;
    level[ inside ] = element.level();
  }
  return std::make_tuple( size, level, a, b );
}
