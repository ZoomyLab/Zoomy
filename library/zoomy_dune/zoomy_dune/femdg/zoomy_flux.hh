#pragma once

#include <algorithm>
#include <dune/fem-dg/operator/fluxes/advection/python.hh>

#include "Numerics.H"
#include "ZoomyParameters.H"

/* Adapter between two fixed calling conventions.
 *
 * zoomy_core prints the Riemann solver as free functions over raw pointers;
 * dune-fem-dg calls a class member over FieldVectors and writes its result
 * into two output states. Nothing here is model-specific -- every size and
 * every expression comes from the generated headers -- so this file is the
 * same for every model and is not itself generated.
 *
 * Sign convention, read off dune/fem-dg/pass/dgpass.hh:904,
 *
 *     fluxEn *= -faceQuadInner.weight(l);   // gLeft  -> inside cell, MINUS
 *     fluxNb *=  faceQuadOuter.weight(l);   // gRight -> neighbour,   PLUS
 *
 * against zoomy_core/fvm/solver_numpy.py:1070, which is the reference
 * implementation of the same scheme,
 *
 *     dQ[A] += -( F + Dm) * |e| / |A|      // A is the cell n points out of
 *     dQ[B] += -(-F + Dp) * |e| / |B|
 *
 * so that gLeft = F + Dm and gRight = F - Dp, with Dp the FIRST block of
 * numerical_fluctuations and Dm the second.
 */
template <class ModelImp>
struct ZoomyFlux : public Dune::Fem::DGAdvectionFluxPythonUserDefine< ModelImp >
{
  typedef Dune::Fem::DGAdvectionFluxPythonUserDefine< ModelImp > BaseType;
  typedef typename BaseType::RangeType          RangeType;
  typedef typename BaseType::JacobianRangeType  JacobianRangeType;
  typedef typename BaseType::DomainType         DomainType;

  typedef Model< double >    ZoomyModel;
  typedef Numerics< double > ZoomyNumerics;

  static constexpr int nvar = ZoomyNumerics::n_dof_q;
  static constexpr int naux = ZoomyModel::n_dof_qaux;
  static constexpr int ndim = DomainType::dimension;

  static_assert( nvar == int(RangeType::dimension),
                 "UFL Model dimRange does not match the printed n_dof_q" );

  ZoomyFlux ( const ModelImp& model ) : BaseType( model ) {}

  std::string name () const { return "zoomy"; }

  template< class LocalContext >
  double numericalFlux ( const LocalContext& left,  const LocalContext& right,
                         const RangeType& uLeft,    const RangeType& uRight,
                         const JacobianRangeType&,  const JacobianRangeType&,
                         RangeType& gLeft,          RangeType& gRight ) const
  {
    const auto& xFace = left.localPosition();
    auto normal = left.intersection().integrationOuterNormal( xFace );
    const double faceArea = normal.two_norm();
    normal *= 1.0 / faceArea;

    const auto xGlobal = left.intersection().geometry().global( xFace );
    double n[ ndim ], X[ ndim ];
    for( int d = 0; d < ndim; ++d ) { n[ d ] = normal[ d ]; X[ d ] = xGlobal[ d ]; }

    /* Aux is evaluated from the state at the face rather than read off a
     * stored field: the Runge-Kutta stages inside femDGOperator give no hook
     * to refresh one, and a stale aux is a silently wrong flux. Exact while
     * aux is pointwise in Q; a model whose aux carries a derivative needs the
     * jacobians this signature also offers. */
    const double zero[ naux ] = {};
    auto auxLeft  = ZoomyModel::update_aux_variables( &uLeft[ 0 ],  zero,
                                                      zoomy_parameters, 0.0, X );
    auto auxRight = ZoomyModel::update_aux_variables( &uRight[ 0 ], zero,
                                                      zoomy_parameters, 0.0, X );

    auto F = ZoomyNumerics::numerical_flux(
        &uLeft[ 0 ], &uRight[ 0 ], auxLeft.begin(), auxRight.begin(),
        zoomy_parameters, n );

    if constexpr ( ZoomyNumerics::fluctuations_are_zero )
    {
      for( int i = 0; i < nvar; ++i )
        gLeft[ i ] = gRight[ i ] = F[ i ] * faceArea;
    }
    else
    {
      auto D = ZoomyNumerics::numerical_fluctuations(
          &uLeft[ 0 ], &uRight[ 0 ], auxLeft.begin(), auxRight.begin(),
          zoomy_parameters, n );
      for( int i = 0; i < nvar; ++i )
      {
        gLeft [ i ] = ( F[ i ] + D[ nvar + i ] ) * faceArea;
        gRight[ i ] = ( F[ i ] - D[ i ]        ) * faceArea;
      }
    }

    auto sLeft  = ZoomyNumerics::local_max_abs_eigenvalue(
        &uLeft[ 0 ],  auxLeft.begin(),  zoomy_parameters, n );
    auto sRight = ZoomyNumerics::local_max_abs_eigenvalue(
        &uRight[ 0 ], auxRight.begin(), zoomy_parameters, n );

    return std::max( sLeft[ 0 ], sRight[ 0 ] ) * faceArea;
  }

  using BaseType::model;
};
