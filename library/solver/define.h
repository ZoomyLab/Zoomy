
#ifndef _CONST_H_
#define _CONST_H_

#include "math.h"
#include "const.h"
#include "mpi.h"
#include <Kokkos_Core.hpp>

#define SERGHEI_FLOAT 1
#define SERGHEI_DOUBLE 2

#ifndef SERGHEI_REAL
  #define SERGHEI_REAL SERGHEI_DOUBLE
#endif

#ifndef SERGHEI_MPI_REAL 
  #if SERGHEI_REAL == SERGHEI_DOUBLE
    #define SERGHEI_MPI_REAL MPI_DOUBLE
  #elif SERGHEI_REAL == SERGHEI_FLOAT
    #define SERGHEI_MPI_REAL MPI_FLOAT
  #endif
#endif

#if SERGHEI_REAL == SERGHEI_DOUBLE
  #define SERGHEI_VTK_REAL "DOUBLE"
#elif SERGHEI_REAL == SERGHEI_FLOAT
  #define SERGHEI_VTK_REAL "FLOAT"
#endif

#if SERGHEI_REAL == SERGHEI_DOUBLE
  typedef double         real;
  #define TOL_MASS_ERROR TOL8
  #define TOLDRY TOL12
  #define TOL_ZERO_MOMENTUM TOL12
  #define TOL_WETDRY TOL12
  #define TOL_MACHINE_ACCURACY TOL12
#endif
#if SERGHEI_REAL == SERGHEI_FLOAT
  typedef float         real;
  #define TOL_MASS_ERROR TOL5
  #define TOLDRY TOL6
  #define TOL_ZERO_MOMENTUM TOL12
  #define TOL_WETDRY TOL12
  #define TOL_MACHINE_ACCURACY TOL6
#endif
typedef unsigned long ulong;
typedef unsigned int  uint;


#ifdef __NVCC__
  typedef Kokkos::View<real*     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> realArr;
  typedef Kokkos::View<real**     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> realArr2;
  typedef Kokkos::View<real***     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> realArr3;
  typedef Kokkos::View<int*     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> intArr;
  typedef Kokkos::View<int**     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> intArr2;
  typedef Kokkos::View<int***     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> intArr3;
  typedef Kokkos::View<bool*     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> boolArr;
  typedef Kokkos::View<bool**     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> boolArr2;
  typedef Kokkos::View<bool***     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> boolArr3;
  typedef Kokkos::View<double*     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> doubleArr;
  typedef Kokkos::View<double**     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> doubleArr2;
  typedef Kokkos::View<double***     ,Kokkos::LayoutLeft,Kokkos::Device<Kokkos::Cuda,Kokkos::CudaUVMSpace>> doubleArr3;
#else
  typedef Kokkos::View<real*     ,Kokkos::LayoutLeft> realArr;
  typedef Kokkos::View<real**     ,Kokkos::LayoutLeft> realArr2;
  typedef Kokkos::View<real***     ,Kokkos::LayoutLeft> realArr3;
  typedef Kokkos::View<int*     ,Kokkos::LayoutLeft> intArr;
  typedef Kokkos::View<int**     ,Kokkos::LayoutLeft> intArr2;
  typedef Kokkos::View<int***     ,Kokkos::LayoutLeft> intArr3;
  typedef Kokkos::View<bool*     ,Kokkos::LayoutLeft> boolArr;
  typedef Kokkos::View<bool**     ,Kokkos::LayoutLeft> boolArr2;
  typedef Kokkos::View<bool***     ,Kokkos::LayoutLeft> boolArr3;
  typedef Kokkos::View<double**     ,Kokkos::LayoutLeft> doubleArr;
  typedef Kokkos::View<double**     ,Kokkos::LayoutLeft> doubleArr2;
  typedef Kokkos::View<double***     ,Kokkos::LayoutLeft> doubleArr3;
#endif

#ifdef __NVCC__
#define _HOSTDEV __host__ __device__
#else
#define _HOSTDEV
#endif

KOKKOS_INLINE_FUNCTION real operator"" _fp( long double x ) {
  return static_cast<real>(x);
}

KOKKOS_INLINE_FUNCTION double mypow ( double const x , double const p ) { return pow (x,p); }
KOKKOS_INLINE_FUNCTION float  mypow ( float  const x , float  const p ) { return powf(x,p); }
KOKKOS_INLINE_FUNCTION double mysqrt( double const x ) { return sqrt (x); }
KOKKOS_INLINE_FUNCTION float  mysqrt( float  const x ) { return sqrtf(x); }
KOKKOS_INLINE_FUNCTION double myfabs( double const x ) { return fabs (x); }
KOKKOS_INLINE_FUNCTION float  myfabs( float  const x ) { return fabsf(x); }

/*
template <class T> KOKKOS_INLINE_FUNCTION T min( T const v1 , T const v2 ) {
  if (v1 < v2) { return v1; }
  else         { return v2; }
}

template <class T> KOKKOS_INLINE_FUNCTION T max( T const v1 , T const v2 ) {
  if (v1 > v2) { return v1; }
  else         { return v2; }
}
*/
template <class T1, class T2> KOKKOS_INLINE_FUNCTION T1 min (T1 const v1, T2 const v2){
  if (v1 < v2) { return (T1) v1; }
  else         { return (T1) v2; }
}

template <class T1, class T2> KOKKOS_INLINE_FUNCTION T1 max (T1 const v1, T2 const v2){
  if (v1 > v2) { return (T1) v1; }
  else         { return (T1) v2; }
}

template <class T> KOKKOS_INLINE_FUNCTION int sgn(T const val) {
    return (T(0) < val) - (val < T(0));
}


/*
// a bit hacky, perhaps better to include in template
KOKKOS_INLINE_FUNCTION float min(float v1, double v2){
  return(min(v1,(float) v2));
}
KOKKOS_INLINE_FUNCTION float max(float v1, double v2){
  return(max(v1,(float) v2));
}
*/


class SergheiTimers{
public:
  real total=0;
  real init=0;
  real out=0;
  real swe=0;
  real raininf=0;
  real sweflux=0;
  real exchange=0;
  real integrate=0;
  real dt=0;
  real sweBC=0;
};

#endif
