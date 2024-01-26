
#ifndef _SARRAY_H_
#define _SARRAY_H_

#include <iostream>
#include <iomanip>

/*
  This is intended to be a simple, low-overhead class to do multi-dimensional arrays
  without pointer dereferencing. It supports indexing and cout only up to 3-D.

  It templates based on array dimension sizes, which conveniently allows overloaded
  functions in the TransformMatrices class.
*/

template <class T, unsigned long D0, unsigned long D1=1, unsigned long D2=1, unsigned long D3=1> class SArray {

protected:

  typedef unsigned long ulong;

  T data[D0*D1*D2];

public :

  KOKKOS_INLINE_FUNCTION SArray() { }
  KOKKOS_INLINE_FUNCTION ~SArray() { }

  KOKKOS_INLINE_FUNCTION T &operator()(ulong const i0)       {
    #ifdef ARRAY_DEBUG
      if (D1*D2*D3 > 1) {std::cout << "SArray: Using 2D or higher array as 1D array\n";}
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
    #endif
    return data[i0];
  }
  KOKKOS_INLINE_FUNCTION T &operator()(ulong const i0, ulong const i1)       {
    #ifdef ARRAY_DEBUG
      if (D2*D3 > 1) {std::cout << "SArray: Using 3D or higher array as 2D array\n";}
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
    #endif
    return data[i0*D1 + i1];
  }
  KOKKOS_INLINE_FUNCTION T &operator()(ulong const i0, ulong const i1, ulong const i2)       {
    #ifdef ARRAY_DEBUG
      if (D3 > 1) {std::cout << "SArray: Using 4D or higher array as 3D array\n";}
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
    #endif
    return data[i0*D1*D2 + i1*D2 + i2];
  }
  KOKKOS_INLINE_FUNCTION T &operator()(ulong const i0, ulong const i1, ulong const i2, ulong const i3)       {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
      if (i3>D3-1) { printf("i3 > D3-1"); exit(-1); }
    #endif
    return data[i0*D1*D2*D3 + i1*D2*D3 + i2*D3 + i3];
  }

  KOKKOS_INLINE_FUNCTION T  operator()(ulong const i0) const {
    #ifdef ARRAY_DEBUG
      if (D1*D2*D3 > 1) {std::cout << "SArray: Using 2D or higher array as 1D array\n";}
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
    #endif
    return data[i0];
  }
  KOKKOS_INLINE_FUNCTION T  operator()(ulong const i0, ulong const i1) const {
    #ifdef ARRAY_DEBUG
      if (D2*D3 > 1) {std::cout << "SArray: Using 3D or higher array as 2D array\n";}
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
    #endif
    return data[i0*D1 + i1];
  }
  KOKKOS_INLINE_FUNCTION T  operator()(ulong const i0, ulong const i1, ulong const i2) const {
    #ifdef ARRAY_DEBUG
      if (D3 > 1) {std::cout << "SArray: Using 4D or higher array as 3D array\n";}
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
    #endif
    return data[i0*D1*D2 + i1*D2 + i2];
  }
  KOKKOS_INLINE_FUNCTION T  operator()(ulong const i0, ulong const i1, ulong const i2, ulong const i3) const {
    #ifdef ARRAY_DEBUG
      if (i0>D0-1) { printf("i0 > D0-1"); exit(-1); }
      if (i1>D1-1) { printf("i1 > D1-1"); exit(-1); }
      if (i2>D2-1) { printf("i2 > D2-1"); exit(-1); }
      if (i3>D3-1) { printf("i3 > D3-1"); exit(-1); }
    #endif
    return data[i0*D1*D2*D3 + i1*D2*D3 + i2*D3 + i3];
  }

  template <class I, ulong E0>  KOKKOS_INLINE_FUNCTION SArray<T,E0> operator*(SArray<I,D0> const &rhs) {
    //This template could match either vector-vector or matrix-vector multiplication
    if ( (D1*D2*D3 == 1) ) {
      // Both 1-D Arrays --> Element-wise multiplication
      SArray<T,D0> ret;
      for (ulong i=0; i<D0; i++) {
        ret.data[i] = data[i] * rhs.data[i];
      }
      return ret;
    } else {
      // Matrix-Vector multiplication
      SArray<T,D1> ret;
      for (ulong j=0; j<D1; j++) {
        T tot = 0;
        for (ulong i=0; i<D0; i++) {
          tot += (*this)(i,j) * rhs(i);
        }
        ret(j) = tot;
      }
      return ret;
    }
  }

  template <class I, ulong E0> KOKKOS_INLINE_FUNCTION SArray<T,E0,D1> operator*(SArray<I,E0,D0> const &rhs) {
    //This template matches Matrix-Matrix multiplication
    SArray<T,E0,D1> ret;
    for (ulong j=0; j<E0; j++) {
      for (ulong i=0; i<D1; i++) {
        T tot = 0;
        for (ulong k=0; k<D0; k++) {
          tot += (*this)(k,i) * rhs(j,k);
        }
        ret(j,i) = tot;
      }
    }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION void operator=(T rhs) {
    //Scalar assignment
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      data[i] = rhs;
    }
  }

  KOKKOS_INLINE_FUNCTION T sum() {
    //Scalar division
    T sum = 0.;
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      sum += data[i];
    }
    return sum;
  }

  KOKKOS_INLINE_FUNCTION void operator/=(T rhs) {
    //Scalar division
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      data[i] = data[i] / rhs;
    }
  }

  KOKKOS_INLINE_FUNCTION void operator*=(T rhs) {
    //Scalar multiplication
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      data[i] = data[i] * rhs;
    }
  }

  KOKKOS_INLINE_FUNCTION SArray<T,D0,D1,D2> operator*(T rhs) {
    //Scalar multiplication
    SArray<T,D0,D1,D2> ret;
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      ret.data[i] = data[i] * rhs;
    }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION SArray<T,D0,D1,D2> operator/(T rhs) {
    //Scalar division
    SArray<T,D0,D1,D2> ret;
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      ret.data[i] = data[i] / rhs;
    }
    return ret;
  }

  inline friend std::ostream &operator<<(std::ostream& os, SArray const &v) {
    if (D1*D2*D3 == 1) {
      for (ulong i=0; i<D0; i++) {
        os << std::setw(12) << v(i) << "\n";
      }
    } else if (D2*D3 == 1) {
      for (ulong j=0; j<D1; j++) {
        for (ulong i=0; i<D0; i++) {
          os << std::setw(12) << v(i,j) << " ";
        }
        os << "\n";
      }
    } else {
      for (ulong i=0; i<D0*D1*D2*D3; i++) {
        os << std::setw(12) << v.data[i] << "\n";
      }
    }
    return os;
  }

KOKKOS_INLINE_FUNCTION  const SArray<T,D0,D1,D2> operator+(SArray<T,D0,D1,D2> const rhs) {
    SArray<T,D0,D1,D2> ret;
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      ret.data[i] = data[i] + rhs.data[i];
    }
    return ret;
  }
KOKKOS_INLINE_FUNCTION  const SArray<T,D0,D1,D2> operator-(SArray<T,D0,D1,D2> const rhs) {
    SArray<T,D0,D1,D2> ret;
    for (ulong i=0; i<D0*D1*D2*D3; i++) {
      ret.data[i] = data[i] - rhs.data[i];
    }
    return ret;
  }

  KOKKOS_INLINE_FUNCTION SArray<T,D0,D1,D2> unit() {
    // Unit vector
    T sum = 0.;
    for (ulong i=0; i<D0*D1*D2*D3; i++){
      sum += data[i]*data[i];
    }
    SArray<T,D0,D1,D2> ret;
    for (ulong i=0; i<D0*D1*D2*D3; i++){
      ret.data[i] = data[i]/mysqrt(sum);
    }
    return ret;
  }


};



#endif
