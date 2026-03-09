//===========================================================================
//
// File: blas_lapack.hpp
//
// Created: Sun Jun 21 18:56:51 2009
//
// Author(s): Bård Skaflestad     <bard.skaflestad@sintef.no>
//            Atgeirr F Rasmussen <atgeirr@sintef.no>
//
// $Date$
//
// $Revision$
//
//===========================================================================

/*
  Copyright 2009, 2010 SINTEF ICT, Applied Mathematics.
  Copyright 2009, 2010 Statoil ASA.

  This file is part of The Open Reservoir Simulator Project (OpenRS).

  OpenRS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OpenRS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OpenRS.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPENRS_BLAS_LAPACK_HEADER
#define OPENRS_BLAS_LAPACK_HEADER

#if ! __has_include(<FCMacros.h>)
#error "Need FCMacros.h to be generated through FortranCInterface"
#endif

#include <opm/common/ErrorMacros.hpp>
#include <opm/porsol/common/fortran.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#include <FCMacros.h>

#define F77_CHARACTER_TYPE const char*

// y <- a1*op(A)*x + a2*y  where  op(X) in {X, X.'}
void FC_GLOBAL(dgemv,DGEMV)(F77_CHARACTER_TYPE,
                            const int*    m   , const int*    n,
                            const double* a1  , const double* A, const int* ldA ,
                            const double* x, const int* incX,
                            const double* a2  ,       double* y, const int* incY);


    // C <- a1*op(A)*op(B) + a2*C  where  op(X) in {X, X.'}
void FC_GLOBAL(dgemm,DGEMM)(F77_CHARACTER_TYPE, F77_CHARACTER_TYPE,
               const int*    m   , const int*    n   , const int* k  ,
               const double* a1  , const double* A   , const int* ldA,
                                   const double* B   , const int* ldB,
               const double* a2  ,       double* C   , const int* ldC);

// C <- a1*A*A' + a2*C   *or*   C <- a1*A'*A + a2*C
void FC_GLOBAL(dsyrk,DSYRK)(F77_CHARACTER_TYPE, F77_CHARACTER_TYPE,
                            const int*    n   , const int*    k   ,
                            const double* a1  , const double* A   , const int* ldA,
                            const double* a2  ,       double* C   , const int* ldC);

// B <- a*op(A)*B  *or*  B <- a*B*op(A)  where op(X) \in {X, X.', X'}
void FC_GLOBAL(dtrmm,DTRMM)(F77_CHARACTER_TYPE, F77_CHARACTER_TYPE,
                            F77_CHARACTER_TYPE, F77_CHARACTER_TYPE,
                            const int*    m   , const int* n      ,
                            const double* a   ,
                            const double* A   , const int* ldA    ,
                                  double* B   , const int* ldB);


void FC_GLOBAL(dgeqrf,DGEQRF)(const int*    m    , const int*    n   ,
                              double* A    , const int*    ld  ,
                              double* tau  ,       double* work,
                              const int*    lwork,       int*    info);

void FC_GLOBAL(dorgqr,DORGQR)(const int*    m   , const int* n    , const int*    k  ,
                              double* A   , const int* ld   , const double* tau,
                              double* work, const int* lwork,       int*    info);

void FC_GLOBAL(dgetrf,DGETRF)(const int*    m   , const int* n ,
                                    double* A   , const int* ld,
                                    int*    ipiv,       int* info);

void FC_GLOBAL(dgetri,DGETRI)(const int*    n   ,
                                    double* A   , const int* ld,
                              const int*    ipiv,
                                    double* work,       int* lwork, int* info);

#ifdef __cplusplus
}
#endif

namespace Opm {
    namespace BLAS_LAPACK {
        //--------------------------------------------------------------------------
        /// @brief GEneral Matrix Vector product (Level 2 BLAS).
        ///
        /// @tparam T Element type of matrix/vector.
        ///
        /// @param transA
        ///    Specifcation of @f$ \mathrm{op} @f$.  @code transA = 'N'
        ///    @endcode yields @f$ \mathrm{op}(A) = A @f$ while @code
        ///    transA = 'T' @endcode yields @f$ \mathrm{op}(A) =
        ///    A^{\mathsf{T}} @f$.
        ///
        /// @param [in] m
        ///    Number of matrix rows (and number of rows for which new
        ///    data will be assigned to the result vector @f$ y @f$.)
        ///
        /// @param [in] n
        ///    Number of matrix columns (and number of rows for which data
        ///    will be retrieved from the input vector @f$ x @f$).
        ///
        /// @param [in] a1
        ///    Scalar coefficient @f$ a_1 @f$.
        ///
        /// @param [in] A
        ///    Pointer to first data element of matrix @f$ A @f$.
        ///
        /// @param [in] ldA
        ///    Leading dimension of storage std::array containing the matrix
        ///    @f$ A @f$.
        ///
        /// @param [in] x
        ///    Input vector @f$ x @f$.
        ///
        /// @param [in] incX
        ///    Data element stride for input vector @f$ x @f$.
        ///
        /// @param [in] a2
        ///    Scalar coefficient @f$ a_2 @f$.
        ///
        /// @param y
        ///    Result vector @f$ y @f$.
        ///
        /// @param [in] incY
        ///    Data element stride for result vector @f$ y @f$.
        template<typename T>
        void GEMV(const char* transA,
                  const int   m     , const int   n,
                  const T&    a1    , const T*    A, const int ldA ,
                                      const T*    x, const int incX,
                  const T&    a2    ,       T*    y, const int incY);

        /// @brief GEneral Matrix Vector product specialization for double.
        template<>
        void GEMV<double>(const char*   transA,
                          const int     m     , const int     n,
                          const double& a1    , const double* A, const int ldA,
                                                const double* x, const int incX,
                          const double& a2    ,       double* y, const int incY);

        //--------------------------------------------------------------------------
        /// @brief GEneral Matrix Matrix product (Level 3 BLAS).
        ///
        /// @tparam T Element type of matrix.
        ///
        /// @brief
        ///    Double precision general matrix-matrix product matrix
        ///    update.  Specifically, @f$ C \leftarrow a_1 \mathrm{op}(A)
        ///    \mathrm{op}(B) + a_2 C @f$.
        ///
        /// @param [in] transA
        ///    Specifcation of @f$ \mathrm{op} @f$.  @code transA = 'N'
        ///    @endcode yields @f$ \mathrm{op}(A) = A @f$ while @code
        ///    transA = 'T' @endcode yields @f$ \mathrm{op}(A) =
        ///    A^{\mathsf{T}} @f$.
        ///
        /// @param [in] transB
        ///    Specifcation of @f$ \mathrm{op} @f$.  @code transB = 'N'
        ///    @endcode yields @f$ \mathrm{op}(B) = A @f$ while @code
        ///    transB = 'T' @endcode yields @f$ \mathrm{op}(B) =
        ///    B^{\mathsf{T}} @f$.
        ///
        /// @param [in] m
        ///    Number of rows of matrix @f$ \mathrm{op}(A) @f$ and of
        ///    matrix @f$ C @f$.
        ///
        /// @param [in] n
        ///    Number of columns of matrix @f$ \mathrm{op}(B) @f$ and of
        ///    matrix @f$ C @f$.
        ///
        /// @param [in] k
        ///    Number of columns of matrix @f$ \mathrm{op}(A) @f$ and
        ///    number of rows of matrix @f$ \mathrm{op}(B) @f$.
        ///
        /// @param [in] a1
        ///    Scalar coefficient @f$ a_1 @f$.
        ///
        /// @param [in] A
        ///    Pointer to first data element of matrix @f$ A @f$.
        ///
        /// @param [in] ldA
        ///    Leading dimension of storage std::array containing the matrix
        ///    @f$ A @f$.
        ///
        /// @param [in] B
        ///    Pointer to first data element of matrix @f$ B @f$.
        ///
        /// @param [in] ldB
        ///    Leading dimension of storage std::array containing the matrix
        ///    @f$ B @f$.
        ///
        /// @param [in] a2
        ///    Scalar coefficient @f$ a_2 @f$.
        ///
        /// @param C
        ///    Pointer to first data element of matrix @f$ C @f$.
        ///
        /// @param [in] ldC
        ///    Leading dimension of storage std::array containing the matrix
        ///    @f$ C @f$.
        template<typename T>
        void GEMM(const char* transA, const char* transB,
                  const int   m     , const int   n     , const int k  ,
                  const T&    a1    , const T*    A     , const int ldA,
                                      const T*    B     , const int ldB,
                  const T&    a2    ,       T*    C     , const int ldC);

        /// @brief GEneral Matrix Matrix product specialization for double.
        template<>
        void GEMM<double>(const char*   transA, const char*   transB,
                          const int     m     , const int     n     , const int k  ,
                          const double& a1    , const double* A     , const int ldA,
                                                const double* B     , const int ldB,
                          const double& a2    ,       double* C     , const int ldC);


        //--------------------------------------------------------------------------
        /// @brief SYmmetric Rank K update of symmetric matrix (Level 3 BLAS)
        ///
        /// @tparam T Matrix element type.
        template<typename T>
        void SYRK(const char* uplo, const char* trans,
                  const int   n   , const int   k    ,
                  const T&    a1  , const T*    A    , const int ldA,
                  const T&    a2  ,       T*    C    , const int ldC);

        /// @brief SYmmetric Rank K update of symmetric matrix specialization for double.
        template<>
        void SYRK<double>(const char*   uplo, const char*   trans,
                          const int     n   , const int     k    ,
                          const double& a1  , const double* A    , const int ldA,
                          const double& a2  ,       double* C    , const int ldC);

        //--------------------------------------------------------------------------
        /// @brief TRiangular Matrix Matrix product (Level 2 BLAS)
        ///
        /// @tparam T Matrix element type.
        template<typename T>
        void TRMM(const char* side  , const char* uplo,
                  const char* transA, const char* diag,
                  const int   m     , const int   n   , const T& a,
                  const T*    A     , const int   ldA ,
                        T*    B     , const int   ldB);

        /// @brief TRiangular Matrix Matrix product specialization for double.
        template<>
        void TRMM<double>(const char*   side  , const char* uplo,
                          const char*   transA, const char* diag,
                          const int     m     , const int   n   , const double& a,
                          const double* A     , const int   ldA ,
                                double* B     , const int   ldB);

        //--------------------------------------------------------------------------
        /// @brief GEneral matrix QR Factorization (LAPACK)
        ///
        /// @tparam T Matrix element type.
        template<typename T>
        void GEQRF(const int m    , const int  n   ,
                         T*  A    , const int  ld  ,
                         T*  tau  ,       T*   work,
                   const int lwork,       int& info);

        /// @brief GEneral matrix QR Factorization specialization for double.
        template<>
        void GEQRF<double>(const int     m    , const int     n   ,
                                 double* A    , const int     ld  ,
                                 double* tau  ,       double* work,
                           const int     lwork,       int&    info);

        //--------------------------------------------------------------------------
        /// @brief ORthogonal matrix Generator from QR factorization (LAPACK).
        ///
        /// @tparam T Matrix element type.
        template<typename T>
        void ORGQR(const int m   , const int n    , const int  k  ,
                         T*  A   , const int ld   , const T*   tau,
                         T*  work, const int lwork,       int& info);

        /// @brief
        ///    ORthogonal matrix Generator from QR factorization
        ///    specialization for double.
        template<>
        void ORGQR<double>(const int     m   , const int n    , const int     k  ,
                                 double* A   , const int ld   , const double* tau,
                                 double* work, const int lwork,       int&    info);

        //--------------------------------------------------------------------------
        /// @brief GEneral matrix TRiangular Factorization (LAPACK).
        ///
        /// @tparam T Matrix element type.
        template<typename T>
        void GETRF(const int m, const int n, T* A,
                   const int ld, int* ipiv, int& info);

        /// @brief
        ///    GEneral matrix TRiangular Factorization specialization for double.
        template<>
        void GETRF<double>(const int m, const int n , double* A,
                           const int ld, int* ipiv, int& info);

        //--------------------------------------------------------------------------
        /// @brief GEneral matrix TRiangular Inversion (LAPACK).
        ///
        /// @tparam T Matrix element type.
        template<typename T>
        void GETRI(const int  n   , T* A   , const int ld,
                   const int* ipiv, T* work, int lwork, int& info);

        /// @brief GEneral matrix TRiangular Inversion specialization for double.
        template<>
        void GETRI(const int  n   , double* A   , const int ld,
                   const int* ipiv, double* work, int lwork, int& info);
    } // namespace BLAS_LAPACK
} // namespace Opm

#endif // OPENRS_BLAS_LAPACK_HEADER
