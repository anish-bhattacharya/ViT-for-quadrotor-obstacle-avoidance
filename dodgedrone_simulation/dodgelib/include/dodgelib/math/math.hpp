#pragma once

#include <dodgelib/math/types.hpp>

namespace agi {

Matrix<3, 3> skew(const Vector<3>& v);

Matrix<4, 4> Q_left(const Quaternion& q);

Matrix<4, 4> Q_right(const Quaternion& q);

Matrix<4, 3> qFromQeJacobian(const Quaternion& q);

Matrix<4, 4> qConjugateJacobian();

Matrix<3, 3> qeRotJacobian(const Quaternion& q, const Matrix<3, 1>& t);

Matrix<3, 3> qeInvRotJacobian(const Quaternion& q, const Matrix<3, 1>& t);

void matrixToTripletList(const SparseMatrix& matrix,
                         std::vector<SparseTriplet>* const list,
                         const int row_offset = 0, const int col_offset = 0);

void matrixToTripletList(const Matrix<>& matrix,
                         std::vector<SparseTriplet>* const list,
                         const int row_offset = 0, const int col_offset = 0);

void insert(const SparseMatrix& from, SparseMatrix* const into,
            const int row_offset = 0, const int col_offset = 0);

void insert(const Matrix<>& from, SparseMatrix* const into,
            const int row_offset = 0, const int col_offset = 0);

void insert(const Matrix<>& from, Matrix<>* const into,
            const int row_offset = 0, const int col_offset = 0);

Vector<> clip(const Vector<>& v, const Vector<>& bound);

inline constexpr Scalar toRad(const Scalar angle_deg) {
  return angle_deg * M_PI / 180.0;
}
inline constexpr Scalar toDeg(const Scalar angle_deg) {
  return angle_deg / M_PI * 180.0;
}

}  // namespace agi
