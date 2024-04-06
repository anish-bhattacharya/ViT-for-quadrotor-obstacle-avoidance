#include "dodgelib/reference/trajectory_reference/polynomial.hpp"

#include <iostream>

namespace agi {

/* Create exponents as in
/  derivative_order = 0: [ 0 1 2 3 4 ... N]
/  derivative_order = 1: [ 0 0 1 2 3 ... N-1]
/  derivative_order = 2: [ 0 0 0 1 2 ... N-2]
/  ...
*/
Array<> createExponents(const int order) {
  const int N = order + 1;
  Array<> P = Array<>::Zero(N, N);
  P.col(0) = ArrayVector<>::LinSpaced(N, 0, order);

  for (int i = 1; i < N; ++i)
    P.col(i) << ArrayVector<>::Zero(i), P.col(0).head(N - i);

  return P;
}

/* Create derivative coefficients as in
/  derivative_order = 0: [ 1 1 1 1 1 ...]
/  derivative_order = 1: [ 0 1 2 3 4 ...]
/  derivative_order = 2: [ 0 0 2 6 12 ...]
/  ...
*/
Array<> createAlpha(const int order) {
  const int N = order + 1;
  const ArrayVector<> p = ArrayVector<>::LinSpaced(N, 0, order);

  Array<> alpha = Array<>::Ones(N, N);

  for (int i = 1; i < N; ++i) {
    alpha.block(i - 1, i, N + 1 - i, N - i) *=
      p.head(N + 1 - i).replicate(1, N - i);
  }

  return alpha;
}

template<typename Solver>
Polynomial<Solver>::Polynomial()
  : order_(11),
    continuity_(-1),
    c_(Vector<>::Constant(12, NAN)),
    exponents_(createExponents(11)),
    alpha_(createAlpha(11)),
    weights_(Vector<4>(0, 0, 0, 1)) {}

template<typename Solver>
Polynomial<Solver>::Polynomial(const int order, const Vector<>& weights,
                               const int continuity)
  : order_(order),
    continuity_(continuity),
    c_(Vector<>::Constant(order + 1, NAN)),
    exponents_(createExponents(order)),
    alpha_(createAlpha(order)),
    weights_(weights) {}

template<>
Polynomial<void>::Polynomial()
  : order_(5),
    continuity_(3),
    c_(Vector<>::Constant(6, NAN)),
    exponents_(createExponents(5)),
    alpha_(createAlpha(5)),
    weights_(Vector<3>(0, 0, 1)),
    b_(Vector<>::Constant(6, NAN)) {}

template<typename Solver>
bool Polynomial<Solver>::scale(const Scalar start_time, const Scalar duration) {
  bool success = false;
  if (std::isfinite(start_time)) {
    t_offset_ = start_time;
    success = true;
  }

  if (std::isfinite(duration)) {
    if (duration > 0.0) {
      t_scale_ = 1.0 / duration;
    } else {
      success = false;
    }
  }

  return success;
}

template<typename Solver>
int Polynomial<Solver>::addConstraint(const Scalar time,
                                      const Vector<>& constraint) {
  const ArrayVector<> constraints =
    continuity_ > 0
      ? constraint.head(std::min((int)constraint.size(), continuity_ + 1))
      : constraint;
  const int n = constraints.isFinite().cast<int>().sum();
  if (n < 1) return 0;

  const Scalar tau = tauFromTime(time);

  const int n_old = A_.allFinite() ? A_.rows() : 0;

  if (n_old > 0) {
    const Matrix<> A_old = A_;
    const Vector<> b_old = b_;

    A_.resize(n_old + n, size());
    b_.resize(n_old + n);

    A_.topRows(n_old) = A_old;
    b_.head(n_old) = b_old;
  } else {
    A_.resize(n, size());
    b_.resize(n);
  }

  int idx = n_old;
  for (int i = 0; i < constraints.size(); ++i) {
    if (std::isfinite(constraints(i))) {
      A_.row(idx) =
        std::pow(t_scale_, i) * alpha_.col(i) * tauVector(tau, i).array();
      b_(idx) = constraints(i);
      ++idx;
    }
  }

  return n;
}

template<>
int Polynomial<void>::addConstraint(const Scalar time,
                                    const Vector<>& constraint) {
  if (constraint.rows() < 3) {
    std::cout << "Closed form polynomial constraints must be of size 3!"
              << std::endl;
    return 0;
  }

  if (std::abs(time - t_offset_) < 1e-3) {
    b_.head<3>() = constraint.head<3>();
  } else if (std::abs(time - t_offset_ - 1.0 / t_scale_) < 1e-3) {
    b_.tail<3>() = constraint.head<3>();
  } else {
    std::cout
      << "Closed form polynomial constraints must be at start or end time!"
      << std::endl;
    return 0;
  }

  return 3;
}

template<typename Solver>
Scalar Polynomial<Solver>::tauFromTime(const Scalar t) const {
  return t_scale_ * (t - t_offset_);
}

template<typename Solver>
Vector<> Polynomial<Solver>::tauVector(const Scalar tau,
                                       const int order) const {
  return ArrayVector<>::Constant(size(), tau).pow(exponents_.col(order));
}

template<typename Solver>
Vector<> Polynomial<Solver>::tauVectorFromTime(const Scalar time,
                                               const int order) const {
  return tauVector(tauFromTime(time), order);
}

template<typename Solver>
Matrix<> Polynomial<Solver>::polyMatrix(const Scalar tau,
                                        const int order) const {
  const ArrayVector<> tau_vec = tauVector(tau);

  const int N = size();
  Array<> M = alpha_.block(0, 0, N, order);

  for (int i = 0; i < order; ++i)
    M.block(i, i, N - i, 1) *= tau_vec.head(N - i);

  return M.transpose();
}

template<typename Solver>
Matrix<> Polynomial<Solver>::polyMatrixFromTime(const Scalar time,
                                                const int order) const {
  return polyMatrix(tauFromTime(time), order);
}

template<typename Solver>
void Polynomial<Solver>::reset() {
  H_ = Matrix<>::Constant(1, order_, NAN);
  A_ = Matrix<>::Constant(1, order_, NAN);
  b_ = Vector<>::Constant(1, NAN);
}

template<typename Solver>
Matrix<> Polynomial<Solver>::createH(const Vector<>& weights) const {
  const int nW = std::min((int)weights.rows(), order_);
  const Matrix<> W = weights.head(nW).asDiagonal();

  const Vector<> time_scales = ArrayVector<>::Constant(nW, t_scale_)
                                 .pow(ArrayVector<>::LinSpaced(nW, 2, 2 * nW));

  const int n = size();
  Array<> H = Array<>::Zero(n, n);

  for (int i = 0; i < nW; ++i) {
    if (weights(i) <= 0.0) continue;
    const Vector<> alpha = alpha_.col(i + 1);
    const Array<> denom = exponents_.col(i + 1).replicate(1, n);
    const Array<> alpha_outer = alpha * alpha.transpose();
    const Array<> denom_outer = (denom + denom.transpose() + 1).max(1.0);
    H += time_scales(i) * weights(i) * alpha_outer / denom_outer;
  }
  return H;
}

template<typename Solver>
bool Polynomial<Solver>::solve() {
  if (A_.rows() < 2) return false;  // Cant solve without constraints.

  H_ = createH(weights_);
  return solve(H_, A_, b_);
}

template<typename Solver>
bool Polynomial<Solver>::solve(const Matrix<>& H, const Matrix<>& A,
                               const Vector<>& b) {
  if (H.cols() != size()) return false;
  if (A.rows() != b.size() || A.cols() != size()) return false;

  const int n = H.rows();
  const int m = A.rows();

  const Matrix<> S =
    (Matrix<>(n + m, n + m) << 2.0 * H, A.transpose(), A, Matrix<>::Zero(m, m))
      .finished();
  const Vector<> s = (Vector<>(m + n) << Vector<>::Zero(n), b).finished();

  Solver solver(S);
  const Vector<> x = solver.solve(s);
  c_ = x.head(n);

  return true;
}

template<>
bool Polynomial<void>::solve() {
  if (!b_.allFinite()) return false;

  // Hardcoded Matrix inversion for the end state constraints.
  // This is the inverse of the right half of the coeffcient matrix
  // [ 1  1  1  1  1  1 ]
  // [ 0  1  1  3  4  5 ]
  // [ 0  0  2  6  12 20]
  // also known as alpha within this implementation.
  static const Matrix<3, 3> A_inv =
    (Matrix<3, 3>() << 10, -4, 0.5, -15, 7, -1, 6, -3, 0.5).finished();

  const ArrayVector<3> t_vec = ArrayVector<3>::Constant(1.0 / t_scale_)
                                 .pow(ArrayVector<3>::LinSpaced(0, 2));

  c_.head<3>() = t_vec * b_.array().head<3>();
  c_(2) *= 0.5;
  c_.tail<3>() = t_vec * b_.array().tail<3>();

  const Matrix<3, 3> alpha_012 = alpha_.topLeftCorner(3, 3).transpose();
  const Vector<3> b_delta = c_.tail<3>() - alpha_012 * c_.head<3>();
  c_.tail<3>() = A_inv * b_delta;

  return true;
}


template<typename Solver>
bool Polynomial<Solver>::eval(const Scalar time, Ref<Vector<>> state) const {
  const int n = state.rows();
  const Vector<> time_scales = ArrayVector<>::Constant(n, t_scale_)
                                 .pow(ArrayVector<>::LinSpaced(n, 0, n - 1));
  state = time_scales.asDiagonal() * polyMatrixFromTime(time, n) * c_;

  return true;
}

template<typename Solver>
Scalar Polynomial<Solver>::operator()(const Scalar time,
                                      const int order) const {
  return std::pow(t_scale_, order) * c_.transpose() *
         (tauVectorFromTime(time, order).array() * alpha_.col(order)).matrix();
}

template class Polynomial<Eigen::HouseholderQR<Matrix<>>>;
template class Polynomial<void>;

}  // namespace agi
