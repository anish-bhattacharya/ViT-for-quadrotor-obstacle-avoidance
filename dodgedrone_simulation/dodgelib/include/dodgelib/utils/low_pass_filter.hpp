#pragma once

#include "dodgelib/math/types.hpp"

namespace agi {


template<int rows>
class LowPassFilter {
 public:
  LowPassFilter(const Vector<rows>& cutoff_frequency,
                const Vector<rows>& sampling_frequency,
                const Vector<rows>& initial_value);

  LowPassFilter(const Scalar cutoff_frequency, const Scalar sampling_frequency,
                const Scalar initial_value)
    : LowPassFilter(Vector<rows>::Constant(cutoff_frequency),
                    Vector<rows>::Constant(sampling_frequency),
                    Vector<rows>::Constant(initial_value)) {}

  LowPassFilter() : LowPassFilter(NAN, NAN, NAN) {}

  ~LowPassFilter() = default;

  Vector<rows> add(const Vector<rows>& sample);

  Vector<rows> operator()() const { return output_.col(0); }
  Scalar operator()(const int i) const { return output_(i, 0); }

  Vector<rows> derivative() const {
    return sampling_frequency_.array() *
           (output_.col(0) - output_.col(1)).array();
  }
  Scalar derivative(const int i) const {
    return sampling_frequency_(i) * (output_(i, 0) - output_(i, 1));
  }

  bool valid() const {
    return sampling_frequency_.allFinite() && denumerator_.allFinite() &&
           numerator_.allFinite() && input_.allFinite() && output_.allFinite();
  };

 private:
  static Array<rows, 2> init_den(const Vector<rows>& fc,
                                 const Vector<rows>& fs);
  static Array<rows, 2> init_num(const Vector<rows>& fc,
                                 const Vector<rows>& fs);
  const Vector<rows> sampling_frequency_;
  const Array<rows, 2> denumerator_;
  const Array<rows, 2> numerator_;
  Array<rows, 2> input_;
  Array<rows, 2> output_;
};

template<int rows>
Array<rows, 2> LowPassFilter<rows>::init_den(const Vector<rows>& fc,
                                             const Vector<rows>& fs) {
  const ArrayVector<rows> K = (M_PI * fc.array() / fs.array()).tan();
  const ArrayVector<rows> poly = K * K + sqrt(2.0) * K + 1.0;

  Array<rows, 2> denumerator = Array<rows, 2>::Zero();
  denumerator.col(0) = 2.0 * (K * K - 1.0) / poly;
  denumerator.col(1) = (K * K - sqrt(2.0) * K + 1.0) / poly;

  return denumerator;
}

template<int rows>
Array<rows, 2> LowPassFilter<rows>::init_num(const Vector<rows>& fc,
                                             const Vector<rows>& fs) {
  const ArrayVector<rows> K = (M_PI * fc.array() / fs.array()).tan();
  const ArrayVector<rows> poly = K * K + sqrt(2.0) * K + 1.0;

  Array<rows, 2> numerator = Array<rows, 2>::Zero();
  numerator.col(0) = K * K / poly;
  numerator.col(1) = 2.0 * numerator.col(0);

  return numerator;
}


template<int rows>
LowPassFilter<rows>::LowPassFilter(const Vector<rows>& cutoff_frequency,
                                   const Vector<rows>& sampling_frequency,
                                   const Vector<rows>& initial_value)
  : sampling_frequency_(sampling_frequency),
    denumerator_(LowPassFilter::init_den(cutoff_frequency, sampling_frequency)),
    numerator_(LowPassFilter::init_num(cutoff_frequency, sampling_frequency)),
    input_(initial_value.replicate(1, 2)),
    output_(initial_value.replicate(1, 2)) {}

template<int rows>
Vector<rows> LowPassFilter<rows>::add(const Vector<rows>& sample) {
  const ArrayVector<rows> x2 = input_.col(1);
  input_.col(1) = input_.col(0);
  input_.col(0) = sample;
  const Vector<rows> out =
    numerator_.col(0) * x2 +
    (numerator_ * input_ - denumerator_ * output_).rowwise().sum();
  output_.col(1) = output_.col(0);
  output_.col(0) = out;

  return out;
}

}  // namespace agi
