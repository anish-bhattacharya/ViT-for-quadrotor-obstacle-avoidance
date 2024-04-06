#pragma once

#include <algorithm>
#include <deque>

#include "dodgelib/math/types.hpp"

namespace agi {

template<typename T, int N>
class Median {
 public:
  Median(const T& initial_value) : median(initial_value) {
    *this += initial_value;
  }
  ~Median() = default;

  T add(const T& value) {
    while ((int)buffer.size() >= N) buffer.pop_front();
    buffer.push_back(value);

    sorted = buffer;
    std::sort(sorted.begin(), sorted.end());
    median = sorted.at(middle());
    return median;
  }

  inline T get() const { return median; }
  inline T operator()() const { return get(); }

  Median<T, N>& operator+=(const T& value) {
    add(value);
    return *this;
  }

 private:
  constexpr int middle() { return sorted.size() / 2; }

  T median;
  std::deque<T> buffer;
  std::deque<T> sorted;
};


}  // namespace agi
