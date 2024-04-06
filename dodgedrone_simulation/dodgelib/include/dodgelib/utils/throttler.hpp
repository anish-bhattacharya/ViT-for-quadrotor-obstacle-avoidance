#pragma once

#include <chrono>
#include <functional>

namespace agi {

template<typename T>
class Throttler {
 public:
  Throttler(T& obj, const Scalar period) : obj(obj), period(period) {}

  template<class F, class... Args>
  void operator()(F&& f, const Args&... args) {
    const std::chrono::steady_clock::time_point t_now =
      std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::microseconds>(t_now - t_last)
          .count() > 1e6 * period) {
      std::invoke(f, obj, args...);
      t_last = t_now;
    }
  }

 private:
  T& obj;
  const Scalar period;
  std::chrono::steady_clock::time_point t_last;
};

}  // namespace agi
