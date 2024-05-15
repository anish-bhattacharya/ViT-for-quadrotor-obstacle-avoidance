#pragma once

#include <memory>

#include "dodgelib/utils/logger.hpp"

namespace agi {

template<typename Derived>
class Module {
 public:
  Module(const std::string& name) : logger_(name) {}
  virtual ~Module() {}

  inline const std::string& name() const { return logger_.name(); }
  virtual void logTiming() const {};

 protected:
  Logger logger_;
};

}  // namespace agi
