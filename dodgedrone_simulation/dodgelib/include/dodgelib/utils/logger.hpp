#pragma once

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "dodgelib/math/types.hpp"

namespace agi {

#ifndef DEBUG_LOG
namespace {
struct NoPrint {
  template<typename T>
  constexpr NoPrint operator<<(const T&) const noexcept {
    return NoPrint();
  }
  constexpr NoPrint operator<<(std::ostream& (*)(std::ostream&)) const  //
    noexcept {
    return NoPrint();
  }
};
}  // namespace
#endif

struct PublishLogContainer {
  Vector<> data;
  bool advertise;
};


class Logger {
 public:
  Logger(const std::string& name, const bool color = true);
  ~Logger();

  inline std::streamsize precision(const std::streamsize n);
  inline void scientific(const bool on = true);

  void info(const char* msg, ...) const;
  void warn(const char* msg, ...) const;
  void error(const char* msg, ...) const;
  void fatal(const char* msg, ...) const;

#ifdef DEBUG_LOG
  void debug(const char* msg, ...) const;
  std::ostream& debug() const;
  void debug(const std::function<void(void)>&& lambda) const;
#else
  inline constexpr void debug(const char*, ...) const noexcept {}
  inline constexpr NoPrint debug() const { return NoPrint(); }
  inline constexpr void debug(const std::function<void(void)>&&) const  //
    noexcept {}
#endif

  inline void addPublishingVariable(const std::string& var_name,
                                    const Vector<>& vec) const {
    publishing_variables_[name_ + var_name].advertise =
      false;  // Not only advertising
    publishing_variables_[name_ + var_name].data = vec;
  }

  inline void addPublishingVariable(const std::string& var_name,
                                    Scalar val) const {
    addPublishingVariable(var_name, Vector<1>(val));
  }

  inline void advertisePublishingVariable(const std::string& var_name) const {
    publishing_variables_[name_ + var_name].advertise =
      true;  // Only advertising
  }

  inline void erasePublishingVariable(const std::string var_name) {
    publishing_variables_.erase(name_ + var_name);
  }

  template<typename T>
  std::ostream& operator<<(const T& printable) const;

  inline const std::string& name() const { return formatted_name_; }

  static constexpr int MAX_CHARS = 256;

  static void for_each_instance(
    std::function<void(const std::string& name,
                       const PublishLogContainer& container)>
      function) {
    for (auto& pub_var : publishing_variables_) {
      function(pub_var.first, pub_var.second);
    }
  }

 private:
  static constexpr int DEFAULT_PRECISION = 3;
  static constexpr int NAME_PADDING = 15;
  static constexpr char RESET[] = "\033[0m";
  static constexpr char RED[] = "\033[31m";
  static constexpr char YELLOW[] = "\033[33m";
  static constexpr char INFO[] = "Info:    ";
  static constexpr char WARN[] = "Warning: ";
  static constexpr char ERROR[] = "Error:   ";
  static constexpr char FATAL[] = "Fatal:   ";
  static constexpr char DEBUGPREFIX[] = "Debug:   ";

  const std::string name_;
  std::string formatted_name_;
  const bool colored_;
  static std::unordered_map<std::string, PublishLogContainer>
    publishing_variables_;  // Map from variable name to a struct containing
                            // info about logging data
};

template<typename T>
std::ostream& Logger::operator<<(const T& printable) const {
  return std::cout << formatted_name_ << printable;
}

}  // namespace agi
