#pragma once

#include <dodgelib/utils/filesystem.hpp>
#include <exception>
#include <sstream>

#include "dodgelib/math/types.hpp"
#include "dodgelib/utils/filesystem.hpp"
#include "dodgelib/utils/yaml.hpp"

namespace agi {


struct ParameterException : public std::exception {
  ParameterException() = default;
  ParameterException(const std::string& msg)
    : msg(std::string("Dodgelib Parameter Exception: ") + msg) {}
  const char* what() const throw() { return msg.c_str(); }

  const std::string msg{"Dodgelib Parameter Exception"};
};

struct ParameterBase {
  virtual ~ParameterBase() = default;
  virtual bool load(const fs::path& filename);
  virtual bool load(const Yaml& yaml);

  virtual bool valid() const;
};

}  // namespace agi
