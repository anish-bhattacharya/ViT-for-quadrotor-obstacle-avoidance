#pragma once

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include "dodgelib/math/types.hpp"
#include "dodgelib/utils/filesystem.hpp"

namespace agi {
namespace {

struct YamlException : public std::exception {
  YamlException() = default;
  YamlException(const std::string& msg) : msg(msg) {}
  YamlException(const std::string& key, const std::string& msg)
    : msg("Key \'" + key + "\':\n" + msg + '\n') {}

  YamlException(const std::string& key, const std::string& msg,
                const std::string_view& doc, const size_t pos)
    : msg("Key \'" + key + "\' in line " +
          std::to_string(getLineNumber(doc.substr(0, pos))) + ":\n" + msg +
          '\n') {}

  YamlException(const std::string& msg, const std::string_view& doc,
                const size_t pos)
    : msg([&] {
        const size_t ln = getLineNumber(doc.substr(0, pos));
        return std::string("In line " + std::to_string(ln) + ":\n" + msg +
                           "\nLine: \'" + getLine(doc, ln) + "\'\n");
      }()) {}

  [[nodiscard]] const char* what() const noexcept override {
    return msg.c_str();
  }

  static size_t getLineNumber(const std::string_view& doc) {
    return std::count(doc.begin(), doc.end(), '\n');
  }

  static std::string getLine(const std::string_view& doc, const size_t line) {
    size_t start = 0;
    size_t end = doc.find_first_of('\n');
    if (line) {
      for (size_t i = 0; i < line; ++i) {
        start = end + 1;
        end = doc.find_first_of('\n', start);
      }
    }
    return std::string(doc.substr(start, end - start));
  }

  const std::string msg{"YAML Exception"};
};
}  // namespace

class Yaml {
 public:
  Yaml() = default;
  Yaml(const Yaml&) = default;
  ~Yaml() = default;
  Yaml(const std::string& keyspace, const std::string& key)
    : keyspace_(keyspace), key_(key) {}

  // Delegating Constructors
  explicit Yaml(const fs::path& filename) : Yaml(loadFile(filename)) {}
  explicit Yaml(const std::string& doc) : Yaml(std::string_view(doc)) {}
  explicit Yaml(const std::string_view& doc)
    : Yaml([](const std::string_view& doc) {
        size_t indent = 0;
        size_t pos = 0;
        return Yaml("", "", doc, pos, doc.size(), indent);
      }(doc)) {}

  template<typename T>
  T as() const {
    if constexpr (std::is_same_v<T, int>)
      return as_int();
    else if constexpr (std::is_same_v<T, unsigned int>)
      return as_uint();
    else if constexpr (std::is_same_v<T, float>)
      return as_float();
    else if constexpr (std::is_same_v<T, double>)
      return as_double();
    else if constexpr (std::is_same_v<T, std::complex<double>>)
      return as_complex<double>();
    else if constexpr (std::is_same_v<T, std::complex<float>>)
      return as_complex<float>();
    else if constexpr (std::is_same_v<T, bool>)
      return as_bool();
    else
      return as_string();
  }

  template<typename T, typename std::enable_if<
                         !std::is_base_of<Eigen::DenseBase<T>, T>::value &&
                           !std::is_base_of<Eigen::QuaternionBase<T>, T>::value,
                         bool>::type = false>
  inline friend void operator>>(const Yaml& yaml, T& value) {
    value = yaml.as<T>();
  }

  template<typename Derived,
           typename std::enable_if<
             std::is_base_of<Eigen::DenseBase<Derived>, Derived>::value,
             bool>::type = true>
  inline friend void operator>>(const Yaml& yaml, Derived& matrix) {
    yaml.as_matrix(matrix);
  }

  template<typename Derived,
           typename std::enable_if<
             std::is_base_of<Eigen::QuaternionBase<Derived>, Derived>::value,
             bool>::type = true>
  inline friend void operator>>(const Yaml& yaml, Derived& matrix) {
    yaml.as_quaternion(matrix);
  }

  const Yaml operator[](const std::string& s) const {
    try {
      return map_.at(s);
    } catch (std::exception& e) {
      return Yaml(getFullKeyName(), s);
    }
  }

  const Yaml operator[](const int i) const {
    const size_t idx = i >= 0 ? (size_t)i : (size_t)(i % (int)seq_.size());
    if (seq_.empty()) return Yaml();
    if (idx < seq_.size())
      return seq_[idx];
    else
      return Yaml(getFullKeyName(), std::to_string(i));
  }

  template<typename T>
  bool getIfDefined(T& val) const {
    if (isNull()) return false;
    if constexpr (std::is_base_of_v<Eigen::DenseBase<T>, T>)
      as_matrix(val);
    else if constexpr (std::is_base_of_v<Eigen::QuaternionBase<T>, T>)
      as_quaternion(val);
    else
      val = as<T>();
    return true;
  }

  [[nodiscard]] inline bool isValid() const {
    return !raw_.empty() ^ !map_.empty() ^ !seq_.empty();
  }
  [[nodiscard]] inline bool isDefined() const {
    return !raw_.empty() || !map_.empty() || !seq_.empty();
  }
  [[nodiscard]] inline bool isNull() const {
    return raw_.empty() && map_.empty() && seq_.empty();
  }
  [[nodiscard]] inline bool isValue() const { return !raw_.empty(); }
  [[nodiscard]] inline bool isNode() const { return !map_.empty(); }
  [[nodiscard]] inline bool isSequence() const { return !seq_.empty(); }

  [[nodiscard]] inline size_t size() const {
    if (isNull())
      return 0;
    else if (isSequence())
      return seq_.size();
    else if (isNode())
      return map_.size();
    else
      return 1;
  }

  friend std::ostream& operator<<(std::ostream& os, const Yaml& yaml) {
    yaml.print(os, 0);
    return os;
  }

  void print(std::ostream& os, const size_t indent) const {
    if (isValue()) {
      os << ' ' << raw_ << '\n';
      return;
    }
    os << '\n';
    for (const auto& node : map_) {
      os << std::string(indent, ' ') << node.first << ':';
      node.second.print(os, indent + 2);
    }
    for (const auto& node : seq_) {
      os << std::string(indent, ' ') << "- ";
      node.print(os, indent + 2);
    }
  }

  // Parsing Constructor
  Yaml(const std::string& keyspace, const std::string& key,
       const std::string_view& doc, size_t& pos, const size_t end,
       const size_t min_indent = 0, bool list = false)
    : keyspace_(keyspace), key_(key) {
    while (pos < end) {
      const size_t first_non_space = doc.find_first_not_of(" \t", pos);
      const size_t first = (list && doc[first_non_space] == '-')
                             ? doc.find_first_not_of(" \t", first_non_space + 1)
                             : first_non_space;
      list = false;
      const size_t line_break =
        std::min(doc.find_first_of('\n', first), std::string::npos - 1);
      const size_t line_end = std::min(doc.find_first_of("#\n", first), end);
      const size_t indent = first - pos;
      const size_t next = std::min(line_break + 1, end);

      if (first >= line_end) {
        pos = next;
        continue;
      }

      if (indent < min_indent) break;

      const std::string_view line = doc.substr(first, line_end - first);

      if (line_end > first) {
        const size_t indicator = line.find_first_of(":[{(");
        if (line[0] == '-' && line[1] == ' ') {
          const size_t list_indent = first_non_space - pos;
          seq_.emplace_back(getFullKeyName(), std::to_string(seq_.size()), doc,
                            pos, end, list_indent + 1, true);
        } else if (indicator == std::string::npos) {
          const size_t value_end = line.find_last_not_of(' ');
          if (value_end < std::string::npos) {
            raw_ = line.substr(0, 1 + value_end);
          } else {
            raw_ = line;
          }
          pos = next;
        } else if (line[indicator] == ':') {
          const size_t key_end = line.find_first_of(" \t\n:{}[]()");
          const size_t key_remainder = line.find_first_not_of(" \t", key_end);
          if (indicator < 1 || key_remainder < indicator)
            throw YamlException("Could not parse line! No valid key!", doc,
                                pos);
          const std::string sub_key =
            (std::string)line.substr(0, std::min(indicator, key_end));
          const size_t value_start =
            line.find_first_not_of(" \t", indicator + 1);
          if (value_start < line_end) {
            pos = first + value_start;
            map_[sub_key] =
              Yaml(getFullKeyName(), sub_key, doc, pos, line_end, 0);
          } else {
            pos = next;
            map_[sub_key] =
              Yaml(getFullKeyName(), sub_key, doc, pos, end, indent + 1);
          }
          pos = std::max(pos, next);
        } else if (line[indicator] == '[' || line[indicator] == '{' ||
                   line[indicator] == '(') {
          pos = doc.find_first_not_of(" \t", first + indicator + 1);
          size_t r = pos;
          int level = 1;
          while (r < end) {
            const char c = doc[r];
            if (c == ':') break;
            if (c == '[' || c == '{' || c == '(') {
              ++level;
              ++r;
              continue;
            } else if (c == ']' || c == '}' || c == ')') {
              if (level == 1) {
                seq_.emplace_back(getFullKeyName(), std::to_string(seq_.size()),
                                  doc, pos, r, 0);
              }
              --level;
              ++r;
              continue;
            } else if (c == ',' && level == 1) {
              seq_.emplace_back(getFullKeyName(), std::to_string(seq_.size()),
                                doc, pos, r, 0);
              pos = ++r;
              continue;
            }
            ++r;
          }
          if (level != 0) throw YamlException("List not closed!", doc, pos);
          pos = doc.find_first_of('\n', pos);
          if (pos < end) ++pos;
        } else {
          pos = next;
        }
      } else {
        pos = next;
      }
    }

    // Clean out sequences of size 1.
    if (seq_.size() == 1 && map_.empty()) {
      raw_ = seq_[0].raw_;
      map_ = seq_[0].map_;
      seq_ = seq_[0].seq_;
    }
  }

 private:
  // Interpreters
  [[nodiscard]] int as_int() const {
    try {
      return std::stoi(raw_);
    } catch (std::exception& e) {
      throw YamlException(getFullKeyName(), "Is not a valid int: " + raw_);
    };
  }

  [[nodiscard]] int as_uint() const {
    try {
      return std::stoul(raw_);
    } catch (std::exception& e) {
      throw YamlException(getFullKeyName(),
                          "Is not a valid unsigned integer: " + raw_);
    };
  }

  [[nodiscard]] float as_float() const {
    try {
      return std::stof(raw_);
    } catch (std::exception& e) {
      throw YamlException(getFullKeyName(), "Is not a valid float: " + raw_);
    };
  }

  [[nodiscard]] double as_double() const {
    try {
      return std::stod(raw_);
    } catch (std::exception& e) {
      throw YamlException(getFullKeyName(), "Is not a valid double: " + raw_);
    };
  }

  [[nodiscard]] bool as_bool() const {
    // Fast catch for obvious cases such as empty or single char.
    if (raw_.empty())
      throw YamlException(getFullKeyName(), "Can't cast empty node to bool!");
    if (raw_.size() == 1) {
      if (raw_[0] == '0' || raw_[0] == 'f' || raw_[0] == 'F') return false;
      if (raw_[0] == '1' || raw_[0] == 't' || raw_[0] == 'T') return true;
      throw YamlException(getFullKeyName(), "Is not a valid bool: " + raw_);
    }

    static constexpr std::array<char, 4> strue{'t', 'r', 'u', 'e'};
    static constexpr std::array<char, 5> sfalse{'f', 'a', 'l', 's', 'e'};

    size_t true_hit = 0;
    size_t false_hit = 0;
    static constexpr size_t case_dist = 'a' - 'A';
    for (const char c : raw_) {
      if (c == strue[true_hit] || (char)(c + case_dist) == strue[true_hit]) {
        ++true_hit;
        if (true_hit >= strue.size()) return true;
      } else {
        true_hit = 0;
      }
      if (c == sfalse[false_hit] ||
          (char)(c + case_dist) == sfalse[false_hit]) {
        ++false_hit;
        if (false_hit >= sfalse.size()) return false;
      } else {
        false_hit = 0;
      }
    }
    throw YamlException(getFullKeyName(), "Is not a valid bool: " + raw_);
  }

  [[nodiscard]] std::string as_string() const {
    const size_t start = raw_.find_first_not_of(" \'\"");
    const size_t end = raw_.find_last_not_of(" \'\"");
    if (start < raw_.size() && end < raw_.size())
      return raw_.substr(start, end - start + 1);
    else
      throw YamlException(getFullKeyName(),
                          std::string("Is not a valid string: ") + raw_);
    return "";
  }

  template<typename T>
  [[nodiscard]] std::complex<T> as_complex() const {
    if (seq_.size() != 2)
      throw YamlException(getFullKeyName(), "Is not a valid complex number!");
    try {
      return std::complex<T>(seq_[0].as<T>(), seq_[1].as<T>());
    } catch (std::exception& e) {
      throw YamlException(
        getFullKeyName(),
        std::string("Can't parse to a complex number:\n") + e.what());
    };
    return std::complex<T>();
  }

  template<typename Derived,
           typename std::enable_if<
             std::is_base_of<Eigen::DenseBase<Derived>, Derived>::value,
             bool>::type = true>
  void as_matrix(Derived& matrix) const {
    if (isNull())
      throw YamlException(getFullKeyName(),
                          "Can't cast empty Yaml node to Eigen object.");
    const size_t n = seq_.size();
    try {
      if (isValue() && (matrix.rows() == 1 || matrix.cols() == 1)) {
        matrix.setConstant(as<typename Derived::Scalar>());
      } else if (matrix.size() == (int)n) {
        for (size_t i = 0; i < n; ++i)
          matrix((int)i) = seq_[i].as<typename Derived::Scalar>();
      } else {
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        if (rows != (int)n)
          throw YamlException(getFullKeyName(),
                              "dimensions wrong\n"
                              "Eigen object: " +
                                std::to_string(rows) + 'x' +
                                std::to_string(cols) + '\n' +
                                "Yaml object:  " + std::to_string(n) + 'x' +
                                std::to_string(seq_[0].size()));
        for (size_t i = 0; i < n; ++i) {
          const size_t m = seq_[i].size();
          if (cols != (int)m)
            throw YamlException(
              getFullKeyName(),
              "dimensions wrong\n"
              "Eigen object: " +
                std::to_string(rows) + 'x' + std::to_string(cols) + '\n' +
                "Yaml object:  " + std::to_string(n) + 'x' + std::to_string(m));
          for (size_t j = 0; j < m; ++j)
            matrix((int)i, (int)j) = seq_[i][j].as<typename Derived::Scalar>();
        }
      }
    } catch (std::exception& e) {
      throw YamlException(
        getFullKeyName(),
        std::string("Can't parse to Eigen object:\n") + e.what());
    }
  }

  template<typename Derived,
           typename std::enable_if<
             std::is_base_of<Eigen::QuaternionBase<Derived>, Derived>::value,
             bool>::type = true>
  void as_quaternion(Derived& quaternion) const {
    if (seq_.size() != 4)
      throw YamlException(getFullKeyName(),
                          "Is not a valid quaternion, because it has " +
                            std::to_string(seq_.size()) + " entries.");

    try {
      quaternion.w() = seq_[0].as<typename Derived::Scalar>();
      quaternion.x() = seq_[1].as<typename Derived::Scalar>();
      quaternion.y() = seq_[2].as<typename Derived::Scalar>();
      quaternion.z() = seq_[3].as<typename Derived::Scalar>();
    } catch (std::exception& e) {
      throw YamlException(
        getFullKeyName(),
        std::string("Can't parse into Eigen quaternion:\n") + e.what());
    }
  }

  // File helper
  static std::string loadFile(const fs::path& filename) {
    std::ifstream fs(filename.c_str());
    if (!fs.is_open())
      throw YamlException("Could not open file \'" + filename.string() + "\'");
    fs.seekg(0, std::ios::end);
    const size_t size = fs.tellg();
    if (size < 1)
      throw YamlException("Empty file \'" + filename.string() + "\'");
    std::string buffer((size_t)size, ' ');
    fs.seekg(0);
    fs.read(&buffer[0], size);
    return buffer;
  }

  [[nodiscard]] std::string getFullKeyName() const {
    return keyspace_.empty() ? key_ : keyspace_ + "/" + key_;
  }

  // Members
  std::string keyspace_;
  std::string key_;
  std::string raw_;
  std::map<std::string, Yaml> map_;
  std::vector<Yaml> seq_;
};


}  // namespace agi
