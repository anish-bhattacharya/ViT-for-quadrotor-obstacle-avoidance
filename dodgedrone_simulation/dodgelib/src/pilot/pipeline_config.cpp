#include "dodgelib/pilot/pipeline_config.hpp"

#include <ostream>

#include "dodgelib/base/parameter_base.hpp"

namespace agi {

bool checkFile(const fs::path& directory, fs::path* const filename) {
  if (filename == nullptr) return false;

  if (filename->empty()) return false;

  if (fs::is_directory(*filename)) return false;

  if (filename->is_absolute()) {
    if (fs::exists(*filename)) return true;
  }

  if (fs::exists(directory / *filename)) {
    *filename = directory / *filename;
    return true;
  }
  return false;
}

bool ModuleConfig::loadIfUndefined(const Yaml& yaml) {
  if (type.empty()) yaml["type"].getIfDefined(type);

  if (file.empty()) yaml["file"].getIfDefined(file);

  return !type.empty() && !file.empty();
}

std::ostream& operator<<(std::ostream& os, const ModuleConfig& config) {
  return os << "Type: " << config.type << "\nFile: " << config.file << '\n';
}

void PipelineConfig::load(const Yaml& yaml, const std::string& directory) {
  estimator_cfg.loadIfUndefined(yaml["estimator"]);
  checkFile(directory, &estimator_cfg.file);

  sampler_cfg.loadIfUndefined(yaml["sampler"]);
  checkFile(directory, &sampler_cfg.file);

  if (!outer_controller_cfg.loadIfUndefined(yaml["outer_controller"]))
    outer_controller_cfg.loadIfUndefined(yaml["controller"]);
  checkFile(directory, &outer_controller_cfg.file);

  inner_controller_cfg.loadIfUndefined(yaml["inner_controller"]);
  checkFile(directory, &inner_controller_cfg.file);

  bridge_cfg.loadIfUndefined(yaml["bridge"]);
  checkFile(directory, &bridge_cfg.file);
}

std::ostream& operator<<(std::ostream& os, const PipelineConfig& config) {
  os << "Estimator:\n" << config.estimator_cfg;
  os << "Sampler:\n" << config.sampler_cfg;
  os << "Outer Controller:\n" << config.outer_controller_cfg;
  os << "Inner Controller:\n" << config.inner_controller_cfg;
  os << "Bridge:\n" << config.bridge_cfg;
  return os;
}

}  // namespace agi
