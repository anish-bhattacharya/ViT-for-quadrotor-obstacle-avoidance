#pragma once

#include <ostream>

#include "dodgelib/utils/filesystem.hpp"
#include "dodgelib/utils/yaml.hpp"

namespace agi {

bool checkFile(const fs::path& directory, fs::path* const filename);

struct ModuleConfig {
  std::string type;
  fs::path file;

  bool loadIfUndefined(const Yaml& yaml);

  friend std::ostream& operator<<(std::ostream& os, const ModuleConfig& config);
};

struct PipelineConfig {
  ModuleConfig estimator_cfg;
  ModuleConfig sampler_cfg;
  ModuleConfig outer_controller_cfg;
  ModuleConfig inner_controller_cfg;
  ModuleConfig bridge_cfg;

  void load(const Yaml& yaml, const std::string& directory);

  friend std::ostream& operator<<(std::ostream& os,
                                  const PipelineConfig& config);
};


}  // namespace agi
