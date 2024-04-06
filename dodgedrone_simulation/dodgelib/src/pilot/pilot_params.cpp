#include "dodgelib/pilot/pilot_params.hpp"

#include <unistd.h>

#include <filesystem>
#include <memory>

#include "dodgelib/base/parameter_base.hpp"
#include "dodgelib/bridge/debug_bridge.hpp"
#include "dodgelib/controller/geometric/controller_geo.hpp"
#include "dodgelib/controller/geometric/geo_params.hpp"
#include "dodgelib/estimator/feedthrough/feedthrough_estimator.hpp"
#include "dodgelib/estimator/feedthrough/feedthrough_params.hpp"
#include "dodgelib/pilot/pipeline_config.hpp"
#include "dodgelib/sampler/time_based/time_sampler.hpp"

namespace agi {


PilotParams::PilotParams(const fs::path& filename, const fs::path& directory,
                         const fs::path& quad_file)
  : directory_(directory), quad_file_(quad_file) {
  fs::path full_filename = filename;
  if (!checkFile(directory_, &full_filename))
    throw ParameterException("Pilot Config file not found!\n" +
                             filename.string() + "\n" + full_filename.string());

  std::cout << "Loading Pilot parameters from " << full_filename << std::endl;

  if (!ParameterBase::load(full_filename))
    throw ParameterException(full_filename);
}

bool PilotParams::load(const Yaml& yaml) {
  if (directory_.empty()) {
    static constexpr int PATH_LEN = 2048;
    char path_cstr[PATH_LEN];
    if (readlink("/proc/self/exe", path_cstr, PATH_LEN)) {
      const fs::path agidir(path_cstr);

      fs::path::iterator start_it = agidir.begin();
      fs::path::iterator end_it = std::lower_bound(
        start_it, agidir.end(), std::string("dodgedrone_simulator"));

      if (end_it == agidir.end()) {
        throw ParameterException(
          "No directory provided and dodgedrone_simulator directory not "
          "found!");
      } else {
        for (fs::path::iterator it = agidir.begin(); it != end_it; it++)
          directory_ /= *it;
        directory_ =
          directory_ / "dodgedrone_simulator" / "dodgelib" / "params";
      }
    }
  }

  pipeline_cfg_.load(yaml["pipeline"], directory_);

  // Quadrotor
  quad_file_ = [&] {
    fs::path quad_file = quad_file_;
    std::vector<fs::path> possible_paths{directory_, directory_ / "quads"};
    if (!quad_file.empty()) {
      if (quad_file.has_filename()) {
        if (quad_file.is_absolute()) {
          possible_paths.clear();
        }
      } else {
        if (quad_file.is_absolute()) {
          possible_paths = {quad_file, quad_file / "quads"};
          quad_file.clear();
        } else {
          throw ParameterException(
            "If quadrotor file path is manually set, it must be an absolute "
            "path.\n"
            "Specified path: " +
            quad_file.string());
        }
      }
    }

    fs::path quad_file_from_yaml = [&yaml] {
      std::string quad_file_from_yaml;
      yaml["quadrotor"].getIfDefined(quad_file_from_yaml);
      return quad_file_from_yaml;
    }();

    if (!quad_file_from_yaml.empty()) {
      if (!quad_file.empty())
        throw ParameterException("Quadrotor file is set manually an in YAML!");

      if (!quad_file_from_yaml.has_filename())
        throw ParameterException("Quadrotor file is no filename!");

      quad_file = quad_file_from_yaml;
      if (quad_file.is_absolute()) possible_paths.clear();
    }

    for (const fs::path& path : possible_paths) {
      if (checkFile(path, &quad_file)) return quad_file;
    }

    return quad_file;
  }();

  if (!quad_.load(quad_file_) || !quad_.valid())
    throw ParameterException("Could not load Quadrotor parameters from: " +
                             quad_file_.string());

  // Pilot Params
  dt_min_ = yaml["dt_min"].as<Scalar>();
  yaml["outerloop_divisor"].getIfDefined(outerloop_divisor_);
  dt_telemetry_ = yaml["dt_telemetry"].as<Scalar>();
  traj_type_ = yaml["traj_type"].as<std::string>();

  yaml["velocity_in_bodyframe"].getIfDefined(velocity_in_bodyframe_);
  yaml["takeoff_height"].getIfDefined(takeoff_heigth_);
  yaml["takeoff_threshold"].getIfDefined(takeoff_threshold_);
  yaml["start_land_speed"].getIfDefined(start_land_speed_);
  yaml["brake_deceleration"].getIfDefined(brake_deceleration_);
  yaml["go_to_pose_mean_vel"].getIfDefined(go_to_pose_mean_vel_);
  yaml["stop_after_feedthrough"].getIfDefined(stop_after_feedthrough_);
  yaml["feedthrough_timeout"].getIfDefined(feedthrough_timeout_);

  yaml["viz_sampler_dt"].getIfDefined(traj_viz_dt_);
  yaml["sphere_size"].getIfDefined(traj_viz_sphere_size_);
  yaml["pub_log_var"].getIfDefined(publish_log_var_);

  return valid();
}

bool PilotParams::valid() const { return quad_.valid(); }

bool PilotParams::createEstimator(std::shared_ptr<EstimatorBase>& estimator,
                                  const ModuleConfig& config) const {
  try {
    if (config.type == "Feedthrough") {
      std::shared_ptr<FeedthroughParameters> params =
        std::make_shared<FeedthroughParameters>();
      if (!config.file.empty() && !params->load(config.file))
        throw ParameterException();
      estimator = std::make_shared<FeedthroughEstimator>(params);
      return true;
    }
  } catch (const ParameterException& e) {
    throw ParameterException("Could not load estimator " + config.type +
                             " from parameter file \'" + config.file.string() +
                             "\':\n" + e.what());
  }

  return false;
}

bool PilotParams::createController(std::shared_ptr<ControllerBase>& controller,
                                   const ModuleConfig& config) const {
  try {
    if (config.type == "GEO") {
      std::shared_ptr<GeometricControllerParams> params =
        std::make_shared<GeometricControllerParams>();
      if (!config.file.empty() && !params->load(config.file))
        throw ParameterException();
      controller = std::make_shared<GeometricController>(quad_, params);
      return true;
    }
  } catch (const ParameterException& e) {
    throw ParameterException("Could not load controller " + config.type +
                             " from parameter file \'" + config.file.string() +
                             "\':\n" + e.what());
  }

  return false;
}

bool PilotParams::createBridge(std::shared_ptr<BridgeBase>& bridge,
                               const TimeFunction& time_function,
                               const ModuleConfig& config) const {
  try {
    if (config.type == "Debug") {
      bridge = std::make_shared<DebugBridge>("DebugBridge", time_function);
      return true;
    }
  } catch (const ParameterException& e) {
    throw ParameterException("Could not load bridge " + config.type +
                             " from parameter file \'" + config.file.string() +
                             "\':\n" + e.what());
  }

  return false;
}

bool PilotParams::createBridge(std::shared_ptr<BridgeBase>& bridge,
                               const TimeFunction& time_function) const {
  return createBridge(bridge, time_function, pipeline_cfg_.bridge_cfg);
}

bool PilotParams::createSampler(
  std::shared_ptr<SamplerBase>& sampler,
  const std::shared_ptr<ControllerBase>& controller,
  const ModuleConfig& config) const {
  if (!controller) return false;
  try {
    if (config.type == "Time") {
      sampler = std::make_shared<TimeSampler>(controller->horizonLength(),
                                              controller->dt());
      return true;
    }
  } catch (const ParameterException& e) {
    throw ParameterException("Could not load sampler " + config.type +
                             " from parameter file \'" + config.file.string() +
                             "\':\n" + e.what());
  }

  return false;
}

bool PilotParams::createPipeline(Pipeline* const pipeline,
                                 const PipelineConfig& config) const {
  if (pipeline == nullptr) return false;
  pipeline->setOuterloopDivisor(outerloop_divisor_);
  pipeline->setStopAfterFeedthrough(stop_after_feedthrough_);
  pipeline->setFeedthroughTimeout(feedthrough_timeout_);

  if (!createEstimator(pipeline->estimator_, pipeline_cfg_.estimator_cfg))
    logger_.warn("Did not create estimator '%s'!",
                 pipeline_cfg_.estimator_cfg.type.c_str());
  if (!createController(pipeline->outer_controller_,
                        pipeline_cfg_.outer_controller_cfg))
    logger_.warn("Did not create outer controller '%s'!",
                 pipeline_cfg_.outer_controller_cfg.type.c_str());
  if (!createController(pipeline->inner_controller_,
                        pipeline_cfg_.inner_controller_cfg))
    logger_.warn("Did not create inner controller '%s'!",
                 pipeline_cfg_.inner_controller_cfg.type.c_str());
  if (!createSampler(pipeline->sampler_, pipeline->outer_controller_,
                     pipeline_cfg_.sampler_cfg))
    logger_.warn("Did not create sampler '%s'!",
                 pipeline_cfg_.sampler_cfg.type.c_str());

  return true;
}

bool PilotParams::createPipeline(Pipeline* const pipeline) const {
  return createPipeline(pipeline, pipeline_cfg_);
}

std::ostream& operator<<(std::ostream& os, const PilotParams& params) {
  os << "Pilot Parameters:\n";
  os << "Directory:                    " << params.directory_ << '\n';
  os << '\n';
  os << "Quad File:                    " << params.quad_file_ << '\n';
  os << '\n';
  os << "Pipeline:\n" << params.pipeline_cfg_;
  os << '\n';

  os << "Trajectory Type:              " << params.traj_type_ << '\n';
  os << "dt min:                       " << params.dt_min_ << '\n';
  os << "dt telemetry:                 " << params.dt_telemetry_ << '\n';
  os << "outerloop divisor:            " << params.outerloop_divisor_ << '\n';
  os << "velocity in bodyframe:        " << params.velocity_in_bodyframe_
     << '\n';
  os << "takeoff height:               " << params.takeoff_heigth_ << '\n';
  os << "takeoff threshold:            " << params.takeoff_threshold_ << '\n';
  os << "start land speed:             " << params.start_land_speed_ << '\n';
  os << "brake deceleration:           " << params.brake_deceleration_ << '\n';
  os << "go to pose velocity:          " << params.go_to_pose_mean_vel_ << '\n';
  os << "stop after feedthough:        " << params.stop_after_feedthrough_
     << '\n';
  os << "trajectory visualization dt:  " << params.traj_viz_dt_ << '\n';
  os << "trajectory sphere size:       " << params.traj_viz_sphere_size_
     << '\n';
  os << "publish log variables:        " << params.publish_log_var_ << '\n';


  return os;
}

}  // namespace agi
