Header header

# Time [s]
float64 t

# Bridge
std_msgs/String bridge_type
std_msgs/Bool bridge_armed
std_msgs/Bool guard_triggered

# Reference
int32 num_references_in_queue
float64 reference_left_duration
float64 rmse
dodgeros_msgs/QuadState reference

# Onboard Telemetry
float64 voltage
