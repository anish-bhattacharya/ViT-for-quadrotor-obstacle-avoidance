# Miscellaneous information about CPU status.
# Written by Marco Tranzatto (see: https://github.com/ethz-asl/mav_eurathlon/blob/master/mav_eurathlon_msgs/msg/CpuInfo.msg)

Header header

float32[] cpu_percent               # Current system-wide CPU utilization as a percentage, per CPU.
ProcessInfo[] heaviest_processes    # Info about heaviest running processes.