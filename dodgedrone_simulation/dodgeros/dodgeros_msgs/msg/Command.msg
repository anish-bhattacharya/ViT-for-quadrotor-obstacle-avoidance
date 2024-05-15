Header header 

# Time [s]
float64 t

# If input is in terms of collective thrust & bodyrates or single rotor thrusts
bool is_single_rotor_thrust

# Collective mass-normalized thrust [m/s^2]
float64 collective_thrust

# Angular rates in body frame [rad/s]
geometry_msgs/Vector3 bodyrates

# Single rotor thrusts [N]
float64[4] thrusts
