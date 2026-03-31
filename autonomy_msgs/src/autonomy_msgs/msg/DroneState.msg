# DroneState.msg - Represents the estimated drone state in real-time.

std_msgs/Header header            # Timestamp and frame reference (from MultiWiiNode clock)

geometry_msgs/PoseStamped pose     # Estimated drone pose (position + orientation)
geometry_msgs/TwistStamped twist   # Estimated drone velocity (linear + angular)