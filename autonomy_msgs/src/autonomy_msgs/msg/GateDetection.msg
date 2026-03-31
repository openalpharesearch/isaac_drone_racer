std_msgs/Header header         # ⏳ Timestamp + frame reference
int32 gate_id                  # 🟦 Guessed from SLAM + seen gates
float32 tracking_confidence    # 🟦 Confidence in gate match (0.0 - 1.0)

geometry_msgs/Pose pose        # 🟩 Gate position & orientation
float32 distance               # 🟦 Distance from drone to gate
float32 confidence             # 🟦 Detection confidence (0.0 - 1.0)
bool detected                  # 🟩 True if gate is visible

geometry_msgs/Point center_point      # 🟩 Gate center for trajectory alignment
geometry_msgs/Point[] outer_corners   # 🟩 Full gate frame (incl. stacked gates)
geometry_msgs/Point[] inner_corners   # 🟩 Fly-through region(s)
