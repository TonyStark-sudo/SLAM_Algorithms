#pragma once

#include "pose.h"
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>

std::vector<PoseStamped> readTumPoses(const std::string& pose_path);