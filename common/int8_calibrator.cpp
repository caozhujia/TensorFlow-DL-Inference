/**
 * @file        - int8_calibrator.cpp
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief       - TensorRT INT8量化类实现
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#include <iterator>
#include <algorithm>
#include <fstream>
#include <memory>
#include <string.h>

#include "int8_calibrator.h"

nvinfer1::IInt8Calibrator* helper::get_int8_calib