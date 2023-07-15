/**
 * @file        - int8_calibrator.h
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief       - tensorrt int8量化类声明
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#ifndef TRT_INT8_CALIBRATOR_H
#define TRT_INT8_CALIBRATOR_H

#include <iostream>
#include <string>
#include <vector>

#include <cudnn.h>
#include <NvInfer.h>

#include "helper.h"

namespace helper
{
nvinfer1::IInt8Calibrator* get_int8_ca