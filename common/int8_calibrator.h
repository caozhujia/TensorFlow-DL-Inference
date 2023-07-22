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
nvinfer1::IInt8Calibrator* get_int8_calibrator(const std::string& calibrator_type,
                int batch_size, const std::vector<std::vector<float>>& data,
                const std::string& calib_data_name, bool read_cache);
}// helper

/**
 * TensorRT INT8量化类
 */
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(int batch_size, const std::vector<std::vector<float>>& data,
            const std::string& calib_data_name = "", bool read_cache = true);
        
    ~Int8EntropyCalibrator2();

    int getBatchSize() const override;

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

    const void* readCalibrationCache(std::size_t& length) override;

    void writeCalibrationCache(const void* ptr, std::size_t length) override;

private:
    std::string