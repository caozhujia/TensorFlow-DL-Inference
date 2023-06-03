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

nvinfer1::IInt8Calibrator* helper::get_int8_calibrator(
                const std::string& calibrator_type,
                int batch_size, 
                const std::vector<std::vector<float>>& data,
                const std::string& calib_data_name, 
                bool read_cache)
{
    if(calibrator_type == "Int8EntropyCalibrator2")
    {
        return new Int8EntropyCalibrator2(batch_size, data, calib_data_name, read_cache);
    }
    else
    {
        helper::debug << "[INT8] : Unsupport calibrator type "<< std::endl;
        return nullptr;
    }
    
}


Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batch_size, 
            const std::vector<std::vector<float>>& data,
            const std::string& calib_data_name , 
            bool read_cache)
            : m_calib_data_name(calib_data_name), 
            m_batch_size(batch_size),
            m_read_cache(read_cache)
{
    m_data.reser