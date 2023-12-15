
/**
 * @file        - demo_plugin.cpp
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief 
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#include "demo_plugin.h"
#include "helper.h"

using namespace nvinfer1;
using nvinfer1::plugin::DemoPlugin;
using nvinfer1::plugin::DemoPluginCreator;

// log
#define log(...) {  \
    char str[100];  \
    sprintf(str, __VA_ARGS__);  \
    helper::debug << "(@ _ @) Here ---> call " << "[" << __FILE__    \
                  << __FUNCTION__ << "][Line " << __LINE__ << "]"   \
                  << str << std::endl;                              \
}

DemoPlugin::DemoPlugin(const std::string layer_name) : m_layer_name(layer_name)
{
    log(" run here now! ");
    m_plugin_name = "demo_plugin";
    m_plugin_version = "01";
}

DemoPlugin::DemoPlugin(const std::string layer_name, const void* data, size_t length)
    : m_layer_name(layer_name)
{
    log(" run here now! ");
}

const char* DemoPlugin::getPluginType() const
{
    log(" run here now! ");
    return m_plugin_name.c_str();
}

const char* DemoPlugin::getPluginVersion() const
{
    log(" run here now! ");
    return m_plugin_version.c_str();
}

int DemoPlugin::getNbOutputs() const
{
    log(" run here now! ");
    return 1;
}

nvinfer1::Dims DemoPlugin::getOutputDimensions(int index, 
    const nvinfer1::Dims* inputs, int nbInputDims)
{
    log(" run here now! ");
    return nvinfer1::Dims3(inputs[1].d[1],inputs[1].d[2],inputs[1].d[3]);
}

bool DemoPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const
{
    log(" run here now! ");
    return false;
}

int DemoPlugin::initialize()
{
    log(" run here now! ");
    return 0;
}

void DemoPlugin::terminate()
{
    log(" run here now! ");
}

size_t DemoPlugin::getWorkspaceSize(int maxBatchSize) const
{
    log(" run here now! ");
    return 0;
}

int DemoPlugin::enqueue(int batchSize, const void* const* inputs, 
            void** outputs, void* workspace, cudaStream_t stream)
{
    log(" run here now! ");
    return 0;
}

size_t DemoPlugin::getSerializationSize() const
{
    log(" run here now! ");
    return 0;
}

void DemoPlugin::serialize(void* buffer) const
{
    log(" run here now! ");
}

nvinfer1::DataType DemoPlugin::getOutputDataType(int index, 
    const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    log(" run here now! ");
    return nvinfer1::DataType::kFLOAT;