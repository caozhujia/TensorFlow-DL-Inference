
/**
 * @file        - demo_plugin.h
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief 
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#ifndef TRT_DEMO_PLUGIN_H
#define TRT_DEMO_PLUGIN_H

#include <string>
#include <vector>

#include <NvInferPlugin.h>
#include <NvInferRuntimeCommon.h>

#include "helper.h"

namespace nvinfer1
{
namespace plugin
{
class DemoPlugin : public nvinfer1::IPluginV2Ext
{
public:
    /**
     * parse 阶段构造函数
    */
    DemoPlugin(const std::string layer_name);

    /**
     * deserialize 阶段构造函数
    */
    DemoPlugin(const std::string layer_name, const void* data, size_t length);

    /**
     * 注意：删掉默认构造函数
    */
    DemoPlugin() = delete;

    /**
     * 返回 plugin 名称
    */
    const char* getPluginType() const override;

    /**
     * 返回 plugin 版本
    */
    const char* getPluginVersion() const override;

    /**
     * 返回op返回多少个Tensor
    */
    int getNbOutputs() const override;

    /**
     * 输出 tensor 维度
    */
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;
    // void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, 
    //         const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type, 
    //         nvinfer1::PluginFormat format, int maxBatchSize) override;

    /**
     * 初始化函数，一般分配内存等
    */
    int initialize() override;

    /**
     * 终止函数，一般释放内存等
    */
    void terminate() override;

    /**
     * 返回此 plugin 所需要显存大小
    */
    size_t getWorkspaceSize(int maxBatchSize) const override;

    /**
     * plugin 实际执行函数
    */
    int enqueue(int batchSize, const void* const* inputs, void** outputs, 
                void* workspace, cudaStream_t stream) override;
    
    /**
     * 获取序列化 plugin 大小
    */
    size_t getSerializationSize() const override;

    /**
     * 获取序列化 plugin 
    */
    void serialize(void* buffer) const override;

    /**
     * 返回结果的类型
    */
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    /**
     * 判断输入和输出类型数量是否正确。
     * 
     * 官方还提到通过这个配置信息可以告知TensorRT去选择合适的算法(algorithm)去调优这个模型。
    */
    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
                                 int nbOutputs, const nvinfer1::DataType* inputTypes, 
                                 const nvinfer1::DataType* outputTypes,
                                 const bool* inputIsBroadcast, const bool* outputIsBroadcast, 
                                 nvinfer1::PluginFormat floatFormat, int maxBatchSize) override;

    /**
     * 销毁
    */
    void destroy() override;

    /**
     * 深拷贝