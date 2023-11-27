
# Plugin

## 1.Description

TensorRT已经只支持了许多常见的神经网络层,比如卷积, 池化, BN等等. 但是依然还有很多操作和算子是不支持的,所以TensorRT提供了接口让我们可以编写插件来实现自己的自定义层。

随着tensorRT的不断发展(v5->v6->v7)，TensorRT的插件的使用方式也在不断更新。插件接口也在不断地变化，由v5版本的`IPluginV2Ext`，到v6版本的`IPluginV2IOExt`和`IPluginV2DynamicExt`。

| TensorRT版本        | 混合精度 | 动态大小输入 | Requires extended runtime |      |
| ------------------- | -------- | ------------ | ------------------------- | ---- |
| IPluginV2Ext        | 5.1      | Limited      | No                        | No   |
| IPluginV2IOExt      | 6.0.1    | General      | No                        | No   |
| IPluginV2DynamicExt | 6.0.1    | General      | Yes                       | Yes  |

## 2.Plugin Class

需要写两个类：

- `DemoPlugin`，继承`IPluginV2Ext`，是插件类，用于写插件具体的实现
- `DemoPluginCreator`，继承`IPluginCreator`，是插件工厂类，用于根据需求创建该插件

对了，插件类继承`IPluginV2DynamicExt`才可以支持动态尺寸，其他插件类接口例如`IPluginV2IOExt`和前者大部分是相似的。

```cpp

class DemoPlugin : public nvinfer1::IPluginV2Ext

class DemoPluginCreator : public nvinfer1::IPluginCreator
```

## 3.DemoPlugin Class

```C++
class DemoPlugin : public nvinfer1::IPluginV2Ext
{
public:
    DemoPlugin(const std::string layer_name);
    DemoPlugin(const std::string layer_name, const void* data, size_t length);
    DemoPlugin() = delete;
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;
    int initialize() override;
    void terminate() override;
    size_t getWorkspaceSize(int maxBatchSize) const override;
    int enqueue(int batchSize, const void* const* inputs, void** outputs, 
                void* workspace, cudaStream_t stream) override;  
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
    bool canBroadcastInputAcrossBatch(int inputIndex) const override;
    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
                                 int nbOutputs, const nvinfer1::DataType* inputTypes, 
                                 const nvinfer1::DataType* outputTypes,
                                 const bool* inputIsBroadcast, const bool* outputIsBroadcast, 
                                 nvinfer1::PluginFormat floatFormat, int maxBatchSize) override;
    void destroy() override;
    nvinfer1::IPluginV2Ext* clone() const override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;
private:
    const std::string m_layer_name;
    size_t m_cpoy_size;
    std::string m_namespace;
    std::string m_plugin_version;
    std::string m_plugin_name;