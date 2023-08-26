# MNIST ONNX

## 1.Description

继承`TrtBase`类实现`mnist`手写数字算法。

## 2.Running

### 2.1 编译生成可执行文件

```shell
cd TensorRT-Base
mkdir build
cd build
cmake ..
make
```

### 2.2 运行 faster rcnn

```shell
PS E:\TensorRT-Base> .\bin\Release\mnist_onnx.exe
[2021-04-26 21:25:52][  WARN ] : empty engine file name, skip save!
[2021-04-26 21:25:52][ DEBUG ] : build onnx engine from mnist_onnx/model/mnist.onnx ...
----------------------------------------------------------------
Input filename:   mnist_onnx/model/mnist.onnx
ONNX IR version:  0.0.3
Opset version:    8
Producer name:    CNTK
Producer version: 2.5.1
Domain:           ai.cntk
Model version:    1
Doc string:
----------------------------------------------------------------
[2021-04-26 21:25:52][  WARN ] : onnx2trt_utils.cpp:220: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[2021-04-26 21:25:52][ DEBUG ] : unmark original output...
[2021-04-26 21:25:52][ DEBUG ] : mark custom output...
[2021-04-26 21:25:52][ DEBUG ] : FP16 support: 1
[2021-04-26 21:25:52][ DEBUG ] : INT8 support: 1
[2021-04-26 21:25:52][ DEBUG ] : Max batchsize: 1
[2021-04-26 21:25:52][ DEBUG ] : Max workspace size: 16777216
[2021-04-26 21:25:52][ DEBUG ] : Number of DLA core: 0
[2021-04-26 21:25:52][ DEBUG ] : Max DLA batchsize: 268435456
[2021-04-26 21:25:52][ DEBUG ] : Current use DLA core: 0
[2021-04-26 21:25:52][ DEBUG ] : build engine ... 
[2021-04-26 21:26:00][ DEBUG ] : build Network done
[2021-04-26 21:26:00][ DEBUG ] : serialize engine to
[2021-04-26 21:26:00][  WARN ] : empty engine file name, skip save!
[2021-04-26 21:26:00][ DEBUG ] : create execute context and malloc device memory...
[2021-04-26 21:26:00][ DEBUG ] : init engine...
[2021-04-26 21:26:00][ DEBUG ] : malloc device memory
[2021-04-26 21:26:00][ DEBUG ] : nbBingdings: 2
[2021-04-26 21:26:00][ DEBUG ] : input:
[2021-04-26 21:26:00][ DEBUG ] : binding bindIndex: 0, name: Input3, size in byte: 3136
[2021-04-26 21:26:00][ DEBUG ] : binding dims with 4 dimemsion
[2021-04-26 21:26:00][ DEBUG ] : output:
[2021-04-26 21:26:00][ DEBUG ] : binding bindIndex: 1, name: Plus214_Output_0, size in byte: 40
[2021-04-26 21:26:00][ DEBUG ] : binding dims with 2 dimemsion
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@