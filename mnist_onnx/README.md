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
[2021-04-26 21:25:52][  WARN ] : onnx2trt_utils.cpp:220: Your ONNX model has 