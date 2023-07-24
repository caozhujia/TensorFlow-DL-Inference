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
[2021-04-26 21:25:52][ DEBUG 