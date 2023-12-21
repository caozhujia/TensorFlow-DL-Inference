
# TensorFlow Deep Learning Inference Base

## 1.Description
This project encapsulates TensorRT to accelerate deep learning models, especially supporting models in caffe and onnx formats. Most of it comes from TensorRT C++ samples along with reference codes from several GitHub sources.

# 2.Environment

Recommended operating system is Ubuntu.

## 2.1 Ubuntu

* TensorRT 7+
* GUN tools (g++ 7.5.0)
* CMake tools (3.10+)

## 2.2 Windows

* TensorRT 7+
* Visual Studio 2017
* CMake tools (3.10+)

# 3.Directory

```
  ├── bin: Directory for executable files
  ├── build: CMake build directory
  ├── common: TeslaRT basic classes and some common functions, compiled into library for invocation
  ├── lib: Library creation directory
  ├── plugin: TensorRT plugin directory for custom plugins
  ├── CMakeLists.txt: Root directory CMake configuration file
  ├── mnist_caffe: TensorRT parsing caffe model demo
  ├── mnist_onnx: TensorRT parsing ONNX model demo
  └── faster_rcnn: TensorRT acceleration Faster RCNN algorithm
```

# 4.Running

Modify the `CUDA` and `TensorRT` directories in the CMakeLists.txt file in the root directory, with cudnn assumed to be installed in the same directory as CUDA.

```cmake
set(CUDA_ROOT_DIR "D:/Software/CUDA11.0/development")
set(TRT_ROOT_DIR "D:/Software/CUDA11.0/development")
```

GO to your project's directory, run the following commands

```shell
cd TensorFlow-DL-Inference
mkdir build
cd build
cmake ..
make
```

After compiling, executable files will be created in the bin folder, libraries will be generated in the lib directory. By default, due to complexity in configuring the dynamic library in the windows environment, a static library is created, which you can change according to your need.

Execute the file in the bin directory to accelerate the deep learning model. Here's an example using the faster rcnn demo.

```shell
PS E:\TensorFlow-DL-Inference> .\bin\Release\faster_rcnn.exe
[2021-04-25 20:48:32][  WARN ] : Dynamic size input setting invalid!
[2021-04-25 20:48:32][ DEBUG ] : deserialize engine from faster_rcnn/model/faster_rcnn.bin
[2021-04-25 20:48:38][ DEBUG ] : max batch size of deserialized engine: 1
[2021-04-25 20:48:38][ DEBUG ] : create execute context and malloc device memory...
[2021-04-25 20:48:38][ DEBUG ] : init engine...
[2021-04-25 20:48:38][ DEBUG ] : malloc device memory
[2021-04-25 20:48:38][ DEBUG ] : nbBingdings: 5
[2021-04-25 20:48:38][ DEBUG ] : input: 
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 0, name: data, size in byte: 2250000
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 3 dimemsion
[2021-04-25 20:48:38][ DEBUG ] : input: 
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 1, name: im_info, size in byte: 12  
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 3 dimemsion
[2021-04-25 20:48:38][ DEBUG ] : output: 
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 2, name: rois, size in byte: 4800
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 3 dimemsion
[2021-04-25 20:48:38][ DEBUG ] : output:
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 3, name: bbox_pred, size in byte: 100800
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 4 dimemsion
[2021-04-25 20:48:38][ DEBUG ] : output:
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 4, name: cls_prob, size in byte: 25200
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 4 dimemsion
[2021-04-25 20:48:38][  WARN ] : output_bbox_pred: 25200
[2021-04-25 20:48:38][  WARN ] : output_cls_prob: 6300
[2021-04-25 20:48:38][  WARN ] : output_rois: 1200
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 aeroplane
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 bicycle
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 bird
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 boat
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 bottle
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 bus
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 car
[2021-04-25 20:48:38][  INFO ] : indices size is: 1 cat
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 chair
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 cow
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 diningtable
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 dog
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 horse
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 motorbike
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 person
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 pottedplant
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 sheep
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 sofa
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 train
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 tvmonitor
```