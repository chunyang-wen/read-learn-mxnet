### base.h

base.h 主要定义了:

+ `Context`: 包括device type和device id
+ `RunContext`：包括一个stream指针，该指针是通过模板函数来得到的(`get\_stream`)

一些基础的宏定义：

+ 使用使用OPENCV/CUDA/CUDNN
+ 编译器版本检查
+ mxnet的版本相关信息
