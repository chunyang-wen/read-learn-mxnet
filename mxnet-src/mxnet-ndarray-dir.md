### autograd.h/.cc

利用 `nnvm` 中模块相关的功能来实现自动梯度计算。

### ndarray\_functin.h/.cc

其主要实现是在 `ndarray_function-inl.h` 中，通过模板和宏来实现很多功能，主要包括如下：

+ 二元的操作
+ 矩阵每个元素的操作
+ 不同的随机数生成器

算子中有两个上下文：

+ Context: 主要是 `cpu`, `gpu` 执行的环境
+ RunContext: 主要是指对应设备上的 `stream`，有 `get_stream` 函数

### ndarray.cc

主要是实现 `ndarray.h` 中声明的接口
