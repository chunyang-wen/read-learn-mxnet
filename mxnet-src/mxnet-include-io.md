### io.h

定义基本的迭代器类型

+ IIterator
+ DataInst/DataBatch: 一个样本和一批样本

注册迭代器的宏：

```c
#define MXNET_REGISTER_IO_ITER(name) \
    DMLC_REGISTRY_REGISTER(::mxnet::DataIteratorReg, DataIteratorReg, name)
```
