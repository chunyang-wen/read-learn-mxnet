### io.h

这个文件中定义了 `IIterator` 需要实现的接口。

+ Init: 利用 map of pair 来初始化 paramter (这个使用了dmlc-core中的宏)
+ BeforeFirst: 初始化迭代器（不清楚为什么用这个名字）
+ Next: 将迭代器移动至下一个元素
+ Value：返回当前元素

`IIterator` 派生自 `DataIter`, 后者相对于前者没有 `Init` 接口。

```c
iter->BeforeFirst();
while (iter->Next()) {
    const DType v = iter->Value();
    // other operations
}
```

### iter\_batchloader.h

从一个 `IIterator<DataInst>` 构造出一个指定 `batch_size` 的 loader。

+ num\_of\_overflow\_: 表示 `round\_batch` 是额外增加的元素

** 变量 `head_` 貌似可以删除 **

### inst\_vector.h

定义了类：

+ InstVector: 存储(label, example)的对
+ TensorVector: 存储不同尺寸的数据，利用连续的内存空间
+ TBlobBatch: 传递给 `NDArray` 前数据的存储位置
+ TBlobContainer: 派生自 `mshadow::TBlob`

### iter\_prefetcher.h

利用一个双向队列，保证使用 `k` 个 `batch` 在内存中。（目前不知道很做的原因）

其利用 `threaditer` 来循环利用 `k` 个 `batch` 的内存。

### iter\_csv.cc

+ CSVIterParam: 存储data和label的路径以及它们的shape
+ CSVIter: 派生自 IIterator\<DataInst\>，其CSV文件的解析是利用 `Parser` 类来完成

round\_batch: 表示当样本数不够一个 `batch` 时的处理方式：

+ round\_batch=False: 直接增加从最后一条数据往前数，直到满足 `batch\_size` 个样本
+ round\_batch=True: 从第一条数据开始，采用round-robin方式产出，此时 `iter.reset()` 接口的语义不
表示重新开始产出数据

### iter\_normalize.h

对数据做预处理，将数据全部减去均值。

### iter\_mnist.cc

+ MNISTParam: 数据和标注的地址
+ MNISTIter: `new PrefetchIter(new MNISTITER())` 会开多线程去准备 `batch`

包括下载数据，解析数据以及对数据进行shuffle。shuffle 使用对序列的 shuffle 功能。


### image\_augementer.h

基于`OpenCV` 做图像的一些增强，定义了接口 `ImageAugementer`:

+ Create
+ Init: 要求在所有操作前调用，类似于很多的 `IIterator`
+ Process

### image\_aug\_default.cc

默认的图像增强类：`DefaultImageAugementer`，派生自 `ImageAugementer`

默认的参数：`DefaultImageAugementParam`

+ 填充
+ 裁剪
+ 旋转
+ HSL通道的增强
+ 差值的方法：0-NN 1-bilinear 2-cubic 3-area 4-lanczos4 9-auto 10-rand

`Process` 函数中具体的操作：

1. Resize
2. normal augmentation by affine transformation
  + shear
  + rotate
  + scale
3. pad logic
4. crop logic
5. color space augementation

### image\_det\_aug\_default.cc

貌似 `det` 是 detection 的简写。（碉堡了，注释啥都没有，意淫的）

增加了一些具体相关的功能，具体还不是很了解。

+ `ImageDetLabel`
+ IOU = Intersection over Union ?
+ 'enum ImageDetAugDefaultCropEmitMode {kCenter, kOverlap};'
+ `enum ImageDetAugDefaultResizeMode {kForce, kShrink, kFit};`

### image\_io.cc

`image\_io` 必须要在编译时开启 `USE_OPENCV=1`

定义公共的函数：

+ get\_jpeg\_size
+ get\_png\_size
+ Imdecode: 解析 `jpeg` 和 `png` 的数据，调用 `cv::imdecode`
+ Imresize: 调用 `cv::resize`
+ MakeBorderShape
+ ResizeShape: 将输入的 `ishape` 转换为指定宽和高的 `oshape`
+ CopyMakeBorder

注册(`NNVM_REGISTER_OP`)了3个 `OP`:

+ `\_cvimdecode`
+ `\_cvresize`
+ `\_cvCopyBorderMaker`

### image\_recordio.h

定义了 `ImageRecordIO` 的结构体。

+ `struct Header`
+ `float *label`
+ `int num_label`
+ `uint8_t *content`
+ `size_t content_size`

定义了辅助函数：

+ `SaveHeader`
+ `Load`

### image\_iter\_common.h

`ImageLabelMap`：存储图像 `id` 以及和它们对应的标记

定义了多个 `Parameter`:

+ `BatchParam`: `batch\_size` 和 `round\_batch`
+ `ImageNormalizeParam`
+ `ImageDetNormalizeParam`
+ `ImageRecParserParam`
+ `ImageRecordParam`: `shuffle`, `seed`, `verbose`
+ `PrefetcherParam`: `prefetch\_buffer` 同时的 `batch` 个数, `dmlc::optional<int> dtype`

### iter\_image\_recordio{\_2}.cc

定义 `ImageRecordIOParser` 和 `ImageRecordIter`。`ImageRecordIter` 封装自 `PrefetcherIter` 和
`BatchLoader`。

其升级版本为 "\_2" 后缀的文件。

+ 加入了 `round\_batch` 的支持
+ 集中了多个功能：把批量预取，图片的normalize都封装在同一个类中
