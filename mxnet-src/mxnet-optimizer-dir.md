### optimizer.h

这部分定义在 `cpp-package/include/mxnet-cpp/optimizer.h`。

`Optimizer` 表示梯度更新的方法, 学习速率变化方法(learing rate, lr\_scheduler) 在 Python 端实现了
（mxnet/python/mxnet/lr\_scheduler.py)

(如何控制学习速率，需要进一步阅读源码)

目前 C\+\+ 单好像并没有支持学习速率的变化。

目前支持的 `Optimizer`:

+ SGDOptimizer
+ RMSPropOptimizer
+ AdamOptimizer
+ AdaGradOptimizer
+ AdaDeltaOptimizer

在 `mxnet/src/optimizer/sgd-inl.h`，定义了一个 `SGDOpt`.
