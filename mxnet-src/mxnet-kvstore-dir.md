### kvstore 相关的内容

这部分内容是基于李沐实现的 `ps-lite` 完成的。在相应的论文中其说明的算法比 `[Github](https://github.com)`
中实现的简单。论文中有使用一致性哈希来保证冗余和一致性。

### comm.h

不同设备间通信的模块。

统一接口: `Comm`

+ Init: 初始化key和数据的形状
+ Broadcast: 类似于挨个赋值
+ Reduce: 做聚合

有两个子类：

+ CommCPU
+ CommDevice

在 `Reduce` 时，每次累加的步骤是以 `4` 为步长。

使用 `NDArray` 中的 `CopyFromTo` 在不同或者相同设备之间传递数据。

### kvstore.cc

先看 `KVStore` 对外暴露的接口：

+ Create(const char\* type="local"): 默认创建本地模式的 `KVStore`。支持的类型包括：
  + local, local\_update\_cpu, local\_allreduce\_cpu
  + device, device\_allreduce\_device
  + dist\_\*
+ type(): 返回具体的类型名字
+ Init(keys, Values)：初始化 `Key` 和 `Value` 键值对。会hang住整个程序，直到所有节点完成初始化
+ Push(Keys, Values): `Push` 前会做相同 `Key` 的聚合
+ Pull(Keys, Values):
+ set\_updater: 用户可以控制 `Push` 时怎么处理，`updater(key, value, &value\_in\_store)`

要求在 `Push` 前必须将所有用到的 `Key` 初始化?

```cpp
/* wait write or read process */
/* local */
ndarray.WaitToWrite();
ndarray.WaitToRead();

/* dist */
Wait(Keys);
```


`KVStore` 的类型：

+ `KVStoreLocal`
+ `KVStoreDist`

`kvstore.cc` 只有一个 `Create` 函数的实现。其主要是根据传入的类型名字来调用具体的构造函数。

### kvstore\_local.cc

`KVStoreLocal` 使用一个哈希表来存储 `std::unordered_map<int, NDArray> local_`。

### kvstore\_dist.cc, kvstore\_dist\_server.h

`KVStoreDistServer` 通过往 `Executor` 中推送一个空的 `function` 来结束整个生命周期。

`kvstore\_dist.cc` 中表示的一个 worker 节点，`kvstore\_dist\_server.h` 中表示一个 server 节点。

`KVStoreDist` 竟然派生自 `KVStoreLocal`。`get_rank() == 0` 的节点负责向 server 发送退出命令。不同
worker 和 server 之间的角色是通过环境变量来判断的(这部分在 `ps-lite` 实现中)。在 `Init` 的调用中，
这个 `rank == 0` 的节点也必须要等所有的 `Key` 都初始化完毕后才会退出 `Init` 函数，而其他节点则可以
往引擎中推送相关的调用即可。

`KVStoreDistServer` 也是通过 `KVStoreDist` 来启动的。其在构造函数中会根据判断当前节点的类型。
