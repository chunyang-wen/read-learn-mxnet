### profiler.h/.cc

用户做性能分析的

### thread\_pool.h

多个线程执行同样的函数

### engine.cc

环境变量：`MXNET_ENGINE_TYPE` 表示执行引擎的类型, 默认是 `ThreadedEnginePerDevice`。

目前有三种引擎：

+ NaiveEngine
+ ThreadedEngine
+ ThreadedEnginePerDevice

如果 `MXNET_PREDICT_ONLY` 不等于 0 的话会启动 `NaiveEngine`。(这是为了方便调试的把？)

### engine\_impl.h

这里面主要是声明了用户在 `engine.cc` 中的三个创建引擎的函数。（在 `engine.cc` 中 include）

定义引擎中变量和操作的基类，注释中说主要用于类型检查。其定义了相关的 `Cast` 函数。

### naive\_engine.cc

只支持同步的执行。

### threaed\_engine.h/.cc

`MXNET_ENGINE_INFO` 环境变量：指示是否打印出更多的引擎执行信息

`OprExecStat`: 统计这个 operation 的执行状态

+ opr\_name: 操作名字
+ opr\_start\_rel\_micros:
+ opr\_end\_rel\_micros:
+ thread\_id
+ device\_type
+ device\_id

`OprBlock` 表示 `Push` 到引擎中的一个操作

+ std::atomic<int> wait{0}: 这个块等待的任务
+ ThreadedOpr \*opr: 这个块上的操作
+ Context ctx: 算子执行的上下文，主要是设备的信息(device\_type)
+ int priority: 优先级（正负好像有不同的含义）
+ bool profiling{false}: 是否开启profile
+ OprExecStat \*stat: 算子执行信息

`VersionVarBlock` 表示 `ThreadedVar` 链表中的一个元素

+ VersionVarBlock \*next: 链表中的下个节点
+ OprBlock \* trigger: 这个block触发的操作
+ bool write{false}: 表示这个变量是否是个写操作（mutate操作）

`ThreadVar` 表示一个双向链表

+ AppendReadDependency
  如果没有pending\_write\_则直接执行;否则加入到队列末尾
+ AppendWriteDependency
  如果num\_pending\_reads\_ == 0和pending\_write\_为 `nullptr`
+ CompleteReadDependency
+ CompleteWriteDependency
+ SetToDelete: 没搞明白这个函数怎么使用的？
+ ready\_to\_read

`ThreadedOpr`: ThreadedEngine 中使用的操作表示

VarHandle: Var\*
OprHandle: Opr\*

`ThreadedEngine`:

+ NewVariable
+ NewOperator
+ DeleteOperator(OprHandle op)
+ DeleteVariable(SyncFn delete\_fn, Context exec\_ctx, VarHandle var)
+ Push
+ PushAsync
+ WaitForVar
+ WaitForAll
+ NotifyShutdown

`Dispatcher` 实际上是函数，其任务是将 `OprBlock` 压入引擎的队列

### threaded\_engine\_pooled.cc

本质上是通过这个 engine 将所有的功能暴露出去的，其派生自 `ThreadedEngine`

