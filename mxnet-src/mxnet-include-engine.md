### engine.h

基本的定义：

```c
/* 从后面看，Var和Opr会支持比较运算符 */
struct Var;
struct Opr;
typedef Var* VarHandle;
typedef Opr* OprHandle;
```

`CallbackOnComplete`: 用于异步函数中的回调

`FnProperty` 枚举类，表示操作的属性

+ kNormal
+ kCopyFromGPU
+ kCopyToGPU
+ kCPUPrioritized
+ kAsync

`Engine` 类的函数接口：

+ Get/\_GetSharedRef: 单例模式访问接口
+ NewVariable/DeleteVariable
+ NewOperator/DeleteOperator
+ Push/DeleteOperator
+ PushAsync
+ PushSync: 利用 PushAsync 来实现（没明白）
+ CreateCallback: 创建回调，引擎是this

等待相关的函数：

+ WaitForVar
+ WaitForAll

通知引擎关闭：

+ NotifyShutdown

