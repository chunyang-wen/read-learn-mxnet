### storage.h

是存储介质的接口定义：

+ Alloc
+ Free
+ DirectFree: 忽略内存池
+ Get/\_GetSharedRef

#### StorageHandle

```c
struct Handle {
    void* dptr;
    size_t size;
    Context ctx;
}
```
