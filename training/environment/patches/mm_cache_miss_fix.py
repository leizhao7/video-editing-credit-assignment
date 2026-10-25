"""让 mm 缓存未命中变成**可恢复**的错误，而不是深埋在引擎进程里的断言。

背景：token-in-token-out 的 generate 是无状态的 —— 第 N 轮的 prompt 含前面所有
轮的 token，里面有 N 个图片占位符，引擎就要 N 份特征去填。但契约要求的是
「占位符能对应到特征」，**没要求特征必须从网络来**：

    if (cached := self._cache.get(mm_hash)) is not None:
        return cached                       # 从缓存满足，合法
    assert mm_item is not None              # 只有缓存也没有才失败

所以客户端可以只发新图、老图只发 hash（就是 mm_thin）。问题出在失败方式：
缓存被 LRU 挤掉时抛的是 `assert`，走 core.py 的
`_send_error_outputs_to_client`，客户端只拿到一个通用请求级错误、**不带 hash**。
实测 142 次失败，客户端按异常文本匹配的兜底一次都没触发。

这个补丁把断言换成带缺失 hash 的 `RuntimeError`。客户端因此能：
  1) 确定地识别出「是缓存未命中」，而不是靠猜字符串
  2) 知道**缺哪几张**，只补发那几张，而不是整批重来

这样缓存开多大就从**正确性问题**降级成**性能问题** —— 挤掉了自己会恢复，
只是多一次重发。原来的做法是「量准字节数再拍一个够大的值」，本质仍是赌。

标记串 MM_CACHE_MISS 是客户端的识别锚点，改这里就要同步改客户端。
"""
import sys

F = ("/opt/SkyRL/.venv/lib/python3.12/site-packages/vllm/multimodal/cache.py")

MARK = "MM_CACHE_MISS"

OLD = '        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"'
NEW = ('        if mm_item is None:\n'
       '            # 见 /patch/mm_cache_miss_fix.py：换成可恢复错误 + 缺失 hash，\n'
       '            # 客户端据此只补发缺的那几张。原来是 assert，客户端拿不到 hash。\n'
       f'            raise RuntimeError(f"{MARK} mm_hash={{mm_hash}}")')


def main() -> int:
    try:
        src = open(F, encoding="utf-8").read()
    except FileNotFoundError:
        print(f"[mm_miss] 找不到 {F}，跳过", flush=True)
        return 0
    if MARK in src:
        print("[mm_miss] 已打过补丁", flush=True)
        return 0
    n = src.count(OLD)
    if n == 0:
        print(f"[mm_miss] ✗ 锚点没匹配到，原文可能已变，不改", flush=True)
        return 1
    src = src.replace(OLD, NEW)
    open(F, "w", encoding="utf-8").write(src)
    import py_compile
    py_compile.compile(F, doraise=True)
    print(f"[mm_miss] ✅ 替换 {n} 处断言并通过语法检查：{F}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
