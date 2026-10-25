"""让 SkyRL 的推理服务地址支持 IPv6。构建镜像时执行一次，直接改源码。

Modal 的容器间网络是 IPv6（fdaa:ac1d:...）。SkyRL 拼 URL 的写法是
``f"http://{ip}:{port}"``，而 IPv6 地址本身满是冒号，拼出来就是

    http://fdaa:ac1d:8078:18f3:93fe:aa3b:1130:e571:8000/health

解析器按最后一个冒号切，把 `...e571:8000` 当成端口去 int()：

    ValueError: invalid literal for int() with base 10:
                'ac1d:8078:18f3:93fe:aa3b:1130:e571:8000'
    RayTaskError(InvalidURL): ray::VLLMServerActor.start()

RFC 3986 要求 IPv6 在 URL 里用方括号包起来。

**为什么不在 get_node_ip() 源头包**：同一个 `_ip` 还喂给两个不接受方括号的地方 ——
``os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = self._ip``（vllm_server_actor.py:250）
和 DP RPC 的 ``(self._ip, dp_rpc_port)``（:280）。那两处要裸地址。所以只改真正
拼 URL 的三处，其余保持原样。

改法是纯文本替换而不是 monkey-patch：这三处都在 f-string 里，运行时替换要重写
函数体，脆得多；而镜像构建时改源码是一次性的、可验证的（下面会 assert 改到了）。
"""
import re
import sys

D = "/opt/SkyRL/skyrl/backends/skyrl_train/inference_servers"
T = "/opt/SkyRL/skyrl/train/utils"

HELPER = '''

def _sk_hosthdr(ip):
    """IPv6 地址在 URL 里要用方括号包起来（RFC 3986）。IPv4 原样返回。"""
    ip = str(ip)
    if ":" in ip and not ip.startswith("["):
        return "[" + ip + "]"
    return ip
'''

# (文件, 原文, 替换)
EDITS = [
    (f"{D}/common.py",
     'return f"http://{self.ip}:{self.port}"',
     'return f"http://{_sk_hosthdr(self.ip)}:{self.port}"'),
    (f"{D}/vllm_server_actor.py",
     'url = f"http://{self._ip}:{self._port}/health"',
     'url = f"http://{_sk_hosthdr(self._ip)}:{self._port}/health"'),
    (f"{D}/vllm_router.py",
     'router_url = f"http://{ip}:{self._router_args.port}"',
     'router_url = f"http://{_sk_hosthdr(ip)}:{self._router_args.port}"'),
    # 端口预留必须和实际监听用同一个协议族，否则预留是假的。
    # find_and_reserve_port 原本用 AF_INET 绑 ("", port) —— 只证明 **IPv4** 的
    # 那个端口空闲；而下面我们把 server/router 的监听改成了 "::"（IPv6 双栈，
    # 要求 IPv4 和 IPv6 两边都空闲）。IPv6 那半从没被预留过，谁碰巧占了它，
    # router 启动就是
    #     Error starting router: Address already in use (os error 98)
    #     VLLMRouter process exited with code 1
    # 这正是那个"重跑一次就好了"的间歇性启动失败。
    # 改成 AF_INET6 + bind(("::", port))：Linux 默认 IPV6_V6ONLY=0，双栈同时
    # 占住两边，预留才和实际绑定等价。
    # 非 colocate（训练/推理分开占卡）时权重同步走 BroadcastTransferStrategy，
    # 这条路上有两处写死 IPv4，在 Modal 的纯 IPv6 容器网络里必然 gaierror：
    #
    # 1) vLLM 建 TCP store 的监听 socket 写死 AF_INET，却拿 IPv6 地址去 bind：
    #        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #        listen_socket.bind((host, port))     # host = "fdaa:..."
    #    → RayTaskError(gaierror)，卡在 init_weight_sync_state。
    # 2) SkyRL 探测空闲端口也用默认的 AF_INET。
    #
    # 按 host 里有没有冒号自动选地址族，IPv4 环境行为不变。
    ("/opt/SkyRL/.venv/lib/python3.12/site-packages/vllm/distributed/utils.py",
     'listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)',
     'listen_socket = socket.socket('
     'socket.AF_INET6 if ":" in str(host) else socket.AF_INET, socket.SOCK_STREAM)'),
    (f"{D.replace('inference_servers', 'weight_sync')}/broadcast_strategy.py",
     '        with socket.socket() as sock:\n'
     '            sock.bind(("", 0))',
     '        with socket.socket(socket.AF_INET6) as sock:\n'
     '            sock.bind(("", 0))'),
    (f"{D}/common.py",
     '        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n'
     '            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n'
     '            sock.bind(("", port))',
     '        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)\n'
     '            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n'
     '            sock.bind(("::", port))'),
    # 监控用的 Prometheus 抓取器，同一类 URL 拼接问题，但它藏在 train/utils 下、
    # 而且要跑到**第一步训练开始**才炸，前面所有关卡全绿了才轮到它：
    #     httpx.InvalidURL: Invalid port: 'ac1d:8077:18f3:93fe:682c:7249:8684:-1'
    # 顺手连 port 一起挡掉 —— Ray 的 MetricsExportPort 在我们这套配置里是 -1
    # （没开 metrics export），原来的 `if not port` 拦不住 -1，加了方括号也还是
    # 个非法端口。URL 列表为空时 scraper 只会 warning 一句，不影响训练。
    (f"{T}/vllm_metrics_scraper.py",
     '        if not ip or not port:\n'
     '            continue\n'
     '        urls.append(f"http://{ip}:{port}/metrics")',
     '        if not ip or not port or int(port) <= 0:\n'
     '            continue\n'
     '        urls.append(f"http://{_sk_hosthdr(ip)}:{port}/metrics")'),
    # —— 下面这两处才是多节点真正卡住的地方 ——
    #
    # server 硬编码绑 "0.0.0.0"（IPv4-only），而健康检查敲的是
    #     http://[fdaa:...]:port/health   （self._ip = get_node_ip()）
    # 协议族对不上，永远连不上。表现为引擎 warmup 明明完成了、日志再无动静，
    # 34 分钟后报 "Server failed to become healthy within 2400s"。
    #
    # 关键证据：超时的两个引擎和 driver 在**同一个节点**上 —— 所以不是跨节点
    # 网络隔离，是 socket 根本没在 IPv6 上监听。
    #
    # 单节点一直没事，是因为 Ray 的 get_node_ip_address() 在单容器里返回 IPv4
    # (172.20.x.x)，敲的也是 IPv4。多节点时 train_multi 给 ray start 传了 IPv6
    # 的 --node-ip-address，整条链路才切到 IPv6。
    #
    # 改成 "::"（IPv6 any）而不是逐个判断：Linux 默认 IPV6_V6ONLY=0，绑 "::"
    # 会同时接受 IPv6 和 IPv4-mapped 连接，所以单节点那条路也照常工作。
    # vLLM 的 create_server_socket 本来就会按地址族选 AF_INET6，不需要额外改。
    (f"{D}/vllm_server_actor.py",
     'self._cli_args.host = "0.0.0.0"',
     'self._cli_args.host = "::"'),
    (f"{D}/vllm_server_actor.py",
     'cli_args.host = "0.0.0.0"',
     'cli_args.host = "::"'),
    # —— CPU render server 要认 file:// 图片 ——
    #
    # 我们把渲染从引擎卸载到 CPURenderServer 池（见 vlm_gen_fix 的 _pool_render），
    # 而 render_server.py 起 vLLM render app 的参数里没有 allowed-local-media-path，
    # file:///tmp/... 的帧会被 400 拒掉。引擎侧我们是靠 engine_init_kwargs 传的，
    # 这个独立进程只能改它的启动参数。
    (f"{D}/render_server.py",
     '["--model", model_path, "--host", _RENDER_HOST, "--port", str(port), "--trust-remote-code"]',
     '["--model", model_path, "--host", _RENDER_HOST, "--port", str(port), "--trust-remote-code", "--allowed-local-media-path", "/tmp"]'),
]


def main() -> None:
    # 每个文件各自追加一份 helper，**不做 import**。
    #
    # 上一版是把 helper 放 common.py、另两个文件插一行 import，结果炸了：
    #   SyntaxError: invalid syntax
    # 插入点是用正则 ^(from|import) .*$ 找最后一个 import 算的，而
    # vllm_server_actor.py 的 import 是多行括号式的
    #     from ...common import (
    #         get_node_ip,
    #     )
    # 正则只匹到第一行 `from ... import (`，插入点落进括号里，语法当场断掉。
    # 而且这种错要等多节点跑到一半才暴露（16 卡都起来了才 import 到那个模块）。
    #
    # 各自定义一份就不需要 import，也就没有插入点的问题。_sk_hosthdr 是在方法体
    # 里调用的，运行时才查模块全局，所以定义追加在文件末尾完全可以。
    for path, old, new in EDITS:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if new in src and old not in src:
            print(f"[ipv6] 已经打过：{path}")
            continue
        if old not in src:
            # 上游改了写法就明着失败 —— 静默跳过的话多节点会在半小时后
            # 用同一个 InvalidURL 挂掉，而那时已经看不出是这里没打上。
            raise SystemExit(f"[ipv6] 找不到待替换的代码，上游可能改了：{path}\n  {old}")
        src = src.replace(old, new)
        if "_sk_hosthdr" not in src.split(new)[0]:
            src += HELPER
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        # 立刻验语法。上一版就是改出了语法错却一路带进镜像。
        import py_compile
        py_compile.compile(path, doraise=True)
        print(f"[ipv6] 已修补并通过语法检查：{path}")

    # 兜底：整个 inference_servers 目录里凡是拿 "0.0.0.0" 当绑定地址的，一律换成
    # "::"。上面 EDITS 是逐处精确替换，覆盖了 VLLMServerActor；但失败点会一层层
    # 往后移 —— 引擎那关过了之后是
    #     RuntimeError: VLLMRouter failed to become healthy within 2400s
    # 也就是 router 自己也绑在 IPv4 上。与其一层层追着改、每验一次赔 40 分钟，
    # 不如把这一类全扫掉。
    #
    # 只替换成对出现的绑定字面量，不碰注释和文档字符串里的示例地址（那些通常带
    # http:// 前缀或在引号外），所以限定为独立的 "0.0.0.0" 字符串字面量。
    import glob
    extra = 0
    for path in glob.glob(f"{D}/*.py"):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if '"0.0.0.0"' not in src:
            continue
        src2 = src.replace('"0.0.0.0"', '"::"')
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src2)
        import py_compile
        py_compile.compile(path, doraise=True)
        n = src.count('"0.0.0.0"')
        extra += n
        print(f"[ipv6] 兜底替换 {n} 处绑定地址：{path}")
    print(f"[ipv6] 兜底共替换 {extra} 处")

    # ── gloo → nccl ──────────────────────────────────────────────────────
    # SkyRL 建进程组用的是混合 backend：
    #     torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl")
    # CPU 张量走 gloo、GPU 张量走 nccl。nccl 在这个环境是通的（单节点权重同步
    # 一直用它），**只有 gloo 建不起跨节点连接**：
    #     Gloo connectFullMesh failed ... timed out connecting: SO_ERROR
    #
    # 根因是地址族。两次失败的错误不同，正好把它指出来：
    #     不设 GLOO_SOCKET_IFNAME    → Timed out waiting for recv（连上了收不到）
    #     设 GLOO_SOCKET_IFNAME=eth0 → connectFullMesh timed out（连都连不上）
    # 设了反而更糟，说明 gloo 拿 eth0 时挑了 IPv4（eth0 上 IPv4/IPv6 都有），
    # 而 Modal 容器间**只有 IPv6 可达**。gloo 没有 NCCL_SOCKET_FAMILY 那样的
    # 地址族开关，这是它的硬限制，改环境变量绕不过去。
    #
    # 所以把 CPU 那半也交给 nccl，让 gloo 完全不参与跨节点通信。
    # SkyRL 用混合 backend 的原因写在它自己的注释里 —— 避免 'invalid device
    # ordinal'，那个错误的前提是没有正确 set_device；FSDP 路径上每个 worker
    # 都会绑定自己的 GPU，所以纯 nccl 大概率没问题。**这是个赌注，不是定论**：
    # 如果重新冒出 invalid device ordinal，就说明这条路不通，得回到 gloo 并另想
    # 办法（例如让 Modal 暴露 IPv4，或改用 Megatron 后端）。
    import glob as _glob
    nc = 0
    for path in _glob.glob(f"{D.replace('inference_servers', 'workers')}/**/*.py",
                           recursive=True):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if 'backend="cpu:gloo,cuda:nccl"' not in src:
            continue
        src = src.replace('backend="cpu:gloo,cuda:nccl"', 'backend="nccl"')
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        import py_compile
        py_compile.compile(path, doraise=True)
        nc += 1
        print(f"[ipv6] gloo→nccl：{path}")
    print(f"[ipv6] gloo→nccl 共改 {nc} 个文件")

    # 【已撤销】曾经在这里给 FSDP 的 init_process_group 前注入
    #     os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET6"
    # 目的是让 FSDP 跨节点走 IPv6 而不影响引擎单节点内的 TP。撤销原因：
    # 那个补丁确实生效了（报错从 ncclRemoteError 变成了 ProcessGroupNCCL.cpp:3135
    # invalid usage），但真正的错误是最后一行 "Bootstrap : no socket interface
    # found" —— 当时 NCCL_SOCKET_IFNAME 被错设成 eth0，而 eth0 上**只有 IPv4**，
    # 和 AF_INET6 的交集是空集。承载 i6pn 的其实是 eth1。
    #
    # 选对网卡（见 modal_s12_grpo.py 的 getifaddrs 反查）之后就不需要强制地址族了：
    # eth1 上只有 IPv6，NCCL 自然用 IPv6。而 NCCL_SOCKET_FAMILY 是全局变量、引擎
    # 也继承，留着只会再弄坏引擎。
    # 自检：确认三处都生效，且 helper 行为正确
    sys.path.insert(0, "/opt/SkyRL")
    ns = {}
    exec(HELPER, ns)
    f = ns["_sk_hosthdr"]
    assert f("fdaa:ac1d::1") == "[fdaa:ac1d::1]", f("fdaa:ac1d::1")
    assert f("10.0.0.1") == "10.0.0.1"
    assert f("[fdaa::1]") == "[fdaa::1]"
    for path, _, new in EDITS:
        with open(path, encoding="utf-8") as fh:
            assert new in fh.read(), f"未生效：{path}"
    print("[ipv6] 三处替换全部生效，helper 自检通过")


if __name__ == "__main__":
    main()
