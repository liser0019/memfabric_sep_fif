# A5 单份 Host KV 池：ACL 共享映射实验

本分支基于 MemFabric `clean` 的 `4b5bfe2490a7e0b9bc7214743411cb173c288945`。
增加可选的 `OFFLOAD_FLAG_ACL_SHARED_HOST`，默认 LOCAL/SHARED 行为保持原样。
这是等待 A5 实机验证的实验实现，不是已验证的整网 PD 修复。

## 1. 旧 `offload.empty()` 为什么可能不能被 AIV 访问

`offload.empty()` 只是 `offload.malloc()` 加上 `torch.frombuffer()`，并不负责设备映射。
底层路径取决于 scene、GVA 版本、dataOpType 和 flags，不能把它概括为一个固定分配器。

* `acc_offload_shared_dram_entry.cpp` 使用 `MTE`、`DRAM_MAP_HOST_VA | UNRESTRICTED_MEM`，
  经 `MemSegment::Create()` 选择 VMM 或连接型 segment。
* VMM 路径使用 `HalMemCreate/Map/SetAccess`，并通过 FABRIC handle 导出、导入共享物理内存。
  工厂明确包含 `ASCEND_950`，源码也针对 A5 调整了物理内存属性和大页类型。
  因此没有依据断言“这些旧函数一概不支持 950”。
* 旧连接型路径曾调用 `HalHostRegister(...HOST_MEM_MAP_DEV..., &dva)`，不是完全没有设备注册。
  但连接型 `Import/Mmap` 只建立远端 GVA 记录，不在当前进程映射对端物理页。
  `AddVaInfoFromExternal({import.gva, 0, 0, ...})` 的两个零明确表示未提供本地 DVA/HVA。
* 原 SHARED `GetDva()` 只检查本 rank 的区间，然后直接返回原指针；这个假设仅适合已正确建立的
  统一 VMM 映射。它不能让一个未映射的 GVA 自动成为 AIV 可访问地址。
  现有 vLLM SHARED 接入也直接使用广播的 rank0 地址，没有逐进程查询 DVA。

历史补丁记载的对照是“旧 offload 地址失败、ACL mapped-host 成功”。这支持优先排查映射与地址使用，
但没有保存完整的 segment 选择、HAL 返回值和 MMU fault 信息，不能据此唯一定位到某一个函数。
`507035` 是 `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`，仅表示 Vector Core 执行异常，
不是一个专门表示“该分配 API 不支持 A5”的错误码。

特别要区分：如果当时实际走 VMM，则应继续核对 import、SetAccess、拓扑可达性和传入 kernel 的地址；
不能拿连接型路径的缺陷，直接解释一次未经确认的 VMM 失败。

## 2. 官方文档能确认什么

1. Host 和 Device 看到的地址默认可能不同；不能默认 `HVA == DVA`。
2. 普通 `malloc/mmap` 申请的内存可通过 `HostRegisterV2(PINNED|MAPPED)` 注册，再查询 DVA。
   官方要求 OS 内核高于 5.10；内核 <=5.10 应使用 `aclrtMallocHost()`。
3. 查询出的 mapped DVA 用于设备 kernel，不能拿它当普通 Host 指针或作为 Runtime memcpy 地址。
4. ACL 的 V2 共享句柄接口接受 `aclrtMallocPhysical()` 返回的物理内存 handle，
   不接受 `aclrtMallocHost()` 返回的普通指针；该接口列出了 950 的进程间共享支持。
   这不等于任意 Host 分配都已被证明可通过该接口共享。

资料：

* [Host 锁页内存使用与地址一致性](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/programug/acldevg/runtime_doc_dev_0006.html)
* [HostRegisterV2 参数与 OS 限制](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2128.html)
* [HostGetDevicePointer 与设备地址使用限制](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1163.html)
* [V2 物理内存共享句柄](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_2006.html)
* [AI Core Error 故障信息解析](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/troubleshooting/troubleshooting_0515.html)

不同版本中 950PR/950DT 的支持表存在差异，务必核对现场 SKU、CANN、驱动版本及实际返回码。
文档没有承诺“同一组共享页在所有进程、所有设备重复注册都一定成功”，该部分必须实测。

## 3. 本次实现

```text
rank0 创建一份 /dev/shm 文件并实际预留物理 backing
    ↓ 通过原有共享池控制组交换文件身份、大小、boot_id、初始化结果
所有 rank 协商共同空闲的 Host VA 区间（不覆盖其他映射）
    ↓ 每个进程把同一 inode MAP_SHARED 到该 Host VA
各进程在自己的 Device 上 HostRegisterV2(PINNED|MAPPED)
    ↓ 每个进程独立 HostGetDevicePointer
公共 HVA：供 CPU / PD 使用
各自 DVA：供本设备 Plan/Transfer/sparse_copy 使用
```

公共 HVA 是显式协商的兼容设计，方便沿用框架广播 rank0 指针的方式；它不表示 DVA 必须相同。
映射尝试从 32 TiB 地址开始、以 1 TiB 为间隔，最多尝试 16 个候选；
使用 `MAP_FIXED_NOREPLACE` 预留，只有本进程已预留的区域才用 `MAP_FIXED` 替换为共享文件映射。
任何 rank 占用某个候选时，全组释放该候选并重试；全部失败则返回错误。

注册后 `get_dva(hva + offset)` 使用本进程 DVA 基址加偏移，不再返回恒等映射。
只允许 rank0 调用分配器；其他 rank 使用 rank0 广播的已分配地址。
该模式是一份 rank0 池，不是每 rank 分配一个虚拟槽的通用对称池。

文件通过 `mkstemp` 创建，权限 0600；导入检查文件类型、uid、inode、设备号、大小和同机 boot_id。
`posix_fallocate()` 先检查容量，避免 tmpfs 稀疏文件在首次访问时因容量不足触发 SIGBUS。
各进程完成映射与注册结果交换后，owner 删除文件名；仍存在的映射保持物理 backing 存活。
每个进程只注销、解除自己的映射；注销失败时保留 backing，避免仍被设备注册的页遭到释放。
初始化普通错误通过集体结果交换传播；进程被杀或网络失败仍依赖现有控制组超时，不能提供故障容错。

所有注册和地址查询发生在初始化阶段，不改变 SIMT LRU 算法，也不增加逐 token Host 调度。
调用方必须在 uninitialize 前结束所有使用该池的 kernel/URMA 请求，并解除 TransferEngine 注册。

## 4. 使用方式和边界

```python
config.scene = offload.Scene.SHARED
config.world_size = tp_size
config.rank_id = tp_rank
config.reserve_size = pool_bytes
config.alloc_size = pool_bytes if tp_rank == 0 else 0
config.store_url = "tcp://127.0.0.1:18500"  # 所有 TP 相同且端口空闲
config.flags = offload.OFFLOAD_FLAG_ACL_SHARED_HOST
```

* 显式选择实验模式；不能与 `GIANT_PAGE` 合用；当前仅接受 A5。
* rank0 `alloc_size` 必须等于 `reserve_size`，其他 rank 为 0；大小沿用 offload 的 GiB 对齐规则，
  单池最多 1 TiB。peer 可以查询 rank0 池中已分配地址的 DVA。
* 所有进程应位于同一台物理机、同一用户、能够访问同一 `/dev/shm` backing。
  文件名相同但不同 IPC mount 不会被当作共享成功。
* OS 内核 <=5.10 明确拒绝本实验模式。不能用 `mlock()` 冒充文档所要求的 ACL 分配路径；
  这种服务器应继续评估 VMM HOST 物理句柄导入方案，或升级至受支持的 OS/驱动组合。
* 必须按实际池大小配置 `/dev/shm`，例如容器使用已评估过的 IPC/共享内存配置。
  默认小容量容器 tmpfs 无法承载数百 GB KV。
* `entry_gather` 依赖整个对称地址空间的统一设备地址算法，本实验明确拒绝注册它，避免静默误读。
  SIMT Transfer 使用每层 DVA 基址，不需要该接口。
* ACL 映射不等于 URMA 注册；URMA 仍须在每个进程注册该进程的 HVA。
* 本次没有把 vLLM 默认 A5 独立池切换为共享池。整网切换还需同步调整池所有权与设备类型判定、
  逐进程 DVA 表、Main KV 分片拉取、写回 owner、传输注册和完成同步。
  Indexer / resident HBM buffer 仍属于各 rank，不共享。
* 地址稳定有利于图捕获，但本分支没有验证 graph replay 或整网 MTP。

## 5. 验证

本地无需 CANN：

```bash
bash test/standalone/run_acl_shared_pool.sh
```

7 项真实双进程 / 模拟 ACL 测试：共享页逐字比较、每进程不同 DVA 与偏移、越界拒绝、
共同 VA 冲突重试、owner 先退出、注销失败重试、设备切换/注册/DVA 查询失败回滚、
分配策略与大小不一致的集体失败。输出明确标记 CPU_ONLY，不能作为 NPU 验证结果。

服务器先正常构建并安装本分支：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 以现场实际安装路径为准
bash script/build_and_pack_run.sh
# 按原部署方式安装 output 下本次新生成的 .run / wheel；确认 Python 没有加载旧包。
python3 -c 'from memfabric_hybrid import offload; print(offload.OFFLOAD_FLAG_ACL_SHARED_HOST)'
```

下面脚本每次只分配一份 1 GiB Host backing，校验其中 256 KiB；父进程不初始化 ACL。
卡号为逻辑卡号，选择空闲设备；测试 `0,0` 可区分跨进程和跨设备这两个条件：

```bash
python3 test/python/acc_offload/probe_acl_shared_pool.py --devices 0,0
python3 test/python/acc_offload/probe_acl_shared_pool.py --devices 0,1
python3 test/python/acc_offload/probe_acl_shared_pool.py --devices 0,1,2,3
```

脚本验证 CPU 写→所有 AIV 读、各 AIV 分片写→CPU/所有 AIV 读，以及已读后缀重写。
每阶段逐字节对照，再输出校验和；发现不一致、初始化失败或子进程异常即失败。
这是通过现有 sparse_copy 做的 AIV 正确性探针，不是 SIMT Transfer、模型输出或性能测试。

可选真实 URMA：额外使用一张独立 P 设备，D 各 rank 只拉取各自分片，之后每个 AIV 读取完整数据。
将地址替换为现场可用于设备 URMA 的本机业务 IP，两类 store 使用不同空闲端口：

```bash
python3 test/python/acc_offload/probe_acl_shared_pool.py \
    --devices 0,1 --urma-source-device 2 \
    --pool-store tcp://127.0.0.1:18500 \
    --bind-ip YOUR_BUSINESS_IP --urma-store tcp://YOUR_BUSINESS_IP:18510
```

这能检验“本机独立 P 进程/HBM→URMA→D 共享池→各 AIV”，尚未覆盖真正两台服务器的 PD 网络。
没有指定 URMA 参数时，PASS 明确写出 URMA not tested。
硬件回传请保留每个阶段日志，以及 `uname -r`、CANN、驱动、SKU 和容器共享内存配置。

### 本机检查结果

* 7 项 CPU_ONLY 测试通过；ASAN/UBSAN 复跑通过。
  本机工具运行于受跟踪环境，LeakSanitizer 的 ptrace 检查无法使用，复跑关闭了 leak 检查；不宣称 LSAN 通过。
* CMake `XPU_TYPE=NONE` 配置及 `acc_offload_objects` 目标编译通过，包含共享入口和 ACL 适配器。
* Python Ruff、语法、CLI 参数检查、Shell 语法、C++ 格式与 diff 检查通过。
* 完整 Host 共享库构建被 clean 基线已有的 `MAP_HUGE_1GB` 常量与本机新 glibc 宏同名冲突阻塞
  （`hybm_conn_based_segment.cpp:444`）。本次没有修改该无关分配器，也没有宣称完整链接通过。
* 没有 A5 设备，未执行 NPU/URMA 探针、图捕获、MTP 或整网推理；未运行完整 UT 套件及 Bazel 构建。

## 6. 后续决策

优先顺序：同卡双进程 → 双卡 → 4 TP → URMA 共注册与分片拉取 → SIMT Transfer → graph/MTP → 全量 PD。
只有这些验证成立，才把默认 A5 独立池改为共享池；不能仅凭 `initialize()==0` 或 HTTP 200 认定成功。
共享池节省 Host 容量与重复 PD 流量，但可能增加 NUMA 远端访问、Host 带宽争用和同步等待，
不保证比每卡本地池快；应分别测容量、端到端吞吐与单步延迟。
