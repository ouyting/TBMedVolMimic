# TBMedVolMimic 开发计划：AI 自驱动的 TB 级医学影像体渲染与 Mimics 类建模系统

## 1. Context

**要解决的问题**：现有工具各管一段。RadiAnt 只在 Windows 上运行，没有已确认发布的曲面重建（CPR）和真 4D 体渲染。Imaris 是商业桌面软件。neuroglancer 能处理 PB 级数据，但只支持 WebGL2，也没有写实渲染和临床影像工作流。Mimics 是封闭的商业套件。项目要做一个局域网 Web 系统（服务端加浏览器客户端），把这四类能力统一到一个流式引擎上，并由 AI 代理全权开发和持续迭代。

**仓库现状**：只有 README.md 和 MIT LICENSE。项目从零开始，所有依赖必须与 MIT 兼容，不链接 GPL/AGPL 组件。

**已确认的决策**

| 决策项 | 结论 |
|---|---|
| Mimic 含义 | Materialise Mimics 类功能：分割、掩膜编辑、三维重建、STL 导出、三维测量、手术规划 |
| 渲染架构 | 混合。浏览器 WebGPU 负责交互式流式渲染；服务端 GPU 负责电影级路径追踪出图、视频导出和弱客户端兜底 |
| AI 底座 | 自建编排器，参考 Hermes Agent 的技能、记忆、定时和网关设计，基于 Claude Agent SDK 和 Claude Code 无头模式 |
| 部署 | 局域网 Linux 服务器，配 NVIDIA GPU、NVMe 热数据层和 NAS 冷数据层，用 Docker Compose 部署；开发机为 Windows 11 |

**默认假设**（可调整）
- 仅限科研使用（RUO），不作为临床诊断器械。所有视图和导出物都带 RUO 标识。
- 客户端以 Chrome/Edge 为主，Firefox 141+ 和 Safari 26 尽力支持。
- 界面支持中英双语。

**预期结果**：先建成一座“AI 软件工厂”，即 Harness 加编排器。再由它按里程碑自动交付产品，并在人类每周设定方向的前提下持续自我迭代。

---

## 2. 总体原则

1. **先建工厂，再造产品。** M0 只做 Harness，任何产品功能都走“验收用例 → 代理实现 → 自动验证”的闭环。
2. **一切皆多尺度分块数据。** DICOM、显微、电镜数据都映射为统一的 Scene/Layer/Source 模型，由同一个引擎渲染。
3. **能原地读就不转换。** PB 级数据无法重新编码。
   - OME-Zarr 和 neuroglancer precomputed 直接服务。
   - .ims（分块 HDF5）和分块 TIFF 生成 kerchunk 风格的引用清单，浏览器直接按偏移读取。
   - 只有 DICOM 和布局不佳的数据才转换为 OME-Zarr。
   - 原始数据始终是真值来源。
4. **数据面静态化。** 块数据由 Caddy 直接按字节范围提供，Python 服务不在热路径上。
5. **客户端和服务端渲染同源。** 服务端用 Node 加 Dawn 运行同一套 TypeScript/WGSL 引擎，保证两端画面一致。
6. **机器可验证优先。** 每个功能都有带度量的验收测试。功能状态只能由 CI 根据测试结果写入，代理不能自己标记“已完成”。

---

## 3. 系统架构

### 3.1 服务（Docker Compose）

| 服务 | 技术 | 职责 |
|---|---|---|
| `edge` | Caddy | TLS、HTTP/2/3、字节范围块服务（NVMe 热层、NAS 只读冷层）、SPA 托管、COOP/COEP 头（SharedArrayBuffer 需要） |
| `api` | FastAPI + pydantic v2 + SQLAlchemy | 数据目录、认证与 RBAC、任务、WebSocket、DICOMweb 子集、用户反馈 |
| `dicom-scp` | pynetdicom | C-STORE 接收、C-ECHO；作为 C-FIND/C-MOVE 客户端对接医院 PACS |
| `worker-cpu` | tensorstore、ngff-zarr、kerchunk、pydicom/highdicom、VTK、manifold3d、meshoptimizer | 导入、金字塔生成、引用清单、逐块统计、网格化、STL 与 DICOM SEG 导出 |
| `worker-gpu` | MONAI、nnU-Net v2、TotalSegmentator、SAM-Med3D-turbo/MedSAM2、Cellpose/StarDist/micro-sam、itk-elastix | AI 分割、配准、GPU 形态学运算 |
| `render-node` | Node + Dawn（`webgpu` npm 包）+ OIDN | 路径追踪高清出图和视频导出（LGPL 版 ffmpeg）；弱客户端的图像流兜底 |
| `synth` | Python | 程序化生成的虚拟多尺度 OME-Zarr，带解析真值，虚拟尺寸可达约 1 PB，不占磁盘 |
| 基础设施 | Postgres + Procrastinate 任务队列、OpenTelemetry、VictoriaMetrics/VictoriaLogs、Jaeger | 元数据、任务、可观测性 |

**许可证取舍**
- 不用 Orthanc（GPL/AGPL）、MinIO（AGPL）、Grafana 栈（AGPL）、Bio-Formats（GPL）和 OpenSlide（LGPL），分别以 pynetdicom、文件系统、Victoria 栈和 tiffslide 替代。
- AI 权重单独登记许可证。nnInteractive、VoxTell、VISTA3D 的权重仅限非商用，TotalSegmentator 的部分子任务需要商用许可。

### 3.2 统一数据模型

- **Scene**：世界坐标系加单位，包含多个 Layer。Layer 类型有 image、label、mesh、skeleton、annotation、points、surface。
- **Source**：多尺度数组，轴取自 (t, c, z, y, x)，另有 dtype、编解码链、每级 4×4 仿射矩阵和存储类型（zarr-http、precomputed、reference、synthetic）。
- **DICOM 映射**
  - 序列按方向、堆叠、回波和期相拆分成体数据。时相映射为 t 轴，b 值和回波映射为 c 轴。
  - 像素以无损 int16/uint16（blosc-zstd）存储并保留 rescale 参数。PET 保留 SUV 因子。
  - 保留 Frame of Reference UID，供融合使用。
  - 机架倾斜或层距不均的序列会重采样，并在界面上标注。
- **OME-Zarr 版本**：读取 0.4、0.5 和 0.6。写入默认用 0.5（Zarr v3 加分片），生态兼容性最好。0.6（2026-09 发布，引入含仿射的坐标变换）在工具链支持后启用。在此之前，完整 DICOM 几何存放在命名空间属性 `tbmv` 中。
- **显微数据**：HCS 板和孔用 OME-Zarr plate 规范。标签为 uint64，在 GPU 上以 rg32uint 存储（WGSL 没有 u64）。
- **逐块统计旁路金字塔**：每块存最小值、最大值和 16 档直方图。它驱动空域跳过、自动窗宽窗位和路径追踪的 majorant 网格。

### 3.3 客户端引擎（out-of-core 流式渲染）

参考 neuroglancer 的块管理器、Hadwiger 2012 的可视化驱动虚拟内存、Herzberger 2024 的 Residency Octree（WebGPU 实现），以及开源项目 kiln-render 和 webgpu-volume-pathtracer。移植代码前先核对许可证。

- **块尺寸分层**：网络分片 256³，内块 64³，按 Morton 序排列，便于合并范围请求。GPU 砖块初定 32³ 加 1 体素边框，最终尺寸由基准测试和架构决策记录（ADR）确定。
- **砖块图集**：每种格式一个 3D 纹理，每维不超过 2048，数量不够时开多个图集。
- **页表**：存储缓冲区里的开放寻址哈希。键为 (layer, level, t, c, bx, by, bz)，值为图集槽位。未命中时回退到更粗一级。最粗的几级常驻显存。
- **反馈缓冲**：着色器用原子操作追加未命中和已使用的键，用 3 帧环形 `mapAsync` 回读。2D 视图和 MPR 用解析方式计算所需砖块，体渲染（DVR/MIP）用反馈方式。
- **细节层级（LOD）**：目标约每屏幕像素 1 个体素，交互时放宽到约 2 个，带滞回。
- **请求调度器**
  - 优先级依次为：常驻、可见、预取（相邻层面、下一时相）、投机。
  - 并发约 64 个请求，按吞吐自适应；过期请求用 AbortController 取消。
  - 解码在 WASM Worker 池中进行（numcodecs.js、cornerstone codecs、draco3d、compressed_segmentation）。
  - 每帧上传预算不超过约 4 ms。
- **三级缓存**：GPU 缓存（独显默认 1.5 GiB，集显 512 MiB），内存中的已解码数据（2 GiB），可选的 OPFS 压缩缓存。按反馈时间戳做 CLOCK 淘汰。
- **4D 与多通道**：t 进入页表键。电影播放时按“每体字节数 ÷ 带宽 ≤ 1/fps”选择 LOD。只请求打开的通道，单遍最多合成约 8 个通道。
- **分割与对象**
  - 标签体用最近邻采样，按 ID 哈希着色，选择集放在 GPU 哈希集合里。
  - 网格采用 neuroglancer multilod Draco 格式和八叉树 LOD。
  - 骨架和纤丝用实例化胶囊体绘制，spots 用 impostor 球体。
  - 网格先光栅化，其深度用来截断体渲染光线。
- **掩膜编辑**：客户端记录稀疏脏砖块增量和操作日志并提交服务端。服务端持有带版本的权威掩膜。重运算交给 GPU worker。
- **渲染通道**
  - 切片、斜位 MPR、曲面 MPR（CPR）。
  - 厚层 MIP/MinIP/AvgIP。
  - 带空域跳过和提前终止的 DVR 和等值面。
  - 电影级路径追踪：delta/ratio tracking 加 majorant 网格、HDR 环境光重要性采样、rgba32f 渐进累积。客户端用 à-trous 降噪，服务端用 OIDN。

**WebGPU 限制与应对**

| 限制 | 应对 |
|---|---|
| 3D 纹理每维最大 2048 | 多图集加软件页表 |
| 缓冲区默认 256 MiB，存储绑定默认 128 MiB | 按默认值设计，适配器允许时再申请更大 |
| float32 可过滤、shader-f16、timestamp-query 均为可选特性 | 运行时检测。16 位数据回退为 r16float 加逐数据集缩放，或 r16uint 加手工三线性插值。测量一律取原始值 |
| 无法查询显存 | 用 `pushErrorScope('out-of-memory')` 自适应预算；`device.lost` 后整体重建 |
| Windows 约 2 s 看门狗 | 单次 dispatch 不超过 16 ms，分块渐进渲染 |
| Linux 上 Chrome 对 NVIDIA 仅在 Wayland 默认开启 | CI 固定 Chrome 版本，加 `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan`；Windows 开发机作为第二个 D3D12 运行器 |

### 3.4 技术栈

- **前端**：TypeScript、Vite、React（仅界面外壳）、Zustand；pnpm 加 Turborepo；zarrita.js；着色器用 WGSL/WESL，由 naga 校验。
- **服务端**：Python 3.12+，用 uv 管理工作区。
- **AI 编排器**：Python、Claude Agent SDK、Claude Code 无头模式、SQLite FTS5、APScheduler、Playwright。

---

## 4. AI 自驱动研发体系（Harness Engineering 加 Hermes 式编排器）

### 4.1 采纳的 Harness Engineering 实践

- **OpenAI（2026-02）**
  - AGENTS.md 控制在约 100 行，只作地图，指向结构化的 `docs/` 知识库。
  - 执行计划写入仓库。
  - 每个 git worktree 启动独立的应用栈，代理可以查询日志、指标和链路追踪，并通过 CDP/Playwright 驱动浏览器。
  - 自定义 lint 的报错信息里写明修复方法；用结构测试约束分层依赖。
  - 设 doc-gardening 和垃圾回收代理。
  - 代理之间互审，直到所有评审满意（Ralph 循环）。
- **Anthropic（2025-11）**：初始化代理生成 `features.json`（全部初始为 failing）、`init` 脚本和 `progress.md`。编码代理每次会话只做一个功能，先跑冒烟测试再开发。
- **Anthropic（2026-03）**：Planner、Generator、Evaluator 三角色分离，冲刺开始前商定“冲刺契约”（验收标准）。评估必须由独立上下文完成，因为自评过于宽松。长任务用上下文重置加文件交接。
- **Böckeler（2026-04）**：控制手段分为引导（事前）和传感器（事后），各自又分计算型和推理型。便宜的检查放在前面，贵的放在后面。

### 4.2 Hermes 组件映射

| Hermes Agent 组件 | 本项目实现 |
|---|---|
| 代理循环（分层提示加缓存断点） | Agent SDK `query` 循环。稳定前缀是 AGENTS.md 加角色提示，以利用提示缓存 |
| 工具注册表加危险命令审批 | 进程内 MCP 服务器：`mcp_app`、指标/日志/追踪查询、数据集、worktree 管理；PreToolUse 钩子负责审批 |
| 技能（SKILL.md，代理用 skill_manage 自建和修补） | 正式技能放 `.claude/skills/`，候选放 `harness/skills-staging/`；复盘时生成候选，评测后通过 PR 晋升 |
| 记忆（MEMORY.md/USER.md 有字符上限，会话开始时冻结快照） | `harness/memory/MEMORY.md`（项目经验，≤8 KB）和 `STEERING.md`（人类方向与偏好，≤4 KB），会话开始时冻结加载 |
| session_search（FTS5） | SQLite FTS5 索引全部会话记录、PR 和评审意见，承担长期记忆 |
| Cron（任务可绑定技能） | APScheduler 负责夜间迭代、每日文档园丁、每周差距分析、许可证与漏洞审计、记忆整理 |
| 网关（多平台适配器） | Web 控制台、Webhook、企业微信/飞书/钉钉/邮件适配器、应用内反馈按钮 |
| delegate_task 子代理 | SDK 子代理，各角色独立上下文，只回传摘要 |
| 终端后端 | 本地（Windows，需人工值守）、Docker 沙箱（Linux 服务器默认）、SSH（GPU 测试） |
| 轨迹导出 | 会话轨迹归档为 JSONL，用于复盘和评估 Harness 本身 |

### 4.3 代理角色与模型分级

模型分级写在 `harness/policies/models.yaml` 中，可随时调整。

| 角色 | 职责 | 建议模型 |
|---|---|---|
| Planner | 从工作队列挑选任务，编写执行计划和冲刺契约 | claude-fable-5-1 |
| Implementer | 在独立 worktree 和独立栈中实现 | 引擎/着色器难题用 claude-fable-5-1；常规功能用 claude-sonnet-5-5 |
| Evaluator | 独立上下文，用 Playwright 在真实 GPU 上驱动应用，输出 JSON 裁决 | claude-opus-5-5 |
| Reviewer | 对照文档和 ADR 评审 diff，最多 3 轮 | claude-opus-5-5 |
| Researcher | 维护与 RadiAnt/Mimics/Imaris/neuroglancer 的功能差距矩阵；只读，无写工具 | claude-sonnet-5-5 |
| Gardener | 文档新鲜度、技术债、重复技能合并、重构 PR | claude-haiku-4-5-20251001 |
| Triage | 日志摘要、失败分类、反馈归并 | claude-haiku-4-5-20251001 |

### 4.4 迭代循环

```
触发：定时 / CI 失败 / 性能回退(p95 > 基线 5%) / 应用内反馈 / 人类指令
  → Observe：从失败功能、CI、性能、反馈、园丁发现、差距矩阵中汇总并打分，形成工作队列
  → Plan    ：Planner 编写 docs/exec-plans/active/NNNN-*.md 和冲刺契约
  → Execute ：Implementer 在 worktree 加独立 compose 栈中实现
  → Verify  ：Evaluator 在独立上下文中用真实浏览器和 GPU 验证，输出 JSON 裁决
  → Review  ：Reviewer 评审，最多 3 轮
  → Merge   ：通过 G1–G6 后进入合并队列；受保护路径还需 G7 人工审批
  → Learn   ：复盘，更新记忆、候选技能、lint 规则和文档
同一事项连续失败 3 轮即标记 blocked，附摘要上报人类。
```

**核心数据文件**
- `harness/state/features.json` 的 schema 在 `schemas/` 中。`status` 和 `evidence` 字段只允许 CI 的 status-sync 机器人写入，`acceptance` 字段受保护。示例：
  ```json
  {"id":"F-M2-014","milestone":"M2","area":"engine-core/scheduler",
   "title":"1 PB 合成数据上的由粗到细 MPR",
   "acceptance":[{"type":"e2e","test":"tests/e2e/mpr_pb.spec.ts::coarse_first",
                  "metric":{"ttfi_ms":{"max":500}}}],
   "depends_on":["F-M2-003"],"gap_ref":"neuroglancer:multiscale",
   "status":"failing","evidence":null}
  ```
- `harness/state/progress.md`：只追加。每条记录会话、角色、完成内容、下一步、阻塞项和成本。
- 执行计划模板包含：目标、背景链接、非目标、设计、步骤清单、验收、进度、决策日志、意外发现、结果。
- `docs/quality/QUALITY_SCORE.md`：每晚生成。按领域给出 A–F 评分，依据是验收通过率、flaky 率、覆盖率、lint 债、性能余量和文档新鲜度。
- `docs/product/gap-matrix.csv`：四款参考产品的功能逐行列出，标注 must/should/could 优先级，并映射到功能 ID。

**质量门禁**

| 门禁 | 内容 |
|---|---|
| G1 静态 | lint、类型、分层、许可证白名单、gitleaks、PHI 扫描 |
| G2 | 单元测试和结构测试 |
| G3 | 基于 compose 栈的集成测试 |
| G4 | GPU 运行器上的 Playwright e2e 和金标准图像。图像比对用 NVIDIA FLIP；2D 数值用回读比对，误差不超过 ±1 LSB |
| G5 | 性能预算 |
| G6 | Evaluator 通过且 Reviewer 批准 |
| G7 | 受保护路径的人工审批 |

**渐进放权**

| 级别 | 权限 |
|---|---|
| L0 | 只提建议 |
| L1 | 自动合并文档、测试、lint 和杂务 |
| L2 | 自动合并非受保护包内的功能 |
| L3 | 代理自主提出目标 |
| L4 | 持续运行，人类只设目标和预算 |

晋级条件：在当前级别合并至少 50 次，回滚率不超过 5%。任何回滚事故或安全事件都会自动降级。

**安全护栏**
- **受保护路径**：harness 本身、CI、策略、阈值、金标准图像、`acceptance` 字段、`.claude/hooks`。三重拦截：SDK PreToolUse 钩子、CODEOWNERS、CI diff 检查。这样代理无法削弱自己的测试。
- **隔离执行**：Implementer 在无 Docker socket 的 rootless 沙箱中运行。出网经允许列表代理，只放行包仓库、Anthropic API 和 git。git 令牌只能推送 `agent/*` 分支。
- **数据分级**：数据分为 public、synthetic、restricted、phi 四级，代理只能挂载 public 和 synthetic。DICOM 标签和网页内容一律视为不可信输入，防范提示注入。
- **紧急停止**：创建 `STOP` 文件或在网关发送 `/stop`，10 秒内停止全部代理。PostToolUse 钩子写 JSONL 审计日志。
- **公开仓库**：如果远程仓库是公开的，自托管 GPU 运行器只执行受信分支和合并队列的任务，从不运行 fork 的 PR。

**成本控制**
- 每个任务限制轮数和美元，依据结果消息中的 `total_cost_usd` 执行。另设每日和每周上限，并配熔断。
- 用稳定前缀提高提示缓存命中；低价值工作交给便宜模型。
- 考核指标：每个合并 PR 的成本，每个新转为 passing 的功能的成本。

**学习与修剪**
- 复盘时提出记忆条目（去重，有上限）、候选技能和文档修正。
- 同一条评审意见出现 3 次以上，就固化为 lint 规则。
- 候选技能先经触发评测和回放评测，再晋升为正式技能。
- 45 天未使用或成功率低于基线的技能会被修剪。正式技能上限 60 个。

**让代理看见应用**
- 浏览器、API 和 worker 全部接入 OpenTelemetry。
- `window.__TBMV_DEBUG__` 暴露帧时间、常驻砖块、缓存命中率、在途请求和显存估计。
- `mcp_app` 工具集提供：`up`/`down`、`open_scene`、`wait_quiescent`、`screenshot`、`engine_stats`、`query_metrics`、`query_logs`、`get_trace`、`run_bench`、`compare_golden`。
- 测试模式固定随机种子，保证渲染确定性。

**人类的角色**
- 每周做一次方向评审：看质量评分、差距矩阵、预算和上报的阻塞项，并更新 `STEERING.md`。
- 审批受保护路径的变更。
- 抽查 10% 的合并。
- 提供需求和反馈。

---

## 5. 仓库结构与机械化约束

```
AGENTS.md (≤120 行地图)   CLAUDE.md (@AGENTS.md)   justfile   init.py
.claude/{settings.json, hooks/, skills/}
docs/{index.md, ARCHITECTURE.md, design-docs/, adr/, product/{module1-radiology.md,
      module2-microscopy.md, gap-matrix.csv}, exec-plans/{active,completed}/,
      quality/QUALITY_SCORE.md, tech-debt.md, golden-principles.md, references/, generated/}
schemas/                      # JSON Schema: scene/layer/source、feature-list、exec-plan，生成 TS 和 pydantic 代码
packages/                     # TS: types → math → formats → codecs-wasm → engine-core → engine-webgpu → tools → ui ; testkit
apps/{web, render-node}
server/                       # Python uv 工作区: tbmv_core → tbmv_formats → tbmv_{dicom,ingest,mesh,ai,synth} → tbmv_{api,workers}
deploy/{compose.yaml, compose.worktree.yaml, caddy/, otel/}
tests/{e2e, golden, perf, structural}
datasets/registry.yaml        # URL、许可证、校验和、数据分级
harness/{orchestrator, roles, tools, gateway, scheduler, memory, skills-staging, evals, policies, state}
tools/lint/{eslint-plugin-tbmv, py_checks}
```

**分层规则**：每层只能依赖它左侧的层。
- `engine-core` 不接触 GPU 和 DOM，可以用模拟的网络和 GPU 做确定性测试。
- `engine-webgpu` 是唯一可以使用 `GPU*` 类型的包，同样不接触 DOM，因此能在 Node 中运行。
- 只有 `ui` 和 `apps` 可以引入 React。
- Python 侧的层次是 Types → Config → Repo → Service → Runtime。

**强制工具**
- TypeScript 用 tsc 项目引用、dependency-cruiser 和 eslint-plugin-tbmv。自定义规则的报错里写修复方法，例如“engine 内禁止直接 fetch，改用 RequestScheduler.enqueue，见 docs/…”。另用 knip 查死代码。
- Python 用 ruff、pyright strict、import-linter 和 deptry。
- 着色器用 naga 校验，Dawn 编译，并检查绑定组反射结果与 TS 端一致。
- 全局还有：lockfile 许可证白名单、gitleaks、PHI 扫描、lychee 链接检查、单文件不超过 600 行、每个功能 ID 都有对应测试、lefthook、`.gitattributes` 强制 LF、Windows 开启长路径。

---

## 6. 里程碑与自动验收门槛

- **参考硬件**：客户端为 RTX 4060 级显卡加 Chrome，分辨率 1080p；弱客户端为集显笔记本；网络为千兆局域网（TB 级场景建议万兆）。
- **工期**：下表工期仅为估计。进度以门禁为准，不以日期为准。M4 和 M5 并行，到 M5 总计约 33 周。

| 里程碑 | 周期 | 主要内容 | 退出门槛（全部自动验证） |
|---|---|---|---|
| **M0 工厂** | ~3 周 | Harness、CI、护栏、合成数据、首个 WebGPU 画面 | `just check` 在 Windows 和 Linux 上全绿。代理在无人改代码的情况下把 3 个种子功能从 failing 变为 passing。红队 10 次试图削弱受保护文件全部被拦截。紧急停止不超过 10 s，预算超支不超过 5%。GPU 运行器上报 NVIDIA 适配器。synth 单块 p95 < 20 ms |
| **M1 数据面 + 2D** | ~6 周 | DICOM 导入与索引、2D 阅片、窗宽窗位、测量、OME-Zarr/WSI 2D 金字塔 | 1000 层 CT 的文件夹导入和 C-STORE 导入均不低于 50 张/s。HU 与 pydicom 逐位一致。6 种方位体模的方位标签全部正确。冷启动首图 < 1 s，滚动 ≥ 60 fps。50 GB 2D 金字塔平移 ≥ 60 fps。测量误差 < 0.5 px |
| **M2 out-of-core 3D** | ~8 周 | 砖块图集、页表、反馈、调度器；MPR、斜位、MIP、DVR；4D；多通道 | 在 1 TB 实体数据和 1 PB 虚拟数据上：粗图 < 500 ms，停止操作后 3 s 内收敛，p95 帧时间 ≤ 33 ms。30 分钟浸泡测试中显存不超预算，零设备丢失。各级 LOD 采样值与解析真值一致。4D 播放 20×(512²×300) 不低于 10 体/s。40 通道中开 8 个时 ≥ 30 fps。集显 ≥ 15 fps |
| **M3 电影级渲染** | ~6 周 | 客户端和服务端路径追踪、HDR 光照、降噪、视频导出 | 客户端 5 s 内达到 FLIP ≤ 0.1（参考图为服务端 16k spp）。交互时 ≥ 20 fps。服务端 4K 静帧 < 60 s。同种子下两端 FLIP ≤ 0.02。单次 dispatch 不超过 50 ms |
| **M4 模块一 + Mimics** | ~10 周 | CPR、PET-CT/MR 融合、配准、SUV、同步与挂片协议、PACS 对接；阈值、区域生长、画笔、AI 分割、网格、STL、三维测量、中心线 | DIR-Lab 配准 TRE ≤ 2 mm。TotalSegmentator 全身分割 < 3 min，Dice 与公开值相差不超过 0.02。512³ 画笔 < 50 ms 并可撤销。网格水密，与掩膜的 Hausdorff 距离 ≤ 1 体素。差距矩阵中模块一的 must 项通过率 ≥ 80% |
| **M5 模块二** | ~10 周（与 M4 并行） | .ims、OME-TIFF、SVS/NDPI、precomputed（分片、分割、网格、骨架、注释）；HCS 板视图；Surfaces/Spots/Filaments；时序；MICrONS 连接组视图 | MICrONS minnie65 选中神经元后 2 s 内出网格。384 孔板总览 < 3 s。100 万个 spots 渲染 60 fps。100 时相光片数据不低于 5 体/s。差距矩阵中模块二的 must 项通过率 ≥ 80% |
| **M6 持续进化与加固** | 持续 | 多用户、OIDC/RBAC、审计、备份，升级到 L3 自治 | 10 个并发用户时块请求 p95 < 50 ms。备份恢复演练通过。L3 自治下回滚率 ≤ 5%。人类每周只做方向评审 |

**测试数据**
- 合成数据：synth 服务提供的 1 PB 虚拟体。
- 放射影像：TCIA 的 4D-Lung（CC BY 3.0）和 FDG-PET-CT-Lesions。
- 显微与电镜：IDR OME-Zarr 0.4/0.5 样例（含 HCS plate）、OpenOrganelle、FlyEM hemibrain、MICrONS 公开 precomputed 桶。

---

## 7. M0 首批任务（批准后由我在本仓库直接执行，之后交给 Harness）

1. **仓库卫生**：`.gitattributes`、`.editorconfig`、长路径、CODEOWNERS、README 加 RUO 声明、NOTICE。
2. **知识库**：AGENTS.md、CLAUDE.md、`docs/` 骨架；ADR-0001 至 0006 记录本计划中的决策；执行计划和 ADR 模板。
3. **工具链**：pnpm 加 Turborepo、uv 工作区、justfile、lefthook、幂等的 `init.py`（Windows 和 Linux 通用）。
4. **包骨架**：所有 TS 和 Python 包各带一个测试，配好 TS 项目引用。
5. **强制规则 v1**：dependency-cruiser、eslint 及 3 条自定义规则、ruff、pyright、import-linter、knip、deptry、许可证白名单、gitleaks、结构测试。
6. **Schema 与代码生成**：feature-list、exec-plan 和 scene/layer/source v0 的 schema，生成 TS 和 pydantic 代码，并检查生成物是否最新。
7. **功能清单**：`features.json` 录入 M0/M1 功能，全部为 failing；约定 `@feature` 测试标签；status-sync 机器人；差距矩阵初版。
8. **CI**：托管运行器跑 G1–G3。自托管 GPU 运行器只跑受信分支，执行 Playwright WebGPU 冒烟测试和 Dawn-in-Node 着色器测试。
9. **最小栈**：compose 包含 Caddy（范围请求、COOP/COEP）、FastAPI health、Postgres 和可观测性栈；worktree 覆盖层负责端口分配。
10. **合成数据与首个画面**：`tbmv_synth` v0 加小型实体夹具。Web 端用 WebGPU 渲染一张合成切片，并用 FLIP 做金标准比对。
11. **编排器核心**：代理循环、角色提示、工具注册、worktree 管理器、SQLite 运行记录和 FTS5、审计日志。
12. **护栏**：settings.json 拒绝规则、受保护路径钩子、Docker 沙箱、出网代理、预算、紧急停止。每项都有测试。
13. **代理可见性**：`mcp_app` 工具集和 `window.__TBMV_DEBUG__`。
14. **记忆、技能、调度、网关**：带上限的 MEMORY.md 和 STEERING.md；技能从候选到正式的晋升流程，附 3 个种子技能；APScheduler 夜间任务；网关 v0（CLI、Webhook、应用内反馈入口）。
15. **M0 演练**：代理在 L1 级别完成 3 个种子功能，跑红队套件并复盘。结果记入 QUALITY_SCORE，Harness 晋升到 L1。

**需要你提供的资源**（在第 8、9、11 步时需要）
- Linux GPU 服务器的 SSH 访问，并装好 Docker 和 NVIDIA Container Toolkit。
- Anthropic API key，供 Agent SDK 使用。
- GitHub 仓库管理权限，用于设置分支保护、合并队列和 Actions 运行器。
- 一个通知渠道，企业微信、飞书、钉钉或邮件任选其一。

在这些资源到位之前，所有不依赖它们的任务先在 Windows 本机完成。

---

## 8. 主要风险与对策

| # | 风险 | 对策 |
|---|---|---|
| 1 | WebGPU 跨平台差异大，Linux NVIDIA 无头 CI 脆弱 | 固定 Chrome 版本和启动参数；着色器测试用 Dawn-in-Node；Windows 机作为 D3D12 第二运行器；限额取保守值；支持设备丢失后恢复 |
| 2 | out-of-core 引擎卡顿或抖动 | `engine-core` 为纯 TS，用模拟网络和 GPU 配合合成真值测试；借鉴成熟设计；从 M0 起设性能预算 |
| 3 | 范围过大，相当于同时做四款成熟产品 | 差距矩阵分级；做薄的垂直切片；两个模块共用一个引擎；没有验收用例的功能不开工 |
| 4 | 代理钻测试空子或削弱测试 | 功能状态只由 CI 写入；受保护路径；独立 Evaluator；每周抽样做变异测试（Stryker/mutmut）；人工抽查 10% |
| 5 | 代码熵增和许可证污染 | 带修复指引的 lint、园丁代理、质量评分、ADR、许可证门禁 |
| 6 | PHI 泄露或安全问题 | 导入时分级，代理不接触受限数据；出网代理；密钥扫描；GPU 运行器只跑受信任务；TLS 和认证 |
| 7 | 被误用于临床 | 全局 RUO 标识；每个派生对象都带来源信息（模型、版本、参数）；体模验证套件；左右方位测试；模型权重许可证登记 |
| 8 | 成本失控或被单一厂商锁定 | 设上限和熔断，利用缓存，便宜模型做分诊；所有状态都是仓库中的普通文件，人类随时可以接管 |

---

## 9. 验证方式

- **M0 完成的判据**：在 Windows 和 Linux 上执行 `just init` 和 `just check` 全部通过。然后运行编排器演练，观察 3 个种子功能是否由 CI 自动翻转为 passing。
- **护栏验证**：运行红队用例，内容包括修改受保护文件、越权推送、出网访问和超预算。还要验证 `STOP` 能否在 10 秒内生效。
- **产品验证**：每个里程碑的门槛由 `tests/e2e`、`tests/golden` 和 `tests/perf` 自动执行，结果汇总到 QUALITY_SCORE.md。人类每周只需看这份报告和差距矩阵。
- **端到端实测**：打开 `http://<server>/`，分别载入合成 1 PB 体、TCIA 4D-Lung 和 MICrONS 公开数据，检查 2D、MPR、DVR 和路径追踪画面，以及调试面板里的指标。
