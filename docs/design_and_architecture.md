# AlphaGPT (AlphaJungle A-Share Edition) - 项目与架构设计文档

## 1. 项目目标 (Project Goal)

本项目旨在构建一个**基于大语言模型 (LLM) 与蒙特卡洛树搜索 (MCTS) 的自动化因子挖掘系统**，专注于 **中国 A 股市场**。

**核心变革：**
- **大脑升级**：从传统的强化学习 (RL) 生成 Token 序列，转变为利用 LLM 的金融先验知识“思考”并生成可解释的 Python 数学公式。
- **搜索算法**：引入 MCTS 算法，通过回测反馈构建搜索树，系统性地探索因子空间，避免 RL 的随机游走。
- **市场适配**：完全剥离原有的 Solana/Crypto 数据源，适配 A 股行情的 Tensor 数据结构。

## 2. 设计思路 (Design Philosophy)

### 2.1 核心流程：严格对齐 Alpha Jungle 论文
1.  **Selection (选择)**:
    *   MCTS 使用 UCT 改进算法选择节点。
    *   **关键点**: 不仅选择叶子节点，也可以选择中间节点进行“再优化”。

2.  **Expansion (扩展 - 论文核心创新)**:
    *   **维度诊断**: 系统分析当前节点的五维得分（IC, RankIC, 换手率, 稳定性, 多样性）。
    *   **目标锁定**: 根据得分短板，概率性选择一个**“主攻方向”**（例如：当前因子 IC 很高但波动大，系统选中“Stability”作为优化目标）。
    *   **双步生成**:
        *   Step 1 (Idea): LLM 接收优化目标，生成自然语言的**改进建议** (Refinement Suggestion)。
        *   Step 2 (Code): LLM 根据建议，生成具体的 Python 公式代码。
    *   **FSA 约束**: 强制 LLM 规避 Alpha Zoo 中出现频率过高的“烂大街”子树结构，逼迫探索新颖结构。

3.  **Evaluation (回测与评分)**:
    *   **相对评分 (Relative Ranking)**: 不使用绝对 IC 值。新因子的得分取决于它在 **Alpha Zoo (精英因子库)** 中的排位。
    *   **LLM 风控官**: 引入专门的 LLM 步骤，阅读生成的公式，定性打分 (0-10) 判断是否存在 **过拟合 (Overfitting Risk)**（如参数过多、逻辑牵强）。
    *   **综合打分**: 综合 相对排名分 + LLM 风控分 = 最终 Reward。
    
4.  **Backpropagation (反向传播)**:
    *   将最终 Reward 更新回树的路径上，指引后续搜索。

### 2.2 技术栈选择
- **计算核心**: PyTorch (利用 GPU 进行矩阵化回测，替代传统 Event-Driven 回测)。
- **推理核心**: OpenAI API / DeepSeek API (负责逻辑生成)。
- **执行方式**: Python `eval()` 动态执行 (替代原有的 StackVM，提升可读性)。
- **数据源**: CSV/Parquet (本地 A 股数据)。

---

## 3. 文件结构详解 (File Structure & Roles)

以下是重构后的推荐文件结构及其详细实现逻辑。

### 根目录
- **`.env`**: 配置文件。存储 OpenAI/DeepSeek API Key，数据库连接串等敏感信息。
- **`requirements.txt`**: 依赖列表 (torch, pandas, numpy, openai, etc.)。

### 模块：`model_core/` (核心算法层)

这是本项目的“心脏”，负责生成和计算因子。

#### 1. `model_core/mcts_agent.py` (新增，核心中枢)
- **作用**: 实现 MCTS 搜索算法的主循环。
- **实现逻辑**:
    - 维护一棵树 `Tree`，每个 `Node`包含一个公式字符串和它的统计分（访问次数 N，总分 Q）。
    - `select_node()`: 使用 UCB1 公式找到最有潜力的叶子节点。
    - `expand_node()`: 构建 Prompt，调用 `llm_client.py`，要求 LLM 基于当前节点公式生成变体。
    - `backpropagate()`: 更新路径上所有祖先节点的统计值。

#### 2. `model_core/llm_client.py` (新增，LLM 接口)
- **作用**: 封装与 DeepSeek/OpenAI 的交互。
- **实现逻辑**:
    - `generate_factor(context)`: 发送 Prompt。Prompt 中包含：现有算子列表、可用字段 (open, close...)、以及“优化方向”（如：降低换手率）。
    - 包含重试机制和 JSON 解析器，确保 LLM 输出的是合法的公式字符串。

#### 3. `model_core/executor.py` (新增，替代 `vm.py`)
- **作用**: 安全且高效地执行 LLM 生成的字符串公式。
- **实现逻辑**:
    - 维护一个全局上下文 `Context`，预先注入所有算子函数 (ADD, RSI...) 和基础数据 Tensor。
    - 使用 Python 的 `eval(formula_str, globals=None, locals=Context)` 直接计算结果。
    - 捕获 `ZeroDivisionError` 或 `Inf`，保证 Tensor 数值稳定性。

#### 4. `model_core/ops_lib.py` (改造自 `ops.py`)
- **作用**: 定义基础算子库。
- **实现逻辑**:
    - 这是一个纯函数库。所有函数输入均为 `torch.Tensor`，输出也为 `torch.Tensor`。
    - 包含：`ts_delay` (时序滞后), `ts_mean` (滚动平均), `rank` (截面排序), `correlation` (滚动相关) 等。
    - **关键点**: 所有操作必须支持 Batch 运算 (Shape: `[num_stocks, time_steps]`)。

#### 5. `model_core/data_loader.py` (改造为 A 股版)
- **作用**: 将 A 股 CSV 数据加载到 GPU 显存中。
- **实现逻辑**:
    - 读取 CSV，进行 `pivot` 操作，转变成 `(股票数, 时间)` 的 2D 矩阵。
    - 处理停牌数据（`NaN` 填充）。
    - 计算 `Target`（未来 N 日收益率），用于后续计算 IC。

#### 6. `model_core/backtest.py` (保留并优化)
- **作用**: 判卷老师。给公式打分。
- **实现逻辑**:
    - 输入：`Executor` 算出的因子矩阵 `Factor(T, N)`。
    - **IC 计算**: 计算 `Factor` 与 `Target` 在每个时间切片上的皮尔逊相关系数。
    - **多头收益**: 选取每日前 10% 分数的股票，计算其平均收益减去交易成本（换手率 * 手续费）。
    - 输出：综合得分 `Reward = IC * w1 + Sharpe * w2 - Turnover * w3`.

---

### 模块：`data_pipeline/` (数据工程层，简化版)

由于不再需要实时抓取 Solana 链上数据，此模块极大简化。

#### 1. `data_pipeline/csv_processor.py`
- **作用**: 数据清洗和格式化。
- **实现逻辑**:
    - 将从 Tushare/聚宽下载的原始这一堆 CSV 清洗成标准格式（Date, Code, Open, High, Low, Close, Vol, Amount）。
    - 剔除上市时间不足的股票或 ST 股。

---

### 模块：`strategy_manager/` (策略管理层 - 暂缓)

目前阶段主要集中在“挖掘”，实盘管理层可暂缓，或仅保留简单的信号生成器。

---

## 4. 数据流向图 (Data Flow)

1. **初始化**: `DataLoader` 读取 A 股 CSV -> 生成基础张量字典 `{OPEN, CLOSE...}` -> 存入显存。
2. **Step 1 (Select)**: `MCTS Agent` 选中一个待优化的因子节点 `F_old`（假设它收益高但回撤大）。
3. **Step 2 (Diagnose)**: 系统计算各维度得分，选中 "Stability" 为优化目标。
4. **Step 3 (Suggest)**: `LLM Client` (Role: Analyst) 提出建议 -> "尝试引入过去20天的波动率作为分母进行平滑"。
5. **Step 4 (Code)**: `LLM Client` (Role: Coder) 将建议转化为公式 -> `"F_new = F_old / (Std(Close, 20) + 1e-6)"`。
6. **Step 5 (Check)**: 检查 `F_new` 是否包含 `Forbidden Subtrees` (FSA)。
7. **Step 6 (Eval)**: `Executor` 计算张量 -> `Backtest` 算出各指标 -> `Alpha Zoo` 进行相对排名。
8. **Step 7 (Risk)**: `LLM Client` (Role: Risk Officer) 审查公式复杂度，给出 `Overfitting Score`。
9. **Step 8 (Update)**: 综合得分回传更新 MCTS 树。
