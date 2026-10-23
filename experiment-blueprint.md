# 实验设计

对于每种inference integrity violation type都做一组实验。每组实验有M条序列，其中均匀随机有N条序列有inference integrity violation，每条序列prefill P个tokens，decode直到max_length个 tokens时停止，每条序列只有一处地方有violation。

## 参数设置
- M=256
- N=32
- P=32
- max_length=128

## Inference Integrity Violation Type

### KV Cache Mismatch:
decode的某一step中，使用了不匹配的KV Cache进行计算

### Serving Configuration:
运行环境发生变化，例如route到不同设备上，或者在不符合用户配置的环境下推理，例如用户给定的thinking budget为0，但推理过程中仍然存在思维链条

### Model Mismatch:
对于整条序列，使用小模型来代替大模型进行推理，却claim是大模型的推理结果

### Token Selection Mismatch：
decode的某一step中，强行更改某个token

### Computation Mismatch：
某一处计算时，kernel出错导致了计算结果错误

### Input Mismatch：
prompt中信息发生删除或重排序

## 检测指标

- Serving时的开销，TTFT，TOPT，ITL，E2E latency，需要记录Mean,Median,P50,P75,P90,P95,P99
- Prover开销，耗时、证明大小、峰值内存
- Verifier开销，耗时
- True/False Postives/Negaives, 收集原始数据用于计算高阶统计数据

**False Positives**: 该序列中无violation，但是工具误判该序列有violation
**True Positives**: 该序列有violation，该工具也判断出该序列确实有violation
**False Negatives**: 该序列有violation，但工具判断出该序列无violation
**True Negatives**: 该序列无violation，该工具也判断出该序列无violation

## 测试模型：

- LLaMA-3.2-1B，本地权重: /home/yliangh/models/meta-llama/Llama-3.2-1B
- Qwen3.5-0.8B，本地权重: /home/yliangh/models/Qwen/Qwen3.5-0.8B

## 交付内容

- 实验具体实施细节
- 自动化实验运行脚本
