"""DSPARK 投机解码 + DP attention 端到端测试 (DeepSeek-V4-Flash, tp8/dp8)。

覆盖 staged change: DraftBlockProposer._fill_dp_moe_sync_metadata() 为 DSPARK
draft 的 ForwardBatch 补齐 original_global_num_tokens_cpu /
num_token_non_padded / num_token_non_padded_cpu。在 --enable-dp-attention +
MoE draft 下, 这些字段分别被:
  * decode CUDA graph 的 DP 准入逻辑使用 (max(original_global_num_tokens_cpu),
    缺失时直接 TypeError 崩溃);
  * MoE topk 的 padding 掩码使用 (num_token_non_padded);
  * radix attention 的真实 token 切片使用 (num_token_non_padded_cpu)。
字段缺失会让 draft 路径崩溃, 或把 DP padding 的垃圾 token 送进 MoE draft，
导致 draft logits 错乱、accept length 塌缩到 ~1.0。

断言策略: 只做服务级检查 (请求成功、输出非空、进程存活) + accept length
下限, 不校验输出内容 (模型措辞/Unicode 变体如 "H₂O" 会造成误报)。

本文件是 test/registered/spec/dspark/test_dspark_dp_tier.py 中
TestDraftDpSyncMetadata 单元测试的端到端对应版本; 启动参数与参考命令一致
(仅新增 --decode-log-interval 1, 让 avg_spec_accept_length 立即滚动上报,
 否则默认 40 次 decode 迭代才 rollup 一次, 短测试窗口内查不到该字段)。
并发不均匀负载用于让各 DP rank 的请求数出现差异 (含空闲 rank 的
idle batch 参与路径)。
"""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# 1. 模型路径: 优先环境变量 (默认与参考命令的本地 checkpoint 对齐)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/DeepSeek-V4-Flash-0731",
)
BASE_URL = "http://127.0.0.1:9985"
SERVER_LAUNCH_TIMEOUT = 3600  # 8 卡大模型加载较慢, 留足启动时间
REQUEST_TIMEOUT = 600

# 短 prompt: 验证基本解码可用。只断言输出非空, 不校验内容 ——
# 模型措辞/Unicode 变体 (如 "H₂O" vs "h2o") 会造成内容断言误报
SHORT_PROMPTS = [
    "What is the capital of France? Answer with just the city name.",
    "What is the capital of Japan? Answer with just the city name.",
    "What is the capital of Italy? Answer with just the city name.",
    "What is 7 + 5? Answer with just the number.",
    "What is 6 times 7? Answer with just the number.",
    "What is the chemical formula of water? Answer with just the formula.",
]

# 长输出 prompt: 拉长尾部请求的真实生成长度, 让各 DP rank 的排空节奏错开
LONG_PROMPTS = [
    "List the first 20 prime numbers, one number per line.",
    "Write the 26 letters of the English alphabet, one letter per line.",
]


class TestDeepSeekV4DsparkDpAttention(CustomTestCase):
    process = None

    @classmethod
    def setUpClass(cls):
        # 2. 启动参数与参考命令完全对齐 (DSPARK + dp attention + dp lm head)
        cls.model = MODEL_PATH
        cls.base_url = BASE_URL
        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "8",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
            "--max-running-requests",
            "64",
            "--cuda-graph-max-bs",
            "64",
            "--mem-fraction-static",
            "0.8",
            "--disable-radix-cache",
            "--disable-custom-all-reduce",
            "--disable-overlap-schedule",
            "--disable-shared-experts-fusion",
            "--disable-piecewise-cuda-graph",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--dist-timeout",
            "60000",
            "--enable-dp-lm-head",
            "--enable-dp-attention",
            "--dp-size",
            "8",
            # 让 avg_spec_accept_length 每次 decode 迭代都 rollup, 否则
            # 默认 40 次迭代才上报一次, 短测试窗口内 /server_info 查不到
            "--decode-log-interval",
            "1",
        ]

        print(f"[Server] 启动 DSPARK + DP attention 服务, 模型: {cls.model}")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )
        cls.served_model_name = cls._get_served_model_name()
        print(f"[Server] served model name: {cls.served_model_name}")

    @classmethod
    def tearDownClass(cls):
        # 测试结束, 清理 Server 进程树
        if cls.process is not None:
            print("[Server] 测试结束, 正在关闭 SGLang 服务...")
            kill_process_tree(cls.process.pid)

    @classmethod
    def _get_served_model_name(cls):
        # 未设置 --served-model-name 时, OpenAI 接口的 model 字段需为模型路径
        try:
            resp = requests.get(cls.base_url + "/v1/models", timeout=60)
            resp.raise_for_status()
            return resp.json()["data"][0]["id"]
        except Exception:
            return cls.model

    def _chat(self, prompt, max_tokens=64):
        payload = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            # 关闭 thinking 得到短且确定的事实型回答; 同时也拼接
            # reasoning_content, 兼容模板忽略该开关的情况
            "chat_template_kwargs": {"thinking": False},
        }
        resp = requests.post(
            self.base_url + "/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        parts = [
            message.get("content") or "",
            message.get("reasoning_content") or "",
        ]
        return "\n".join(part for part in parts if part)

    def test_01_concurrent_uneven_load_generation(self):
        """并发混合负载: 长/短请求混合, 各 DP rank 的请求数不均 (排空阶段
        部分 rank 归零), 覆盖 draft ForwardBatch 的 DP 同步元数据填充路径。
        断言所有请求成功返回非空输出且服务进程存活 (不校验输出内容, 避免
        模型措辞/Unicode 变体造成误报)。"""
        tasks = []
        for i in range(24):
            if i % 4 == 3:
                # 每 4 个请求掺 1 个长输出请求, 制造 rank 间的长度差
                tasks.append((LONG_PROMPTS[(i // 4) % len(LONG_PROMPTS)], 192))
            else:
                tasks.append((SHORT_PROMPTS[i % len(SHORT_PROMPTS)], 64))

        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [
                pool.submit(self._chat, prompt, max_tokens)
                for prompt, max_tokens in tasks
            ]
            outputs = [future.result() for future in futures]

        for i, (output, (prompt, _)) in enumerate(zip(outputs, tasks)):
            self.assertTrue(
                output.strip(),
                f"request {i} 返回空输出, prompt={prompt!r}",
            )
        self.assertIsNone(self.process.poll())

    def test_02_lone_request_after_drain(self):
        """全部请求排空后 (server 经历过全空闲 DP 轮次) 的单请求:
        global_num_tokens 形如 [1, 0, ..., 0], 其余 7/8 rank 走 idle
        参与路径 (run_idle_participation), 单独验证空闲分支的元数据填充。"""
        output = self._chat(SHORT_PROMPTS[0], max_tokens=64)
        self.assertTrue(output.strip(), "排空后的单请求返回空输出")
        self.assertIsNone(self.process.poll())

    def test_03_spec_accept_length(self):
        """DSPARK 真正生效: 各 DP rank 的平均 accept length 必须显著大于 1。
        若 draft 的 DP padding 处理出错 (num_token_non_padded 缺失/错误),
        draft 提案会被 target 全部拒绝, accept length 塌缩到 ~1.0。
        依赖 --decode-log-interval 1 使指标立即 rollup (见 setUpClass)。"""
        server_info = requests.get(self.base_url + "/server_info", timeout=60).json()
        values = [
            state["avg_spec_accept_length"]
            for state in server_info.get("internal_states", [])
            if "avg_spec_accept_length" in state
        ]
        self.assertTrue(
            values,
            "没有任何 DP rank 上报 avg_spec_accept_length, 投机解码可能没有运行",
        )
        avg = sum(values) / len(values)
        print(f"[Spec] {len(values)} 个 DP rank 的平均 accept length = {avg:.3f}")
        # 故障模式 (draft 元数据错误) 下 accept length ≈ 1.0 (仅 bonus 被接受);
        # 阈值取 1.2: 明确高于故障基线, 又不苛求具体加速比, 减少误报
        self.assertGreater(avg, 1.2)


if __name__ == "__main__":
    unittest.main()
