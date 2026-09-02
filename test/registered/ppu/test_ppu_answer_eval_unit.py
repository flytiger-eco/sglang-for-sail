"""CPU tests for the deterministic PPU Answer evaluator."""

import copy
import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci, register_ppu_ci
from sglang.test.kits.answer_eval_kit import (
    AnswerEvalError,
    CandidateRequestError,
    _repeated_ngram_coverage,
    answer_expected_hardware,
    apply_cross_case_checks,
    build_answer_server_args,
    build_report,
    canonical_digest,
    default_provenance,
    evaluate_case,
    load_json,
    normalize_answer,
    parse_thinking_blocks,
    redact_report,
    render_candidates,
    render_junit,
    request_chat_completion,
    validate_annotation_record,
    validate_dataset,
    validate_profile,
    validate_test_config,
    write_report_files,
)

# Hardware-free (pure stdlib evaluator), so the CPU suite owns it. It is also
# registered on the PPU per-commit chain: this file is the only guard for the
# evaluator that gates the nightly-answer-*-ppu suites, and a PR that breaks a
# threshold or a redaction rule should turn the PPU gate red immediately rather
# than surface hours later in the nightly Answer run.
register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_ppu_ci(est_time=10, suite="stage-b-test-1-gpu-ppu")

DATA_DIR = Path(__file__).with_name("answer_eval") / "data"


class TestPPUAnswerEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_config = load_json(
            DATA_DIR / "qwen3_5_397b_a17b_w8a8_int8_test_config.json"
        )
        cls.dataset = load_json(DATA_DIR / "answer_cases_zh_v1.json")
        cls.profile = load_json(DATA_DIR / "quality_profile.json")
        cls.cases = {case["id"]: case for case in cls.dataset["cases"]}

    def evaluate(self, case_id, answer, finish_reason="stop"):
        return evaluate_case(
            self.cases[case_id],
            answer,
            finish_reason,
            self.profile,
            self.dataset["revision"],
        )

    def reason_codes(self, result):
        return {finding["reason_code"] for finding in result["findings"]}

    def test_data_contract_and_digest_are_stable(self):
        validate_test_config(self.test_config)
        validate_dataset(self.dataset)
        validate_profile(self.profile)
        self.assertEqual(
            self.test_config["server"]["parameters"]["tp_size"],
            len(self.test_config["hardware"]["visible_devices"]),
        )
        # The contract under test is the flag spelling and the argument order.
        # The values belong to the reviewed configuration, so they are read from
        # it rather than copied here, where a copy would silently become a
        # second, competing source of truth.
        parameters = self.test_config["server"]["parameters"]
        self.assertEqual(
            build_answer_server_args(self.test_config),
            [
                "--trust-remote-code",
                "--tp-size",
                str(parameters["tp_size"]),
                "--attention-backend",
                parameters["attention_backend"],
                "--mem-fraction-static",
                str(parameters["mem_fraction_static"]),
                "--quantization",
                parameters["quantization"],
                "--reasoning-parser",
                parameters["reasoning_parser"],
                "--served-model-name",
                self.test_config["model"]["served_model_name"],
            ],
        )
        # Likewise, provenance depends on the shape of the hardware string, not
        # on one particular accelerator generation; the rendering rule itself is
        # pinned by test_expected_hardware_renders_the_declared_topology.
        self.assertRegex(
            answer_expected_hardware(self.test_config), r"^[0-9a-z.]+-\d+x[0-9.]+g$"
        )
        self.assertEqual(len(canonical_digest(self.dataset)), 64)
        self.assertEqual(len({case["id"] for case in self.dataset["cases"]}), 10)

    def test_every_reviewed_test_config_is_executable(self):
        # One config per nightly Answer job, and only the workflow names the ones
        # this class does not load, so they are validated here instead of first
        # failing on the machine after a checkpoint has been warmed.
        config_paths = sorted(DATA_DIR.glob("*_test_config.json"))
        self.assertGreaterEqual(len(config_paths), 2)
        test_ids = set()
        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                config = load_json(config_path)
                validate_test_config(config)
                test_ids.add(config["test_id"])
                for field in ("dataset", "quality_profile"):
                    self.assertTrue(
                        (config_path.parent / config["evaluation"][field]).is_file(),
                        f"evaluation.{field} does not resolve to a file",
                    )
                args = build_answer_server_args(config)
                self.assertEqual(
                    args[args.index("--tp-size") + 1],
                    str(len(config["hardware"]["visible_devices"])),
                )
                # The schema only requires a non-empty string, so a value that
                # argparse would reject would otherwise surface as a server that
                # never starts. server_args is imported here rather than at
                # module scope to keep the evaluator's own import path stdlib
                # only; "unquant" is that module's spelling for an explicit
                # opt-out, which is how a BF16 checkpoint satisfies the field.
                from sglang.srt.server_args import QUANTIZATION_CHOICES

                self.assertIn(
                    config["server"]["parameters"]["quantization"],
                    QUANTIZATION_CHOICES,
                )
        self.assertEqual(len(test_ids), len(config_paths), "test_id must be unique")

    def test_expected_hardware_renders_the_declared_topology(self):
        config = {
            "hardware": {
                "generation": "zw810e",
                "visible_devices": [0, 1, 2, 3],
                "memory_gib_per_device": 96,
            }
        }
        self.assertEqual(answer_expected_hardware(config), "zw810e-4x96g")

        # A capacity that JSON carries as a float must not leak a ".0" into
        # provenance, while a genuinely fractional capacity must survive.
        config["hardware"]["memory_gib_per_device"] = 96.0
        self.assertEqual(answer_expected_hardware(config), "zw810e-4x96g")
        config["hardware"]["memory_gib_per_device"] = 97.5
        self.assertEqual(answer_expected_hardware(config), "zw810e-4x97.5g")

        config["hardware"]["generation"] = "btv1.5"
        config["hardware"]["visible_devices"] = [0, 1, 2, 3, 4, 5, 6, 7]
        config["hardware"]["memory_gib_per_device"] = 144
        self.assertEqual(answer_expected_hardware(config), "btv1.5-8x144g")

    def test_execution_config_rejects_inconsistent_or_unsafe_values(self):
        invalid = copy.deepcopy(self.test_config)
        invalid["server"]["parameters"]["tp_size"] = 4
        with self.assertRaisesRegex(AnswerEvalError, "visible device count"):
            validate_test_config(invalid)

        invalid = copy.deepcopy(self.test_config)
        invalid["evaluation"]["dataset"] = "../private/prompts.json"
        with self.assertRaisesRegex(AnswerEvalError, "relative"):
            validate_test_config(invalid)

    def test_provenance_prefers_the_actual_checked_out_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen3.5-397B-A17B-W8A8-INT8"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"qwen3_5_moe"}\n', encoding="utf-8"
            )
            with mock.patch.dict(
                os.environ,
                {
                    "GITHUB_SHA": "dispatcher-sha",
                    "SGLANG_PPU_SOURCE_REVISION": "tested-sha",
                    "PPU_BASE_IMAGE_DIGEST": "sha256:image",
                },
                clear=True,
            ):
                provenance = default_provenance(
                    "Qwen3.5-397B-A17B-W8A8-INT8",
                    str(model_dir),
                    server_config=self.test_config["server"]["parameters"],
                    generation_config=self.test_config["request"]["generation"],
                    expected_hardware=answer_expected_hardware(self.test_config),
                )
        self.assertEqual(provenance["source_revision"], "tested-sha")
        self.assertEqual(provenance["base_image_digest"], "sha256:image")
        self.assertEqual(provenance["checkpoint_name"], model_dir.name)
        self.assertEqual(
            provenance["expected_hardware"],
            answer_expected_hardware(self.test_config),
        )
        self.assertEqual(len(provenance["checkpoint_config_sha256"]), 64)
        self.assertEqual(
            provenance["generation_config"],
            self.test_config["request"]["generation"],
        )

    def test_normalization_and_balanced_thinking_parser(self):
        final, findings = parse_thinking_blocks(
            "<think>这里是推理，不参与事实判分。</think> ４\r\n"
        )
        self.assertEqual(normalize_answer(final), "4")
        self.assertEqual(findings, [])

    def test_malformed_and_empty_thinking_blocks_fail(self):
        for answer in ("</think>答案", "<think>未闭合", "<think>a<think>b</think>"):
            with self.subTest(answer=answer):
                _, findings = parse_thinking_blocks(answer)
                self.assertIn(
                    "malformed_thinking_block",
                    {finding["reason_code"] for finding in findings},
                )
        _, findings = parse_thinking_blocks("<think>只有推理</think>")
        self.assertIn("empty_final_answer", {item["reason_code"] for item in findings})

    def test_unicode_hard_fail_boundaries(self):
        invalid = self.evaluate("deepseek-letter-count", "答案是4\ufffd")
        self.assertIn("invalid_unicode", self.reason_codes(invalid))

        abnormal = self.evaluate("deepseek-letter-count", "答案是4\ue000\ue001\ue002")
        self.assertIn("abnormal_codepoint_ratio", self.reason_codes(abnormal))

        control = self.evaluate("deepseek-letter-count", "答案是4\x01")
        self.assertIn("invalid_unicode", self.reason_codes(control))

        surrogate = self.evaluate("deepseek-letter-count", "答案是4\ud800")
        self.assertIn("invalid_unicode", self.reason_codes(surrogate))
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_report(
                self.dataset,
                self.profile,
                {
                    case["id"]: {
                        "content": (
                            "答案是4\ud800"
                            if case["id"] == "deepseek-letter-count"
                            else "候选回答"
                        ),
                        "finish_reason": "stop",
                        "model": "Qwen3.5-397B-A17B-W8A8-INT8",
                    }
                    for case in self.dataset["cases"]
                },
                {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
            )
            write_report_files(report, Path(temp_dir), include_raw_outputs=True)

    def test_repetition_hard_fail_boundaries(self):
        same = self.evaluate("deepseek-letter-count", "4" + "哈" * 8)
        self.assertIn("same_codepoint_run", self.reason_codes(same))

        periodic = self.evaluate("deepseek-letter-count", "4" + "甲乙丙丁戊己" * 4)
        self.assertIn("periodic_fragment_repeat", self.reason_codes(periodic))

        sentence = self.evaluate(
            "deepseek-letter-count",
            "答案确实就是四个。答案确实就是四个。答案确实就是四个。",
        )
        self.assertIn("exact_sentence_repeat", self.reason_codes(sentence))

        newline_sentence = self.evaluate(
            "deepseek-letter-count",
            "答案就是四个字母e\n答案就是四个字母e\n答案就是四个字母e",
        )
        self.assertIn("exact_sentence_repeat", self.reason_codes(newline_sentence))

    def test_ngram_coverage_thresholds(self):
        settings = self.profile["repetition"]
        repetitive = ("甲乙丙丁" * 20) + "戊己庚辛壬癸天地玄黄宇宙洪荒"
        self.assertGreaterEqual(
            _repeated_ngram_coverage(repetitive, settings),
            settings["ngram_hard_fail_coverage"],
        )
        normal = "这是一段包含不同词汇和自然语序的中文文本，用于确认正常表达不会因为少量常见字词而被判定成异常复读。"
        self.assertEqual(_repeated_ngram_coverage(normal, settings), 0.0)

    def test_repeated_mojibake_marker_is_hard_fail(self):
        result = self.evaluate("deepseek-letter-count", "答案是4，Ã¤ 与 Ã¥ 是异常片段")
        marker = next(
            finding
            for finding in result["findings"]
            if finding["reason_code"] == "mojibake_marker"
        )
        self.assertEqual(marker["action"], "hard_fail")
        self.assertEqual(result["verdict"], "failed")

        latin1 = self.evaluate("deepseek-letter-count", "答案是4，ä½ å¥½")
        self.assertIn("reversible_utf8_mojibake", self.reason_codes(latin1))
        self.assertEqual(latin1["verdict"], "failed")

    def test_finish_reason_length_fails(self):
        result = self.evaluate("deepseek-letter-count", "答案是4", "length")
        self.assertIn("finish_reason_length", self.reason_codes(result))

        for finish_reason in (None, "content_filter", "tool_calls"):
            with self.subTest(finish_reason=finish_reason):
                result = self.evaluate(
                    "deepseek-letter-count", "答案是4", finish_reason
                )
                self.assertIn("finish_reason_incomplete", self.reason_codes(result))

    def test_chat_completion_contract_and_strict_utf8(self):
        captured = {}

        def post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            body = {
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "答案是 4",
                            "reasoning_content": None,
                        },
                    }
                ],
                "usage": {"completion_tokens": 4},
            }
            return SimpleNamespace(
                status_code=200,
                content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            )

        fake_requests = SimpleNamespace(RequestException=Exception, post=post)
        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            result = request_chat_completion(
                "http://127.0.0.1:30000/",
                "Qwen3.5-397B-A17B-W8A8-INT8",
                "deepseek这个单词中有几个e？",
                "请直接回复最终的答案。",
                timeout_seconds=10,
                generation_config=self.test_config["request"]["generation"],
            )

        self.assertEqual(captured["url"], "http://127.0.0.1:30000/v1/chat/completions")
        self.assertEqual(captured["timeout"], 10)
        self.assertEqual(captured["json"]["temperature"], 0)
        self.assertFalse(captured["json"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(result["content"], "答案是 4")

        def invalid_post(*_args, **_kwargs):
            return SimpleNamespace(status_code=200, content=b"\xff")

        fake_requests.post = invalid_post
        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaisesRegex(CandidateRequestError, "not UTF-8") as caught:
                request_chat_completion(
                    "http://127.0.0.1:30000",
                    "model",
                    "prompt",
                    "suffix",
                    timeout_seconds=10,
                    generation_config=self.test_config["request"]["generation"],
                )
        self.assertEqual(caught.exception.reason_code, "response_decode_error")

    def test_objective_reference_rules_accept_correct_variants(self):
        answers = {
            "deepseek-letter-count": "deepseek 中共有 4 个字母 e。",
            "spring-dawn-poem": "作者孟浩然：春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
            "mount-everest-height": "世界最高峰是珠穆朗玛峰，最新公认高程为 8848.86 米。",
            "henan-bordering-provinces": "河南与河北、山西、陕西、湖北、安徽、山东六省接壤。",
            "baijiaxing-first-ten": "赵、钱、孙、李、周、吴、郑、王、冯、陈。",
            "art-primary-colors": "美术颜料三原色是红色、黄色和蓝色。",
            "chinese-zodiac": "十二生肖为鼠、牛、虎、兔、龙、蛇、马、羊、猴、鸡、狗、猪。",
            "red-ball-probability": "概率为 (3/7)×(2/6)=1/7，约为14.29%。",
        }
        for case_id, answer in answers.items():
            with self.subTest(case_id=case_id):
                result = self.evaluate(case_id, answer)
                self.assertEqual(result["verdict"], "passed", result["findings"])

    def test_number_extraction_respects_chinese_context_and_compound_numerals(self):
        correct = self.evaluate("deepseek-letter-count", "答案是4个字母e。")
        correct_with_context = self.evaluate(
            "deepseek-letter-count", "deepseek共有8个字母，其中4个是字母e。"
        )
        correct_with_asserted_context = self.evaluate(
            "deepseek-letter-count",
            "deepseek 的总字符数为8，其中 e 的数量为4。",
        )
        correct_probability = self.evaluate("red-ball-probability", "答案是1/7。")
        correct_probability_derivation = self.evaluate(
            "red-ball-probability",
            "第一次抽中红球的概率为3/7，第二次为2/6，"
            "所以两次都是红球的概率为1/7，约14.29%。",
        )
        wrong = self.evaluate("deepseek-letter-count", "答案是十四个字母e。")
        conflicting = self.evaluate("deepseek-letter-count", "答案是4个，而实际是3个。")
        also_conflicting = self.evaluate(
            "deepseek-letter-count", "答案是4个，也是3个。"
        )
        self.assertEqual(correct["verdict"], "passed", correct["findings"])
        self.assertEqual(
            correct_with_context["verdict"], "passed", correct_with_context["findings"]
        )
        self.assertEqual(
            correct_probability["verdict"], "passed", correct_probability["findings"]
        )
        self.assertEqual(
            correct_with_asserted_context["verdict"],
            "passed",
            correct_with_asserted_context["findings"],
        )
        self.assertEqual(
            correct_probability_derivation["verdict"],
            "passed",
            correct_probability_derivation["findings"],
        )
        self.assertEqual(wrong["verdict"], "failed", wrong["findings"])
        self.assertEqual(conflicting["verdict"], "failed", conflicting["findings"])
        self.assertEqual(
            also_conflicting["verdict"], "failed", also_conflicting["findings"]
        )

    def test_negated_extra_and_interleaved_facts_fail(self):
        poem = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
        negated_authors = [
            self.evaluate("spring-dawn-poem", f"作者不是孟浩然，而是李白：{poem}"),
            self.evaluate(
                "spring-dawn-poem", f"孟浩然不是《春晓》的作者，李白才是。{poem}"
            ),
            self.evaluate(
                "spring-dawn-poem", f"作者不是唐代诗人孟浩然，而是李白。{poem}"
            ),
        ]
        negated_number = self.evaluate(
            "deepseek-letter-count", "答案不是4个，而是3个。"
        )
        negated_probability = self.evaluate(
            "red-ball-probability", "概率不是1/7，而是1/2。"
        )
        negated_percentage = self.evaluate(
            "red-ball-probability", "答案不是14.29%，而是50%。"
        )
        negated_province = self.evaluate(
            "henan-bordering-provinces",
            "河南不与河北接壤，却与山西、陕西、湖北、安徽、山东接壤。",
        )
        extra_province = self.evaluate(
            "henan-bordering-provinces",
            "河南与河北、山西、陕西、湖北、安徽、山东和广东接壤。",
        )
        self_province = self.evaluate(
            "henan-bordering-provinces",
            "河南与河北、山西、陕西、湖北、安徽、山东以及河南自身接壤。",
        )
        interleaved_surname = self.evaluate(
            "baijiaxing-first-ten",
            "赵、钱、孙、刘、李、周、吴、郑、王、冯、陈。",
        )
        leading_surname = self.evaluate(
            "baijiaxing-first-ten",
            "刘、赵、钱、孙、李、周、吴、郑、王、冯、陈。",
        )
        trailing_surname = self.evaluate(
            "baijiaxing-first-ten",
            "赵、钱、孙、李、周、吴、郑、王、冯、陈、刘。",
        )
        for result in (
            *negated_authors,
            negated_number,
            negated_probability,
            negated_percentage,
            negated_province,
            extra_province,
            self_province,
            interleaved_surname,
            leading_surname,
            trailing_surname,
        ):
            self.assertEqual(result["verdict"], "failed", result["findings"])
            self.assertIn("fact_rule_failed", self.reason_codes(result))

    def test_exact_sequence_allows_numbering_and_item_labels(self):
        surname = self.evaluate(
            "baijiaxing-first-ten",
            "1.赵、2.钱、3.孙、4.李、5.周、6.吴、7.郑、8.王、9.冯、10.陈",
        )
        poem = self.evaluate(
            "spring-dawn-poem",
            "作者孟浩然。第一句：春眠不觉晓；第二句：处处闻啼鸟；"
            "第三句：夜来风雨声；第四句：花落知多少。",
        )
        poem_with_comma = self.evaluate(
            "spring-dawn-poem",
            "作者孟浩然，春眠不觉晓，处处闻啼鸟，夜来风雨声，花落知多少。",
        )
        self.assertEqual(surname["verdict"], "passed", surname["findings"])
        self.assertEqual(poem["verdict"], "passed", poem["findings"])
        self.assertEqual(
            poem_with_comma["verdict"], "passed", poem_with_comma["findings"]
        )

    def test_known_old_henan_golden_is_rejected(self):
        result = self.evaluate(
            "henan-bordering-provinces",
            "河南与山东、安徽、湖北、湖南、山西、陕西、河北等七省接壤。",
        )
        self.assertEqual(result["verdict"], "failed")
        finding = next(
            item
            for item in result["findings"]
            if item["reason_code"] == "fact_rule_failed"
        )
        self.assertEqual(finding["observed"]["unexpected"], ["湖南"])

    def test_wrong_objective_facts_fail(self):
        wrong_answers = {
            "deepseek-letter-count": "共有 3 个 e。",
            "spring-dawn-poem": "作者李白：春眠不觉晓，处处闻啼鸟。",
            "mount-everest-height": "最高峰是泰山，高度 1545 米。",
            "baijiaxing-first-ten": "钱、赵、孙、李、周、吴、郑、王、冯、陈。",
            "art-primary-colors": "三原色是红绿蓝。",
            "chinese-zodiac": "鼠牛虎兔龙蛇马羊猴鸡狗。",
            "red-ball-probability": "概率为 28.57%。",
        }
        for case_id, answer in wrong_answers.items():
            with self.subTest(case_id=case_id):
                result = self.evaluate(case_id, answer)
                self.assertEqual(result["verdict"], "failed")
                self.assertEqual(result["failure_class"], "candidate_failed")
                self.assertIn("fact_rule_failed", self.reason_codes(result))

    def test_open_ended_cases_are_only_hard_constraint_covered(self):
        xian = self.evaluate(
            "xian-three-day-trip",
            "西安三日游：首日逛钟楼、城墙和回民街；次日参观兵马俑与华清池；第三日游大雁塔及陕西历史博物馆，感受古都风貌。",
        )
        tengwang = self.evaluate(
            "tengwangge-reflection",
            "读王勃的《滕王阁序》，我最先感受到的是文字铺陈出的辽阔气象。山川、楼阁与宾客在整饬的骈句中次第展开，落霞与秋水的画面尤其明净。文章并不只写宴游之乐，也把个人际遇、生命短暂和进取之心交织在一起。作者虽感叹关山难越、萍水相逢，却没有停在失意中，而以老当益壮、穷且益坚表达不甘沉沦的选择。这种由盛景转入身世、再由感伤振起精神的结构，使作品既华美又有力量。今天重读它，我理解到真正动人的才华不仅是辞藻繁复，更是人在困顿中仍能确认志向。我们面对不确定的环境，也应珍惜相遇，保持清醒，在有限时间里做值得留下的事。",
        )
        self.assertEqual(xian["verdict"], "passed", xian["findings"])
        self.assertEqual(tengwang["verdict"], "passed", tengwang["findings"])
        self.assertEqual(xian["semantic_coverage"], "hard_constraints_only")
        self.assertEqual(xian["judge"]["status"], "deferred")

    def test_cross_case_duplicate_and_near_duplicate(self):
        base = "这是一段长度足够并且被错误复用于两个不同问题的统一候选回答文本。"
        left = self.evaluate("xian-three-day-trip", base + "钟楼大雁塔三日游")
        right = self.evaluate("tengwangge-reflection", base + "钟楼大雁塔三日游")
        apply_cross_case_checks([left, right], self.profile)
        self.assertIn("cross_case_duplicate", self.reason_codes(left))
        self.assertIn("cross_case_duplicate", self.reason_codes(right))

        common = (
            "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黄宇宙洪荒日月盈昃辰宿列张"
            "寒来暑往秋收冬藏闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈"
        )
        left = {
            "case_id": "left",
            "normalized_answer": common + "甲",
            "findings": [],
        }
        right = {
            "case_id": "right",
            "normalized_answer": common + "乙",
            "findings": [],
        }
        apply_cross_case_checks([left, right], self.profile)
        self.assertTrue(
            {"cross_case_duplicate", "cross_case_near_duplicate"}
            & self.reason_codes(left)
        )

    def test_build_report_classifies_l0_failure_and_model_mismatch(self):
        responses = {
            case["id"]: {
                "content": "4" if case["id"] == "deepseek-letter-count" else "临时答案",
                "finish_reason": "stop",
                "model": "unexpected",
            }
            for case in self.dataset["cases"]
        }
        responses["spring-dawn-poem"] = {
            "error": "Timeout",
            "reason_code": "request_error",
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        mismatch = report["cases"][0]
        self.assertIn("unexpected_model_name", self.reason_codes(mismatch))
        self.assertEqual(report["cases"][1]["failure_class"], "server_error")
        self.assertNotIn(
            "spring-dawn-poem",
            {candidate["case_id"] for candidate in report["label_candidates"]},
        )
        public = redact_report(report)
        junit = ET.fromstring(render_junit(public))
        spring_dawn = next(
            case
            for case in junit.findall("testcase")
            if case.attrib["name"] == "spring-dawn-poem"
        )
        self.assertEqual(spring_dawn.find("failure").attrib["type"], "server_error")

    def test_label_candidates_conform_to_public_schema_surface(self):
        responses = {
            case["id"]: {
                "content": "候选回答",
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        schema = load_json(DATA_DIR / "annotation_record.schema.json")
        allowed = set(schema["properties"])
        required = set(schema["required"])
        self.assertTrue(report["label_candidates"])
        for candidate in report["label_candidates"]:
            self.assertFalse(set(candidate) - allowed)
            self.assertFalse(required - set(candidate))
            self.assertRegex(candidate["sample_id"], r"^[0-9a-f]{64}$")
            self.assertRegex(candidate["answer_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(candidate["selection_reasons"])
            self.assertEqual(candidate["case_revision"], self.dataset["revision"])
            self.assertEqual(
                candidate["quality_profile"]["sha256"],
                canonical_digest(self.profile),
            )
            self.assertIn("candidate_answer", candidate)

        pending_rule = schema["properties"]["annotation"]["allOf"][0]["then"]
        self.assertEqual(pending_rule["properties"]["reviews"]["maxItems"], 0)

        candidate = report["label_candidates"][0]
        validate_annotation_record(candidate)
        review = {
            "reviewer": "reviewer-a",
            "reviewed_at": "2026-09-01T00:00:00Z",
            "verdict": "pass",
            "scores": {
                "factual_correctness": 3,
                "relevance": 3,
                "completeness": 3,
                "coherence": 3,
                "no_repetition": 3,
            },
            "critical_errors": [],
        }
        invalid_pending = copy.deepcopy(candidate)
        invalid_pending["annotation"]["reviews"] = [review]
        with self.assertRaisesRegex(AnswerEvalError, "lifecycle state"):
            validate_annotation_record(invalid_pending)

        duplicate_reviewers = copy.deepcopy(candidate)
        duplicate_reviewers["annotation"].update(
            {
                "status": "reviewed",
                "reviews": [review, {**review, "reviewer": "Reviewer-A"}],
            }
        )
        with self.assertRaisesRegex(AnswerEvalError, "different reviewers"):
            validate_annotation_record(duplicate_reviewers)

    def test_public_report_and_files_do_not_contain_candidate_text(self):
        responses = {
            case["id"]: {
                "content": f"PRIVATE-CANDIDATE-{case['id']}",
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        public = redact_report(report)
        serialized = json.dumps(public)
        self.assertNotIn("PRIVATE-CANDIDATE", serialized)
        self.assertNotIn("raw_response", serialized)
        self.assertNotIn("answer_sha256", serialized)
        self.assertNotIn("sample_id", serialized)
        self.assertNotIn("label_candidates", public)
        ET = render_junit(public)
        self.assertNotIn(b"PRIVATE-CANDIDATE", ET)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_report_files(report, output_dir)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"result.json", "summary.md", "junit.xml"},
            )
            for path in output_dir.iterdir():
                self.assertNotIn(b"PRIVATE-CANDIDATE", path.read_bytes())

        report["cases"][0]["returned_model"] = "/private/model/path"
        public = redact_report(report)
        serialized = json.dumps(public)
        self.assertNotIn("/private/model/path", serialized)
        self.assertEqual(
            public["cases"][0]["returned_model"],
            "<redacted-unexpected-model-name>",
        )
        self.assertNotIn("returned_model_sha256", public["cases"][0])

    def test_candidates_render_for_the_job_log(self):
        # Whoever opens a red nightly reads the log first, so the block has to
        # carry the answer text, the prompt it answered, and the observed value
        # that tripped the rule.  A candidate that UTF-8 cannot encode must come
        # out escaped rather than raise while the block is being built.
        responses = {
            case["id"]: {
                "content": (
                    "3个" if case["id"] == "deepseek-letter-count" else "候选回答\ud800"
                ),
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        rendered = render_candidates(report, self.dataset)
        self.assertIn("3个", rendered)
        self.assertIn("[failed] deepseek-letter-count", rendered)
        self.assertIn("observed=", rendered)
        first_prompt = self.dataset["cases"][0]["prompt"]
        self.assertIn(first_prompt, rendered)
        # The lone surrogate survives as an escape and the block stays writable
        # to a UTF-8 stream.
        self.assertNotIn("\ud800", rendered)
        self.assertIn("\\ud800", rendered)
        rendered.encode("utf-8")
        # Without the dataset the block still renders, only without prompts.
        self.assertNotIn(first_prompt, render_candidates(report))

    def test_raw_outputs_land_next_to_the_redacted_report(self):
        # The nightly workflow publishes these two file names as its only record
        # of what the model actually answered, so a rename has to break here
        # rather than quietly ship an artifact with no candidate text in it.
        responses = {
            case["id"]: {
                "content": "答案是3",
                "finish_reason": "stop",
                "model": "Qwen3.5-397B-A17B-W8A8-INT8",
            }
            for case in self.dataset["cases"]
        }
        report = build_report(
            self.dataset,
            self.profile,
            responses,
            {"served_model_name": "Qwen3.5-397B-A17B-W8A8-INT8"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_report_files(report, output_dir, include_raw_outputs=True)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "result.json",
                    "summary.md",
                    "junit.xml",
                    "result.raw.json",
                    "label_candidates.jsonl",
                },
            )
            raw = json.loads((output_dir / "result.raw.json").read_text())
            public = json.loads((output_dir / "result.json").read_text())
            candidates = [
                json.loads(line)
                for line in (output_dir / "label_candidates.jsonl")
                .read_text()
                .splitlines()
            ]

        self.assertEqual({case["final_answer"] for case in raw["cases"]}, {"答案是3"})
        self.assertNotIn("final_answer", public["cases"][0])
        self.assertEqual({row["candidate_answer"] for row in candidates}, {"答案是3"})


if __name__ == "__main__":
    unittest.main()
