"""GPU-free checks of DSA Python/C++ FFI contracts."""

import ast
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsaFusedStoreFfi(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        kernels = Path(__file__).resolve().parents[4] / "python/sglang/kernels"
        cls.wrapper_tree = ast.parse(
            (kernels / "ops/kvcache/dsa_fused_store_index_cache.py").read_text()
        )
        cls.header = (kernels / "jit/csrc/dsa/fused_store_index_cache.cuh").read_text()

    def _ffi_contract(self, name):
        functions = {
            node.name: node
            for node in self.wrapper_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        wrapper = functions[name]
        module_assignment = next(
            node
            for node in wrapper.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "module" for t in node.targets)
        )
        factory = functions[module_assignment.value.func.id]
        load = next(
            node
            for node in ast.walk(factory)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_jit"
        )
        exports = next(k.value for k in load.keywords if k.arg == "cuda_wrappers")
        target = next(
            pair.elts[1] for pair in exports.elts if pair.elts[0].value == name
        )
        cpp_class = target.values[0].value.split("<", 1)[0]
        body = self.header.split(f"struct {cpp_class} {{", 1)[1].split("\n};", 1)[0]
        signature = re.search(r"static\s+void\s+run\s*\(([^)]*)\)", body)
        self.assertIsNotNone(signature)
        parameters = [p.strip() for p in signature.group(1).split(",")]
        call = next(
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "module"
            and node.func.attr == name
        )
        self.assertEqual(len(parameters), len(call.args), f"{name}: Python/C++ arity")
        self.assertEqual(call.keywords, [])
        return parameters, call, body

    def test_fp8_forwards_dcp_geometry_to_the_kernel(self):
        parameters, call, body = self._ffi_contract("fused_store_index_k_cache")
        self.assertEqual(parameters[3:], ["int64_t dcp_size", "int64_t dcp_rank"])
        self.assertEqual(
            [ast.unparse(arg) for arg in call.args[3:]], ["dcp_size", "dcp_rank"]
        )
        self.assertRegex(body, r"\.dcp_size\s*=\s*dcp_size\b")
        self.assertRegex(body, r"\.dcp_rank\s*=\s*dcp_rank\b")

    def test_mxfp4_keeps_the_three_tensor_interface(self):
        parameters, _, _ = self._ffi_contract("fused_store_index_k_mxfp4_cache")
        self.assertEqual(
            parameters,
            [
                "tvm::ffi::TensorView input",
                "tvm::ffi::TensorView cache",
                "tvm::ffi::TensorView indices",
            ],
        )


class TestTopkCandidateFfi(CustomTestCase):
    def test_ragged_parameter_types_compile_without_redefinitions(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required for the header syntax check")
        header = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh"
        ).read_text()
        declarations = re.findall(
            r"^struct (?:PageTransform|TopKRaggedKpoolParams) \{.*?^\};",
            header,
            re.MULTILINE | re.DOTALL,
        )
        source = (
            "#include <cstdint>\n#define SGL_DEVICE\n"
            + "\n".join(declarations)
            + "\nstatic_assert(sizeof(TopKRaggedKpoolParams) > 0);\n"
        )
        result = subprocess.run(
            [compiler, "-std=c++20", "-fsyntax-only", "-x", "c++", "-"],
            input=source,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_block_and_dcp_candidates_keep_distinct_bindings(self):
        """Candidate blocks return indices; DCP returns score/index pairs."""
        kernels = Path(__file__).resolve().parents[4] / "python/sglang/kernels"
        tree = ast.parse((kernels / "ops/attention/dsv4/topk.py").read_text())
        header = (kernels / "jit/csrc/deepseek_v4/topk_v2.cuh").read_text()
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        exports = next(
            node.value
            for node in functions["_jit_topk_v2_module"].body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "wrappers" for t in node.targets
            )
        )
        names = [pair.elts[0].value for pair in exports.elts]
        self.assertEqual(len(names), len(set(names)))
        for public_name, ffi_name, cpp_name, arity, output_shape in (
            (
                "topk_transform_candidates",
                "topk_transform_candidates",
                "transform_candidates",
                5,
                "{B, K}",
            ),
            (
                "topk_transform_candidates_v2",
                "topk_transform_candidates_dcp",
                "transform_candidates_dcp",
                6,
                "{B, K, 2}",
            ),
        ):
            with self.subTest(public_name=public_name):
                target = next(
                    pair.elts[1]
                    for pair in exports.elts
                    if pair.elts[0].value == ffi_name
                )
                self.assertEqual(target.values[-1].value, "::" + cpp_name)
                call = next(
                    node
                    for node in ast.walk(functions[public_name])
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == ffi_name
                )
                signature = re.search(
                    r"static\s+void\s+" + cpp_name + r"\s*\(([^)]*)\)", header
                )
                self.assertIsNotNone(signature)
                self.assertEqual(len(signature.group(1).split(",")), arity)
                self.assertEqual(len(call.args), arity)
                body = header[signature.end() :].split("\n  }", 1)[0]
                self.assertIn("TensorMatcher(" + output_shape + ")", body)


if __name__ == "__main__":
    unittest.main()
