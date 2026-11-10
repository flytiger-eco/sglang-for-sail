"""Pruning spatial selection; handoff cases live in unit/disaggregation."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from sglang.test.test_utils import CustomTestCase

ROOT = Path(__file__).resolve().parents[3] / 'python/sglang/srt'


def extract(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in found:
        node.decorator_list = []
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + found, type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), 'exec'), namespace)
    return namespace


class Positions:
    def __init__(self, rows): self.rows = rows
    def __getitem__(self, key):
        return Positions([[row[i] for i in key[1]] for row in self.rows])


class SelectionTests(CustomTestCase):
    def setUp(self):
        self.select = extract('multimodal/processors/qwen_vl.py', {'_select_epd_image_tokens'}, {'Modality': NS(IMAGE='image')})['_select_epd_image_tokens']

    def test_two_images_and_video_original_coordinates(self):
        rows = [[i * axis for i in range(15)] for axis in (1, 2, 3)]
        ids, offsets, positions, delta = self.select(list(range(15)), [(1, 4), (6, 8), (10, 12)], ['image', 'video', 'image'], Positions(rows), 20, [[0, 3], [1]])
        keep = [0, 1, 4, 5, 6, 7, 8, 9, 11, 13, 14]
        self.assertEqual(ids, keep)
        self.assertEqual(offsets, [(1, 2), (4, 6), (8, 8)])
        self.assertEqual(positions.rows, [[row[i] for i in keep] for row in rows])
        self.assertEqual(len(ids) + delta, 15 + 20)
        for output_index in range(5):
            self.assertEqual(len(ids) + delta + output_index, 35 + output_index)

    def test_identity(self):
        result = self.select([1, 2, 3], [(1, 2)], ['image'], Positions([[0, 1, 2]] * 3), -1, [[0, 1]])
        self.assertEqual(result[0], [1, 2, 3]); self.assertEqual(result[3], -1)

    def test_malformed(self):
        for indices in ([], [[]], [[1, 0]], [[0, 0]], [[2]], [[True]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.select([1, 2], [(0, 1)], ['image'], Positions([[0, 1]] * 3), 0, indices)


if __name__ == '__main__': unittest.main()
