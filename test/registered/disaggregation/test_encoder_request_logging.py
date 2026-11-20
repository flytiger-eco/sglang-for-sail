"""JSON request logging must tolerate internal EPD modality maps."""
import dataclasses
import io
import json
import logging
import unittest
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.utils.request_logger import RequestLogger, _transform_data_for_logging


class RequestLoggingTests(unittest.TestCase):
    def test_early_encoder_metadata_is_logged_without_mutating_the_request(self):
        @dataclasses.dataclass
        class Request:
            rid: str
            num_items_assigned: dict
            text: str = "test"
            input_ids: object = None
        request = Request("test-id", {Modality.IMAGE: [12]})
        buffer = io.StringIO()
        target = logging.Logger("test.epd.json")
        target.addHandler(logging.StreamHandler(buffer))
        logger = RequestLogger(True, 0, "json", None)
        logger.targets = [target]
        logger.log_received_request(request)
        record = json.loads(buffer.getvalue())
        self.assertEqual(record["event"], "request.received")
        self.assertEqual(record["obj"]["num_items_assigned"][str(Modality.IMAGE)], [12])
        self.assertEqual(request.num_items_assigned, {Modality.IMAGE: [12]})

    def test_supported_keys_and_nested_enum_keys(self):
        data = {"nested": {Modality.IMAGE: {7: ["x"]}}, "normal": 1}
        result = _transform_data_for_logging(data)
        self.assertEqual(result["nested"][str(Modality.IMAGE)], {7: ["x"]})
        self.assertEqual(json.loads(json.dumps(result))["normal"], 1)
