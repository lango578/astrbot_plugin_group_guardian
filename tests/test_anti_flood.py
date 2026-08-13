"""Regression tests for anti-flood repeat and rate detection."""

import unittest
from unittest.mock import patch

import anti_flood


AntiFloodMixin = anti_flood.AntiFloodMixin


class _Harness(AntiFloodMixin):
    def __init__(self, **overrides):
        self.values = {
            "anti_flood_rate_per_second": 0,
            "anti_flood_rate_per_minute": 0,
            "anti_flood_rate_per_hour": 0,
            "anti_flood_night_enabled": False,
            "repeat_detect_enabled": True,
            "repeat_detect_window_seconds": 120,
            "repeat_detect_count": 3,
            "long_text_detect_enabled": False,
            "long_text_threshold": 0,
        }
        self.values.update(overrides)
        self._init_anti_flood()

    def _cfg(self, key, default=False, group_id=None):
        return self.values.get(key, default)

    def _cfg_int(self, key, default=0, group_id=None):
        return int(self.values.get(key, default))


class AntiFloodRepeatTests(unittest.TestCase):
    @staticmethod
    def _record(harness, *texts):
        with patch.object(anti_flood.time, "time", return_value=1000.0):
            for index, value in enumerate(texts, start=1):
                harness._record_message("100", "200", str(index), value)
            return harness._check_anti_flood("100", "200")

    def test_separate_images_do_not_trigger_repeat_detection(self):
        harness = _Harness(repeat_detect_count=5)

        detected, info = self._record(
            harness, "[图片]", "[图片]", "[图片]", "[图片]", "[图片]"
        )

        self.assertFalse(detected)
        self.assertIsNone(info)

    def test_multiple_media_placeholders_are_not_repeat_keys(self):
        placeholders = (
            "[图片][图片]",
            " [图片] \n [商城表情] [表情] ",
        )
        for value in placeholders:
            with self.subTest(value=value):
                detected, info = self._record(_Harness(), value, value, value)
                self.assertFalse(detected)
                self.assertIsNone(info)

    def test_repeated_text_still_triggers_on_configured_count(self):
        detected, info = self._record(
            _Harness(), "same message", "same message", "same message"
        )

        self.assertTrue(detected)
        self.assertEqual(info["rate"], "重复消息")
        self.assertEqual(info["count"], 3)
        self.assertEqual(info["msg_ids"], ["3", "2", "1"])

    def test_text_with_image_remains_repeatable(self):
        detected, info = self._record(
            _Harness(), "same caption[图片]", "same caption[图片]", "same caption[图片]"
        )

        self.assertTrue(detected)
        self.assertEqual(info["rate"], "重复消息")

    def test_latest_media_does_not_fall_back_to_older_text(self):
        detected, info = self._record(
            _Harness(repeat_detect_count=2), "older text", "older text", "[图片]"
        )

        self.assertFalse(detected)
        self.assertIsNone(info)

    def test_images_still_count_toward_rate_limits(self):
        harness = _Harness(anti_flood_rate_per_second=2, repeat_detect_count=2)

        detected, info = self._record(harness, "[图片]", "[图片]", "[图片]")

        self.assertTrue(detected)
        self.assertEqual(info["rate"], "每秒")
        self.assertEqual(info["count"], 3)


if __name__ == "__main__":
    unittest.main()
