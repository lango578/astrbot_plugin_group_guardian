import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConfigSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(
            (ROOT / "_conf_schema.json").read_text(encoding="utf-8")
        )
        tree = ast.parse((ROOT / "web.py").read_text(encoding="utf-8"))
        cls.web_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "WebMixin"
        )
        cls.dashboard = (
            ROOT / "pages" / "dashboard" / "index.html"
        ).read_text(encoding="utf-8")

    @classmethod
    def _class_literal(cls, name):
        assignment = next(
            node
            for node in cls.web_class.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        )
        return ast.literal_eval(assignment.value)

    @classmethod
    def _static_return_literal(cls, name):
        function = next(
            node
            for node in cls.web_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        )
        return ast.literal_eval(next(
            node.value for node in function.body if isinstance(node, ast.Return)
        ))

    def test_full_message_moderation_is_optional_and_group_overridable(self):
        setting = self.schema["llm_moderation_always"]
        self.assertEqual(setting["type"], "bool")
        self.assertFalse(setting["default"])

        categories = self._class_literal("_CONFIG_CATEGORIES")
        excluded = self._class_literal("_GROUP_CONFIG_EXCLUDE")
        self.assertEqual(categories["llm_moderation_always"], "审核规则")
        self.assertNotIn("llm_moderation_always", excluded)
        self.assertIn("key: 'llm_moderation_always'", self.dashboard)
        self.assertIn("llm_moderation_always: true", self.dashboard)

        concurrency = self.schema["llm_max_concurrency"]
        self.assertEqual(concurrency["type"], "int")
        self.assertEqual(concurrency["default"], 12)
        self.assertIn("llm_max_concurrency", excluded)
        ranges = self._static_return_literal("_config_int_ranges")
        self.assertEqual(ranges["llm_max_concurrency"], (1, 32))

    def test_card_admin_exemption_is_enabled_and_exposed(self):
        setting = self.schema["card_audit_admin_exempt"]
        self.assertEqual(setting["type"], "bool")
        self.assertTrue(setting["default"])

        keys = self._class_literal("_CARD_MONITOR_KEYS")
        excluded = self._class_literal("_GROUP_CONFIG_EXCLUDE")
        self.assertIn("card_audit_admin_exempt", keys)
        self.assertNotIn("card_audit_admin_exempt", excluded)
        self.assertIn("'card_audit_admin_exempt'", self.dashboard)


if __name__ == "__main__":
    unittest.main()
