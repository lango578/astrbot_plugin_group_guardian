# -*- coding: utf-8 -*-
import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARAM_PATTERN = re.compile(
    r"^\s*(?P<name>\w+)\((?P<type>\w+)(?:\[(?P<item_type>\w+)\])?\):",
    re.MULTILINE,
)
PY_TO_JSON_TYPE = {"list": "array"}


def _class_methods(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _parameter_schema(method, parameter_name):
    docstring = ast.get_docstring(method) or ""
    for match in PARAM_PATTERN.finditer(docstring):
        if match.group("name") != parameter_name:
            continue
        parameter_type = PY_TO_JSON_TYPE.get(match.group("type"), match.group("type"))
        schema = {"type": parameter_type}
        item_type = match.group("item_type")
        if parameter_type == "array" and item_type:
            schema["items"] = {
                "type": PY_TO_JSON_TYPE.get(item_type, item_type),
            }
        return schema
    raise AssertionError(
        "{} is missing from the Args section of {}".format(
            parameter_name, method.name
        )
    )


def _registered_tool_name(method):
    for decorator in method.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        if not isinstance(target, ast.Attribute) or target.attr != "llm_tool":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
    return None


def _annotation_is_list_of_strings(method, parameter_name):
    parameter = next(arg for arg in method.args.args if arg.arg == parameter_name)
    annotation = parameter.annotation
    if not isinstance(annotation, ast.Subscript):
        return False
    if not isinstance(annotation.value, ast.Name) or annotation.value.id != "List":
        return False
    item_annotation = annotation.slice
    if hasattr(ast, "Index") and isinstance(item_annotation, ast.Index):
        item_annotation = item_annotation.value
    return isinstance(item_annotation, ast.Name) and item_annotation.id == "str"


class LlmToolSchemaTests(unittest.TestCase):
    TOOL_METHODS = {
        "batch_ban_members_tool": "batch_ban_members",
        "batch_kick_members_tool": "batch_kick_members",
    }

    def test_registered_batch_tools_emit_string_array_schema(self):
        methods = _class_methods(ROOT / "main.py", "Main")
        expected = {"type": "array", "items": {"type": "string"}}

        for method_name, tool_name in self.TOOL_METHODS.items():
            with self.subTest(method=method_name):
                self.assertEqual(_registered_tool_name(methods[method_name]), tool_name)
                self.assertEqual(
                    _parameter_schema(methods[method_name], "user_ids"), expected
                )

    def test_registered_batch_tool_annotations_match_schema(self):
        methods = _class_methods(ROOT / "main.py", "Main")

        for method_name in self.TOOL_METHODS:
            with self.subTest(method=method_name):
                self.assertTrue(
                    _annotation_is_list_of_strings(
                        methods[method_name], "user_ids"
                    )
                )

    def test_business_docstrings_match_registered_schema(self):
        methods = _class_methods(ROOT / "llm_tools.py", "LlmToolsMixin")
        expected = {"type": "array", "items": {"type": "string"}}

        for method_name in self.TOOL_METHODS:
            with self.subTest(method=method_name):
                self.assertEqual(
                    _parameter_schema(methods[method_name], "user_ids"), expected
                )


if __name__ == "__main__":
    unittest.main()
