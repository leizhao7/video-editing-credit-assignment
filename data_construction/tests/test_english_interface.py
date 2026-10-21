"""User-facing English interfaces, without invoking models or cloud services."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
CONSTRUCTION = REPO / "data_construction"
sys.path.insert(0, str(CONSTRUCTION))
import annotation_script_stages as script
import annotation_s4_stages as cuts
import quality_tool_contract as contract

HAN = re.compile(r"[\u3400-\u9fff]")


class EnglishInterfaceTests(unittest.TestCase):
    def test_cli_help_is_english_without_credentials(self):
        files = sorted(p for p in CONSTRUCTION.glob("*.py")
                       if "Usage: python data_construction/" in p.read_text())
        files += sorted((CONSTRUCTION / "modal").glob("*.py"))
        files += [REPO / "video_editing.py", REPO / "evaluation/task_pipeline.py"]
        self.assertGreater(len(files), 20)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("VIDEO_EDITING_", "S12_", "GOOGLE_AI_STUDIO_API_KEY", "AZURE_"))}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for path in files:
            with self.subTest(script=path.name):
                result = subprocess.run([sys.executable, str(path), "--help"], env=env,
                                        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsNone(HAN.search(result.stdout + result.stderr), path.name)

    def test_all_annotation_stages_request_english_and_preserve_input_text(self):
        prompts = [
            script.build_shot_prompt("/private/input.json", {
                "manifest_path": "/private/manifest.json", "draft_path": "/private/draft.json"}),
            script.build_structure_prompt("/private/frozen.json", "/private/input.json", "/private/structure.json"),
            cuts.build_cut_prompt("/private/batch", "/private/output.json", "cut: Direct cut."),
            (CONSTRUCTION / "prompts/PROMPT_SCRIPT.md").read_text(),
            (CONSTRUCTION / "prompts/PROMPT_S4.md").read_text(),
        ]
        for prompt in prompts:
            self.assertIsNone(HAN.search(prompt))
            self.assertIn("English", prompt)
            self.assertIn("preserve", prompt.lower())
        self.assertIn("insert its exact text/null mechanically", prompts[0])
        self.assertIn("supplied dialogue", prompts[2])

    def test_report_reads_both_new_and_historical_vocabulary_headings(self):
        # Extract this pure helper without executing report.main or touching media.
        tree = ast.parse((CONSTRUCTION / "build_report.py").read_text())
        helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_vocab")
        module = ast.Module(body=[helper], type_ignores=[])
        namespace = {"re": re}
        exec(compile(ast.fix_missing_locations(module), "report-vocabulary", "exec"), namespace)
        template = (CONSTRUCTION / "prompts/PROMPT_S4.md").read_text()
        result = namespace["_vocab"](template)
        self.assertIn("no_change", result["transition"])
        self.assertIn("match_cut", result["technique"])
        self.assertNotIn("match_cut", result["transition"])
        historical = template.replace("### Transition vocabulary", "### 一、 Transition vocabulary").replace(
            "### Technique vocabulary", "### 二、 Technique vocabulary")
        self.assertEqual(namespace["_vocab"](historical), result)

    def test_tool_instructions_and_submission_policy_are_english(self):
        sys.path.insert(0, str(REPO / "evaluation"))
        import submission_policy
        self.assertIsNone(HAN.search(json.dumps(contract.FUNCTIONS, ensure_ascii=False)))
        self.assertIsNone(HAN.search(contract.TOOLS_FALLBACK_SYSTEM))
        # Public instructions must retain the validated delivery contract.
        policy_strings = [n.value for n in ast.walk(ast.parse(Path(submission_policy.__file__).read_text()))
                          if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertIsNone(HAN.search("\n".join(policy_strings)))
        self.assertTrue(any("10%" in text and "3 seconds" in text for text in policy_strings))


if __name__ == "__main__":
    unittest.main()
