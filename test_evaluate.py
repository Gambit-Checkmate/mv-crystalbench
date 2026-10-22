"""Offline evaluation tests: mock responses only; never contact a model API."""

import asyncio
import base64
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import evaluate


def question(identifier="q1", qtype="Q3", difficulty="hard", applicable=True):
    return {
        "id": identifier, "qtype": qtype, "difficulty": difficulty,
        "structure": "INTERNAL_SECRET_STRUCTURE", "base_structure": "BASE_SECRET",
        "scene_id": "scene", "view_ids": ["scene:v00", "scene:v01"],
        "image_paths": ["one.png", "two.png"],
        "question": "Which marked atom is closer to A in three-dimensional space?",
        "options": {"A": "B is closer", "B": "C is closer", "C": "Equally close", "D": "Cannot be determined"},
        "answer": "B", "answer_text": "C is closer",
        "meta": {"text_applicable": applicable, "depth_trap": True},
        "text_repr": "Lattice parameters: a=4 b=4 c=4\n0 C 0 0 0 [A]" if applicable else "",
    }


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "results.jsonl"
        for name, color in (("one.png", "red"), ("two.png", "blue")):
            Image.new("RGB", (32, 40), color).save(self.root / name)
        self.config = {
            "alias": "mock", "provider": "test", "model": "mock-vl",
            "base_url": "http://unused.invalid/v1", "api_key_env": "TEST_EVALUATE_KEY",
            "temperature": 0, "max_tokens": 100, "supports_multi_image": True,
        }
        self.environment = patch.dict(os.environ, {"TEST_EVALUATE_KEY": "test-only-not-real"})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def run_evaluation(self, questions=None, **kwargs):
        return asyncio.run(evaluate.evaluate_questions(
            questions or [question()], self.config, self.output, data_dir=self.root, **kwargs
        ))

    def test_strict_last_line_then_last_independent_uppercase_letter(self):
        cases = {
            "Maybe B.\nAnswer: A": "A", "Answer: D\n\n": "D", "Answer: E": None,
            "I select (C).": "C", "A first, B next, finally D.": "D",
            "ABCD": None, "CAD structure": None, "a possible choice": None,
            "abc_A": None, "12B": None, "": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(evaluate.parse_answer(text), expected)
        self.assertEqual(evaluate._parse_answer("Answer: A")[1], "final_answer_line")

    def test_example_model_configs_use_environment_variables_only(self):
        path = Path(__file__).with_name("models.yaml")
        for alias in ("aggregator", "qwen_vl", "local_vllm"):
            with self.subTest(alias=alias):
                config = evaluate.load_model_config(path, alias)
                self.assertNotIn("api_key", config)
                self.assertTrue(config["api_key_env"])
                self.assertEqual(config["temperature"], 0)
        qwen = evaluate.load_model_config(path, "qwen_vl")
        self.assertEqual(qwen["model"], "qwen3-vl-plus")
        self.assertTrue(qwen["base_url"].endswith("/compatible-mode/v1"))

    def test_conditions_have_only_requested_evidence_and_no_private_fields(self):
        q = question()
        image = evaluate.build_messages(q, "image", self.root, self.config)
        text = evaluate.build_messages(q, "text", self.root, self.config)
        combined = evaluate.build_messages(q, "image_text", self.root, self.config)
        self.assertNotIn("Lattice parameters", json.dumps(image))
        self.assertIn("Lattice parameters", json.dumps(text))
        self.assertIn("Lattice parameters", json.dumps(combined))
        self.assertNotIn("image_url", json.dumps(text))
        self.assertEqual(sum(p["type"] == "image_url" for p in image[0]["content"]), 2)
        self.assertEqual([p["type"] for p in image[0]["content"]],
                         ["text", "image_url", "text", "image_url", "text"])
        self.assertEqual(image[0]["content"][0]["text"], "Image 1")
        self.assertIn(q["question"], image[0]["content"][-1]["text"])
        self.assertTrue(image[0]["content"][-1]["text"].endswith(
            "Answer with a single letter (A, B, C, or D) on the last line in the format: Answer: X"
        ))
        for messages in (image, text, combined):
            serialized = json.dumps(messages)
            for forbidden in ("INTERNAL_SECRET_STRUCTURE", "BASE_SECRET", "depth_trap", "answer_text"):
                self.assertNotIn(forbidden, serialized)

    def test_q5_groups_and_single_horizontal_montage(self):
        q = question(qtype="Q5")
        q["meta"]["view_groups"] = [
            {"label": "private label ignored", "image_indices": [0]},
            {"label": "another private label", "image_indices": [1]},
        ]
        messages = evaluate.build_messages(
            q, "image", self.root, {**self.config, "supports_multi_image": False}
        )
        text = messages[0]["content"][-1]["text"]
        self.assertIn("Group 1, Image 1", text)
        self.assertIn("Group 2, Image 2", text)
        self.assertNotIn("private label", text)
        images = [part for part in messages[0]["content"] if part["type"] == "image_url"]
        self.assertEqual(len(images), 1)
        data = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.size, (64, 72))
            self.assertEqual(image.getpixel((1, 35)), (255, 0, 0))
            self.assertEqual(image.getpixel((33, 35)), (0, 0, 255))

    def test_dry_run_is_deterministic_and_skips_visual_only_text_without_network(self):
        questions = [question(), question("q2", "Q1a", applicable=False)]
        with patch.object(evaluate, "_http_post", side_effect=AssertionError("No network allowed")):
            first = self.run_evaluation(questions, dry_run=True, conditions=evaluate.CONDITIONS)
            self.output.unlink()
            second = self.run_evaluation(questions, dry_run=True, conditions=evaluate.CONDITIONS,
                                         concurrency=1)
        self.assertEqual([r["prediction"] for r in first], [r["prediction"] for r in second])
        q1_predictions = [r["prediction"] for r in first if r["question_id"] == "q1"]
        self.assertEqual(len(set(q1_predictions)), 1)
        skipped = [r for r in first if r["status"] == "skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertIsNone(skipped[0]["correct"])
        self.assertEqual(skipped[0]["attempts"], 0)
        self.assertTrue(all(r["dry_run"] for r in first))

    def test_retry_success_invalid_and_result_metadata(self):
        responses = [
            evaluate.APIError("rate limit", retryable=True, status_code=429),
            {"model": "actual-vl-version", "choices": [{"message": {"content": "Answer: B"}}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}},
        ]
        with patch.object(evaluate, "_http_post", side_effect=responses) as post:
            records = self.run_evaluation(backoff_base=0)
        record = records[0]
        self.assertEqual(post.call_count, 2)
        self.assertEqual(record["attempts"], 2)
        self.assertTrue(record["correct"])
        self.assertEqual(record["response_model"], "actual-vl-version")
        self.assertEqual(record["usage"]["total_tokens"], 15)
        self.assertEqual(record["base_structure"], "BASE_SECRET")
        self.assertEqual(record["meta"]["depth_trap"], True)
        self.assertFalse(record["invalid"])
        self.assertEqual(record["temperature"], 0)
        self.output.unlink()
        with patch.object(evaluate, "_http_post", return_value={
            "choices": [{"message": {"content": "I cannot decide."}}]
        }):
            invalid = self.run_evaluation()[0]
        self.assertEqual(invalid["status"], "invalid_response")
        self.assertTrue(invalid["invalid"])
        self.assertFalse(invalid["correct"])
        self.assertIsNone(invalid["prediction"])

    def test_completed_resume_also_skips_errors_and_rejects_changed_dataset(self):
        with patch.object(evaluate, "_http_post", side_effect=evaluate.APIError(
            "bad request", retryable=False, status_code=400
        )) as post:
            record = self.run_evaluation()[0]
        self.assertEqual(post.call_count, 1)
        self.assertEqual(record["status"], "error")
        with patch.object(evaluate, "_http_post", side_effect=AssertionError("Already recorded")) as post:
            resumed = self.run_evaluation()
            again = self.run_evaluation()
        self.assertEqual(post.call_count, 0)
        self.assertEqual(resumed[0]["status"], "error")
        self.assertEqual(again, resumed)
        self.assertEqual(len(self.output.read_text("utf-8").splitlines()), 1)
        changed = question()
        changed["question"] = "Changed experimental prompt?"
        with self.assertRaisesRegex(ValueError, "different dataset/configuration"):
            self.run_evaluation([changed])

    def test_maximum_five_retries_then_error(self):
        with patch.object(evaluate, "_http_post", side_effect=evaluate.APIError(
            "timeout", retryable=True
        )) as post:
            record = self.run_evaluation(max_retries=5, backoff_base=0)[0]
        self.assertEqual(post.call_count, 6)
        self.assertEqual(record["attempts"], 6)
        self.assertEqual(record["status"], "error")
        self.assertFalse(record["invalid"])

    def test_majority_uses_full_answer_text_per_difficulty_and_absence_is_wrong(self):
        full = []
        for index, (difficulty, answer) in enumerate(
            [("easy", "6")] * 3 + [("hard", "8")] * 4 + [("hard", "6")]
        ):
            q = question(str(index), "Q2", difficulty)
            q["answer_text"] = answer
            full.append(q)
        self.assertEqual(evaluate.majority_answers(full), {"Q2:easy": "6", "Q2:hard": "8"})
        target = question(qtype="Q2", difficulty="hard")
        target["options"] = {"A": "2", "B": "6", "C": "4", "D": "12"}
        target["answer"] = "B"
        target["answer_text"] = "6"
        with patch.object(evaluate, "_http_post", side_effect=AssertionError("No API")):
            result = self.run_evaluation([target], baseline="majority", full_questions=full)[0]
        self.assertIsNone(result["prediction"])
        self.assertEqual(result["prediction_text"], "8")
        self.assertFalse(result["correct"])
        self.assertFalse(result["invalid"])
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["baseline_detail"]["majority_absent"])

    def test_concurrency_limit_and_limit_order(self):
        counts = {"active": 0, "peak": 0}

        async def fake_thread(function, *args):
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            await asyncio.sleep(0)
            counts["active"] -= 1
            return {"choices": [{"message": {"content": "Answer: B"}}]}

        with patch.object(evaluate.asyncio, "to_thread", side_effect=fake_thread):
            results = self.run_evaluation(
                [question(f"q{i}") for i in range(8)], concurrency=2, limit=5
            )
        self.assertEqual(len(results), 5)
        self.assertEqual(counts["peak"], 2)

    def test_config_change_rejected_and_no_resume_never_overwrites(self):
        self.run_evaluation(dry_run=True)
        original = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.run_evaluation(dry_run=True, resume=False)
        changed = copy.deepcopy(self.config)
        changed["temperature"] = 1
        with self.assertRaisesRegex(ValueError, "different dataset/configuration"):
            asyncio.run(evaluate.evaluate_questions(
                [question()], changed, self.output, data_dir=self.root, dry_run=True
            ))
        self.assertEqual(self.output.read_bytes(), original)

    def test_resume_recovers_only_an_interrupted_final_record_with_backup(self):
        questions = [question("q1"), question("q2")]
        first = self.run_evaluation(questions, dry_run=True, limit=1)
        with self.output.open("ab") as handle:
            handle.write(b'{"question_id": "q2", "raw_response":')
        resumed = self.run_evaluation(questions, dry_run=True)
        self.assertEqual(len(resumed), 2)
        self.assertEqual(first[0]["prediction"], resumed[0]["prediction"])
        self.assertEqual(len(self.output.read_text("utf-8").splitlines()), 2)
        self.assertEqual(len(list(self.root.glob("*.interrupted-*.txt"))), 1)
        with self.output.open("ab") as handle:
            handle.write(b'{"bad completed line":\n')
        original = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "malformed completed line"):
            self.run_evaluation(questions, dry_run=True)
        self.assertEqual(self.output.read_bytes(), original)

    def test_dry_run_validates_images_and_reasoning_is_opt_in(self):
        q = question()
        q["image_paths"][0] = "missing.png"
        with patch.object(evaluate, "_http_post", side_effect=AssertionError("No API")):
            result = self.run_evaluation([q], dry_run=True)[0]
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "FileNotFoundError")
        prompt = evaluate.build_prompt(question(), "image")
        self.assertIn("Do not include reasoning", prompt)
        reasoning_prompt = evaluate.build_prompt(question(), "image", allow_reasoning=True)
        self.assertIn("brief explanation", reasoning_prompt)

    def test_base_structure_prefers_internal_meta(self):
        q = question()
        q["meta"]["base_structure"] = "meta-base"
        result = self.run_evaluation([q], dry_run=True)[0]
        self.assertEqual(result["base_structure"], "meta-base")

    def test_no_image_has_only_question_choices_and_runs_visual_only_tasks(self):
        questions = [question(f'q-{kind}', qtype=kind, applicable=kind not in {'Q1a', 'Q6'})
                     for kind in ('Q1a', 'Q1b', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6')]
        for q in questions:
            q['image_paths'] = ['missing.png']
            messages = evaluate.build_messages(q, 'no_image', self.root, self.config)
            prompt = messages[0]['content']
            self.assertIsInstance(prompt, str)
            self.assertIn(q['question'], prompt)
            for letter, option in q['options'].items():
                self.assertIn(f'{letter}. {option}', prompt)
            for private in ('Lattice parameters', 'image_url', 'Images are supplied', 'INTERNAL_SECRET', 'depth_trap'):
                self.assertNotIn(private, prompt)
        with patch.object(evaluate, '_http_post', side_effect=AssertionError('No API')):
            rows = self.run_evaluation(questions, conditions=('no_image',), dry_run=True)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row['status'] == 'ok' for row in rows))

    def test_provider_can_omit_temperature(self):
        self.config['omit_temperature'] = True
        def responder(config, payload, timeout):
            self.assertNotIn('temperature', payload)
            return {'choices':[{'message':{'content':'Answer: B'}}]}
        with patch.object(evaluate, '_http_post', side_effect=responder):
            row = self.run_evaluation(conditions=('no_image',))[0]
        self.assertEqual(row['status'], 'ok')
        self.assertIsNone(row['temperature'])

    def test_finish_reason_retained_for_empty_truncated_response(self):
        response = {'choices':[{'message':{'content':''},'finish_reason':'length'}],
                    'usage':{'prompt_tokens':100,'completion_tokens':1024}}
        with patch.object(evaluate,'_http_post',return_value=response):
            row = self.run_evaluation(conditions=('no_image',))[0]
        self.assertEqual(row['finish_reason'],'length')
        self.assertEqual(row['status'],'invalid_response')
        self.assertEqual(row['usage']['completion_tokens'],1024)

    def test_user_agent_header_is_forwarded_without_credential_override(self):
        self.config['headers'] = {'User-Agent':'Mozilla/5.0'}
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{}'
        def open_request(request, timeout):
            self.assertEqual(request.get_header('User-agent'), 'Mozilla/5.0')
            self.assertEqual(request.get_header('Authorization'), 'Bearer test-only-not-real')
            return FakeResponse()
        with patch.object(evaluate.urlrequest, 'urlopen', side_effect=open_request):
            self.assertEqual(evaluate._http_post(self.config, {}, 1), {})
        self.config['headers']['Authorization'] = 'bad'
        with self.assertRaises(evaluate.APIError):
            evaluate._http_post(self.config, {}, 1)

    def test_cli_all_defaults_to_one_result_file_per_condition(self):
        question_file = self.root / "questions.jsonl"
        question_file.write_text(json.dumps(question()) + "\n", encoding="utf-8")
        script = Path(evaluate.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(script), "--questions", str(question_file),
             "--dry-run", "--condition", "all"],
            cwd=self.root, capture_output=True, text=True, check=True,
        )
        paths = sorted((self.root / "results").glob("*.jsonl"))
        self.assertEqual({path.name for path in paths},
                         {"dry-run_image.jsonl", "dry-run_text.jsonl", "dry-run_image_text.jsonl", "dry-run_no_image.jsonl"})
        self.assertEqual({json.loads(path.read_text("utf-8"))["condition"] for path in paths},
                         {"image", "text", "image+text", "no_image"})
        self.assertEqual(result.stdout.count('"output"'), 4)


if __name__ == "__main__":
    unittest.main()
