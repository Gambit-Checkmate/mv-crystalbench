"""A real local HTTP transport test; no external model endpoints are contacted."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from evaluate import evaluate_questions
from summarize import load_results, summarize_records


class LocalTransportTest(unittest.TestCase):
    def test_three_conditions_retry_usage_prompt_order_and_resume(self):
        requests = []
        request_lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with request_lock:
                    requests.append((self.path, self.headers.get("Authorization"), payload))
                    is_first = len(requests) == 1
                if is_first:
                    self.send_response(429)
                    content = {"error": {"message": "test-only throttling"}}
                else:
                    self.send_response(200)
                    content = {
                        "model": "returned-local-mock", "choices": [
                            {"message": {"content": "Answer: C"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                    }
                encoded = json.dumps(content).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(
                    os.environ, {"LOCAL_TEST_TOKEN": "test-only-key", "no_proxy": "127.0.0.1"}):
                root = Path(directory)
                Image.new("RGB", (16, 16), "white").save(root / "view.png")
                question = {
                    "id": "transport", "qtype": "Q4", "difficulty": "symmetry",
                    "structure": "PRIVATE_INTERNAL_NAME",
                    "question": "Select the symmetry class.",
                    "options": {"A": "Triclinic", "B": "Monoclinic", "C": "Cubic", "D": "Hexagonal"},
                    "answer": "C", "answer_text": "Cubic",
                    "meta": {"base_structure": "base-only-in-results", "text_applicable": True},
                    "image_paths": ["view.png"], "view_ids": ["scene:v00"],
                    "scene_id": "scene", "text_repr": "a=1 b=1 c=1\n0 C 0 0 0",
                }
                config = {
                    "alias": "local-mock", "provider": "offline-test", "model": "requested-mock",
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "api_key_env": "LOCAL_TEST_TOKEN", "temperature": 0,
                    "max_tokens": 32, "supports_multi_image": True,
                }
                output = root / "mock.jsonl"
                kwargs = dict(data_dir=root, conditions=("image", "text", "image+text"),
                              concurrency=2, timeout=5, max_retries=5, backoff_base=0)
                rows = asyncio.run(evaluate_questions([question], config, output, **kwargs))
                self.assertEqual(len(requests), 4)
                self.assertTrue(all(row["correct"] for row in rows))
                self.assertTrue(all(row["response_model"] == "returned-local-mock" for row in rows))
                self.assertTrue(all(row["usage"]["total_tokens"] == 13 for row in rows))
                self.assertTrue(all(row["base_structure"] == "base-only-in-results" for row in rows))
                self.assertEqual(sum(row["attempts"] for row in rows), 4)
                for path, authorization, payload in requests:
                    self.assertEqual(path, "/v1/chat/completions")
                    self.assertEqual(authorization, "Bearer test-only-key")
                    self.assertEqual(payload["temperature"], 0)
                    serialized = json.dumps(payload)
                    self.assertNotIn("PRIVATE_INTERNAL_NAME", serialized)
                    self.assertNotIn("base-only-in-results", serialized)
                    content = payload["messages"][0]["content"]
                    if isinstance(content, list):
                        self.assertEqual(content[0]["type"], "text")  # image label only
                        self.assertEqual(content[1]["type"], "image_url")
                        self.assertNotIn("symmetry class", content[0]["text"])
                        self.assertIn("symmetry class", content[-1]["text"])
                before = output.read_bytes()
                again = asyncio.run(evaluate_questions([question], config, output, **kwargs))
                self.assertEqual(len(requests), 4)
                self.assertEqual(rows, again)
                self.assertEqual(before, output.read_bytes())
                summaries = summarize_records(load_results([output]))
                overall = [row for row in summaries if row["scope"] == "overall"]
                self.assertEqual(len(overall), 3)
                self.assertTrue(all(row["accuracy"] == 1 for row in overall))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
