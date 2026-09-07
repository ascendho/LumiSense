from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mini_cares"
for relative in (
    "code/sft",
    "code/kd",
    "code/wise_ft",
    "code/quantization",
    "code/cascade",
    "code/baselines",
):
    sys.path.insert(0, str(ROOT / relative))

from lumisense_baselines.features import event_text, extract_numeric_features  # noqa: E402
from lumisense_baselines.prompt import export_fewshot_demos  # noqa: E402
from lumisense_cascade.cascade import risk_curve, select_threshold  # noqa: E402
from lumisense_cascade.gguf import restricted_confidence  # noqa: E402
from lumisense_kd.distill import tokenize_training_records  # noqa: E402
from lumisense_kd.gates import teacher_gate  # noqa: E402
from lumisense_kd.prompt import training_record as kd_training_record  # noqa: E402
from lumisense_kd.teacher import class_token_ids, teacher_response_accepted  # noqa: E402
from lumisense_quant.benchmark import docker_benchmark_command, parse_memory_bytes  # noqa: E402
from lumisense_quant.export import gguf_export_commands, llama_server_command  # noqa: E402
from lumisense_quant.gates import quantization_gate  # noqa: E402
from lumisense_sft.data_io import CLASS_IDS, load_pairs  # noqa: E402
from lumisense_sft.metrics import evidence_checks, parse_response, score_predictions  # noqa: E402
from lumisense_sft.prompt import SYSTEM_PROMPT, inference_messages, training_record  # noqa: E402
from prepare_training_data import prepare_training_data  # noqa: E402


class FakeTokenizer:
    class_map = {"A": 10, "B": 11, "C": 12, "D": 13}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [self.class_map[text]]

    def apply_chat_template(self, messages, **options):
        assert options["tokenize"] is True
        if options["add_generation_prompt"]:
            return [1, 2, 3]
        class_id = json.loads(messages[-1]["content"])["class_id"]
        return [1, 2, 3, 90, self.class_map[class_id], 91]


class PublicWorkflowTests(unittest.TestCase):
    def test_fixture_shape_and_prompt_contract(self) -> None:
        records, labels = load_pairs(FIXTURE, "train")
        self.assertEqual(len(records), 4)
        self.assertEqual({item["class_id"] for item in labels}, set("ABCD"))
        self.assertEqual(inference_messages(records[0])[0]["content"], SYSTEM_PROMPT)

        item = training_record(records[0], labels[0])
        self.assertEqual(item["class_id"], labels[0]["class_id"])
        self.assertEqual(item["messages"][-1]["role"], "assistant")

    def test_metrics_require_strict_grounded_json(self) -> None:
        records, labels = load_pairs(FIXTURE, "validation")
        fact = labels[0]["target"]["evidence"][0]
        response = {
            "class_id": labels[0]["class_id"],
            "evidence": [{"path": fact["path"], "value": fact["value"]}],
        }
        parsed = parse_response(json.dumps(response))
        self.assertEqual(parsed, response)
        checks = evidence_checks(
            parsed,
            records[0]["diagnostic_snapshot"],
            set(labels[0]["acceptable_evidence_paths"]),
        )
        self.assertTrue(all(item["grounded"] and item["relevant"] for item in checks))
        metrics = score_predictions(
            records,
            labels,
            [
                {"sample_id": records[0]["sample_id"], "response": json.dumps(response)},
                *[
                    {"sample_id": record["sample_id"], "response": "not-json"}
                    for record in records[1:]
                ],
            ],
        )
        self.assertLess(metrics["strict_output_rate"], 1.0)

    def test_sft_preparation_uses_train_and_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = prepare_training_data(
                FIXTURE,
                output_dir,
                samples_per_class=None,
                selection_seed=17,
            )
            self.assertEqual(manifest["splits"]["train"]["samples"], 4)
            self.assertEqual(manifest["splits"]["validation"]["samples"], 4)
            self.assertFalse((output_dir / "test.jsonl").exists())

    def test_baseline_feature_and_demo_helpers(self) -> None:
        records, _ = load_pairs(FIXTURE, "train")
        snapshot = records[0]["diagnostic_snapshot"]
        self.assertTrue(extract_numeric_features(snapshot))
        self.assertIsInstance(event_text(snapshot), str)
        with tempfile.TemporaryDirectory() as directory:
            payload = export_fewshot_demos(
                data_dir=FIXTURE,
                output=Path(directory) / "demos.json",
                shots_per_class=1,
                seed=17,
            )
            self.assertEqual(len(payload["demos"]), 4)

    def test_kd_class_token_and_teacher_gate(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(class_token_ids(tokenizer), [10, 11, 12, 13])
        record = {
            "sample_id": "one",
            "label": "PROCESS_ANOMALY",
            "class_id": "A",
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": '{"class_id":"A","evidence":[]}'},
            ],
            "teacher_logits": {"A": 2.0, "B": 0.0, "C": -1.0, "D": -2.0},
            "teacher_accepted": True,
        }
        tokenized = tokenize_training_records([record], tokenizer)
        self.assertEqual(tokenized[0][2], 4)
        self.assertEqual(tokenized[0][4], 1.0)

        records, labels = load_pairs(FIXTURE, "validation")
        fact = labels[0]["target"]["evidence"][0]
        response = {"class_id": labels[0]["class_id"], "evidence": [fact]}
        self.assertTrue(teacher_response_accepted(response, records[0], labels[0]))
        with self.assertRaises(ValueError):
            kd_training_record(records[0], labels[0], target={"class_id": "Z", "evidence": []})

        summary = {
            "accepted_rate": 1.0,
            "minimum_class_accepted_rate": 1.0,
            "raw_metrics": {
                "macro_f1": 1.0,
                "strict_output_rate": 1.0,
                "grounded_evidence_precision": 1.0,
                "relevant_evidence_precision": 1.0,
            },
        }
        self.assertTrue(teacher_gate(summary, summary)["passed"])

    def test_quantization_commands_and_gate(self) -> None:
        command = docker_benchmark_command(
            model_path=Path("/tmp/student.gguf"),
            profile="tiny_gateway",
            image="llama:test",
            container_name="lumisense-test",
        )
        joined = " ".join(command)
        self.assertIn("--memory 2g", joined)
        self.assertNotIn("--gpus", joined)
        self.assertEqual(parse_memory_bytes("123.5MiB / 2GiB"), int(123.5 * 1024**2))

        export = gguf_export_commands(
            model_path=Path("/models/08b"),
            adapter_path=Path("/runs/adapter"),
            output_dir=Path("/runs/export"),
            llama_cpp=Path("/tools/llama.cpp"),
            quantization="Q4_K_M",
        )
        self.assertIn("lumisense_quant.merge_lora_hf", " ".join(export["commands"][0]))
        self.assertEqual(export["commands"][2][-1], "Q4_K_M")

        server = llama_server_command(
            server=Path("/tools/llama-server"),
            model=Path("/models/student.gguf"),
            port=18080,
            threads=2,
        )
        self.assertEqual(server[-1], "0")
        self.assertTrue(
            quantization_gate(
                {"metrics": {"macro_f1": 0.95}},
                {"metrics": {"macro_f1": 0.94, "strict_output_rate": 1.0}},
            )["passed"]
        )

    def test_cascade_confidence_and_threshold_selection(self) -> None:
        predicted, confidence = restricted_confidence({"A": 0.2, "B": 0.6, "C": 0.1})
        self.assertEqual(predicted, "B")
        self.assertAlmostEqual(confidence, 0.6 / 0.9)

        rows = [
            {"sample_id": "one", "gold_class_id": "A", "class_id": "A", "confidence": 0.9, "correct": True},
            {"sample_id": "two", "gold_class_id": "B", "class_id": "A", "confidence": 0.7, "correct": False},
            {"sample_id": "three", "gold_class_id": "C", "class_id": "C", "confidence": 0.6, "correct": True},
            {"sample_id": "four", "gold_class_id": "D", "class_id": "D", "confidence": 0.4, "correct": True},
        ]
        cloud = {"one": "A", "two": "B", "three": "C", "four": "D"}
        selected = select_threshold(risk_curve(rows, cloud))
        self.assertEqual(selected["tau"], 0.9)
        self.assertEqual(selected["cascade_macro_f1"], 1.0)

    def test_wise_ft_helpers(self) -> None:
        from lumisense_wise.adapter import alpha_label, read_config

        self.assertEqual(alpha_label(0.5), "a05")
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text('{"lora_parameters":{"rank":16}}\n')
            self.assertEqual(read_config(adapter)["lora_parameters"]["rank"], 16)


if __name__ == "__main__":
    unittest.main()
