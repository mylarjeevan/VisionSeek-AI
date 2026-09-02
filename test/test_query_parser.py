import unittest

from src.query_parser import builder_to_query, parse_query, tokenize
from src.search_engine import SearchFilters, eval_ast, structured_search


class QueryParserTests(unittest.TestCase):
    def test_and_or_not(self):
        and_q = parse_query("person AND motorcycle")
        self.assertIsNone(and_q.error)
        self.assertEqual(and_q.ast["operator"], "AND")
        self.assertEqual(and_q.mode_hint, "structured")

        or_q = parse_query("car OR motorcycle")
        self.assertEqual(or_q.ast["operator"], "OR")

        not_q = parse_query("person NOT car")
        self.assertEqual(not_q.ast["operator"], "AND")
        self.assertEqual(not_q.ast["conditions"][1]["operator"], "NOT")

        nested = parse_query("(person AND motorcycle) OR bicycle")
        self.assertEqual(nested.ast["operator"], "OR")

    def test_multiword_class(self):
        tokens = tokenize("dining table AND person")
        values = [t.value for t in tokens]
        self.assertIn("dining table", values)
        self.assertIn("AND", values)

    def test_semantic_hint(self):
        parsed = parse_query("person wearing a black jacket")
        self.assertEqual(parsed.mode_hint, "hybrid")
        self.assertIn("person", parsed.extracted_classes)
        self.assertTrue(parsed.unknown_terms)

    def test_unknown_in_boolean(self):
        parsed = parse_query("person AND jacket")
        self.assertIsNotNone(parsed.error)

    def test_natural_language(self):
        cars = parse_query("cars and motorcycles")
        self.assertIsNone(cars.error)
        self.assertEqual(cars.mode_hint, "structured")
        self.assertEqual(cars.ast["operator"], "AND")
        self.assertEqual(
            {c["value"] for c in cars.ast["conditions"]},
            {"car", "motorcycle"},
        )

        bikes = parse_query("car or bike")
        self.assertEqual(bikes.ast["operator"], "OR")
        self.assertIn("bicycle", bikes.extracted_classes)

        without = parse_query("people without bicycles")
        self.assertIsNone(without.error)
        self.assertEqual(without.ast["operator"], "AND")
        self.assertEqual(without.ast["conditions"][0]["value"], "person")
        self.assertEqual(without.ast["conditions"][1]["operator"], "NOT")

        grouped = parse_query("(car OR motorcycle) AND person")
        self.assertEqual(grouped.ast["operator"], "AND")

    def test_wearing_query_is_hybrid(self):
        parsed = parse_query("person wearing black")
        self.assertEqual(parsed.mode_hint, "hybrid")
        self.assertEqual(parsed.extracted_classes, ["person"])
        self.assertNotIn("wearing", parsed.extracted_classes)
        self.assertEqual(parsed.ast["operator"], "TERM")
        self.assertEqual(parsed.ast["value"], "person")

        jacket = parse_query("person wearing a black jacket")
        self.assertEqual(jacket.mode_hint, "hybrid")
        self.assertEqual(jacket.extracted_classes, ["person"])
        self.assertTrue({"wearing", "black", "jacket"}.issubset(set(jacket.unknown_terms)))

        red_car = parse_query("red car")
        self.assertEqual(red_car.mode_hint, "hybrid")
        self.assertEqual(red_car.extracted_classes, ["car"])

    def test_builder(self):
        q = builder_to_query(["person", "motorcycle"], "AND", ["car"])
        self.assertEqual(q, "person AND motorcycle AND NOT car")

    def test_eval(self):
        ast = parse_query("person AND motorcycle").ast
        self.assertTrue(eval_ast(ast, {"person", "motorcycle"}, {"person": 1, "motorcycle": 1}, {}))
        self.assertFalse(eval_ast(ast, {"person"}, {"person": 1}, {}))
        not_ast = parse_query("person NOT car").ast
        self.assertTrue(eval_ast(not_ast, {"person"}, {"person": 1}, {}))
        self.assertFalse(eval_ast(not_ast, {"person", "car"}, {"person": 1, "car": 1}, {}))

    def test_structured_search_sample(self):
        metadata = [
            {"image_path": "a.jpg", "detections": [
                {"class": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1], "count": 1},
                {"class": "motorcycle", "confidence": 0.8, "bbox": [0, 0, 1, 1], "count": 1},
            ], "class_counts": {"person": 1, "motorcycle": 1}},
            {"image_path": "b.jpg", "detections": [
                {"class": "person", "confidence": 0.9, "bbox": [0, 0, 1, 1], "count": 1},
                {"class": "car", "confidence": 0.8, "bbox": [0, 0, 1, 1], "count": 1},
            ], "class_counts": {"person": 1, "car": 1}},
        ]
        ast = parse_query("person AND motorcycle").ast
        hits = structured_search(metadata, ast, SearchFilters())
        self.assertEqual(len(hits), 1)
        ast2 = parse_query("person OR motorcycle").ast
        self.assertEqual(len(structured_search(metadata, ast2)), 2)
        ast3 = parse_query("person NOT car").ast
        hits3 = structured_search(metadata, ast3)
        self.assertEqual(len(hits3), 1)

    def test_search_on_indexed_metadata(self):
        from src.utils import load_metadata
        from src.search_engine import run_search

        path = "data/processed/coco-val-2017-500/metadata.json"
        metadata = load_metadata(path)
        known = ["person", "car", "motorcycle", "bicycle"]
        hits_and, parsed, mode, _ = run_search(
            "person AND motorcycle", metadata, known, "structured"
        )
        self.assertGreater(len(hits_and), 0)
        self.assertEqual(mode, "structured")
        self.assertIsNone(parsed.error)
        hits_or, _, _, warn_or = run_search("car OR motorcycle", metadata, known, "structured")
        self.assertGreaterEqual(len(hits_or), len(hits_and))
        self.assertTrue(len(hits_or) <= 48)
        hits_not, _, _, _ = run_search("person AND NOT bicycle", metadata, known, "structured")
        self.assertGreater(len(hits_not), 0)
        nl, parsed_nl, _, _ = run_search("cars and motorcycles", metadata, known, "auto")
        self.assertIsNone(parsed_nl.error)
        self.assertGreater(len(nl), 0)

    def test_portable_image_resolution(self):
        import os
        import tempfile
        from pathlib import Path

        from src.utils import (
            PROJECT_ROOT,
            filename_candidates,
            infer_source_dir,
            portable_relpath,
            resolve_image_path,
        )

        old = (
            r"C:\Monal\Work\AllLight\Krish-sir\KN-CV-Udemy\Phase-2\Projects"
            r"\Yolo_image_search\data\raw\coco-val-2017-500\000000000139.jpg"
        )
        self.assertEqual(
            portable_relpath(old),
            "data/raw/coco-val-2017-500/000000000139.jpg",
        )
        self.assertIn("000000000139.jpg", filename_candidates("0000000139.jpg"))
        inferred = infer_source_dir(
            metadata=[{"image_path": old}],
            metadata_path="data/processed/coco-val-2017-500/metadata.json",
        )
        self.assertEqual(
            inferred.resolve(),
            (PROJECT_ROOT / "data" / "raw" / "coco-val-2017-500").resolve(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "000000000139.jpg"
            dest.write_bytes(b"fake")
            found = resolve_image_path("0000000139.jpg", extra_roots=[Path(tmp)])
            self.assertEqual(found, dest)
            short_old = (
                r"C:\Monal\Work\AllLight\Krish-sir\KN-CV-Udemy\Phase-2\Projects"
                r"\Yolo_image_search\data\raw\coco-val-2017-500\0000000139.jpg"
            )
            found_old = resolve_image_path(short_old, extra_roots=[Path(tmp)])
            project_file = (
                PROJECT_ROOT / "data" / "raw" / "coco-val-2017-500" / "000000000139.jpg"
            )
            if project_file.exists():
                self.assertEqual(resolve_image_path(short_old), project_file)
            else:
                self.assertEqual(found_old, dest)
            env_dir = Path(tmp) / "envroot"
            env_dir.mkdir()
            env_file = env_dir / "zzz_env_only_0001.jpg"
            env_file.write_bytes(b"fake2")
            previous = os.environ.get("YOLO_IMAGE_ROOT")
            os.environ["YOLO_IMAGE_ROOT"] = str(env_dir)
            try:
                found_env = resolve_image_path("zzz_env_only_0001.jpg")
                self.assertEqual(found_env, env_file)
            finally:
                if previous is None:
                    os.environ.pop("YOLO_IMAGE_ROOT", None)
                else:
                    os.environ["YOLO_IMAGE_ROOT"] = previous


if __name__ == "__main__":
    unittest.main()
