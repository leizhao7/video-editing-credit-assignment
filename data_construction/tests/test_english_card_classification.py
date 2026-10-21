"""English card evidence and compatibility with historical Chinese shot logs."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_s1_tasks as builder
import s1_card_classification as classification


class EnglishCardClassificationTests(unittest.TestCase):
    def marked_kind(self, description):
        inputs = {"shots": [{"id": "g0001", "kind": "source"}]}
        script = {"shots": [{"id": "g0001", "desc": description}]}
        with mock.patch.object(builder, "_LUMA", {"sample/g0001": [80, 25]}):
            builder.mark_cards(script, inputs, "sample")
        return inputs["shots"][0]["kind"]

    def test_english_and_historical_chinese_cards_are_recognized(self):
        for description in (
                "A full-screen title card with white lettering.",
                "Release-date card on a black background.",
                "A STUDIO LOGO fills the screen.",
                "A title card, not a studio logo, fills the screen.",
                "White text on black announces the film title.",
                "片名卡，黑底白字。", "全屏字幕卡。"):
            with self.subTest(description=description):
                self.assertEqual(self.marked_kind(description), "card")

    def test_temporal_references_overlays_and_unrelated_words_are_not_cards(self):
        for description in (
                "After a title card, two people walk through the street.",
                "A live-action shot before the release-date card.",
                "A studio logo is superimposed on a live-action shot.",
                "A title card overlay appears over the actor.",
                "This is not a title card; it is live-action footage.",
                "This isn't a standalone title card; the actor is holding a sign.",
                "There is no release-date card in this shot.",
                "A man carries a postcard through the room.",
                "片名卡之后，两个人走在街上。",
                "厂标叠在实拍画面上。"):
            with self.subTest(description=description):
                self.assertEqual(self.marked_kind(description), "source")

    def test_english_correction_notes_match_historical_evidence_classes(self):
        pairs = (
            ("This is actually live-action footage.", "实际是实拍", "footage"),
            ("This is actually a full-screen title card.", "实际是全屏字幕卡", "card"),
            ("This is actually a solid-black shot.", "实际是纯黑画面", "featureless"),
            ("A title card alternates with live-action footage.", "字幕卡交替实拍", "mixed_card_footage"),
            ("Live-action footage is intercut with title cards.", "字幕卡混合实拍交替", "mixed_card_footage"),
        )
        for english, historical, expected in pairs:
            for note in (english, historical):
                with self.subTest(note=note):
                    self.assertEqual(classification.note_classification(note), expected)

    def test_opening_card_descriptions_require_review_and_negations_do_not_assert_cards(self):
        for description in ("A full-screen text card with white lettering.", "字幕卡：黑底白字。"):
            with self.subTest(description=description):
                self.assertEqual(classification.note_classification("", description),
                                 "card_description_requires_review")
        for note in ("This is not a title card.", "The shot follows a title card."):
            with self.subTest(note=note):
                self.assertEqual(classification.note_classification(note), "unresolved")
        self.assertEqual(classification.note_classification("", "After a title card, a man enters."),
                         "no_correction_note")

    def test_english_overlay_evidence_preserves_footage_for_review(self):
        raw = {"shots": [{"id": "g0001", "kind": "source"}]}
        script = {"shots": [{"id": "g0001", "card_misflag": "This is actually a title card.",
                             "desc": "White lettering is superimposed on live-action footage."}]}
        fixed, _, audit = classification.correct_s1_classification(
            script, raw, deepcopy(raw), {}, "sample", luma={"sample/g0001": [80, 25]})
        self.assertEqual(fixed["shots"][0]["kind"], "source")
        self.assertTrue(audit[0]["review_required"])
        self.assertEqual(audit[0]["reason"], "card_overlay_or_description_requires_review")


if __name__ == "__main__":
    unittest.main()
