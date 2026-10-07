# Fine-cut boundary annotation

This is a generic prompt rewritten for the public code release. It preserves the observed schema and general vocabulary, but is not the historical prompt byte for byte. Corpus-specific vocabulary extensions, film examples, statistics, and answers are omitted.

Read the locked `stages/s3_units.json`, `stages/cut_list.json`, `stages/audio_measured.json`, and `stages/picture_measured.json`. Inspect `reference/golden.mp4` around each boundary. Do not change unit identities or timings. Audio labels are measured inputs: use their speech `type` (`hard`, `J`, or `L`) and music events (`hit`, `change`, `swell`, or `bridge`) rather than inventing an auditory observation. Only use `no_change` when the supplied picture measurement supports continuity. Do not delegate to other agents.

Write `stages/s4_cuts.json` with top-level `stage: "S4"` and `cuts`, an array with exactly one object per input cut in the same order.

| Field in each cut | Contract |
| --- | --- |
| `idx` | Copy the numeric input cut index. |
| `after`, `before` | Copy the adjacent input unit IDs without alteration. |
| `transition` | Object containing a nonempty `type`; include the observed parameters when relevant. |
| `technique` | Nonempty editing-technique label supported by the pictures. |
| `note` | Concrete visual evidence and the relevant supplied audio evidence for this particular cut. |

### 一、 Transition vocabulary

The headings below also support the inherited HTML report parser. These labels are not a closed ontology; use an explained new label only when the listed ones do not describe the evidence.

| Type | Meaning and observed parameter fields |
| --- | --- |
| `cut` | Direct cut. |
| `dip_to_black` | Fade through black; numeric seconds in `fade_out`, `black_hold`, `fade_in`. |
| `dip_to_white` | Fade through white; numeric seconds in `fade_out`, `black_hold`, `fade_in` in the inherited schema. |
| `dissolve` | Overlap transition; numeric seconds in `dur`. |
| `flash` | Brief flash transition. |
| `wipe` | Wipe; describe `direction`. |
| `blur_whip` | Blur or whip transition. |
| `no_change` | Continuous picture, only with measured continuity support. |

### 二、 Technique vocabulary

| Technique | Meaning |
| --- | --- |
| `plain` | Direct connection without a more specific observed technique. |
| `on_action` | Cut coordinated with an action. |
| `match_cut` | Correspondence between the adjacent shots. |
| `jump_cut` | Discontinuous jump within related action or framing. |
| `smash_cut` | Abrupt contrast between adjacent shots. |
| `intercut` | Alternation between related actions or settings. |
| `reaction` | Cut to a visible reaction. |
| `reveal` | Cut that reveals relevant visible information. |
| `graphic_match` | Correspondence in visual composition. |

## Review

Every note must refer to the particular cut, not repeat a generic sentence. The inherited queue flags sequences with at least 20 cuts and fewer than three technique labels. That diversity threshold is a quality-control heuristic, not a reason to fabricate varied labels: report for human review when the evidence does not support the heuristic. Automated checks validate adjacency and required fields, not interpretive accuracy. Review the final file before ending.
