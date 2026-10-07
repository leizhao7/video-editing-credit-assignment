# Shot, beat, and chunk annotation

This is a generic prompt rewritten for the public code release. It preserves the observed input/output contract, but is not the historical prompt byte for byte. No historical film, shot annotation, dialogue, or task answer is included.

Work in the task directory supplied by the queue. Read `stages/script_inputs.json` and inspect `reference/golden.mp4`; use `cards/` for the listed card stills and `work/` for temporary inspection frames. The inputs supply ordered shot IDs, `start_s`, `dur_s`, `kind`, timed speech, per-shot `line`, and measured music events. Preserve measured values; annotate visible content from the reference. Do not invent dialogue or infer visual content solely from a title. Do not delegate to other agents.

Write one JSON object to `stages/script.json`. The following table specifies fields, not a complete sample annotation.

| Location | Required content |
| --- | --- |
| `film` | Copy the input film slug. |
| `logline` | A concise summary grounded in the visible sequence. |
| `shots` | One object per input shot, in exactly the original order. |
| `shots[].id` | Copy its input ID. |
| `shots[].desc` | Nonempty description of what is visible. |
| `shots[].line` | The supplied spoken text, or null when absent. |
| `shots[].card_text` | Visible text on a card, or null when absent. |
| `shots[].card_misflag` | Optional nonempty explanation when visual evidence contradicts a card classification. Do not use it merely to avoid transcribing a card. |
| `beats` | Ordered narrative beats; each has `id`, `name`, `shots`, `what`, `function`, and `why_here`. |
| `beats[].shots` | Two input shot IDs delimiting a closed interval: first and last. |
| `beats[].what` | What occurs in this beat. |
| `beats[].function` | Its narrative role. |
| `beats[].why_here` | Visible evidence supporting the boundary. |
| `chunks` | Ordered editing assignments; each has `id`, `beat`, `shots`, `what`, and `why_here`. |
| `chunks[].beat` | ID of the containing beat. |
| `chunks[].shots` | Two input shot IDs delimiting a closed interval. |
| `chunks[].what` | A concrete local editing brief supported by the sequence. |
| `chunks[].why_here` | Evidence supporting the chunk boundary. |

Beats must cover every shot once, with no gaps or overlaps. Chunks must do the same and remain entirely within their named parent beat. Prefer coherent boundaries to an arbitrary fixed length. The inherited validator requires at least two beats and nonempty chunks; it rejects a chunk longer than 15 shots when its containing beat is also longer than 15 shots. These are construction heuristics, not evidence that every sequence naturally has that structure. If the supplied material cannot support a valid annotation, report that limitation for human review rather than inventing content.

Inspect the finished JSON against the inputs before ending. The queue checks coverage and required fields; those checks cannot establish the accuracy of descriptions or narrative interpretation.
