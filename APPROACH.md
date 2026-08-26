# Frame & Dialogue Extractor

**Problem:** Given a media URL and a target dialogue string, find the exact frame where that dialogue *first* appears, extract the text, and save the frame.

**Reference input:** `https://ok.ru/video/248244667877`, Granada TV, *The Adventures of Sherlock Holmes: A Scandal in Bohemia* (1984), 54:22.
**Reference target:** `"My mind rebels at stagnation"`

This document answers, in order, the four questions the brief asks this design to answer.

> **A note on how this design was reached.** The architecture below isn't the first one I built. It's the second, arrived at after empirical evidence contradicted the first. §1 explains why, with the evidence trail left in rather than smoothed over. The full decision-by-decision record is in `prompts.txt`.

---

## 0. Interpretation of the Problem Statement

The statement is ambiguous about whether the dialogue text is an **input** to the program or something the program **discovers**. I resolved it as an input: the program is a *search* tool, not a bulk text extractor. Evidence, from the statement itself:

1. **"We may also choose a different video / dialogue text during our evaluation."** This is the decisive one. If the program merely reported whatever text was on screen, there'd be no such thing as choosing a different *dialogue text*, since you could only choose a different video. Varying the text independently of the video means the text is a parameter.
2. **"3. The actual dialogue: 'My mind rebels at stagnation'"**: the target is handed to me in the task description directly.
3. **"How your solution determines where to look in the video"** doesn't make sense if I extracted everything. "Where to look" presupposes searching for something specific.
4. The example output prints `Text : "My mind rebels at stagnation"`.

The `Input` section names only the URL, which is what creates the confusion. I read that as an omission in how it's written, contradicted by all four points above.

**Consequence:** the CLI takes `--url` and `--text`. This also happens to be the only design that survives a different dialogue being substituted.

---

## 1. How the solution determines *where to look*

The reference video runs 54:22, roughly **81,500 frames at 25 fps**. Running OCR on every single one isn't a usable engineering artifact. Measured EasyOCR latency on this hardware is ~1.1s/call (see the correction below, an earlier draft of this document estimated ~75ms/call before I actually measured it), which puts naive full-frame OCR at roughly **a full day per run**, not the couple of hours I originally assumed. Reducing that search space is part of the problem. But which channel to search turned out to be the bigger question, and I got the first answer wrong before evidence corrected it.

### Why the architecture is dual-channel, not visual-only

I initially built this as a 100%-visual pipeline, with no audio processing at all. The reasoning at the time: the problem asks for *on-screen* dialogue and visual text extraction, and an audio-first pipeline is silently wrong on a silent video, a dubbed track, subtitles that disagree with the spoken track, or on-screen text that's never spoken (titles, captions, signage). Since the video might get swapped, correctness depending on audio agreeing with the picture looked like a liability.

That decision was then contradicted by evidence on the reference video itself. I watched it and confirmed, at 05:26/54:22: Jeremy Brett speaking the target line, with no on-screen text in frame at all. Corroborated independently by an IMDB quotes page for this exact episode and an academic source describing the line as imported from Conan Doyle's *The Sign of the Four* into the teleplay. I then sampled 7 of 23 frames across the full runtime (title cards, multiple dialogue scenes, end credits) via archive.org's auto-generated thumbnails: zero on-screen text in any dialogue scene, anywhere. Text appears only in title and credit cards. The target line is spoken dialogue, full stop. There's nothing for a visual-only pipeline to extract here, no matter how well engineered.

I considered just flipping to audio-primary instead. Rejected that too. An ASR-primary design with OCR only as a fallback-on-miss has a silent-failure mode. If the audio transcript produces a confident but wrong match, the OCR fallback never fires, and the tool reports a plausible wrong frame without ever checking what's actually on screen. My original concern about a swapped-in silent or visually-driven video is still valid; I now just have direct proof of the mirror-image failure too. Neither single-channel design survives both cases.

Final design: both channels run independently, and disagreement between them is a reported signal, not something to resolve silently.

```
                ┌─ VISUAL channel: coarse-to-fine OCR sweep ─┐
video ──────────┤                                             ├──► fuse ──► report
                └─ AUDIO channel: ASR, word-level timestamps ─┘
```

Both channels normalise and score candidates through the same `matcher.py`, but their scores are never compared directly against each other. See §4 for why, and for the two fusion designs I tried and rejected before landing on this one.

`--mode auto|visual|audio` (default `auto`) lets either channel be forced independently, directly serving the statement's *"we may ask you to modify your implementation."*

### The visual channel: four compounding filters

| # | Filter | Mechanism | Effect |
|---|---|---|---|
| 1 | **Temporal sampling** | Decode at 1 fps instead of 25 fps | ~81,500 → ~3,260 candidate frames |
| 2 | **Spatial ROI crop** | Crop to the lower ~40% band where subtitles render | Each OCR call is ~3× cheaper |
| 3 | **Frame dedup** | Skip a frame whose ROI is near-identical to the last one OCR'd (mean-absolute-difference) | On real footage, measured ~30% rejection, the dominant real-world contributor |
| 4 | **Text-presence gate** | Edge-density heuristic, ~1 ms | Cheap and never rejects legible text, but see the limitation below |

The run summary reports each filter's rejection count (`diagnostics` in `result.json`), so the reduction is measured, not asserted.

> **Measured, not assumed: the text-presence gate barely filters real footage.** An earlier draft of this document claimed filters 3 and 4 "matter more than the crop" and implied comparable contributions from both. Profiling against a real 90-second window of the reference-adjacent footage showed the text-presence gate rejecting essentially nothing (0–1 of 90 frames), because real footage (film grain, fabric, furniture) is edge-rich enough that its Canny edge-density distribution (measured: min 0.0043–0.0061, median ~0.02) directly overlaps the density of genuinely legible text (measured: 0.0136). No threshold separates two overlapping distributions on one feature. That's not a mis-calibration, it's a limitation of using a generic edge-density heuristic as a text detector. Frame dedup does the real work on real content (~30% rejection measured, since consecutive frames of a static shot are legitimately near-identical). The text-presence gate is correctly conservative (cheap, zero false negatives) rather than a major contributor, which is a narrower and more honest claim than the original draft made.

**Sampling rate.** Subtitle cards typically hold for 1.5–5 seconds, so 1 fps (default) rarely misses one, and it halves the visual channel's OCR cost relative to 2 fps. This was a deliberate trade-off, accepted specifically to fund the audio channel's build and runtime cost within budget (dual-channel is, by my own risk analysis, "essentially building two projects instead of one"). The accepted risk is a subtitle card shorter than ~1s; `--sample-fps` overrides for cases where that matters more than it does on the reference video.

### The ROI is a fast path, not an assumption

The statement requires robustness to *"variations in ... the appearance of the dialogue."* A hardcoded lower-third crop would return a confident "not found" on a video with centered or top-positioned text: a silent wrong answer, the worst failure mode available.

So the ROI is tiered:

1. Sweep the ROI band first (fast path).
2. If the ROI sweep yields no match, automatically re-sweep the full frame before reporting failure.
3. `--roi` and `--full-frame` allow manual override.

The extra pass only gets paid for in the case where I was otherwise about to fail. That turns the optimisation from a correctness risk into a defensible tiered strategy.

---

## 2. How it determines the relevant frame

The coarse sweep finds *a* frame containing the dialogue. The statement asks for the frame where it **first appears**, a stricter requirement.

**Coarse-to-fine refinement:**

1. Coarse sweep returns the best-scoring sample at time *T* (accurate only to the sampling interval, ±0.5 s).
2. Re-decode the window `[T − 2s, T + 2s]` at the video's native frame rate.
3. OCR every frame in that window.
4. Walk backward from *T* while the match score stays above threshold. The earliest frame still matching is the subtitle's onset, the reported answer.
5. Walk forward similarly to report the subtitle's on-screen duration.

The backward walk is capped (default 10 s) so a pathological match can't run away.

**Frame number** is derived from the decoder's reported position rather than computed from wall-clock time, so it stays correct on videos whose container timebase disagrees with a naive `t × fps`. For variable-frame-rate sources, frame index isn't well defined; I report the timestamp as authoritative and flag the result. The statement's own hedge, *"the frame number, where applicable,"* suggests it anticipates this.

---

## 3. How it extracts the text

### Visual channel: EasyOCR

**Engine:** EasyOCR (PyTorch backend). Chosen over Tesseract, which needs a system-level binary install and is notably weak on anti-aliased video subtitle text without heavy preprocessing. EasyOCR is pip-installable, more accurate on rendered subtitles, and its multilingual model coverage is real insurance for a future swapped-in video with non-English on-screen text. Verified working with the installed numpy 2.5.2 (I had an initial concern about a version mismatch with EasyOCR 1.7.2 that turned out unfounded, checked directly rather than assumed).

**Line assembly.** EasyOCR returns a list of detected text boxes, not a sentence. Subtitles routinely wrap across two lines, and a target string can straddle that break. I sort detected boxes by `(y, x)` and join them into a single string before matching, so a wrapped subtitle gets matched as one unit rather than two fragments that each fail on their own.

**Preprocessing** is kept deliberately minimal (grayscale, upscale small ROIs). Aggressive binarisation helps clean white-on-black subtitles and actively hurts subtitles over bright or busy backgrounds. Since the input video can't be predicted in advance, I'd rather lean on the engine's own robustness than hand-tune thresholds fitted to one sample.

### Audio channel: faster-whisper

**Engine:** `faster-whisper` (CTranslate2 backend), `small.en` by default. Chosen over `openai-whisper` for CPU throughput. CTranslate2 is meaningfully faster on CPU-only inference, which matters since this channel processes the full runtime (audio-only decode; there's no equivalent to the visual channel's frame-sampling reduction needed here, since transcribing continuous audio is already a single linear pass rather than a per-frame search).

**Word-level timestamps are the point of this channel, not just line detection.** `faster-whisper` exposes per-word start/end times directly. I run a sliding fuzzy match of the target string over the transcript; once the containing segment is found, the first word's start timestamp is the audio channel's answer for "where the dialogue first appears," the speech-onset analogue of the visual channel's text-onset backward walk (§2). Reported and converted to a frame number via the video's fps, same as the visual channel's output, so both channels produce directly comparable `Result` objects downstream.

**Same normalisation as the visual channel.** The transcript passes through the identical `matcher.py` normalisation (NFKC, casefold, punctuation strip, whitespace collapse) before fuzzy scoring. Deliberately shared code, so both channels get scored by the same rules even though (§4) their raw scores are never compared to each other directly.

### Audio confirmation isn't the same thing as on-screen confirmation

An ASR match only proves the target line exists somewhere in the audio track. It doesn't prove the
speaker is actually visible when they say it: a voiceover, an internal monologue, or a line delivered
off-camera would all match just as confidently. On-screen dialogue, by the working definition this
project uses, is a line spoken by a character who's both present and visible in the current shot, so
audio evidence alone isn't sufficient to claim that.

`core/face.py` checks for a face near the audio onset frame (`cv2.FaceDetectorYN`, backed by a
bundled ~227KB ONNX model from the OpenCV Zoo, `core/models/face_detection_yunet_2023mar.onnx`).
Not mediapipe: checked first, and its latest release has no wheel for Python 3.13 on any platform.
Not the classic Haar-cascade detector either, that's been the default answer to "cheap local face
detection in OpenCV" for two decades, but OpenCV 5.0 (already a dependency here) dropped both
`cv2.CascadeClassifier` and the bundled cascade XML files from the Python wheel entirely. YuNet is
the modern replacement and is more accurate anyway, so this isn't a downgrade forced by necessity.

A single frame is not a reliable read for this. A face turned three-quarters away, mid-blink, or hit
by a motion-blurred pan can make the detector miss a face that's genuinely there, the same "don't
trust one frame" lesson already applied to the visual channel's OCR onset walk (§2). So the check
looks at a small window of frames around the onset (`--face-check-window`, default ±0.5s), not just
the exact computed timestamp, and counts a face as present if *any* frame in that window shows one.

**This never discards a correct audio match, it only relabels it.** Three outcomes:

| Case | What happens |
|---|---|
| A face is confirmed near the onset (possibly by preferring a different, face-confirmed occurrence of the same line over a higher-confidence but faceless one) | Reported exactly as before: `CONFIRMED_BY_AUDIO` (or `CORROBORATED` if visual agrees) |
| No face found at any confident audio candidate's onset | Still returns the best-confidence candidate, full timestamp/frame/text/confidence, nothing thrown away, but under a new outcome, `CONFIRMED_BY_AUDIO_OFF_SCREEN`, with an explicit note that no on-screen presence was confirmed |
| The detector itself is unavailable (missing model file, load failure) | Total bypass, `face_detected` stays `None` everywhere, behaves identically to a build without this feature at all |

The failure-open design here matters as much as the detection itself. A single-frame face read from a
2-decade-old-style detector's modern successor is still not something to trust enough to silently drop
an otherwise-correct, already-independently-verified audio match. That would trade a labeling
inaccuracy for a much worse failure: reporting no result at all when the pipeline actually found the
right answer. `--no-face-check` turns the whole thing off if needed.

**The face-detection result is stated explicitly, not just implied by the outcome name.** The whole
point of this feature is answering "is this genuinely on-screen dialogue or not." Burying that answer
inside an enum value (`CONFIRMED_BY_AUDIO` vs `CONFIRMED_BY_AUDIO_OFF_SCREEN`) and leaving the reader to
infer it defeats the purpose. So every surface says it plainly: the console prints a dedicated
`On-screen : Yes / No — not confirmed on screen / Not checked` line whenever the primary result is
audio, `result.json` carries `face_detected` both at the top level and per candidate, and
`report.html`'s badge itself spells it out (`Confirmed · Audio (on-screen)` vs `(off-screen)`), with a
matching stat row and, on `CORROBORATED`/`AMBIGUOUS`, an `On-screen face` column in the per-channel
evidence table. Same three-value story (`True`/`False`/`None`) everywhere it appears, via one shared
`face_status_label()` helper in `core/report.py`, not three independently-worded copies that could
drift out of sync with each other.

### An addition on top: mouth motion, still report-only, still not gating anything

Face presence alone can't distinguish a character actively delivering a line from one silently
listening while a voiceover plays over them. Both show a static "face detected" reading, since
that check has no temporal dimension at all. `FacePresenceDetector.mouth_motion_elevated()` adds one:
it tracks the two mouth-corner landmarks `cv2.FaceDetectorYN.detect()` already returns on every call
(15 columns per face: bbox, five landmarks including both mouth corners, and a score; nothing new
to compute, these were already being thrown away), normalizes their distance by inter-ocular
distance (standard scale-invariance trick so the metric means the same thing regardless of how close
the face is to camera), and measures how much that normalized distance moves frame-to-frame across
the onset window. That's compared against the same measurement over a same-length window shortly
before the match (`--face-check-window` before, offset back by
`mouth_motion_baseline_offset_s`, default 1.5s), a *relative* comparison rather than one fixed
global threshold, so it adapts to each video's own idle-motion level instead of guessing a universal
cutoff.

**Verified against the real reference video, not just synthetic tests, and the honest result is a
useful negative, not a clean confirmation.** At the actual onset (00:05:24.640), the raw numbers:
speech-window energy 0.00130, baseline-window energy 0.00163. The baseline is not lower, it's
*higher*. The reason is specific and instructive: Jeremy Brett delivers this line as part of a
continuous monologue, so the window "shortly before" the matched line is not actually quiet, it's
him still mid-speech from the preceding sentence. The baseline-offset design assumes a nearby quiet
reference exists; that assumption fails on continuous dialogue, which is common, not a rare edge
case. This is exactly why the decision to keep this report-only rather than gating was correct.
Had this signal been wired to demote the outcome, it would have taken an already-correct,
already-independently-verified result and cast doubt on it based on a baseline comparison that
didn't hold. No threshold or offset was retuned to make this one case read differently after seeing
the result, that would be fitting a parameter to a single data point, the same trap already flagged
above (§1) about not hand-tuning thresholds to one sample. The honest scope of this feature: real
evidence, reported plainly, that a static face-presence check alone doesn't have, not a solved
active-speaker detector, and it doesn't claim to be one.

---

## 4. How it handles ambiguous or uncertain results

### The query / observation distinction

This is the conceptual core of the matching design.

> **The target text is a *query*. The extracted text is an *empirical observation*.**

They're different kinds of thing and must never get conflated. The target is what I'm looking for; the OCR output is what the camera and the model actually produced at that frame. The two will routinely disagree in harmless ways:

- punctuation the query omits, e.g. `"My mind rebels at stagnation!"`
- line-wrap artifacts, e.g. `"My mind rebels\nat stagnation"`
- character-level OCR noise: `"rebeis"`, `"stagnatlon"`, `l`/`I`/`1` and `O`/`0` confusions
- surrounding dialogue captured in the same subtitle card

Two consequences follow directly:

1. **Matching has to be fuzzy.** Exact string comparison would reject a correct frame just because OCR read one character imperfectly. I normalise (Unicode NFKC, casefold, strip punctuation, collapse whitespace) and score with `rapidfuzz.partial_ratio`, with `token_set_ratio` as a secondary signal.
2. **The reported `Text:` is the OCR output, not the input echoed back.** Printing the query back would be answering a question nobody asked, and it would hide exactly the uncertainty this tool exists to report. The observation is the evidence that the search actually succeeded.

### Confidence bands (per channel)

Each channel is thresholded independently. A visual score of 85 and an audio score of 85 don't mean the same thing, because they measure different error distributions (glyph-level noise on rendered text vs. acoustic/language-model error on speech). Treating them as the same scale would be comparing unlike units.

| Score | Verdict | Behaviour |
|---|---|---|
| ≥ 85 | Confident | Channel produces a candidate |
| 70–84 | Uncertain | Candidate produced, flagged `confidence: "low"` |
| < 70 | No match | Channel abstains |

### Cross-channel fusion: two rejected designs before this one

**Rejected design 1: max-score wins.** *"Whichever channel returns the highest fuzzy-match score above 85% wins."* Rejected for the same reason the per-channel thresholds above are kept separate: an 82 from EasyOCR and an 80 from Whisper aren't on a shared scale, so `82 > 80` across channels isn't a meaningful comparison. Worse, max-score throws away the disagreement itself. If audio says 00:15 @ 80 and visual says 00:45 @ 82, this design reports 00:45 with no caveat, when two channels landing 30 seconds apart is the single most informative signal the system can produce: one of them is confidently wrong, and collapsing that to a single number hides it.

**Rejected design 2: visual channel always wins ties.** As a stopgap, I proposed defaulting to the visual channel whenever both fire, reasoning the brief says "on-screen dialogue." Self-contradictory, and caught before it got adopted: the entire reason this pipeline has an audio channel at all is that I verified, on the reference video, that the target line isn't on screen (§1). Deriving a tie-break from the same phrase that evidence had just undercut is circular. A fixed modality preference is exactly the kind of baked-in assumption that fails the moment the video changes and the true answer turns out to be visual. `resolve_strict()` in `core/fuse.py` implements no such preference.

**Final design: labelled outcome states, not a collapsed score.** Fusion compares timestamps between channels before it ever looks at scores:

| # | Case | Outcome state | Behaviour |
|---|---|---|---|
| 1 | Audio confident, visual not, a face confirmed near the onset | `CONFIRMED_BY_AUDIO` | Return the audio candidate |
| 1b | Audio confident, visual not, no face confirmed at any candidate onset | `CONFIRMED_BY_AUDIO_OFF_SCREEN` | Return the same audio candidate, flagged as audio-only evidence (§3) |
| 2 | Visual confident, audio not | `CONFIRMED_BY_VISUAL` | Return the visual candidate |
| 3 | Both confident, timestamps agree within tolerance | `CORROBORATED` | Highest-confidence result; independent corroboration reported |
| 4 | Both confident, timestamps substantially differ | `AMBIGUOUS` | Both candidates returned, never silently resolved |
| 5 | Neither confident | — | Top-N near-misses from both channels, with scores, so a failure is diagnosable instead of silent |

`--strict` (only when a single answer is explicitly required): selection follows modality validation and temporal evidence, never a fixed channel preference. If modality can't be established, the candidate with the strongest independent validation is returned with `uncertain=true` retained. The uncertainty never gets dropped just because a single value was requested.

`Result` exposes evidence, not a verdict: `channel`, `timestamp`, `confidence`, `matching_method`, `temporal_consistency`, `ambiguity_status` per candidate. ASR and OCR are evidence sources being cross-examined here, not two competitors scored against each other on one scale.

### Other ambiguity cases

| Case | Response |
|---|---|
| Target appears at multiple distinct times within one channel | Rank all clusters, return the earliest confident one, set `ambiguous: true`, list every candidate with scores in `result.json` |
| Score lands in the uncertain band | Return it, flagged, with the raw text attached |
| No visual match in ROI | Automatic full-frame re-sweep before that channel reports failure |
| Target spans two subtitle cards (visual) | Sliding-window join across consecutive sampled frames |
| Fade-in subtitles (visual) | Onset is a score ramp, not a step; onset is the first frame crossing threshold, and the ramp is recorded |
| VFR video | Timestamp authoritative, frame number flagged |
| Cross-channel score comparison | Methodologically invalid if attempted naively, see the two rejected fusion designs above. Designed around from the outset, not patched in afterward |

The guiding principle: the program should never claim a precision it doesn't have, and should never fail without explaining what it saw instead.

---

## 5. Architecture

### Decoupled core / interface

```
core/          pure Python. No I/O framework, no CLI, no HTTP. Takes (video_url,
               target_text) and returns a structured result object.
cli.py         thin presentation layer over core/ (Typer + Rich).
app.py         [future] Streamlit UI, snaps onto the same core, zero core changes.
```

`core/` has no knowledge of how it's being called. That's what makes a UI a later additive change rather than a refactor.

### Module layout

```
core/
  acquire.py     Input acquisition, auto-detects yt-dlp URL / direct media URL /
                 local file path (see "Input genericity" below); local cache
                 keyed by URL hash
  video.py       Metadata (fps, duration, dims, VFR flag), sample_at(fps),
                 iter_range(t0, t1)
  roi.py         ROI band computation + full-frame fallback (visual channel)
  gate.py        Frame dedup (MAD) + text-presence heuristic (visual channel)
  ocr.py         OCR engine interface + EasyOCR implementation (visual channel)
  asr.py         faster-whisper wrapper, word-level timestamps (audio channel)
  face.py        On-screen presence check for audio matches (YuNet, audio channel)
  models/        Bundled face_detection_yunet_2023mar.onnx (~227KB, OpenCV Zoo)
  matcher.py     Normalisation (NFKC, casefold, punctuation strip) + rapidfuzz
                 scoring, shared by both channels so they're scored by
                 identical rules even though never compared cross-channel
  fuse.py        Timestamp-agreement logic, per-channel thresholds, labelled
                 outcome states (§4)
  locate.py      Orchestration: run both channels → fuse → refine → onset walk
  report.py      Result dataclass, result.json, frame PNG
  report_html.py Self-contained report.html (embedded CSS, base64 frame)
  config.py      All tunables in one dataclass
cli.py           Typer entrypoint: --url --text --out --mode --roi --full-frame
                 --sample-fps --strict --no-report --no-face-check
                 --face-check-window --no-mouth-motion-check
tests/
```

### Input genericity: any URL, by design, not as later hardening

The brief says *"a YouTube video link **or equivalent**"* and states a different video may be substituted. `acquire.py` is source-agnostic from the start as a consequence:

| Input | Handling |
|---|---|
| Any yt-dlp-supported URL (~1800 sites: YouTube, Vimeo, Dailymotion, VK, ok.ru, …) | yt-dlp's extractor resolves it |
| A direct media URL (`https://…/video.mp4`) | Streamed/downloaded directly |
| A local file path | Used as-is, no network involved |

The input kind is auto-detected. That's why yt-dlp got chosen over a hand-rolled ok.ru-specific fetcher at the original stack decision: genericity is inherited from the library choice, not built as a special case. Worth being precise about what this does and doesn't solve, though. URL genericity is a property of the code, and it's satisfied today. It's a separate concern from *this specific network's* inability to reach `ok.ru` (§7), which no code change fixes.

### Ingestion failure-handling contract

`acquire()`'s contract: given a source string, return a path to a verified-playable local file, or raise a typed exception (`InvalidSourceError`, `NetworkUnreachableError`, `IncompleteDownloadError`, `MissingDependencyError`). Never a raw network/library exception, and never a silently-returned bad file.

This exists because every one of these failure modes happened for real during this project's own development, not as speculative hardening:

- **Transient network failure**: bounded retry with backoff (3 attempts). Permanent failures (an unsupported/non-video URL) are never retried, that would just waste the budget on something no retry can fix.
- **A download that "succeeds" while actually incomplete**: not hypothetical. yt-dlp's `--skip-unavailable-fragments` is on by default, which is precisely the mechanism that let the reference video's own download report success while silently missing 9.5 minutes of fragments. `acquire()` now compares the source's reported duration against the actual demuxed duration (reusing `core/video.py`) before accepting a download as done.
- **The source is genuinely unreachable from this network**: after the retry budget, fail with a message that's generic across any host, never named after whichever specific site motivated this design, and points at the one thing that reliably works: passing a local file path.

Explicitly not built: a custom HLS/DASH segment-level fetcher. yt-dlp already does fragment-level retry/resume correctly when configured (`fragment_retries`, `continuedl`, `hls_use_mpegts`). Duplicating it would be pure risk for the fixable case, and it wouldn't have fixed the actual failure this project hit anyway, a network-blocked manifest page, which no amount of fragment-level cleverness can route around.

None of this is visible to `core/locate.py` or anything downstream. They only ever see a path `acquire()` has already verified.

### Why CLI first, and why not a web backend

I deliberately chose a CLI over a web service for the primary deliverable:

- **A full run is minutes long, not milliseconds.** Download plus sweep plus refinement is a long-lived job. Behind HTTP that means request timeouts, proxy timeouts, and browser timeouts, which forces a job queue, polling or websockets, and progress persistence. That's a meaningful amount of infrastructure whose only purpose is to work around a constraint I can just decline to adopt.
- **Failure modes get worse, not better.** A timed-out HTTP request is indistinguishable to the user from a crashed pipeline. A CLI process streams progress to the terminal and exits with a real status code.
- **Reliability over surface area.** A 100% working core is worth more than a UI that introduces a class of bugs orthogonal to the actual problem.

Streamlit is the intended future UI precisely because it runs the pipeline in-process. It sidesteps the HTTP timeout problem entirely instead of engineering around it.

---

## 6. Technology choices and trade-offs

| Concern | Choice | Rationale | Trade-off accepted |
|---|---|---|---|
| Acquisition | `yt-dlp` (as a library) | ~1800 supported sites; the statement says *"a YouTube video link or equivalent,"* so URL-agnostic fetch is required, not a bonus | ok.ru specifically has region-lock/extractor issues from some networks (§7); mitigated by local-file input |
| Decode | OpenCV | Ubiquitous, simple, straightforward to reason about line by line | Seeking can be imprecise on some codecs; PyAV exposes true PTS but costs complexity |
| OCR (visual) | EasyOCR | Accurate on subtitles, pip-only, multilingual insurance | Pulls ~2.5 GB of PyTorch (shared cost with the audio channel below); slower per call than an ONNX runtime |
| ASR (audio) | `faster-whisper` | CTranslate2 backend, CPU-fast, word-level timestamps natively (needed for speech onset, not just line detection) | Additional model download (`small.en`); one more dependency to maintain |
| Face detection | `cv2.FaceDetectorYN` (bundled ONNX) | Already-installed OpenCV, no new pip dependency; more accurate than Haar cascade | mediapipe has no Python 3.13 wheel yet (checked directly); Haar cascade itself is gone from OpenCV 5.0's Python wheel, so this is genuinely the current option, not a preference; the tradeoff is a small (~227KB) binary model file committed to the repo |
| Matching | `rapidfuzz` | C++ speed, well-maintained; shared by both channels so they're normalised identically | — |
| CLI | Typer + Rich | Typed args, good `--help`, readable progress output | — |

---

## 7. Known risks

1. **Resolved: no on-screen text in the reference video, at all.** Original concern was language (Russian upload, possible Russian burned-in subs). Actual finding, from a directly-verified frame plus independent sourcing plus my own sampling of 7/23 archive.org thumbnails across the full runtime: there are no burned-in subtitles whatsoever, text appears only in title/credit cards. This is the finding that drove the architecture to dual-channel (§1). Worth noting as a general lesson: a risk framed as "which of these known values will it be" can turn out to have an answer outside the enumerated set entirely.

2. **`ok.ru` is unreachable from the primary development network, confirmed rather than hypothetical, and now resolved via an independently-verified reference file.** Diagnosed as an SNI-level TLS block on the exact hostname `ok.ru` (connection reset mid-handshake, reproduced with bare `curl`, independent of yt-dlp; every alias, `odnoklassniki.ru`, `ok.me`, `mobile.ok.ru`, either redirects back into the block or lacks the stream-URL endpoint). The block pattern matches a consumer ISP, so it may affect other users on similar networks. Consequently, local-file input (already in `acquire.py`, §5) is a correctness requirement, not a development convenience, and `README.md` documents the symptom and workaround up front.

   The actual reference video was subsequently obtained on a different network path (mobile hotspot) and, after a partial-download recovery, verified before use, not taken on trust despite a plausible recovery narrative. A full decode-integrity scan found a real, localized defect (missing-reference-picture and timestamp warnings) confined to a ~8-minute window in the recovered tail (44:55–52:57), confirmed to sit entirely outside the target dialogue's location. The earlier `archive.org` copy (a different cut, 52:07 vs. the reference's 54:22) remains useful as a disposable dev/test corpus but isn't load-bearing for final validation anymore.

3. **Windows console encoding.** Extracted subtitle text is Unicode; the default Windows console codepage (cp1252) raises `UnicodeEncodeError` on characters outside its range. Already hit this once while parsing the source PDF. All text output has to be explicitly UTF-8 encoded.

---

## 8. Validation against the real reference video

Run against the verified reference file (`Sherlock_video/sherlock_complete.mp4`, `--mode audio`):

```
Outcome    : CONFIRMED_BY_AUDIO
Timestamp  : 00:05:24.640
Frame      : 7771
Text       : " My mind rebels at stagnation."
Channel    : audio
Confidence : 100.0
On-screen  : Yes — confirmed on screen
```

That's 1.36 seconds from the reference-video timestamp (05:26) I observed directly while watching the video. The saved frame was visually confirmed to show the correct scene, Jeremy Brett in close-up, matching the framing of my own screenshot. No timestamp was hardcoded anywhere in the search path to produce this; the pipeline located the line independently via ASR, word-level alignment, and fuzzy matching, exactly as designed in §1 and §3. The face check (§3) independently confirms the same thing the original observation did: `face_detected: true` at this exact frame.

This result was obtained on a file whose provenance required independent verification before use. A four-check process (header inspection, decoder consistency, splice-boundary inspection, full decode-integrity scan) confirmed the file was sound in the region this search needed, and precisely localized a real defect elsewhere in the file that the search never touches.

(The mouth-motion signal on this exact case reads `False`, not `True`. That's the honest, real-footage finding documented in §3, not a contradiction of the result above. It reflects the baseline-window assumption breaking down on continuous monologue delivery, not doubt about whether the line is on screen; face presence, independently verified twice over, already settles that.)
