# Recognizer on real handwriting: the RxHand word dataset

First rigorous measurement of the M4/M5 recognizer (Kraken, fine-tuned from the Rev City base model) on real
handwriting. Everything here is reproducible from the scripts in `scripts/rxhand_*.py`; no dataset content is in the
repo (the dataset licence is unknown, so it is neither redistributed nor committed; a handful of label strings are
quoted below as examples).

**Read this first.** These are 128x128 crops of single handwritten words (mostly medication names), by an unknown
number of writers. They test the *recognizer* on real handwriting at word level. They cannot test segmentation,
they are not paragraphs, and the app's single-writer assumption does **not** hold for this data. The numbers are an
upper-level orientation for the real project, not a prediction of its accuracy (caveats in section 9).

## 1. Setup and reproduction

Hardware: GTX 1050 Ti (4 GB, about 1.2 GB used by the Windows display), WSL2 Ubuntu, Kraken 7.1.1, PyTorch 2.9.1.

```
# copy the dataset once into the gitignored data/external/rxhand/ (data/*.jpg + Prescription_Labels.csv)
make base-model                                           # Rev City base model (Apache-2.0)
uv run --project backend python scripts/rxhand_import.py --data-dir data/experiments/rxhand/x3 --stretch 3
uv run --project backend python scripts/rxhand_experiment.py baseline --project data/experiments/rxhand/x3
uv run --project backend python scripts/rxhand_experiment.py train --project data/experiments/rxhand/x3 \
    --name curve_nall --n 0 --lr 3e-4 --batch 8 --auto --keep-model          # --n 100|250|500|1000|2500|0 (all)
uv run --project backend python scripts/rxhand_analyze.py curve   --project data/experiments/rxhand/x3
uv run --project backend python scripts/rxhand_analyze.py analyze --project data/experiments/rxhand/x3 --run curve_nall
```

* `scripts/rxhand_import.py` builds an isolated project (its own `--data-dir`, never the app's; it refuses the default
  one). Each image becomes a one-line page via the new service `services/presegmented.py` (not reachable from the
  API): original stored byte-identical (verified with `cmp`), working copy = the word canvas, one line whose polygon
  is the whole image and whose baseline spans it, label as manual ground truth. No segmentation, no line-height
  rescale, no deskew. The existing import/segmentation behaviour and all existing tests are unchanged.
* `scripts/rxhand_experiment.py` uses the app's own `KrakenTrainer` and `KrakenRecognizer` (not the job queue, because
  the learning curve needs a validation set disjoint from the training set and two held-out sets scored separately;
  see "Finding about the app" in section 9). One smoke run through the app's real train and predict **jobs** on a
  400-image project also passed on the GPU (3 epochs, line preprocessing key recorded on the model row, predictions
  written).
* Outputs (results JSON with every prediction, models, the contact sheet) live under
  `<project>/experiment/` inside `data/`, which git ignores.

## 2. The data

5,578 images, all 128x128 JPEG, one word each, labels in a CSV (`Images,Text`, utf-8-sig, trailing empty columns).

| observation | count |
|---|---|
| rows / unique raw labels / unique after case-folding | 5,578 / 1,787 / 1,536 |
| labels that occur once (after case-folding) | 969 (1,147 raw) |
| case-variant groups of the same word (e.g. `Algin`/`algin`) | 218 groups, 2,757 samples (49%) |
| labels with outer whitespace (stripped by the importer) | 24 distinct labels, 31 rows |
| labels with a `?` (`???`, a half-readable word) | 3 labels, 4 rows (dropped) |
| multi-word labels (inner whitespace) | 344 distinct labels (e.g. `Ace Power`) |
| imported | 5,574 |

* **DPI tag is a source marker.** 4,587 images are tagged 240, 990 are tagged 609.6 and form one **contiguous block
  of file ids** (P2106-P3095), one image is tagged 72. The 610 block is darker and flatter (mean grey 173, std 25 vs
  mean 200, std 44 for the 240 images), so it is most plausibly a different scanning/photo batch and probably different
  writers. Content is mixed between groups (same drug names appear in both). The importer records the tag per image;
  all results below are also split by it.
* **Labels follow a convention that is not always "what is written".** Case is inconsistent for the same word, and
  labels sometimes expand an abbreviation that the writer clearly abbreviated (images reading `Inj.` or `Tab.` labelled
  `Injection` / `Tablet`), and in a visible share of the worst predictions the label does not match the visible text
  at all (section 8). Treat the CSV as noisy ground truth.
* Writer count unknown, so no writer-disjoint split is possible. **The single-writer assumption of the app does not hold
  here**: a model trained and tested on this data learns "many hands", which is a different (usually easier to
  generalize across, harder to fit tightly) problem than one person's notes.

## 3. The CTC time-step pitfall and the preprocessing decision

Kraken rescales a line crop to the **model's input height keeping the aspect ratio**, and the network emits one CTC
time step per 8 px of the rescaled width (plus 16 px of padding on each side). The task description assumed about 48 px; the Rev City model's input height is actually **120 px**. Measured with the real network on every 4th image
(`rxhand_experiment.py ctc`):

| horizontal stretch of the crop | time steps per image | mean steps per character | labels needing more steps than available | labels with fewer than 2 steps per required step ("tight") |
|---|---|---|---|---|
| none (x1; 128x128 -> 120x120) | 19 | 3.1 | 0 | 12.7% |
| x2 | 34 | 5.5 | 0 | 0% |
| x3 | 49 | 8.0 | 0 | 0% |
| x4 | 64 | 10.4 | 0 | 0% |

So training is not impossible (no label is longer than the 19 available steps; the CTC minimum includes the blank
between repeated letters), but the density is low and 1 in 8 labels is tight. Ordinary text lines give roughly 7-8 steps
per character (an estimate, not measured here), so the unstretched square is also far from the shape the base model
knows (these crops look horizontally squeezed).

**Decision: stretch each crop horizontally by 3 (the "word canvas"), once, at import.** `pipeline/wordcrop.py`:
`apply_word_canvas(image, WordCanvas(stretch_x=3))` is the single function (grayscale, bicubic horizontal stretch,
no other change). It writes the page's *working copy*, which is the image both the training export and the predict job
crop from, so training and prediction cannot see different preprocessing (a test asserts that the crop the predict job
passes to the recognizer and the training-export crops are exactly the working copy). The choice is recorded on the page
(`pages.preprocess`: `{"mode": "word_canvas", "key": "word_canvas:x3"}`) and on every fine-tuned model
(`hyperparameters.line_preprocess`); the train and predict jobs refuse to mix keys, and refuse a model on pages with
another key (a pretrained base model has no constraint). Ordinary page imports are untouched (key `page`).

Pilot (validation set only, N=1000 training words, same recipe for every arm, Rev City base model):

| stretch | validation CER, lr 3e-4, volume-based epochs (1 seed) | validation CER, lr 1e-3 (seeds 1234 / 77, an unstable recipe, see section 5) |
|---|---|---|
| x1 (without the fix) | 0.398 | 0.406 / 0.400 |
| x2 | 0.243 | 0.566 / 0.495 |
| **x3 (chosen)** | **0.223** | 0.250 / 0.588 |
| x4 | 0.224 | 0.230 / 0.626 |

With a stable recipe the stretch matters a lot (CER 0.398 -> 0.223 at N=1000; on the held-out sets at N = all: A 0.185
-> 0.137, B 0.468 -> 0.335, section 6). Stretch x2 is slightly worse and x4 equal to x3 but twice as slow and with
2.6x the memory (peak 1,344 MB vs 517 MB), so x3. A per-image autocontrast (percentile clipping) was also tried at the
unstable lr (validation 0.567 / 0.500 against 0.250 / 0.588 without): no benefit, not adopted. The word canvas has an
optional `autocontrast` parameter (off) for it. The pretrained model without any fine-tuning also gains from the
stretch: baseline CER on A/B is 0.898/0.902 at x1, 0.744/0.759 at x2, 0.701/0.734 at x3.

## 4. Evaluation design (two fixed held-out sets)

The data has many repeats of the same word, so a random split mostly measures memorization. Two fixed held-out sets, the
same for every training size (`services/wordsplits.py`, decided before any model was trained):

* **B, unseen words**: 12% of the distinct *case-folded* labels (hash `htr-unseen-v1`), **all** their samples:
  702 samples, 181 labels. Never trained on, never used for validation. Case variants go out together.
* **A, seen words**: of the remaining samples, those picked by the app's existing hash rule (`natural_split`, ~12%),
  *kept only if their label still occurs in training* (a last-of-its-label sample goes back to training, so A is
  "seen words" by construction): 464 samples, 207 labels. A is biased toward frequent words (it has no singleton labels).
* **validation**: 250 samples (lowest `htr-val-v1` hash of the rest), for early stopping and for every decision
  below. Disjoint from A, B and the training samples. **train**: the remaining 4,158 samples in a fixed pseudo-random
  order, so N = 100, 250, 500, 1000, 2500 are nested prefixes of it, N = all is 4,158.
* The app's `line_splits` table is filled with A and B as `eval` and everything else as `train`
  (`WordSplits.app_split`), so the app's own train job is also leak-free with respect to these sets. Tests
  (`test_rxhand.py`) assert: parts partition the data and are disjoint; no unseen label (case-folded) occurs in
  train, validation or A; every A label occurs in training; nested subsets never contain held-out samples; the check
  detects an injected leak; results are deterministic and independent of input order.
* The share of A whose label is in the training *subset* grows with N (44% at N=100, 62%, 74%, 85%, 95%, 100% at all):
  at small N "seen words" is mostly unseen, so A converges to B-like difficulty on the left of the curve. Reported below.
* Metrics (`pipeline/evaluate.py`, case-sensitive, micro-averaged): CER; WER (the share of words wrong; for
  single-word labels this equals 1 - exact match, for the 344 multi-word labels it differs slightly, WER can exceed 1
  when a model adds words); exact-match accuracy; **95% bootstrap CI of the CER** (resampling samples, 2,000
  resamples, fixed seed; it covers test-sample variance, not training-run variance).
* B is dominated by a few words: `Tablet` 146 samples (21%), `Algin` 78 (11%), `Normal` 33 (5%). `Tablet` alone is
  40% of all B edits at N = all (the writers often wrote `Tab.`, the label says `Tablet`, section 8). Column "B CER
  without its top label" gives the effect; it was computed after seeing the results and is a diagnostic, not a
  second held-out set.

## 5. Training recipe, and what was changed (honest log)

Constant for every N (stated before the final curve): start = Rev City base model, AdamW, **lr 3e-4**, batch 8,
Kraken augmentation on, 3-stretch canvas, validation on the fixed 250 samples, seed 1234. The recipe is defined in
training *volume*, not epochs (`auto_recipe`): an epoch always draws at least 1,000 samples (a smaller training set is
repeated, with fresh augmentation each time), at most 100 such units (12,500 steps), early stopping after 15 units
without validation improvement (min. 4 epochs). So N=100 is 10 repetitions per epoch, N=4,158 is 23 epochs.

What was iterated, on the validation set only:

1. First recipe (lr 1e-3, patience 6, epochs capped at 40-60): **unstable**. The same setting landed at validation CER
   0.25 or 0.59 depending on the seed (x3), and the stretch comparison made with it was noise. A learning curve with it was
   non-monotonic (N=500 worse than N=100). Discarded (kept in the result files as `recipe1_*`).
2. Learning rate: 5e-4 -> 0.570, 1e-3 -> 0.250/0.588, 2e-3 -> 0.763 (diverges) at the first recipe; with the
   volume-based recipe 1e-4 -> 0.230 (90 epochs, slow) and **3e-4 -> 0.223** (51 epochs). A too-high learning rate was
   destroying the pretrained features; the app's default (1e-4) is safe but slow, 3e-4 is chosen on validation CER.
3. Autocontrast and "no augmentation" were tried at the unstable lr and did not help.
4. Then stretch factors were re-compared at the stable recipe (section 3) and the curve was run once.

Caveat: my pilot runs printed held-out numbers in their summary lines, so I saw A/B values for the pilot arms; every
decision above was taken from the validation CER. Run-to-run noise at identical seed and recipe (GPU nondeterminism):
N=1000, lr 3e-4 gave validation 0.223 and 0.218, A 0.216 and 0.204, B 0.386 and 0.391. Differences smaller than about
0.01-0.02 CER between curve points are not meaningful. One seed per N is reported for the final curve.

## 6. Results

### 6.1 Baseline: the pretrained models, no fine-tuning

(CER [95% CI] | WER | exact match; A n=464, B n=702.)

| model, preprocessing | A seen words | B unseen words |
|---|---|---|
| Rev City, x1 (no stretch) | 0.898 [0.887-0.910] \| 1.019 \| 0.000 | 0.902 [0.891-0.912] \| 1.009 \| 0.000 |
| Rev City, x2 | 0.744 [0.727-0.763] \| 1.052 \| 0.004 | 0.759 [0.743-0.775] \| 1.035 \| 0.004 |
| **Rev City, x3 (the pipeline's base model + the fix)** | 0.701 [0.683-0.720] \| 1.048 \| 0.000 | 0.734 [0.717-0.751] \| 1.020 \| 0.006 |
| Rev City, x4 | 0.701 [0.681-0.722] \| 1.029 \| 0.002 | 0.738 [0.720-0.756] \| 1.027 \| 0.004 |
| McCATMuS (alternative, x3) | 0.530 [0.507-0.553] \| 1.061 \| 0.017 | 0.608 [0.587-0.629] \| 1.056 \| 0.026 |

The base model alone is unusable on this data (CER 0.70, effectively no exact matches). WER above 1 means the models
insert words.

### 6.2 Learning curve (Rev City, x3 canvas, recipe of section 5)

| N train | epochs | train s | peak VRAM MB | A seen words: CER [95% CI] | WER | exact | B unseen words: CER [95% CI] | WER | exact | B CER without `Tablet` | A labels present in training subset | validation CER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (baseline) | - | - | - | 0.701 [0.683-0.720] | 1.048 | 0.000 | 0.734 [0.717-0.751] | 1.020 | 0.006 | - | - | 0.674 |
| 100 | 57 | 1000 | 517 | 0.369 [0.343-0.396] | 0.845 | 0.162 | 0.512 [0.491-0.534] | 0.944 | 0.058 | 0.440 | 44% | 0.372 |
| 250 | 35 | 614 | 517 | 0.310 [0.287-0.335] | 0.788 | 0.220 | 0.456 [0.436-0.476] | 0.913 | 0.087 | 0.402 | 62% | 0.317 |
| 500 | 69 | 1210 | 517 | 0.236 [0.215-0.260] | 0.683 | 0.332 | 0.405 [0.384-0.426] | 0.873 | 0.121 | 0.345 | 74% | 0.259 |
| 1000 | 75 | 1302 | 517 | 0.204 [0.184-0.227] | 0.618 | 0.369 | 0.391 [0.369-0.414] | 0.841 | 0.151 | 0.316 | 85% | 0.218 |
| 2500 | 24 | 964 | 517 | 0.163 [0.143-0.185] | 0.522 | 0.470 | 0.350 [0.328-0.374] | 0.774 | 0.218 | 0.269 | 95% | 0.178 |
| 4158 (all) | 23 | 1506 | 1428 | **0.137 [0.117-0.158]** | 0.444 | 0.552 | **0.335 [0.314-0.358]** | 0.756 | 0.235 | 0.251 | 100% | 0.155 |

(Epochs are volume-based units; "train s" is the fine-tune only. Peak VRAM is PyTorch's peak allocation measured by
the app's trainer; the driver's view adds the CUDA context. Throughput is about 60 training samples per second at x3.)

Reading it: the curve is monotone and still falling at 4,158 words. Word accuracy on *seen* words reaches 55%, on
*unseen* words 24%. The gap between A and B (0.137 vs 0.335 CER) is the cost of never having seen the word; it does not
close with more data in this range (A improves 0.369 -> 0.137, B 0.512 -> 0.335).

### 6.3 With and without the stretch, at N = all (same recipe, one seed)

| | A CER [CI] | A exact | B CER [CI] | B exact | peak VRAM |
|---|---|---|---|---|---|
| without (x1) | 0.185 [0.164-0.208] | 0.429 | 0.468 [0.448-0.487] | 0.111 | 455 MB |
| with the word canvas (x3) | 0.137 [0.117-0.158] | 0.552 | 0.335 [0.314-0.358] | 0.235 | 1,428 MB |

At N=1000 the same comparison (x1 vs x3) is A 0.340 vs 0.204 (pilot arm 0.216), B 0.569 vs 0.391 (pilot arm 0.386).

### 6.4 Training from scratch, and an alternative pretrained model

| run (N = all unless noted) | A CER [CI] | A exact | B CER [CI] | B exact | notes |
|---|---|---|---|---|---|
| Rev City fine-tune (above) | 0.137 [0.117-0.158] | 0.552 | 0.335 [0.314-0.358] | 0.235 | 23 epochs, 25 min |
| **from scratch** (random weights, same architecture, lr 1e-3, 60 epochs max, patience 10) | 0.524 [0.495-0.553] | 0.138 | 0.808 [0.794-0.822] | 0.000 | still improving at the end of its 65-minute budget: it needs far more data or time; not competitive |
| McCATMuS fine-tune (same recipe) | 0.149 [0.128-0.171] | 0.532 | 0.364 [0.341-0.387] | 0.207 | 14 epochs; baseline was better than Rev City (0.530 / 0.608) but fine-tuned it is not (differences within noise at A, slightly worse at B) |
| McCATMuS fine-tune, N=1000 | 0.201 [0.180-0.224] | 0.405 | 0.385 [0.364-0.407] | 0.172 | vs Rev City N=1000: 0.204 / 0.391: equal |

Alternative model tried: of the 60 entries in the HTRMoPo listing I looked for native Kraken recognition models that
read handwriting and the Latin alphabet in a format `KrakenTrainer` can fine-tune. **McCATMuS**
(doi 10.5281/zenodo.13788177, CC-BY-4.0, "handwritten, printed and typewritten documents, 16th-21st century", reported
CER 7.2% on its own test data) was downloaded (setup-time only, into `data/models/alt/`, not registered as an app base
model) and used. Not used: PP-OCRv6 (another architecture, not trainable by this trainer), the Party multilingual models
(different framework), and the period/language specific models (medieval, Fraktur, Kurrent, Hebrew, Arabic, French
administration). Nothing there is English *modern* handwriting besides Rev City.

### 6.5 App path

The app's own train and predict jobs (queue, worker, real Kraken, GPU) were run on a 400-image project: 3 epochs on 304
words, 56 held out (combined A+B, 0.744 -> 0.529 CER), model row has `line_preprocess: word_canvas:x3`, 40 unlabeled
words predicted in 0.8 s. The learning-curve numbers above come from the same `KrakenTrainer` / `KrakenRecognizer`
classes driven directly.

## 7. Confidence and active learning on real data

Best model (Rev City, N = all). Confidence is the geometric mean of per-character maxima (M5; a ranking, not a
probability).

| | A seen words | B unseen words |
|---|---|---|
| Spearman correlation, confidence vs per-sample CER | -0.573 | -0.111 |
| AUROC, wrong predictions ranked below right ones | 0.823 | 0.631 |
| mean confidence, right / wrong predictions | 0.976 / 0.905 | 0.945 / 0.896 |
| wrong predictions found in the lowest-confidence 20% (random order: 20%) | **38.0%** | 23.8% |
| share of all character edits in the lowest 20% | 46.9% | 28.6% |
| CER 0.137 / 0.335 after a reviewer corrects the lowest 20% (random 20%: 0.110 / 0.268) | **0.073** | 0.239 |
| after correcting the lowest 30% (random: 0.096 / 0.234) | 0.057 | 0.207 |

The M5 premise holds on words the model has seen (the lowest-confidence fifth holds nearly twice its share of wrong
predictions, and correcting it halves the CER), but it is weak on unseen words, where the model is confidently wrong
(0.945 vs 0.896) and three quarters of all predictions are wrong anyway, so there is little ranking left to do. In the
real project (the user's own words recur in their notes) the A situation is the one that applies as labelling goes on.

## 8. Error analysis (best model, N = all)

CER by group (A / B); case-insensitive CER is 0.113 / 0.281 (case-sensitive 0.137 / 0.335), so case errors are
about 0.024-0.054 CER.

| group | A CER (n) | B CER (n) |
|---|---|---|
| label length 1-4 | 0.205 (63) | 0.254 (78) |
| 5-7 | 0.112 (241) | 0.376 (490) |
| 8-10 | 0.130 (126) | 0.255 (112) |
| 11+ | 0.213 (34) | 0.292 (22) |
| DPI tag 240 | 0.148 (383) | 0.366 (596) |
| DPI tag 610 | **0.083** (81) | **0.163** (106) |
| label occurs once in the dataset | (none by construction) | 0.303 (127) |
| label occurs 2-9 times | 0.161 (148) | 0.243 (167) |
| label occurs 10+ times | 0.125 (316) | 0.394 (408, mostly `Tablet`) |
| multi-word label | 0.160 (55) | 0.290 (62) |

* The 610 DPI block (a different source) is read about twice as well. Confounded (different writers, darker, flatter
  scans): it shows the data is not homogeneous, not that "610 is better".
* Shortest (1-4) and longest (11+) labels are worst; the longest are multi-word or truncated.
* Top confusions: on A mainly **case** (`M->m` 9, `c->C` 5, `n->N` 5, `X->x` 5, `d->D` 4) and deleted `e`; on B the
  deletions `e` 147, `l` 137, `t` 125 and `T->t` 125 are almost entirely `Tablet` being read as `Tab`/`tos`.
* Edit operations on A: 297 substitutions, 107 deletions, 35 insertions; on B: 860 / 537 / 92. The model under-reads
  (deletes) more than it hallucinates.

**Contact sheet of the 40 worst predictions** (saved under `data/experiments/rxhand/x3/experiment/out/`, gitignored; I
looked at all 40). Failure modes, by eye (counts are my reading, subjective):

1. **The label does not match the visible word** (about 14 of 40): the image clearly shows one brand name and the
   label is another (for example a clearly written `Domin` labelled `Sergel`). In most of these the model read the
   image correctly and is scored wrong. A noticeable share of the CSV is wrong or
   follows a rule (generic vs brand name?) that cannot be learned from the image. This sets a floor under CER that no
   better model can pass.
2. **Abbreviation expanded in the label** (about 8 of 40): images reading `Inj` / `Tab.` labelled `Injection` /
   `Tablet`; the model, trained on both conventions, answers `Injection` for an image labelled `inj` and `Tb`/`tos`
   for `Injection`/`Tablet`. This is exactly the project's "diplomatic transcription" question, seen from the other side:
   mixed conventions in the labels are costly.
3. **Hard cursive, distortion, multi-word crops** (the rest): long multi-syllable drug names, compressed three-word
   labels (one such label appears three times in the sheet, each read with a word dropped or garbled), margin ink and
   noisy backgrounds (a crop with extra handwriting above the word), a few crops cut at the edge.
4. Case is genuinely ambiguous (an upper-case label read as a capitalized word).

## 9. Interpretation and caveats

* **What the numbers say.** With about 4,000 labelled words and the stretch fix, the pretrained-and-fine-tuned
  recognizer gets a word right 55% of the time when it has seen the word and 24% of the time when it has not
  (CER 0.14 / 0.34). With 100 words it is already much better than the base model (CER 0.37 vs 0.70) but not usable.
  Fine-tuning is worth it from the first 100 samples; the benefit keeps growing to the end of the tested range.
* **Caveats, explicitly.** (1) Word-level *drug names*: short, mostly Latin-script brand names with a strong prior
  (a lexicon would help, see below); not paragraphs, no context, no segmentation. (2) Unknown number of **writers**; the
  app assumes one, so this says little about how fast a single writer's model improves (one hand is usually *easier*
  for a given amount of data, but the data here cannot show it). (3) 128x128 crops resized from unknown shapes: aspect
  ratio and stroke thickness are distorted, which the stretch only partly repairs. (4) **Lexicon overlap**: A is built
  from words that occur in training, and a closed list of 1,536 drug names is repeated across 5,574 samples; the
  figures for seen words reflect that closed vocabulary. Real notes have a different mix of personal vocabulary and
  free text. (5) **Label noise** (section 8) caps the
  measurable accuracy; true CER is lower than measured. (6) One training seed per curve point, run-to-run noise
  about 0.01 CER, CIs cover test sampling only. (7) A, B and validation come from the same pool of sources; B was
  chosen by hash and happens to contain `Tablet`.
* **Not representative of the user's own paragraph notes.** It is evidence about the recognizer stage on real
  handwriting; line segmentation, line length (30+ characters), punctuation and abbreviations in running text are
  untested here.
* **Finding about the app.** In the app's train job (`services/training.export_run`) the validation lines used for
  early stopping and best-epoch selection are a *subset of the training lines* (the `val` XML files are written from
  rows that are also in `train`). Early stopping therefore watches data the model trains on, which favours later
  epochs and overstates the "best validation accuracy". The held-out set is unaffected, so reported CER is honest.
  Not changed here (outside the brief and the existing behaviour is tested); worth fixing for small data, where it
  matters most. The experiment driver uses a disjoint validation set.

## 10. Recommendations for the real project

1. **Keep the Rev City base model, with fine-tuning; it is adequate but not strong.** It cannot read modern
   handwriting on its own (CER 0.70 here), and the alternative McCATMuS is only better *before* fine-tuning; after
   fine-tuning they are equivalent. Training from scratch is not an option at this data size. No further base model
   is worth chasing from the public repository for English cursive; revisit TrOCR (the plan's fallback) only if
   fine-tuned CER on the user's own lines stays above about 10% after a few hundred lines.
2. **Fine-tune with a learning rate of about 3e-4, not 1e-3.** At 1e-3 the same setting gave CER 0.25 or 0.59 by
   seed at N=1000, at 1e-4 it needed twice the time. The app's default is 1e-4: it is safe; consider 3e-4 once more
   data is in, and keep early stopping patient (15 validations or more) because learning is slow before it jumps.
3. **How many labelled lines?** On this word data, CER roughly halves between 100 and 1,000 words (0.37 -> 0.20 on seen
   words) and halves again by 4,000 (0.14). Treat about 100 lines as "a model exists, review everything", about 500
   as "predictions save time on repeated words" (exact match 33% on seen words), and 1,000+ as the range where
   active-learning review (section 7) really pays. Lines are longer than words, so expect the useful thresholds in
   *lines* to be lower than the word counts here, but this experiment cannot say by how much: measure it on the first
   real labelled batch with the app's held-out CER and its learning-curve view.
4. **Review lowest-confidence first only once the model has seen the vocabulary.** The confidence ranking is
   informative on seen words (38% of errors in the lowest 20%) and nearly useless on unseen ones; early rounds with a
   weak model are mostly "review everything" regardless of order.
5. **Preprocessing matters as much as data.** Any pipeline that crops words or short lines must make sure the
   network sees about 7-8 time steps per character; the word canvas must be recorded with the model (done).
6. **A lexicon / abbreviation-aware post-processing would help** for drug names: a large part of the remaining errors
   are near-misses of known words (a long drug name read with two or three wrong letters) and a dictionary of the user's own words (their
   own transcribed lines, plus the abbreviation dictionary of M6) could snap predictions with low confidence to the
   nearest known word. Not built; test it on this data first (it can be evaluated offline from the saved predictions:
   A is the best case, B the failure case where a lexicon would only hurt).
7. **Decide the transcription convention before labelling.** The dataset shows the cost of mixing "diplomatic" and
   "expanded" labels (`Inj.` vs `Injection`). The plan's diplomatic policy is the right one; keep expansion in the
   abbreviation layer.
8. **Fix the validation leak in the train job** (section 9) before relying on early stopping with a few hundred lines.
