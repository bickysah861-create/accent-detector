# Accent Coach

Speak into your microphone. Get your accent, a pronunciation score, and a
fluency review. All of it is measured from your audio by machine learning
models running on your own computer.

No GPU needed. No API key needed. Nothing is sent anywhere.

## What you get back

Every time you speak, the system tells you:

* **Accent** picked from 23 options, with a confidence level
* **Pronunciation** scored out of 100, plus which sounds and words to practise
* **Fluency** scored out of 100, covering pace, pausing and expressiveness
* **A reply** that keeps the conversation going

It takes about 2 seconds.

## Install

You need Python 3.12 and ffmpeg.

```bash
git clone https://github.com/sakrit204-byte/accentlearning_ml.git
cd accentlearning_ml

python3 -m venv .venv
./.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r requirements.txt
```

That installs the CPU build of PyTorch. It is about 1 GB.

## Run

```bash
./.venv/bin/python -m uvicorn src.server:app --port 8000
```

Open **http://127.0.0.1:8000** and press the microphone button.

The first start takes a minute while the speech models download. After that it
starts immediately.

## How well it works

Tested on speakers the model had never heard before:

* Accuracy: **50.3 percent**
* Macro F1: **0.427**
* Random guessing would give: 4.3 percent

It is good at spotting a first language accent. Finnish, Dutch and Indian all
score above 0.80. It is weak at telling apart native varieties of English,
because Irish, Scottish and Northern Irish genuinely sound alike. When it is
unsure it says so, so trust the pronunciation and fluency numbers more than a
single accent guess.

Full details are in [the paper](paper/accent_paper.pdf).

## Retraining it yourself

Only do this if you want to rebuild the model. It takes a few hours and
downloads about 9 GB.

```bash
./.venv/bin/python -m src.data.scan          # index the corpus
./.venv/bin/python -m src.data.build_split   # split it by speaker
./.venv/bin/python -m src.features.extract   # cache the audio features
./run_training.sh                            # calibrate and train
```

While the last step runs, open **http://127.0.0.1:8000/train** to watch it
train live. You can see the loss curve, which part of the speech model it is
learning to trust, and the score for each accent updating in real time.

## How it works

1. **Whisper** writes down what you said.
2. A frozen **WavLM** encoder turns your audio into a fingerprint. A small
   trained classifier reads that fingerprint and picks the accent.
3. A **phoneme model** checks each sound you made against the sound the word
   should have. That gives the pronunciation score.
4. Plain signal processing measures your pauses, speed and pitch.
5. A writer turns those numbers into a friendly reply.

The important part: **steps 1 to 4 produce every number.** Step 5 only chooses
words. It cannot invent or change a score.

## A note on the data

The corpus this model learns from ships with train and test sets that share
every speaker. Because each speaker has one accent, a model can score well just
by recognising voices. This project splits the data again so that no speaker
appears in more than one set. That is why the accuracy above looks lower than
numbers you may see elsewhere. It is the honest one.

## Project layout

```
src/analysis/   speech recognition, pronunciation scoring, prosody
src/features/   the WavLM feature extractor
src/train/      model, training loop, live dashboard data
src/data/       corpus indexing and speaker safe splitting
src/server.py   the web backend
web/            the chat page and the training dashboard
paper/          the write up
```
