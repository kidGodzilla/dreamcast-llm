# Dreamcast: It's Actually Thinking

A Sega Dreamcast you chat with. A 7M-parameter language model runs on the console's 200 MHz SH-4, from a
CD-R, with no network. It writes about 3.6–3.9 tokens per second on real hardware. It answers as **DC**, the
console itself, living in 1999–2001.

```
YOU  sup?
DC   Not much, just sitting here being 128 bit. How about you? Anything fun going on today?
```

The full story is in [WRITEUP.md](WRITEUP.md): the memory budget, the training data, the CD fact book, the
FIPR inner loop, and the dead laser.

## What's in the repo

| Path | Contents |
|---|---|
| `brain/dcbrain.c`, `dcbrain.h` | The inference engine: int8 weights, FIPR dot products, tokenizer, sampler. Portable C. |
| `brain/main_dc.c` | The Dreamcast front end (KallistiOS): boot splash, chat, on-screen keyboard, fact-book lookups, credits. |
| `brain/host_test.c` | Runs the same engine on a desktop and compares it with PyTorch token for token. |
| `brain/disc/` | The files that go on the CD: `brain.bin` (int8 model, 8.1 MB), `tok.bin`, `factbook.txt` (12,499 entries). |
| `train/` | PyTorch model, data preparation, training, int8 export, desktop chat. |
| `data/` | Scripts that generated the training conversations with `claude -p`, and the persona bible (`persona.md`). |
| `sources/` | Wikipedia and StrategyWiki text used for the fact book (CC BY-SA 4.0), and the scripts that fetched it. |
| `app/` | Milestone 0: a keyboard console with a mat-vec benchmark (naive C against FIPR). |
| `tools/inspect_disc.py` | A read-only report on a burned Dreamcast CD-R in a Mac's drive. |

The generated conversations (~80 MB) and the PyTorch checkpoints aren't in the repo.

## Build the disc

You need KallistiOS with its sh-elf toolchain, and [mkdcdisc](https://gitlab.com/simulant/mkdcdisc). The
Makefile expects them in `sdk/` next to `brain/`:

```
sdk/kos/                     KallistiOS (source sdk/kos/environ.sh before building)
sdk/mkdcdisc/builddir/       mkdcdisc
```

1. Build the app and the image:
   ```bash
   cd brain
   source ../sdk/kos/environ.sh
   make && make cdi
   ```
2. Test `thinking.cdi` in [Flycast](https://github.com/flyinghead/flycast). To type on the emulated keyboard,
   set port 1 to a keyboard in Flycast's settings.
3. Burn `thinking.cdi` to a CD-R with any Dreamcast CDI burner. On a Mac,
   [dcdib-mac](https://github.com/jfsantos/dcdib-mac) works:
   ```bash
   sudo dcdib-mac thinking.cdi
   ```

Building KallistiOS on macOS 27 needed two changes:
- Add `--disable-plugin` to the GCC configure flags in `utils/kos-chain/scripts/init.mk`.
- Append Homebrew's `CPATH` and `LIBRARY_PATH` to `environ.sh`.

If a burned disc won't boot, test a disc that is known to work before you change the image.
`tools/inspect_disc.py` shows whether a burned disc matches what mkdcdisc wrote.

## On the console

- **Keyboard:** type and press Enter.
- **Controller only:** the app shows an on-screen keyboard.
  - A types. B deletes. X is space. Y is shift. Start sends.
- **Hot-plug:** you can connect or disconnect a keyboard while the app runs.
- **Commands:** type `/credits`, or press C on the title screen, for the credits.

## Check the engine on a desktop

```bash
cd brain
cc -O2 -o host_test host_test.c dcbrain.c -lm
./host_test <model.bin> <tok.bin> <model.test.txt>
```

`train/export.py` writes all three files. The test file holds PyTorch's greedy output for a few prompts.
`host_test` must produce the same tokens.

## Train your own

Python packages: `torch`, `numpy`, `sentencepiece`. The source fetchers also need `mwparserfromhell`. Training
uses PyTorch MPS on Apple Silicon.

1. Fetch the sources:
   ```bash
   cd sources
   python3 fetch_wikipedia.py
   python3 fetch_strategywiki.py
   ```
2. Generate the data. Each script can resume: run the same command again to continue.
   ```bash
   cd data
   python3 gen.py --calls 4000 --jobs 12                  # persona chats -> raw/
   python3 factbook.py dialogs --jobs 8 --calls 1500      # lookup chats  -> fb/dialogs/
   python3 grounded.py entries --jobs 8 --model opus      # fact book     -> gfb/entries/
   python3 grounded.py chats --jobs 8 --model sonnet      # grounded chats -> gfb/chats/
   python3 judge.py --jobs 6                              # fact-check raw/ -> judged/
   ```
3. Prepare the data, train and export:
   ```bash
   cd train
   python3 prepare.py                                     # 4096-token BPE tokenizer + token/mask files
   python3 train.py --name dc7m --dim 288 --layers 6 --heads 6 --hidden 768 --iters 30000
   python3 export.py --name dc7m                          # out/dc7m.bin, out/tok.bin, out/dc7m.test.txt
   python3 chat.py --name dc7m                            # chat on the desktop
   ```
4. Build the disc files:
   - Copy `out/dc7m.bin` to `brain/disc/brain.bin` and `out/tok.bin` to `brain/disc/tok.bin`.
   - Run `python3 brain/make_factbook.py` to rebuild `factbook.txt`.

## The model

llama2.c architecture: RMSNorm, rotary embeddings, SwiGLU and tied embeddings.

| | |
|---|---|
| Size | dim 288, 6 layers, 6 heads, FFN 768, context 256 tokens, 7.16M parameters |
| Vocabulary | 4,096 SentencePiece BPE tokens with byte fallback |
| Weights | int8, groups of 32, with an fp32 scale per group |
| Chat format | `<u>user<dc>reply<e>`. Loss is computed only on DC's tokens. |
| Lookups | `<look>Topic</look><book>entry<dc>answer`. The app reads the entry from `factbook.txt` on the CD. |

The shipped weights were trained on about 66K persona chats from Sonnet, 7.8K from Opus, 18K lookup dialogs and
18K chats grounded in source text.

## Ideas for later

- Hand-written SH-4 assembly for the matmul that overlaps int8 loads with FIPR. Real hardware is memory-bound,
  and the emulator hides that.
- Grouped-query attention (2 KV heads). It frees about 2.3 MB of RAM, room for an 11–12M model.

## License

- Code: MIT, see [LICENSE](LICENSE). The inference code follows Andrej Karpathy's llama2.c (MIT) and
  gameblabla's Dreamcast port (MIT).
- `brain/disc/factbook.txt` and `sources/`: CC BY-SA 4.0, derived from Wikipedia and StrategyWiki.
- Attribution details: [CREDITS.md](CREDITS.md).

Sega, Dreamcast and all game names are trademarks of their owners. This is an unofficial, non-commercial fan
project.
