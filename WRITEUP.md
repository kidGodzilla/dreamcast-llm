# Dreamcast Thinking: a language model on a 1999 console

The Dreamcast launched with the slogan "It's thinking." This project makes that literal. It's a CD-R that boots on
a stock Dreamcast and runs a 7-million-parameter language model on the console's SH-4 CPU, with no network. You type
to it on the Dreamcast keyboard, or on an on-screen keyboard with a controller. It answers as **DC**, the console
itself, living in 1999–2001: proud of its 128 bits, Sonic and Shenmue, and its 56k modem.

On real hardware it writes about **3.6 tokens per second**.

The idea came from bxfoundry's ChatGPT on the SNES, which runs a 2.2M model on a Super FX 2 chip. The Dreamcast has
more headroom: a 200 MHz SH-4 with a 4-wide dot-product instruction, 16 MB of RAM and a CD drive.

## Prior art

gameblabla's llama2.c port for the Dreamcast (MIT) already existed. It runs Karpathy's 260K-parameter TinyStories
model at 30–50 tok/s, and the output is story completion ("Once upon a time..."). That proves the chip can do it,
but it isn't something you'd want to talk to. The goals here were a model big enough to hold a conversation, a
personality, knowledge about its own world, and a keyboard-first chat UI.

## The budget

Everything has to fit in 16 MB of main RAM next to KOS, the framebuffer and the app:

| Item | Size |
|---|---|
| Weights, int8 with an fp32 scale per 32 values (`brain.bin`) | 8.1 MB |
| KV cache (6 layers × 256 positions × 288 dims × K and V, fp32) | 3.5 MB |
| Tokenizer, activations, framebuffer, fact-book index | ~1–2 MB |

That puts the model at about 7M parameters. At fp32, the weights alone (29 MB) wouldn't fit.

## The model

The model uses the llama2.c architecture: RMSNorm, rotary embeddings, SwiGLU and tied input/output embeddings.

- dim 288, 6 layers, 6 heads, FFN hidden 768, context 256 tokens: **7.16M parameters**
- A 4096-token SentencePiece BPE vocabulary with byte fallback
- Conversations use special tokens:

  ```
  <u>what is crazy taxi<dc>It's a game where you drive people around...<e>
  ```
- Loss is computed only on DC's tokens, so the model learns to answer, not to imitate the user.
- Training ran on Apple Silicon with PyTorch MPS. The shipped checkpoint is the best by validation loss, at
  iteration 12,500.

## The data

There was no corpus of "a Dreamcast talking about itself," so the training data was generated with Claude through
`claude -p`, in parallel:

- **Persona chats.** About 66K conversations from Sonnet and about 7.8K from Opus, across 13 weighted categories:
  greetings, hardware, games, Sega history, Y2K-era culture, jokes, feelings, insults, and off-topic questions.
  All of them follow a persona bible (`data/persona.md`) with a checked fact sheet: $199 on 9/9/99, 12x GD-ROM, a
  56k modem in North America, SegaNet, 200-block VMUs.
- **Grounded knowledge.** Plot and spec details from the model's own memory were unreliable, so source text was
  pulled from 462 Wikipedia articles and 427 StrategyWiki pages covering 179 Dreamcast games (CC BY-SA). From that
  source text, Opus wrote about 12K fact-book entries and Sonnet wrote about 18K chats grounded in those entries.
- **A judge pass.** A fact-checking prompt flagged persona chats with wrong facts, and those were dropped.

The hardest rule in the persona bible was about time. DC lives in 1999–2001. Asked directly, a small leak about its
own future is fine. But it never lectures, and it never corrects a user who is playing along. If you say "it's
Christmas 2000 and I just got you!!", DC is thrilled.

## A fact book on the CD

A 7M model can't memorize hundreds of games, so it learned to look things up. When it needs a fact it emits
`<look>Jet Set Radio</look>`. The app then finds that title in `factbook.txt` on the disc and feeds the entry back
as `<book>...`, and the model writes its answer from the entry. The book has 12,499 entries (2.4 MB, sorted by
title). Only the title index is kept in RAM, so the CD does the rest. That's about 18K lookup dialogs' worth of
training.

## Inference on the SH-4

`brain/dcbrain.c` is portable C. The same file builds on the Mac, where `host_test.c` checks it token for token
against PyTorch.

The time goes into matrix-vector products over int8 weights. The SH-4's `FIPR` instruction computes a 4-element
float dot product in one go, so the inner loop converts four int8 weights to float, does one FIPR against the
activations, and repeats eight times per group of 32 before applying the group's scale. It keeps four
accumulators and prefetches the next weights. The code stays readable C (`fipr()` from KOS's `<dc/fmath.h>`)
rather than hand-written assembly, on purpose for the first release.

| Step | Speed |
|---|---|
| Benchmark: naive C mat-vec | 28.4 MMAC/s (Flycast) |
| Benchmark: FIPR mat-vec | 49.6 MMAC/s (Flycast) |
| First full model, plain C | 2.5 tok/s (Flycast) |
| FIPR matmul and attention, precomputed RoPE | 5.3 tok/s (Flycast) |
| Final, with UI and partial redraw | ~5.2 tok/s (Flycast) |
| **Real Dreamcast** | **~3.6 tok/s** |

The emulator overstates speed. Flycast doesn't model memory or cache timing, and every token streams all 8 MB of
weights through a 16 KB data cache from slow RAM. On real hardware the work is memory-bound, which is why the
planned assembly version should interleave the loads with the math.

## The front end

`brain/main_dc.c` uses KallistiOS. It draws into its own RGB565 framebuffer and copies it to video RAM with store
queues.

- **Boot splash.** The model loads from CD in a background thread. A swirl animates on its own clock and a progress
  bar eases toward the actual load progress, so the swirl stays smooth instead of stuttering with each read. The
  VMU screen shows a swirl too.
- **Chat.** The log is word-wrapped with YOU/DC tags. While the model works, the input box says "It's thinking...".
  Only the regions that change are redrawn while tokens stream, which brought about 10% back.
- **Input.** With a keyboard plugged in you just type. With only a controller there's an on-screen keyboard:
  A types, B deletes, X is space, Y is shift and Start sends. Devices are rescanned twice a second, so you can plug
  a keyboard in mid-chat.
- **Context.** When the 256-token context fills, DC says "(my VMU is full, so I'm starting fresh)".
- **Credits.** A credits screen covers the CC BY-SA sources and the open-source pieces.

## Toolchain on a 2026 Mac

- **KallistiOS + GCC 15.2 (sh-elf):** built without sudo. It needed `--disable-plugin` and Homebrew include and
  library paths on macOS 27.
- **mkdcdisc:** builds the self-booting CDI image (IP.BIN, scrambled 1ST_READ.BIN, data files).
- **Flycast:** 2.7 crashes on macOS 27, so it was built from master with a few CMake fixes.
- **dcdib-mac:** burns the CDI's two sessions with cdrecord. It worked under sudo with SIP left on.

## What's next

- **Assembly matmul.** Hand-written SH-4 code that overlaps int8 loads and conversion with FIPR. That targets
  exactly the memory stalls real hardware shows, maybe 1.3–1.5x.
- **A larger model.** Grouped-query attention (2 KV heads) would shrink the KV cache from 3.5 MB to about 1.2 MB,
  leaving room for about 11–12M parameters at int8. It would be more coherent and somewhat slower.

## Credits

See [CREDITS.md](CREDITS.md): Wikipedia and StrategyWiki (CC BY-SA 4.0), KallistiOS, mkdcdisc, dcdib-mac,
Karpathy's llama2.c, and gameblabla's Dreamcast port (MIT). Sega and Dreamcast are trademarks of Sega; this is an
unofficial fan project.
