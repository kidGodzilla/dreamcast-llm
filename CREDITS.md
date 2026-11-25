# Credits and licenses

## Source text (CC BY-SA 4.0)

DC's fact book and parts of its training conversations are derived from:

- **Wikipedia** (https://en.wikipedia.org), articles on the Dreamcast, Sega, Dreamcast games and
  related topics. Text available under the Creative Commons Attribution-ShareAlike 4.0 License.
  Per-article titles, URLs and revision ids: `sources/wikipedia/*.json`.
- **StrategyWiki** (https://strategywiki.org), guides for Dreamcast games. Text available under the
  Creative Commons Attribution-ShareAlike 4.0 License. Per-page titles, URLs and revision ids:
  `sources/strategywiki/*.json`.

Because of the ShareAlike terms, the fact book shipped on the disc (and the entries it is built from)
is licensed CC BY-SA 4.0 as well. The disc must show an attribution screen crediting Wikipedia and
StrategyWiki contributors.

## Generated data

Training conversations were written by Anthropic's Claude (Sonnet and Opus) from a persona brief and,
for grounded batches, from the source passages above.

## Software

- KallistiOS and its toolchain builder (BSD-style license), https://github.com/KallistiOS/KallistiOS
- mkdcdisc (Simulant), https://gitlab.com/simulant/mkdcdisc
- Model architecture and inference follow llama2.c by Andrej Karpathy (MIT); the Dreamcast port by
  gameblabla (MIT) is a reference implementation.

Sega, Dreamcast and all game names are trademarks of their owners. This is an unofficial,
non-commercial fan project.
