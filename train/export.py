#!/usr/bin/env python3
"""Export a checkpoint for the Dreamcast.

  ../.venv/bin/python export.py --name dc7m
    -> out/dc7m.bin      int8 model (group-quantized, little-endian)
    -> out/tok.bin       tokenizer (llama2.c format)
    -> out/dc7m.test.txt greedy-decode reference from PyTorch, for checking the C code

dc7m.bin layout:
  char[4] "DCT1"; int32 dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, group_size
  Q8 tok_embeddings (also the classifier: tied)
  per layer: f32 attention_norm[dim]; Q8 wq, wk, wv, wo; f32 ffn_norm[dim]; Q8 w1, w2, w3
  f32 norm[dim]
  Q8 tensor = int8 q[n] followed by f32 scale[n / group_size], row-major (out, in)
"""
import argparse, struct
from pathlib import Path
import numpy as np
import torch
import sentencepiece as spm
from model import Config, Transformer

OUT = Path(__file__).resolve().parent / "out"


def q8(w, gs):
    w = w.detach().float().cpu().numpy().reshape(-1, gs)
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.round(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    err = np.abs(q * scale[:, None] - w).max()
    return q.tobytes() + scale.astype("<f4").tobytes(), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dc7m")
    ap.add_argument("--gs", type=int, default=32)
    args = ap.parse_args()
    ck = torch.load(OUT / f"{args.name}.pt", map_location="cpu")
    c = Config(**ck["cfg"])
    m = Transformer(c)
    m.load_state_dict(ck["model"])
    m.eval()

    buf, maxerr = bytearray(b"DCT1"), 0.0
    buf += struct.pack("<8i", c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads,
                       c.vocab_size, c.max_seq_len, args.gs)

    def addq(w):
        nonlocal maxerr
        b, e = q8(w, args.gs)
        maxerr = max(maxerr, e)
        buf.extend(b)

    def addf(w):
        buf.extend(w.detach().float().numpy().astype("<f4").tobytes())

    addq(m.tok_embeddings.weight)
    for L in m.layers:
        addf(L.attention_norm.weight)
        for w in (L.wq, L.wk, L.wv, L.wo):
            addq(w.weight)
        addf(L.ffn_norm.weight)
        for w in (L.w1, L.w2, L.w3):
            addq(w.weight)
    addf(m.norm.weight)
    (OUT / f"{args.name}.bin").write_bytes(buf)
    print(f"{args.name}.bin: {len(buf) / 1e6:.2f} MB, max quant error {maxerr:.4f}")

    # tokenizer, llama2.c format: int32 max_len, then per token: f32 score, int32 len, bytes
    sp = spm.SentencePieceProcessor(model_file=str(OUT / f"tok{c.vocab_size}.model"))
    toks = []
    for i in range(sp.vocab_size()):
        p = sp.id_to_piece(i)
        if sp.is_byte(i):
            b = bytes([int(p[3:5], 16)])
        else:
            b = p.replace("▁", " ").encode("utf-8")
        toks.append((sp.get_score(i), b))
    tb = struct.pack("<i", max(len(b) for _, b in toks))
    for s, b in toks:
        tb += struct.pack("<fi", s, len(b)) + b
    (OUT / "tok.bin").write_bytes(tb)
    print(f"tok.bin: {len(tb) / 1e3:.0f} KB, {len(toks)} tokens")

    # reference greedy decode (fp32 PyTorch) for the C implementation to match
    I = {s: sp.piece_to_id(s) for s in ("<u>", "<dc>", "<e>")}
    lines = []
    for prompt in ("hi", "who are you", "what is your favorite game"):
        x = [sp.bos_id(), I["<u>"]] + sp.encode(prompt) + [I["<dc>"]]
        y = m.generate(torch.tensor([x]), 40, temperature=0, stop={I["<e>"], I["<u>"]})[0].tolist()
        lines.append(f"{prompt}\t{' '.join(map(str, x))}\t{' '.join(map(str, y[len(x):]))}\t{sp.decode(y[len(x):])}")
    (OUT / f"{args.name}.test.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(l.split("\t")[0] + " -> " + l.split("\t")[3] for l in lines))


if __name__ == "__main__":
    main()
