#!/usr/bin/env python3
"""Train DC's brain on Apple Silicon (MPS).

  ../.venv/bin/python train.py --name smoke --dim 128 --layers 4 --heads 4 --hidden 352 --iters 1500
  ../.venv/bin/python train.py --name dc7m  --dim 288 --layers 6 --heads 6 --hidden 768 --iters 30000
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm
from model import Config, Transformer

OUT = Path(__file__).resolve().parent / "out"
PROMPTS = ["hi", "who are you", "what's your favorite game", "what is a smartphone",
           "i'm sad today", "tell me a joke", "how much ram do you have", "what is a lion"]


def batches(split, bs, T, dev):
    tok = np.memmap(OUT / f"{split}.tok", dtype=np.uint16, mode="r")
    msk = np.memmap(OUT / f"{split}.mask", dtype=np.uint8, mode="r")
    while True:
        ix = np.random.randint(0, len(tok) - T - 1, bs)
        x = torch.from_numpy(np.stack([tok[i:i + T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([tok[i + 1:i + T + 1] for i in ix]).astype(np.int64))
        m = torch.from_numpy(np.stack([msk[i + 1:i + T + 1] for i in ix]).astype(np.float32))
        yield x.to(dev), y.to(dev), m.to(dev)


def loss_fn(model, x, y, m):
    logits = model(x)
    l = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
    return (l * m.view(-1)).sum() / m.sum().clamp(min=1)


def chat_samples(model, sp, dev, temperature=0.7):
    ids = {s: sp.piece_to_id(s) for s in ("<u>", "<dc>", "<e>")}
    out = []
    for p in PROMPTS:
        x = torch.tensor([[sp.bos_id()] + [ids["<u>"]] + sp.encode(p) + [ids["<dc>"]]], device=dev)
        y = model.generate(x, 60, temperature=temperature, stop={ids["<e>"], ids["<u>"]})
        out.append(f"  > {p}\n    {sp.decode(y[0, x.shape[1]:].tolist()).replace('<e>', '').replace('<u>', '')}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dim", type=int, default=288)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--kv-heads", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(1999)
    np.random.seed(1999)
    c = Config(dim=args.dim, n_layers=args.layers, n_heads=args.heads,
               n_kv_heads=args.kv_heads or args.heads, vocab_size=args.vocab,
               hidden_dim=args.hidden, max_seq_len=args.seq, dropout=args.dropout)
    model = Transformer(c).to(dev)
    sp = spm.SentencePieceProcessor(model_file=str(OUT / f"tok{args.vocab}.model"))
    assert sp.vocab_size() == args.vocab
    n = model.n_params()
    print(f"{args.name}: {n / 1e6:.2f}M params ({n / 1e6:.1f} MB int8, {n * 4 / 1e6:.1f} MB fp32) on {dev}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    train, val = batches("train", args.bs, args.seq, dev), batches("val", args.bs, args.seq, dev)
    ckpt = OUT / f"{args.name}.pt"
    best, t0 = 1e9, time.time()
    for it in range(args.iters + 1):
        lr = args.lr * min(1, (it + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / args.iters)))
        for g in opt.param_groups:
            g["lr"] = lr
        if it % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(loss_fn(model, *next(val)).item() for _ in range(20)) / 20
            print(f"it {it:6d}  val {vl:.3f}  lr {lr:.2e}  {time.time() - t0:.0f}s", flush=True)
            if vl < best:
                best = vl
                torch.save({"cfg": c.__dict__, "model": model.state_dict(), "it": it, "val": vl}, ckpt)
            if it % (args.eval_every * 4) == 0 and it:
                print(chat_samples(model, sp, dev), flush=True)
            model.train()
        x, y, m = next(train)
        loss = loss_fn(model, x, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.load_state_dict(torch.load(ckpt)["model"])
    model.eval()
    print(f"best val {best:.3f}\n" + chat_samples(model, sp, dev))


if __name__ == "__main__":
    main()
