#!/usr/bin/env python3
"""Read-only report on a Dreamcast CD-R in the Mac's drive: sessions, boot header, ISO, XA subheaders.
  python3 tools/inspect_disc.py            (finds the optical disc automatically)"""
import collections, re, struct, subprocess, sys

info = subprocess.run(["drutil", "status"], capture_output=True, text=True).stdout
m = re.search(r"Name:\s+(/dev/disk\d+)", info)
if not m:
    sys.exit("no disc found (drutil status shows no media)")
raw_dev = m.group(1).replace("/dev/disk", "/dev/rdisk")
print(info.strip().splitlines()[2].strip(), "|", raw_dev)

tracks = []
for blk in subprocess.run(["drutil", "trackinfo"], capture_output=True, text=True).stdout.split("info:")[1:]:
    g = lambda k: int(re.search(k + r":\s+(\d+)", blk).group(1))
    tracks.append(dict(n=len(tracks) + 1, session=g("sessionNumber"), mode=g("trackMode"),
                       start=g("trackStartAddress"), size=g("trackSize")))


def read(lba, count):
    with open(raw_dev, "rb", buffering=0) as f:
        f.seek(lba * 2352)
        return f.read(count * 2352)


for t in tracks:
    kind = "AUDIO" if t["mode"] in (0, 1, 2, 3) else "DATA"
    print(f"\nTrack {t['n']}  session {t['session']}  {kind}  start {t['start']}  size {t['size']}")
    if kind == "AUDIO":
        continue
    raw = read(t["start"], min(t["size"], 400))
    secs = [raw[i * 2352:(i + 1) * 2352] for i in range(len(raw) // 2352)]
    d0 = secs[0][24:24 + 2048]
    if d0.startswith(b"SEGA SEGAKATANA"):
        print(f"  boot header: area {d0[0x30:0x38]!r} device {d0[0x20:0x30]!r} boot file {d0[0x60:0x70].strip()!r}")
    else:
        print("  no boot header at track start")
    if len(secs) > 16 and secs[16][25:30] == b"CD001":
        p = secs[16][24:24 + 2048]
        print(f"  ISO volume {p[40:72].strip()!r} root dir at LBA {struct.unpack('<I', p[158:162])[0]}"
              f" (track starts at {t['start']})")
    sub = collections.Counter(s[16:24].hex() for s in secs)
    print(f"  subheaders (first {len(secs)} sectors): " + ", ".join(f"{k}x{v}" for k, v in sub.most_common(4)))
