"""Write cpu_sites.txt: the executable's Zen 2 instruction sites, which gta6rt replaces before running the game.

Usage:
    python -I tools/cpu_sites.py <game_scarlett_master.exe> <out/cpu_sites.txt> [--scan scan.json]

Runs tools/cpu_feature_scan.py --sites --json on the image (or reads a scan it already wrote, --scan) and writes the
RVAs of its SSE4a, CLZERO, MONITORX and CLWB sites, after an `image <SizeOfImage> <TimeDateStamp>` line that ties
them to this executable (runtime/src/cpu_patch.h refuses a file made for another one). The file holds addresses
only, no code, but it describes the player's copy: it stays beside gta6rt.exe and is never shared.
Prints the number of sites per form. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ("SSE4A", "CLZERO", "MONITORX", "CLWB")


def image_identity(path: str) -> tuple[int, int]:
    """(SizeOfImage, TimeDateStamp) of a PE32+ image."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if len(head) < 0x40 or head[:2] != b"MZ":
        raise ValueError(f"{path} is not a PE image")
    pe = struct.unpack_from("<I", head, 0x3C)[0]
    if pe + 24 + 60 > len(head) or head[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"{path} is not a PE image")
    stamp = struct.unpack_from("<I", head, pe + 8)[0]
    size_of_image = struct.unpack_from("<I", head, pe + 24 + 56)[0]
    return size_of_image, stamp


def site_rvas(scan: dict) -> list[int]:
    rvas = set()
    for name in FEATURES:
        for site in scan.get("features", {}).get(name, {}).get("sites", []):
            rvas.add(int(site["rva"]))
    return sorted(rvas)


def write_sites(path: str, identity: tuple[int, int], rvas: list[int]) -> None:
    staged = path + ".part"
    with open(staged, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Zen 2 instruction sites of this executable (tools/cpu_sites.py); keep on this machine\n")
        f.write(f"image {identity[0]:x} {identity[1]:x}\n")
        for rva in rvas:
            f.write(f"{rva:x}\n")
    os.replace(staged, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--scan", help="a cpu_feature_scan.py --sites --json output to read instead of scanning")
    a = ap.parse_args()
    identity = image_identity(a.image)
    if a.scan:
        with open(a.scan, encoding="utf-8") as f:
            scan = json.load(f)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "scan.json")
            with open(out, "w", encoding="utf-8", newline="\n") as f:
                r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "cpu_feature_scan.py"), a.image, "--sites",
                                    "--json"], stdout=f)
            if r.returncode != 0:
                print(f"cpu_sites: cpu_feature_scan.py exited with {r.returncode}", file=sys.stderr)
                return 1
            with open(out, encoding="utf-8") as f:
                scan = json.load(f)
    rvas = site_rvas(scan)
    write_sites(a.out, identity, rvas)
    for name in FEATURES:
        print(f"{name}: {len(scan.get('features', {}).get(name, {}).get('sites', []))} sites")
    print(f"cpu_sites: {len(rvas)} sites -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
