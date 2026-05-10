#!/usr/bin/env python3
"""Download a small RWC-P subset from Zenodo without fetching the whole ZIP.

Zenodo distributes RWC-P as one large ZIP file. This script uses HTTP range
requests to read the ZIP central directory, select a small number of WAV files,
and download only the compressed bytes needed for those members.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


RWC_P_URL = "https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"


@dataclass
class ZipMember:
    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def http_get_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "rwc-p-subset-downloader/0.1"}
    response = session.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    if response.status_code not in (200, 206):
        raise RuntimeError(f"Unexpected HTTP status for range request: {response.status_code}")
    return response.content


def get_size(session: requests.Session, url: str) -> int:
    response = session.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    return int(response.headers["Content-Length"])


def read_central_directory(session: requests.Session, url: str) -> list[ZipMember]:
    size = get_size(session, url)
    tail_size = min(size, 1024 * 1024)
    tail_start = size - tail_size
    tail = http_get_range(session, url, tail_start, size - 1)

    eocd_sig = b"PK\x05\x06"
    eocd_pos = tail.rfind(eocd_sig)
    if eocd_pos < 0:
        raise RuntimeError("Could not find ZIP end-of-central-directory record.")

    eocd = tail[eocd_pos : eocd_pos + 22]
    (
        _sig,
        _disk_no,
        _cd_disk,
        _entries_disk,
        entries_total,
        cd_size,
        cd_offset,
        comment_len,
    ) = struct.unpack("<4s4H2IH", eocd)
    if comment_len:
        pass

    if entries_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        raise RuntimeError("ZIP64 central directories are not supported by this prototype.")

    central = http_get_range(session, url, cd_offset, cd_offset + cd_size - 1)
    members: list[ZipMember] = []
    pos = 0
    header_fmt = "<4s6H3I5H2I"
    header_size = struct.calcsize(header_fmt)
    for _ in range(entries_total):
        header = central[pos : pos + header_size]
        values = struct.unpack(header_fmt, header)
        sig = values[0]
        if sig != b"PK\x01\x02":
            raise RuntimeError(f"Bad central directory signature at byte {pos}: {sig!r}")
        method = values[4]
        crc32 = values[7]
        compressed_size = values[8]
        uncompressed_size = values[9]
        name_len = values[10]
        extra_len = values[11]
        comment_len = values[12]
        local_header_offset = values[16]

        name_start = pos + header_size
        name_end = name_start + name_len
        name = central[name_start:name_end].decode("utf-8")
        members.append(
            ZipMember(
                name=name,
                method=method,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
            )
        )
        pos = name_end + extra_len + comment_len
    return members


def member_data_offset(session: requests.Session, url: str, member: ZipMember) -> int:
    local_header = http_get_range(
        session,
        url,
        member.local_header_offset,
        member.local_header_offset + 30 - 1,
    )
    (
        sig,
        _version,
        _flag,
        _method,
        _mtime,
        _mdate,
        _crc32,
        _compressed_size,
        _uncompressed_size,
        name_len,
        extra_len,
    ) = struct.unpack("<4s5H3I2H", local_header)
    if sig != b"PK\x03\x04":
        raise RuntimeError(f"Bad local file header signature for {member.name}")
    return member.local_header_offset + 30 + name_len + extra_len


def decompress_member(raw: bytes, member: ZipMember) -> bytes:
    if member.method == 0:
        data = raw
    elif member.method == 8:
        data = zlib.decompress(raw, -15)
    else:
        raise RuntimeError(f"Unsupported ZIP compression method {member.method} for {member.name}")

    crc = binascii.crc32(data) & 0xFFFFFFFF
    if crc != member.crc32:
        raise RuntimeError(f"CRC mismatch for {member.name}: got {crc:08x}, expected {member.crc32:08x}")
    if len(data) != member.uncompressed_size:
        raise RuntimeError(
            f"Size mismatch for {member.name}: got {len(data)}, expected {member.uncompressed_size}"
        )
    return data


def download_member(session: requests.Session, url: str, member: ZipMember, out_root: Path) -> Path:
    data_offset = member_data_offset(session, url, member)
    raw = http_get_range(session, url, data_offset, data_offset + member.compressed_size - 1)
    data = decompress_member(raw, member)
    out_path = out_root / member.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


def select_members(members: Iterable[ZipMember], count: int) -> list[ZipMember]:
    wavs = sorted(
        (m for m in members if m.name.startswith("RWC-P/") and m.name.lower().endswith(".wav")),
        key=lambda m: m.name,
    )
    return wavs[:count]


def write_manifest(out_dir: Path, selected: list[ZipMember], paths: list[Path]) -> None:
    manifest_csv = out_dir / "manifest.csv"
    manifest_json = out_dir / "manifest.json"
    rows = []
    for member, path in zip(selected, paths):
        rows.append(
            {
                "rwc_id": Path(member.name).stem,
                "member": member.name,
                "path": str(path),
                "compressed_size": member.compressed_size,
                "uncompressed_size": member.uncompressed_size,
                "crc32": f"{member.crc32:08x}",
            }
        )

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--url", default=RWC_P_URL)
    parser.add_argument("--out-dir", type=Path, default=Path("data/rwc_p_20"))
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        members = read_central_directory(session, args.url)
        selected = select_members(members, args.count)
        if len(selected) < args.count:
            raise RuntimeError(f"Only found {len(selected)} RWC-P WAV files.")

        if args.list_only:
            for member in selected:
                print(member.name)
            return 0

        paths = []
        for idx, member in enumerate(selected, start=1):
            print(f"[{idx:02d}/{len(selected):02d}] {member.name}", flush=True)
            paths.append(download_member(session, args.url, member, args.out_dir))
        write_manifest(args.out_dir, selected, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
