#!/usr/bin/env python3
"""Download the official Spain GPT-2 artifacts with pinned size and MD5 checks.

Small parallel ranges avoid slow single-response Zenodo downloads. The .partial
file is a contiguous prefix from an interrupted sequential download, if present.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.request

FILES = {
    2: (649085903, "c9e01115f61ae8306f0a897b499b29f7"),
    32: (649182147, "ec452e7995bbae2b13245027428e0476"),
}


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reuse_weights(directory):
    """Reconstruct seq32 using identical seq2 TensorProto bytes, then check MD5.

    The release stores the same 162M float weights in both protobuf files. Only
    small sequence-dependent tensors and graph metadata need downloading again.
    No guessed bytes are accepted: the entire resulting official file must match
    its publisher's checksum before it receives the final .onnx name.
    """
    import mmap
    source = directory / "gpt2-seq-2.onnx"
    if source.stat().st_size != FILES[2][0] or md5(source) != FILES[2][1]:
        raise ValueError("reused seq2 source failed publisher checksum")
    target = directory / "gpt2-seq-32.onnx"
    size, expected = FILES[32]
    url = f"https://zenodo.org/api/records/21045357/files/{target.name}/content"
    prefix = target.with_suffix(".partial")
    prefix_bytes = prefix.read_bytes() if prefix.exists() else b""
    blocks = {}
    transferred = 0

    def remote(offset, length):
        nonlocal transferred
        if offset + length <= len(prefix_bytes):
            return prefix_bytes[offset:offset + length]
        chunks = []
        while length:
            block = offset // 65536
            if block not in blocks:
                start, end = block * 65536, min(size, (block + 1) * 65536) - 1
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("server did not honor exact protobuf range")
                    blocks[block] = response.read()
                if len(blocks[block]) != end - start + 1:
                    raise ValueError("truncated protobuf byte range")
                transferred += len(blocks[block])
            within = offset % 65536
            part = blocks[block][within:within + length]
            chunks.append(part); length -= len(part); offset += len(part)
        return b"".join(chunks)

    def varint(read, at):
        result, shift = 0, 0
        while True:
            value = read(at, 1)[0]; at += 1
            result |= (value & 127) << shift
            if value < 128:
                return result, at
            shift += 7
            if shift > 63:
                raise ValueError("invalid protobuf varint")

    def fields(read, start, end):
        while start < end:
            tag, at = varint(read, start)
            wire = tag & 7
            if wire == 2:
                length, payload = varint(read, at); after = payload + length
            elif wire == 0:
                _, after = varint(read, at); payload = at
            elif wire in (1, 5):
                payload, after = at, at + (8 if wire == 1 else 4)
            else:
                raise ValueError(f"unsupported protobuf wire type {wire}")
            if after > end:
                raise ValueError("protobuf field exceeds message")
            yield tag >> 3, start, payload, after
            start = after

    def tensor_name(read, start, end):
        for tag, _, payload, after in fields(read, start, end):
            if tag == 8:
                return read(payload, after - payload).decode()
        raise ValueError("TensorProto name absent")

    with source.open("rb") as source_file, mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ) as source_map:
        read_source = lambda offset, length: source_map[offset:offset + length]
        tensors = {}
        for tag, _, payload, after in fields(read_source, 0, len(source_map)):
            if tag == 7:
                for graph_tag, _, body, end in fields(read_source, payload, after):
                    if graph_tag == 5 and end - body >= 1 << 20:
                        tensors[tensor_name(read_source, body, end)] = (body, end)
        temporary = target.with_suffix(".reuse")
        reused = 0
        with temporary.open("wb") as out:
            for tag, start, payload, after in fields(remote, 0, size):
                out.write(remote(start, payload - start))
                if tag != 7:
                    out.write(remote(payload, after - payload)); continue
                for graph_tag, field_start, body, end in fields(remote, payload, after):
                    out.write(remote(field_start, body - field_start))
                    if graph_tag == 5 and end - body >= 1 << 20:
                        name = tensor_name(remote, body, end)
                        begin_source, end_source = tensors[name]
                        if end_source - begin_source != end - body:
                            raise ValueError(f"tensor {name} differs in byte length")
                        out.write(source_map[begin_source:end_source])
                        reused += end - body
                        print(f"Reused {name}: cumulative {reused} bytes", flush=True)
                    else:
                        out.write(remote(body, end - body))
        if temporary.stat().st_size != size or md5(temporary) != expected:
            raise ValueError("reconstructed official file failed publisher checksum")
        temporary.replace(target)
        print(f"Verified {target}: MD5 {expected}, reused {reused}, downloaded {transferred} bytes", flush=True)


def download(sequence_length, directory, workers=4):
    size, expected = FILES[sequence_length]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"gpt2-seq-{sequence_length}.onnx"
    if target.exists():
        if target.stat().st_size != size or md5(target) != expected:
            raise ValueError(f"checksum mismatch for existing {target}")
        print(f"Verified existing {target}", flush=True)
        return
    partial = target.with_suffix(".partial")
    ranged = target.with_suffix(".ranged")
    prefix = partial.stat().st_size if partial.exists() else 0
    if prefix > size:
        raise ValueError("partial file exceeds expected artifact length")
    if partial.exists():
        shutil.copyfile(partial, ranged)
    fd = os.open(ranged, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    url = f"https://zenodo.org/api/records/21045357/files/{target.name}/content"

    def fetch(start):
        end = min(start + (8 << 20), size) - 1
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("server did not honor exact byte range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError("truncated byte range")
                written = 0
                while written < len(data):
                    written += os.pwrite(fd, data[written:], start + written)
                return len(data)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    complete = prefix
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, start) for start in range(prefix, size, 8 << 20)]
            for future in as_completed(futures):
                complete += future.result()
                print(f"Downloaded {complete}/{size} bytes", flush=True)
    finally:
        os.close(fd)
    if md5(ranged) != expected:
        raise ValueError("downloaded artifact failed pinned MD5 verification")
    ranged.replace(target)
    if partial.exists():
        partial.unlink()
    print(f"Verified {target}: {size} bytes, MD5 {expected}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_length", type=int, choices=FILES)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-seq2-weights", action="store_true")
    args = parser.parse_args()
    if args.reuse_seq2_weights:
        if args.sequence_length != 32:
            raise ValueError("weight reuse is only for seq32")
        reuse_weights(args.directory)
    else:
        download(args.sequence_length, args.directory, args.workers)
