#!/usr/bin/env python3
"""Relay MAME wavwrite FIFO output into bounded rotating WAV files."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time


def _read_exact(fh, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = max(0, int(count))
    while remaining > 0:
        chunk = fh.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _valid_wav_header(header: bytes) -> bool:
    return len(header) >= 44 and header[0:4] == b"RIFF" and header[8:12] == b"WAVE"


def _replace_output(output_path: str, header: bytes):
    tmp_path = f"{output_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "wb") as out:
        out.write(header)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp_path, output_path)
    return open(output_path, "ab", buffering=0), len(header)


def _write_bounded_chunk(out, written: int, chunk: bytes, header: bytes, output_path: str, max_bytes: int):
    start = 0
    while start < len(chunk):
        if written >= max_bytes:
            out.close()
            out, written = _replace_output(output_path, header)
        remaining = max_bytes - written
        take = min(remaining, len(chunk) - start)
        if take <= 0:
            out.close()
            out, written = _replace_output(output_path, header)
            continue
        piece = chunk[start:start + take]
        out.write(piece)
        written += len(piece)
        start += len(piece)
    return out, written


def _slot_loop(slot: int, input_fifo: str, output_wav: str, max_bytes: int, stop_event: threading.Event):
    header = None
    while not stop_event.is_set():
        try:
            with open(input_fifo, "rb", buffering=0) as src:
                local_header = _read_exact(src, 44)
                if not _valid_wav_header(local_header):
                    time.sleep(0.05)
                    continue
                header = local_header
                out, written = _replace_output(output_wav, header)
                try:
                    while not stop_event.is_set():
                        chunk = src.read(65536)
                        if not chunk:
                            break
                        out, written = _write_bounded_chunk(out, written, chunk, header, output_wav, max_bytes)
                finally:
                    try:
                        out.close()
                    except Exception:
                        pass
        except FileNotFoundError:
            time.sleep(0.10)
        except Exception as exc:
            print(f"[audio_relay slot={slot}] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            time.sleep(0.10)

    if header is None and os.path.exists(output_wav):
        try:
            os.unlink(output_wav)
        except Exception:
            pass


def _run_supervisor(slot_count: int, audio_dir: str, fifo_template: str, wav_template: str, max_bytes: int) -> int:
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    def _handle_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    for slot in range(max(1, int(slot_count))):
        fifo_path = os.path.join(audio_dir, fifo_template.format(slot=slot))
        wav_path = os.path.join(audio_dir, wav_template.format(slot=slot))
        t = threading.Thread(
            target=_slot_loop,
            args=(slot, fifo_path, wav_path, max_bytes, stop_event),
            daemon=True,
            name=f"audio-relay-{slot}",
        )
        t.start()
        threads.append(t)

    try:
        while not stop_event.is_set():
            time.sleep(0.25)
    except KeyboardInterrupt:
        stop_event.set()

    deadline = time.time() + 1.0
    for t in threads:
        remaining = max(0.0, deadline - time.time())
        t.join(timeout=remaining)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot-count", type=int, required=True)
    parser.add_argument("--audio-dir", default="/tmp")
    parser.add_argument("--fifo-template", default="robotron_audio_client{slot}.fifo")
    parser.add_argument("--wav-template", default="robotron_audio_client{slot}.wav")
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    args = parser.parse_args()

    audio_dir = os.path.abspath(str(args.audio_dir))
    max_bytes = max(200_000, int(args.max_bytes))
    os.makedirs(audio_dir, exist_ok=True)
    return _run_supervisor(
        slot_count=max(1, int(args.slot_count)),
        audio_dir=audio_dir,
        fifo_template=str(args.fifo_template),
        wav_template=str(args.wav_template),
        max_bytes=max_bytes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
