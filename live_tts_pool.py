#!/usr/bin/env python3
"""
live_tts_pool.py
----------------
Romanian TTS for YouTube live chat (macOS) with:
• Random Romanian voice
• Parallel 'say' synthesis (default 3 workers)
• 150 wpm default (override --rate)
• 5-second freshness budget
• Compresses repeated emoji codes (e.g. ':foo::foo:' → ':foo: 2 times')
• Instant Ctrl-C shutdown
"""

from __future__ import annotations
import argparse, asyncio, datetime as dt, itertools, os, random, re, signal, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import AsyncIterator, Deque, NamedTuple, Tuple

import pytchat

# ───────── constants / paths ───────── #
MAX_LAG_SEC = 5.0
TMP_DIR     = Path("wav_queue"); TMP_DIR.mkdir(exist_ok=True)
LOG_PATH    = Path("chat.log")

# ───────── emoji compressor ───────── #
EMO_RE = re.compile(r'(:[^:\s]+:)(?=\1)')

def compress_emojis(text: str) -> str:
    """Collapse consecutive identical emoji codes into 'emoji n times'."""
    def repl(match: re.Match) -> str:
        token = match.group(1)
        # count how many times token repeats starting here
        span_start = match.start()
        count = 2  # at least two (look-ahead ensures one extra)
        i = span_start + len(token)*2
        while text.startswith(token, i):
            count += 1
            i += len(token)
        return f"{token} de {count} ori"
    # use sub with a function so we can count length
    return re.sub(r'(:[^:\s]+:)(?:\1)+', repl, text)

# ───────── voice probe ───────── #
def ro_voices() -> list[str]:
    out = subprocess.check_output(["say", "-v", "?"], text=True, encoding="utf-8")
    return [ln.split()[0] for ln in out.splitlines() if "(ro_RO)" in ln]

RO_VOICES = ro_voices() or ["Ioana"]

# ───────── data class ───────── #
class ChatLine(NamedTuple):
    ts: dt.datetime
    author: str
    text: str

# ───────── synthesis helper (runs in pool) ───────── #
def synth_to_aiff(args: Tuple[str, str, int, str]) -> str:
    text, voice, rate, outfile = args
    subprocess.run(
        ["say", "-v", voice, "-r", str(rate), "-o", outfile, text],
        check=True
    )
    return outfile

stop_now = asyncio.Event()   # global shutdown flag

# ───────── chat generator + logger ───────── #
async def chat_stream(video_id: str) -> AsyncIterator[ChatLine]:
    chat = pytchat.create(video_id=video_id)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        while chat.is_alive() and not stop_now.is_set():
            for r in chat.get().sync_items():
                raw_msg  = r.message
                clean_msg = compress_emojis(raw_msg)
                line = ChatLine(
                    ts=dt.datetime.fromisoformat(r.datetime.replace("Z", "+00:00")),
                    author=r.author.name,
                    text=clean_msg,
                )
                print(f"[queued] {line.author}: {line.text}", flush=True)
                print(f"{line.ts.isoformat()} {line.author}: {raw_msg}",
                      file=log, flush=True)         # log original
                yield line
            await asyncio.sleep(0.25)

# ───────── sampler drops stale / backlog ───────── #
async def sampler_stream(
    video_id: str,
    speech_wps: float,
) -> AsyncIterator[Tuple[float, ChatLine]]:
    from collections import deque
    pending: Deque[Tuple[float, ChatLine]] = deque()
    queued_words = 0
    async for line in chat_stream(video_id):
        if stop_now.is_set():
            return
        now = time.monotonic()
        wc = len(line.text.split())
        pending.append((now, line)); queued_words += wc

        while pending and now - pending[0][0] > MAX_LAG_SEC:
            queued_words -= len(pending.popleft()[1].text.split())
        while queued_words / speech_wps > MAX_LAG_SEC and pending:
            queued_words -= len(pending.popleft()[1].text.split())

        yield now, line

# ───────── main async runner ───────── #
async def run(video_id: str, rate: int, workers: int) -> None:
    voices      = RO_VOICES
    speech_wps  = rate / 60
    pool        = ProcessPoolExecutor(max_workers=workers)
    idx_counter = itertools.count(1)
    q: asyncio.Queue[Tuple[int, float, ChatLine, str, asyncio.Future[str]]] = (
        asyncio.Queue(maxsize=500)
    )
    loop = asyncio.get_running_loop()

    async def produce() -> None:
        async for arrival, line in sampler_stream(video_id, speech_wps):
            if stop_now.is_set():
                break
            idx   = next(idx_counter)
            voice = random.choice(voices)
            out   = TMP_DIR / f"{idx}.aiff"
            fut   = loop.run_in_executor(
                pool, synth_to_aiff,
                (f"{line.author} spune {line.text}", voice, rate, str(out))
            )
            await q.put((idx, arrival, line, voice, fut))

    async def play() -> None:
        expected = 1
        pending: dict[int, Tuple[float, ChatLine, str, asyncio.Future[str]]] = {}
        while not stop_now.is_set():
            # fill map
            while expected not in pending and not stop_now.is_set():
                try:
                    item = await asyncio.wait_for(q.get(), timeout=0.2)
                    pending[item[0]] = item[1:]
                    q.task_done()
                except asyncio.TimeoutError:
                    continue

            if stop_now.is_set():
                break
            arrival, line, voice, fut = pending[expected]
            path = await fut
            if time.monotonic() - arrival <= MAX_LAG_SEC:
                proc = await asyncio.create_subprocess_exec("afplay", path)
                await proc.wait()
                print(f"[played|{voice}] {line.author}: {line.text}", flush=True)
            os.unlink(path)
            del pending[expected]
            expected += 1

    tasks = [asyncio.create_task(produce()), asyncio.create_task(play())]

    def on_sigint() -> None:
        stop_now.set()
        for t in tasks:
            t.cancel()

    loop.add_signal_handler(signal.SIGINT, on_sigint)

    try:
        await asyncio.gather(*tasks)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for p in pool._processes.values():
            p.kill()
        for f in TMP_DIR.glob("*.aiff"):
            try: f.unlink()
            except FileNotFoundError: pass

# ───────── CLI entry ───────── #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--video-id", required=True,
                   help="YouTube video ID or full URL")
    p.add_argument("--rate", type=int, default=150,
                   help="Speech rate in wpm (default 150)")
    p.add_argument("--workers", type=int, default=3,
                   help="Parallel synthesis processes (default 3)")
    args = p.parse_args()
    vid = args.video_id.split("v=")[-1][:11]

    try:
        asyncio.run(run(vid, args.rate, args.workers))
    except KeyboardInterrupt:
        print("\n⏹  Stopped by user.", file=sys.stderr)
