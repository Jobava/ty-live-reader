from __future__ import annotations
"""
Offline Romanian live-chat TTS
  engines: vits | your_tts | mms | fastspeech2 (local)

Examples
  python live_tts_ro_chat2.py --self-test-all
  python live_tts_ro_chat2.py -v eRXVIXWnqAA                  # default vits
"""

import argparse, asyncio, datetime as dt, itertools, logging, regex as re, subprocess, time, os, signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator, Deque, NamedTuple, Tuple

import pytchat                # mandatory dependency
from ro_diacritics import restore_diacritics

# ───────── logging
logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)

# ───────── paths
TMP_DIR = Path("wav_tmp"); TMP_DIR.mkdir(exist_ok=True)
LOG_PATH = Path("chat2.log")
EMO_PATH = Path("emojis.txt")

# ───────── model catalogue
MODEL_CATALOG = {
    "vits"        : ("coqui",       "tts_models/ro/cv/vits"),
    # "your_tts"    : ("coqui",       "tts_models/multilingual/multi-dataset/your_tts"),
    # "mms"         : ("transformer", "facebook/mms-tts-ron"),
    # "fastspeech2" : ("coqui_local", None),  # needs --fs2-path
}
DEFAULT_MODEL = "vits"

# ───────── emoji dictionary
def load_emojis(fp: Path) -> tuple[dict[str, str], dict[str, str]]:
    d = {}
    unicode_map = {}
    if fp.is_file():
        for ln in fp.read_text("utf-8").splitlines():
            if ln and not ln.startswith("#"):
                try:
                    parts = dict(seg.split("=", 1) for seg in ln.split(";") if "=" in seg)
                    en = parts.get("en", "").strip()
                    ro = parts.get("ro", "").strip()
                    uni = parts.get("unicode", "").strip() if "unicode" in parts else None
                    if en and ro:
                        d[en] = ro
                    if en and uni:
                        unicode_map[uni] = f":{en}:"
                except Exception:
                    pass
    return d, unicode_map

EMO_RO, UNICODE_EMOJI_TO_NAME = load_emojis(EMO_PATH)

EMOJI = r':([^\s:]+):'
WORD = r'\p{L}+'

def unicode_to_shortcode(text):
    for uni, name in UNICODE_EMOJI_TO_NAME.items():
        text = text.replace(uni, name)
    return text

def normalize_emojis(text: str) -> str:
    """
    Normalize all :emoji_name: to lowercase if in EMO_RO.
    Unicode emoji normalization is not supported unless a mapping is provided.
    """
    return re.sub(EMOJI, lambda m: f":{m.group(1).lower()}:" if m.group(1).lower() in EMO_RO else m.group(0), text)

def fold(pat, s):
    """
    Compress repeated patterns (words or :emoji_name:).
    E.g., ':smile: :smile: :smile:' -> ':smile: de 3 ori'
    """
    return re.sub(
        rf'({pat})(?:\s*\1){{2,}}',
        lambda m: f"{m.group(1)} {' ' if not m.group(1).endswith(':') else ''}{ro_times(len(re.findall(pat, m.group(0), flags=re.I)))}",
        s, flags=re.I
    )

def tokenize_for_compression(text):
    # Tokenize by splitting on spaces, but keep :emoji_name: and hyphenated words (with diacritics) as single tokens
    # Romanian diacritics: ăâîșşțţĂÂÎȘŞȚŢ
    return re.findall(r':[^\n\s:]+:|[\wăâîșşțţĂÂÎȘŞȚŢ]+(?:-[\wăâîșşțţĂÂÎȘŞȚŢ]+)*', text, flags=re.UNICODE)

def compress_sequences(text, min_ngram=2, max_ngram=5):
    tokens = tokenize_for_compression(text)
    n = max_ngram
    while n >= min_ngram:
        i = 0
        new_tokens = []
        while i < len(tokens):
            # Look for at least 3 consecutive n-gram repeats
            if i + n*3 <= len(tokens):
                ngram = tokens[i:i+n]
                reps = 1
                while i + reps*n < len(tokens) and tokens[i+reps*n:i+(reps+1)*n] == ngram:
                    reps += 1
                if reps >= 3:
                    new_tokens.extend(ngram)
                    new_tokens.append(f"de {reps} ori")
                    i += reps * n
                    continue
            new_tokens.append(tokens[i])
            i += 1
        tokens = new_tokens
        n -= 1
    return ' '.join(tokens)

def replace_emojis_with_ro(text: str) -> str:
    # Replace :emoji_name: with Romanian translation if available
    return re.sub(EMOJI, lambda m: EMO_RO.get(m.group(1).lower(), m.group(0)), text)

def compress(t: str) -> str:
    t = unicode_to_shortcode(t)
    t = normalize_emojis(t)
    t = compress_sequences(t)
    t = fold(EMOJI, t)
    t = fold(WORD, t)
    t = replace_emojis_with_ro(t)
    return t

DIGITS = "zero unu doi trei patru cinci șase șapte opt nouă".split()
DIGMAP  = {str(i): DIGITS[i] for i in range(10)}
ALLOWED = re.compile(r"[a-zA-ZăâîșşțţÁÂÎȘŞȚŢ ]")
def clean(t:str) -> str: return "".join(DIGMAP.get(c,c) if not ALLOWED.match(c) and c!=" " else c for c in t)

def ro_times(n:int) -> str: return f"de {n} ori"

# ───────── Synth wrapper
class Synth:
    warn_lang = False
    def __init__(self, key:str, fs2_path:Path|None=None):
        engine, mid = MODEL_CATALOG[key]
        if key == "fastspeech2":
            if fs2_path: engine, mid = "coqui_local", str(fs2_path.expanduser())
            else:        raise RuntimeError("--fs2-path needed for fastspeech2")

        if engine == "coqui":
            from TTS.api import TTS as CoquiTTS
            model = CoquiTTS(model_name=mid, gpu=False)
            kw = {}
            if getattr(model, "speakers", None): kw["speaker"] = model.speakers[0]
            if getattr(model, "languages", None):
                kw["language"] = "ro" if "ro" in model.languages else model.languages[0]
                if kw["language"] != "ro" and not Synth.warn_lang:
                    logging.warning("%s: no 'ro' embedding, using '%s'", key, kw["language"])
                    Synth.warn_lang = True
            self.tts = lambda txt, path, spd: model.tts_to_file(txt, file_path=path, speed=spd, **kw)

        elif engine == "coqui_local":
            from TTS.utils.synthesizer import Synthesizer
            p = Path(mid)
            model = Synthesizer(tts_config_path=p/"config.json",
                                tts_checkpoint_path=p/"model.pth",
                                vocoder_config_path=p/"vocoder_config.json",
                                vocoder_checkpoint_path=p/"vocoder.pth")
            self.tts = lambda txt, path, spd: model.tts_to_file(txt, file_path=path, speaker_idx=0)

        elif engine == "transformer":          # Meta MMS
            from transformers import pipeline
            pipe = pipeline("text-to-audio", model=mid, device="cpu")
            self.tts = lambda txt, path, spd: Path(path).write_bytes(pipe(txt)["audio"])

        else:
            raise RuntimeError("engine not supported")

    def to_file(self, text:str, path:str, speed:float) -> str:
        self.tts(text, path, speed)
        return path

# ───────── async pipeline
class Line(NamedTuple):
    ts: dt.datetime; auth: str; txt: str
stop_flag = asyncio.Event()

async def chat_stream(video_id:str) -> AsyncIterator[Line]:
    chat = pytchat.create(video_id=video_id)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        while chat.is_alive() and not stop_flag.is_set():
            for r in chat.get().sync_items():
                ln = Line(dt.datetime.fromisoformat(r.datetime.replace("Z","+00:00")),
                          r.author.name, compress(r.message))
                logging.info("[queued] %s: %s", ln.auth, ln.txt)
                log.write(f"{ln.ts.isoformat()} {ln.auth}: {r.message}\n"); log.flush()
                yield ln
            await asyncio.sleep(0.25)

async def sampler(video_id:str, wps:float) -> AsyncIterator[Tuple[float, Line]]:
    from collections import deque
    dq:Deque[Tuple[float,Line]] = deque(); words = 0
    async for ln in chat_stream(video_id):
        now=time.monotonic(); words+=len(ln.txt.split()); dq.append((now,ln))
        while dq and now-dq[0][0]>5: words-=len(dq.popleft()[1].txt.split())
        while words/wps>5 and dq:    words-=len(dq.popleft()[1].txt.split())
        yield now, ln

async def run(video_id:str, speed:float, workers:int, model:str, fs2:Path|None):
    synth = Synth(model, fs2)
    wps   = 3/speed
    loop  = asyncio.get_running_loop()
    pool  = ThreadPoolExecutor(max_workers=workers)
    q: asyncio.Queue[Tuple[int,float,Line,asyncio.Future]] = asyncio.Queue(600)
    idx = itertools.count(1)

    async def producer():
        async for arr, ln in sampler(video_id, wps):
            if stop_flag.is_set(): break
            i=next(idx); wav=str(TMP_DIR/f"{i}.wav")
            diacritized_text = restore_diacritics(clean(ln.txt))
            fut = loop.run_in_executor(pool, synth.to_file,
                                       f"{ln.auth} spune {diacritized_text}", wav, speed)
            await q.put((i,arr,ln,fut))

    async def player():
        expect=1; pend={}
        while not stop_flag.is_set():
            while expect not in pend and not stop_flag.is_set():
                try:
                    item=await asyncio.wait_for(q.get(),0.3)
                    pend[item[0]]=item[1:]; q.task_done()
                except asyncio.TimeoutError:
                    continue
            if stop_flag.is_set(): break
            arr,ln,fut=pend[expect]; path=await fut
            if time.monotonic()-arr<=5:
                proc = await asyncio.create_subprocess_exec("afplay", path)
                await proc.wait()
                logging.info("[played] %s: %s", ln.auth, ln.txt)
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            del pend[expect]; expect+=1

    tasks=[asyncio.create_task(producer()),asyncio.create_task(player())]

    async def shutdown():
        print("Received termination signal, shutting down...", flush=True)
        # Start a watchdog thread to force exit if shutdown hangs
        def force_exit_watchdog():
            import time as _time
            _time.sleep(2)
            print("[watchdog] Forcing exit after timeout.", flush=True)
            os._exit(1)
        import threading
        threading.Thread(target=force_exit_watchdog, daemon=True).start()
        for t in tasks:
            t.cancel()
        stop_flag.set()
        await asyncio.sleep(0.1)
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
        except NotImplementedError:
            # add_signal_handler may not be available on Windows
            pass

    try:
        await asyncio.gather(*tasks)
    finally:
        pool.shutdown(wait=False,cancel_futures=True)
        for f in TMP_DIR.glob("*.wav"):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

# ───────── self-tests
def self_test(key:str, fs2:Path|None):
    try:
        logging.info("▶ testing %s …", key)
        synth = Synth(key, fs2)
        out = TMP_DIR / f"{key}.wav"
        synth.to_file("Bună ziua! Acesta este un test.", str(out), 1.0)
        subprocess.run(["afplay", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out.unlink()
        logging.info("✅ %s OK", key)
    except Exception as exc:
        logging.warning("⏭ skipped %s: %s", key, exc)

def self_test_all(fs2:Path|None):
    for k in MODEL_CATALOG:
        try: self_test(k, fs2)
        except Exception as e:
            logging.warning("⏭ skipped %s: %s", k, e)

# ───────── CLI
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-v","--video-id", help="YouTube video ID or URL")
    ap.add_argument("--model",choices=MODEL_CATALOG,default=DEFAULT_MODEL)
    ap.add_argument("--speed",type=float,default=1.0)
    ap.add_argument("--workers",type=int,default=1)
    ap.add_argument("--fs2-path",type=Path, help="Dir with FastSpeech2 model files")
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--self-test-all",action="store_true")
    ap.add_argument("--log-level",default="INFO",choices=["DEBUG","INFO","WARNING","ERROR"])
    args=ap.parse_args()
    logging.getLogger().setLevel(args.log_level)

    if args.self_test_all:
        self_test_all(args.fs2_path); sys.exit(0)
    if args.self_test:
        self_test(args.model, args.fs2_path); sys.exit(0)
    if not args.video_id:
        ap.error("need --video-id or self-test flag")

    vid = args.video_id.split("v=")[-1][:11]
    asyncio.run(run(vid, args.speed, args.workers, args.model, args.fs2_path))
