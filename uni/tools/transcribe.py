"""Transcribe Chinese audio with faster-whisper."""
import sys
from faster_whisper import WhisperModel
import time

model_name = "large-v3"
audio = "/home/txy/lb/uni_videos/BV1VZLi6PE2B.mp3"
out_txt = "/home/txy/lb/uni_videos/BV1VZLi6PE2B.large-v3.txt"
out_srt = "/home/txy/lb/uni_videos/BV1VZLi6PE2B.large-v3.srt"

print(f"Loading model {model_name}...")
t0 = time.time()
model = WhisperModel(model_name, device="cpu", compute_type="int8")
print(f"Loaded in {time.time()-t0:.0f}s")

print("Transcribing...")
t1 = time.time()
segments, info = model.transcribe(audio, language="zh", beam_size=5, vad_filter=True)
print(f"Detected lang={info.language} prob={info.language_probability:.2f} dur={info.duration:.0f}s")

def srt_time(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

text_parts = []
srt_parts = []
n = 0
for seg in segments:
    n += 1
    text_parts.append(seg.text.strip())
    srt_parts.append(f"{n}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{seg.text.strip()}\n")
    if n % 20 == 0:
        print(f"  {n} segs at t={seg.end:.0f}s ({time.time()-t1:.0f}s elapsed)")

with open(out_txt, "w") as f:
    f.write("\n".join(text_parts))
with open(out_srt, "w") as f:
    f.write("\n".join(srt_parts))
print(f"Done in {time.time()-t1:.0f}s; {n} segments")
print(f"Saved: {out_txt}, {out_srt}")
