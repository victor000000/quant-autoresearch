#!/usr/bin/env python3
"""Transcribe Chinese audio with faster-whisper LARGE-V3. Outputs .txt and .srt.

Upgrade from `base` model:
  - large-v3 is ~3 GB (~20x bigger than base ~140 MB)
  - Much better accuracy on Chinese technical content
  - Slower: expect 1-3x realtime on 8 CPU threads with int8
  - First run downloads the model (~3 GB) from HuggingFace
"""
import os, sys, json, time
from faster_whisper import WhisperModel

OUT = "/home/txy/lb/uni_yt"
MODEL_NAME = "large-v3"
COMPUTE_TYPE = "int8"     # int8 quantization → ~2-3x faster than fp16 on CPU
                          # with negligible accuracy loss vs fp16


def fmt_ts(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def transcribe_file(model, src, txt_out, srt_out):
    print(f"[{time.strftime('%H:%M:%S')}] start {os.path.basename(src)}", flush=True)
    t0 = time.time()
    segments, info = model.transcribe(
        src,
        language="zh",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
        initial_prompt="以下是一段关于量化交易、机器学习、金融的中文技术分享，"
                       "包含术语：自定义轴、重采样、标签、特征工程、降维、模型组合、"
                       "概率聚类、变点检测、自编码器、变分自编码器、PCA、HMM、CRF、"
                       "波动率、套利、趋势、震荡、回测、推理、实盘、机器学习、"
                       "分数阶差分、整数阶差分、样本熵、信息熵、卡马、最大回撤。",
    )
    print(f"  detected lang={info.language} prob={info.language_probability:.2f} "
          f"dur={info.duration:.0f}s", flush=True)
    with open(txt_out, "w") as ftxt, open(srt_out, "w") as fsrt:
        for i, seg in enumerate(segments, 1):
            ftxt.write(seg.text.strip() + "\n")
            fsrt.write(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n"
                       f"{seg.text.strip()}\n\n")
            if i % 20 == 0:
                el = time.time() - t0
                rate = seg.end / el if el > 0 else 0
                eta = (info.duration - seg.end) / rate if rate > 0 else 0
                print(f"  seg {i:>4}  audio_pos={seg.end:.0f}s  "
                      f"elapsed={el:.0f}s  rate={rate:.1f}x  ETA={eta:.0f}s", flush=True)
    print(f"  DONE {os.path.basename(src)} in {time.time()-t0:.0f}s "
          f"(rate={(info.duration or 0)/(time.time()-t0):.2f}x realtime)", flush=True)


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Loading {MODEL_NAME} model "
          f"(compute_type={COMPUTE_TYPE}, this may download ~3 GB on first run)...",
          flush=True)
    t_load = time.time()
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE,
                           cpu_threads=os.cpu_count() or 4)
    print(f"Model loaded in {time.time()-t_load:.1f}s, CPU threads: {os.cpu_count()}",
          flush=True)

    # Model-tagged outputs so a re-run with a different model never silently
    # reuses old work. Tag is the model name with a safe separator.
    tag = MODEL_NAME.replace("/", "_")
    bv_map = json.load(open(f"{OUT}/bv_map.json"))
    todo = []
    for yt_id, info in bv_map.items():
        src = f"{OUT}/{yt_id}.m4a"
        txt = f"{OUT}/{yt_id}.{tag}.txt"
        srt = f"{OUT}/{yt_id}.{tag}.srt"
        if not os.path.exists(src):
            print(f"MISSING {src}", flush=True); continue
        # Skip only when THIS model's output already exists and is non-empty
        if os.path.exists(txt) and os.path.getsize(txt) > 100:
            print(f"SKIP {yt_id} ({tag} txt already exists at >100 bytes)", flush=True)
            continue
        todo.append((yt_id, info, src, txt, srt))

    print(f"\n{len(todo)} files to transcribe.\n", flush=True)
    t_batch = time.time()
    for yt_id, info, src, txt, srt in todo:
        try:
            transcribe_file(model, src, txt, srt)
        except Exception as e:
            print(f"ERR {yt_id}: {type(e).__name__}: {e}", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] Batch done in {time.time()-t_batch:.0f}s "
          f"({(time.time()-t_batch)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
