#!/usr/bin/env python3
"""Download audio-only m4a from Bilibili using DASH playurl API."""
import json, urllib.request, sys, os, time

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
OUT = "/home/txy/lb/uni_yt"

def fetch_json(url, referer):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer})
    return json.load(urllib.request.urlopen(req, timeout=15))

def best_audio_url(bv):
    v = fetch_json(f"https://api.bilibili.com/x/web-interface/view?bvid={bv}",
                   f"https://www.bilibili.com/video/{bv}/")
    cid = v["data"]["cid"]
    p = fetch_json(
        f"https://api.bilibili.com/x/player/playurl?bvid={bv}&cid={cid}&fnval=4048&fnver=0&fourk=1",
        f"https://www.bilibili.com/video/{bv}/")
    audios = p["data"]["dash"]["audio"]
    # pick highest bandwidth
    a = max(audios, key=lambda x: x.get("bandwidth", 0))
    return a["baseUrl"], cid

def download(url, dest, referer):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer})
    with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0; last = time.time()
        while True:
            chunk = r.read(1 << 16)
            if not chunk: break
            f.write(chunk); got += len(chunk)
            if time.time() - last > 2:
                pct = 100*got/total if total else 0
                print(f"  {os.path.basename(dest)}: {got/1e6:.1f} MB ({pct:.0f}%)", flush=True)
                last = time.time()
    return got

def main():
    bv_map = json.load(open(f"{OUT}/bv_map.json"))
    for yt_id, info in bv_map.items():
        bv = info["bv"]; title = info["title"]
        dest = f"{OUT}/{yt_id}.m4a"
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"SKIP {yt_id} ({title}) — already downloaded")
            continue
        print(f"\n=== {yt_id} | {bv} | {title} ===", flush=True)
        try:
            url, cid = best_audio_url(bv)
            print(f"  cid={cid}  audio={url[:80]}...", flush=True)
            size = download(url, dest, f"https://www.bilibili.com/video/{bv}/")
            print(f"  DONE: {size/1e6:.1f} MB", flush=True)
        except Exception as e:
            print(f"  ERR: {type(e).__name__}: {e}", flush=True)

if __name__ == "__main__":
    main()
