# uni 的量化日记 — YouTube channel inventory

Channel: https://www.youtube.com/@uni%E7%9A%84%E9%87%8F%E5%8C%96%E6%97%A5%E8%AE%B0/videos
Fetched: 2026-05-18

| # | Video ID | Title | Duration | Notes |
|---|----------|-------|----------|-------|
| 1 | tGzbK8c3R_A | 波动率预测框架 | 52:50 | Volatility prediction framework (also on Bilibili BV1JvAdzmERy, 19k views there vs 62 on YouTube) |
| 2 | WkZ6TUg_gK4 | 频域分析在金融量化研究中的应用 | 1:45:20 | Frequency-domain analysis in financial quant research |
| 3 | vL8DY2NP96I | 自编码器在金融特征工程中的应用 | 1:14:31 | Autoencoders in financial feature engineering (course module 6) |
| 4 | Ca1n2jgKjrs | 熵在金融特征工程中的应用 | 1:31:00 | Entropy in financial feature engineering (AFML Ch 18) |
| 5 | q922tUTWmCY | 不一样的套利框架 | 1:24:18 | A different arbitrage framework |
| 6 | uVnOeOcoivw | 基于机器学习构建量化模型的全流程拆解 | 1:28:15 | Full ML quant model pipeline (course module 1 / overview) |

**Total duration:** ~7h 56min across all 6 videos.

## Subtitle / script access status

- **yt-dlp** per-video metadata + caption download: blocked — "Sign in to confirm you're not a bot" (IP geolocates to Korea data center)
- **youtube-transcript-api**: blocked — `RequestBlocked` (same IP issue)
- **YouTube direct timedtext API** (`/api/timedtext`): returns empty (now requires signature param)
- **WebFetch / r.jina.ai**: only see stripped page (description not in initial HTML, loaded by JS)
- **oEmbed**: returns title + channel only, no description or transcript

## Paths forward

1. User exports YouTube cookies (Chrome/Firefox extension → `cookies.txt`) → use with `yt-dlp --cookies cookies.txt --write-auto-subs --sub-langs zh-Hans <url>`
2. User opens video, clicks "..." → "Show transcript", copies, pastes into `/home/txy/lb/uni_yt/<video_id>.txt`
3. Audio download + Whisper transcription (audio download also requires cookies due to bot block; ~8 hours of audio is a heavy job)
