import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import streamlit as st

st.set_page_config(page_title="Bamanankan Video Dubber", page_icon="🎬", layout="wide")

NUMBER_WORDS = {0: "wolofila", 1: "kelen", 2: "fila", 3: "saba", 4: "naani", 5: "duuru", 6: "wɔɔrɔ", 7: "wolonwula", 8: "seyiŋ", 9: "kɔnɔntɔn"}
TENS = {10: "tan", 20: "mugan", 30: "bi saba", 40: "bi naani", 50: "bi duuru", 60: "bi wɔɔrɔ", 70: "bi wolonwula", 80: "bi seyiŋ", 90: "bi kɔnɔntɔn"}


def number_to_bambara(value: int) -> str:
    n = int(value)
    if n < 0: return "tɛmɛnen " + number_to_bambara(-n)
    if n < 10: return NUMBER_WORDS[n]
    if n < 20: return "tan" if n == 10 else f"tan ni {NUMBER_WORDS[n - 10]}"
    if n < 100:
        base = TENS[(n // 10) * 10]
        return base if n % 10 == 0 else f"{base} ni {NUMBER_WORDS[n % 10]}"
    if n < 1000:
        base = "kɛmɛ" if n // 100 == 1 else f"kɛmɛ {NUMBER_WORDS[n // 100]}"
        return base if n % 100 == 0 else f"{base} ni {number_to_bambara(n % 100)}"
    if n < 1_000_000:
        base = "waga kelen" if n // 1000 == 1 else f"waga {number_to_bambara(n // 1000)}"
        return base if n % 1000 == 0 else f"{base} ni {number_to_bambara(n % 1000)}"
    if n < 1_000_000_000:
        base = f"miliyɔn {number_to_bambara(n // 1_000_000)}"
        return base if n % 1_000_000 == 0 else f"{base} ni {number_to_bambara(n % 1_000_000)}"
    base = f"miliyari {number_to_bambara(n // 1_000_000_000)}"
    return base if n % 1_000_000_000 == 0 else f"{base} ni {number_to_bambara(n % 1_000_000_000)}"


def numbers_to_bambara(text: str) -> str:
    return re.sub(r"(?<![\w])\d+(?![\w])", lambda m: number_to_bambara(int(m.group())), text)


def chunks(text: str, limit: int = 850) -> List[str]:
    parts = re.split(r"(?<=[.!?؟。\n;:])\s+", text.strip())
    result, current = [], ""
    for part in parts:
        part = part.strip()
        if not part: continue
        if len(current) + len(part) + 1 <= limit:
            current = f"{current} {part}".strip()
        else:
            if current: result.append(current)
            while len(part) > limit:
                result.append(part[:limit]); part = part[limit:]
            current = part
    if current: result.append(current)
    return result or [text[:limit]]


@st.cache_resource(show_spinner=False)
def load_models():
    from whosper import WhosperTranscriber
    from maliba_ai.tts.inference import BambaraTTSInference
    from maliba_ai.config.settings import Speakers
    # Multilingual Whisper is used only for automatic source-language detection
    # and transcription of non-Bambara audio. Whosper remains the preferred
    # recognizer whenever Bambara is detected.
    from faster_whisper import WhisperModel
    multilingual = WhisperModel("small", device="cpu", compute_type="int8")
    return WhosperTranscriber(model_id="MALIBA-AI/bambara-asr-v3"), BambaraTTSInference(), Speakers, multilingual


def extract_audio(video_path: str, wav_path: str) -> float:
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(video_path)
    try:
        if clip.audio is None: raise ValueError("الفيديو لا يحتوي على مسار صوتي")
        clip.audio.write_audiofile(wav_path, fps=24000, nbytes=2, codec="pcm_s16le", logger=None)
        return float(clip.duration)
    finally:
        clip.close()


def _looks_like_bambara(text: str) -> bool:
    lower = text.lower()
    markers = ("bɛ", "ɔ", "ɛ", "ɲ", "ŋ", "wolonwula", "bamanankan", "cogoya", "kɛmɛ", "mugan", "ni ce", "aw ni")
    return sum(marker in lower for marker in markers) >= 2


def transcribe_auto(bambara_asr, multilingual, audio_path: str) -> Tuple[str, str, float]:
    """Detect the source language, then use the best recognizer for it.

    faster-whisper supplies ISO language code and confidence. Bambara is not
    consistently represented in general Whisper checkpoints, so a second
    orthography/lexicon check selects the dedicated Whosper model when needed.
    """
    segments, info = multilingual.transcribe(audio_path, beam_size=1, vad_filter=True)
    general_text = " ".join(segment.text.strip() for segment in segments).strip()
    detected = getattr(info, "language", "unknown") or "unknown"
    confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
    if detected in {"bm", "bambara"} or _looks_like_bambara(general_text):
        result = bambara_asr.transcribe_audio(audio_path)
        if isinstance(result, str): text = result.strip()
        elif isinstance(result, dict): text = str(result.get("text", "")).strip()
        else: text = str(getattr(result, "text", result)).strip()
        return text, "Bambara (bamanankan)", max(confidence, 0.95)
    return general_text, detected, confidence


def translate_to_bambara(text: str, api_key: str, model: str, base_url: Optional[str]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    prompt = ("Translate into natural, correct Bamanankan (Bambara). Preserve names, numbers, "
              "meaning, timing cues and emotion. Use ɛ, ɔ, ɲ and ŋ. Return ONLY the translation.\n\n" + text)
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.15)
    return response.choices[0].message.content.strip()


def generate_tts(tts, speaker, text: str, output_path: str, progress=None) -> None:
    rendered, parts = [], chunks(text)
    for index, part in enumerate(parts):
        clean = ("Read ONLY this Bamanankan text. Use a dignified, steady, literary voice; precise pronunciation; "
                 "clean studio delivery; no humming, crackle, music, or extra words. Preserve wolonwula, cogoya and cogo exactly.\n\n" + part)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp: tmp_path = tmp.name
        try:
            tts.generate_speech(text=clean, speaker_id=speaker, output_filename=tmp_path)
            rendered.append(tmp_path)
        finally:
            if progress: progress((index + 1) / max(len(parts), 1))
    if not rendered: raise ValueError("لم ينتج محرك الصوت أي ملف")
    segments = [sf.read(path) for path in rendered]
    rate = segments[0][1]
    wave = np.concatenate([x[0] if x[0].ndim == 1 else x[0].mean(axis=1) for x in segments])
    sf.write(output_path, wave, rate, subtype="PCM_16")
    for path in rendered: Path(path).unlink(missing_ok=True)


def fit_audio(audio_path: str, target_seconds: float, output_path: str) -> None:
    import librosa
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    current = max(librosa.get_duration(y=y, sr=sr), 0.01)
    rate = min(max(current / max(target_seconds, 0.01), 0.70), 1.35)
    stretched = librosa.effects.time_stretch(y, rate=rate)
    wanted = int(target_seconds * sr)
    sf.write(output_path, np.pad(stretched, (0, max(0, wanted - len(stretched))))[:wanted], sr, subtype="PCM_16")


def mux_video(video_path: str, dub_path: str, output_path: str, keep_original: bool, original_volume: float) -> None:
    from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
    video = VideoFileClip(video_path); dub = AudioFileClip(dub_path).volumex(1.0)
    tracks = [dub]; original = None
    if keep_original and video.audio is not None:
        original = video.audio.volumex(original_volume); tracks.append(original)
    final_audio = CompositeAudioClip(tracks); final = video.set_audio(final_audio)
    try: final.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4, logger=None)
    finally:
        final.close(); final_audio.close(); dub.close(); video.close()
        if original: original.close()


st.title("🎬 دبلجة فيديو احترافية إلى البامبارا")
st.caption("اكتشاف تلقائي للغة المصدر · Bamanankan · أرقام صحيحة · صوت وقور ثابت · إخراج MP4")
with st.sidebar:
    st.header("الإعدادات")
    use_numbers = st.checkbox("تحويل الأرقام إلى كلمات بامبارا", True)
    keep_original = st.checkbox("إبقاء الصوت الأصلي منخفضاً", False)
    original_volume = st.slider("مستوى الصوت الأصلي", 0.0, 0.35, 0.08, 0.01)
    speaker_name = st.text_input("اسم المتحدث (MALIBA)", "Bourama")
    st.info("سيتم اكتشاف اللغة تلقائياً. يستخدم التطبيق Whosper للبامبارا وWhisper للغات الأخرى.")

video_file = st.file_uploader("ارفع الفيديو", type=["mp4", "mov", "mkv", "avi", "webm"])
translation_mode = st.radio("المعالجة", ["ترجمة تلقائية إلى البامبارا", "بامبارا جاهز / بدون ترجمة"], horizontal=True)
api_key = model = base_url = ""
if translation_mode.startswith("ترجمة"):
    api_key = st.text_input("مفتاح OpenAI/DeepSeek/Grok", type="password")
    model = st.text_input("اسم النموذج", "gpt-4o-mini")
    base_url = st.text_input("Base URL (اختياري)", "")

if video_file and st.button("🚀 ابدأ الدبلجة", type="primary", use_container_width=True):
    try:
        asr, tts, speakers, multilingual = load_models()
        speaker = getattr(speakers, speaker_name, getattr(speakers, "Bourama"))
        with tempfile.TemporaryDirectory() as work:
            work = Path(work); source = work / "source.mp4"; source.write_bytes(video_file.getbuffer())
            source_audio = work / "source.wav"
            st.write("1/5 استخراج الصوت…"); duration = extract_audio(str(source), str(source_audio))
            st.write("2/5 اكتشاف اللغة والتفريغ…")
            original, language, confidence = transcribe_auto(asr, multilingual, str(source_audio))
            if not original: raise ValueError("لم يتم العثور على كلام في الفيديو")
            st.success(f"اللغة المكتشفة: {language} — الثقة: {confidence:.0%}")
            st.text_area("النص المستخرج", original, height=140)
            text = original
            if translation_mode.startswith("ترجمة"):
                if not api_key: raise ValueError("أدخل مفتاح الترجمة أو اختر بامبارا جاهز")
                st.write("3/5 ترجمة دقيقة إلى البامبارا…"); text = translate_to_bambara(original, api_key, model, base_url)
            if use_numbers: text = numbers_to_bambara(text)
            st.text_area("النص النهائي للدبلجة", text, height=180)
            st.write("4/5 توليد صوت بامبارا نظيف…")
            dub_raw = work / "dub_raw.wav"; bar = st.progress(0)
            generate_tts(tts, speaker, text, str(dub_raw), lambda p: bar.progress(int(p * 100)))
            fitted = work / "dub_fitted.wav"; fit_audio(str(dub_raw), duration, str(fitted))
            st.write("5/5 مزج الصوت وإنتاج الفيديو…")
            output = work / "bamanankan_dub.mp4"; mux_video(str(source), str(fitted), str(output), keep_original, original_volume)
            st.success("اكتملت الدبلجة بنجاح"); st.video(str(output))
            st.download_button("⬇️ تنزيل الفيديو المدبلج", output.read_bytes(), "bamanankan_dub.mp4", "video/mp4")
    except Exception as exc:
        st.error(f"تعذر إكمال الدبلجة: {exc}")
else:
    st.info("ارفع فيديو للبدء. ثبّت FFmpeg على النظام قبل التشغيل.")
