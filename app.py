import json
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Bamanankan Video Dubber Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

NUMBER_WORDS = {
    0: "wolofila", 1: "kelen", 2: "fila", 3: "saba", 4: "naani",
    5: "duuru", 6: "wɔɔrɔ", 7: "wolonwula", 8: "seyiŋ", 9: "kɔnɔntɔn",
}
TENS = {
    10: "tan", 20: "mugan", 30: "bi saba", 40: "bi naani", 50: "bi duuru",
    60: "bi wɔɔrɔ", 70: "bi wolonwula", 80: "bi seyiŋ", 90: "bi kɔnɔntɔn",
}


def number_to_bambara(value: int) -> str:
    """تحويل رقم إلى كلمات بامبارا"""
    n = int(value)
    if n < 0:
        return "tɛmɛnen " + number_to_bambara(-n)
    if n < 10:
        return NUMBER_WORDS[n]
    if n < 20:
        return "tan" if n == 10 else f"tan ni {NUMBER_WORDS[n - 10]}"
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
    """تحويل جميع الأرقام في النص إلى كلمات بامبارا"""
    return re.sub(r"(?<!\w)\d+(?!\w)", lambda m: number_to_bambara(int(m.group())), text)


def chunks(text: str, limit: int = 850) -> List[str]:
    """تقسيم النص إلى مقاطع للمعالجة"""
    parts = re.split(r"(?<=[.!?؟。\n;:])\s+", text.strip())
    result, current = [], ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(current) + len(part) + 1 <= limit:
            current = f"{current} {part}".strip()
        else:
            if current:
                result.append(current)
            while len(part) > limit:
                result.append(part[:limit])
                part = part[limit:]
            current = part
    if current:
        result.append(current)
    return result or [text[:limit]]


@st.cache_resource(show_spinner=False)
def load_models():
    """تحميل نماذج ASR و TTS"""
    try:
        from whosper import WhosperTranscriber
    except Exception:
        WhosperTranscriber = None

    from maliba_ai.tts.inference import BambaraTTSInference
    from maliba_ai.config.settings import Speakers
    from faster_whisper import WhisperModel

    multilingual = WhisperModel("small", device="cpu", compute_type="int8")
    return WhosperTranscriber, BambaraTTSInference(), Speakers, multilingual


def extract_audio(video_path: str, wav_path: str) -> float:
    """استخراج الصوت من الفيديو"""
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(video_path)
    try:
        if clip.audio is None:
            raise ValueError("الفيديو لا يحتوي على مسار صوتي")
        clip.audio.write_audiofile(wav_path, fps=24000, nbytes=2, codec="pcm_s16le", logger=None)
        return float(clip.duration)
    finally:
        clip.close()


def looks_like_bambara(text: str) -> bool:
    """التحقق من ما إذا كان النص بامبارا"""
    lower = text.lower()
    markers = (
        "bɛ", "ɔ", "ɛ", "ɲ", "ŋ", "bamanankan", "wolonwula", "cogoya", "kɛmɛ",
        "mugan", "aw ni ce", "ni ce", "bari", "saba", "duuru", "waga",
    )
    return sum(marker in lower for marker in markers) >= 2


def transcribe_auto(bambara_asr, multilingual, audio_path: str) -> Tuple[str, str, float]:
    """اكتشاف اللغة والتفريغ التلقائي"""
    segments, info = multilingual.transcribe(audio_path, beam_size=1, vad_filter=True)
    general_text = " ".join(segment.text.strip() for segment in segments).strip()
    detected = (getattr(info, "language", "unknown") or "unknown").lower()
    confidence = float(getattr(info, "language_probability", 0.0) or 0.0)

    if bambara_asr is not None:
        try:
            result = bambara_asr.transcribe_audio(audio_path)
            if isinstance(result, str):
                text = result.strip()
            elif isinstance(result, dict):
                text = str(result.get("text", "")).strip()
            else:
                text = str(getattr(result, "text", result)).strip()

            if text and (detected in {"bm", "bambara"} or looks_like_bambara(text) or looks_like_bambara(general_text)):
                return text, "Bambara (bamanankan)", max(confidence, 0.95)
        except Exception:
            pass

    lang_names = {"ar": "العربية", "en": "English", "fr": "Français", "bm": "Bambara", "unknown": "Unknown"}
    return general_text, lang_names.get(detected, detected), confidence


def translate_to_bambara(text: str, api_key: str, model: str, base_url: Optional[str]) -> str:
    """ترجمة النص إلى البامبارا باستخدام LLM"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    prompt = (
        "Translate the following text into clear, natural Bamanankan (Bambara). "
        "Keep names, numbers, time references, emotional tone, and cultural meaning. "
        "Use the correct Bambara orthography with ɛ, ɔ, ɲ and ŋ. Return only the translation.\n\n"
        + text
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
    )
    return response.choices[0].message.content.strip()


def generate_tts(tts, speaker, text: str, output_path: str, progress=None) -> None:
    """توليد الصوت من النص"""
    rendered = []
    parts = chunks(text)
    for index, part in enumerate(parts):
        clean = (
            "Read ONLY this Bamanankan text. Use a dignified, steady, literary voice; "
            "precise pronunciation; clear studio delivery; no humming, crackle, music, "
            "or extra words. Preserve wolonwula, cogoya and cogo exactly.\n\n" + part
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            tts.generate_speech(text=clean, speaker_id=speaker, output_filename=tmp_path)
            rendered.append(tmp_path)
        finally:
            if progress:
                progress((index + 1) / max(len(parts), 1))

    if not rendered:
        raise ValueError("لم ينتج محرك الصوت أي ملف")

    segments = [sf.read(path) for path in rendered]
    sample_rate = segments[0][1]
    wave = np.concatenate([seg[0] if seg[0].ndim == 1 else seg[0].mean(axis=1) for seg in segments])
    sf.write(output_path, wave, sample_rate, subtype="PCM_16")
    for path in rendered:
        Path(path).unlink(missing_ok=True)


def fit_audio(audio_path: str, target_seconds: float, output_path: str) -> None:
    """مزامنة مدة الصوت مع الفيديو"""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = max(librosa.get_duration(y=y, sr=sr), 0.01)
    rate = duration / max(target_seconds, 0.01)
    rate = min(max(rate, 0.70), 1.35)
    stretched = librosa.effects.time_stretch(y, rate=rate)
    wanted = int(target_seconds * sr)
    out = np.pad(stretched, (0, max(0, wanted - len(stretched))))[:wanted]
    sf.write(output_path, out, sr, subtype="PCM_16")


def mux_video(video_path: str, dub_path: str, output_path: str, keep_original: bool, original_volume: float) -> None:
    """دمج الفيديو والصوت"""
    from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip

    video = VideoFileClip(video_path)
    dub = AudioFileClip(dub_path).volumex(1.0)
    tracks = [dub]
    original = None
    if keep_original and video.audio is not None:
        original = video.audio.volumex(original_volume)
        tracks.append(original)
    final_audio = CompositeAudioClip(tracks)
    final = video.set_audio(final_audio)
    try:
        final.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4, logger=None)
    finally:
        final.close()
        final_audio.close()
        dub.close()
        video.close()
        if original:
            original.close()


st.title("🎬 Bamanankan Video Dubber Pro")
st.caption("نظام دبلجة فيديو احترافي • اكتشاف لغة تلقائي • ترجمة ذكية • صوت نظيف • إخراج MP4")

with st.sidebar:
    st.header("⚙️ الإعدادات الاحترافية")
    st.subheader("الإخراج")
    use_numbers = st.checkbox("تحويل الأرقام إلى كلمات بامبارا", True)
    keep_original = st.checkbox("إبقاء الصوت الأصلي منخفضاً", False)
    original_volume = st.slider("مستوى الصوت الأصلي", 0.0, 0.35, 0.08, 0.01)
    speaker_name = st.text_input("اسم المتحدث", "Bourama")

    st.divider()
    st.subheader("نوع المعالجة")
    language_mode = st.selectbox(
        "الخيار",
        [
            "اكتشاف تلقائي + ترجمة",
            "ترجمة مباشرة",
            "بامبارا جاهز (بدون ترجمة)",
        ],
    )

    st.divider()
    st.subheader("مفتاح الترجمة (اختياري)")
    api_key = st.text_input("مفتاح OpenAI / DeepSeek / Grok", type="password")
    model_name = st.text_input("اسم النموذج", "gpt-4o-mini")
    custom_base_url = st.text_input("Base URL", value="")

    st.divider()
    st.caption(
        "🔧 الإعدادات الحالية:\n"
        "• Whosper للبامبارا\n"
        "• Whisper متعدد اللغات\n"
        "• MALIBA TTS\n"
        "• Librosa للمعالجة الصوتية"
    )

uploaded_file = st.file_uploader("رفع الفيديو", type=["mp4", "mov", "mkv", "avi", "webm"])

if uploaded_file:
    st.video(uploaded_file)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("حجم الملف", f"{len(uploaded_file.getvalue()) / 1024 / 1024:.2f} MB")
    with col2:
        st.metric("اللغة", "Auto detect")
    with col3:
        st.metric("الإصدار", "Pro")

    if st.button("🚀 بدء الدبلجة الاحترافية", type="primary", use_container_width=True):
        try:
            asr, tts, speakers, multilingual = load_models()
            speaker = getattr(speakers, speaker_name, None)
            if speaker is None:
                speaker = getattr(speakers, "Bourama", None)

            if speaker is None:
                raise RuntimeError("لم يتم العثور على صوت متاح. تأكد من تثبيت maliba-ai بشكل صحيح.")

            with tempfile.TemporaryDirectory() as work:
                work = Path(work)
                source = work / "source_video.mp4"
                source.write_bytes(uploaded_file.getbuffer())
                source_audio = work / "source_audio.wav"

                progress = st.progress(0, text="1/6 استخراج الصوت…")
                duration = extract_audio(str(source), str(source_audio))
                progress.progress(17, text="2/6 اكتشاف اللغة والتفريغ…")

                source_text, lang_name, confidence = transcribe_auto(asr, multilingual, str(source_audio))

                if not source_text.strip():
                    raise ValueError("لم يتم العثور على كلام في الفيديو")

                st.success(f"✅ اللغة المكتشفة: {lang_name} • الثقة: {confidence:.0%}")

                text = source_text
                if language_mode in ["اكتشاف تلقائي + ترجمة", "ترجمة مباشرة"]:
                    if language_mode == "ترجمة مباشرة" or lang_name != "Bambara (bamanankan)":
                        if not api_key:
                            raise ValueError("أدخل مفتاح الترجمة أو اختر 'بامبارا جاهز'.")
                        progress.progress(35, text="3/6 الترجمة إلى البامبارا…")
                        text = translate_to_bambara(source_text, api_key, model_name, custom_base_url)

                if use_numbers:
                    text = numbers_to_bambara(text)

                st.subheader("📝 النص المستخرج")
                st.text_area("المصدر", source_text, height=120, disabled=True)

                st.subheader("🎯 النص النهائي للدبلجة")
                final_text = st.text_area("البامبارا", text, height=150)

                progress.progress(55, text="4/6 توليد الصوت…")
                dub_raw = work / "dub_raw.wav"
                generate_tts(tts, speaker, final_text, str(dub_raw), lambda p: progress.progress(int(55 + (p * 25)), text=f"4/6 توليد الصوت… {int(p * 100)}%"))

                progress.progress(80, text="5/6 مزامنة الصوت…")
                dub_fit = work / "dub_fit.wav"
                fit_audio(str(dub_raw), duration, str(dub_fit))

                progress.progress(90, text="6/6 دمج الفيديو…")
                output_file = work / "final_dub.mp4"
                mux_video(str(source), str(dub_fit), str(output_file), keep_original, original_volume)

                st.success("✅ اكتمل الإنتاج بنجاح!")
                st.video(str(output_file))

                with open(output_file, "rb") as f:
                    st.download_button(
                        "⬇️ تنزيل الفيديو المدبلج",
                        f.read(),
                        "bamanankan_dub_pro.mp4",
                        "video/mp4",
                    )

        except Exception as exc:
            st.error(f"❌ خطأ: {exc}")
else:
    st.info("ارفع فيديو لبدء المعالجة. تأكد من تثبيت FFmpeg.")

st.markdown("---")
with st.expander("📖 تعليمات التشغيل"):
    st.code(
        """
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install git+https://github.com/sudoping01/whosper.git
pip install maliba-ai
streamlit run app.py
""".strip()
    )
    st.caption("مطلوب: FFmpeg في PATH")

if __name__ == "__main__":
    pass
