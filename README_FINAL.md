# Bamanankan Video Dubber Pro — الإصدار النهائي

تطبيق احترافي لدبلجة الفيديو إلى اللغة البامبارا (Bamanankan) بميزات متقدمة:

## الميزات

- ✅ **اكتشاف اللغة تلقائياً** — يتعرف على لغة الفيديو الأصلية
- ✅ **ترجمة ذكية** — ترجمة احترافية إلى البامبارا عبر OpenAI/DeepSeek/Grok
- ✅ **ASR متخصص** — Whosper للبامبارا + Whisper متعدد اللغات
- ✅ **توليد صوت** — MALIBA TTS مع اختيار المتحدث
- ✅ **تحويل الأرقام** — تحويل تلقائي للأرقام إلى كلمات بامبارا
- ✅ **مزج احترافي** — دمج الصوت الجديد مع الفيديو أو الصوت الأصلي
- ✅ **إخراج MP4** — فيديو نهائي جاهز للتنزيل

## المتطلبات

- Python 3.8+
- FFmpeg (مثبت في PATH)
- pip

## التشغيل

### Windows

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pip install git+https://github.com/sudoping01/whosper.git
pip install maliba-ai
streamlit run app.py
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/sudoping01/whosper.git
pip install maliba-ai
streamlit run app.py
```

بعد التشغيل، افتح:
```
http://localhost:8501
```

## الإعدادات الاحترافية

- **اللغة**: اكتشاف تلقائي + ترجمة، ترجمة مباشرة، أو بامبارا جاهز
- **المتحدث**: Bourama (افتراضي)، Adama، Moussa، وغيرهم
- **الصوت الأصلي**: إبقاء الصوت الأصلي منخفضاً أو حذفه كلياً
- **الأرقام**: تحويل تلقائي إلى كلمات بامبارا
- **نموذج الترجمة**: gpt-4o-mini (افتراضي) أو أي نموذج OpenAI متوافق

## المشاريع المستخدمة

- [Whosper](https://github.com/sudoping01/whosper) — ASR للبامبارا
- [MALIBA-AI](https://huggingface.co/MALIBA-AI) — TTS متعدد اللغات الأفريقية
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper متعدد اللغات
- [Streamlit](https://streamlit.io/) — الواجهة الرسومية

## التخصيص

### تغيير صوت المتحدث

في الشريط الجانبي، أدخل اسم المتحدث من MALIBA:
- Bourama (الافتراضي)
- Adama
- Moussa
- Modibo
- Seydou
- Amadou
- Bakary
- Ngolo
- Ibrahima
- Amara

### تغيير نموذج الترجمة

في الإعدادات، عدّل اسم النموذج:
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- DeepSeek: `deepseek-chat`
- Grok: `grok-2`

## استكشاف الأخطاء

### خطأ: "FFmpeg غير مثبت"
```bash
# Windows (Chocolatey)
choco install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Linux (apt)
sudo apt install ffmpeg
```

### خطأ: "لا توجد لغة بامبارا"
تأكد من تثبيت Whosper:
```bash
pip install git+https://github.com/sudoping01/whosper.git
```

### خطأ: "لا يوجد متحدث متاح"
تأكد من تثبيت maliba-ai:
```bash
pip install maliba-ai
```

## الترخيص

هذا المشروع يستخدم مشاريع مفتوحة المصدر. تحقق من رخص المشاريع المستخدمة قبل الاستخدام التجاري.

## الدعم

إذا واجهت مشاكل، تحقق من:
1. تثبيت Python 3.8+
2. تثبيت FFmpeg
3. تثبيت جميع المتطلبات من requirements.txt
4. تثبيت Whosper و maliba-ai

---

**الإصدار**: 1.0.0 Pro  
**آخر تحديث**: 2026-09-25
