#!/bin/bash

# Bamanankan Video Dubber Pro — setup.sh
# سكريبت التشغيل السريع للـ Linux/macOS

echo "🎬 Bamanankan Video Dubber Pro — Setup"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 غير مثبت. الرجاء تثبيت Python 3.8+"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg غير مثبت."
    echo "macOS: brew install ffmpeg"
    echo "Linux: sudo apt install ffmpeg"
    exit 1
fi

echo "✅ Python و FFmpeg مثبتان"
echo ""

echo "📦 إنشاء البيئة الافتراضية…"
python3 -m venv .venv
source .venv/bin/activate

echo "📦 تثبيت المتطلبات…"
pip install --upgrade pip
pip install -r requirements.txt

echo "📦 تثبيت Whosper…"
pip install git+https://github.com/sudoping01/whosper.git

echo "📦 تثبيت MALIBA-AI…"
pip install maliba-ai

echo ""
echo "✅ اكتمل الإعداد بنجاح!"
echo ""
echo "لتشغيل التطبيق:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
echo ""
