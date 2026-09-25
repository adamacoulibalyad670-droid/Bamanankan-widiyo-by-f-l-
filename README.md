Bamanankan widiyo by-fɔlɔ 
Bmtss fɔlɔ -a

pip install fastapi uvicorn google-genai pydub python-multipart
# ffmpeg مطلوب للدمج النظيف
python main.py

# main.py — Bamanankan TTS (ملف واحد)
# pip install fastapi uvicorn google-genai pydub python-multipart
# ثم: python main.py

import os, re, io, uuid, base64, json, asyncio
from pathlib import Path
from typing import Optional, List, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
KEY_FILE = ROOT / ".gemini_key"
OUT.mkdir(exist_ok=True)

# ——— أرقام بامبارا ———
U = {0:"wolofila",1:"kelen",2:"fila",3:"saba",4:"naani",5:"duuru",6:"wɔɔrɔ",7:"wolonwula",8:"seyin",9:"kɔnɔntɔn"}
T10 = {10:"tan",11:"tan ni kelen",12:"tan ni fila",13:"tan ni saba",14:"tan ni naani",
       15:"tan ni duuru",16:"tan ni wɔɔrɔ",17:"tan ni wolonwula",18:"tan ni seyin",19:"tan ni kɔnɔntɔn"}
T20 = {20:"mugan",30:"bi saba",40:"bi naani",50:"bi duuru",60:"bi wɔɔrɔ",70:"bi wolonwula",80:"bi seyin",90:"bi kɔnɔntɔn"}

def n2bm(n: int) -> str:
    n = int(n)
    if n < 0: return "tɛmɛnen " + n2bm(-n)
    if n < 10: return U[n]
    if n < 20: return T10[n]
    if n < 100:
        t, r = (n//10)*10, n%10
        b = T20.get(t, f"bi {U[t//10]}")
        return b if r == 0 else f"{b} ni {U[r]}"
    if n < 1000:
        h, r = n//100, n%100
        b = "kɛmɛ" if h == 1 else f"kɛmɛ {U[h]}"
        return b if r == 0 else f"{b} ni {n2bm(r)}"
    if n < 10**6:
        th, r = n//1000, n%1000
        b = "waga kelen" if th == 1 else f"waga {n2bm(th)}"
        return b if r == 0 else f"{b} ni {n2bm(r)}"
    m, r = n//10**6, n%10**6
    b = f"miliyɔn {n2bm(m)}"
    return b if r == 0 else f"{b} ni {n2bm(r)}"

def nums_to_bm(t: str) -> str:
    return re.sub(r"\b\d+\b", lambda m: n2bm(int(m.group())), t)

VOICES = ["Kore","Orus","Alnilam","Schedar","Gacrux","Sulafat","Charon","Iapetus",
          "Zephyr","Puck","Aoede","Leda","Fenrir","Achird","Algieba","Despina"]
STYLE = ("Calm, dignified steady Bambara narration. Warm authoritative voice. "
         "Clear precise pronunciation, even pacing. Clean studio, no noise no crackle.")
MODELS = ["gemini-2.5-flash-preview-tts","gemini-2.5-pro-preview-tts","gemini-2.5-flash-tts"]
CHUNK = 1000
jobs = {}

def load_key() -> str:
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k: return k
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    return ""

def save_key(k: str):
    KEY_FILE.write_text(k.strip(), encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = k.strip()

def client():
    k = load_key()
    if not k: raise RuntimeError("لا يوجد مفتاح — أدخله من الواجهة")
    if not genai: raise RuntimeError("ثبّت: pip install google-genai")
    return genai.Client(api_key=k)

def split_text(t: str) -> List[str]:
    parts = re.split(r"(?<=[.!?؟。\n;:])\s+", t.strip())
    out, cur = [], ""
    for s in parts:
        s = s.strip()
        if not s: continue
        if len(cur)+len(s)+1 <= CHUNK:
            cur = (cur+" "+s).strip()
        else:
            if cur: out.append(cur)
            if len(s) > CHUNK:
                for i in range(0, len(s), CHUNK):
                    out.append(s[i:i+CHUNK])
                cur = ""
            else:
                cur = s
    if cur: out.append(cur)
    return out or [t[:CHUNK]]

def extract_audio(resp) -> bytes:
    for p in resp.candidates[0].content.parts:
        d = getattr(getattr(p, "inline_data", None), "data", None)
        if d:
            return base64.b64decode(d) if isinstance(d, str) else d
    raise RuntimeError("لا بيانات صوت")

def to_seg(raw: bytes):
    if not AudioSegment:
        raise RuntimeError("ثبّت: pip install pydub + ffmpeg")
    bio = io.BytesIO(raw)
    try:
        seg = AudioSegment.from_file(bio, format="wav")
    except Exception:
        bio.seek(0)
        try:
            seg = AudioSegment.from_file(bio)
        except Exception:
            seg = AudioSegment.from_raw(io.BytesIO(raw), sample_width=2, frame_rate=24000, channels=1)
    try:
        seg = seg.strip_silence(silence_len=80, silence_thresh=-42, padding=40)
    except Exception:
        pass
    return seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)

def synth_one(text: str, voice: str, style: str) -> bytes:
    c = client()
    prompt = f"{style or STYLE}\n\nRead this Bambara clearly and steadily. No extra words.\n\n{text}"
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice if voice in VOICES else "Kore")
            )
        ),
    )
    err = None
    for m in MODELS:
        try:
            r = c.models.generate_content(model=m, contents=prompt, config=cfg)
            return extract_audio(r)
        except Exception as e:
            err = e
    raise RuntimeError(str(err))

def synthesize(text: str, voice="Kore", style=STYLE, gap_ms=220, read_numbers=True, progress=None) -> Tuple[bytes, list, int]:
    logs = []
    text = text.strip()
    if not text: raise ValueError("نص فارغ")
    if read_numbers:
        text = nums_to_bm(text)
        logs.append("✔ أرقام → بامبارا")
    chunks = split_text(text)
    logs.append(f"مقاطع: {len(chunks)}")
    segs = []
    for i, ch in enumerate(chunks):
        logs.append(f"▶ {i+1}/{len(chunks)}")
        if progress: progress(int(i/max(len(chunks),1)*90))
        segs.append(to_seg(synth_one(ch, voice, style)))
    gap = AudioSegment.silent(duration=max(0, gap_ms), frame_rate=24000)
    comb = segs[0]
    for s in segs[1:]:
        comb = comb + gap + s
    buf = io.BytesIO()
    comb.export(buf, format="wav", parameters=["-ac","1","-ar","24000"])
    if progress: progress(100)
    logs.append(f"✅ {len(comb)/1000:.1f}ث")
    return buf.getvalue(), logs, len(segs)

# ——— API ———
app = FastAPI()

class TTSIn(BaseModel):
    text: str
    voice: str = "Kore"
    style: str = STYLE
    read_numbers: bool = True
    gap_ms: int = 220
    voice_ref: Optional[str] = None

class KeyIn(BaseModel):
    key: str

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/api/health")
def health():
    k = bool(load_key())
    return {"api_key_present": k, "engine_ready": k and genai is not None}

@app.post("/api/key")
def set_key(body: KeyIn):
    if len(body.key.strip()) < 10:
        return {"ok": False, "error": "مفتاح غير صالح"}
    save_key(body.key)
    return {"ok": True}

@app.get("/api/voices")
def voices():
    rec = ["Kore","Orus","Alnilam","Schedar","Gacrux","Sulafat"]
    return {"voices": [{"name":v,"style":"","recommended":v in rec} for v in VOICES], "recommended": rec}

@app.get("/api/number")
def number(n: int):
    return {"n": n, "bambara": n2bm(n)}

@app.post("/api/tts")
def tts(body: TTSIn):
    try:
        data, logs, parts = synthesize(body.text, body.voice, body.style or STYLE, body.gap_ms, body.read_numbers)
        fn = f"t_{uuid.uuid4().hex[:10]}.wav"
        (OUT/fn).write_bytes(data)
        return {"ok": True, "file": f"/out/{fn}", "bytes": len(data), "parts": parts, "voice": body.voice, "log": logs}
    except Exception as e:
        return {"ok": False, "error": str(e), "log": [str(e)]}

@app.post("/api/tts-long")
async def tts_long(body: TTSIn):
    jid = uuid.uuid4().hex[:10]
    jobs[jid] = {"status":"running","percent":0,"log":[],"file":None,"bytes":0,"error":None,"total":0}

    def run():
        try:
            def prog(p): jobs[jid]["percent"] = p
            data, logs, parts = synthesize(body.text, body.voice, body.style or STYLE, body.gap_ms, body.read_numbers, prog)
            fn = f"L_{jid}.wav"
            (OUT/fn).write_bytes(data)
            jobs[jid].update(status="done", percent=100, log=logs, file=f"/out/{fn}", bytes=len(data), total=parts)
        except Exception as e:
            jobs[jid].update(status="failed", error=str(e), log=[str(e)])

    asyncio.get_event_loop().run_in_executor(None, run)
    return {"ok": True, "job": jid}

@app.get("/api/job/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if not j: return {"ok": False}
    return {"ok": True, "job": j, "percent": j["percent"], "log": j.get("log", [])}

@app.post("/api/clone")
async def clone(source_audio: UploadFile = File(...), consent_audio: UploadFile = File(...),
                display_name: str = Form("Bambara"), store: str = Form("true")):
    # حفظ العينات محلياً — التوليد يبقى بالصوت الجاهز + أسلوب وقور
    sid = uuid.uuid4().hex[:8]
    (OUT/f"src_{sid}.bin").write_bytes(await source_audio.read())
    (OUT/f"con_{sid}.bin").write_bytes(await consent_audio.read())
    ref = f"voice_{sid}"
    return {"ok": True, "voice_ref": ref, "log": ["✔ حُفظت البصمة. استخدم صوتاً جاهزاً قوياً (Kore/Orus) حتى يتوفر Instant Voice في حسابك."]}

@app.get("/api/myvoices")
def myvoices():
    return {"ok": True, "voices": []}

@app.post("/api/design")
def design(body: dict):
    ref = f"voice_{uuid.uuid4().hex[:8]}"
    return {"ok": True, "voice_ref": ref, "preview": None, "log": ["✔ وُصف الصوت. طبّق الأسلوب في خانة الأسلوب."]}

@app.get("/out/{name}")
def out_file(name: str):
    p = OUT / name
    if not p.exists(): return HTMLResponse("404", 404)
    return FileResponse(p, media_type="audio/wav")

# ——— واجهة مصغّرة ———
HTML = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bamanankan TTS</title>
<style>
:root{--bg:#0b1020;--card:#161f3d;--line:#26325c;--t:#e8edff;--m:#8fa0cc;--a:#4f8cff;--ok:#22c55e;--err:#ef4444}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:radial-gradient(900px 400px at 50% -10%,#1b2a55,var(--bg));color:var(--t);min-height:100vh;padding:16px;line-height:1.6}
.w{max-width:720px;margin:auto}
h1{text-align:center;font-size:1.5rem;margin-bottom:4px}h1 span{color:var(--a)}
.sub{text-align:center;color:var(--m);font-size:.85rem;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin-bottom:12px}
.card h2{font-size:.95rem;margin-bottom:10px;border-right:3px solid var(--a);padding-right:8px}
textarea,input,select{width:100%;background:#121a33;border:1px solid var(--line);border-radius:10px;color:var(--t);padding:10px;font:inherit;margin-top:4px}
textarea{min-height:120px;resize:vertical}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media(max-width:560px){.row{grid-template-columns:1fr}}
label{font-size:.78rem;color:var(--m)}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
button{background:var(--a);color:#fff;border:0;border-radius:10px;padding:10px 16px;font-weight:700;cursor:pointer;font:inherit}
button.g{background:transparent;border:1px solid var(--line);color:var(--t)}
button:disabled{opacity:.5}
.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.72rem;border:1px solid var(--line);color:var(--m);margin:0 4px}
.pill.on{border-color:var(--ok);color:var(--ok)}.pill.off{border-color:var(--err);color:var(--err)}
#log{background:#080d1c;border:1px solid var(--line);border-radius:8px;padding:10px;font:12px monospace;color:#9fb3e0;max-height:160px;overflow:auto;direction:ltr;text-align:left;white-space:pre-wrap}
.bar{height:6px;background:#121a33;border-radius:4px;margin-top:8px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--a),var(--ok));transition:.3s}
audio{width:100%;margin-top:10px}
.stats{font-size:.8rem;color:var(--m);margin-top:6px}.stats b{color:var(--t)}
.chk{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:.85rem}
.chk input{width:auto}
</style></head><body><div class="w">
<header>
<h1>🎙️ Bamanankan <span>TTS</span></h1>
<div class="sub">نص + أرقام بالبامبارا · صوت وقور نقي · نصوص طويلة</div>
<div style="text-align:center">
<span class="pill" id="kp">المفتاح…</span>
<span class="pill" id="ep">المحرّك…</span>
</div>
</header>

<div class="card">
<h2>مفتاح Gemini (مرة واحدة — يُحفظ محلياً)</h2>
<div class="row">
<input type="password" id="key" placeholder="الصق مفتاح Google AI Studio هنا">
<button id="btnKey">💾 حفظ المفتاح</button>
</div>
</div>

<div class="card">
<h2>١. النص</h2>
<textarea id="text">I ni ce! N tɔgɔ ye Awa ye.
N yɛrɛ bɛ san 2026. A ye 1250 fɔ.
Hɛrɛ sira? Hɛrɛ dɔrɔn.</textarea>
<div class="stats">حروف: <b id="cc">0</b> · كلمات: <b id="wc">0</b> · مقاطع: <b id="ch">0</b></div>
<div class="chk"><input type="checkbox" id="rn" checked><label for="rn">أرقام بامبارا (7→wolonwula)</label></div>
</div>

<div class="card">
<h2>٢. الصوت</h2>
<div class="row">
<div><label>صوت</label><select id="voice"></select></div>
<div><label>وقفة ms</label><input type="number" id="gap" value="220" min="0" max="1000" step="20"></div>
</div>
<label style="margin-top:8px;display:block">أسلوب</label>
<textarea id="style" style="min-height:70px"></textarea>
</div>

<div class="card">
<h2>٣. توليد</h2>
<div class="btns">
<button id="go">▶️ ولّد</button>
<button class="g" id="gol">🧵 نص طويل</button>
<button class="g" id="stop">■ إيقاف</button>
</div>
<div class="bar"><i id="bar"></i></div>
<audio id="player" controls style="display:none"></audio>
<div id="log" style="margin-top:8px"></div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
const log=s=>{$("log").textContent+=s+"\n";$("log").scrollTop=99999};
const bar=p=>$("bar").style.width=Math.min(100,p)+"%";
let JOB=null,POLL=null;

async function health(){
  try{
    const d=await(await fetch("/api/health")).json();
    $("kp").textContent=d.api_key_present?"✔ المفتاح محفوظ":"✖ أدخل المفتاح";
    $("kp").className="pill "+(d.api_key_present?"on":"off");
    $("ep").textContent=d.engine_ready?"✔ جاهز":"⏳";
    $("ep").className="pill "+(d.engine_ready?"on":"");
  }catch(e){log("⚠️ "+e)}
}
async function loadVoices(){
  const d=await(await fetch("/api/voices")).json();
  $("voice").innerHTML=d.voices.map(v=>`<option value="${v.name}">${v.recommended?"⭐ ":""}${v.name}</option>`).join("");
  $("voice").value="Kore";
}
function count(){
  const t=$("text").value||"";
  $("cc").textContent=t.length;
  $("wc").textContent=t.trim()?t.trim().split(/\s+/).length:0;
  $("ch").textContent=t.length<=1000?(t.trim()?1:0):Math.ceil(t.length/1000);
}
$("text").oninput=count;

$("btnKey").onclick=async()=>{
  const k=$("key").value.trim();
  if(!k)return log("❌ الصق المفتاح");
  const d=await(await fetch("/api/key",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key:k})})).json();
  log(d.ok?"✔ المفتاح حُفظ محلياً":("❌ "+d.error));
  $("key").value="";
  health();
};

async function synth(long){
  const text=$("text").value.trim();
  if(!text)return log("❌ اكتب نصاً");
  const body={text,voice:$("voice").value,style:$("style").value,read_numbers:$("rn").checked,gap_ms:+$("gap").value||220};
  $("go").disabled=$("gol").disabled=true;bar(10);log(long?"🧵 مهمة طويلة…":"▶️ توليد…");
  try{
    if(!long){
      const d=await(await fetch("/api/tts",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
      (d.log||[]).forEach(log);
      if(!d.ok){log("❌ "+d.error);bar(0);return}
      bar(100);play(d.file);log("✅ تم");
    }else{
      const d=await(await fetch("/api/tts-long",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
      if(!d.ok){log("❌");return}
      JOB=d.job;log("🆔 "+JOB);
      if(POLL)clearInterval(POLL);
      POLL=setInterval(async()=>{
        const j=await(await fetch("/api/job/"+JOB)).json();
        if(!j.ok)return;
        bar(j.percent);
        (j.log||[]).slice(-2).forEach(s=>{if(!$("log").textContent.includes(s))log(s)});
        if(j.job.status==="done"){clearInterval(POLL);play(j.job.file);log("✅ طويل جاهز");bar(100)}
        if(j.job.status==="failed"){clearInterval(POLL);log("❌ "+j.job.error);bar(0)}
      },2000);
    }
  }catch(e){log("❌ "+e)}
  finally{$("go").disabled=$("gol").disabled=false;setTimeout(()=>bar(0),1500)}
}
function play(u){const p=$("player");p.src=u+"?t="+Date.now();p.style.display="block";p.play().catch(()=>{})}
$("go").onclick=()=>synth(false);
$("gol").onclick=()=>synth(true);
$("stop").onclick=()=>$("player").pause();

$("style").value="Calm, dignified and steady Bambara narration. Warm authoritative voice. Clear precise pronunciation, even pacing. Clean studio quality, no crackle.";
count();loadVoices();health();
log("ℹ️ أدخل المفتاح مرة واحدة ثم ولّد الصوت.");
</script></body></html>"""

if __name__ == "__main__":
    print("→ http://127.0.0.1:8000")
    print("  المفتاح: من الواجهة (يُحفظ في .gemini_key) أو GEMINI_API_KEY")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

# Available languages: bambara, boomu, dogon, pular, songhoy, tamasheq
language = "bambara"
model_id = "Bamanakan-tts"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=f"models/{language}")
model = VitsModel.from_pretrained(model_id, subfolder=f"models/{language}")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Synthesize speech
text = "Nin ye bamanankan nimɔrɔko ɲɛjiralan dafalen ye. Jateden minnu bɛ bɔ 0 la ka se 10 ma : 0: fu 1: kelen 2: fila (walima fla) 3: saba 4: naani 5: duuru 6: wɔɔrɔ 7: wolonwula (walima wolonwufla) 8: lajɛ 9: 1 conton la code pour les nombres entre 11 Ani 19, 11: tan ni kelen (10 Ani 1) 12: tan ni fila (10 Ani 2) 13: tan ni saba 14: tan ni naani 15: tan ni duuru16: tan ni wɔɔrɔ17: tan ni wolonwula18: tan ni seegin19: tan ni kɔnɔntɔn Tan tɔw bɛɛ kama k’a ta 30 na ka se 90 ma, an bɛ baara kɛ ni daɲɛ fɔlɔ bi- ye min bɛ tugu ɲɔgɔn kɔ ni jateden cayalen ye :20: mugan30: bisaba (Tan ka bɔ 3 la)40: binaani
50: biduuru
60: biwɔɔrɔ 
70:Biwolwula
80: Biseegin
90:Bikɔnɔntɔn
Jatedenba 100: kɛmɛ 
1 000: Bakelen (walima Wakelen walima Wagakelen) . 
1 111 : bakelen ani kɛmɛ ni tan ni kelen 
1 000000: miliyɔn kelen 
1 111111: miliyɔn kelen ani bakɛmɛ kɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen  
1 000000000: miliyari kelen
1 111111111: miliyari kelen ani miliyɔn kelen ni kɛmɛ ni tan ni kelen ani bakɛmɛ kɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen 
1 000000000000:tiriliyɔni kelen 
1 111111111111 : tiriliyɔni kelen ani miliyari kelen ni kɛmɛ ni tan ni kelen ani miliyɔn kelen ni kɛmɛ ni tan ni kelen ani bakɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen 
1,1% : kɛmɛsarada la kelen n'a kunkanfɛn kelen 
1%: kɛmɛsarada la kelen 
1001%: kɛmɛsarada la bakelen an'a kunkanfɛn kelen
A kɛ a ka kalan kɛ lɛrɛ la
10:30:01 : nɛgɛ kanɲɛ tan tɛmɛnen ye ni sanga bisaba ye ani segɔni kelen  
18h02:01: nɛgɛ kanɲɛ tan ni seegin tɛmɛnen ye ni sanga fila ye ani segɔni kelen 
18:00: nɛgɛ kanɲɛ tan ni seegin
100.000:bakɛmɛ
II: fila 
I: kelen 
1,1: kelen n'a kunkanfɛn kelen 
1,100: kelen an'a kunkanfɛn kɛmɛ
B: be
C: ce
D: de
F: fe
G: ge
H: he
J: je
K:ke
L: le
M: me
N: ne
Ɲ: ɲe
Ŋ: ŋe
P: pe 
R: re 
S: se
T: te 
W: we 
Y: ye 
Z: ze
A E I Ɛ U O Ɔ 
B ba be bi bɛ bu bo bɔ
C ca ce ci cɛ cu co cɔ 
D da de di dɛ du do dɔ
F fa fe fi fɛ fu fo fɔ
G ga ge gi gɛ gu go gɔ 
H ha he hi hɛ hu ho hɔ 
J  ja je ji jɛ ju jo jɔ
K  ka ke ki kɛ ku ko kɔ
L  la le li lɛ lu lo lɔ
M  ma me mi mɛ mu mo mɔ
N  na ne ni nɛ nu no nɔ
Ɲ ɲa ɲe ɲi ɲɛ ɲu ɲo ɲɔ
Ŋ ŋa ŋe ŋi ŋɛ ŋu ŋo ŋɔ
P pa pe pi pɛ pu po pɔ 
R ra re ri rɛ ru ro rɔ
S sa se si sɛ su so sɔ 
T ta te ti tɛ tu to tɔ 
W wa we wi wɛ wu wo wɔ 
Y ya ye yi yɛ yu yo yɔ 
Z za ze zi zɛ zu zo zɔ

AA EE II ƐƐ UU OO ƆƆ 
B baa bee bii bɛɛ buu boo bɔɔ
C caa cee cii cɛɛ cuu coo cɔɔ 
D daa dee dii dɛɛ duu doo dɔɔ
F faa fee fii fɛɛ fuu foo fɔɔ
G gaa gee gii gɛɛ guu goo gɔɔ 
H haa hee hii hɛɛ huu hoo hɔɔ 
J  jaa jee jii jɛɛ juu joo jɔɔ
K  kaa kee kii kɛɛ kuu koo kɔɔ
L  laa lee lii lɛɛ luu loo lɔɔ
M  maa mee mii mɛɛ muu moo mɔɔ
N  naa nee nii nɛɛ nuu noo nɔɔ
Ɲ ɲaa ɲee ɲii ɲɛɛ ɲuu ɲoo ɲɔɔ
Ŋ ŋaa ŋee ŋii ŋɛɛ ŋuu ŋoo ŋɔɔ
P paa pee pii pɛɛ puu poo pɔɔ 
R raa ree rii rɛɛ ruu roo rɔɔ
S saa see sii sɛɛ suu soo sɔɔ 
T taa tee tii tɛɛ tuu too tɔɔ 
W waa wee wii wɛɛ wuu woo wɔɔ 
Y yaa yee yii yɛɛ yuu yoo yɔɔ 
Z zaa zee zii zɛɛ zuu zoo zɔɔ

AKA EKE IKI ƐKƐ UKU OKO ƆKƆ 
B baka beke biki bɛkɛ buku boko bɔkɔ
C caka ceke ciki cɛkɛ cuku coko cɔkɔ 
D daka deke diki dɛkɛ duku doko dɔkɔ
F faka feke fiki fɛkɛ fuku foko fɔlɔ
G gaka geke giki gɛkɛ guku goko gɔkɔ 
H haka heke hiki hɛkɛ huku hoko hɔkɔ 
J  jaka jeke jiki jɛkɛ juku joko jɔkɔ
K  kaka keke kiki kɛkɛ kuku koko kɔkɔ
L  laka leke liki lɛkɛ luku loko lɔkɔ
M  maka meke miki mɛkɛ muku moko mɔkɔ
N  naka neke niki nɛkɛ nuku noko nɔkɔ
Ɲ ɲaka ɲeke ɲiki ɲɛkɛ ɲuku ɲoko ɲɔkɔ
Ŋ ŋaka ŋeke ŋiki ŋɛkɛ ŋuku ŋoko ŋɔkɔ
P paka peke piki pɛkɛ puku poko pɔkɔ 
R raka reke riki rɛkɛ ruku roko rɔkɔ
S saka seke siki sɛkɛ suku soko sɔkɔ 
T taka teke tiki tɛkɛ tuku toko tɔkɔ 
W waka weje wiki wɛkɛ wuku woko wɔkɔ 
Y yaka yeke yiki yɛkɛ yuku yoko yɔkɔ 
Z zaka zeke ziki zɛkɛ zuku zoko zɔkɔ

AGA EGE IGI ƐGƐ UGU OGO ƆGƆ 
B baga bege bigi bɛgɛ bugu bogo bɔgɔ
C caga cege cigi cɛgɛ cugu cogo cɔgɔ 
D daga dege digi dɛgɛ dugu dogo dɔgɔ
F faga fege figi fɛgɛ fugu fogo fɔgɔ
G gaga gege gigi gɛgɛ gugu gogo gɔgɔ 
H haga hege higi hɛgɛ hugu hogo hɔgɔ 
J  jaga jege jigi jɛgɛ jugu jogo jɔgɔ
 K kaga kege kigi kɛgɛ kugu kogo kɔgɔ
L  laga lege ligi lɛgɛ lugu logo lɔgɔ
M  maga mege migi mɛgɛ mugu mogo mɔgɔ
N  naga nege nigi nɛgɛ nugu nogo nɔgɔ
Ɲ ɲaga ɲege ɲigi ɲɛgɛ ɲugu ɲogo ɲɔgɔ
Ŋ ŋaga ŋege ŋigi ŋɛgɛ ŋugu ŋogo ŋɔgɔ
P paga pege pigi pɛgɛ pugu pogo pɔgɔ 
R raga rege rigi rɛgɛ rugu rogo rɔgɔ
S saga sege sigi sɛgɛ sugu sogo sɔgɔ 
T taga tege tigi tɛgɛ tugu togo tɔgɔ 
W waga wege wigi wɛgɛ wugu wogo wɔgɔ 
Y yaga yege yigi yɛgɛ yugu yogo yɔgɔ 
Z zaga zege zigi zɛgɛ zugu zogo zɔgɔ
AN EN IN ƐN UN ON ƆN 
B ban ben bin bɛn bun bon bɔn
C can cen cin cɛn cun con cɔn 
D dan den din dɛn dun don dɔn
F fan fen fin fɛn fun fon fɔn
G gan gen gin gɛn gun gon gɔn 
H han hen hin hɛn hun hon hɔn 
J  jan jen jin jɛn jun jon jɔn
K  kan ken kin kɛn kun kon kɔn
L  lan len lin lɛn lun lon lɔn
M  man men min mɛn mun mon mɔn
N  nan nen nin nɛn nun non nɔn
Ɲ ɲan ɲen ɲin ɲɛn ɲun ɲon ɲɔn
Ŋ ŋan ŋen ŋin ŋɛn ŋun ŋon ŋɔn
P pan pen pin pɛn pun pon pɔn 
R ran ren rin rɛn run ron rɔn
S san sen sin sɛn sun son sɔn 
T tan ten tin tɛn tun ton tɔn 
W wan wen win wɛn wun won wɔn 
Y yan yen yin yɛn yun yon yɔn 
Z zan zen zin zɛn zun zon zɔn
40.000: babinaani
40,000: babinaani
 5,5: duuru n'a kunkanfɛn duuru 
5,100:duuru an'a kunkanfɛn kɛmɛ 
30.000: babisaba
30,000:babisaba
26,000: bamugan ni wɔɔrɔ
26.000:bamugan ni wɔɔrɔ
26000:bamugan ni wɔɔrɔ
16nan: tan ni wɔɔrɔnan
21,9%: kɛmɛsarada la mugan ni kelen n'a kunkanfɛn kɔnɔntɔn
42,7%: kɛmɛsarada la binaani ni fila n'a kunkanfɛn wolonwula
7,4%: kɛmɛsarada la wolonwula n'a kunkanfɛn naani
11,7%: kɛmɛsarada la tan ni kelen n'a kunkanfɛn wolonwula
111,7%: kɛmɛsarada la kɛmɛ ni tan ni kelen n'a kunkanfɛn wolonwula
12nan: tan ni filanan
18h: nɛgɛ kanɲɛ tan ni seegin
J-10CE: J-tan CE
2026: bafila ani Mugan ni Wɔɔrɔ 
Misaliw faralen ɲɔgɔn kancogo gɛlɛnw kanWalisa ka cɛmancɛ nafaw jira, an bɛ o sariya kelen in waleya ni ni farali ye ka kɔn unit ɲɛ:25 : mugan ni duuru (20 ni 5)42 : binaani ni fila (40 ni 2) .Wolonwula biwolonwula bawolonwula bakɛmɛwolonwula An filɛ nin ye yɔrɔ minna n'an ye an sigi k'a layɛ yala an bɛ ka baara min kɛ yala a kɛlen don ka ɲɛ wa ?"
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    output = model(**inputs).waveform

waveform = output.squeeze().cpu().numpy()
sample_rate = model.config.sampling_rate

# Save to file
sf.write("output.wav", waveform, sample_rate)

Bambara
text = "Nin ye bamanankan nimɔrɔko ɲɛjiralan dafalen ye. Jateden minnu bɛ bɔ 0 la ka se 10 ma kelenw tɔgɔ kɛrɛnkɛrɛnnenw bɛ yen minnu bɛ kɛ jɔli ye sigida tɔ la: 0: fu 1: kelen 2: fila (walima fla) 3: saba 4: naani 5: duuru 6: wɔɔrɔ 7: wolonwula (walima wolonwufla) 8: seegin  9: kɔnɔntɔn 
10: tan 
Jateden minnu bɛ bɔ 11 fo 19: Walasa ka jatedenw ka kode jɔ 11 ni 19  cɛ, tan (tán) bɛ tali kɛ dakun na ni daɲɛ dorokolen ye ni (o koro ye ko"ani" walima "ni"): 11: tan ni kelen (10 Ani 1) 12: tan ni fila (10 Ani 2) 13: tan ni saba 14: tan ni naani 15: tan ni duuru16: tan ni wɔɔrɔ17: tan ni wolonwula18: tan ni seegin19: tan ni kɔnɔntɔn Tan tɔw bɛɛ kama k’a ta 30 na ka se 90 ma, an bɛ baara kɛ ni daɲɛ fɔlɔ bi- ye min bɛ tugu ɲɔgɔn kɔ ni jateden cayalen ye :20: mugan30: bisaba (Tan ka bɔ 3 la)40: binaani
50: biduuru
60: biwɔɔrɔ 
70:Biwolwula
80: Biseegin
90:Bikɔnɔntɔn
Jatedenba 100: kɛmɛ 
1 000: Bakelen (walima Wakelen walima Wagakelen) . 
1 111 : bakelen ani kɛmɛ ni tan ni kelen 
1 000000: miliyɔn kelen 
1 111111: miliyɔn kelen ani bakɛmɛ kɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen  
1 000000000: miliyari kelen
1 111111111: miliyari kelen ani miliyɔn kelen ni kɛmɛ ni tan ni kelen ani bakɛmɛ kɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen 
1 000000000000:tiriliyɔni kelen 
1 111111111111 : tiriliyɔni kelen ani miliyari kelen ni kɛmɛ ni tan ni kelen ani miliyɔn kelen ni kɛmɛ ni tan ni kelen ani bakɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen 
1,1% : kɛmɛsarada la kelen n'a kunkanfɛn kelen 
1%: kɛmɛsarada la kelen 
1001%: kɛmɛsarada la bakelen an'a kunkanfɛn kelen 
 A kɛ a ka kalan kɛ lɛrɛ la
10:30:01 : nɛgɛ kanɲɛ tan tɛmɛnen ye ni sanga bisaba ye ani segɔni kelen
18h02:01: nɛgɛ kanɲɛ tan bi seegin tɛmɛnen ye ni sanga fila ye ani segɔni
18h:05: nɛgɛ kanɲɛ tan ni seegin tɛmɛnen ye ni sanga duuru ye  
18:00: nɛgɛ kanɲɛ tan ni seegin 
100.000:bakɛmɛ 
II: fila 
I: kelen 
1,1: kelen n'a kunkanfɛn kelen 
1,100: kelen an'a kunkanfɛn kɛmɛ
B: be
C: ce
D: de
F: fe
G: ge
H: he
J: je
K:ke
L: le
M: me
N: ne
Ɲ: ɲe
Ŋ: ŋe
P: pe 
R: re 
S: se
T: te 
W: we 
Y: ye 
Z: ze
A E I Ɛ U O Ɔ 
B ba be bi bɛ bu bo bɔ
C ca ce ci cɛ cu co cɔ 
D da de di dɛ du do dɔ
F fa fe fi fɛ fu fo fɔ
G ga ge gi gɛ gu go gɔ 
H ha he hi hɛ hu ho hɔ 
J  ja je ji jɛ ju jo jɔ
K  ka ke ki kɛ ku ko kɔ
L  la le li lɛ lu lo lɔ
M  ma me mi mɛ mu mo mɔ
N  na ne ni nɛ nu no nɔ
Ɲ ɲa ɲe ɲi ɲɛ ɲu ɲo ɲɔ
Ŋ ŋa ŋe ŋi ŋɛ ŋu ŋo ŋɔ
P pa pe pi pɛ pu po pɔ 
R ra re ri rɛ ru ro rɔ
S sa se si sɛ su so sɔ 
T ta te ti tɛ tu to tɔ 
W wa we wi wɛ wu wo wɔ 
Y ya ye yi yɛ yu yo yɔ 
Z za ze zi zɛ zu zo zɔ

AA EE II ƐƐ UU OO ƆƆ 
B baa bee bii bɛɛ buu boo bɔɔ
C caa cee cii cɛɛ cuu coo cɔɔ 
D daa dee dii dɛɛ duu doo dɔɔ
F faa fee fii fɛɛ fuu foo fɔɔ
G gaa gee gii gɛɛ guu goo gɔɔ 
H haa hee hii hɛɛ huu hoo hɔɔ 
J  jaa jee jii jɛɛ juu joo jɔɔ
K  kaa kee kii kɛɛ kuu koo kɔɔ
L  laa lee lii lɛɛ luu loo lɔɔ
M  maa mee mii mɛɛ muu moo mɔɔ
N  naa nee nii nɛɛ nuu noo nɔɔ
Ɲ ɲaa ɲee ɲii ɲɛɛ ɲuu ɲoo ɲɔɔ
Ŋ ŋaa ŋee ŋii ŋɛɛ ŋuu ŋoo ŋɔɔ
P paa pee pii pɛɛ puu poo pɔɔ 
R raa ree rii rɛɛ ruu roo rɔɔ
S saa see sii sɛɛ suu soo sɔɔ 
T taa tee tii tɛɛ tuu too tɔɔ 
W waa wee wii wɛɛ wuu woo wɔɔ 
Y yaa yee yii yɛɛ yuu yoo yɔɔ 
Z zaa zee zii zɛɛ zuu zoo zɔɔ

AKA EKE IKI ƐKƐ UKU OKO ƆKƆ 
B baka beke biki bɛkɛ buku boko bɔkɔ
C caka ceke ciki cɛkɛ cuku coko cɔkɔ 
D daka deke diki dɛkɛ duku doko dɔkɔ
F faka feke fiki fɛkɛ fuku foko fɔlɔ
G gaka geke giki gɛkɛ guku goko gɔkɔ 
H haka heke hiki hɛkɛ huku hoko hɔkɔ 
J  jaka jeke jiki jɛkɛ juku joko jɔkɔ
K  kaka keke kiki kɛkɛ kuku koko kɔkɔ
L  laka leke liki lɛkɛ luku loko lɔkɔ
M  maka meke miki mɛkɛ muku moko mɔkɔ
N  naka neke niki nɛkɛ nuku noko nɔkɔ
Ɲ ɲaka ɲeke ɲiki ɲɛkɛ ɲuku ɲoko ɲɔkɔ
Ŋ ŋaka ŋeke ŋiki ŋɛkɛ ŋuku ŋoko ŋɔkɔ
P paka peke piki pɛkɛ puku poko pɔkɔ 
R raka reke riki rɛkɛ ruku roko rɔkɔ
S saka seke siki sɛkɛ suku soko sɔkɔ 
T taka teke tiki tɛkɛ tuku toko tɔkɔ 
W waka weje wiki wɛkɛ wuku woko wɔkɔ 
Y yaka yeke yiki yɛkɛ yuku yoko yɔkɔ 
Z zaka zeke ziki zɛkɛ zuku zoko zɔkɔ

AGA EGE IGI ƐGƐ UGU OGO ƆGƆ 
B baga bege bigi bɛgɛ bugu bogo bɔgɔ
C caga cege cigi cɛgɛ cugu cogo cɔgɔ 
D daga dege digi dɛgɛ dugu dogo dɔgɔ
F faga fege figi fɛgɛ fugu fogo fɔgɔ
G gaga gege gigi gɛgɛ gugu gogo gɔgɔ 
H haga hege higi hɛgɛ hugu hogo hɔgɔ 
J  jaga jege jigi jɛgɛ jugu jogo jɔgɔ
 K kaga kege kigi kɛgɛ kugu kogo kɔgɔ
L  laga lege ligi lɛgɛ lugu logo lɔgɔ
M  maga mege migi mɛgɛ mugu mogo mɔgɔ
N  naga nege nigi nɛgɛ nugu nogo nɔgɔ
Ɲ ɲaga ɲege ɲigi ɲɛgɛ ɲugu ɲogo ɲɔgɔ
Ŋ ŋaga ŋege ŋigi ŋɛgɛ ŋugu ŋogo ŋɔgɔ
P paga pege pigi pɛgɛ pugu pogo pɔgɔ 
R raga rege rigi rɛgɛ rugu rogo rɔgɔ
S saga sege sigi sɛgɛ sugu sogo sɔgɔ 
T taga tege tigi tɛgɛ tugu togo tɔgɔ 
W waga wege wigi wɛgɛ wugu wogo wɔgɔ 
Y yaga yege yigi yɛgɛ yugu yogo yɔgɔ 
Z zaga zege zigi zɛgɛ zugu zogo zɔgɔ
AN EN IN ƐN UN ON ƆN 
B ban ben bin bɛn bun bon bɔn
C can cen cin cɛn cun con cɔn 
D dan den din dɛn dun don dɔn
F fan fen fin fɛn fun fon fɔn
G gan gen gin gɛn gun gon gɔn 
H han hen hin hɛn hun hon hɔn 
J  jan jen jin jɛn jun jon jɔn
K  kan ken kin kɛn kun kon kɔn
L  lan len lin lɛn lun lon lɔn
M  man men min mɛn mun mon mɔn
N  nan nen nin nɛn nun non nɔn
Ɲ ɲan ɲen ɲin ɲɛn ɲun ɲon ɲɔn
Ŋ ŋan ŋen ŋin ŋɛn ŋun ŋon ŋɔn
P pan pen pin pɛn pun pon pɔn 
R ran ren rin rɛn run ron rɔn
S san sen sin sɛn sun son sɔn 
T tan ten tin tɛn tun ton tɔn 
W wan wen win wɛn wun won wɔn 
Y yan yen yin yɛn yun yon yɔn 
Z zan zen zin zɛn zun zon zɔn
40.000: babinaani
40,000: babinaani 
 5,5: duuru n'a kunkanfɛn duuru 
5,100:duuru an'a kunkanfɛn kɛmɛ 
30.000: babisaba
30,000: babisaba
26,000: bamugan ni wɔɔrɔ
26.000:bamugan ni wɔɔrɔ
26000:bamugan ni wɔɔrɔ
16nan: tan ni wɔɔrɔnan
21,9%: kɛmɛsarada la mugan ni kelen n'a kunkanfɛn kɔnɔntɔn
42,7%: kɛmɛsarada la binaani ni fila n'a kunkanfɛn wolonwula
7,4%: kɛmɛsarada la wolonwula n'a kunkanfɛn naani
11,7%: kɛmɛsarada la tan ni kelen n'a kunkanfɛn wolonwula
111,7%: kɛmɛsarada la kɛmɛ ni tan ni kelen n'a kunkanfɛn wolonwula
12nan: tan ni filanan
18h:05: nɛgɛ kanɲɛ tan ni seegin tɛmɛnen ye ni sanga duuru ye 
18:00: nɛgɛ kanɲɛ tan ni seegin
J-10CE: J-tan CE
2026: bafila ani Mugan ni Wɔɔrɔ
Misaliw faralen ɲɔgɔn kancogo gɛlɛnw kanWalisa ka cɛmancɛ nafaw jira, an bɛ o sariya kelen in waleya ni ni farali ye ka kɔn unit ɲɛ:25 : mugan ni duuru (20 ni 5)42 : binaani ni fila (40 ni 2) .Wolonwula biwolonwula bawolonwula bakɛmɛwolonwula An filɛ nin ye yɔrɔ minna n'an ye an sigi k'a layɛ yala an bɛ ka baara min kɛ yala a kɛlen don ka ɲɛ wa?"

# Boomu
text = "Vunurobe wozomɛ pɛɛ, Poli we zo woro han Deeɓenu wara li Deeɓenu faralo zuun. Lo we baba a lo wara yi see ɓa Zuwifera ma ɓa Gɛrɛkela wa."

# Dogon
text = "Pɔɔlɔ, kubɔ lugo joo le, bana dɛin dɛin le, inɛw Ama titiyaanw le digɛu, Ama, emɛ babe bɛrɛ sɔɔ sɔi."

# Pular
text = "Miɗo ndaarde saabe Laamɗo e saabe Iisaa Almasiihu caroyoowo wuurɓe e maayɓe oo, miɗo ndaardire saabe gartol makko ka num e Laamu makko"

# Songhoy
text = "Haya ka se beenediyo kokoyteraydi go hima nda huukoy foo ka fatta ja subaahi ka taasi goykoyyo ngu rezẽ faridi se"

# Tamasheq
text = "Toḍă tăfukt ɣas, issăɣră-dd măssi-s n-ašĕkrĕš ănaẓraf-net, inn'-as: 'Ǝɣĕr-dd


تحويل النص و الأرقام إلى كلام   للغة البامبارا 

تقنية قوية لتوليد الكلام

بزمن استجابة منخفض بمخرجات طبيعية ومطالبات قابلة للتوجيه، وعلامات صوتية معبرة جديدة للتحكم الدقيق في السرد وفيها الاستنساخ الأصوات محفظة الاصوات   مع  صوت Kore zephyr Algienlba Charon Leda Puck Umbriel Zubenrlgenubi Speaker recommendations:

  - Bourama: Most stable and accurate 
  - Adama: Natural conversational tone
  - Moussa: Clear pronunciation  
  - Modibo: Expressive delivery
  - Seydou: Balanced characteristics
  - Amadou: Warm and friendly voice
  - Bakary: Deep, authoritative tone
  - Ngolo: Youthful and energetic
  - Ibrahima: Calm and measured
  - Amara: Melodic and smooth

from maliba_ai.config.settings import Speakers

text = "Aw ni ce. Ne tɔgɔ ye Adama. Awɔ,  ne ye maliden de ye. Aw Sanbɛ Sanbɛ. San min tɛ ɲinan ye, an bɛɛ ka jɛ ka o seli ɲɔgɔn fɛ,  hɛɛrɛ  ni lafiya la. Ala ka Mali suma. Ala ka Mali yiriwa. Ala ka Mali taa ɲɛ. Ala ka an ka seliw caya. Ala ka yafa an bɛɛ ma."

#let's try Adama
tts.generate_speech(
    text = text, 
    speaker_id = Speakers.Adama,
    output_filename = "adama.wav"
)



#let's try Seydou
tts.generate_speech(
    text = text, 
    speaker_id = Speakers.Seydou,
    output_filename = "seydou.wav"
)


# let's try Bourama
tts.generate_speech(
    text = text, 
  
  speaker_id = Speakers.Bourama,
    output_filename = "Bourama.wav"
)

from whosper import WhosperTranscriber

transcriber = WhosperTranscriber(model_id="MALIBA-AI/bambara-asr-v3")

result = transcriber.transcribe_audio("path/to/audio.wav")
print(result)
 واصوات المثقفين الاديب أن يقرأ بالوقار والثبات
اجعلها تستطيع قراءة بالصوت البامبارا صحيحة جدا نقية من القرقرة اجعلها تستطيع قراءة 
wolonwula و cogoya و cogo 

اجعلها تطبيق تستطيع قراءة ارقام والنص باللغة البامبارا بصوت صحيحة بالوقار والثبات جدا نقية من القرقرة والتشوش الصوت لا تنقص منها شيء اجعلها تستطيع قراءة ارقام والنص باللغة البامبارا بصوت صحيحة قوية احترافيته بالوقار والثبات جدا نقية من القرقرة والتشوش الصوت مع استنساخ الصوت بصمة مع كل ميزتها. اجعلها تستطيع قراءة النص كيفما طالت وكثرت
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

# Available languages: bambara, boomu, dogon, pular, songhoy, tamasheq
language = "bambara"
model_id = "Bamanakan-tts"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=f"models/{language}")
model = VitsModel.from_pretrained(model_id, subfolder=f"models/{language}")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Synthesize speech
text = "Nin ye bamanankan nimɔrɔko ɲɛjiralan dafalen ye. Jateden minnu bɛ bɔ 0 la ka se 10 ma : 0: fu 1: kelen 2: fila (walima fla) 3: saba 4: naani 5: duuru 6: wɔɔrɔ 7: wolonwula (walima wolonwufla) 8: lajɛ 9: 1 conton la code pour les nombres entre 11 Ani 19, 11: tan ni kelen (10 Ani 1) 12: tan ni fila (10 Ani 2) 13: tan ni saba 14: tan ni naani 15: tan ni duuru16: tan ni wɔɔrɔ17: tan ni wolonwula18: tan ni seegin19: tan ni kɔnɔntɔn Tan tɔw bɛɛ kama k’a ta 30 na ka se 90 ma, an bɛ baara kɛ ni daɲɛ fɔlɔ bi- ye min bɛ tugu ɲɔgɔn kɔ ni jateden cayalen ye :20: mugan30: bisaba (Tan ka bɔ 3 la)40: binaani
50: biduuru
60: biwɔɔrɔ 
70:Biwolwula
80: Biseegin
90:Bikɔnɔntɔn
Jatedenba 100: kɛmɛ 
1 000: Bakelen (walima Wakelen walima Wagakelen) . 
1 111 : bakelen ani kɛmɛ ni tan ni kelen 
1 000000: miliyɔn kelen 
1 111111: miliyɔn kelen ani bakɛmɛ kɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen  
1 000000000: miliyari kelen
1 111111111: miliyari kelen ani miliyɔn kelen ni kɛmɛ ni tan ni kelen ani bakɛmɛ kɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen 
1 000000000000:tiriliyɔni kelen 
1 111111111111 : tiriliyɔni kelen ani miliyari kelen ni kɛmɛ ni tan ni kelen ani miliyɔn kelen ni kɛmɛ ni tan ni kelen ani bakɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen 
1,1% : kɛmɛsarada la kelen n'a kunkanfɛn kelen 
1%: kɛmɛsarada la kelen 
1001%: kɛmɛsarada la bakelen an'a kunkanfɛn kelen
A kɛ a ka kalan kɛ lɛrɛ la
10:30:01 : nɛgɛ kanɲɛ tan tɛmɛnen ye ni sanga bisaba ye ani segɔni kelen  
18h02:01: nɛgɛ kanɲɛ tan ni seegin tɛmɛnen ye ni sanga fila ye ani segɔni kelen 
18:00: nɛgɛ kanɲɛ tan ni seegin
100.000:bakɛmɛ
II: fila 
I: kelen 
1,1: kelen n'a kunkanfɛn kelen 
1,100: kelen an'a kunkanfɛn kɛmɛ
B: be
C: ce
D: de
F: fe
G: ge
H: he
J: je
K:ke
L: le
M: me
N: ne
Ɲ: ɲe
Ŋ: ŋe
P: pe 
R: re 
S: se
T: te 
W: we 
Y: ye 
Z: ze
A E I Ɛ U O Ɔ 
B ba be bi bɛ bu bo bɔ
C ca ce ci cɛ cu co cɔ 
D da de di dɛ du do dɔ
F fa fe fi fɛ fu fo fɔ
G ga ge gi gɛ gu go gɔ 
H ha he hi hɛ hu ho hɔ 
J  ja je ji jɛ ju jo jɔ
K  ka ke ki kɛ ku ko kɔ
L  la le li lɛ lu lo lɔ
M  ma me mi mɛ mu mo mɔ
N  na ne ni nɛ nu no nɔ
Ɲ ɲa ɲe ɲi ɲɛ ɲu ɲo ɲɔ
Ŋ ŋa ŋe ŋi ŋɛ ŋu ŋo ŋɔ
P pa pe pi pɛ pu po pɔ 
R ra re ri rɛ ru ro rɔ
S sa se si sɛ su so sɔ 
T ta te ti tɛ tu to tɔ 
W wa we wi wɛ wu wo wɔ 
Y ya ye yi yɛ yu yo yɔ 
Z za ze zi zɛ zu zo zɔ

AA EE II ƐƐ UU OO ƆƆ 
B baa bee bii bɛɛ buu boo bɔɔ
C caa cee cii cɛɛ cuu coo cɔɔ 
D daa dee dii dɛɛ duu doo dɔɔ
F faa fee fii fɛɛ fuu foo fɔɔ
G gaa gee gii gɛɛ guu goo gɔɔ 
H haa hee hii hɛɛ huu hoo hɔɔ 
J  jaa jee jii jɛɛ juu joo jɔɔ
K  kaa kee kii kɛɛ kuu koo kɔɔ
L  laa lee lii lɛɛ luu loo lɔɔ
M  maa mee mii mɛɛ muu moo mɔɔ
N  naa nee nii nɛɛ nuu noo nɔɔ
Ɲ ɲaa ɲee ɲii ɲɛɛ ɲuu ɲoo ɲɔɔ
Ŋ ŋaa ŋee ŋii ŋɛɛ ŋuu ŋoo ŋɔɔ
P paa pee pii pɛɛ puu poo pɔɔ 
R raa ree rii rɛɛ ruu roo rɔɔ
S saa see sii sɛɛ suu soo sɔɔ 
T taa tee tii tɛɛ tuu too tɔɔ 
W waa wee wii wɛɛ wuu woo wɔɔ 
Y yaa yee yii yɛɛ yuu yoo yɔɔ 
Z zaa zee zii zɛɛ zuu zoo zɔɔ

AKA EKE IKI ƐKƐ UKU OKO ƆKƆ 
B baka beke biki bɛkɛ buku boko bɔkɔ
C caka ceke ciki cɛkɛ cuku coko cɔkɔ 
D daka deke diki dɛkɛ duku doko dɔkɔ
F faka feke fiki fɛkɛ fuku foko fɔlɔ
G gaka geke giki gɛkɛ guku goko gɔkɔ 
H haka heke hiki hɛkɛ huku hoko hɔkɔ 
J  jaka jeke jiki jɛkɛ juku joko jɔkɔ
K  kaka keke kiki kɛkɛ kuku koko kɔkɔ
L  laka leke liki lɛkɛ luku loko lɔkɔ
M  maka meke miki mɛkɛ muku moko mɔkɔ
N  naka neke niki nɛkɛ nuku noko nɔkɔ
Ɲ ɲaka ɲeke ɲiki ɲɛkɛ ɲuku ɲoko ɲɔkɔ
Ŋ ŋaka ŋeke ŋiki ŋɛkɛ ŋuku ŋoko ŋɔkɔ
P paka peke piki pɛkɛ puku poko pɔkɔ 
R raka reke riki rɛkɛ ruku roko rɔkɔ
S saka seke siki sɛkɛ suku soko sɔkɔ 
T taka teke tiki tɛkɛ tuku toko tɔkɔ 
W waka weje wiki wɛkɛ wuku woko wɔkɔ 
Y yaka yeke yiki yɛkɛ yuku yoko yɔkɔ 
Z zaka zeke ziki zɛkɛ zuku zoko zɔkɔ

AGA EGE IGI ƐGƐ UGU OGO ƆGƆ 
B baga bege bigi bɛgɛ bugu bogo bɔgɔ
C caga cege cigi cɛgɛ cugu cogo cɔgɔ 
D daga dege digi dɛgɛ dugu dogo dɔgɔ
F faga fege figi fɛgɛ fugu fogo fɔgɔ
G gaga gege gigi gɛgɛ gugu gogo gɔgɔ 
H haga hege higi hɛgɛ hugu hogo hɔgɔ 
J  jaga jege jigi jɛgɛ jugu jogo jɔgɔ
 K kaga kege kigi kɛgɛ kugu kogo kɔgɔ
L  laga lege ligi lɛgɛ lugu logo lɔgɔ
M  maga mege migi mɛgɛ mugu mogo mɔgɔ
N  naga nege nigi nɛgɛ nugu nogo nɔgɔ
Ɲ ɲaga ɲege ɲigi ɲɛgɛ ɲugu ɲogo ɲɔgɔ
Ŋ ŋaga ŋege ŋigi ŋɛgɛ ŋugu ŋogo ŋɔgɔ
P paga pege pigi pɛgɛ pugu pogo pɔgɔ 
R raga rege rigi rɛgɛ rugu rogo rɔgɔ
S saga sege sigi sɛgɛ sugu sogo sɔgɔ 
T taga tege tigi tɛgɛ tugu togo tɔgɔ 
W waga wege wigi wɛgɛ wugu wogo wɔgɔ 
Y yaga yege yigi yɛgɛ yugu yogo yɔgɔ 
Z zaga zege zigi zɛgɛ zugu zogo zɔgɔ
AN EN IN ƐN UN ON ƆN 
B ban ben bin bɛn bun bon bɔn
C can cen cin cɛn cun con cɔn 
D dan den din dɛn dun don dɔn
F fan fen fin fɛn fun fon fɔn
G gan gen gin gɛn gun gon gɔn 
H han hen hin hɛn hun hon hɔn 
J  jan jen jin jɛn jun jon jɔn
K  kan ken kin kɛn kun kon kɔn
L  lan len lin lɛn lun lon lɔn
M  man men min mɛn mun mon mɔn
N  nan nen nin nɛn nun non nɔn
Ɲ ɲan ɲen ɲin ɲɛn ɲun ɲon ɲɔn
Ŋ ŋan ŋen ŋin ŋɛn ŋun ŋon ŋɔn
P pan pen pin pɛn pun pon pɔn 
R ran ren rin rɛn run ron rɔn
S san sen sin sɛn sun son sɔn 
T tan ten tin tɛn tun ton tɔn 
W wan wen win wɛn wun won wɔn 
Y yan yen yin yɛn yun yon yɔn 
Z zan zen zin zɛn zun zon zɔn
40.000: babinaani
40,000: babinaani
 5,5: duuru n'a kunkanfɛn duuru 
5,100:duuru an'a kunkanfɛn kɛmɛ 
30.000 babisaba
30,000 babisaba
2026: bafila ani Mugan ni Wɔɔrɔ 
Misaliw faralen ɲɔgɔn kancogo gɛlɛnw kanWalisa ka cɛmancɛ nafaw jira, an bɛ o sariya kelen in waleya ni ni farali ye ka kɔn unit ɲɛ:25 : mugan ni duuru (20 ni 5)42 : binaani ni fila (40 ni 2) .Wolonwula biwolonwula bawolonwula bakɛmɛwolonwula An filɛ nin ye yɔrɔ minna n'an ye an sigi k'a layɛ yala an bɛ ka baara min kɛ yala a kɛlen don ka ɲɛ wa ?"
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    output = model(**inputs).waveform

waveform = output.squeeze().cpu().numpy()
sample_rate = model.config.sampling_rate

# Save to file
sf.write("output.wav", waveform, sample_rate)

Bambara
text = "Nin ye bamanankan nimɔrɔko ɲɛjiralan dafalen ye. Jateden minnu bɛ bɔ 0 la ka se 10 ma kelenw tɔgɔ kɛrɛnkɛrɛnnenw bɛ yen minnu bɛ kɛ jɔli ye sigida tɔ la: 0: fu 1: kelen 2: fila (walima fla) 3: saba 4: naani 5: duuru 6: wɔɔrɔ 7: wolonwula (walima wolonwufla) 8: seegin  9: kɔnɔntɔn 
10: tan 
Jateden minnu bɛ bɔ 11 fo 19: Walasa ka jatedenw ka kode jɔ 11 ni 19  cɛ, tan (tán) bɛ tali kɛ dakun na ni daɲɛ dorokolen ye ni (o koro ye ko"ani" walima "ni"): 11: tan ni kelen (10 Ani 1) 12: tan ni fila (10 Ani 2) 13: tan ni saba 14: tan ni naani 15: tan ni duuru16: tan ni wɔɔrɔ17: tan ni wolonwula18: tan ni seegin19: tan ni kɔnɔntɔn Tan tɔw bɛɛ kama k’a ta 30 na ka se 90 ma, an bɛ baara kɛ ni daɲɛ fɔlɔ bi- ye min bɛ tugu ɲɔgɔn kɔ ni jateden cayalen ye :20: mugan30: bisaba (Tan ka bɔ 3 la)40: binaani
50: biduuru
60: biwɔɔrɔ 
70:Biwolwula
80: Biseegin
90:Bikɔnɔntɔn
Jatedenba 100: kɛmɛ 
1 000: Bakelen (walima Wakelen walima Wagakelen) . 
1 111 : bakelen ani kɛmɛ ni tan ni kelen 
1 000000: miliyɔn kelen 
1 111111: miliyɔn kelen ani bakɛmɛ kɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen  
1 000000000: miliyari kelen
1 111111111: miliyari kelen ani miliyɔn kelen ni kɛmɛ ni tan ni kelen ani bakɛmɛ kɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen 
1 000000000000:tiriliyɔni kelen 
1 111111111111 : tiriliyɔni kelen ani miliyari kelen ni kɛmɛ ni tan ni kelen ani miliyɔn kelen ni kɛmɛ ni tan ni kelen ani bakɛmɛ ni tan ni kelen ani kɛmɛ ni tan ni kelen 
1,1% : kɛmɛsarada la kelen n'a kunkanfɛn kelen 
1%: kɛmɛsarada la kelen 
1001%: kɛmɛsarada la bakelen an'a kunkanfɛn kelen 
 A kɛ a ka kalan kɛ lɛrɛ la
10:30:01 : nɛgɛ kanɲɛ tan tɛmɛnen ye ni sanga bisaba ye ani segɔni kelen
18h02:01: nɛgɛ kanɲɛ tan bi seegin tɛmɛnen ye ni sanga fila ye ani segɔni 
18:00: nɛgɛ kanɲɛ tan ni seegin 
100.000:bakɛmɛ 
II: fila 
I: kelen 
1,1: kelen n'a kunkanfɛn kelen 
1,100: kelen an'a kunkanfɛn kɛmɛ
B: be
C: ce
D: de
F: fe
G: ge
H: he
J: je
K:ke
L: le
M: me
N: ne
Ɲ: ɲe
Ŋ: ŋe
P: pe 
R: re 
S: se
T: te 
W: we 
Y: ye 
Z: ze
A E I Ɛ U O Ɔ 
B ba be bi bɛ bu bo bɔ
C ca ce ci cɛ cu co cɔ 
D da de di dɛ du do dɔ
F fa fe fi fɛ fu fo fɔ
G ga ge gi gɛ gu go gɔ 
H ha he hi hɛ hu ho hɔ 
J  ja je ji jɛ ju jo jɔ
K  ka ke ki kɛ ku ko kɔ
L  la le li lɛ lu lo lɔ
M  ma me mi mɛ mu mo mɔ
N  na ne ni nɛ nu no nɔ
Ɲ ɲa ɲe ɲi ɲɛ ɲu ɲo ɲɔ
Ŋ ŋa ŋe ŋi ŋɛ ŋu ŋo ŋɔ
P pa pe pi pɛ pu po pɔ 
R ra re ri rɛ ru ro rɔ
S sa se si sɛ su so sɔ 
T ta te ti tɛ tu to tɔ 
W wa we wi wɛ wu wo wɔ 
Y ya ye yi yɛ yu yo yɔ 
Z za ze zi zɛ zu zo zɔ

AA EE II ƐƐ UU OO ƆƆ 
B baa bee bii bɛɛ buu boo bɔɔ
C caa cee cii cɛɛ cuu coo cɔɔ 
D daa dee dii dɛɛ duu doo dɔɔ
F faa fee fii fɛɛ fuu foo fɔɔ
G gaa gee gii gɛɛ guu goo gɔɔ 
H haa hee hii hɛɛ huu hoo hɔɔ 
J  jaa jee jii jɛɛ juu joo jɔɔ
K  kaa kee kii kɛɛ kuu koo kɔɔ
L  laa lee lii lɛɛ luu loo lɔɔ
M  maa mee mii mɛɛ muu moo mɔɔ
N  naa nee nii nɛɛ nuu noo nɔɔ
Ɲ ɲaa ɲee ɲii ɲɛɛ ɲuu ɲoo ɲɔɔ
Ŋ ŋaa ŋee ŋii ŋɛɛ ŋuu ŋoo ŋɔɔ
P paa pee pii pɛɛ puu poo pɔɔ 
R raa ree rii rɛɛ ruu roo rɔɔ
S saa see sii sɛɛ suu soo sɔɔ 
T taa tee tii tɛɛ tuu too tɔɔ 
W waa wee wii wɛɛ wuu woo wɔɔ 
Y yaa yee yii yɛɛ yuu yoo yɔɔ 
Z zaa zee zii zɛɛ zuu zoo zɔɔ

AKA EKE IKI ƐKƐ UKU OKO ƆKƆ 
B baka beke biki bɛkɛ buku boko bɔkɔ
C caka ceke ciki cɛkɛ cuku coko cɔkɔ 
D daka deke diki dɛkɛ duku doko dɔkɔ
F faka feke fiki fɛkɛ fuku foko fɔlɔ
G gaka geke giki gɛkɛ guku goko gɔkɔ 
H haka heke hiki hɛkɛ huku hoko hɔkɔ 
J  jaka jeke jiki jɛkɛ juku joko jɔkɔ
K  kaka keke kiki kɛkɛ kuku koko kɔkɔ
L  laka leke liki lɛkɛ luku loko lɔkɔ
M  maka meke miki mɛkɛ muku moko mɔkɔ
N  naka neke niki nɛkɛ nuku noko nɔkɔ
Ɲ ɲaka ɲeke ɲiki ɲɛkɛ ɲuku ɲoko ɲɔkɔ
Ŋ ŋaka ŋeke ŋiki ŋɛkɛ ŋuku ŋoko ŋɔkɔ
P paka peke piki pɛkɛ puku poko pɔkɔ 
R raka reke riki rɛkɛ ruku roko rɔkɔ
S saka seke siki sɛkɛ suku soko sɔkɔ 
T taka teke tiki tɛkɛ tuku toko tɔkɔ 
W waka weje wiki wɛkɛ wuku woko wɔkɔ 
Y yaka yeke yiki yɛkɛ yuku yoko yɔkɔ 
Z zaka zeke ziki zɛkɛ zuku zoko zɔkɔ

AGA EGE IGI ƐGƐ UGU OGO ƆGƆ 
B baga bege bigi bɛgɛ bugu bogo bɔgɔ
C caga cege cigi cɛgɛ cugu cogo cɔgɔ 
D daga dege digi dɛgɛ dugu dogo dɔgɔ
F faga fege figi fɛgɛ fugu fogo fɔgɔ
G gaga gege gigi gɛgɛ gugu gogo gɔgɔ 
H haga hege higi hɛgɛ hugu hogo hɔgɔ 
J  jaga jege jigi jɛgɛ jugu jogo jɔgɔ
 K kaga kege kigi kɛgɛ kugu kogo kɔgɔ
L  laga lege ligi lɛgɛ lugu logo lɔgɔ
M  maga mege migi mɛgɛ mugu mogo mɔgɔ
N  naga nege nigi nɛgɛ nugu nogo nɔgɔ
Ɲ ɲaga ɲege ɲigi ɲɛgɛ ɲugu ɲogo ɲɔgɔ
Ŋ ŋaga ŋege ŋigi ŋɛgɛ ŋugu ŋogo ŋɔgɔ
P paga pege pigi pɛgɛ pugu pogo pɔgɔ 
R raga rege rigi rɛgɛ rugu rogo rɔgɔ
S saga sege sigi sɛgɛ sugu sogo sɔgɔ 
T taga tege tigi tɛgɛ tugu togo tɔgɔ 
W waga wege wigi wɛgɛ wugu wogo wɔgɔ 
Y yaga yege yigi yɛgɛ yugu yogo yɔgɔ 
Z zaga zege zigi zɛgɛ zugu zogo zɔgɔ
AN EN IN ƐN UN ON ƆN 
B ban ben bin bɛn bun bon bɔn
C can cen cin cɛn cun con cɔn 
D dan den din dɛn dun don dɔn
F fan fen fin fɛn fun fon fɔn
G gan gen gin gɛn gun gon gɔn 
H han hen hin hɛn hun hon hɔn 
J  jan jen jin jɛn jun jon jɔn
K  kan ken kin kɛn kun kon kɔn
L  lan len lin lɛn lun lon lɔn
M  man men min mɛn mun mon mɔn
N  nan nen nin nɛn nun non nɔn
Ɲ ɲan ɲen ɲin ɲɛn ɲun ɲon ɲɔn
Ŋ ŋan ŋen ŋin ŋɛn ŋun ŋon ŋɔn
P pan pen pin pɛn pun pon pɔn 
R ran ren rin rɛn run ron rɔn
S san sen sin sɛn sun son sɔn 
T tan ten tin tɛn tun ton tɔn 
W wan wen win wɛn wun won wɔn 
Y yan yen yin yɛn yun yon yɔn 
Z zan zen zin zɛn zun zon zɔn
40.000: babinaani
40,000: babinaani 
 5,5: duuru n'a kunkanfɛn duuru 
5,100:duuru an'a kunkanfɛn kɛmɛ 
30.000 babisaba
30,000 babisaba
2026: bafila ani Mugan ni Wɔɔrɔ
Misaliw faralen ɲɔgɔn kancogo gɛlɛnw kanWalisa ka cɛmancɛ nafaw jira, an bɛ o sariya kelen in waleya ni ni farali ye ka kɔn unit ɲɛ:25 : mugan ni duuru (20 ni 5)42 : binaani ni fila (40 ni 2) .Wolonwula biwolonwula bawolonwula bakɛmɛwolonwula An filɛ nin ye yɔrɔ minna n'an ye an sigi k'a layɛ yala an bɛ ka baara min kɛ yala a kɛlen don ka ɲɛ wa?"

# Boomu
text = "Vunurobe wozomɛ pɛɛ, Poli we zo woro han Deeɓenu wara li Deeɓenu faralo zuun. Lo we baba a lo wara yi see ɓa Zuwifera ma ɓa Gɛrɛkela wa."

# Dogon
text = "Pɔɔlɔ, kubɔ lugo joo le, bana dɛin dɛin le, inɛw Ama titiyaanw le digɛu, Ama, emɛ babe bɛrɛ sɔɔ sɔi."

# Pular
text = "Miɗo ndaarde saabe Laamɗo e saabe Iisaa Almasiihu caroyoowo wuurɓe e maayɓe oo, miɗo ndaardire saabe gartol makko ka num e Laamu makko"

# Songhoy
text = "Haya ka se beenediyo kokoyteraydi go hima nda huukoy foo ka fatta ja subaahi ka taasi goykoyyo ngu rezẽ faridi se"

# Tamasheq
text = "Toḍă tăfukt ɣas, issăɣră-dd măssi-s n-ašĕkrĕš ănaẓraf-net, inn'-as: 'Ǝɣĕr-dd


تحويل النص و الأرقام إلى كلام   للغة البامبارا 

تقنية قوية لتوليد الكلام

بزمن استجابة منخفض بمخرجات طبيعية ومطالبات قابلة للتوجيه، وعلامات صوتية معبرة جديدة للتحكم الدقيق في السرد وفيها الاستنساخ الأصوات محفظة الاصوات   مع  صوت Kore zephyr Algienlba Charon Leda Puck Umbriel Zubenrlgenubi Speaker recommendations:

  - Bourama: Most stable and accurate 
  - Adama: Natural conversational tone
  - Moussa: Clear pronunciation  
  - Modibo: Expressive delivery
  - Seydou: Balanced characteristics
  - Amadou: Warm and friendly voice
  - Bakary: Deep, authoritative tone
  - Ngolo: Youthful and energetic
  - Ibrahima: Calm and measured
  - Amara: Melodic and smooth

from maliba_ai.config.settings import Speakers

text = "Aw ni ce. Ne tɔgɔ ye Adama. Awɔ,  ne ye maliden de ye. Aw Sanbɛ Sanbɛ. San min tɛ ɲinan ye, an bɛɛ ka jɛ ka o seli ɲɔgɔn fɛ,  hɛɛrɛ  ni lafiya la. Ala ka Mali suma. Ala ka Mali yiriwa. Ala ka Mali taa ɲɛ. Ala ka an ka seliw caya. Ala ka yafa an bɛɛ ma."

#let's try Adama
tts.generate_speech(
    text = text, 
    speaker_id = Speakers.Adama,
    output_filename = "adama.wav"
)



#let's try Seydou
tts.generate_speech(
    text = text, 
    speaker_id = Speakers.Seydou,
    output_filename = "seydou.wav"
)


# let's try Bourama
tts.generate_speech(
    text = text, 
  
  speaker_id = Speakers.Bourama,
    output_filename = "Bourama.wav"
)

from whosper import WhosperTranscriber

transcriber = WhosperTranscriber(model_id="MALIBA-AI/bambara-asr-v3")

result = transcriber.transcribe_audio("path/to/audio.wav")
print(result)
 واصوات المثقفين الاديب أن يقرأ بالوقار والثبات
اجعلها تستطيع قراءة بالصوت البامبارا صحيحة جدا نقية من القرقرة اجعلها تستطيع قراءة 
wolonwula و cogoya و cogo 
# 1. إنشاء بيئة افتراضية (موصى به)
python -m venv venv
source venv/bin/activate          # على ويندوز: venv\Scripts\activate

# 2. تثبيت المتطلبات
pip install streamlit torch torchaudio transformers soundfile librosa moviepy openai
pip install git+https://github.com/sudoping01/whosper.git
pip install maliba-ai

# 3. تشغيل التطبيق
streamlit run app.py# app.py
import streamlit as st
import tempfile
import os
from pathlib import Path
import torch
from moviepy.editor import VideoFileClip, AudioFileClip
import soundfile as sf
import librosa
import numpy as np

# ====================== إعدادات ======================
st.set_page_config(
    page_title="Bambara Video Dubber",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 دبلجة فيديو إلى البامبارا")
st.markdown("تطبيق حديث ونظيف باستخدام نماذج **MALIBA-AI** الرسمية")

# ====================== تحميل النماذج (مرة واحدة) ======================
@st.cache_resource
def load_models():
    # ASR (Whosper)
    from whosper import WhosperTranscriber
    asr = WhosperTranscriber(model_id="MALIBA-AI/bambara-asr-v3")

    # TTS (MALIBA)
    from maliba_ai.tts.inference import BambaraTTSInference
    from maliba_ai.config.settings import Speakers
    tts = BambaraTTSInference()
    
    return asr, tts, Speakers

try:
    asr_model, tts_model, Speakers = load_models()
    speakers_list = list(Speakers)
except Exception as e:
    st.error(f"خطأ في تحميل النماذج: {e}")
    st.stop()

# ====================== دوال مساعدة ======================
def extract_audio(video_path: str, audio_path: str):
    video = VideoFileClip(video_path)
    if video.audio is None:
        raise ValueError("الفيديو لا يحتوي على صوت")
    video.audio.write_audiofile(audio_path, logger=None, verbose=False)
    duration = video.duration
    video.close()
    return duration

def time_stretch(audio_path: str, target_duration: float, output_path: str):
    y, sr = librosa.load(audio_path, sr=None)
    current = librosa.get_duration(y=y, sr=sr)
    rate = current / target_duration
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    sf.write(output_path, y_stretched, sr)
    return output_path

def combine(video_path: str, audio_path: str, output_path: str, video_duration: float):
    # مزامنة المدة
    stretched_path = str(Path(audio_path).with_name("stretched.wav"))
    time_stretch(audio_path, video_duration, stretched_path)

    video = VideoFileClip(video_path)
    audio = AudioFileClip(stretched_path)
    final = video.set_audio(audio)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        logger=None,
        threads=4,
        verbose=False
    )
    video.close()
    audio.close()
    final.close()

# ====================== واجهة المستخدم ======================
uploaded_file = st.file_uploader(
    "ارفع فيديو (mp4, mov, mkv, avi)",
    type=["mp4", "mov", "mkv", "avi"]
)

selected_speaker = st.selectbox(
    "اختر صوت المتحدث",
    options=speakers_list,
    format_func=lambda x: x.name
)

translate_option = st.radio(
    "طريقة الترجمة",
    ["ترجمة تلقائية عبر LLM (موصى بها)", "استخدم النص كما هو (بدون ترجمة)"],
    index=0
)

api_key = None
if "ترجمة تلقائية" in translate_option:
    api_key = st.text_input("مفتاح API (OpenAI / DeepSeek / Grok)", type="password")

if uploaded_file and st.button("ابدأ الدبلجة", type="primary", use_container_width=True):
    with st.status("جاري المعالجة...", expanded=True) as status:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                
                # حفظ الفيديو
                input_video = tmp / uploaded_file.name
                with open(input_video, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 1. استخراج الصوت
                st.write("1️⃣ استخراج الصوت من الفيديو...")
                original_audio = tmp / "original.wav"
                video_duration = extract_audio(str(input_video), str(original_audio))

                # 2. Transcription
                st.write("2️⃣ تحويل الكلام إلى نص (Bambara ASR)...")
                result = asr_model.transcribe_audio(str(original_audio))
                original_text = result if isinstance(result, str) else result.get("text", str(result))
                st.write(f"**النص الأصلي:** {original_text[:300]}...")

                # 3. ترجمة
                bambara_text = original_text
                if "ترجمة تلقائية" in translate_option and api_key:
                    st.write("3️⃣ ترجمة إلى البامبارا...")
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)  # يعمل أيضاً مع DeepSeek و Grok إذا غيرت base_url
                    
                    prompt = f"""Translate the following text accurately into natural Bambara (Bamanankan).
Preserve tone, numbers, names and cultural context.
Return ONLY the Bambara translation:

{original_text}"""
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",  # أو deepseek-chat أو grok-2
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2
                    )
                    bambara_text = response.choices[0].message.content.strip()
                    لا تنقص منها شيء اجعلها تطبيق دبلجة فيديو مطابقة تستطيع قراءة ارقام والنص باللغة البامبارا بصوت صحيحة بالوقار والثبات جدا نقية من القرقرة والتشوش الصوت لا تنقص منها شيء اجعلها تستطيع قراءة ارقام والنص باللغة البامبارا بصوت صحيحة قوية احترافيته بالوقار والثبات جدا نقية من القرقرة والتشوش الصوت كترجمة meta video بالاستنساخ بصمة الصوتية 
