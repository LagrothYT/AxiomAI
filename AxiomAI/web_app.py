import json
import os
import re
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

HOST = "127.0.0.1"
PORT = 7860

chat_engine = None
chat_lock = threading.RLock()
video_jobs = {}
video_lock = threading.Lock()


def json_response(handler, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler, text, status=200, content_type="text/html; charset=utf-8"):
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


def safe_name(text):
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower()).strip("_")[:36]
    return base or "video"


def get_chat_engine():
    global chat_engine
    with chat_lock:
        if chat_engine is None:
            from chat import AxiomChatEngine
            chat_engine = AxiomChatEngine()
        return chat_engine


def run_video_job(job_id, prompt):
    from video_generation.generate import generate_video

    def progress(phase, detail="", current=None, total=None, elapsed=None):
        with video_lock:
            job = video_jobs.get(job_id)
            if not job:
                return
            job["phase"] = str(phase)
            job["detail"] = str(detail or "")
            job["current"] = current
            job["total"] = total
            job["elapsed"] = elapsed
            job["updated_at"] = time.time()

    with video_lock:
        video_jobs[job_id].update({
            "status": "running",
            "phase": "Starting",
            "detail": "Preparing video generation",
        })

    try:
        output_name = f"web_{time.strftime('%Y%m%d_%H%M%S')}_{safe_name(prompt)}.mp4"
        path = generate_video(prompt, output_name=output_name, progress_callback=progress)
        with video_lock:
            video_jobs[job_id].update({
                "status": "done",
                "phase": "Done",
                "detail": "Video saved",
                "path": os.path.abspath(path),
                "url": "/media/video/" + os.path.basename(path),
                "updated_at": time.time(),
            })
    except Exception as exc:
        with video_lock:
            video_jobs[job_id].update({
                "status": "error",
                "phase": "Error",
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "updated_at": time.time(),
            })


def list_files(folder, suffixes):
    if not os.path.isdir(folder):
        return []
    rows = []
    for name in os.listdir(folder):
        if not name.lower().endswith(suffixes):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            rows.append({
                "name": name,
                "size": os.path.getsize(path),
                "mtime": os.path.getmtime(path),
            })
    rows.sort(key=lambda item: item["mtime"], reverse=True)
    return rows


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AxiomAI Web</title>
  <style>
    :root { color-scheme: dark; --bg:#080c11; --panel:#111821; --panel2:#0d131b; --line:#243143; --text:#edf3fb; --muted:#91a0b5; --blue:#2f6df6; --bad:#ff6b6b; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Segoe UI, Arial, sans-serif; }
    header { padding:18px 22px; border-bottom:1px solid var(--line); background:#0d1219; display:flex; justify-content:space-between; align-items:center; }
    h1 { margin:0; font-size:22px; }
    .muted { color:var(--muted); }
    main { padding:18px; max-width:1200px; margin:0 auto; }
    nav { display:flex; gap:8px; margin-bottom:14px; }
    button { background:#1c283a; color:var(--text); border:1px solid #32445d; border-radius:8px; padding:10px 14px; font-weight:650; cursor:pointer; }
    button.primary { background:var(--blue); border-color:#6ea0ff; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .tab { display:none; }
    .tab.active { display:block; }
    .grid { display:grid; grid-template-columns:1fr 320px; gap:14px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
    .chat-log { min-height:440px; max-height:58vh; overflow:auto; display:flex; flex-direction:column; gap:10px; background:#080d13; border:1px solid #1e2a39; border-radius:8px; padding:14px; }
    .msg { max-width:78%; padding:10px 12px; border-radius:12px; line-height:1.45; white-space:pre-wrap; }
    .human { align-self:flex-end; background:#2563eb; }
    .gpt { align-self:flex-start; background:#141f2e; border:1px solid #293d59; }
    .system { align-self:center; background:#211d2c; border:1px solid #4d4268; color:#d8c9ff; }
    textarea, input { width:100%; background:#080d13; color:var(--text); border:1px solid #2b3a4d; border-radius:8px; padding:11px; font:inherit; }
    textarea { resize:vertical; min-height:90px; }
    .row { display:flex; gap:10px; align-items:center; }
    .row > * { flex:1; }
    label { display:block; font-size:12px; color:var(--muted); margin:10px 0 6px; }
    .status { padding:10px; border:1px solid #26384f; border-radius:8px; background:var(--panel2); margin-bottom:10px; }
    .thinking { min-height:170px; display:grid; place-items:center; background:#070b10; border:1px solid #1f2c3c; border-radius:8px; overflow:hidden; }
    .latent { display:grid; grid-template-columns:repeat(16, 10px); gap:4px; }
    .latent span { width:10px; height:10px; border-radius:3px; background:#1b2a3e; animation:pulse 1.4s infinite ease-in-out; }
    @keyframes pulse { 0%,100%{opacity:.25; transform:scale(.85)} 50%{opacity:1; transform:scale(1.1)} }
    progress { width:100%; height:18px; }
    video { width:100%; max-height:420px; background:#000; border-radius:8px; border:1px solid var(--line); }
    .library { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px,1fr)); gap:12px; }
    .file { background:var(--panel2); border:1px solid #243143; border-radius:8px; padding:10px; overflow:hidden; }
    a { color:#9ec5ff; }
    pre { background:#05080d; border:1px solid #1f2c3c; border-radius:8px; padding:10px; overflow:auto; max-height:340px; color:#cbd5e1; }
    @media (max-width:850px) { .grid { grid-template-columns:1fr; } .msg { max-width:95%; } }
  </style>
</head>
<body>
<header>
  <div><h1>AxiomAI</h1><div class="muted">basic web chat + video</div></div>
  <div id="topStatus" class="muted">local</div>
</header>
<main>
  <nav>
    <button onclick="showTab('chat')" id="chatBtn" class="primary">Chat</button>
    <button onclick="showTab('video')" id="videoBtn">Video</button>
    <button onclick="showTab('library')" id="libraryBtn">Library</button>
    <button onclick="showTab('debug')" id="debugBtn">Debug</button>
  </nav>

  <section id="chat" class="tab active">
    <div class="grid">
      <div class="panel">
        <div id="chatLog" class="chat-log"><div class="msg system">Ready.</div></div>
        <label>Message</label>
        <textarea id="chatInput" placeholder="Message Axiom..."></textarea>
        <div class="row" style="margin-top:10px">
          <button onclick="newChat()">New Chat</button>
          <button class="primary" onclick="sendChat()">Send</button>
        </div>
      </div>
      <aside class="panel">
        <div class="status"><b>Model</b><div id="modelStatus" class="muted">checking...</div></div>
        <label>Temperature</label><input id="temperature" type="number" min="0.01" max="2" step="0.05" value="0.8">
        <label>Top-K</label><input id="topK" type="number" min="0" max="500" step="1" value="40">
        <label>Top-P</label><input id="topP" type="number" min="0" max="1" step="0.05" value="0.9">
        <label>Repeat Penalty</label><input id="repPenalty" type="number" min="1" max="2" step="0.05" value="1.15">
        <label>Max Tokens</label><input id="maxTokens" type="number" min="1" max="512" step="1" value="200">
      </aside>
    </div>
  </section>

  <section id="video" class="tab">
    <div class="grid">
      <div class="panel">
        <label>Prompt</label>
        <textarea id="videoPrompt" placeholder="a dog drinking water"></textarea>
        <div class="row" style="margin-top:10px"><button class="primary" onclick="startVideo()">Generate Video</button><button onclick="refreshLibrary()">Refresh Library</button></div>
        <h3>Thinking</h3>
        <div class="thinking"><div id="latentGrid" class="latent"></div></div>
      </div>
      <aside class="panel">
        <div class="status"><b id="videoPhase">Ready</b><div id="videoDetail" class="muted">No video running.</div></div>
        <progress id="videoProgress" max="100" value="0"></progress>
        <p><a id="videoLink" href="#" target="_blank" style="display:none">Open video file</a></p>
        <video id="videoPlayer" controls style="display:none"></video>
      </aside>
    </div>
  </section>

  <section id="library" class="tab">
    <div class="panel">
      <h3>Generated Videos</h3>
      <div id="videoLibrary" class="library"></div>
    </div>
  </section>

  <section id="debug" class="tab">
    <div class="panel">
      <button onclick="loadDebug()">Refresh Debug</button>
      <pre id="debugBox">Debug info appears here.</pre>
    </div>
  </section>
</main>
<script>
let history = [];
let activeVideoJob = null;

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(name).classList.add('active');
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('primary'));
  document.getElementById(name + 'Btn').classList.add('primary');
  if (name === 'library') refreshLibrary();
}

function addMsg(role, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + role;
  el.textContent = text;
  const log = document.getElementById('chatLog');
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function newChat() {
  history = [];
  document.getElementById('chatLog').innerHTML = '<div class="msg system">New chat.</div>';
}

function settings() {
  return {
    temperature: Number(document.getElementById('temperature').value),
    top_k: Number(document.getElementById('topK').value),
    top_p: Number(document.getElementById('topP').value),
    repetition_penalty: Number(document.getElementById('repPenalty').value),
    max_gen_length: Number(document.getElementById('maxTokens').value)
  };
}

async function sendChat() {
  const box = document.getElementById('chatInput');
  const text = box.value.trim();
  if (!text) return;
  box.value = '';
  history.push({role:'human', value:text});
  addMsg('human', text);
  const pending = addMsg('gpt', 'Thinking...');
  try {
    const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({history, settings:settings()})});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'chat failed');
    pending.textContent = data.response || '(empty response)';
    history.push({role:'gpt', value:data.response || ''});
    document.getElementById('modelStatus').textContent = `${data.source || 'model'} | PPL ${Number(data.ppl || 0).toFixed(1)} | context left ${data.context_left}`;
  } catch (err) {
    pending.className = 'msg system';
    pending.textContent = 'Chat failed: ' + err.message;
  }
}

function buildLatentGrid() {
  const grid = document.getElementById('latentGrid');
  grid.innerHTML = '';
  for (let i = 0; i < 128; i++) {
    const s = document.createElement('span');
    s.style.animationDelay = ((i % 19) * 0.04) + 's';
    grid.appendChild(s);
  }
}

async function startVideo() {
  const prompt = document.getElementById('videoPrompt').value.trim();
  if (!prompt) return;
  document.getElementById('videoPlayer').style.display = 'none';
  document.getElementById('videoLink').style.display = 'none';
  const res = await fetch('/api/video/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({prompt})});
  const data = await res.json();
  if (!res.ok) {
    document.getElementById('videoDetail').textContent = data.error || 'video failed';
    return;
  }
  activeVideoJob = data.job_id;
  pollVideo();
}

async function pollVideo() {
  if (!activeVideoJob) return;
  const res = await fetch('/api/video/status?id=' + encodeURIComponent(activeVideoJob));
  const data = await res.json();
  document.getElementById('videoPhase').textContent = data.phase || data.status;
  document.getElementById('videoDetail').textContent = data.detail || '';
  if (data.total && data.current !== null && data.current !== undefined) {
    document.getElementById('videoProgress').value = Math.round((data.current / data.total) * 100);
  }
  if (data.status === 'done') {
    activeVideoJob = null;
    const player = document.getElementById('videoPlayer');
    const link = document.getElementById('videoLink');
    player.src = data.url;
    player.style.display = 'block';
    link.href = data.url;
    link.style.display = 'inline';
    refreshLibrary();
  } else if (data.status === 'error') {
    activeVideoJob = null;
    document.getElementById('videoProgress').value = 0;
  } else {
    setTimeout(pollVideo, 800);
  }
}

async function refreshLibrary() {
  const res = await fetch('/api/library');
  const data = await res.json();
  const box = document.getElementById('videoLibrary');
  box.innerHTML = '';
  for (const file of data.videos || []) {
    const card = document.createElement('div');
    card.className = 'file';
    card.innerHTML = `<b>${file.name}</b><div class="muted">${Math.round(file.size/1024)} KB</div><video controls src="/media/video/${encodeURIComponent(file.name)}"></video>`;
    box.appendChild(card);
  }
}

async function loadDebug() {
  const res = await fetch('/api/debug');
  document.getElementById('debugBox').textContent = JSON.stringify(await res.json(), null, 2);
}

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('modelStatus').textContent = data.chat ? `${data.chat.source} | ${data.chat.max_seq_len} ctx` : 'No chat model';
    document.getElementById('topStatus').textContent = data.cwd;
  } catch (e) {
    document.getElementById('modelStatus').textContent = 'status failed';
  }
}

buildLatentGrid();
loadStatus();
refreshLibrary();
</script>
</body>
</html>"""


class AxiomWebHandler(BaseHTTPRequestHandler):
    server_version = "AxiomAIWeb/0.1"

    def log_message(self, fmt, *args):
        print("[web] " + fmt % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            return text_response(self, HTML)
        if path == "/api/status":
            try:
                engine = get_chat_engine()
                chat_status = engine.status()
            except Exception as exc:
                chat_status = {"error": str(exc)}
            return json_response(self, {"cwd": PROJECT_ROOT, "chat": chat_status})
        if path == "/api/video/status":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            with video_lock:
                job = dict(video_jobs.get(job_id, {"status": "missing", "phase": "Missing", "detail": "Unknown job"}))
            return json_response(self, job)
        if path == "/api/library":
            videos = list_files(os.path.join(PROJECT_ROOT, "out_video"), (".mp4", ".avi", ".mov"))
            previews = list_files(os.path.join(PROJECT_ROOT, "model", "video_model", "previews"), (".png", ".jpg", ".jpeg"))
            return json_response(self, {"videos": videos, "previews": previews})
        if path == "/api/debug":
            with video_lock:
                jobs = dict(video_jobs)
            return json_response(self, {"jobs": jobs, "cwd": PROJECT_ROOT})
        if path.startswith("/media/video/"):
            return self.serve_file(os.path.join(PROJECT_ROOT, "out_video"), unquote(path.rsplit("/", 1)[-1]), "video/mp4")
        return text_response(self, "Not found", status=404, content_type="text/plain; charset=utf-8")

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = read_json(self)
            if parsed.path == "/api/chat":
                history = payload.get("history", [])
                settings = payload.get("settings", {})
                with chat_lock:
                    engine = get_chat_engine()
                    result = engine.generate(history, settings)
                return json_response(self, result)
            if parsed.path == "/api/video/start":
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    return json_response(self, {"error": "Missing prompt"}, status=400)
                job_id = f"job_{int(time.time() * 1000)}"
                with video_lock:
                    video_jobs[job_id] = {
                        "id": job_id,
                        "prompt": prompt,
                        "status": "queued",
                        "phase": "Queued",
                        "detail": "Waiting to start",
                        "current": None,
                        "total": None,
                        "created_at": time.time(),
                    }
                thread = threading.Thread(target=run_video_job, args=(job_id, prompt), daemon=True)
                thread.start()
                return json_response(self, {"job_id": job_id})
            return json_response(self, {"error": "Not found"}, status=404)
        except Exception as exc:
            return json_response(self, {"error": str(exc), "traceback": traceback.format_exc()}, status=500)

    def serve_file(self, folder, raw_name, content_type):
        name = os.path.basename(raw_name)
        path = os.path.abspath(os.path.join(folder, name))
        root = os.path.abspath(folder)
        if not path.startswith(root) or not os.path.isfile(path):
            return text_response(self, "Not found", status=404, content_type="text/plain; charset=utf-8")
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def launch_web(open_browser=True, host=HOST, port=PORT):
    server = ThreadingHTTPServer((host, int(port)), AxiomWebHandler)
    url = f"http://{host}:{port}"
    print(f"AxiomAI web UI running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    launch_web()
