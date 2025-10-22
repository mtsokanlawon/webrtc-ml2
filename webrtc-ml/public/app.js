let localStream;
let roomId, peerId;
let isHost = false;
let audioRecorder;
let audioChunks = [];
let frameTimer = null;

let peerConnections = {};
let remoteStreams = {};
let isPolite = {};
let socket;
let hostPeerId = null;

const pendingCandidates = {};
const remoteAudioRecorders = {};

function log(msg, ...rest) {
  console.log(msg, ...rest);
  const el = document.getElementById('log');
  if (el) {
    el.textContent += new Date().toLocaleTimeString() + ' - ' + msg + "\n";
    el.scrollTop = el.scrollHeight;
  }
}

function updateStatus(status) {
  const statusEl = document.getElementById('connectionStatus');
  if (statusEl) statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

// ===== Audio upload queue (NEW) =====
let audioUploadQueue = [];
let audioUploading = false;
const AUDIO_QUEUE_MAX = 12; // cap to avoid unbounded memory growth

function enqueueAudioChunk(fd) {
  if (audioUploadQueue.length >= AUDIO_QUEUE_MAX) {
    console.warn("⚠️ Audio queue full — dropping oldest chunk");
    audioUploadQueue.shift();
  }
  audioUploadQueue.push(fd);
  processAudioQueue();
}

async function processAudioQueue() {
  if (audioUploading) return;
  audioUploading = true;
  while (audioUploadQueue.length > 0) {
    const fd = audioUploadQueue[0];
    try {
      const resp = await fetch('/ingest/audio', { method: 'POST', body: fd });
      if (!resp.ok) {
        log(`❌ Audio upload failed: ${resp.status} ${resp.statusText}`);
        // on server error drop and continue to avoid infinite loops
      } else {
        // log success (try to show file size if possible)
        try {
          const f = fd.get('file');
          const size = f && f.size ? f.size : 'unknown';
          log(`🎤 Uploaded audio chunk (${size} bytes)`);
        } catch (e) {
          log('🎤 Uploaded audio chunk');
        }
      }
      audioUploadQueue.shift();
    } catch (err) {
      log(`❌ Audio upload error: ${err.message}. Retrying in 500ms`);
      // exponential backoff could be added; for now simple wait then retry
      await new Promise((r) => setTimeout(r, 500));
    }
  }
  audioUploading = false;
}

async function flushAudioQueue(timeoutMs = 3000) {
  const start = Date.now();
  while (audioUploadQueue.length > 0 && (Date.now() - start) < timeoutMs) {
    await new Promise((r) => setTimeout(r, 100));
  }
  if (audioUploadQueue.length > 0) {
    console.warn("⚠️ flushAudioQueue timed out, dropping remaining chunks");
    audioUploadQueue = [];
  }
}

// ===== Socket.IO
function initSocket() {
  // ensure websockets are used (avoid long-polling fallbacks that add delay)
  socket = io(window.location.origin, { transports: ["websocket"] });

  socket.on('connect', () => { log('🔌 Connected'); updateStatus('connected'); });
  socket.on('disconnect', () => { log('❌ Disconnected'); updateStatus('disconnected'); });

  socket.on('host-info', ({ hostPeerId: hostId }) => {
    hostPeerId = hostId;
    updateHostBanner();
  });

  socket.on('peers', async (peers) => {
    for (const id of peers) {
      isPolite[id] = false;
      const pc = await createPeerConnection(id);
      await createOffer(pc, id);
    }
  });

  socket.on('peer-initiate', async ({ peerId: id }) => {
    isPolite[id] = true;
    await createPeerConnection(id);
  });

  socket.on('peers-updated', ({ peers = [], host }) => {
    document.getElementById('peerCount').textContent = peers.length;
    hostPeerId = host;
    updateHostBanner();
  });

  socket.on('peer-left', async ({ peerId: id }) => {
    log(`👋 Peer ${id} left`);
    await removePeerConnection(id);
  });

  socket.on('signal', async (msg) => {
    const remoteId = msg.from;
    const pc = peerConnections[remoteId] || await createPeerConnection(remoteId);
    if (!pendingCandidates[remoteId]) pendingCandidates[remoteId] = [];

    if (msg.sdp) {
      const desc = new RTCSessionDescription(msg.sdp);
      try {
        await pc.setRemoteDescription(desc);
        log(`✅ Applied SDP from ${remoteId}`);
        for (const c of pendingCandidates[remoteId]) {
          try { await pc.addIceCandidate(new RTCIceCandidate(c)); }
          catch(e){ log('❌ Failed to add queued candidate: ' + e); }
        }
        pendingCandidates[remoteId] = [];
        if (desc.type === 'offer') {
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          socket.emit('signal', { to: remoteId, sdp: answer });
          log(`📨 Sent answer to ${remoteId}`);
        }
      } catch(e) {
        log('❌ setRemoteDescription failed: ' + e);
      }
    } else if (msg.candidate) {
      if (!pc.remoteDescription) {
        pendingCandidates[remoteId].push(msg.candidate);
      } else {
        try { await pc.addIceCandidate(new RTCIceCandidate(msg.candidate)); }
        catch(e) { log('❌ ICE candidate error: ' + e); }
      }
    }
  });

  socket.on('engagement-event', ({ peerId: pId, eventType, description, timestamp }) => {
    // Only host sees all; peer sees own events
    if (isHost || pId === peerId) {
      log(`⚠️ Engagement: ${eventType} (${pId}) - ${description}`);
      showEngagementBanner(`${eventType} (${pId}): ${description}`);
    }
  });

  // socket.on('transcription-event', ({ transcripts}) => {
  //   // All users see transcription events
  //   log(`⚠️ Transcription: ${transcripts}`);
  //   showTranscriptBanner(`${transcripts}`);
  // });

  socket.on('transcription-event', ({ transcripts }) => {
    // backward-compat fallback: sometimes server sends batch arrays
    if (Array.isArray(transcripts)) {
      // append only latest item to the UI to avoid large concatenations
      const last = transcripts[transcripts.length - 1];
      if (last) {
        const short = `[${last.timestamp || last.timestamp_iso}] (${last.peerId || 'Unknown'}): ${last.text}`;
        log(`⚠️ Transcription: ${short}`);
        showTranscriptBanner(short);
      }
    } else {
      log(`⚠️ Transcription: ${transcripts}`);
      showTranscriptBanner(`${transcripts}`);
    }
  });

  // handle single-transcript emits (lightweight)
  socket.on('transcript', (t) => {
    // t is { roomId, peerId, text, final, timestamp, timestamp_iso }
    try {
      const short = `[${t.timestamp || t.timestamp_iso}] (${t.peerId||'Unknown'}): ${t.text}`;
      // minimal work: log and render banner quickly
      log(`🔉 ${short}`);
      // schedule UI update on next frame
      requestAnimationFrame(() => showTranscriptBanner(short));
    } catch (e) {
      console.error('❌ transcript handler error', e);
    }
  });

  socket.onAny((ev, ...args) => console.log('[socket event]', ev, args));
  socket.on('error', err => log('Socket error: ' + JSON.stringify(err)));
}

function updateHostBanner() {
  const banner = document.getElementById('hostBanner') || document.createElement('div');
  banner.id = 'hostBanner';
  banner.style.background = '#222';
  banner.style.color = '#fff';
  banner.style.padding = '5px 10px';
  banner.style.borderRadius = '8px';
  banner.style.marginBottom = '8px';
  banner.style.fontWeight = 'bold';
  if (peerId === hostPeerId) {
    banner.textContent = "You are the HOST (live feedback enabled)";
    isHost = true;
    document.getElementById('finalizeBtn').style.display = '';
  } else {
    banner.textContent = `Host: ${hostPeerId}`;
    isHost = false;
    document.getElementById('finalizeBtn').style.display = 'none';
  }
  const container = document.querySelector('.container');
  if (container && !document.getElementById('hostBanner')) {
    container.insertBefore(banner, container.firstChild);
  }
}

// ===== Show Feedback Banner
function showEngagementBanner(msg) {
  let banner = document.getElementById('engagementBanner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'engagementBanner';
    banner.style.position = 'fixed';
    banner.style.top = '10px';
    banner.style.left = '50%';
    banner.style.transform = 'translateX(-50%)';
    banner.style.background = '#ff9800';
    banner.style.color = '#222';
    banner.style.padding = '12px 28px';
    banner.style.borderRadius = '10px';
    banner.style.fontSize = '1.2rem';
    banner.style.zIndex = 9999;
    document.body.appendChild(banner);
  }
  banner.textContent = msg;
  banner.style.display = 'block';
  setTimeout(() => { banner.style.display = 'none'; }, 4000);
}

// ===== Show Transcript Banner
function showTranscriptBanner(msg) {
  let banner = document.getElementById('transcriptBanner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'transcriptBanner';
    banner.style.position = 'fixed';
    banner.style.bottom = '10px';
    banner.style.left = '50%';
    banner.style.transform = 'translateX(-50%)';
    banner.style.background = '#2196f3';
    banner.style.color = '#fff';
    banner.style.padding = '10px 24px';
    banner.style.borderRadius = '8px';
    banner.style.fontSize = '1rem';
    banner.style.maxWidth = '80%';
    banner.style.textAlign = 'center';
    banner.style.zIndex = 9999;
    document.body.appendChild(banner);
  }
  // use innerText (fast) and avoid recreating DOM nodes repeatedly
  banner.innerText = msg;
  banner.style.display = 'block';
  // shorter display so UI cycles faster
  setTimeout(() => { banner.style.display = 'none'; }, 3000);
}


// ===== Media
async function initLocalMedia() {
  try {
    localStream = await navigator.mediaDevices.getUserMedia({ video: { width:640,height:480 }, audio:true });
    document.getElementById('localVideo').srcObject = localStream;
    log('📹 Local media ready');
    return true;
  } catch(e) { alert('Please allow camera & mic.'); return false; }
}

// ===== Frame Ingest
function startFrameIngest(intervalMs=5000) {
  stopFrameIngest();
  const videoEl = document.getElementById('localVideo');
  frameTimer = setInterval(() => {
    if (!videoEl.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoEl.videoWidth; canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl,0,0,canvas.width,canvas.height);
    const ts = Date.now();
    ctx.fillStyle = 'white';
    ctx.font = '16px monospace';
    ctx.fillText(new Date(ts).toISOString(), 10, 22);

    const frameData = canvas.toDataURL('image/jpeg');
    fetch('/ingest/frame', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ roomId, peerId, frame: frameData, timestamp: ts }) });
  }, intervalMs);
}

function stopFrameIngest(){ if(frameTimer){ clearInterval(frameTimer); frameTimer=null; } }

// ===== Local audio recording
function startAudioRecording() {
  const audioTracks = localStream.getAudioTracks();
  if (audioTracks.length === 0) {
    log("❌ No local audio track available");
    return;
  }

  const audioStream = new MediaStream(audioTracks);
  audioRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm;codecs=opus' });

  audioRecorder.ondataavailable = async (e) => {
    console.log("Audio chunk available:", e.data.size);

    if (e.data && e.data.size > 0) {
      const blob = e.data;
      const fd = new FormData();
      fd.append('roomId', roomId);
      fd.append('peerId', peerId);
      fd.append('timestamp', Date.now());
      fd.append('file', blob, `audio_${peerId}_${Date.now()}.webm`);

      // enqueue instead of sending immediately; queue worker will await responses
      enqueueAudioChunk(fd);
    }
  };

  // start recording and emit data every 3s
  audioRecorder.start(3000);
  log('🎙️ Local audio recording started (chunked)');
}

function stopAudioRecording(){ 
  if(audioRecorder && audioRecorder.state !== 'inactive') audioRecorder.stop();
  audioRecorder = null; 
}

// ===== Remote audio recording
function startRemoteAudioRecording(id, track) {
  if (remoteAudioRecorders[id]) return;

  const recStream = new MediaStream([track]);
  const recorder = new MediaRecorder(recStream, { mimeType: 'audio/webm;codecs=opus' });

  recorder.ondataavailable = async (e) => {
    if (e.data && e.data.size > 0) {
      const blob = e.data;
      const fd = new FormData();
      fd.append('roomId', roomId);
      fd.append('peerId', id);
      fd.append('timestamp', Date.now());
      fd.append('file', blob, `audio_${id}_${Date.now()}.webm`);

      // enqueue remote chunk as well
      enqueueAudioChunk(fd);
      log(`🎤 Enqueued remote audio chunk from ${id} (${blob.size} bytes)`);
    }
  };

  recorder.start(3000); // every 3s
  remoteAudioRecorders[id] = { recorder };
  log(`🎙️ Remote audio recording started for ${id}`);
}

// ===== WebRTC
async function createPeerConnection(remoteId){
  if(peerConnections[remoteId]) return peerConnections[remoteId];
  const pc = new RTCPeerConnection({
    iceServers:[
      {urls:'stun:stun.l.google.com:19302'},
      {urls:'stun:stun1.l.google.com:19302'},
      {urls:'turn:your.turn.server:3478', username:'user', credential:'pass'}
    ]
  });
  peerConnections[remoteId]=pc;
  remoteStreams[remoteId]=new MediaStream();
  pendingCandidates[remoteId] = [];

  if(localStream) localStream.getTracks().forEach(t=>{
    if(!pc.getSenders().find(s=>s.track?.kind===t.kind)) pc.addTrack(t,localStream);
  });

  const receivedTracks=new Set();
  pc.ontrack=event=>{
    const stream=event.streams[0];
    if(stream) stream.getTracks().forEach(t=>{
      if(!receivedTracks.has(t.id)){
        receivedTracks.add(t.id);
        remoteStreams[remoteId].addTrack(t);
        if(t.kind==='audio') startRemoteAudioRecording(remoteId,t);
      }
    });
    updateRemoteVideoElement(remoteId);
  };

  pc.onicecandidate=event=>{
    if(event.candidate) socket?.emit('signal',{to:remoteId,candidate:event.candidate});
  };

  pc.oniceconnectionstatechange=()=>{
    const state=pc.iceConnectionState;
    log(`ICE ${remoteId}: ${state}`);
    if(state==='failed'||state==='disconnected'){ setTimeout(()=>createOffer(pc,remoteId),3000); }
  };

  pc.onnegotiationneeded=async ()=>{
    if(isPolite[remoteId]===false) await createOffer(pc,remoteId);
  };

  return pc;
}

async function createOffer(pc,remoteId){
  if(!pc||pc.signalingState!=='stable') return;
  const offer=await pc.createOffer();
  await pc.setLocalDescription(offer);
  socket.emit('signal',{to:remoteId,sdp:offer});
}

// ===== Remote video
function updateRemoteVideoElement(remoteId){
  const containerId = `remoteContainer_${remoteId}`;
  let container = document.getElementById(containerId);
  if (!container) {
    container = document.createElement('div');
    container.id = containerId;
    container.style.marginBottom = '10px';
    container.style.textAlign = 'center';

    const video = document.createElement('video');
    video.id = `remoteVideo_${remoteId}`;
    video.autoplay = true;
    video.playsInline = true;
    video.style.border = '3px solid #FF9800';

    const label = document.createElement('div');
    label.textContent = remoteId;
    label.style.color = 'white';
    label.style.background = '#333';
    label.style.padding = '2px 6px';
    label.style.borderRadius = '6px';
    label.style.marginTop = '4px';

    container.appendChild(video);
    container.appendChild(label);
    document.getElementById('remoteVideos').appendChild(container);
  }

  const videoEl = document.getElementById(`remoteVideo_${remoteId}`);
  if (videoEl.srcObject !== remoteStreams[remoteId]) {
    videoEl.srcObject = remoteStreams[remoteId];
  }
}

// ===== Remove peer
async function removePeerConnection(remoteId){
  if (remoteAudioRecorders[remoteId]) remoteAudioRecorders[remoteId].recorder.stop();
  if (peerConnections[remoteId]) peerConnections[remoteId].close();
  delete peerConnections[remoteId];
  delete remoteStreams[remoteId];
  delete isPolite[remoteId];
  delete pendingCandidates[remoteId];

  const container = document.getElementById(`remoteContainer_${remoteId}`);
  if (container) container.remove();
}

// ===== Join/Leave
async function joinRoom(){
  roomId = document.getElementById('roomId').value || 'demo-room';
  peerId = document.getElementById('peerName').value || 'Guest-' + Math.random().toString(36).substr(2, 6);
  // Host logic: if first in room, or if name contains [host], set host
  isHost = false;
  if (!window.joinedOnce && (!hostPeerId || peerId.toLowerCase().includes("host"))) {
    isHost = true;
  }
  window.joinedOnce = true;

  if (!await initLocalMedia()) return;

  initSocket();
  // ensure we send the join after socket connected (avoid race)
  if (socket && socket.connected) {
    socket.emit('join', { roomId, peerId, isHost });
  } else if (socket) {
    socket.once('connect', () => socket.emit('join', { roomId, peerId, isHost }));
  }

  startFrameIngest();
  startAudioRecording();

  document.getElementById('joinBtn').disabled = true;
  document.getElementById('leaveBtn').disabled = false;
  updateStatus('connecting');
}

async function leaveRoom(){
  stopFrameIngest();
  stopAudioRecording();

  for (const id of Object.keys(peerConnections)) {
    await removePeerConnection(id);
  }

  if (localStream) localStream.getTracks().forEach(t => t.stop());

  // flush queued uploads (wait up to 3s) to avoid abrupt client aborts
  try {
    await flushAudioQueue(3000);
  } catch (e) {
    console.warn("⚠️ flushAudioQueue error:", e);
  }

  socket.emit('leave', { roomId, peerId });
  socket.disconnect();

  document.getElementById('joinBtn').disabled = false;
  document.getElementById('leaveBtn').disabled = true;
  updateStatus('disconnected');
  document.getElementById('localVideo').srcObject = null;
}

// ===== Host: Finalize Room and Download Report
async function finalizeRoom() {
  if (!isHost) return;
  log('⏳ Generating report...');
  const btn = document.getElementById('finalizeBtn');
  btn.disabled = true;
  try {
    const resp = await fetch(`/finalize/${roomId}`, { method: 'POST' });
    // If backend returned JSON error
    if (resp.headers.get("content-type")?.includes("application/json")) {
      const err = await resp.json();
      alert("Error: " + err.error);
      return;
    }

    // Otherwise it’s a PDF file
    const blob = await resp.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${roomId}.pdf`;
    a.click();
    log('✅ Report downloaded');

    window.URL.revokeObjectURL(url);
  } catch (e) {
    alert("Request failed: " + e.message);
  }

  btn.disabled = false;
}

async function finalizeAndDownload(roomId) {
  try {
    const resp = await fetch(`/finalize/${roomId}`, { method: "POST" });

    // If backend returned JSON error
    if (resp.headers.get("content-type")?.includes("application/json")) {
      const err = await resp.json();
      alert("Error: " + err.error);
      return;
    }

    // Otherwise it’s a PDF file
    const blob = await resp.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${roomId}.pdf`;
    a.click();

    window.URL.revokeObjectURL(url);
  } catch (e) {
    alert("Request failed: " + e.message);
  }
}


// ===== Bind buttons
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('joinBtn').addEventListener('click', joinRoom);
  document.getElementById('leaveBtn').addEventListener('click', leaveRoom);
  document.getElementById('finalizeBtn').addEventListener('click', finalizeRoom);
});