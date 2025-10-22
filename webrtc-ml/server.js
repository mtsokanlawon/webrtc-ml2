const express = require("express");
const http = require("http");
const path = require("path");
const fs = require("fs");
const multer = require("multer");
const { exec } = require("child_process");
const { Server } = require("socket.io");
const axios = require("axios");
const { spawn } = require("child_process");
const ffmpegSessions = {}; // key: roomId+peerId → ffmpeg process
// Track active sessions locally in Node (per room)
const sessions = {};
const app = express();
const server = http.createServer(app);
// const io = new Server(server);
const io = new Server(server, {
  transports: ["websocket"], // prefer websockets (avoid polling over ngrok)
  pingInterval: 20000,
  pingTimeout: 5000,
  cors: { origin: "*" }
});
const upload = multer({ limits: { fileSize: 6 * 1024 * 1024 } }); // 6 MB max chunks

// simple in-memory transcript store (room and per-room_peer)
const transcripts = {}; // key -> [{ peerId, transcript, final, ts }]

// parse JSON from analyzer
app.use(express.json({ limit: '1mb' }));

// analyzer -> Node: store transcript and notify clients
app.post('/analyze/realtime', (req, res) => {
  const { roomId, peerId, transcript, final, ts } = req.body || {};
  if (!roomId || !peerId || typeof transcript !== 'string') {
    return res.status(400).json({ error: 'roomId, peerId and transcript required' });
  }

  const roomKey = roomId;
  const peerKey = `${roomId}_${peerId}`;
  const entry = {
    peerId,
    transcript,
    final: !!final,
    ts: ts || new Date().toISOString(),
  };

  transcripts[roomKey] = transcripts[roomKey] || [];
  transcripts[peerKey] = transcripts[peerKey] || [];
  transcripts[roomKey].push(entry);
  transcripts[peerKey].push(entry);

  // notify connected clients in the room immediately
  try {
    emitToRoom(roomId, 'transcript', { roomId, ...entry });
  } catch (err) {
    console.warn('emit transcript error', err && err.message);
  }

  return res.json({ ok: true });
});

// existing frontend polling endpoint (or add if missing)
app.get('/get_transcripts/:room', (req, res) => {
  const key = req.params.room;
  return res.json(transcripts[key] || []);
});

app.use(express.static(path.join(__dirname, "public")));
app.use(express.json({ limit: "50mb" }));
app.use('/data', express.static(path.join(__dirname, 'data')));

let audioBuffer = Buffer.alloc(0);
let rooms = {}; // { [roomId]: Set(peerId) }
let hosts = {}; // { [roomId]: peerId }

function stopFFmpeg(sessionKey) {
    const ffmpeg = ffmpegSessions[sessionKey];
    if (!ffmpeg) return;

    try {
        ffmpeg.stdin.end(); // signal no more input
        ffmpeg.kill();     // force kill if needed
    } catch (err) {
        console.error("❌ Error stopping FFmpeg:", err.message);
    }

    delete ffmpegSessions[sessionKey];
}

// helper: stop every per-peer session for a room (ffmpeg + python session)
function stopAllRoomSessions(roomId) {
    // stop ffmpeg sessions whose key starts with `${roomId}_`
    for (const key of Object.keys(ffmpegSessions)) {
        if (key.startsWith(`${roomId}_`)) {
            try {
                stopFFmpeg(key);
                console.log(`🛑 Stopped ffmpeg session ${key}`);
            } catch (err) {
                console.error(`❌ Error stopping ffmpeg ${key}:`, err.message);
            }
            axios.post(`http://localhost:5000/stop_session/${key}`)
              .then(() => console.log(`🛑 Notified Python to stop session ${key}`))
              .catch(err => console.error(`❌ Failed to stop python session ${key}:`, err.message));
        }
    }

    // defensive: also request a room-level stop (compat fallback)
    axios.post(`http://localhost:5000/stop_session/${roomId}`)
      .then(() => console.log(`🛑 Notified Python to stop room session ${roomId}`))
      .catch(() => { /* ignore if endpoint not used */ });
}

// ========== Ingest Frame, Analyze Real-Time ==========
app.post("/ingest/frame", async (req, res) => {
  try {
    const { roomId, peerId, frame, timestamp } = req.body;
    if (!roomId || !peerId || !frame) return res.status(400).json({ error: "Missing required fields" });

    const dir = path.join(__dirname, "data", roomId, peerId, "frames");
    fs.mkdirSync(dir, { recursive: true });

    const buf = Buffer.from(frame.split(",")[1], "base64");
    const fname = path.join(dir, `frame_${timestamp}.jpg`);
    fs.writeFileSync(fname, buf);

    // Real-time analyzer call
    try {
      const analyzeUrl = "http://localhost:5000/analyze/realtime";
      const analyzeResp = await axios.post(analyzeUrl, {
        roomId, peerId, frame, timestamp
      });
      const { eventType, description, timestamp: ts } = analyzeResp.data || {};
      if (eventType) {
        // Find host for this room
        const hostPeerId = hosts[roomId];
        if (hostPeerId) {
          for (let [sid, s] of io.of("/").sockets) {
            if (s.roomId === roomId && s.peerId === hostPeerId) {
              s.emit("engagement-event", { peerId, eventType, description, timestamp: ts });
            }
          }
        }
      }
    } catch (e) {
      console.error("❌ Real-time analysis error", e.message);
    }

    console.log(`🖼️ Saved frame: ${fname}`);
    res.json({ status: "ok" });
  } catch (e) {
    console.error("❌ Frame ingest error", e);
    res.status(500).json({ error: "Frame ingest failed" });
  }
});

// ========== Ingest Audio ==========
app.post("/ingest/audio", upload.single("file"), async (req, res) => {
  const { roomId, peerId, timestamp } = req.body;
  // detect client aborts early (prevents raw-body noisy stack traces)
  req.on && req.on("aborted", () => {
    console.warn(`⚠️ Upload aborted by client for room=${roomId} peer=${peerId}`);
  });

  if (!roomId || !peerId || !req.file) {
    return res.status(400).json({ error: "Missing required fields" });
  }

  try {
    // ensure audio dir exists (optional, only if you still want raw saving)
    const dir = path.join(__dirname, "data", roomId, peerId, "audio");
    fs.mkdirSync(dir, { recursive: true });

    // save raw .webm chunk (optional for debugging)
    const webmPath = path.join(dir, `audio_${timestamp}.webm`);
    fs.writeFileSync(webmPath, req.file.buffer);
    console.log(`🎙️ Saved raw audio: ${webmPath}`);

    // send to Python (defensive: timeout + swallow network errors)
    const chunkB64 = req.file.buffer.toString("base64");
    try {
      await axios.post(`http://localhost:5000/transcribe/realtime/${roomId}`, {
        audio: chunkB64,
        peerId: peerId,
      }, { timeout: 10_000 });
    } catch (err) {
      console.error("❌ Real-time transcription proxy error:", err.message);
    }

    res.json({ status: "ok" });
  } catch (e) {
    console.error("❌ Ingest audio handler error:", e && e.message ? e.message : e);
    // if the client aborted mid-upload, respond gracefully
    if (res.headersSent) return;
    res.status(500).json({ error: "Audio ingest failed" });
  }
});

// Finalize and auto-download report
app.post("/finalize/:roomId", async (req, res) => {
  try {
    const { roomId } = req.params;

    // Step 1: Finalize on Python
    // await axios.post(`http://localhost:5000/finalize/${roomId}`);
    const finalizeResp = await axios.post(`http://localhost:5000/finalize/${roomId}`);
    if (finalizeResp.data.error) {
      return res.status(400).json(finalizeResp.data);
    }

    // Step 2: Fetch PDF from Python
    const pdfResp = await axios.get(`http://localhost:5000/download/${roomId}`, {
      responseType: "arraybuffer"
    });
    

    // Step 3: Stream PDF back to client
    res.setHeader("Content-Type", "application/pdf");
    res.setHeader("Content-Disposition", `attachment; filename="report_${roomId}.pdf"`);
    res.send(pdfResp.data);

  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ========== Peer Management ==========
io.on("connection", (socket) => {
  socket.on("join", ({ roomId, peerId, isHost }) => {
    socket.join(roomId);
    socket.roomId = roomId;
    socket.peerId = peerId;
    // send recent transcripts to the newly joined socket so they catch up
    (async () => {
      try {
        const peerIds = rooms[roomId] ? [...rooms[roomId]] : [];
        const sessionKeys = [roomId, ...peerIds.map(p => `${roomId}_${p}`)];
        const all = [];
        for (const sk of sessionKeys) {
          try {
            const r = await axios.get(`http://localhost:5000/get_transcripts/${sk}`, { timeout: 3000 });
            const ts = r.data.transcripts || [];
            all.push(...ts);
          } catch (e) {
            // ignore per-session fetch failures
          }
        }
        if (all.length) {
          // emit only to the joining socket so it gets history immediately
          socket.emit('transcription-event', { transcripts: all, sessionKey: roomId });
        }
      } catch (e) {
        console.warn('Failed to fetch transcripts for join:', e && e.message);
      }
    })();

    if (!rooms[roomId]) rooms[roomId] = new Set();
    rooms[roomId].add(peerId);

    // Host logic: first peer is host
    if (!(roomId in hosts)) {
      hosts[roomId] = peerId; // first peer in room
    }

    // Start transcription session ONCE per room
    if (!sessions[roomId]) {   // 👈 only if no active session
      try {
        axios.post(`http://localhost:5000/start_session/${roomId}`);
        sessions[roomId] = true; // mark as active
        console.log(`✅ Transcription session started for ${roomId}`);
      } catch (err) {
        console.error("❌ Failed to start transcription session:", err.message);
      }
    }
    // axios.post(`http://localhost:5000/start_session/${roomId}`)
    // .then(() => console.log(`✅ Transcription session started for ${roomId}`))
    // .catch(err => console.error("❌ Failed to start transcription session:", err.message));

    const others = [...rooms[roomId]].filter((id) => id !== peerId);
    socket.emit("peers", others);

    for (let [sid, s] of io.of("/").sockets) {
      if (s.roomId === roomId && s.peerId !== peerId) {
        s.emit("peer-initiate", { peerId });
      }
    }

    io.to(roomId).emit("peers-updated", {
      peers: [...rooms[roomId]],
      count: rooms[roomId].size,
      host: hosts[roomId]
    });

    socket.emit("host-info", { hostPeerId: hosts[roomId] });

    console.log(`👥 ${peerId} joined ${roomId}${hosts[roomId] === peerId ? " (host)" : ""}`);
  });

  socket.on("signal", (msg) => {
    if (socket.roomId) {
      socket.to(socket.roomId).emit("signal", { ...msg, from: socket.peerId });
    }
  });

  socket.on("leave", ({ roomId, peerId }) => {
    if (rooms[roomId]) {
      rooms[roomId].delete(peerId);
      if (rooms[roomId].size === 0) {
        delete rooms[roomId];
        delete hosts[roomId];

    //     // Stop AssemblyAI session
    //     axios.post(`http://localhost:5000/stop_session/${roomId}`)
    //       .then(() => console.log(`🛑 Session stopped for ${roomId}`))
    //       .catch(err => console.error("❌ Failed to stop session:", err.message));
    // const sessionKey = `${roomId}_${peerId}`;
    // stopFFmpeg(sessionKey);
        // stop all per-peer sessions for this room
        stopAllRoomSessions(roomId);
      } else if (hosts[roomId] === peerId) {
        // Reassign host
        hosts[roomId] = [...rooms[roomId]][0];
      }
    }
    socket.leave(roomId);
    socket.to(roomId).emit("peer-left", { peerId });
    io.to(roomId).emit("peers-updated", {
      peers: rooms[roomId] ? [...rooms[roomId]] : [],
      count: rooms[roomId] ? rooms[roomId].size : 0,
      host: hosts[roomId]
    });
    console.log(`👋 ${peerId} left ${roomId}`);
  });

  socket.on("disconnect", () => {
    const { roomId, peerId } = socket;
    if (roomId && peerId && rooms[roomId]) {
      rooms[roomId].delete(peerId);
      if (rooms[roomId].size === 0) {
        delete rooms[roomId];
        delete hosts[roomId];

      //   // Stop AssemblyAI session
      //   axios.post(`http://localhost:5000/stop_session/${roomId}`)
      //     .then(() => console.log(`🛑 Session stopped for ${roomId}`))
      //     .catch(err => console.error("❌ Failed to stop session:", err.message));

      // const sessionKey = `${roomId}_${peerId}`;
      // stopFFmpeg(sessionKey);
        // stop all per-peer sessions for this room
        stopAllRoomSessions(roomId);
      } else if (hosts[roomId] === peerId) {
        hosts[roomId] = [...rooms[roomId]][0];
      }
      socket.to(roomId).emit("peer-left", { peerId });
      io.to(roomId).emit("peers-updated", {
        peers: rooms[roomId] ? [...rooms[roomId]] : [],
        count: rooms[roomId] ? rooms[roomId].size : 0,
        host: hosts[roomId]
      });
      console.log(`❌ ${peerId} disconnected`);
    }
  });
});

// ========== Poll Transcripts & Emit to all Users ==========
// Track sent transcripts per sessionKey (room or room_peer)
const sentTranscripts = {}; // { sessionKey: Set<string> }

// Poll faster and emit individual events to reduce client processing time
setInterval(async () => {
  for (const roomId of Object.keys(rooms)) {
    const peerIds = [...rooms[roomId]];
    const sessionKeys = [roomId, ...peerIds.map(p => `${roomId}_${p}`)];
    for (const sessionKey of sessionKeys) {
      try {
        const res = await axios.get(`http://localhost:5000/get_transcripts/${sessionKey}`, { timeout: 3000 });
        const transcripts = res.data.transcripts || [];
        if (!sentTranscripts[sessionKey]) sentTranscripts[sessionKey] = new Set();
        for (const t of transcripts) {
          const key = t.id || t.timestamp_iso || `${t.timestamp}-${t.peerId}-${t.text}`;
          if (sentTranscripts[sessionKey].has(key)) continue;
          sentTranscripts[sessionKey].add(key);
          // emit a single lightweight event per transcript (clients append incrementally)
          emitToRoom(roomId, "transcript", { roomId, ...t });
        }
      } catch (err) {
        console.error("❌ Transcript fetch failed for", sessionKey, err.message);
      }
    }
  }
}, 500);

// ========== Start Server ==========
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));

// global express error handler to catch raw-body / parse errors
app.use((err, req, res, next) => {
  if (!err) return next();
  console.error("❌ Express error:", err && err.message ? err.message : err);
  if (res.headersSent) return next(err);
  res.status(err.status || 400).json({ error: err.message || "Bad request" });
});

// helper: emit to room with debug info
const lastEmitTs = new Map();
function emitToRoom(room, event, payload) {
  const now = Date.now();
  const prev = lastEmitTs.get(room) || now;
  const delta = now - prev;
  lastEmitTs.set(room, now);
  const sockets = io.sockets.adapter.rooms.get(room);
  const size = sockets ? sockets.size : 0;
  console.log(`EMIT -> ${event} to room=${room} sockets=${size} delta_ms=${delta} payloadSize=${JSON.stringify(payload).length}`);
  io.to(room).emit(event, payload);
}