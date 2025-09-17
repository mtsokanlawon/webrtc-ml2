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
const io = new Server(server);
const upload = multer();

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
  if (!roomId || !peerId || !req.file) {
    return res.status(400).json({ error: "Missing required fields" });
  }

  // ensure audio dir exists (optional, only if you still want raw saving)
  const dir = path.join(__dirname, "data", roomId, peerId, "audio");
  fs.mkdirSync(dir, { recursive: true });

  // save raw .webm chunk (optional for debugging)
  const webmPath = path.join(dir, `audio_${timestamp}.webm`);
  fs.writeFileSync(webmPath, req.file.buffer);
  console.log(`🎙️ Saved raw audio: ${webmPath}`);

  // instead of writing into ffmpeg stdin:
  const chunkB64 = req.file.buffer.toString("base64");

  await axios.post(`http://localhost:5000/transcribe/realtime/${roomId}`, {
    audio: chunkB64,
    peerId: peerId,
  }).catch(err => console.error("❌ Real-time transcription error", err.message));

  res.json({ status: "ok" });
  
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

        // Stop AssemblyAI session
        axios.post(`http://localhost:5000/stop_session/${roomId}`)
          .then(() => console.log(`🛑 Session stopped for ${roomId}`))
          .catch(err => console.error("❌ Failed to stop session:", err.message));
      const sessionKey = `${roomId}_${peerId}`;
      stopFFmpeg(sessionKey);
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

        // Stop AssemblyAI session
        axios.post(`http://localhost:5000/stop_session/${roomId}`)
          .then(() => console.log(`🛑 Session stopped for ${roomId}`))
          .catch(err => console.error("❌ Failed to stop session:", err.message));

      const sessionKey = `${roomId}_${peerId}`;
      stopFFmpeg(sessionKey);
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
// Track sent transcripts per room
const sentTranscripts = {}; // { roomId: Set<string> }

setInterval(async () => {
  for (const roomId of Object.keys(rooms)) {
    try {
      const res = await axios.get(`http://localhost:5000/get_transcripts/${roomId}`);
      const transcripts = res.data.transcripts || [];

      // console.log("📦 Raw transcripts:", JSON.stringify(res.data.transcripts, null, 2));

      if (!sentTranscripts[roomId]) {
        sentTranscripts[roomId] = new Set();
      }

      // Only pick up new finalized transcripts
      const newOnes = transcripts.filter(t => {
        const key = `${t.timestamp}-${t.peerId}-${t.text}`;
        if (sentTranscripts[roomId].has(key)) return false;
        sentTranscripts[roomId].add(key);
        return true;
      });

      if (newOnes.length > 0) {
        // Format each transcript on its own line
        const joinedText = newOnes
          .map(t => `[${t.timestamp}] (${t.peerId || "Unknown"}): ${t.text.trim()}`)
          .join("\n");

        console.log(`📝 Sent:\n${joinedText}\n➡️ to users in room ${roomId}`);

        // Emit to all users
        io.to(roomId).emit("transcription-event", { transcripts: joinedText });

        // // if (hostPeerId) {
        //   for (let [sid, s] of io.of("/").sockets) {
        //     if (s.roomId === roomId)// && s.peerId === hostPeerId) 
        //     {
        //       // Send as plain string
        //       s.emit("transcription-event", { transcripts: joinedText} );
        //     }
        //   }
        // }
      }
    } catch (err) {
      console.error("❌ Transcript fetch failed for room", roomId, err.message);
    }
  }
}, 2000);


// ========== Start Server ==========
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));