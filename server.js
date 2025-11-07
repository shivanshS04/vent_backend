// server.js
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs").promises;
const path = require("path");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 3000;

// ---------- Multer setup for file uploads ----------
const upload = multer({
  storage: multer.diskStorage({
    destination: async (req, file, cb) => {
      const uploadDir = path.join(__dirname, "uploads");
      try {
        await fs.mkdir(uploadDir, { recursive: true });
        cb(null, uploadDir);
      } catch (err) {
        cb(err);
      }
    },
    filename: (req, file, cb) => {
      const uniqueName = `${Date.now()}-${Math.random()
        .toString(36)
        .substring(7)}${path.extname(file.originalname)}`;
      cb(null, uniqueName);
    },
  }),
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = /\.(mp3|mp4|m4a|wav|webm)$/i.test(file.originalname);
    if (allowed) cb(null, true);
    else cb(new Error("Only audio/video files are allowed."));
  },
});

// ---------- Middleware ----------
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ---------- Environment variable check ----------
if (!process.env.OPENROUTER_API_KEY) {
  console.error("❌ OPENROUTER_API_KEY not set in .env");
  process.exit(1);
}

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const GROQ_API_KEY = process.env.GROQ_API_KEY || process.env.OPENROUTER_API_KEY;

const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions";
const GROQ_TRANSCRIPTION_URL =
  "https://api.groq.com/openai/v1/audio/transcriptions";

const MODEL_FALLBACKS = [
  process.env.OPENROUTER_MODEL || "deepseek/deepseek-r1:free",
  "google/gemini-2.0-flash-exp:free",
  "meta-llama/llama-3.2-3b-instruct:free",
  "nousresearch/hermes-3-llama-3.1-405b:free",
  "openai/gpt-3.5-turbo",
];

// ---------- Utility ----------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function extractNature(aiResponse) {
  const match = aiResponse.match(/\[SENTIMENT:\s*(positive|negative)\]/i);
  return match ? match[1].toLowerCase() : "positive";
}

// ---------- Groq Whisper Transcription ----------
async function transcribeAudio(filePath) {
  const FormData = (await import("form-data")).default;
  const fileBuffer = await fs.readFile(filePath);
  const fileName = path.basename(filePath);

  const formData = new FormData();
  formData.append("file", fileBuffer, {
    filename: fileName,
    contentType: "audio/mp4",
  });
  formData.append("model", "whisper-large-v3-turbo");
  formData.append("response_format", "json");

  const response = await fetch(GROQ_TRANSCRIPTION_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
      ...formData.getHeaders(),
    },
    body: formData,
  });

  if (!response.ok) {
    const errData = await response.text();
    throw new Error(`Groq API failed: ${response.status} - ${errData}`);
  }

  const data = await response.json();
  return data.text || "";
}

// ---------- OpenRouter API helper ----------
async function callOpenRouter(
  systemPrompt,
  userMessage,
  retryCount = 0,
  modelIndex = 0
) {
  const model = MODEL_FALLBACKS[modelIndex];
  const response = await fetch(OPENROUTER_BASE_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
      "HTTP-Referer": process.env.APP_URL || "http://localhost:3000",
      "X-Title": "Journal AI App",
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage },
      ],
      temperature: 0.7,
      max_tokens: 800,
    }),
  });

  if (response.status === 429 && modelIndex < MODEL_FALLBACKS.length - 1) {
    console.log(`⚠️ Rate limited on ${model}, switching model...`);
    await sleep(2000);
    return callOpenRouter(systemPrompt, userMessage, 0, modelIndex + 1);
  }

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`OpenRouter failed: ${response.status} - ${errText}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || "No response";
}

// ---------- ROUTES ----------

// 🎙️ Transcribe audio
app.post("/transcribe", upload.single("audio"), async (req, res) => {
  let uploadedFile = null;
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "Invalid input",
        message: "Request must include an audio file (field name: 'audio')",
      });
    }

    uploadedFile = req.file.path;
    console.log("🎧 Audio received:", req.file.originalname);

    const text = await transcribeAudio(uploadedFile);
    if (!text.trim()) {
      return res.status(400).json({
        error: "Transcription failed",
        message: "Could not extract text from the audio file.",
      });
    }

    res.status(200).json({ success: true, text });
  } catch (err) {
    console.error("❌ /transcribe error:", err.message);
    res.status(500).json({
      error: "Transcription error",
      message: err.message,
    });
  } finally {
    if (uploadedFile) {
      await fs.unlink(uploadedFile).catch(() => {});
    }
  }
});

// 🧠 Analyze a journal entry
app.post("/analyze", async (req, res) => {
  try {
    const { entry } = req.body;
    if (!entry || typeof entry !== "string" || !entry.trim()) {
      return res
        .status(400)
        .json({
          error: "Invalid input",
          message: "Provide a valid entry string.",
        });
    }

    const systemPrompt = `You are an empathetic AI journal companion... (same as your previous long prompt)`;

    const aiResponse = await callOpenRouter(systemPrompt, entry);
    const nature = extractNature(aiResponse);
    const cleaned = aiResponse
      .replace(/\[SENTIMENT:\s*(positive|negative)\]\s*/i, "")
      .trim();

    res.status(200).json({ success: true, analysis: cleaned, nature });
  } catch (err) {
    console.error("❌ /analyze error:", err.message);
    res.status(500).json({ error: "Internal error", message: err.message });
  }
});

// 📅 Monthly recap
app.post("/recap", async (req, res) => {
  try {
    const { entries } = req.body;
    if (!entries || typeof entries !== "string" || !entries.trim()) {
      return res
        .status(400)
        .json({ error: "Invalid input", message: "Entries string required." });
    }

    const systemPrompt = `You are an insightful AI journal analyst... (same as your previous prompt)`;
    const recap = await callOpenRouter(systemPrompt, entries);

    res.status(200).json({ success: true, recap });
  } catch (err) {
    console.error("❌ /recap error:", err.message);
    res.status(500).json({ error: "Internal error", message: err.message });
  }
});

// Health check
app.get("/health", (req, res) =>
  res.json({
    status: "healthy",
    time: new Date().toISOString(),
    models: MODEL_FALLBACKS,
  })
);

// Catch-all 404
app.use((req, res) => {
  res
    .status(404)
    .json({
      error: "Not Found",
      message: `${req.method} ${req.path} does not exist`,
    });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res
    .status(500)
    .json({ error: "Internal server error", message: err.message });
});

// Start
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
