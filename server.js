const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs").promises;
const path = require("path");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 3000;

// Configure multer for file uploads
const upload = multer({
  storage: multer.diskStorage({
    destination: async (req, file, cb) => {
      const uploadDir = path.join(__dirname, "uploads");
      try {
        await fs.mkdir(uploadDir, { recursive: true });
        cb(null, uploadDir);
      } catch (error) {
        cb(error);
      }
    },
    filename: (req, file, cb) => {
      const uniqueName = `${Date.now()}-${Math.random()
        .toString(36)
        .substring(7)}${path.extname(file.originalname)}`;
      cb(null, uniqueName);
    },
  }),
  limits: {
    fileSize: 25 * 1024 * 1024, // 25MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = [
      "audio/mp4",
      "audio/mpeg",
      "audio/wav",
      "audio/webm",
      "video/mp4",
      "audio/m4a",
    ];
    if (
      allowedTypes.includes(file.mimetype) ||
      file.originalname.match(/\.(mp4|m4a|mp3|wav|webm)$/i)
    ) {
      cb(null, true);
    } else {
      cb(new Error("Invalid file type. Only audio/video files are allowed."));
    }
  },
});

// Middleware
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Validate required environment variables
if (!process.env.OPENROUTER_API_KEY) {
  console.error("ERROR: OPENROUTER_API_KEY is not set in .env file");
  process.exit(1);
}

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions";

// Model fallback chain - will try these in order if rate limited
const MODEL_FALLBACKS = [
  process.env.OPENROUTER_MODEL || "deepseek/deepseek-r1:free",
  "google/gemini-2.0-flash-exp:free", // Free Gemini model
  "meta-llama/llama-3.2-3b-instruct:free", // Smaller Llama model
  "nousresearch/hermes-3-llama-3.1-405b:free", // Free Hermes model
  "openai/gpt-3.5-turbo", // Paid fallback if all free options exhausted
];

/**
 * Transcribes audio file using OpenRouter Whisper API
 * @param {string} filePath - Path to the audio file
 * @returns {Promise<string>} - Transcribed text
 */
async function transcribeAudio(filePath) {
  try {
    console.log("Transcribing audio file:", filePath);

    // Read the file as a buffer
    const fileBuffer = await fs.readFile(filePath);
    const fileName = path.basename(filePath);

    // Create form data
    const FormData = (await import("form-data")).default;
    const formData = new FormData();
    formData.append("file", fileBuffer, {
      filename: fileName,
      contentType: "audio/mp4",
    });
    formData.append("model", "openai/whisper-1");

    const response = await fetch(
      "https://openrouter.ai/api/v1/audio/transcriptions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${OPENROUTER_API_KEY}`,
          ...formData.getHeaders(),
        },
        body: formData,
      }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        `Transcription failed: ${response.status} - ${
          errorData.error?.message || response.statusText
        }`
      );
    }

    const data = await response.json();
    console.log("✓ Transcription successful");

    return data.text || "";
  } catch (error) {
    console.error("Transcription error:", error.message);
    throw error;
  }
}

/**
 * Sleep utility for retry delays
 */
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Extracts the sentiment from AI response
 * @param {string} aiResponse - The AI's full response
 * @returns {string} - "positive" or "negative"
 */
function extractNature(aiResponse) {
  // Look for the sentiment marker in the response
  const match = aiResponse.match(/\[SENTIMENT:\s*(positive|negative)\]/i);
  if (match) {
    return match[1].toLowerCase();
  }

  // Fallback: if no explicit marker found, default to positive
  return "positive";
}

/**
 * Helper function to call OpenRouter API with retry logic and model fallbacks
 * @param {string} systemPrompt - The system instruction for the AI
 * @param {string} userMessage - The user's message/content
 * @param {number} retryCount - Current retry attempt (internal use)
 * @param {number} modelIndex - Current model index in fallback chain (internal use)
 * @returns {Promise<string>} - The AI's response text
 */
async function callOpenRouter(
  systemPrompt,
  userMessage,
  retryCount = 0,
  modelIndex = 0
) {
  const maxRetries = 2; // Reduced for free tier
  const baseDelay = 3000; // 3 seconds base delay for free tier
  const currentModel = MODEL_FALLBACKS[modelIndex];

  try {
    console.log(
      `Attempting API call with model: ${currentModel} (attempt ${
        retryCount + 1
      })`
    );

    const response = await fetch(OPENROUTER_BASE_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
        "HTTP-Referer": process.env.APP_URL || "http://localhost:3000",
        "X-Title": "Journal AI App",
      },
      body: JSON.stringify({
        model: currentModel,
        messages: [
          {
            role: "system",
            content: systemPrompt,
          },
          {
            role: "user",
            content: userMessage,
          },
        ],
        temperature: 0.7,
        max_tokens: 800, // Reduced for free tier efficiency
      }),
    });

    const data = await response.json();

    // Handle rate limiting (429) with retry and model fallback
    if (response.status === 429) {
      console.warn(`Rate limit hit for model: ${currentModel}`);

      // Try next model in fallback chain
      if (modelIndex < MODEL_FALLBACKS.length - 1) {
        console.log(
          `Switching to fallback model: ${MODEL_FALLBACKS[modelIndex + 1]}`
        );
        await sleep(2000); // 2 second pause before trying next model
        return callOpenRouter(systemPrompt, userMessage, 0, modelIndex + 1);
      }

      // If all models exhausted, retry with exponential backoff
      if (retryCount < maxRetries) {
        const delay = baseDelay * Math.pow(2, retryCount);
        console.log(
          `Retrying in ${delay}ms (attempt ${retryCount + 1}/${maxRetries})`
        );
        await sleep(delay);
        return callOpenRouter(systemPrompt, userMessage, retryCount + 1, 0);
      }

      throw new Error(
        "Rate limit exceeded. Free tier models are currently busy. Please try again in a few minutes."
      );
    }

    // Handle other API errors
    if (!response.ok) {
      const errorMessage = data.error?.message || response.statusText;

      // Retry on server errors (5xx)
      if (response.status >= 500 && retryCount < maxRetries) {
        const delay = baseDelay * Math.pow(2, retryCount);
        console.log(`Server error. Retrying in ${delay}ms...`);
        await sleep(delay);
        return callOpenRouter(
          systemPrompt,
          userMessage,
          retryCount + 1,
          modelIndex
        );
      }

      throw new Error(
        `OpenRouter API error: ${response.status} - ${errorMessage}`
      );
    }

    // Extract the AI's response text
    if (!data.choices || !data.choices[0] || !data.choices[0].message) {
      throw new Error("Invalid response structure from OpenRouter API");
    }

    console.log(`✓ Successfully got response from: ${currentModel}`);
    return data.choices[0].message.content;
  } catch (error) {
    // Network errors - retry with exponential backoff
    if (error.message.includes("fetch failed") && retryCount < maxRetries) {
      const delay = baseDelay * Math.pow(2, retryCount);
      console.log(`Network error. Retrying in ${delay}ms...`);
      await sleep(delay);
      return callOpenRouter(
        systemPrompt,
        userMessage,
        retryCount + 1,
        modelIndex
      );
    }

    console.error("OpenRouter API call failed:", error.message);
    throw error;
  }
}

/**
 * POST /analyze
 * Analyzes a single journal entry and returns empathetic follow-up questions
 */
app.post("/analyze", async (req, res) => {
  try {
    const { entry } = req.body;

    // Validate input
    if (!entry || typeof entry !== "string") {
      return res.status(400).json({
        error: "Invalid input",
        message:
          'Request body must contain an "entry" field with a string value',
      });
    }

    if (entry.trim().length === 0) {
      return res.status(400).json({
        error: "Invalid input",
        message: "Journal entry cannot be empty",
      });
    }

    // Limit entry length to prevent abuse
    let trimmedEntry = entry;
    if (entry.length > 10000) {
      trimmedEntry = entry.slice(0, 10000);
    }

    // System prompt for mood analysis and empathetic response
    const systemPrompt = `You are an empathetic AI journal companion. Your role is to:
1. Analyze the emotional tone and mood of the user's journal entry
2. Identify key themes, concerns, or highlights
3. Generate 2-3 thoughtful, empathetic follow-up questions that encourage deeper reflection
4. Provide validation and support

IMPORTANT GUIDELINES:
- Use SIMPLE, everyday language that anyone can understand
- AVOID jargon, technical terms, or complex vocabulary
- Write like you're talking to a friend - warm, natural, and conversational
- If the user writes in Hindi or uses Hindi words, respond in Hinglish (Hindi words written in English script mixed with English)
- Keep sentences short and clear
- KEEP YOUR RESPONSE BRIEF - Maximum 3-4 short sentences total
- Use line breaks between thoughts for easy reading
- At the very beginning of your response, include a sentiment marker: [SENTIMENT: positive] or [SENTIMENT: negative]

After the sentiment marker, provide a SHORT empathetic response (50-100 words max).

Example format for English entry:
[SENTIMENT: positive]
That sounds amazing! I can feel your excitement.

What made this moment special for you?
How are you planning to celebrate?

Example format for Hindi/Hinglish entry:
[SENTIMENT: negative]
Aaj kaafi tough raha lagta hai. It's okay to feel overwhelmed.

Kya hua specifically?
Abhi kaise feel kar rahe ho?`;

    // Call OpenRouter API with retry logic
    const aiResponse = await callOpenRouter(systemPrompt, trimmedEntry);

    // Extract sentiment from AI response
    const nature = extractNature(aiResponse);

    // Remove the sentiment marker from the response before sending to client
    const cleanedAnalysis = aiResponse
      .replace(/\[SENTIMENT:\s*(positive|negative)\]\s*/i, "")
      .trim();

    // Return successful response
    res.status(200).json({
      success: true,
      analysis: cleanedAnalysis,
      nature: nature,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error in /analyze route:", error);

    // Handle rate limiting specifically
    if (error.message.includes("Rate limit exceeded")) {
      return res.status(429).json({
        error: "Rate limit exceeded",
        message: "Too many requests. Please wait a moment and try again.",
        retryAfter: 60, // Suggest retry after 60 seconds
      });
    }

    // Return appropriate error response
    if (error.message.includes("OpenRouter API error")) {
      return res.status(502).json({
        error: "External API error",
        message:
          "Failed to communicate with AI service. Please try again later.",
        details: error.message,
      });
    }

    res.status(500).json({
      error: "Internal server error",
      message: "An unexpected error occurred while analyzing your entry",
    });
  }
});

/**
 * POST /recap
 * Generates a monthly summary of all journal entries
 */
app.post("/recap", async (req, res) => {
  try {
    const { entries } = req.body;

    // Validate input
    if (!entries || typeof entries !== "string") {
      return res.status(400).json({
        error: "Invalid input",
        message:
          'Request body must contain an "entries" field with a string value',
      });
    }

    if (entries.trim().length === 0) {
      return res.status(400).json({
        error: "Invalid input",
        message: "Entries cannot be empty",
      });
    }

    // Limit entries length
    if (entries.length > 50000) {
      return res.status(400).json({
        error: "Invalid input",
        message: "Entries are too long. Maximum 50,000 characters allowed.",
      });
    }

    // System prompt for monthly recap
    const systemPrompt = `You are an insightful AI journal analyst. Create a comprehensive monthly recap that includes:

1. **Overall Emotional Trend**: Describe the general emotional trajectory over the month
2. **Key Highlights**: Identify 3-5 significant moments or achievements
3. **Recurring Themes**: Note patterns in thoughts, concerns, or activities
4. **Personal Growth**: Observe any signs of growth or change
5. **Supportive Advice**: Offer 2-3 pieces of constructive, encouraging advice for moving forward

Be empathetic, constructive, and focus on helping the user gain insights about themselves. Structure your response with clear sections. Keep it concise (under 500 words).`;

    // Call OpenRouter API with retry logic
    const aiResponse = await callOpenRouter(systemPrompt, entries);

    // Return successful response
    res.status(200).json({
      success: true,
      recap: aiResponse,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error in /recap route:", error);

    // Handle rate limiting specifically
    if (error.message.includes("Rate limit exceeded")) {
      return res.status(429).json({
        error: "Rate limit exceeded",
        message: "Too many requests. Please wait a moment and try again.",
        retryAfter: 60,
      });
    }

    // Return appropriate error response
    if (error.message.includes("OpenRouter API error")) {
      return res.status(502).json({
        error: "External API error",
        message:
          "Failed to communicate with AI service. Please try again later.",
        details: error.message,
      });
    }

    res.status(500).json({
      error: "Internal server error",
      message: "An unexpected error occurred while generating your recap",
    });
  }
});

/**
 * POST /transcribe
 * Transcribes an uploaded audio file and returns the text
 */
app.post("/transcribe", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "No file uploaded",
        message: 'Please upload an audio file with the field name "audio"',
      });
    }

    // Transcribe the uploaded audio file
    const text = await transcribeAudio(req.file.path);

    // Optionally, delete the file after transcription
    await fs.unlink(req.file.path).catch(() => {});

    res.status(200).json({
      success: true,
      text,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error in /transcribe route:", error);
    res.status(500).json({
      error: "Internal server error",
      message: "An error occurred while transcribing the audio file",
    });
  }
});

/**
 * GET /health
 * Health check endpoint
 */
app.get("/health", (req, res) => {
  res.status(200).json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    service: "Journal AI Backend",
    models: MODEL_FALLBACKS,
  });
});

/**
 * GET /models
 * Returns available models and their status
 */
app.get("/models", (req, res) => {
  res.status(200).json({
    primary: MODEL_FALLBACKS[0],
    fallbacks: MODEL_FALLBACKS,
    note: "Models are tried in order if rate limits are encountered",
  });
});

/**
 * 404 handler for undefined routes
 */
app.use((req, res) => {
  res.status(404).json({
    error: "Not found",
    message: `Route ${req.method} ${req.path} does not exist`,
  });
});

/**
 * Global error handler
 */
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({
    error: "Internal server error",
    message: "An unexpected error occurred",
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Journal AI Backend running on port ${PORT}`);
  console.log(`📝 Primary model: ${MODEL_FALLBACKS[0]}`);
  console.log(`🔄 Fallback models: ${MODEL_FALLBACKS.slice(1).join(", ")}`);
  console.log(`✅ Server is ready to accept requests`);
});
