# Digital Liveness & Forensics Suite (WIP)

> **Status: Actively Under Development.** > The isolated Python forensic microservices are currently being built and tested. The central Java Spring Boot API Gateway and JavaScript Client-Side Tracker are pending integration.

## Main Purpose
A polyglot, microservice-based fusion engine designed to verify human interaction and detect AI-generated media (Deepfakes, LLM text, cloned audio, and synthesized images). 

## Architecture Overview
This system utilizes a Scatter-Gather pattern. A central Java Gateway routes media to isolated, domain-specific Python forensic services, fusing their confidence scores into a final "Probability of Human Origin."

### Completed Microservices (Isolated Testing Phase)
* **Biological Liveness Service (Python/FastAPI):** Uses rPPG and MediaPipe to detect micro-fluctuations in facial blood volume.
* **Audio Forensics Service (Python/FastAPI):** Uses Wav2Vec2 and librosa to detect missing breath sounds, phase irregularities, and unnatural MFCC acceleration in synthetic voices.
* **Reverse Engineering Service (Python/FastAPI):** Analyzes raw hex streams in chunks to prevent Zip Bombs while detecting programmatic compiler fingerprints (e.g., automated FFmpeg wrappers).
* **Vision Artifact Service (Python/FastAPI):** Uses Vision Transformers (ViT) and FFT analysis with adversarial blurring to detect latent diffusion tiling.
* **Text & NLP Service (Python/FastAPI):** Uses local GPT-2 deterministic math to measure sentence perplexity and burstiness variance to catch LLM-generated text.

### Pending Microservices
* **API Gateway / Central Brain (Java Spring Boot):** The orchestrator utilizing Resilience4j circuit breakers and async routing.
* **Client-Side Behavioral Tracker (JavaScript):** Detects botnet keystroke dynamics and mouse trajectories.
* **Metadata & Provenance Service (Java):** Cryptographic C2PA and EXIF header inspection.

## Prerequisites & Setup
Because this suite relies heavily on system-level signal processing and local AI inference, standard package managers are not enough.

1.  **System Dependencies:** You *must* install `ffmpeg` on your host machine (via `winget`, `brew`, or `apt`) to decode complex audio/video codecs.
2.  **Local AI Models:** To prevent runtime crashes and save bandwidth, the Hugging Face models (Wav2Vec2, ViT, GPT-2) must be downloaded manually into their respective `local_*_model` directories before starting the FastAPI servers.
3.  **Docker:** Required for the Reverse Engineering service to enforce a strict 512MB RAM quota against decompression attacks.

## Author
**Niroshan Dh**
