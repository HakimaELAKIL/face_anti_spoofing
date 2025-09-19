# 🔒 Anti-Spoofing Embedded System Project

> A lightweight, real-time **anti-spoofing system** designed for embedded devices.  
> This project combines computer vision and machine learning techniques to detect and prevent spoofing attacks (e.g. photo or video replay) on face-based authentication systems.

---

## 🎯 Project Goals
- Provide **real-time liveness detection** on low-power embedded hardware (ESP32-CAM, Raspberry Pi…)
- **Detect spoofing attempts** such as printed photos, screen replays, or masks
- Integrate **blink detection** and **motion analysis** to verify user presence
- Extract **face embeddings** using FaceNet (or similar models) with **PCA** for dimensionality reduction
- Run a **classifier** (SVM/Random Forest) on device or edge to decide “genuine vs. spoof”

---
