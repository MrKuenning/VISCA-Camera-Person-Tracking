# Changelog

All notable changes to this project will be documented in this file.

---

## [2026-01-11]

### Added

**Asynchronous Camera Control**
Implemented a background thread handler (`AsyncCamera`) for all VISCA commands. This decouples camera movement from the main UI thread, completely eliminating the "app not responding" freezes and video stuttering caused by network latency or camera response times.

**Robust Exception Handling & Crash Protection**
Wrapped all camera communication (Pan/Tilt/Zoom) in try-except blocks. Network timeouts, socket errors, or hardware hiccups no longer cause the Python script to crash; instead, the app now gracefully logs the error and continues operating.

### Changed

**Refined Hysteresis Centering Logic**
Updated the tracking algorithm to ensure the camera centers the face precisely on the target crosshair. The "Dead Zone" now acts strictly as a trigger to start movement, while the centering logic continues until the target is perfectly aligned, preventing the camera from stopping at the edge of the zone.

**Persistent Tracking Settings**
The Advanced Tracking parameters (Target X/Y, Dead Zone W/H) are now part of the global configuration. Added buttons to Save and Reset these settings, which are automatically reloaded every time the app starts.

### Fixed

**Ghost Tracking Overlay**
Fixed an issue where tracking boxes and status text would freeze on screen when tracking was disabled but "Show Center Target" was enabled. Now, dynamic tracking info is only drawn when tracking is active.

---

## [2026-01-11]

### Fixed

**Centering Behavior**
Fixed `_centering_step` to respect user-defined Target X/Y ratios instead of using hardcoded center values.

---

## [2026-01-04]

### Added

**Camera Position Presets Grid**
Added a 3x3 grid of numbered buttons (1-9) to quickly recall the camera's built-in position presets.

**Store/Recall Presets 14-16**
Added store and recall buttons for camera preset slots 14, 15, and 16, allowing users to save and restore additional camera positions.

**Visual Preset Status Indicators**
Recall buttons for presets 14-16 show visual feedback - grey when empty, green when a position is stored.

### Changed

**Position Presets Panel Reorganization**
Split the presets panel into two sections: "Camera Presets" for hardware presets and "App Presets (Temporary)" for app-stored positions.

### Fixed

**Camera Presets Off-By-One Error**
Fixed preset numbering so button 1 calls camera preset 0, button 2 calls preset 1, etc. (camera uses 0-indexed presets).

**App Preset Storage Using Camera Slots**
Changed app presets (A, B, C) to use camera's built-in preset slots 14, 15, 16 instead of position queries, fixing compatibility with cameras that don't support position inquiry commands.

---
