# Changelog

All notable changes to this project will be documented in this file.

---

## [2026-01-04]

### Added

**Camera Position Presets Grid**
Added a 3x3 grid of numbered buttons (1-9) to quickly recall the camera's built-in position presets.

**App-Level Temporary Presets**
Added store and recall functionality for temporary position presets (A, B, C) that save the camera's current pan, tilt, and zoom positions in app memory.

**Visual Preset Status Indicators**
Recall buttons for app presets show visual feedback - grey when empty, green when a position is stored.

**Position Data in Status Bar**
When storing an app preset, the status bar displays the saved pan, tilt, and zoom values for reference.

### Changed

**Position Presets Panel Reorganization**
Split the presets panel into two sections: "Camera Presets" for hardware presets and "App Presets (Temporary)" for app-stored positions.

### Fixed

**Camera Presets Off-By-One Error**
Fixed preset numbering so button 1 calls camera preset 0, button 2 calls preset 1, etc. (camera uses 0-indexed presets).

**App Preset Storage Reliability**
Added retry logic for position queries when storing app presets to handle cameras that don't respond immediately.

---
