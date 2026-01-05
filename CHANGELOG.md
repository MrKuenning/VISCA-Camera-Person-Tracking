# Changelog

All notable changes to this project will be documented in this file.

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
