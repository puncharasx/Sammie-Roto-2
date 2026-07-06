# PLAN: Sammie-Roto-2 Fork — Precision & Performance Upgrade (v2, grilled 2026-07-06)

**Status:** Ready for implementation (all open questions resolved — see §6 decision log)
**Supersedes:** PLAN_sammie_fork_segmentation.md (v1)
**Target repo (upstream):** https://github.com/Zarxrax/Sammie-Roto-2 (v2.3.4.a1, Python 3.10+, PySide6, SAM2/EfficientTAM vendored)
**Decision log:**
- Full Rust rewrite evaluated and REJECTED — bottleneck is GPU inference (PyTorch), not Python; SAM2 video predictor / diffusers pipelines are impractical to port and upstream adds new models frequently. This fork stays Python.
- This is a PERSONAL TOOL. Upstream is PINNED at v2.3.4.a1; no per-milestone rebasing. Cherry-pick upstream fixes only when actually needed. Editing vendored `sam2/` is acceptable (though M3 no longer requires it).
- Target hardware: NVIDIA 8–12 GB VRAM. CPU remains a crash-not floor, not a performance target.

**Scope:** Two phases.
- **Phase 1 — Precision Segmentation:** box prompts, draggable points, multimask selection, crop-based HQ refinement, trimap matting for fine detail (hair).
- **Phase 2 — Architecture & Workflow:** worker-thread refactor, manual mask brush, RAM cache for scrubbing, full undo stack, timeline keyframe markers, autosave. (Stretch: proxy workflow, CLI mode, multi-channel EXR.)

---

## 0. Context: How the Current Code Works

Read these before writing any code:

| Area | File | Key symbols |
|---|---|---|
| SAM manager | `sammie/sammie.py` | `SamManager`, `segment_image()`, `preview_point()`, `replay_points()`, `_propagate()` |
| Image viewer | `sammie/gui_widgets.py` | `ImageViewer` (QGraphicsView), `point_clicked = Signal(int, int, bool)`, shift-preview debounce (`_preview_timer`, 150 ms), `FrameSlider` |
| Main window | `sammie_main.py` | `add_point_from_click()`, `point_manager.points`, `undo_last_point()` (Ctrl+Z), `PointTable` sync |
| Core state | `sammie/core.py` | `frames_dir`, `mask_dir`, `masks_backup`, `DeviceManager`, `VideoInfo`, `PALETTE` |
| Matting | `sammie/matting.py` | `MattingManager` base class + per-model subclasses |
| Export | `sammie/export_workers.py`, `export_formats.py` | `BaseExportWorker(QThread)` — the ONLY long op already on a worker thread |
| Models | `sammie/model_downloader.py` | `ensure_models()` — lazy download with hash check |
| Settings | `sammie/settings_manager.py` | session vs persistent app settings |

Key facts (verified in source):

1. **Point data model** is a flat list of dicts `{'frame': int, 'object_id': int, 'positive': bool, 'x': int, 'y': int}` in `point_manager.points`; `PointTable` is a synced view.
2. **SAM2 predictor** is called via `predictor.add_new_points_or_box(inference_state, frame_idx, obj_id, points, labels, ...)`. The API **already accepts `box=`** and `clear_old_points=True`; `preview_point()` uses a try-and-revert pattern worth reusing.
3. **Masks are binary PNGs** at `temp/masks/{frame:05d}/{obj_id}.png` (0/255), threshold `out_mask_logits > 0.0`.
4. **Threading model is the main weakness:** `_propagate()` (and matting/removal loops) run ON THE GUI THREAD using `QApplication.processEvents()` inside the loop, with a modal `QProgressDialog`. Export is the correct reference implementation (`QThread` + Signals).
5. **Everything is disk-backed:** video is fully extracted to `temp/frames/` as PNG (threaded writer pool, `save_worker` queue in `load_video`); every scrub re-reads frame + masks from disk. No RAM cache exists.
6. **Undo:** only `undo_last_point` exists. No general undo stack.
7. GUI↔logic coupling is via callback lists (`_notify(action, **kwargs)`) plus module-level globals in `core.py`. Extend the pattern; do not rewrite it.

---

## 1. Fork Setup

- Fork upstream, keep `upstream` remote for cherry-picks. **Pinned at v2.3.4.a1 — no routine rebasing.**
- Branch per milestone: `feat/m1-box-prompt`, `feat/m7-worker-threads`, etc.
- Add `FORK_NOTES.md` at repo root documenting every divergence (file + reason) and a manual test checklist per milestone. Update it in every milestone. (Purpose: documentation for future-you, not a rebase conflict map.)

---

## 2. Phase 1 — Precision Segmentation

### M1 — Box Prompt (highest value / lowest risk)

**Goal:** Drag a rectangle to select an object; box is sent to SAM2 as a prompt. Boxes coexist with points.

1. **Data model:** add `point_manager.boxes`: `{'frame', 'object_id', 'x1', 'y1', 'x2', 'y2'}`. Max ONE box per (frame, object_id); a new box replaces the old.
2. **Viewer:** add a "Box" tool mode to `ImageViewer` (toolbar toggle AND hold-`B` modifier). Rubber-band `QGraphicsRectItem` while dragging; on release emit `box_drawn = Signal(int, int, int, int)` (scene coords, clamped to image bounds). Ignore boxes < 5×5 px.
3. **SamManager:** extend `segment_image()` with optional `box: np.ndarray | None` (shape `(4,)`, XYXY float32), passed together with points for the same object in the SAME `add_new_points_or_box` call. **Constraint (verified in `sam2/sam2_video_predictor.py:197-203`): `box=` requires `clear_old_points=True`.** Therefore any (frame, object) that has a box cannot use incremental point appends — every prompt change on it resends box + ALL points in one call. `replay_points()` for boxed objects does the same (single combined call per frame/object).
4. **Main window:** `box_drawn` → store → segment → refresh overlay. Render stored box as outlined rect in the object's `core.PALETTE` color.
5. **Persistence:** serialize `boxes` alongside `points` in the session JSON; old project files without the key must load fine.
6. **PointTable:** box rows (or compact separate list) with per-box delete; deleting triggers full replay rebuild.

**Acceptance:** box segments an object; negative points refine it afterwards; delete works; save/load round-trips; old projects load.

### M2 — Draggable Points

**Goal:** Grab and drag existing points; mask updates live (debounced) and commits on release.

1. **Rendering:** replace pixmap-painted points with `DraggablePointItem(QGraphicsEllipseItem)` per point on the current frame — `ItemIsMovable`, `ItemSendsGeometryChanges`, `ItemIgnoresTransformations` (scale-invariant, ~6 px radius). Green positive / red negative, thin white border.
2. **Hit priority:** in `ImageViewer.mousePressEvent`, if `itemAt(pos)` is a `DraggablePointItem`, delegate to the item and skip point-add logic.
3. **Drag flow (DECIDED — commit on release only):** dragging moves the Qt item freely with NO inference during the drag (synchronous `preview_point`-style inference on the GUI thread would freeze mouse tracking). On release: commit coords, re-segment with `clear_old_points=True` replay for that frame/object, sync `PointTable`. Live preview during drag is a small post-M7 follow-up (inference off-thread).
4. **Edge cases:** clamp to image bounds; preserve tracking data on modification (upstream 2.1.0 already has this code path — find and reuse); re-run only the affected object on multi-object frames.

**Acceptance:** smooth drag at 4K / Base model on GPU (pure Qt, no inference mid-drag); re-segment on release; PointTable updates.

### M3 — Multimask Candidate Selection

**Goal:** For the FIRST prompt on an object, cycle SAM's 3 candidate masks.

**DECIDED — no vendored edits; depends on M4.** Finding: the video predictor ALREADY runs multimask internally on the first click (`multimask_output_in_sam: true` in every model config) and auto-picks best-IoU inside `sam2_base.py`; plumbing the 3 candidates out of `track_step` is deep surgery. Instead:

1. First prompt on an object → run M4's lazily-loaded `SAM2ImagePredictor` with `multimask_output=True` (natively returns 3 masks + IoU scores) on the full frame.
2. Store candidates transiently in `SamManager`; `Tab` (viewer focus) cycles → rewrite mask PNG → refresh → notify.
3. Status hint: `Candidate 2/3 (IoU 0.94) — Tab to cycle`.
4. On selection (or next action), commit the chosen mask to the video predictor via `add_new_mask()` (`sam2/sam2_video_predictor.py:300`) so tracking follows the pick.
5. A second prompt on the object collapses to standard single-mask points/box behavior (clear + replay).

**Acceptance:** one click on a shirt can cycle whole-person / shirt / torso variants; propagation follows the chosen candidate.

### M4 — Crop-Based High-Resolution Refinement

**Goal:** Fix small-object/fine-structure loss from SAM2's 1024 px internal resolution on HD/4K footage.

1. New `sammie/refine.py` with `refine_mask_crop(frame_idx, object_id) -> np.ndarray`:
   - Coarse mask → bbox → expand 25% margin (clamped) → crop frame → remap prompts into crop coords.
   - Run the vendored `SAM2ImagePredictor` on the crop (image predictor, NOT video predictor — never touch `inference_state`).
   - Paste refined mask back at full resolution (still binary).
2. Lazy-load the image predictor with the SAME checkpoint already on disk; release VRAM after batch runs per `DeviceManager.clear_cache()` conventions. (M3 reuses this predictor — keep the loader generic.)
3. UI: "Refine (HQ)" button (current frame + selected object) + checkbox "Refine all frames after propagation" running per-frame with cancellable progress (M7 lands first in the revised order — build batch mode on the worker infra).
4. Skip refinement when crop ≥ 90% of frame area.
5. **Storage (DECIDED):** refine overwrites `temp/masks/` in place, but first copies affected masks to a refine-owned `temp/masks_prerefine/` slot, with a "Revert refinement" action. **Never use `temp/masks_backup/`** — the duplicate-frame handler owns that slot and blindly restores it as "source data" (`duplicate_frame_handler.py:123-126`); sharing it would silently destroy refinements. Pipeline ordering: refine runs AFTER duplicate-frame processing.

**Acceptance:** visible edge improvement on 4K with a small subject; 300-frame batch refine completes with progress + cancel; no VRAM accumulation.

### M5 — Trimap-Based Fine-Detail Matting (Hair)

**Goal:** True soft alpha for hair/fur via automatic trimap + ViTMatte, complementing (not replacing) MatAnyone/VideoMaMa.

1. New `sammie/trimap_matting.py`:
   - `build_trimap(mask, erode_px=10, dilate_px=20)` → uint8 {0, 128, 255}; kernel sizes user-adjustable, defaults tuned at 1080p and scaled by frame height.
   - `ViTMatteRefiner` following `MattingManager` conventions (callbacks, `_prepare_device`, progress helpers). Model: ViTMatte-B (`hustvl/vitmatte-base-composition-1k`) registered in `model_downloader.py` with URL + SHA256.
   - VRAM strategy at 4K: tile the unknown-band region — **MANDATORY on the 8–12 GB target hardware, not optional** (document it).
2. Output to `temp/matting/` in the exact format the export path consumes — **verified:** `temp/matting/{frame:05d}/{object_id}.png` grayscale, same layout as masks (`matting.py:531`, `export_dialog.py:842-855`).
3. UI: new entry "ViTMatte (trimap)" in the EXISTING matting model dropdown; erode/dilate sliders in its settings section. Zero new tabs. **Default (DECIDED): ViTMatte-B, with Small selectable.**
4. Optional temporal smoothing checkbox (EMA over alpha, window 3) — ViTMatte is per-frame and will flicker; MatAnyone stays the default for temporal-critical work (note in FORK_NOTES).

**Acceptance:** flyaway hair strands preserved vs MatAnyone default output; export unchanged.

### M6 (Stretch) — Text-Prompt Object Selection

Grounding-DINO-tiny or Florence-2 → box → M1's box path. Separate lazy download, feature-flagged in settings. Start only after M1–M5 are stable.

---

## 3. Phase 2 — Architecture & Workflow

### M7 — Worker-Thread Refactor (do this EARLY in Phase 2; unblocks everything else)

**Goal:** Move propagation, matting, and removal off the GUI thread. Kill every `QApplication.processEvents()` in a compute loop.

1. Create `sammie/workers.py` modeled on `export_workers.py`: `class TrackingWorker(QThread)` with signals `progress(int)`, `frame_done(int)`, `finished(object)`, `error(str)`, and a cooperative `request_cancel()` flag checked between frames.
2. Move the body of `SamManager._propagate()` into the worker; the manager keeps the API but dispatches to the worker and returns via signals. Same treatment for matting (`MattingManager.run_matting` loops) and removal.
3. Main window: replace modal `QProgressDialog` + `processEvents` with a non-modal **inline status-bar progress + cancel button** (DECIDED) driven by signals; frame slider nudge via `frame_done`.
4. **Thread-safety rules (document in FORK_NOTES) — DECIDED:** interactive single-frame segmentation (`segment_image` on click, `preview_point` on shift-hover) STAYS synchronous on the GUI thread (~50–300ms block is imperceptible; async click UX isn't worth the plumbing). The invariant is: **no two threads touch `inference_state` concurrently** — while a worker runs, all prompt input is disabled via a busy flag. Mask PNG writes stay on the worker; GUI reads masks only after `frame_done(idx)` for that index; one compute worker at a time.
5. Cancellation must release VRAM properly (call the same cleanup upstream does).
6. **Out of scope:** `duplicate_frame_handler.py:159`'s `processEvents` (video-load path) — load is inherently modal; leave it.

**Acceptance:** UI stays fully responsive during a long propagation; cancel is immediate (≤1 frame); no re-entrancy (rapid clicking during tracking cannot start a second job); matting + removal behave identically.

**Note:** M4's batch refine and M5's batch matting should be built on this worker infra if M7 is done first — recommended ordering below reflects this.

### M8 — Manual Mask Brush / Eraser

**Goal:** Paint/erase masks directly — the escape hatch every production roto tool has.

1. New viewer tool modes: Brush / Eraser, circular cursor preview, size via `[` `]` + slider, applies to the SELECTED object's mask on the current frame.
2. Implementation: edit the mask PNG in memory (numpy circle stamping along interpolated stroke path), write on stroke end, refresh overlay via existing notify path.
3. Strokes are undoable (see M10) — store per-stroke diff (bbox + before-patch) rather than full-frame copies.
4. Brushed frames must be flagged so the duplicate-frame handler / re-propagation logic doesn't silently overwrite them: add a `temp/masks/{frame:05d}/.edited` marker (or a JSON registry) and warn the user before a propagation pass would overwrite edited frames. **DECIDED: ask once per propagation with remember-choice (skip edited / overwrite).**
5. **Matting staleness:** export prefers `temp/matting/` when present, so a brush edit MUST invalidate that frame+object's matting PNG (delete it) — otherwise the edit silently never reaches export.

**Acceptance:** fix a bad SAM frame by hand in seconds; brush strokes survive unless the user explicitly allows overwrite; undo per stroke works.

### M9 — RAM Cache for Scrubbing

**Goal:** Smooth timeline scrubbing; stop hitting disk for every frame view.

1. New `sammie/frame_cache.py`: LRU cache keyed `(frame_idx)` for decoded frames and `(frame_idx, obj_id)` for masks; capacity in MB from a persistent setting (**DECIDED: default min(2048 MB, 25% of system RAM)**, computed from decoded frame size).
2. Route all frame/mask reads in the display path (`update_image` and the `_handle_*_view` functions in `sammie/sammie.py`) through the cache.
3. Invalidation: any mask write (segmentation, propagation `frame_done`, brush, refine, matting) must invalidate the affected keys — add a single `invalidate(frame_idx, obj_id=None)` helper and call it at every write site. **Threading rule:** worker-thread writes invalidate via the queued `frame_done` signal on the GUI thread, never by calling the cache directly from the worker.
4. Optional prefetch of ±N frames around the playhead on a low-priority thread.

**Acceptance:** scrubbing a cached region is visibly smoother at 4K; memory stays under the configured cap; no stale-mask display after any edit (test: brush → scrub away → scrub back).

### M10 — Full Undo/Redo Stack

**Goal:** Replace "remove last point" with a real `QUndoStack`.

1. `QUndoCommand` subclasses: AddPoint, DeletePoint(s), MovePoint (M2 drag = one command per drag), AddBox, DeleteBox, BrushStroke (M8 diff), ClearTracking.
2. Ctrl+Z / Ctrl+Shift+Z (and Ctrl+Y) wired to the stack; Edit menu shows command names.
3. Segmentation side-effects: point/box commands re-run the existing replay path on undo/redo (correctness over speed); BrushStroke restores the stored patch directly.
4. Keep upstream's `undo_last_point` button working by routing it to the stack.
5. **Heavy ops clear the stack (DECIDED):** any propagation / matting / batch-refine run calls `QUndoStack.clear()` with a status-bar note. Rationale: propagation rewrites hundreds of masks (too big to snapshot), and stale BrushStroke patch-diffs applied over post-propagation masks would produce corrupted hybrids. Undo covers the editing session BETWEEN heavy ops.

**Acceptance:** 20-step mixed undo/redo (points, boxes, drags, strokes) leaves masks and PointTable consistent with a fresh replay; running propagation clears history and undo does nothing afterwards.

### M11 — Timeline Keyframe Markers + Autosave

1. **Markers:** paint ticks on `FrameSlider` — one color for frames with points/boxes, another for brush-edited frames (from M8's registry); in/out markers already exist, follow their painting code.
2. **Autosave:** reuse the existing session-save path on a `QTimer` (default every 3 min, setting-controlled) writing to `temp/autosave_session.json`; on startup, if an autosave is newer than the last explicit save, offer recovery.

**Acceptance:** markers update live as prompts are added/removed; kill -9 during work → relaunch offers recovery with points/boxes intact.

### M12 (Stretch) — Proxy Workflow

Work at 1080p proxy end-to-end; on export, upscale masks/alphas to source resolution, with optional per-frame HQ refine pass (M4) at full res. Large design surface (two frame dirs, coordinate scaling for all prompts, export remap) — write a short design note for approval before implementing.

### M13 (Stretch) — Headless CLI Mode

`python -m sammie.cli --project x.sammie --track --matting vitmatte --export prores4444 out.mov`. Requires managers to run without a QApplication — audit Qt imports in compute paths (progress dialogs move behind an interface with a no-op CLI impl). Feasible after M7 since workers already decouple compute from GUI.

### M14 (Stretch) — Multi-Channel EXR Export

One EXR per frame with each object as a named channel (OpenEXR already in deps). New `ExportFormat` subclass in `export_formats.py`; verify channel naming against Nuke import.

---

## 4. Cross-Cutting Rules

- **Style:** match upstream (manager classes, `_notify` callbacks, tqdm for console). New long ops use M7 worker infra, not `processEvents`.
- **Settings:** all new options via `settings_manager.py` (follow existing session/persistent key examples).
- **Models:** every new checkpoint via `model_downloader.py::ensure_models` with size/hash; never bundle weights.
- **Device support:** must run on CPU (slow ok, crash not); gate half-precision by device type like upstream.
- **Compat:** old project files must load; missing keys default gracefully.
- **Testing:** no upstream test suite. Per milestone: manual checklist in `FORK_NOTES.md` + headless unit tests under `tests/` for pure functions (`build_trimap`, box clamping, crop remapping, cache invalidation, undo command apply/revert).
- **Upstream policy:** pinned at v2.3.4.a1; cherry-pick individual upstream fixes only when needed. No routine rebasing.

## 5. Recommended Order & Sizing

| # | Milestone | Size | Depends on | Rationale |
|---|---|---|---|---|
| 1 | M1 Box prompt | M | — | Highest value, lowest risk |
| 2 | M2 Draggable points | M | — | Coordinate with M1 (both touch ImageViewer); commit-on-release only |
| 3 | M7 Worker threads | M–L | — | Do early; M4/M5 batch jobs build on it; unlocks M2 live-preview follow-up |
| 4 | M4 Crop refinement | M | M1, M7 | Batch mode uses workers; introduces the image predictor M3 needs |
| 5 | M3 Multimask | S–M | M1, **M4** | Rebuilt on M4's image predictor + `add_new_mask` — no vendored edits |
| 6 | M8 Mask brush | M | M7 recommended | |
| 7 | M9 RAM cache | M | — | Touches every read path; land after brush to cover invalidation |
| 8 | M10 Undo stack | M | M1, M2, M8 | Commands cover all edit types |
| 9 | M5 ViTMatte trimap | L | M4, M7 | |
| 10 | M11 Markers + autosave | S | M8 | |
| — | M6 / M12 / M13 / M14 | — | stable core | Stretch, in any order |

## 6. Resolved Decisions (grilling session, 2026-07-06)

All former open questions are resolved and folded into the milestone sections above. Summary:

1. **Fork strategy:** personal tool; upstream pinned at v2.3.4.a1; no rebase policy; cherry-pick only.
2. **Hardware target:** NVIDIA 8–12 GB VRAM; CPU is a crash-not floor.
3. **M1:** box activation = toolbar toggle AND hold-`B`. `box=` forces `clear_old_points=True` → boxed objects always resend box + all points in one call.
4. **M2:** commit on release only; live drag-preview deferred until after M7.
5. **M3:** image predictor multimask + `add_new_mask()`; no vendored edits; depends on M4; ordered after M4.
6. **M4:** in-place refine with refine-owned `temp/masks_prerefine/` backup + Revert action; `masks_backup` is off-limits (duplicate-frame handler owns it); refine runs after dup-frame processing.
7. **M5:** ViTMatte-B default, Small selectable; unknown-band tiling mandatory at 4K; output format verified as `temp/matting/{frame:05d}/{obj}.png`.
8. **M7:** interactive segmentation stays synchronous on GUI thread + busy guard; invariant = no two threads touch `inference_state` concurrently; inline status-bar progress + cancel; load-path `processEvents` out of scope.
9. **M8:** ask once per propagation (remember choice) for edited frames; brush edits delete stale matting PNGs for that frame/object.
10. **M9:** cache default min(2 GB, 25% RAM); worker invalidation marshaled through queued signals.
11. **M10:** propagation/matting/batch-refine clear the undo stack; undo covers the session between heavy ops.
