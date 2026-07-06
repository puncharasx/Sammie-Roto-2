# sammie/workers.py
"""
Worker threads for long-running compute operations (tracking, matting, removal).

Moves propagation, matting, and removal off the GUI thread. Every worker
emits progress/frame_done/finished/error signals and supports cooperative
cancellation via request_cancel().

Thread-safety invariant: interactive single-frame segmentation (segment_image
on click, preview_point on shift-hover) stays synchronous on the GUI thread.
No two threads touch inference_state concurrently — while a worker runs, all
prompt input is disabled via a busy flag. One compute worker at a time.
"""

import os
import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal

from sammie import core
from sammie.core import DeviceManager, VideoInfo


class BaseWorker(QThread):
    """Base worker thread for compute operations.

    Signals:
        progress(int): Percentage progress (0-100).
        frame_done(int): Emitted after each frame is processed. GUI can
            nudge the frame slider to this index.
        finished(object): Emitted on success with a result dict.
        error(str): Emitted on failure with an error message.
    """

    progress = Signal(int)
    frame_done = Signal(int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_requested = False

    def request_cancel(self):
        """Request cooperative cancellation. Checked between frames."""
        self._cancel_requested = True

    @property
    def is_cancelled(self):
        return self._cancel_requested

    def _emit_progress(self, current, total):
        """Emit progress percentage."""
        if total > 0:
            self.progress.emit(int(current * 100 / total))

    def _emit_frame_done(self, frame_idx):
        """Emit frame_done signal for GUI slider nudge."""
        self.frame_done.emit(frame_idx)


class TrackingWorker(BaseWorker):
    """Worker thread for SAM2 propagation (tracking).

    Moves the body of SamManager._propagate() off the GUI thread.
    The manager keeps its API but dispatches to this worker and
    returns via signals.

    Args:
        predictor: SAM2 video predictor instance.
        inference_state: Current inference state for the predictor.
        start_frame_idx: Frame to start propagating from.
        max_frame_num_to_track: How many additional frames to propagate,
            or None to propagate to the end of the video.
        reverse: If True, propagate backward toward frame 0.
        display_update_frequency: How often to emit frame_done (every N frames).
        total_frames: Total number of frames in the video.
        parent: Parent QObject.
    """

    def __init__(self, predictor, inference_state, start_frame_idx,
                 max_frame_num_to_track, reverse=False,
                 display_update_frequency=5, total_frames=None, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.inference_state = inference_state
        self.start_frame_idx = start_frame_idx
        self.max_frame_num_to_track = max_frame_num_to_track
        self.reverse = reverse
        self.display_update_frequency = display_update_frequency
        self.total_frames = total_frames or VideoInfo.total_frames

    def run(self):
        """Execute the propagation loop on a background thread."""
        try:
            total_frames = (
                (self.max_frame_num_to_track + 1)
                if self.max_frame_num_to_track is not None
                else self.total_frames
            )

            last_frame_idx = None

            for out_frame_idx, out_obj_ids, out_mask_logits in \
                    self.predictor.propagate_in_video(
                        self.inference_state,
                        start_frame_idx=self.start_frame_idx,
                        max_frame_num_to_track=self.max_frame_num_to_track,
                        reverse=self.reverse):

                if self._cancel_requested:
                    break

                # Save masks to disk
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask_filename = os.path.join(
                        core.mask_dir,
                        f"{out_frame_idx:05d}",
                        f"{out_obj_id}.png"
                    )
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    mask = (mask * 255).astype(np.uint8)
                    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                    cv2.imwrite(mask_filename, mask)

                last_frame_idx = out_frame_idx

                # Emit progress
                frames_processed = abs(out_frame_idx - self.start_frame_idx) + 1
                self._emit_progress(frames_processed, total_frames)

                # Emit frame_done at the specified frequency
                if out_frame_idx % self.display_update_frequency == 0:
                    self._emit_frame_done(out_frame_idx)

            # Always emit final frame_done
            if last_frame_idx is not None:
                self._emit_frame_done(last_frame_idx)

            # Release VRAM on cancel (matches upstream cleanup behavior)
            if self._cancel_requested:
                DeviceManager.clear_cache()

            self.finished.emit({
                'last_frame_idx': last_frame_idx,
                'cancelled': self._cancel_requested,
            })

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"Tracking failed: {e}\n\n{tb}")


class MattingWorker(BaseWorker):
    """Worker thread for matting operations.

    Moves the matting loop off the GUI thread. Supports both forward
    and backward processing with multiple keyframes per object.

    Args:
        matting_manager: The MattingManager instance (MatAnyManager or VideoMaMaManager).
        points_list: List of point dictionaries.
        combined: If True, combine all objects into one pass.
        parent: Parent QObject.
    """

    def __init__(self, matting_manager, points_list, combined=False, parent=None):
        super().__init__(parent)
        self.matting_manager = matting_manager
        self.points_list = points_list
        self.combined = combined

    def run(self):
        """Execute the matting loop on a background thread."""
        try:
            from sammie.matting import get_settings_manager
            settings_mgr = get_settings_manager()

            if self.matting_manager.processor is None:
                self.error.emit("Matting model not loaded")
                return

            DeviceManager.clear_cache()
            device = DeviceManager.get_device()
            frame_count = VideoInfo.total_frames

            start_frame, end_frame, frames_to_process = \
                self.matting_manager._get_frame_range()

            # Get unique object IDs
            object_ids = sorted(list(set(
                point['object_id'] for point in self.points_list
                if 'object_id' in point
            )))
            if not object_ids:
                self.error.emit("No objects found for matting")
                return

            # Find keyframes per object
            object_keyframes = {}
            for object_id in object_ids:
                keyframes = sorted(list(set(
                    point['frame'] for point in self.points_list
                    if point.get('object_id') == object_id
                    and start_frame <= point['frame'] <= end_frame
                )))
                if keyframes:
                    object_keyframes[object_id] = keyframes

            if not object_keyframes:
                self.error.emit("No valid keyframes found")
                return

            # Combined mode
            combine_ids = None
            if self.combined and len(object_ids) > 1:
                combine_ids = object_ids
                earliest_keyframe = min(kf[0] for kf in object_keyframes.values())
                object_ids = [0]
                object_keyframes = {0: [earliest_keyframe]}

            # Calculate total operations
            total_operations = 0
            for object_id, keyframes in object_keyframes.items():
                first_keyframe = keyframes[0]
                total_operations += first_keyframe - start_frame
                for i in range(len(keyframes)):
                    if i == len(keyframes) - 1:
                        total_operations += end_frame - keyframes[i] + 1
                    else:
                        total_operations += keyframes[i + 1] - keyframes[i]

            # Create matting directory
            os.makedirs(core.matting_dir, exist_ok=True)

            # If combined mode, delete existing non-zero matting files
            if self.combined and os.path.exists(core.matting_dir):
                for frame_dirname in os.listdir(core.matting_dir):
                    frame_dir = os.path.join(core.matting_dir, frame_dirname)
                    if os.path.isdir(frame_dir):
                        for f in os.listdir(frame_dir):
                            if f != "0.png":
                                os.remove(os.path.join(frame_dir, f))

            images = self.matting_manager._collect_image_paths(start_frame, end_frame)
            operations_completed = 0
            display_update_frequency = settings_mgr.get_app_setting(
                "display_update_frequency", 5
            )

            # Process each object
            for object_id, keyframes in object_keyframes.items():
                if self._cancel_requested:
                    break

                # Process segments for this object
                success = self._process_object_matting(
                    images, object_id, keyframes, end_frame + 1, device,
                    start_frame, combine_ids, total_operations,
                    operations_completed, display_update_frequency
                )

                if not success:
                    break

                # Update operations completed
                first_keyframe = keyframes[0]
                operations_completed += first_keyframe - start_frame
                for i in range(len(keyframes)):
                    if i == len(keyframes) - 1:
                        operations_completed += end_frame - keyframes[i] + 1
                    else:
                        operations_completed += keyframes[i + 1] - keyframes[i]

            # Final cleanup
            DeviceManager.clear_cache()

            self.finished.emit({
                'cancelled': self._cancel_requested,
                'propagated': not self._cancel_requested and frame_count == frames_to_process,
            })

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"Matting failed: {e}\n\n{tb}")

    def _process_object_matting(self, images, object_id, keyframes, frame_count,
                                 device, start_frame, combine_ids,
                                 total_operations, operations_completed,
                                 display_update_frequency):
        """Process a single object using multiple keyframes."""
        first_keyframe = keyframes[0]

        # Load first keyframe mask
        mask, original_size = self.matting_manager._load_mask_for_matting(
            object_id, first_keyframe, device, combine_ids=combine_ids
        )
        if mask is None:
            return False

        # Single frame special case
        if len(images) == 1:
            return self.matting_manager._process_single_frame(
                images[0], mask, object_id, original_size, device
            )

        current_operations = operations_completed

        # Process backward from first keyframe
        if first_keyframe > start_frame:
            success = self._process_backward_segment(
                images, mask, object_id, first_keyframe, original_size,
                device, current_operations, total_operations,
                display_update_frequency, start_frame_offset=start_frame
            )
            if not success:
                return False
            current_operations += first_keyframe - start_frame

        # Process forward segments between keyframes
        for i in range(len(keyframes)):
            if self._cancel_requested:
                return False

            current_keyframe = keyframes[i]

            # Refresh mask for each segment
            mask, original_size = self.matting_manager._load_mask_for_matting(
                object_id, current_keyframe, device, combine_ids=combine_ids
            )
            if mask is None:
                return False

            # Determine end frame
            if i == len(keyframes) - 1:
                end_frame = frame_count
            else:
                end_frame = keyframes[i + 1]

            # Process forward segment
            if end_frame > current_keyframe:
                success = self._process_forward_segment(
                    images, mask, object_id, current_keyframe, original_size,
                    device, current_operations, total_operations,
                    display_update_frequency, end_frame, start_frame_offset=start_frame
                )
                if not success:
                    return False
                current_operations += end_frame - current_keyframe

        return True

    def _process_forward_segment(self, images, mask, object_id, start_frame,
                                  original_size, device, operations_completed,
                                  total_operations, display_update_frequency,
                                  end_frame=None, start_frame_offset=0):
        """Process frames forward from start_frame."""
        if end_frame is None:
            end_frame = start_frame + len(images)

        try:
            for frame_number in range(start_frame, end_frame):
                if self._cancel_requested:
                    return False

                # Map to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.matting_manager._resize_image(img)
                img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

                if frame_number == start_frame:
                    output_prob = self.matting_manager.processor.step(img, mask, objects=[1])
                    for _ in range(10):  # Warmup
                        output_prob = self.matting_manager.processor.step(img, first_frame_pred=True)
                        DeviceManager.clear_cache()
                else:
                    output_prob = self.matting_manager.processor.step(img)

                # Convert to matte
                mat = self.matting_manager.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self.matting_manager._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(
                    core.matting_dir, f"{frame_number:05d}", f"{object_id}.png"
                )
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                DeviceManager.clear_cache()

                # Emit signals
                current_progress = int(
                    ((operations_completed + (frame_number - start_frame) + 1) * 100)
                    / total_operations
                )
                self.progress.emit(current_progress)

                if frame_number % display_update_frequency == 0:
                    self._emit_frame_done(frame_number)

            return True

        except Exception as e:
            print(f"Error in forward processing: {e}")
            return False

    def _process_backward_segment(self, images, mask, object_id, start_frame,
                                   original_size, device, operations_completed,
                                   total_operations, display_update_frequency,
                                   start_frame_offset=0):
        """Process frames backward from start_frame."""
        try:
            for frame_number in range(start_frame, start_frame_offset - 1, -1):
                if self._cancel_requested:
                    return False

                # Map to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.matting_manager._resize_image(img)
                img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

                if frame_number == start_frame:
                    output_prob = self.matting_manager.processor.step(img, mask, objects=[1])
                    for _ in range(10):  # Warmup
                        output_prob = self.matting_manager.processor.step(img, first_frame_pred=True)
                        DeviceManager.clear_cache()
                else:
                    output_prob = self.matting_manager.processor.step(img)

                # Convert to matte
                mat = self.matting_manager.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self.matting_manager._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(
                    core.matting_dir, f"{frame_number:05d}", f"{object_id}.png"
                )
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                DeviceManager.clear_cache()

                # Emit signals
                operations_completed += 1
                self.progress.emit(operations_completed * 100 // total_operations)

                if frame_number % display_update_frequency == 0:
                    self._emit_frame_done(frame_number)

            return True

        except Exception as e:
            print(f"Error in backward processing: {e}")
            return False


class RemovalWorker(BaseWorker):
    """Worker thread for object removal operations.

    Moves the removal loop off the GUI thread. Supports both OpenCV
    inpainting and MiniMax-Remover.

    Args:
        removal_manager: The RemovalManager instance.
        points_list: List of point dictionaries.
        method: 'cv' for OpenCV inpainting, 'minimax' for MiniMax-Remover.
        parent: Parent QObject.
    """

    def __init__(self, removal_manager, points_list, method='cv', parent=None):
        super().__init__(parent)
        self.removal_manager = removal_manager
        self.points_list = points_list
        self.method = method

    def run(self):
        """Execute the removal loop on a background thread."""
        try:
            if self.method == 'cv':
                self._run_cv_removal()
            else:
                self.error.emit(f"Unknown removal method: {self.method}")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(f"Removal failed: {e}\n\n{tb}")

    def _run_cv_removal(self):
        """Run OpenCV inpainting removal on a background thread."""
        from sammie.matting import get_settings_manager
        settings_mgr = get_settings_manager()

        frame_count = VideoInfo.total_frames
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)

        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1

        # Get settings
        inpaint_method = settings_mgr.get_session_setting("inpaint_method", "Telea")
        inpaint_radius = settings_mgr.get_session_setting("inpaint_radius", 3)
        grow = settings_mgr.get_session_setting("grow", 0)
        inpaint_grow = settings_mgr.get_session_setting("inpaint_grow", 0) + grow
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        if inpaint_method == "Telea":
            cv2_method = cv2.INPAINT_TELEA
        elif inpaint_method == "Navier-Stokes":
            cv2_method = cv2.INPAINT_NS
        else:
            cv2_method = cv2.INPAINT_TELEA

        # Get unique object IDs
        object_ids = sorted(list(set(
            p['object_id'] for p in self.points_list if 'object_id' in p
        )))
        if not object_ids:
            self.error.emit("No objects found for removal")
            return

        # Create output directory
        os.makedirs(core.removal_dir, exist_ok=True)
        extension = core.get_frame_extension()
        operations_completed = 0

        for frame_number in range(start_frame, end_frame + 1):
            if self._cancel_requested:
                break

            frame_filename = os.path.join(
                core.frames_dir, f"{frame_number:05d}.{extension}"
            )
            if not os.path.exists(frame_filename):
                operations_completed += 1
                self._emit_progress(operations_completed, frames_to_process)
                continue

            frame = cv2.imread(frame_filename)
            if frame is None:
                operations_completed += 1
                self._emit_progress(operations_completed, frames_to_process)
                continue

            # Combine masks
            combined_mask = np.zeros(frame.shape[:2], np.uint8)
            for object_id in object_ids:
                mask_filename = os.path.join(
                    core.mask_dir, f"{frame_number:05d}", f"{object_id}.png"
                )
                if os.path.exists(mask_filename):
                    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Skip if no mask
            if not np.any(combined_mask):
                output_filename = os.path.join(
                    core.removal_dir, f"{frame_number:05d}.png"
                )
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, frame)
                operations_completed += 1
                self._emit_progress(operations_completed, frames_to_process)
                if frame_number % display_update_frequency == 0:
                    self._emit_frame_done(frame_number)
                continue

            # Apply grow/shrink
            if inpaint_grow != 0:
                combined_mask = core.grow_shrink(combined_mask, inpaint_grow)

            # Run inpainting
            try:
                result = cv2.inpaint(frame, combined_mask, inpaint_radius, cv2_method)
                output_filename = os.path.join(
                    core.removal_dir, f"{frame_number:05d}.png"
                )
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, result)
            except Exception as e:
                print(f"Error inpainting frame {frame_number}: {e}")
                output_filename = os.path.join(
                    core.removal_dir, f"{frame_number:05d}.png"
                )
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, frame)

            operations_completed += 1
            self._emit_progress(operations_completed, frames_to_process)

            if frame_number % display_update_frequency == 0:
                self._emit_frame_done(frame_number)

        self.finished.emit({
            'cancelled': self._cancel_requested,
            'propagated': not self._cancel_requested and frame_count == frames_to_process,
        })
