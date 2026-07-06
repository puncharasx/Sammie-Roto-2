# tests/test_workers.py
"""Unit tests for worker thread classes (M7 Worker-Thread Refactor).

Tests the TrackingWorker, MattingWorker, and RemovalWorker classes
for initialization, cancellation, and error handling. Signal tests
use QCoreApplication + processEvents instead of pytest-qt fixtures.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch

# PySide6 needs a QCoreApplication before signals/slots work
from PySide6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)


from sammie.workers import BaseWorker, TrackingWorker, MattingWorker, RemovalWorker


# ---------------------------------------------------------------------------
# BaseWorker
# ---------------------------------------------------------------------------

class TestBaseWorker:
    """Tests for BaseWorker base class."""

    def test_initial_state(self):
        worker = BaseWorker()
        assert worker.is_cancelled is False

    def test_request_cancel(self):
        worker = BaseWorker()
        worker.request_cancel()
        assert worker.is_cancelled is True

    def test_emit_progress_captures_value(self):
        """_emit_progress should emit progress signal with correct percentage."""
        worker = BaseWorker()
        received = []
        worker.progress.connect(lambda v: received.append(v))

        worker._emit_progress(50, 100)
        _app.processEvents()

        assert received == [50]

    def test_emit_progress_zero_total(self):
        """_emit_progress should not emit when total is zero."""
        worker = BaseWorker()
        received = []
        worker.progress.connect(lambda v: received.append(v))

        worker._emit_progress(0, 0)
        _app.processEvents()

        assert received == []

    def test_emit_frame_done(self):
        """_emit_frame_done should emit frame_done signal with frame index."""
        worker = BaseWorker()
        received = []
        worker.frame_done.connect(lambda v: received.append(v))

        worker._emit_frame_done(42)
        _app.processEvents()

        assert received == [42]


# ---------------------------------------------------------------------------
# TrackingWorker
# ---------------------------------------------------------------------------

class TestTrackingWorker:
    """Tests for TrackingWorker."""

    def test_initialization(self):
        predictor = MagicMock()
        inference_state = MagicMock()

        worker = TrackingWorker(
            predictor=predictor,
            inference_state=inference_state,
            start_frame_idx=10,
            max_frame_num_to_track=50,
            reverse=True,
            display_update_frequency=3,
            total_frames=100,
        )

        assert worker.predictor is predictor
        assert worker.inference_state is inference_state
        assert worker.start_frame_idx == 10
        assert worker.max_frame_num_to_track == 50
        assert worker.reverse is True
        assert worker.display_update_frequency == 3
        assert worker.total_frames == 100

    @patch('sammie.workers.core')
    @patch('sammie.workers.cv2')
    @patch('sammie.workers.os')
    def test_run_success(self, mock_os, mock_cv2, mock_core):
        """TrackingWorker should emit finished on successful completion."""
        predictor = MagicMock()
        inference_state = MagicMock()

        # Mock mask logits: > 0 so mask is saved
        import torch
        logits_0 = torch.tensor([[[[1.0]]]])
        logits_1 = torch.tensor([[[[1.0]]]])

        predictor.propagate_in_video.return_value = iter([
            (0, [0], logits_0),
            (1, [0], logits_1),
        ])

        worker = TrackingWorker(
            predictor=predictor,
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
            display_update_frequency=1,
        )

        received_finished = []
        received_frame_done = []
        worker.finished.connect(lambda v: received_finished.append(v))
        worker.frame_done.connect(lambda v: received_frame_done.append(v))

        worker.run()
        _app.processEvents()

        assert len(received_finished) == 1
        result = received_finished[0]
        assert result['last_frame_idx'] == 1
        assert result['cancelled'] is False
        assert len(received_frame_done) >= 2

    @patch('sammie.workers.core')
    @patch('sammie.workers.cv2')
    @patch('sammie.workers.os')
    def test_run_cancellation(self, mock_os, mock_cv2, mock_core):
        """TrackingWorker should stop and emit cancelled when request_cancel is called."""
        import torch
        predictor = MagicMock()
        inference_state = MagicMock()

        def mock_propagate(*args, **kwargs):
            for i in range(100):
                if worker.is_cancelled:
                    break
                yield (i, [0], torch.tensor([[[[1.0]]]]))

        predictor.propagate_in_video.return_value = mock_propagate()

        worker = TrackingWorker(
            predictor=predictor,
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=99,
            display_update_frequency=1,
        )

        received = []
        worker.finished.connect(lambda v: received.append(v))

        worker.request_cancel()
        worker.run()
        _app.processEvents()

        assert len(received) == 1
        assert received[0]['cancelled'] is True

    @patch('sammie.workers.core')
    @patch('sammie.workers.cv2')
    @patch('sammie.workers.os')
    def test_run_error(self, mock_os, mock_cv2, mock_core):
        """TrackingWorker should emit error on exception."""
        predictor = MagicMock()
        inference_state = MagicMock()

        predictor.propagate_in_video.side_effect = RuntimeError("GPU out of memory")

        worker = TrackingWorker(
            predictor=predictor,
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=10,
        )

        received = []
        worker.error.connect(lambda v: received.append(v))
        worker.run()
        _app.processEvents()

        assert len(received) == 1
        assert "GPU out of memory" in received[0]


# ---------------------------------------------------------------------------
# MattingWorker
# ---------------------------------------------------------------------------

class TestMattingWorker:
    """Tests for MattingWorker."""

    def test_initialization(self):
        matting_manager = MagicMock()
        points_list = [{'frame': 0, 'object_id': 0, 'x': 100, 'y': 100}]

        worker = MattingWorker(
            matting_manager=matting_manager,
            points_list=points_list,
            combined=True,
        )

        assert worker.matting_manager is matting_manager
        assert worker.points_list == points_list
        assert worker.combined is True

    def test_run_no_processor(self):
        """MattingWorker should emit error when processor is None."""
        matting_manager = MagicMock()
        matting_manager.processor = None

        worker = MattingWorker(
            matting_manager=matting_manager,
            points_list=[],
        )

        received = []
        worker.error.connect(lambda v: received.append(v))
        worker.run()
        _app.processEvents()

        assert len(received) == 1
        assert "not loaded" in received[0].lower()

    @patch('sammie.workers.DeviceManager')
    @patch('sammie.workers.core')
    def test_run_no_objects(self, mock_core, mock_dm):
        """MattingWorker should emit error when no objects found."""
        matting_manager = MagicMock()
        matting_manager.processor = MagicMock()
        matting_manager._get_frame_range.return_value = (0, 10, 11)

        worker = MattingWorker(
            matting_manager=matting_manager,
            points_list=[],  # No points
        )

        received = []
        worker.error.connect(lambda v: received.append(v))
        worker.run()
        _app.processEvents()

        assert len(received) == 1
        assert "No objects" in received[0]


# ---------------------------------------------------------------------------
# RemovalWorker
# ---------------------------------------------------------------------------

class TestRemovalWorker:
    """Tests for RemovalWorker."""

    def test_initialization(self):
        removal_manager = MagicMock()
        points_list = [{'frame': 0, 'object_id': 0, 'x': 100, 'y': 100}]

        worker = RemovalWorker(
            removal_manager=removal_manager,
            points_list=points_list,
            method='cv',
        )

        assert worker.removal_manager is removal_manager
        assert worker.points_list == points_list
        assert worker.method == 'cv'

    def test_run_unknown_method(self):
        """RemovalWorker should emit error for unknown method."""
        removal_manager = MagicMock()

        worker = RemovalWorker(
            removal_manager=removal_manager,
            points_list=[],
            method='unknown',
        )

        received = []
        worker.error.connect(lambda v: received.append(v))
        worker.run()
        _app.processEvents()

        assert len(received) == 1
        assert "Unknown removal method" in received[0]


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestWorkerIntegration:
    """Integration tests for worker thread lifecycle."""

    def test_busy_flag_management(self):
        """Busy flag should be managed correctly during worker lifecycle."""
        from sammie.sammie import SamManager

        manager = SamManager()
        assert manager.is_busy is False

        manager._busy = True
        assert manager.is_busy is True

        manager._on_tracking_finished({'cancelled': False, 'last_frame_idx': 0})
        assert manager.is_busy is False

    def test_busy_flag_on_error(self):
        """Busy flag should reset on error."""
        from sammie.sammie import SamManager

        manager = SamManager()
        manager._busy = True

        manager._on_tracking_error("test error")
        assert manager.is_busy is False

    def test_cancel_tracking(self):
        """cancel_tracking should request cancel on worker."""
        from sammie.sammie import SamManager

        manager = SamManager()
        mock_worker = MagicMock()
        manager._worker = mock_worker

        manager.cancel_tracking()
        mock_worker.request_cancel.assert_called_once()

    def test_matting_busy_flag(self):
        """MattingManager busy flag lifecycle."""
        from sammie.matting import MattingManager

        manager = MattingManager()
        assert manager.is_busy is False

        manager._busy = True
        manager._on_matting_finished({'cancelled': False, 'propagated': True})
        assert manager.is_busy is False
        assert manager.propagated is True

    def test_matting_cancel(self):
        """cancel_matting should request cancel on worker."""
        from sammie.matting import MattingManager

        manager = MattingManager()
        mock_worker = MagicMock()
        manager._worker = mock_worker

        manager.cancel_matting()
        mock_worker.request_cancel.assert_called_once()

    def test_removal_busy_flag(self):
        """RemovalManager busy flag lifecycle."""
        from sammie.removal import RemovalManager

        manager = RemovalManager()
        assert manager.is_busy is False

        manager._busy = True
        manager._on_removal_finished({'cancelled': False, 'propagated': True})
        assert manager.is_busy is False
        assert manager.propagated is True

    def test_removal_cancel(self):
        """cancel_removal should request cancel on worker."""
        from sammie.removal import RemovalManager

        manager = RemovalManager()
        mock_worker = MagicMock()
        manager._worker = mock_worker

        manager.cancel_removal()
        mock_worker.request_cancel.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
