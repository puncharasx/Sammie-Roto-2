"""Tests for M2 Draggable Points — PointManager.move_point and clamping logic"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
from sammie.core import PointManager


class TestMovePoint:
    """Tests for PointManager.move_point()"""

    @pytest.fixture(autouse=True)
    def _mock_settings(self):
        """Patch get_settings_manager for all tests in this class"""
        mock_settings = MagicMock()
        patcher = patch("sammie.core.get_settings_manager", return_value=mock_settings)
        patcher.start()
        yield mock_settings
        patcher.stop()

    def _make_manager(self, points=None):
        """Create a PointManager with optional initial points"""
        pm = PointManager()
        pm.callbacks = []
        if points:
            pm.points = [dict(p) for p in points]  # deep copy
        return pm

    def test_move_point_updates_coords(self):
        """move_point should change x,y of the matching point"""
        initial = [{"frame": 0, "object_id": 1, "positive": True, "x": 100, "y": 200}]
        pm = self._make_manager(initial)

        result = pm.move_point(0, 1, 100, 200, 150, 250)

        assert result is not None
        assert result["x"] == 150
        assert result["y"] == 250
        assert pm.points[0]["x"] == 150
        assert pm.points[0]["y"] == 250

    def test_move_point_preserves_other_fields(self):
        """move_point should not change frame, object_id, or positive"""
        initial = [{"frame": 5, "object_id": 2, "positive": False, "x": 10, "y": 20}]
        pm = self._make_manager(initial)

        result = pm.move_point(5, 2, 10, 20, 30, 40)

        assert result["frame"] == 5
        assert result["object_id"] == 2
        assert result["positive"] is False

    def test_move_point_notifies_callback(self):
        """move_point should fire 'move_point' action via _notify"""
        initial = [{"frame": 0, "object_id": 1, "positive": True, "x": 10, "y": 20}]
        pm = self._make_manager(initial)
        callback = MagicMock()
        pm.add_callback(callback)

        pm.move_point(0, 1, 10, 20, 30, 40)

        callback.assert_called_once()
        action = callback.call_args[0][0]
        kwargs = callback.call_args[1]
        assert action == "move_point"
        assert kwargs["old_x"] == 10
        assert kwargs["old_y"] == 20
        assert kwargs["new_x"] == 30
        assert kwargs["new_y"] == 40

    def test_move_point_saves_persistence(self, _mock_settings):
        """move_point should call settings_mgr.save_points"""
        initial = [{"frame": 0, "object_id": 1, "positive": True, "x": 10, "y": 20}]
        pm = self._make_manager(initial)

        pm.move_point(0, 1, 10, 20, 30, 40)

        _mock_settings.save_points.assert_called_once_with(pm.points)

    def test_move_point_not_found_returns_none(self):
        """move_point should return None when no matching point exists"""
        initial = [{"frame": 0, "object_id": 1, "positive": True, "x": 10, "y": 20}]
        pm = self._make_manager(initial)

        result = pm.move_point(0, 1, 999, 999, 30, 40)

        assert result is None
        assert pm.points[0]["x"] == 10
        assert pm.points[0]["y"] == 20

    def test_move_point_wrong_frame_returns_none(self):
        """move_point should return None when frame doesn't match"""
        initial = [{"frame": 0, "object_id": 1, "positive": True, "x": 10, "y": 20}]
        pm = self._make_manager(initial)

        result = pm.move_point(99, 1, 10, 20, 30, 40)

        assert result is None

    def test_move_point_wrong_object_returns_none(self):
        """move_point should return None when object_id doesn't match"""
        initial = [{"frame": 0, "object_id": 1, "positive": True, "x": 10, "y": 20}]
        pm = self._make_manager(initial)

        result = pm.move_point(0, 99, 10, 20, 30, 40)

        assert result is None

    def test_move_point_multiple_points_only_moves_target(self):
        """move_point should only change the matching point, not others"""
        initial = [
            {"frame": 0, "object_id": 1, "positive": True, "x": 10, "y": 20},
            {"frame": 0, "object_id": 1, "positive": False, "x": 50, "y": 60},
            {"frame": 1, "object_id": 1, "positive": True, "x": 10, "y": 20},
        ]
        pm = self._make_manager(initial)

        pm.move_point(0, 1, 10, 20, 30, 40)

        assert pm.points[0]["x"] == 30
        assert pm.points[0]["y"] == 40
        assert pm.points[1]["x"] == 50
        assert pm.points[1]["y"] == 60
        assert pm.points[2]["x"] == 10
        assert pm.points[2]["y"] == 20


class TestClampToBounds:
    """Tests for coordinate clamping logic"""

    def test_clamp_within_bounds(self):
        """Coords inside bounds should pass through unchanged"""
        from sammie.gui_widgets import DraggablePointItem
        x, y = DraggablePointItem.clamp_to_bounds(100, 200, 1920, 1080)
        assert x == 100
        assert y == 200

    def test_clamp_negative_x(self):
        """Negative x should clamp to 0"""
        from sammie.gui_widgets import DraggablePointItem
        x, y = DraggablePointItem.clamp_to_bounds(-5, 200, 1920, 1080)
        assert x == 0
        assert y == 200

    def test_clamp_negative_y(self):
        """Negative y should clamp to 0"""
        from sammie.gui_widgets import DraggablePointItem
        x, y = DraggablePointItem.clamp_to_bounds(100, -10, 1920, 1080)
        assert x == 100
        assert y == 0

    def test_clamp_exceeds_width(self):
        """x >= width should clamp to width - 1"""
        from sammie.gui_widgets import DraggablePointItem
        x, y = DraggablePointItem.clamp_to_bounds(2000, 200, 1920, 1080)
        assert x == 1919
        assert y == 200

    def test_clamp_exceeds_height(self):
        """y >= height should clamp to height - 1"""
        from sammie.gui_widgets import DraggablePointItem
        x, y = DraggablePointItem.clamp_to_bounds(100, 1100, 1920, 1080)
        assert x == 100
        assert y == 1079

    def test_clamp_corner_case(self):
        """Coords at exact boundary should be valid"""
        from sammie.gui_widgets import DraggablePointItem
        x, y = DraggablePointItem.clamp_to_bounds(0, 0, 1920, 1080)
        assert x == 0
        assert y == 0
        x, y = DraggablePointItem.clamp_to_bounds(1919, 1079, 1920, 1080)
        assert x == 1919
        assert y == 1079
