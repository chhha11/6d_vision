import unittest
from unittest.mock import Mock, patch

from core.camera_identity import CameraDevice, PinnedCamera


class CameraIdentityTest(unittest.TestCase):
    def test_bind_records_physical_path_and_resolves_new_index(self):
        first = [CameraDevice(0, 'USB Camera', 'astra-path'),
                 CameraDevice(1, 'Integrated Camera', 'laptop-path')]
        reordered = [CameraDevice(0, 'Integrated Camera', 'laptop-path'),
                     CameraDevice(1, 'USB Camera', 'astra-path')]
        with patch('core.camera_identity.enumerate_cameras', side_effect=[first, reordered]):
            identity = PinnedCamera.bind(0, 'USB Camera')
            self.assertEqual(identity.resolve(), 1)

    def test_bind_rejects_wrong_friendly_name(self):
        with patch('core.camera_identity.enumerate_cameras', return_value=[
                CameraDevice(0, 'Integrated Camera', 'laptop-path')]):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                PinnedCamera.bind(0, 'USB Camera')

    def test_resolve_rejects_missing_physical_device(self):
        identity = PinnedCamera('USB Camera', 'astra-path')
        with patch('core.camera_identity.enumerate_cameras', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                identity.resolve()

    def test_open_uses_new_index_for_same_device(self):
        identity = PinnedCamera('USB Camera', 'astra-path')
        cap = Mock()
        opener = Mock(return_value=cap)
        with patch('core.camera_identity.enumerate_cameras', return_value=[
                CameraDevice(0, 'Integrated Camera', 'laptop-path'),
                CameraDevice(1, 'USB Camera', 'astra-path')]):
            self.assertIs(identity.open(opener, 640, 480), cap)
        opener.assert_called_once_with(1, 640, 480)

    def test_open_releases_camera_if_order_changes_during_open(self):
        identity = PinnedCamera('USB Camera', 'astra-path')
        before = [CameraDevice(0, 'USB Camera', 'astra-path')]
        after = [CameraDevice(0, 'Integrated Camera', 'laptop-path'),
                 CameraDevice(1, 'USB Camera', 'astra-path')]
        cap = Mock()
        with patch('core.camera_identity.enumerate_cameras', side_effect=[before, after]):
            with self.assertRaisesRegex(RuntimeError, 'order changed'):
                identity.open(Mock(return_value=cap), 640, 480)
        cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
