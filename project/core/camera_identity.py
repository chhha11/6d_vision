"""Pin DirectShow RGB capture to one physical device across reopen cycles."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    device_path: str


def enumerate_cameras() -> list[CameraDevice]:
    # COM initialization belongs to the calling thread. Live camera opens and
    # recovery happen on worker threads as well as on the main thread.
    import comtypes
    from comtypes import GUID
    from comtypes.persist import IPropertyBag
    from pygrabber.dshow_graph import SystemDeviceEnum
    from pygrabber.dshow_ids import DeviceCategories

    enumerator = moniker = bag = system = None
    comtypes.CoInitialize()
    try:
        system = SystemDeviceEnum()
        enumerator = system.system_device_enum.CreateClassEnumerator(
            GUID(DeviceCategories.VideoInputDevice), dwFlags=0)
        devices = []
        if not enumerator:
            return devices
        while True:
            try:
                moniker, count = enumerator.Next(1)
            except ValueError:
                break
            if not count:
                break
            bag = moniker.BindToStorage(0, 0, IPropertyBag._iid_).QueryInterface(IPropertyBag)
            name = str(bag.Read('FriendlyName', pErrorLog=None))
            try:
                path = str(bag.Read('DevicePath', pErrorLog=None))
            except comtypes.COMError:
                path = ''
            devices.append(CameraDevice(len(devices), name, path))
        return devices
    except Exception as exc:
        raise RuntimeError(f'Cannot enumerate RGB camera identities: {exc}') from exc
    finally:
        # Release COM interfaces while the apartment is still initialized.
        enumerator = moniker = bag = system = None
        comtypes.CoUninitialize()


@dataclass(frozen=True)
class PinnedCamera:
    name: str
    device_path: str

    @classmethod
    def bind(cls, index: int, expected_name: str | None):
        devices = enumerate_cameras()
        device = next((device for device in devices if device.index == index), None)
        if device is None or (expected_name is not None and
                              device.name.casefold() != expected_name.casefold()):
            raise RuntimeError('Selected RGB camera changed before its identity was pinned')
        if not device.device_path:
            raise RuntimeError(f'Cannot verify physical device path for {device.name!r}')
        return cls(device.name, device.device_path)

    def resolve(self) -> int:
        matches = [device for device in enumerate_cameras()
                   if device.device_path.casefold() == self.device_path.casefold()]
        if len(matches) != 1:
            raise RuntimeError(f'Waiting for original RGB camera {self.name!r}; device unavailable')
        return matches[0].index

    def open(self, opener, width: int, height: int, *args):
        index = self.resolve()
        cap = opener(index, width, height, *args)
        try:
            # OpenCV's DirectShow API accepts a numeric index only. Recheck
            # after open so a re-enumeration race cannot hand us another camera.
            if self.resolve() != index:
                raise RuntimeError('RGB device order changed while opening; retry original camera')
            return cap
        except BaseException:
            cap.release()
            raise
