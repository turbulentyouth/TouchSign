"""Read V4L2 capabilities without starting a video stream (Ubuntu/Linux ABI)."""
import errno
import os
import struct

# From linux/videodev2.h: _IOR('V', 0, v4l2_capability),
# _IOWR('V', 2, v4l2_fmtdesc). The structures contain only fixed-width fields.
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xC0405602
COLOR_FORMATS = {b'MJPG', b'JPEG', b'YUYV', b'UYVY', b'YVYU', b'NV12',
                 b'NV21', b'YU12', b'YV12', b'RGB3', b'BGR3', b'RGB4', b'BGR4'}


def is_color_capture(index):
    import fcntl
    fd = os.open(f'/dev/video{index}', os.O_RDONLY | os.O_NONBLOCK)
    try:
        capability = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, capability)
        capabilities, device_caps = struct.unpack_from('=II', capability, 84)
        caps = device_caps if capabilities & 0x80000000 else capabilities
        if not caps & 1:  # VIDEO_CAPTURE; excludes metadata-only nodes.
            return False
        for number in range(256):
            description = bytearray(64)
            struct.pack_into('=II', description, 0, number, 1)  # VIDEO_CAPTURE
            try:
                fcntl.ioctl(fd, VIDIOC_ENUM_FMT, description)
            except OSError as exc:
                if exc.errno == errno.EINVAL:  # End of enumeration.
                    return False
                raise
            if bytes(description[44:48]) in COLOR_FORMATS:
                return True
        return False
    finally:
        os.close(fd)
