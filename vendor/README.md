# Robot SDK dependencies

These wheels are repackaged by TouchSign from unmodified upstream Python sources
and upstream prebuilt x64 DLLs. They are not official upstream wheel releases.
Source blobs and DLLs were checked against Git blob SHA-1 before packaging;
manifest.json records the resulting wheel SHA-256. Wheels include upstream
license files and Python source. wrapt uses its upstream pure-Python fallback.

| Package | Source |
| --- | --- |
| pyAgxArm | https://github.com/agilexrobotics/pyAgxArm/tree/d90e234ba8e408cfa964b38aa9120b3131fc9f85 |
| python-can-agx-cando | https://github.com/agilexrobotics/python-can-agx-cando/tree/fbd76e5c3e6b6d9b30eeb0610c5b595994d8ad94 |
| python-can 4.6.1 | https://github.com/hardbyte/python-can/tree/v4.6.1 |
| wrapt 1.17.3 | https://github.com/GrahamDumpleton/wrapt/tree/1.17.3 |

The x64 CAN plugin contains cando.dll and agx_receive.dll, ABI 2. Hardware drivers,
firmware compatibility, native drag compensation with the installed payload, and
physical motions have not been tested. No hardware is opened during build checks.

pyAgxArm and python-can: LGPL-3.0; CAN plugin: MIT; wrapt: BSD-2-Clause.
Keep these notices, included licenses, and the corresponding source available
when redistributing the application. The wheels can be opened as ZIP archives.
