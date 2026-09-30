"""PyInstaller runtime hook: make ``multiprocessing`` work in the frozen build.

Without this, a ``spawn``-based child in a frozen app re-executes the whole
``PULSE.exe`` bootloader and starts a second GUI instead of a worker, because the
frozen child has no ``__main__`` module for multiprocessing to re-import. The hook
runs before any user code, which is the only point where it can fix that up.

Guard on ``sys.frozen``: the hook is also loaded in development by some
workflows, and ``freeze_support()`` outside a frozen build would install a
no-op that shadows the real behaviour.
"""

import multiprocessing
import sys

if getattr(sys, "frozen", False):
    multiprocessing.freeze_support()
