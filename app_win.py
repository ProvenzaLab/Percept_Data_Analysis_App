"""Windows entry point.

On Windows we additionally tag the process with an explicit
``AppUserModelID`` so the taskbar groups windows under the correct app
identity. All actual UI lives in :mod:`app` (and ``src.ui``); this file
only sets the OS-level metadata and forwards to ``app.main()``.
"""

# Set the AppUserModelID before importing Qt so it takes effect on the
# first Windows shell call this process makes.
try:
    from ctypes import windll

    myappid = "Provenza_Labs.Percept_Data_Analysis App.v2.0"
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

import app

if __name__ == "__main__":
    app.main()
