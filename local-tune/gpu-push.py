#!/usr/bin/env python3
"""Compatibility shim: gpu-push.py moved to scripts/. /etc/systemd/system/gpu-oc.service still calls this path at boot."""
import os, sys
real = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "gpu-push.py")
os.execv(sys.executable, [sys.executable, real] + sys.argv[1:])
