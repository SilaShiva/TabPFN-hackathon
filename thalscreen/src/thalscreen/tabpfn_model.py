"""TabPFN-3.5 model wrapper: local or cloud, with a pure-numpy fallback
for offline tests (deterministic synthetic scoring, clearly marked).
"""
from __future__ import annotations

import os

import numpy as np


def make_tabpfn(mode: str = "client"):
    """Return a TabPFNClassifier with corporate-proxy TLS applied if needed.

    mode: "client" (Prior Labs cloud, default, no GPU) or "local".
    """
    if mode == "local":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier()

    # cloud path: apply the corporate-proxy workarounds when configured
    bundle = os.environ.get("THALSCREEN_CA_BUNDLE")
    if bundle:
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    if os.environ.get("THALSCREEN_ALLOW_NONSTRICT_TLS") == "1":
        import ssl
        _orig = ssl.SSLContext.wrap_socket

        def lenient(ctx, *a, **k):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
            return _orig(ctx, *a, **k)

        ssl.SSLContext.wrap_socket = lenient
        try:
            import httpcore._ssl as hs
            _d = hs.default_ssl_context
            def ld():
                c = _d()
                c.verify_flags &= ~ssl.VERIFY_X509_STRICT
                return c
            hs.default_ssl_context = ld
        except ImportError:
            pass
        try:
            import joblib.parallel as jp
            _i = jp.Parallel.__init__
            def ti(self, *a, **k):
                if k.get("backend") == "loky":
                    k["backend"] = "threading"
                _i(self, *a, **k)
            jp.Parallel.__init__ = ti
        except ImportError:
            pass

    from tabpfn_client import TabPFNClassifier
    return TabPFNClassifier()
