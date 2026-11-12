"""
scripts/check_routes.py
-----------------------
Manual HTTP smoke-tester for Clario routes.
Requires a live server running on port 5000: python app.py

Run from the project root:
    python scripts/check_routes.py
"""

import urllib.request
import urllib.error


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


opener = urllib.request.build_opener(NoRedirect)

tests = [
    ("GET /",          "http://127.0.0.1:5000/",          200),
    ("GET /register",  "http://127.0.0.1:5000/register",  200),
    ("GET /login",     "http://127.0.0.1:5000/login",     200),
    ("GET /dashboard", "http://127.0.0.1:5000/dashboard", 302),
    ("GET /upload",    "http://127.0.0.1:5000/upload",    302),
    ("GET /result/1",  "http://127.0.0.1:5000/result/1",  302),
    ("GET /notfound",  "http://127.0.0.1:5000/notfound",  404),
]

all_pass = True
for label, url, expected in tests:
    try:
        code = opener.open(url, timeout=4).getcode()
    except urllib.error.HTTPError as e:
        code = e.code
    ok = (code == expected)
    if not ok:
        all_pass = False
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  {label:<22}  HTTP {code}  (expected {expected})")

print()
print("Result:", "ALL PASS" if all_pass else "SOME FAILURES")
