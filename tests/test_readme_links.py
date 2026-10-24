"""README.md: every relative link and image points to an existing file; no local absolute paths.

Also runs as a script and prints a report:  python tests/test_readme_links.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
ABS_PATH = re.compile(r"(?<![\w/])[A-Za-z]:[\\/]|/Users/|/home/|\\Users\\")


def relative_links(text):
    for target in LINK.findall(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
            continue                      # http(s), mailto, in-page anchors
        yield target.split("#", 1)[0]


def broken_links():
    text = README.read_text(encoding="utf-8")
    links = sorted(set(relative_links(text)))
    return links, [t for t in links if not (ROOT / t).exists()]


def absolute_paths():
    hits = []
    files = [README] + [p for p in (ROOT / "docs").rglob("*") if p.suffix in (".md", ".txt", ".svg", ".html")]
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if ABS_PATH.search(line):
                hits.append(f"{f.relative_to(ROOT).as_posix()}:{n}: {line.strip()[:80]}")
    return hits


def test_relative_links_exist():
    links, broken = broken_links()
    assert links, "no relative links found - the regex is broken"
    assert broken == [], broken


def test_no_absolute_local_paths():
    assert absolute_paths() == []


if __name__ == "__main__":
    links, broken = broken_links()
    for t in links:
        print(("MISSING " if t in broken else "ok      ") + t)
    hits = absolute_paths()
    for h in hits:
        print("ABSOLUTE PATH", h)
    print(f"\n{len(links)} relative links checked, {len(broken)} missing, {len(hits)} absolute paths")
    sys.exit(1 if broken or hits else 0)
