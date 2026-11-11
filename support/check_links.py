#!/usr/bin/env python3
"""
Check for broken internal markdown links.
Scans .md files for links like [text](Something.md) and reports
any that point to files that don't exist. Links with a URL scheme
or host (http, https, mailto, ...) and bare anchors are skipped,
as are targets that resolve outside the repository (they cannot
be validated from a checkout).

Paths listed in .check-links-ignore at the repository root (one
fnmatch glob per line, relative to the root) are also skipped:
markdown files matching a pattern are not scanned (e.g. vendored
third-party docs), and link targets matching a pattern are not
flagged as broken (e.g. targets that moved to another repository).

Usage: python3 check_links.py <directory>
       python3 check_links.py <file.md> [<file.md> ...]

The file form is used by pre-commit, which passes the tracked
markdown files to check.

Kept in sync between ml-explore/mlx-swift-examples (support/)
and ml-explore/mlx-swift (tools/).
"""

import fnmatch
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


# Directories to skip when scanning a directory
SKIP_DIRS = {'.git', '.build', '.swiftpm', 'vendor', 'node_modules'}

IGNORE_FILE = '.check-links-ignore'

LINK_PATTERN = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')


def load_ignore_patterns(base):
    """Load path patterns to skip from .check-links-ignore."""
    ignore_file = base / IGNORE_FILE
    if not ignore_file.is_file():
        return []
    patterns = []
    for line in ignore_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            patterns.append(line)
    return patterns


def is_ignored(path, base, ignore_patterns):
    """True if path (relative to base) matches an ignore pattern."""
    if not path.is_relative_to(base):
        return False
    rel = str(path.relative_to(base))
    return any(fnmatch.fnmatch(rel, p) for p in ignore_patterns)


def find_all_md_files(directory):
    """Build a set of all .md file paths under the directory."""
    md_files = set()
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        md_files.update(
            Path(dirpath) / name for name in filenames if name.endswith('.md'))
    return md_files


def check_file(filepath, directory, ignore_patterns):
    """Check all internal links in a file. Returns a list of
    (line_num, link_text, target) tuples for broken links."""
    broken = []
    in_code_block = False
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip fenced code blocks -- their contents are not links
            if line.lstrip().startswith(('```', '~~~')):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            for match in LINK_PATTERN.finditer(line):
                display = match.group(1)
                target = urlsplit(match.group(2))

                # Skip external links and bare anchors
                if target.scheme or target.netloc or not target.path:
                    continue

                # URL-decode (e.g. %20 -> space) and resolve relative
                # to the file's directory
                target_path = (filepath.parent / unquote(target.path)).resolve()

                # Targets outside the repository cannot be validated here
                if not target_path.is_relative_to(directory):
                    continue

                if is_ignored(target_path, directory, ignore_patterns):
                    continue

                if not target_path.exists():
                    broken.append((line_num, display, target.path))

    return broken


def main():
    args = sys.argv[1:]
    if len(args) == 1 and Path(args[0]).is_dir():
        directory = Path(args[0]).resolve()
        md_files = find_all_md_files(directory)
    elif args:
        directory = Path.cwd()
        md_files = {Path(arg).resolve() for arg in args}
    else:
        print("Usage: python3 check_links.py <directory or markdown files>")
        sys.exit(1)

    ignore_patterns = load_ignore_patterns(directory)

    by_file = {}
    for filepath in sorted(md_files):
        if is_ignored(filepath, directory, ignore_patterns):
            continue
        broken = check_file(filepath, directory, ignore_patterns)
        if broken:
            rel = (filepath.relative_to(directory)
                   if filepath.is_relative_to(directory) else filepath)
            by_file[rel] = broken

    if not by_file:
        print("No broken links found.")
        sys.exit(0)

    for source, broken in by_file.items():
        print(f"\n{source}:")
        for line_num, display, target in broken:
            print(f"  line {line_num}: [{display}]({target}) -> NOT FOUND")

    total = sum(len(broken) for broken in by_file.values())
    print(f"\nTotal: {total} broken link(s) in {len(by_file)} file(s)")
    sys.exit(1)


if __name__ == '__main__':
    main()
