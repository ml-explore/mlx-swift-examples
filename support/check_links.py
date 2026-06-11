#!/usr/bin/env python3
"""
Check for broken internal markdown links.
Scans .md files for links like [text](Something.md) and reports
any that point to files that don't exist. External links (http,
https, mailto) and bare anchors are skipped, as are targets that
resolve outside the repository (they cannot be validated from a
checkout).

Targets listed in .check-links-ignore at the repository root (one
fnmatch glob per line, relative to the root) are also skipped.

Usage: python3 check_links.py <directory>
       python3 check_links.py <file.md> [<file.md> ...]

The file form is used by pre-commit, which passes the tracked
markdown files to check.
"""

import fnmatch
import re
import sys
from pathlib import Path
from urllib.parse import unquote


# Directories to skip when scanning a directory
SKIP_DIRS = {'.git', '.build', '.swiftpm', 'vendor', 'node_modules'}

IGNORE_FILE = '.check-links-ignore'


def load_ignore_patterns(base):
    """Load link-target patterns to skip from .check-links-ignore."""
    ignore_file = base / IGNORE_FILE
    if not ignore_file.is_file():
        return []
    patterns = []
    for line in ignore_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            patterns.append(line)
    return patterns


def find_all_md_files(directory):
    """Build a set of all .md file paths under the directory."""
    md_files = set()
    for filepath in directory.rglob('*.md'):
        if any(part in SKIP_DIRS for part in filepath.parts):
            continue
        md_files.add(filepath)
    return md_files


def check_file(filepath, directory, ignore_patterns):
    """Check all internal links in a file. Returns list of (source, line_num, link_text, target) tuples."""
    broken = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Match [text](target) but skip external URLs
    link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')

    for line_num, line in enumerate(lines, 1):
        for match in link_pattern.finditer(line):
            display = match.group(1)
            target = match.group(2)

            # Skip external links
            if target.startswith(('http://', 'https://', 'mailto:', '#')):
                continue

            # Strip anchor fragments
            target = target.split('#')[0]
            if not target:
                continue

            # URL-decode (e.g. %20 -> space)
            target = unquote(target)

            # Resolve relative to the file's directory
            target_path = (filepath.parent / target).resolve()

            # Targets outside the repository cannot be validated here
            try:
                rel_target = target_path.relative_to(directory)
            except ValueError:
                continue

            if any(fnmatch.fnmatch(str(rel_target), p) for p in ignore_patterns):
                continue

            if not target_path.exists():
                try:
                    rel_source = filepath.relative_to(directory)
                except ValueError:
                    rel_source = filepath
                broken.append((rel_source, line_num, display, target))

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

    all_broken = []
    for filepath in sorted(md_files):
        all_broken.extend(check_file(filepath, directory, ignore_patterns))

    if not all_broken:
        print("No broken links found.")
        sys.exit(0)

    # Group by source file
    by_file = {}
    for source, line_num, display, target in all_broken:
        by_file.setdefault(source, []).append((line_num, display, target))

    for source in sorted(by_file):
        print(f"\n{source}:")
        for line_num, display, target in by_file[source]:
            print(f"  line {line_num}: [{display}]({target}) -> NOT FOUND")

    print(f"\nTotal: {len(all_broken)} broken link(s) in {len(by_file)} file(s)")
    sys.exit(1)


if __name__ == '__main__':
    main()
