"""Check every internal link and anchor in a rendered site directory.

Usage: python3 check_site_links.py <directory>
"""

from __future__ import annotations

import collections
import html
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ATTRIBUTE = re.compile(r'<(?:a|link|img|script|source|iframe)\b[^>]*?\b(?:href|src)="([^"]+)"')
IDENTIFIER = re.compile(r'\bid="([^"]+)"')


def main() -> int:
    """Report broken internal links and return a process status."""
    root = Path(sys.argv[1]).resolve()
    pages = sorted(root.rglob('*.html'))
    anchors = {page: set(IDENTIFIER.findall(page.read_text(encoding='utf-8', errors='replace'))) for page in pages}
    broken: collections.Counter[str] = collections.Counter()
    checked = 0
    for page in pages:
        for raw in ATTRIBUTE.findall(page.read_text(encoding='utf-8', errors='replace')):
            value = html.unescape(raw)
            parts = urlsplit(value)
            if parts.scheme or value.startswith('//') or not value or value.startswith(('javascript:', 'data:')):
                continue
            checked += 1
            if parts.path.startswith('/'):
                target = root / unquote(parts.path).lstrip('/')
            elif parts.path:
                target = (page.parent / unquote(parts.path)).resolve()
            else:
                target = page
            if target.is_dir():
                target = target / 'index.html'
            if not target.exists():
                broken[f'{page.relative_to(root)} -> {value}'] += 1
            elif parts.fragment and target.suffix == '.html' and target in anchors and unquote(parts.fragment) not in anchors[target]:
                broken[f'{page.relative_to(root)} -> {value} (no anchor)'] += 1
    print(f'pages {len(pages)}, internal links checked {checked}, broken {sum(broken.values())}')
    for item, count in broken.most_common(40):
        print(f'  {count} {item[:150]}')
    return 1 if broken else 0


if __name__ == '__main__':
    sys.exit(main())
