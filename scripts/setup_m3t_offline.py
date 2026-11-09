#!/usr/bin/env python3
"""Build a pinned, project-local M3T region backend without sudo or pip changes."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parent.parent
COMMIT = 'f02106186a1e756450226b3b793e309506817c58'
PACKAGES = ('libeigen3-dev', 'libglew-dev', 'libglfw3', 'libglfw3-dev',
            'libgl-dev', 'libglx-dev', 'libopengl-dev', 'libglvnd-core-dev',
            'libegl-dev', 'libgles-dev', 'libglu1-mesa-dev')


def run(args, cwd=ROOT):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--offline', action='store_true', help='Use already downloaded source/packages')
    args = parser.parse_args()
    source = ROOT / 'third_party/3DObjectTracking'
    deps = ROOT / 'third_party/m3t_deps'
    packages = deps / 'packages'
    packages.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        if args.offline:
            parser.error('Upstream source is missing')
        run(['git', 'clone', '--filter=blob:none', '--no-checkout',
             'https://github.com/DLR-RM/3DObjectTracking.git', source])
        run(['git', 'sparse-checkout', 'init', '--cone'], source)
        run(['git', 'sparse-checkout', 'set', 'M3T/include', 'M3T/src', 'M3T/third_party'], source)
        run(['git', 'checkout', '--detach', COMMIT], source)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if revision != COMMIT:
        raise RuntimeError(f'Expected upstream {COMMIT}, found {revision}; source left intact')
    if not args.offline:
        run(['apt-get', 'download', *PACKAGES], packages)
    if not list(packages.glob('*.deb')):
        parser.error('Local dependency packages are missing')
    for package in sorted(packages.glob('*.deb')):
        run(['dpkg-deb', '-x', package, deps / 'root'])
    manifest = {'upstream': 'DLR-RM/3DObjectTracking', 'commit': COMMIT,
                'backend': 'M3T region only; upstream sources unmodified',
                'packages': {p.name: subprocess.check_output(
                    ['dpkg-deb', '-f', str(p), 'Version'], text=True).strip()
                    for p in sorted(packages.glob('*.deb'))}}
    (deps / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    run(['cmake', '-S', ROOT / 'native/m3t', '-B', ROOT / 'build/m3t', '-DCMAKE_BUILD_TYPE=Release'])
    run(['cmake', '--build', ROOT / 'build/m3t', '-j', max(1, args.jobs)])


if __name__ == '__main__':
    main()
