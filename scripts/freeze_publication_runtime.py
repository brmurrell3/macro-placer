#!/usr/bin/env python3
"""Archive the current corrected runtime without relying on an uncommitted tree."""
from __future__ import annotations
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'writeup/data/publication'


def main():
    files = []
    for folder in ['thinkorplace', 'macro_place']:
        files.extend((ROOT / folder).rglob('*.py'))
    files.extend(ROOT / name for name in [
        'pyproject.toml', 'requirements.txt', 'LICENSE.md', 'README.md', 'placer.py',
        'thinkorplace/README.md', 'thinkorplace/MIGRATION.json',
        'submissions/thinkorplace/placer.py', 'submissions/thinkorplace-v2/placer.py',
        'submissions/thinkorplace-v2/lane_worker.py',
        'scripts/run_publication_study.py', 'scripts/publication_worker.py',
        'writeup/study/pilot.json', 'writeup/study/repair_validation.json', 'writeup/study/final_ng45_integration.json',
        'writeup/study/main_ibm17.json', 'writeup/study/README.md',
        'test/test_submission_safety.py', 'test/test_thinkorplace_package.py',
        'test/test_publication_runner.py', 'test/test_incremental_evaluator.py',
        'test/test_postpilot_repairs.py', 'test/test_sdf_topology.py'])
    payloads = {str(path.relative_to(ROOT)): path.read_bytes() for path in sorted(set(files))}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name,payload in payloads.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, payload in payloads.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))
    target = OUT / f'corrected-runtime-{identity[:12]}.tar.gz'
    data = buffer.getvalue()
    if target.exists() and target.read_bytes() != data:
        raise SystemExit('Refusing to replace a different immutable archive.')
    for name, payload in payloads.items():
        if (ROOT/name).read_bytes() != payload:
            raise SystemExit(f'Source changed while freezing: {name}')
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    manifest = {
        'classification': 'corrected_postcompetition_runtime',
        'base_git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'canonical_upstream_commit': '996193c83eaad151e5ae3fb166cedbd83372d060',
        'tilos_commit': subprocess.check_output(['git','-C','external/MacroPlacement','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_identity_sha256': identity,
        'archive': target.name, 'archive_sha256': hashlib.sha256(data).hexdigest(),
        'files': hashes,
        'scope': 'Solver package, canonical local infrastructure, compatibility entries, runner, focused tests and study configurations. Benchmark inputs and external TILOS dependency remain separate and are hashed in each run manifest.',
    }
    (OUT/'corrected_runtime_snapshot.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'{target.name}: {len(hashes)} files; {len(data)} bytes')


if __name__ == '__main__':
    main()
