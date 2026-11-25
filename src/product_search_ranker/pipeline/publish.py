"""`psr publish`: copy the model, its metrics and the demo index to S3 or a local folder."""

import json
import shutil
from pathlib import Path

from product_search_ranker import config
from product_search_ranker.evaluate.gate import ndcg_all

PUBLISHED_METRICS = "published_metrics.json"


def parse_target(target: str) -> tuple[str, str, str]:
    """-> (kind, bucket_or_dir, prefix). kind is "s3" or "local"."""
    if target.startswith("s3://"):
        bucket, _, prefix = target[5:].partition("/")
        if not bucket:
            raise SystemExit(f"bad target {target!r}: expected s3://bucket/prefix")
        return "s3", bucket, prefix.strip("/")
    # Relative paths would resolve against each Airflow task's temp cwd: always "first run".
    if target.startswith("local:") and Path(target[6:]).is_absolute():
        return "local", target[6:], ""
    raise SystemExit(f"bad target {target!r}: expected s3://bucket/prefix or local:<abs dir>")


def _key(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _s3_client(s3=None):
    if s3 is not None:
        return s3
    import boto3

    if boto3.Session().get_credentials() is None:
        raise SystemExit("no AWS credentials found; set them or use --dry-run")
    return boto3.client("s3")


def files_to_publish(fixtures: bool) -> dict[str, Path]:
    """Published name -> local file, metrics last. The demo index is included once P7 builds it."""
    models = config.models_dir(fixtures)
    # The gate compares the next candidate with this, so it must be the candidate's own run.
    metrics = config.candidate_dir(fixtures) / "metrics.json"
    files = {"ltr.txt": models / "ltr.txt", "ltr_meta.json": models / "ltr_meta.json"}
    missing = [str(p) for p in (*files.values(), metrics) if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)} (run train and eval --candidate)")
    if metrics.stat().st_mtime < files["ltr.txt"].stat().st_mtime:
        raise SystemExit(f"{metrics} is older than the model: rerun `psr eval --candidate`")
    if ndcg_all(json.loads(metrics.read_text())) is None:
        raise SystemExit(f"{metrics} has no ltr NDCG@10: run `psr eval --systems ltr --candidate`")
    demo = config.indexes_dir(fixtures) / "demo"
    if demo.is_dir():
        for p in sorted(demo.rglob("*")):
            if p.is_file():
                files[f"demo/{p.relative_to(demo).as_posix()}"] = p
    # Last, so a failed upload never leaves a new baseline next to a half-published model.
    files[PUBLISHED_METRICS] = metrics
    return files


def read_published(target: str, s3=None) -> dict | None:
    """The metrics of the last publish, or None on the first run."""
    kind, where, prefix = parse_target(target)
    if kind == "local":
        path = Path(where) / PUBLISHED_METRICS
        return json.loads(path.read_text()) if path.exists() else None
    client = _s3_client(s3)
    try:
        obj = client.get_object(Bucket=where, Key=_key(prefix, PUBLISHED_METRICS))
    except client.exceptions.NoSuchKey:
        return None
    return json.loads(obj["Body"].read())


def publish(target: str, fixtures: bool = False, dry_run: bool = False, s3=None) -> None:
    kind, where, prefix = parse_target(target)
    files = files_to_publish(fixtures)
    client = None if kind == "local" or dry_run else _s3_client(s3)
    for name, src in files.items():
        dest = f"s3://{where}/{_key(prefix, name)}" if kind == "s3" else str(Path(where) / name)
        if dry_run:
            print(f"would copy {src} -> {dest}")
        elif kind == "local":
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"wrote {dest}")
        else:
            client.upload_file(str(src), where, _key(prefix, name))
            print(f"wrote {dest}")
