"""Integrity-checked case archives and bounded durable release bundles."""

import hashlib
import json
from pathlib import Path
import zipfile


MAX_RELEASE_ASSET_BYTES = int(1.5 * 1024**3)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_case(case_directory, case_id, archive_path):
    """Create one raw case ZIP with safe, stable archive paths and CRC check."""
    case_directory, archive_path = Path(case_directory), Path(archive_path)
    if not case_directory.is_dir() or not case_id or "/" in case_id or "\\" in case_id:
        raise ValueError("invalid case archive source or identity")
    paths = sorted(path for path in case_directory.rglob("*") if path.is_file())
    if not paths:
        raise ValueError("cannot package an empty case")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    inventory = {}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            relative = path.relative_to(case_directory).as_posix()
            if ".." in Path(relative).parts or Path(relative).is_absolute():
                raise ValueError("case archive path escapes case root")
            name = f"cases/{case_id}/{relative}"
            payload = path.read_bytes()
            archive.writestr(name, payload)
            inventory[name] = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or {
            name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
        } != inventory:
            raise ValueError("packed case archive differs from source files")
    return {"path": archive_path, "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path), "members": inventory}


def _validate_case_records(case_archives):
    records = sorted(case_archives, key=lambda row: row["case_id"])
    ids = [row["case_id"] for row in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("cell evidence requires distinct case archives")
    for row in records:
        archive = Path(row["archive"])
        if not archive.is_file():
            raise ValueError("case archive is missing")
        actual = sha256_file(archive)
        if actual != row.get("sha256"):
            raise ValueError("case archive SHA256 differs from inventory")
    return records


def build_release_bundles(case_archives, output_dir, *, max_asset_bytes=MAX_RELEASE_ASSET_BYTES,
                          cell_name="tree-null-cell"):
    """Pack checked raw case ZIPs into stable assets below the release size cap."""
    if type(max_asset_bytes) is not int or max_asset_bytes < 256:
        raise ValueError("max asset size must be an integer of at least 256 bytes")
    records = _validate_case_records(case_archives)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    chunks, current, current_bytes = [], [], 22
    for row in records:
        archive = Path(row["archive"])
        size = archive.stat().st_size
        member_name = f"cases/{row['case_id']}.zip"
        cost = size + len(member_name.encode()) + 128
        if cost + 22 > max_asset_bytes:
            raise ValueError(f"single case archive exceeds release asset cap: {row['case_id']}")
        if current and current_bytes + cost > max_asset_bytes:
            chunks.append(current)
            current, current_bytes = [], 22
        current.append(row)
        current_bytes += cost
    if current:
        chunks.append(current)

    assets = []
    for part, chunk in enumerate(chunks, start=1):
        path = output_dir / f"{cell_name}-part{part:02d}.zip"
        expected = {}
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as bundle:
            for row in chunk:
                name = f"cases/{row['case_id']}.zip"
                payload = Path(row["archive"]).read_bytes()
                bundle.writestr(name, payload)
                expected[name] = {
                    "sha256": row["sha256"],
                    "bytes": len(payload),
                    "case_id": row["case_id"],
                }
        size = path.stat().st_size
        if size > max_asset_bytes:
            raise ValueError("release bundle exceeded configured asset cap")
        verify_release_bundle(path, expected)
        assets.append({
            "name": path.name,
            "path": path,
            "bytes": size,
            "sha256": sha256_file(path),
            "members": expected,
        })
    return assets


def verify_release_bundle(path, expected_members):
    """Verify ZIP CRC, exact member list and each nested raw archive hash."""
    path = Path(path)
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None or set(bundle.namelist()) != set(expected_members):
            raise ValueError("release bundle CRC or member list differs")
        for name, expected in expected_members.items():
            payload = bundle.read(name)
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError("release bundle member hash differs")
    return True


def verify_uploaded_asset(local_path, downloaded_path):
    """Confirm a downloaded GitHub Release asset exactly matches its local file."""
    if sha256_file(local_path) != sha256_file(downloaded_path):
        raise ValueError("uploaded release asset SHA256 differs")
    return True


def write_inventory(path, assets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = []
    for asset in assets:
        asset_path = Path(asset["path"])
        try:
            relative = asset_path.resolve().relative_to(path.parent.resolve()).as_posix()
        except ValueError:
            relative = asset_path.name
        document.append({key: value for key, value in asset.items() if key != "path"}
                        | {"path": relative})
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    return document
