"""Atomic publication and strict, offline verification of layer-consensus bundles."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Mapping

from open.graph import _freeze
from open.layer_artifacts import FILES, build_payloads, reconstruct_payloads, table_texts
from open.layer_types import LayeredOPENResult
from open.metadata import MetadataValidation
from open.io import _publish_no_replace


@dataclass(frozen=True)
class LayerBundle:
    metadata: Mapping[str, object]
    layers: tuple[Mapping[str, object], ...]
    assessments: tuple[Mapping[str, object], ...]

    def __post_init__(self):
        object.__setattr__(self,"metadata",_freeze(self.metadata))
        object.__setattr__(self,"layers",_freeze(self.layers))
        object.__setattr__(self,"assessments",_freeze(self.assessments))


def _json_write(path, value):
    with path.open("w",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        handle.write("\n")


def write_layer_outputs(result: LayeredOPENResult, output_dir: Path, *,
                         metadata: MetadataValidation) -> tuple[Path, ...]:
    if not isinstance(result,LayeredOPENResult) or not isinstance(metadata,MetadataValidation):
        raise TypeError("expected LayeredOPENResult and MetadataValidation")
    payloads = build_payloads(result,metadata)
    tables = table_texts(result,metadata)
    destination=Path(output_dir)
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.",dir=destination.parent) as temporary:
        stage=Path(temporary)
        for name,payload in payloads.items(): _json_write(stage/name,payload)
        for name,text in tables.items():
            with (stage/name).open("w",encoding="utf-8",newline="") as handle: handle.write(text)
        _publish_no_replace(stage,destination)
    return tuple(destination/name for name in FILES)


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError("duplicate JSON key")
        result[key]=value
    return result


def load_layer_bundle(output_dir: Path) -> LayerBundle:
    directory=Path(output_dir)
    try:
        payloads=[]
        for name in FILES[:3]:
            payloads.append(json.loads((directory/name).read_text(encoding="utf-8"),object_pairs_hook=_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value {value}"))))
        result,metadata=reconstruct_payloads(*payloads)
        for name,expected in table_texts(result,metadata).items():
            # Universal-newline reading tolerates a CSV copied by a Windows editor.
            if (directory/name).read_text(encoding="utf-8") != expected:
                raise ValueError(f"{name}: CSV disagrees with saved assessment evidence")
        return LayerBundle(payloads[0],tuple(payloads[1]["layers"]),tuple(payloads[2]["assessments"]))
    except (KeyError,TypeError,IndexError,AttributeError,OverflowError) as exc:
        raise ValueError(f"invalid layer-consensus bundle: {exc}") from exc
