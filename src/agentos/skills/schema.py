"""JSON-Schema validation for skill parameters.

The skill parameters are a *contract* between the model and the robot.  We use
the ``jsonschema`` library when available and fall back to a small internal
validator (same subset) so the repo still runs in a bare environment.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import guard
    import jsonschema

    HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]
    HAS_JSONSCHEMA = False


def validate_params(schema: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty == valid)."""
    if not isinstance(params, dict):
        return [f"parameters must be an object, got {type(params).__name__}"]
    if HAS_JSONSCHEMA and schema:
        validator = jsonschema.Draft202012Validator(schema)
        errors = []
        for error in sorted(validator.iter_errors(params), key=lambda e: list(e.absolute_path)):
            location = ".".join(str(part) for part in error.absolute_path) or "(root)"
            errors.append(f"{location}: {error.message}")
        return errors
    return _validate_fallback(schema, params)


def _validate_fallback(schema: dict[str, Any], params: dict[str, Any], path: str = "") -> list[str]:
    """Minimal validator: type/required/enum/min/max/properties/additionalProperties."""
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type == "object" and not isinstance(params, dict):
        return [f"{path or '(root)'}: expected object"]
    required = schema.get("required", [])
    for key in required:
        if key not in params:
            errors.append(f"{path or '(root)'}: missing required property {key!r}")
    properties = schema.get("properties", {})
    for key, value in params.items():
        if key not in properties:
            if schema.get("additionalProperties") is False:
                errors.append(f"{path or '(root)'}: additional property {key!r} is not allowed")
            continue
        errors.extend(_validate_value(properties[key], value, f"{path}.{key}".lstrip(".")))
    return errors


def _validate_value(schema: dict[str, Any], value: Any, path: str) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    if expected and expected in type_map:
        # bool is a subclass of int; reject it for number/integer fields
        if expected in ("number", "integer") and isinstance(value, bool):
            return [f"{path}: expected {expected}, got boolean"]
        if not isinstance(value, type_map[expected]):
            return [f"{path}: expected {expected}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
    if expected == "object":
        errors.extend(_validate_fallback(schema, value, path))
    if expected == "array" and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(_validate_value(schema["items"], item, f"{path}[{index}]"))
    return errors
