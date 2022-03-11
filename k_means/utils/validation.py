"""Module for json schema validation."""
import json
import k_means.resources.schemas as schemas
from k_means.resources.os_type import dir_slash
import jsonschema


def validate_json_file(filename):
    with open(filename) as file:
        try:
            return json.load(file)
        except json.decoder.JSONDecodeError as expt:
            print(f"Invalid JSON: {expt}")


def validate_message_vs_schema(message, filename) -> [str, bool]:
    schema = validate_json_file(schemas.__path__._path[0] +
                                dir_slash +
                                filename)
    try:
        jsonschema.validate(message, schema)
        return 'Schema Validation Success', True
    except jsonschema.exceptions.ValidationError as expt:
        return f'Schema Validation Failed: {expt}', False
