from pathlib import Path
import re

__all__ = ["get_root_dir", "compress_file_from_txt"]


def get_root_dir():
    """Return the root directory of the project."""
    return str(Path(__file__).resolve().parents[1])


def _compress_code(code: str) -> str:
    lines = code.split('\n')
    compressed_lines = []

    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        line = re.sub(r'\s*([=+\-*/(,)])\s*', r'\1', line)  # Remove spaces around operators and brackets
        compressed_lines.append(line)
    compressed_code = ' '.join(compressed_lines)
    return compressed_code


def compress_file_from_txt(input_filename: str, output_filename: str):
    with open(input_filename, 'r') as file:
        code = file.read()
    compressed_code = _compress_code(code)

    with open(output_filename, 'w') as file:
        file.write(compressed_code)


