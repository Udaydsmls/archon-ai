import os
import subprocess
import tempfile


class CodeExecutorTool:
    """Runs Python code in an isolated subprocess and returns its output."""

    name = "execute_python"
    description = (
        "Execute Python code and return stdout/stderr. "
        "Use for calculations, data analysis, and generating structured outputs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    }

    _TIMEOUT_SECONDS = 10
    _MAX_OUTPUT_CHARS = 3000

    def run(self, code: str) -> str:
        """Write code to a temp file, execute it, and return truncated output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=self._TIMEOUT_SECONDS,
            )
            output = result.stdout or result.stderr
            return output[: self._MAX_OUTPUT_CHARS] if output else "No output."
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {self._TIMEOUT_SECONDS}s."
        finally:
            os.unlink(tmp_path)

    def to_anthropic_schema(self) -> dict:
        """Return the Anthropic-compatible tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
