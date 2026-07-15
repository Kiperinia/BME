import importlib
import importlib.metadata
import platform
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def print_header(title: str) -> None:
	"""brief:
	    Print a section header for the import diagnostic report.

	parameter:
	    - title: Input value for title.

	retrival:
	    - Returns None; writes formatted text to stdout.
	"""
	print(f"\n=== {title} ===")


def print_kv(key: str, value: object) -> None:
	"""brief:
	    Print one key-value row for the import diagnostic report.

	parameter:
	    - key: Input value for key.
	    - value: Input value for value.

	retrival:
	    - Returns None; writes formatted text to stdout.
	"""
	print(f"{key}: {value}")


def get_distribution_version(name: str) -> str:
	"""brief:
	    Resolve an installed Python distribution version.

	parameter:
	    - name: Input value for name.

	retrival:
	    - Returns the installed version string or a diagnostic fallback string.
	"""
	try:
		return importlib.metadata.version(name)
	except importlib.metadata.PackageNotFoundError:
		return "not installed"
	except Exception as exc:
		return f"error: {exc}"


def try_import(module_name: str) -> bool:
	"""brief:
	    Import a module and print a structured success or failure report.

	parameter:
	    - module_name: Input value for module_name.

	retrival:
	    - Returns True when the module imports successfully, otherwise False.
	"""
	print_header(f"Import {module_name}")
	try:
		module = importlib.import_module(module_name)
		module_file = getattr(module, "__file__", "built-in")
		print_kv("status", "ok")
		print_kv("module", module_name)
		print_kv("file", module_file)
		return True
	except Exception as exc:
		print_kv("status", "failed")
		print_kv("module", module_name)
		print_kv("error_type", type(exc).__name__)
		print_kv("error", exc)
		print(traceback.format_exc().rstrip())
		return False


def main() -> int:
	"""brief:
	    Run the SAM3 and MedicalSAM3 import diagnostic check.

	parameter:
	    - None.

	retrival:
	    - Returns 0 when all required imports pass, otherwise 1.
	"""
	root_str = str(ROOT)
	if root_str not in sys.path:
		sys.path.insert(0, root_str)

	print_header("Environment")
	print_kv("python", sys.version.replace("\n", " "))
	print_kv("executable", sys.executable)
	print_kv("platform", platform.platform())
	print_kv("cwd", Path.cwd())
	print_kv("workspace", ROOT)

	print_header("Packages")
	print_kv("torch", get_distribution_version("torch"))
	print_kv("triton", get_distribution_version("triton"))
	print_kv("numpy", get_distribution_version("numpy"))
	print_kv("pillow", get_distribution_version("pillow"))
	print_kv("pandas", get_distribution_version("pandas"))
	print_kv("sam3", get_distribution_version("sam3"))
	print_kv("local triton wheel", (ROOT / "triton-3.0.0-cp312-cp312-win_amd64.whl").exists())

	checks = [
		"torch",
		"triton",
		"pandas",
		"sam3",
		"sam3.model.edt",
		"MedicalSAM3.sam3_official.build_model",
		"MedicalSAM3.sam3_official.tensor_forward",
		"MedicalSAM3.adapters.lora",
		"MedicalSAM3.scripts.common",
	]
	results = [try_import(module_name) for module_name in checks]
	return 0 if all(results) else 1


if __name__ == "__main__":
	raise SystemExit(main())
