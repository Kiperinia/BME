import importlib
import importlib.metadata
import platform
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def print_header(title: str) -> None:
	print(f"\n=== {title} ===")


def print_kv(key: str, value: object) -> None:
	print(f"{key}: {value}")


def get_distribution_version(name: str) -> str:
	try:
		return importlib.metadata.version(name)
	except importlib.metadata.PackageNotFoundError:
		return "not installed"
	except Exception as exc:
		return f"error: {exc}"


def try_import(module_name: str) -> bool:
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

	try_import("torch")
	try_import("triton")
	try_import("pandas")
	try_import("sam3")
	try_import("sam3.model.edt")
	try_import("MedicalSAM3.models.medsam3_wrapper")
	try_import("MedicalSAM3.models.medsam3_base")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
