"""Run pipeline for a single group. Usage: python run_group.py <regime> <direction> <instrument> <label_file>"""
import sys, importlib, re, shutil, time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AF_PATH = "/Users/simon/Desktop/AlphaForge"
for p in (str(_PROJECT_ROOT), _AF_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.dev_pipeline import run_single_strategy_pipeline

regime, direction, instrument, label_file = sys.argv[1:5]
TIMEFRAMES = ["daily", "1h", "2h", "4h"]

# Switch labels
src = _PROJECT_ROOT / "data" / "regime_labels" / label_file
dst = _PROJECT_ROOT / "data" / "regime_labels" / f"{instrument}.yaml"
shutil.copy(src, dst)

results = []
for freq in TIMEFRAMES:
    strategy_dir = _PROJECT_ROOT / "strategies" / regime / direction / instrument / freq
    if not strategy_dir.exists():
        continue
    for py_file in sorted(strategy_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        v_match = re.search(r"v(\d+)", py_file.stem)
        if not v_match:
            continue
        v_num = v_match.group(0)
        mod_path = str(py_file.relative_to(_PROJECT_ROOT)).replace("/", ".").replace(".py", "")
        if mod_path in sys.modules:
            del sys.modules[mod_path]
        try:
            mod = importlib.import_module(mod_path)
            cls = None
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (isinstance(obj, type) and hasattr(obj, "_generate_signal")
                    and obj.__name__ not in ("QBaseStrategy", "TrendingStrategy", "MeanReversionStrategy")):
                    cls = obj
                    break
            if cls is None:
                continue
        except Exception as e:
            print(f"  [skip] {py_file.name}: {e}")
            continue

        t0 = time.time()
        try:
            result = run_single_strategy_pipeline(
                strategy_class=cls, symbol=instrument, direction=direction,
                regime=regime, horizon="medium", version=v_num, freq=freq,
                params_override={},
            )
            elapsed = time.time() - t0
            sr = result.get("validation", {}).get("oos_sharpe")
            fn = result.get("folder_name", v_num)
            st = result.get("status")
            results.append((freq, v_num, fn, sr, st, elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            results.append((freq, v_num, "ERROR", None, str(e)[:60], elapsed))

print(f"\n{'=' * 60}")
print(f"  {regime}/{direction}/{instrument} — {len(results)} strategies")
print(f"{'=' * 60}")
for freq, v, fn, sr, st, t in results:
    sr_str = f"sharpe={sr:.2f}" if sr is not None else "N/A"
    print(f"  {freq:>5} {v:>4}  {fn:<28}  {sr_str}  [{st}]  {t:.0f}s")
