import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETRAIN_LOG_PATH = PROJECT_ROOT / "models" / "retrain_log.json"


def run_command(command):
    print("\nRunning:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}"
        )

    return result


def write_retrain_log(status, message):
    RETRAIN_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    log = {
        "last_retrain_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "message": message,
    }

    RETRAIN_LOG_PATH.write_text(
        json.dumps(log, indent=2),
        encoding="utf-8"
    )


def main():
    print("\n" + "=" * 70)
    print("GRIDLOCK IQ 30-DAY RETRAIN JOB")
    print("=" * 70)

    try:
        run_command(
            [
                sys.executable,
                "train_all.py"
            ]
        )

        run_command(
            [
                sys.executable,
                "scripts/project_health_check.py"
            ]
        )

        write_retrain_log(
            status="success",
            message="30-day retrain completed successfully."
        )

        print("\nRetrain completed successfully.")

    except Exception as e:
        write_retrain_log(
            status="failed",
            message=str(e)
        )

        print("\nRetrain failed.")
        print(str(e))
        raise


if __name__ == "__main__":
    main()