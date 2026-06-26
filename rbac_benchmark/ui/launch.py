"""
launch.py — Console entry point that boots the Streamlit dashboard.

Streamlit apps must be started via ``streamlit run <script>`` (the bare module can't
just be imported), so the ``rbac-dashboard`` entry point shells out to it, passing
through any extra CLI args (e.g. ``--server.port 8502``). Equivalent to:

    streamlit run rbac_benchmark/ui/app.py
"""
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parent / "app.py"


def main():
    # Imported lazily so importing this module doesn't require streamlit's CLI.
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(_APP), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
