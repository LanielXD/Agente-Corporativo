# ──────────────────────────────────────────────
# RUN — inicializador do Streamlit com path absoluto
# ──────────────────────────────────────────────

import subprocess
import sys
from pathlib import Path

VERSAO_MINIMA = (3, 10)
if sys.version_info < VERSAO_MINIMA:
    print(f"Erro: Python {'.'.join(str(v) for v in VERSAO_MINIMA)}+ requerido. Versão atual: {sys.version}")
    sys.exit(1)

DIRETORIO_PROJETO = Path(__file__).parent
resultado = subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(DIRETORIO_PROJETO / "app.py")],
    cwd=str(DIRETORIO_PROJETO),
)

if resultado.returncode != 0:
    print(f"Erro ao iniciar o Streamlit. Código de saída: {resultado.returncode}")
    sys.exit(resultado.returncode)