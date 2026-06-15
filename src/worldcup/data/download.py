"""Descarga de datos vía la API de Kaggle.

Credenciales: el cliente moderno de Kaggle usa un **token único**
(`KAGGLE_API_TOKEN`, generado en https://www.kaggle.com/settings/api). Este
módulo carga automáticamente un archivo `.env` en la raíz del proyecto y acepta
la variable bajo cualquiera de estos nombres:

    KAGGLE_API_TOKEN   (nombre oficial)
    KAGGLE_API         (alias usado en este proyecto)

También admite el esquema clásico usuario/clave (KAGGLE_USERNAME + KAGGLE_KEY)
o un ~/.kaggle/kaggle.json, si prefieres esa vía.

Nota: `kaggle` se importa de forma perezosa DENTRO de la función porque al
importarlo intenta autenticar de inmediato y fallaría si no hay credenciales.
"""
from __future__ import annotations

import os
from pathlib import Path

from worldcup.config import PROJECT_ROOT, load_config

# Alias aceptados → nombre de variable que entiende el cliente de Kaggle.
_TOKEN_ALIASES = ("KAGGLE_API_TOKEN", "KAGGLE_API")


class KaggleCredentialsError(RuntimeError):
    """No se encontraron credenciales de Kaggle."""


def load_credentials(env_path: str | Path | None = None) -> None:
    """Carga el .env y normaliza la credencial a las variables de entorno.

    Mapea cualquier alias de token (p. ej. KAGGLE_API) a KAGGLE_API_TOKEN, que
    es lo que consume el cliente moderno de Kaggle. Idempotente.
    """
    from dotenv import load_dotenv

    dotenv_file = Path(env_path) if env_path else PROJECT_ROOT / ".env"
    if dotenv_file.exists():
        load_dotenv(dotenv_file)

    # Normaliza el token: el primer alias presente gana.
    if not os.environ.get("KAGGLE_API_TOKEN"):
        for alias in _TOKEN_ALIASES:
            val = os.environ.get(alias)
            if val:
                os.environ["KAGGLE_API_TOKEN"] = val.strip()
                break


def _has_any_credential() -> bool:
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    has_user_key = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return has_token or has_user_key or has_json


def _make_api():
    """Crea y autentica el cliente de Kaggle (import perezoso)."""
    load_credentials()
    if not _has_any_credential():
        raise KaggleCredentialsError(
            "No se encontró ninguna credencial de Kaggle. Define KAGGLE_API "
            "(o KAGGLE_API_TOKEN) en el archivo .env de la raíz del proyecto, "
            "o coloca ~/.kaggle/kaggle.json. Genera tu token en "
            "https://www.kaggle.com/settings/api"
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Falta el paquete 'kaggle'. Instala con: pip install kaggle") from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise KaggleCredentialsError(
            "Las credenciales de Kaggle no fueron aceptadas. Verifica que el "
            "token en .env (KAGGLE_API) sea válido y esté vigente."
        ) from exc
    return api


def download_dataset(dataset: str | None = None, dest: str | Path | None = None) -> Path:
    """Descarga y descomprime el dataset de Kaggle en data/raw/.

    Args:
        dataset: slug 'usuario/dataset'. Por defecto, el de config.yaml.
        dest: carpeta destino. Por defecto, data/raw/.

    Returns:
        Ruta de la carpeta donde quedaron los CSV.
    """
    cfg = load_config()
    dataset = dataset or cfg["data"]["kaggle_dataset"]
    dest = Path(dest) if dest else Path(cfg["paths"]["data_raw"])
    dest.mkdir(parents=True, exist_ok=True)

    api = _make_api()
    api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=False)
    return dest
