"""Resolve native provider audio policy without changing built-in or command defaults."""

from tools.transcription_chunking import upload_limit
from tools.transcription_common import CLOUD_STT_PROVIDERS


def provider_audio_policy(provider: str, model: str | None, plugin):
    """Return ((inclusive byte cap, duration cap), cloud preprocessing opt-in).

    Capability errors fail before preprocessing/upload. The caller binds the resolved
    instance for the entire recording, including all chunk uploads.
    """
    if plugin is None:
        return upload_limit(provider, model), provider in CLOUD_STT_PROVIDERS
    try:
        cap = plugin.max_upload_bytes
        trim = plugin.cloud_preprocessing
        if type(cap) is not int or cap <= 0:
            raise ValueError("max_upload_bytes must be a positive integer")
        if type(trim) is not bool:
            raise ValueError("cloud_preprocessing must be a boolean")
    except Exception as exc:
        raise ValueError(f"Invalid STT audio policy for '{provider}': {exc}") from exc
    return (cap, None), trim
