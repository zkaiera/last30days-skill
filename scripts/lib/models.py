"""Model auto-selection for last30days skill."""

import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple

from . import cache, http

# OpenAI API
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_FALLBACK_MODELS = ["gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o"]

# xAI API - Agent Tools API requires grok-4 family
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
XAI_ALIASES = {
    "latest": "grok-4-1-fast",  # Required for x_search tool
    "stable": "grok-4-1-fast",
}


def parse_version(model_id: str) -> Optional[Tuple[int, ...]]:
    """Parse semantic version from model ID.

    Examples:
        gpt-5 -> (5,)
        gpt-5.2 -> (5, 2)
        gpt-5.2.1 -> (5, 2, 1)
    """
    match = re.search(r'(\d+(?:\.\d+)*)', model_id)
    if match:
        return tuple(int(x) for x in match.group(1).split('.'))
    return None


def parse_model_map(model_map_str: Optional[str]) -> Dict[str, str]:
    """Parse model mapping from JSON or key=value list.

    Supported formats:
    - JSON object: {"gpt-5.2":"provider/model-a","gpt-4.1":"provider/model-b"}
    - KV list: gpt-5.2=provider/model-a,gpt-4.1=provider/model-b
    - KV list with semicolon: gpt-5.2=provider/model-a;gpt-4.1=provider/model-b
    """
    if not model_map_str:
        return {}

    raw = model_map_str.strip()
    if not raw:
        return {}

    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        result = {}
        for key, value in data.items():
            if key is None or value is None:
                continue
            key_str = str(key).strip()
            value_str = str(value).strip()
            if key_str and value_str:
                result[key_str] = value_str
        return result

    result = {}
    parts = [segment.strip() for segment in re.split(r"[,;]", raw) if segment.strip()]
    for part in parts:
        if "=" in part:
            left, right = part.split("=", 1)
        elif ":" in part:
            left, right = part.split(":", 1)
        else:
            continue

        left = left.strip()
        right = right.strip()
        if left and right:
            result[left] = right

    return result


def get_openai_fallback_chain(config: Dict) -> List[str]:
    """Get OpenAI fallback chain after applying model mapping.

    Supports optional OPENAI_FALLBACK_MODELS override:
    - JSON array: ["gpt-4.1", "gpt-4o"]
    - CSV: gpt-4.1,gpt-4o
    """
    raw = config.get("OPENAI_FALLBACK_MODELS")
    fallback = OPENAI_FALLBACK_MODELS

    if raw:
        parsed: List[str] = []
        text = str(raw).strip()
        if text.startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = []
            if isinstance(data, list):
                parsed = [str(item).strip() for item in data if str(item).strip()]
        else:
            parsed = [part.strip() for part in text.split(",") if part.strip()]

        if parsed:
            fallback = parsed

    model_map = parse_model_map(config.get("OPENAI_MODEL_MAP"))
    return [apply_model_mapping(model_id, model_map) for model_id in fallback]


def apply_model_mapping(model_id: str, model_map: Optional[Dict[str, str]]) -> str:
    """Map canonical model ID to provider-specific model ID."""
    if not model_map:
        return model_id
    return model_map.get(model_id, model_id)


def _models_url(base_url: str) -> str:
    """Build models endpoint URL from base URL."""
    return f"{base_url.rstrip('/')}/models"


def _cache_key(
    provider: str,
    base_url: str,
    policy: str,
    pin: Optional[str],
    model_map: Optional[Dict[str, str]],
) -> str:
    """Build stable cache key including provider routing context."""
    payload = {
        "provider": provider,
        "base_url": base_url.rstrip('/'),
        "policy": policy,
        "pin": pin or "",
        "map": model_map or {},
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"model:{provider}:{digest}"


def is_mainline_openai_model(model_id: str) -> bool:
    """Check if model is a mainline GPT model (not mini/nano/chat/codex/pro)."""
    model_lower = model_id.lower()

    # Must be gpt-4o, gpt-4.1+, or gpt-5+ series (mainline, not mini/nano/etc)
    if not re.match(r'^gpt-(?:4o|4\.1|5)(\.\d+)*$', model_lower):
        return False

    # Exclude variants
    excludes = ['mini', 'nano', 'chat', 'codex', 'pro', 'preview', 'turbo']
    for exc in excludes:
        if exc in model_lower:
            return False

    return True


def select_openai_model(
    api_key: str,
    policy: str = "auto",
    pin: Optional[str] = None,
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    model_map: Optional[Dict[str, str]] = None,
    mock_models: Optional[List[Dict]] = None,
) -> str:
    """Select the best OpenAI model based on policy.

    Args:
        api_key: OpenAI API key
        policy: 'auto' or 'pinned'
        pin: Model to use if policy is 'pinned'
        mock_models: Mock model list for testing

    Returns:
        Selected model ID
    """
    model_map = model_map or {}

    if policy == "pinned" and pin:
        return apply_model_mapping(pin, model_map)

    # Check cache first
    cache_key = _cache_key("openai", base_url, policy, pin, model_map)
    cached = cache.get_cached_model(cache_key)
    if cached:
        return cached

    # Fetch model list
    if mock_models is not None:
        models = mock_models
    else:
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = http.get(_models_url(base_url), headers=headers)
            models = response.get("data", [])
        except http.HTTPError:
            # Fall back to known models
            selected = apply_model_mapping(OPENAI_FALLBACK_MODELS[0], model_map)
            cache.set_cached_model(cache_key, selected)
            return selected

    # Filter to mainline models
    candidates = [m for m in models if is_mainline_openai_model(m.get("id", ""))]

    if not candidates:
        # No gpt-5 models found, use fallback
        selected = apply_model_mapping(OPENAI_FALLBACK_MODELS[0], model_map)
        cache.set_cached_model(cache_key, selected)
        return selected

    # Sort by version (descending), then by created timestamp
    def sort_key(m):
        version = parse_version(m.get("id", "")) or (0,)
        created = m.get("created", 0)
        return (version, created)

    candidates.sort(key=sort_key, reverse=True)
    canonical = candidates[0]["id"]
    selected = apply_model_mapping(canonical, model_map)

    # Cache the selection
    cache.set_cached_model(cache_key, selected)

    return selected


def select_xai_model(
    api_key: str,
    policy: str = "latest",
    pin: Optional[str] = None,
    base_url: str = DEFAULT_XAI_BASE_URL,
    model_map: Optional[Dict[str, str]] = None,
    mock_models: Optional[List[Dict]] = None,
) -> str:
    """Select the best xAI model based on policy.

    Args:
        api_key: xAI API key
        policy: 'latest', 'stable', or 'pinned'
        pin: Model to use if policy is 'pinned'
        mock_models: Mock model list for testing

    Returns:
        Selected model ID
    """
    del api_key, mock_models
    model_map = model_map or {}

    if policy == "pinned" and pin:
        return apply_model_mapping(pin, model_map)

    # Use alias system
    if policy in XAI_ALIASES:
        canonical = XAI_ALIASES[policy]
        alias = apply_model_mapping(canonical, model_map)

        # Check cache first
        cache_key = _cache_key("xai", base_url, policy, pin, model_map)
        cached = cache.get_cached_model(cache_key)
        if cached:
            return cached

        # Cache the alias
        cache.set_cached_model(cache_key, alias)
        return alias

    # Default to latest
    return apply_model_mapping(XAI_ALIASES["latest"], model_map)


def get_models(
    config: Dict,
    mock_openai_models: Optional[List[Dict]] = None,
    mock_xai_models: Optional[List[Dict]] = None,
) -> Dict[str, Optional[str]]:
    """Get selected models for both providers.

    Returns:
        Dict with 'openai' and 'xai' keys
    """
    result = {"openai": None, "xai": None}

    openai_model_map = parse_model_map(config.get("OPENAI_MODEL_MAP"))
    xai_model_map = parse_model_map(config.get("XAI_MODEL_MAP"))

    if config.get("OPENAI_API_KEY"):
        result["openai"] = select_openai_model(
            config["OPENAI_API_KEY"],
            config.get("OPENAI_MODEL_POLICY", "auto"),
            config.get("OPENAI_MODEL_PIN"),
            config.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            openai_model_map,
            mock_openai_models,
        )

    if config.get("XAI_API_KEY"):
        result["xai"] = select_xai_model(
            config["XAI_API_KEY"],
            config.get("XAI_MODEL_POLICY", "latest"),
            config.get("XAI_MODEL_PIN"),
            config.get("XAI_BASE_URL", DEFAULT_XAI_BASE_URL),
            xai_model_map,
            mock_xai_models,
        )

    return result
