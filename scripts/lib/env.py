"""Environment and API key management for last30days skill."""

import os
from pathlib import Path
from typing import Optional, Dict, Any

# 默认读取当前技能目录下的 .env
# 可通过 LAST30DAYS_CONFIG_DIR 覆盖（用于测试/特殊部署）
# - LAST30DAYS_CONFIG_DIR=""  => 禁用文件配置（只读系统环境变量）
# - LAST30DAYS_CONFIG_DIR="/path/to/dir" => 读取 /path/to/dir/.env
# - LAST30DAYS_CONFIG_DIR="/path/to/.env" => 直接读取该文件
SKILL_DIR = Path(__file__).resolve().parents[2]


def _resolve_config_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Resolve config directory/file based on override and defaults."""
    override = os.environ.get('LAST30DAYS_CONFIG_DIR')
    if override == "":
        return None, None

    if override:
        path = Path(override).expanduser()
        if path.name == ".env":
            return path.parent, path
        return path, path / ".env"

    return SKILL_DIR, SKILL_DIR / ".env"


CONFIG_DIR, CONFIG_FILE = _resolve_config_paths()


def _pick_value(file_env: Dict[str, str], *keys: str, default: Optional[str] = None) -> Optional[str]:
    """Pick the first non-empty value from env vars, then .env file."""
    for key in keys:
        value = os.environ.get(key)
        if value not in (None, ""):
            return value

    for key in keys:
        value = file_env.get(key)
        if value not in (None, ""):
            return value

    return default


def load_env_file(path: Path) -> Dict[str, str]:
    """Load environment variables from a file."""
    env = {}
    if not path.exists():
        return env

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                if key and value:
                    env[key] = value
    return env


def get_env_file_path() -> Optional[Path]:
    """Get the active .env file path, or None when file config is disabled."""
    return CONFIG_FILE


def get_env_file_display_path() -> str:
    """Get user-friendly config path for logs/prompts."""
    return str(CONFIG_FILE) if CONFIG_FILE else "(file config disabled)"


def get_config() -> Dict[str, Any]:
    """Load configuration from skill-local .env and process environment."""
    # Load from config file first (if configured)
    file_env = load_env_file(CONFIG_FILE) if CONFIG_FILE else {}

    # Environment variables override file
    config = {
        'OPENAI_API_KEY': _pick_value(file_env, 'OPENAI_API_KEY'),
        'XAI_API_KEY': _pick_value(file_env, 'XAI_API_KEY'),
        'OPENAI_MODEL_POLICY': _pick_value(file_env, 'OPENAI_MODEL_POLICY', default='auto'),
        'OPENAI_MODEL_PIN': _pick_value(file_env, 'OPENAI_MODEL_PIN'),
        'XAI_MODEL_POLICY': _pick_value(file_env, 'XAI_MODEL_POLICY', default='latest'),
        'XAI_MODEL_PIN': _pick_value(file_env, 'XAI_MODEL_PIN'),
        # 第三方中转/网关（OpenAI/xAI 兼容）
        'OPENAI_BASE_URL': _pick_value(
            file_env,
            'OPENAI_BASE_URL',
            'OPENAI_API_BASE',
            default='https://api.openai.com/v1',
        ),
        'XAI_BASE_URL': _pick_value(
            file_env,
            'XAI_BASE_URL',
            'XAI_API_BASE',
            default='https://api.x.ai/v1',
        ),
        # 模型名称映射：支持 JSON 或 key=value,key2=value2
        'OPENAI_MODEL_MAP': _pick_value(file_env, 'OPENAI_MODEL_MAP'),
        'XAI_MODEL_MAP': _pick_value(file_env, 'XAI_MODEL_MAP'),
        'OPENAI_FALLBACK_MODELS': _pick_value(file_env, 'OPENAI_FALLBACK_MODELS'),
    }

    return config


def config_exists() -> bool:
    """Check if configuration file exists."""
    return bool(CONFIG_FILE and CONFIG_FILE.exists())


def get_available_sources(config: Dict[str, Any]) -> str:
    """Determine which sources are available based on API keys.

    Returns: 'both', 'reddit', 'x', or 'web' (fallback when no keys)
    """
    has_openai = bool(config.get('OPENAI_API_KEY'))
    has_xai = bool(config.get('XAI_API_KEY'))

    if has_openai and has_xai:
        return 'both'
    elif has_openai:
        return 'reddit'
    elif has_xai:
        return 'x'
    else:
        return 'web'  # Fallback: WebSearch only (no API keys needed)


def get_missing_keys(config: Dict[str, Any]) -> str:
    """Determine which sources are missing (accounting for Bird).

    Returns: 'both', 'reddit', 'x', or 'none'
    """
    has_openai = bool(config.get('OPENAI_API_KEY'))
    has_xai = bool(config.get('XAI_API_KEY'))

    # Check if Bird provides X access (import here to avoid circular dependency)
    from . import bird_x
    has_bird = bird_x.is_bird_installed() and bird_x.is_bird_authenticated()

    has_x = has_xai or has_bird

    if has_openai and has_x:
        return 'none'
    elif has_openai:
        return 'x'  # Missing X source
    elif has_x:
        return 'reddit'  # Missing OpenAI key
    else:
        return 'both'  # Missing both


def validate_sources(requested: str, available: str, include_web: bool = False) -> tuple[str, Optional[str]]:
    """Validate requested sources against available keys.

    Args:
        requested: 'auto', 'reddit', 'x', 'both', or 'web'
        available: Result from get_available_sources()
        include_web: If True, add WebSearch to available sources

    Returns:
        Tuple of (effective_sources, error_message)
    """
    # WebSearch-only mode (no API keys)
    if available == 'web':
        if requested == 'auto':
            return 'web', None
        elif requested == 'web':
            return 'web', None
        else:
            config_path = get_env_file_display_path()
            return 'web', f"No API keys configured. Using WebSearch fallback. Add keys to {config_path} for Reddit/X."

    if requested == 'auto':
        # Add web to sources if include_web is set
        if include_web:
            if available == 'both':
                return 'all', None  # reddit + x + web
            elif available == 'reddit':
                return 'reddit-web', None
            elif available == 'x':
                return 'x-web', None
        return available, None

    if requested == 'web':
        return 'web', None

    if requested == 'both':
        if available not in ('both',):
            missing = 'xAI' if available == 'reddit' else 'OpenAI'
            return 'none', f"Requested both sources but {missing} key is missing. Use --sources=auto to use available keys."
        if include_web:
            return 'all', None
        return 'both', None

    if requested == 'reddit':
        if available == 'x':
            return 'none', "Requested Reddit but only xAI key is available."
        if include_web:
            return 'reddit-web', None
        return 'reddit', None

    if requested == 'x':
        if available == 'reddit':
            return 'none', "Requested X but only OpenAI key is available."
        if include_web:
            return 'x-web', None
        return 'x', None

    return requested, None


def get_x_source(config: Dict[str, Any]) -> Optional[str]:
    """Determine the best available X/Twitter source.

    Priority: Bird (free) → xAI (paid API)

    Args:
        config: Configuration dict from get_config()

    Returns:
        'bird' if Bird is installed and authenticated,
        'xai' if XAI_API_KEY is configured,
        None if no X source available.
    """
    # Import here to avoid circular dependency
    from . import bird_x

    # Check Bird first (free option)
    if bird_x.is_bird_installed():
        username = bird_x.is_bird_authenticated()
        if username:
            return 'bird'

    # Fall back to xAI if key exists
    if config.get('XAI_API_KEY'):
        return 'xai'

    return None


def get_x_source_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get detailed X source status for UI decisions.

    Returns:
        Dict with keys: source, bird_installed, bird_authenticated,
        bird_username, xai_available, can_install_bird
    """
    from . import bird_x

    bird_status = bird_x.get_bird_status()
    xai_available = bool(config.get('XAI_API_KEY'))

    # Determine active source
    if bird_status["authenticated"]:
        source = 'bird'
    elif xai_available:
        source = 'xai'
    else:
        source = None

    return {
        "source": source,
        "bird_installed": bird_status["installed"],
        "bird_authenticated": bird_status["authenticated"],
        "bird_username": bird_status["username"],
        "xai_available": xai_available,
        "can_install_bird": bird_status["can_install"],
    }
