"""Hindsight's declared config surface — rendered by the generic desktop panel."""

from plugins.memory.config_schema import (
    KIND_BOOL, KIND_SECRET, KIND_SELECT, KIND_TEXT, ProviderConfigSchema, ProviderField, ProviderFieldOption,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="hindsight",
    label="Hindsight",
    fields=(
        ProviderField(
            key="mode", label="Mode", kind=KIND_SELECT, default="cloud",
            description="How Hermes connects to Hindsight.",
            options=(
                ProviderFieldOption("cloud", "Cloud", "Hindsight Cloud API (lightweight, just needs an API key)"),
                ProviderFieldOption("local_external", "Local External", "Connect to an existing Hindsight instance"),
            ),
            inline=True,
        ),
        ProviderField(
            key="api_key", label="API key", kind=KIND_SECRET, env_key="HINDSIGHT_API_KEY",
            description="Used to authenticate with the Hindsight API.",
            placeholder="Enter Hindsight API key", inline=True,
        ),
        ProviderField(
            key="api_url", label="API URL", kind=KIND_TEXT, default="https://api.hindsight.vectorize.io",
            aliases=("apiUrl",), env_fallbacks=("HINDSIGHT_API_URL",), inline=True,
        ),
        ProviderField(key="bank_id", label="Bank ID", kind=KIND_TEXT, default="hermes", aliases=("bankId",), inline=True),
        ProviderField(
            key="recall_budget", label="Recall budget", kind=KIND_SELECT, default="mid", aliases=("budget",),
            options=tuple(ProviderFieldOption(b, b) for b in ("low", "mid", "high")),
            inline=True,
        ),
        ProviderField(
            key="bank_id_template",
            label="Bank ID template",
            kind=KIND_TEXT,
            default="",
            description="Dynamic bank ID; supports {profile}, {workspace}, {platform}, {user}, and {session}.",
            group="Memory bank",
            inline=True,
        ),
        ProviderField(
            key="recall_tags",
            label="Recall tags",
            kind=KIND_TEXT,
            default="",
            description="Comma-separated tags used to filter recall.",
            env_fallbacks=("HINDSIGHT_RECALL_TAGS",),
            group="Recall",
            inline=True,
        ),
        ProviderField(
            key="observation_scopes",
            label="Observation scopes",
            kind=KIND_TEXT,
            default="",
            description="Explicit keyword or JSON scopes; when set, overrides derived scope filtering.",
            env_fallbacks=("HINDSIGHT_RETAIN_OBSERVATION_SCOPES",),
            group="Retain",
            inline=True,
        ),
        ProviderField(
            key="observation_scope_exclude_tag_prefixes",
            label="Excluded scope tag prefixes",
            kind=KIND_TEXT,
            default="",
            description="Comma-separated volatile prefixes excluded from derived observation scopes.",
            env_fallbacks=(
                "HINDSIGHT_RETAIN_OBSERVATION_SCOPE_EXCLUDE_TAG_PREFIXES",
            ),
            group="Retain",
            inline=True,
        ),
        ProviderField(
            key="expose_retain_tool",
            label="Expose retain tool",
            kind=KIND_BOOL,
            default=True,
            description="Expose hindsight_retain to the model; automatic retention remains enabled independently.",
            group="Tools",
            inline=True,
        ),
    ),
)
