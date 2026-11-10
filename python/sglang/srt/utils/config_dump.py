"""Credential sanitization for readbacks and dumps; live config/IPC stays raw."""

SECRET_FIELDS = frozenset(
    {"api_key", "admin_api_key", "disagg_decode_control_key", "ssl_keyfile_password"}
)


def redact_cli_secrets(argv, parser):
    """Scrub every credential option's value before retaining the launch argv."""
    result = []
    redact_next = False
    for argument in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        option = parser._parse_optional(argument)
        # CPython 3.14 returns a list of option tuples; older releases return
        # one tuple. Unknown options can have a None action in either shape.
        options = option if isinstance(option, list) else [option]
        if any(
            item is not None and item[0] is not None and item[0].dest in SECRET_FIELDS
            for item in options
        ):
            name, separator, _ = argument.partition("=")
            result.append(name + "=<redacted>" if separator else name)
            redact_next = not separator
        else:
            result.append(argument)
    return result


def redact_config_secrets(
    snapshot, *, secret_values=(), updates=(), redact_strings=True
):
    """Return a scrubbed copy, including credentials in command strings."""
    secrets = {value for value in secret_values if isinstance(value, str)}

    def collect(value):
        if isinstance(value, dict):
            for name, item in value.items():
                if name in SECRET_FIELDS and isinstance(item, str):
                    secrets.add(item)
                else:
                    collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(snapshot)
    collect(updates)
    secrets = sorted(
        (key for key in secrets if key and key != "<redacted>"), key=len, reverse=True
    )

    def clean(value, scrub_strings=redact_strings):
        if isinstance(value, dict):
            return {
                key: (
                    "<redacted>"
                    if key in SECRET_FIELDS and item
                    else clean(
                        item,
                        scrub_strings or key in ("launch_command", "_launch_command"),
                    )
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return type(value)(clean(item, scrub_strings) for item in value)
        if scrub_strings and isinstance(value, str):
            for key in secrets:
                value = value.replace(repr(key), "'<redacted>'").replace(
                    key, "<redacted>"
                )
        return value

    return clean(snapshot)


def redact_config_for_dump(snapshot, server_args, updates):
    return redact_config_secrets(
        snapshot,
        secret_values=[getattr(server_args, name, None) for name in SECRET_FIELDS],
        updates=updates,
    )
