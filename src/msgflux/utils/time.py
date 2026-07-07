from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def utc_now_isoformat() -> str:
    return utc_now().isoformat()


def utc_current_date() -> str:
    return utc_now().strftime("%A, %B %d, %Y")


def parse_utc_timestamp(value: str | None) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None
