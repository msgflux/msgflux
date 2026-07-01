from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def utc_now_isoformat() -> str:
    return utc_now().isoformat()


def utc_current_date() -> str:
    return utc_now().strftime("%A, %B %d, %Y")
