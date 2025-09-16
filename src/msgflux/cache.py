from functools import wraps
from typing import Optional

from msgflux.data.dbs.base import BaseDB
from msgflux.message import Message
from msgflux.models.types import TextEmbedderModel


def response_cache(
    db: BaseDB,
    task_inputs: Optional[str] = None,
    response_mode: Optional[str] = None,
    model: Optional[TextEmbedderModel] = None,
):
    def decorator(func):
        @wraps(func)
        def wrapper(msg):
            if isinstance(msg, Message):
                if task_inputs is None:
                    raise ValueError(
                        "`task_inputs` is required when input is a Message instance"
                    )
                key = msg.get(task_inputs)
            else:
                key = msg

            if model:
                key = model(key)

            cached_response = db(key)

            if cached_response is not None:
                if isinstance(msg, Message):
                    if response_mode is None:
                        raise ValueError(
                            "`response_mode` is required when "
                            "input is a Message instance"
                        )
                    msg.set(response_mode, cached_response)
                    return msg
                return cached_response

            response = func(msg)

            if response is not None:
                db.add(key, response)

            return response

        return wrapper

    return decorator
