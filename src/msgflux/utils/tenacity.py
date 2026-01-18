from tenacity import retry, stop_after_attempt, stop_after_delay

from msgflux.envs import envs

model_retry = retry(
    reraise=True,
    stop=(
        stop_after_delay(envs.model_stop_after_delay)
        | stop_after_attempt(envs.model_stop_after_attempt)
    ),
)

tool_retry = retry(
    reraise=True,
    stop=(
        stop_after_delay(envs.tool_stop_after_delay)
        | stop_after_attempt(envs.tool_stop_after_attempt)
    ),
)
