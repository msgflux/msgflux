# dotdict

## ✦₊⁺ Overview

`dotdict` is a dictionary with **dot notation access** and **nested path support**. It extends Python's built-in `dict`, so anything that works with a regular dict works here too.

It is the foundation of [`Message`](nn/message.md) and can be used standalone wherever flexible, deeply-nested data structures are needed.

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"user": {"name": "Maria", "age": 30}})

    print(d.user.name)   # "Maria"
    print(d.user.age)    # 30
    ```

---

## 1. **Creating a dotdict**

### From a dict

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"key": "value", "nested": {"x": 1}})
    ```

### From keyword arguments

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict(name="Alice", score=100)
    ```

### Combined

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"role": "admin"}, name="Alice", score=100)
    ```

### Empty

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()
    d.name = "Bob"
    ```

---

## 2. **Reading Values**

### Dot access

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"user": {"name": "Clark"}})

    print(d.user.name)  # "Clark"
    ```

### Bracket access

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"user": {"name": "Clark"}})

    print(d["user"]["name"])  # "Clark"
    ```

### Nested path with `get()`

Use `get(path, default=None)` to traverse nested keys via a dot-separated string:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"user": {"profile": {"city": "Gotham"}}})

    print(d.get("user.profile.city"))        # "Gotham"
    print(d.get("user.profile.zip", "N/A"))  # "N/A" (key doesn't exist)
    ```

`get()` never raises — it returns `default` when any key in the path is missing.

---

## 3. **Writing Values**

### Dot assignment

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()
    d.name = "Diana"
    d.score = 99
    ```

### Bracket assignment

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()
    d["name"] = "Diana"
    ```

### Nested path with `set()`

Use `set(path, value)` to write deeply nested values. Intermediate keys are created automatically:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()

    d.set("user.profile.city", "Metropolis")
    d.set("user.profile.age", 28)

    print(d.user.profile.city)  # "Metropolis"
    print(d.user.profile.age)   # 28
    ```

---

## 4. **Nested Paths**

Both `get()` and `set()` accept dot-separated strings to traverse any depth:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()

    d.set("a.b.c.d", "deep value")
    print(d.get("a.b.c.d"))  # "deep value"
    ```

### List index access

Use integer segments in the path to index into lists:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()
    d.set("items", [{"name": "Alpha"}, {"name": "Beta"}])

    print(d.get("items.0.name"))  # "Alpha"
    print(d.get("items.1.name"))  # "Beta"
    ```

`set()` also supports writing into existing list positions:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()
    d.set("items", [{"name": "Alpha"}, {"name": "Beta"}])

    d.set("items.0.name", "Updated")
    print(d.get("items.0.name"))  # "Updated"
    ```

---

## 5. **Auto-wrapping**

Any `dict` assigned to a `dotdict` — whether at creation, via `set()`, or via attribute assignment — is automatically converted to a `dotdict`, so dot access always works:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()
    d.config = {"debug": True, "timeout": 30}

    print(d.config.debug)    # True
    print(d.config.timeout)  # 30
    ```

Lists of dicts are also wrapped recursively:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()
    d.users = [{"name": "A"}, {"name": "B"}]

    print(d.users[0].name)  # "A"
    ```

---

## 6. **`update()`**

`update()` extends `dict.update` with two extras:

**1. Dotted keys are written as nested paths:**

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()
    d.update({"user.name": "Bruce", "user.age": 35})

    print(d.user.name)  # "Bruce"
    ```

**2. Dict values are merged recursively when the key already holds a `dotdict`:**

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"config": {"debug": False, "timeout": 30}})
    d.update({"config": {"debug": True}})

    print(d.config.debug)    # True   (updated)
    print(d.config.timeout)  # 30     (preserved)
    ```

Standard positional argument and keyword argument forms are both supported:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict()
    d.update({"key": "value"})
    d.update(key="other")
    ```

---

## 7. **Serialization**

### `to_dict()`

Converts the `dotdict` (and all nested `dotdict` values) back to a plain Python `dict`:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"user": {"name": "Lois"}})

    plain = d.to_dict()
    print(type(plain))           # <class 'dict'>
    print(type(plain["user"]))   # <class 'dict'>
    ```

### `to_json()`

Returns a JSON-encoded `bytes` object, powered by `msgspec`:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"score": 42})

    print(d.to_json())  # b'{"score":42}'
    ```

---

## 8. **Immutability**

Pass `frozen=True` to create a read-only `dotdict`. Any attempt to write raises `AttributeError`:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"key": "value"}, frozen=True)

    d.key = "new"        # raises AttributeError: Cannot modify frozen dotdict
    d["key"] = "new"     # raises AttributeError
    d.set("key", "new")  # raises AttributeError
    del d.key            # raises AttributeError
    ```

Nested dicts inherit the `frozen` flag automatically:

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict({"user": {"name": "Bruce"}}, frozen=True)

    d.user.name = "Clark"  # raises AttributeError
    ```

---

## 9. **Hidden Keys**

`hidden_keys` marks keys as invisible to enumeration and discovery. They won't appear in iteration, serialization, or string representations — but can always be accessed directly if you know they exist.

???+ example

    ```python
    import msgflux as mf

    d = mf.dotdict(
        {"api_key": "sk-secret", "username": "john"},
        hidden_keys=["api_key"]
    )

    # Enumeration — api_key is invisible
    print("api_key" in d)      # False
    print(list(d.keys()))      # ["username"]
    print(list(d.values()))    # ["john"]
    print(list(d.items()))     # [("username", "john")]
    for k in d:
        print(k)               # "username"
    print(d.to_dict())         # {"username": "john"}
    print(d.to_json())         # b'{"username":"john"}'
    print(d)                   # {'username': 'john'}

    # Direct access — api_key is reachable
    print(d.api_key)              # "sk-secret"
    print(d["api_key"])           # "sk-secret"
    print(d.get("api_key"))       # "sk-secret"
    print(d.get("api_key", "x"))  # "sk-secret"
    ```

---

## 10. **Extending dotdict**

`dotdict` is designed to be subclassed. You can add default fields, metadata, or custom behavior by overriding `__init__`:

???+ example

    ```python
    import msgflux as mf


    class Config(mf.dotdict):
        def __init__(self, env="production", **kwargs):
            super().__init__(**kwargs)
            self.env = env
            self.debug = env != "production"


    cfg = Config(env="development", timeout=30)

    print(cfg.env)      # "development"
    print(cfg.debug)    # True
    print(cfg.timeout)  # 30
    ```

[`Message`](nn/message.md) follows this exact pattern — it extends `dotdict` with default AI workflow fields (`content`, `images`, `context`, etc.) and metadata (`user_id`, `chat_id`, `execution_id`).
