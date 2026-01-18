from typing import Any, Dict, Optional, Union
from uuid import uuid4

from msgflux.dotdict import dotdict


class Message(dotdict):
    def __init__(
        self,
        *,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        texts: Optional[Dict[str, Any]] = None,
        audios: Optional[Dict[str, Any]] = None,
        images: Optional[Dict[str, Any]] = None,
        videos: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        if extra is None:
            extra = {}
        if videos is None:
            videos = {}
        if images is None:
            images = {}
        if audios is None:
            audios = {}
        if texts is None:
            texts = {}
        if context is None:
            context = {}
        super().__init__()
        self.metadata = {
            "execution_id": str(uuid4()),
            "user_id": user_id,
            "user_name": user_name,
            "chat_id": chat_id,
        }
        self.content = content
        self.texts = texts
        self.context = context
        self.audios = audios
        self.images = images
        self.videos = videos
        self.extra = extra
        self.outputs = {}
        self.response = {}

    def get_response(self):
        if self.get("response"):
            return next(iter(self.get("response").values()))
        else:
            return self.get("response")
