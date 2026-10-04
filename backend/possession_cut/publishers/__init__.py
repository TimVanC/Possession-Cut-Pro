"""Publishing targets (TikTok, Reels, Shorts...). Interface only, no implementations.

Auto-posting is out of scope for this build. A scheduler integration can register a
``Publisher`` here later without touching the pipeline.
"""

from .base import Publisher, PublishRequest, PublishResult, get_publisher, register_publisher

__all__ = ["Publisher", "PublishRequest", "PublishResult", "get_publisher", "register_publisher"]
