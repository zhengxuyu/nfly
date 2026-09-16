"""Visualisation layer: watch any (policy, gym env) session in the browser.

    from nfly.viz import SessionConfig, serve
    server = serve(SessionConfig(suite="atari", game="pong", checkpoint="runs/ppo-atari-pong.pt"))
    print(server.url)
"""

from .anatomy import BrainAtlas, ActivityScale, build_atlas, calibrate_activity
from .server import VizServer, serve
from .session import Policy, RandomPolicy, Session, SessionConfig, action_names_of, build_session
from .streamer import Broadcast, EpisodeStreamer, StepEvent

__all__ = ["VizServer", "serve", "BrainAtlas", "ActivityScale", "build_atlas", "calibrate_activity", "Policy", "RandomPolicy", "Session", "SessionConfig", "build_session",
           "action_names_of", "Broadcast", "EpisodeStreamer", "StepEvent"]
