from dataclasses import dataclass
from typing import Optional

@dataclass
class NavigationOutput:
    timestamp: float
    mode: str
    latitude: float
    longitude: float
    enu_position: tuple
    speed_mps: float
    heading: float
    gnss_available: bool
    stationary_probability: float
    map_confidence: float
    map_candidate_count: int
    fallback_reason: Optional[str]
    authoritative_dr_position: tuple
    map_context_position: Optional[tuple]
