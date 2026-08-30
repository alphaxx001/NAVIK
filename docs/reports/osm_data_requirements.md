# Map Matching Validation: OSM Data Requirements

## Geographic Coverage
The IDR-System validation map matcher requires an offline OpenStreetMap (OSM) extract encompassing all 12 held-out IO-VNBD test sessions. The precise bounding box (with a `0.05` degree geographic margin for routing context) is:

*   **Min Latitude (South):** `51.97152`
*   **Min Longitude (West):** `-2.25177`
*   **Max Latitude (North):** `53.27134`
*   **Max Longitude (East):** `-0.68391`
*   **Approximate Area:** 15,331.38 sq km (144.7 km x 106.0 km)

*(This region covers parts of the West and East Midlands in the UK, roughly spanning from Gloucester in the south to Nottingham/Sheffield in the north, and from the Welsh border in the west to Lincolnshire in the east).*

## Required Format
The data must be provided as a standard OSM extract.
*   **Preferred Format:** `.osm.pbf` (Protocolbuffer Binary Format - highly compressed and fast to parse)
*   **Acceptable Format:** `.osm` (XML format)

## Why the Data is Needed
The NAVIK pipeline requires strict GNSS-denied trajectory constraint via road map matching (Phase 7B). However, the execution environment blocks outbound HTTP/HTTPS traffic to the OSM Overpass API, meaning the map data cannot be dynamically downloaded during pipeline execution. Providing this file manually bridges the firewall gap.

## How it Will be Used
The OSM extract will be processed **offline** by a preprocessing script (`build_graph.py`). The script will:
1. Parse the local `.pbf` / `.osm` file.
2. Filter for drivable highways (discarding footways, bike paths, etc.).
3. Project the WGS84 (Lat/Lon) coordinates into the local ENU tangent plane corresponding to each session's origin.
4. Extract nodes, edges, and connectivity data.
5. Save a lightweight JSON map cache per session in `map/runtime/`.

## GNSS-Free Runtime Confirmation
The map matching system strictly abides by the blackout requirements:
*   **Offline Preparation:** GNSS is used *only once* at $t=0$ to establish the local tangent plane origin (ENU) and to extract the correct sub-graph from the OSM file.
*   **Runtime:** During the simulated GNSS blackout, the HMM/Viterbi map matcher relies **only** on:
    *   Dead-Reckoned (DR) state
    *   SpeedNet velocities
    *   HeadingNet yaw rates
    *   MotionStateNet ZUPT triggers
    *   The offline local OSM graph

GNSS latitude, longitude, and CAN-bus velocity/heading are strictly disconnected and inaccessible to the map-matching logic during the blackout inference.
