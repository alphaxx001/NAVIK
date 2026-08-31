import os
import json
import pandas as pd
import numpy as np
import osmium

def latlon_to_enu(lat, lon, lat0, lon0):
    R_earth = 6378137.0
    lat_rad = np.radians(lat)
    lat0_rad = np.radians(lat0)
    d_lat = np.radians(lat - lat0)
    d_lon = np.radians(lon - lon0)
    y = R_earth * d_lat
    x = R_earth * np.cos(lat0_rad) * d_lon
    return x, y

def build_offline_map():
    pbf_file = "data/raw/OSM/uk_midlands.osm.pbf"
    if not os.path.exists(pbf_file):
        print(f"Error: OSM data not found at {pbf_file}")
        return
        
    inventory = pd.read_csv("data/manifests/session_inventory.csv")
    with open("data/manifests/test_sessions.json", "r") as f:
        test_sessions = json.load(f)
    with open("data/manifests/val_sessions.json", "r") as f:
        val_sessions = json.load(f)
    test_sessions = test_sessions + val_sessions
        
    # 1. Determine bounding boxes and origins for all sessions
    session_bboxes = {}
    session_origins = {}
    margin = 0.05
    for sess in test_sessions:
        row = inventory[inventory['session_id'] == sess]
        if len(row) == 0: continue
        v_path = row.iloc[0]['v_file']
        try:
            df = pd.read_csv(v_path, encoding='latin1', on_bad_lines='skip')
            df.columns = [c.strip() for c in df.columns]
            lat = df['Latitude (degrees)'].dropna().values
            lon = df['Longitude (degrees)'].dropna().values
            if len(lat) > 0:
                min_lat, max_lat = lat.min(), lat.max()
                min_lon, max_lon = lon.min(), lon.max()
                session_bboxes[sess] = (min_lat - margin, max_lat + margin, min_lon - margin, max_lon + margin)
                session_origins[sess] = (lat[0], lon[0])
        except Exception as e:
            pass

    # Excluded highway types
    excluded_highways = {'footway', 'pedestrian', 'steps', 'corridor', 'path', 'cycleway', 'track', 'proposed', 'construction', 'abandoned', 'platform', 'raceway'}

    class HighwayHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways_per_session = {sess: [] for sess in session_bboxes.keys()}
            
        def way(self, w):
            if 'highway' not in w.tags: return
            if w.tags['highway'] in excluded_highways: return
            try:
                matched_sessions = set()
                for n in w.nodes:
                    lat, lon = n.lat, n.lon
                    for sess, bbox in session_bboxes.items():
                        if sess not in matched_sessions and bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]:
                            matched_sessions.add(sess)
                
                if matched_sessions:
                    way_dict = {
                        'id': w.id,
                        'oneway': w.tags.get('oneway', 'no') == 'yes',
                        'type': w.tags['highway'],
                        'nodes': [(nd.ref, nd.lat, nd.lon) for nd in w.nodes]
                    }
                    for sess in matched_sessions:
                        self.ways_per_session[sess].append(way_dict)
            except osmium.InvalidLocationError:
                pass

    print("Parsing OSM PBF file... (This may take a minute for 300MB)")
    handler = HighwayHandler()
    handler.apply_file(pbf_file, locations=True)

    print("Building local ENU graphs...")
    os.makedirs("map/runtime", exist_ok=True)
    
    for sess in session_bboxes.keys():
        origin = session_origins[sess]
        ways = []
        
        for w in handler.ways_per_session[sess]:
            way_nodes = w['nodes']
            if len(way_nodes) < 2: continue
            
            for i in range(len(way_nodes)-1):
                n1_ref, n1_lat, n1_lon = way_nodes[i]
                n2_ref, n2_lat, n2_lon = way_nodes[i+1]
                
                x1, y1 = latlon_to_enu(n1_lat, n1_lon, origin[0], origin[1])
                x2, y2 = latlon_to_enu(n2_lat, n2_lon, origin[0], origin[1])
                
                dx = x2 - x1
                dy = y2 - y1
                heading = (np.arctan2(dx, dy) + 2*np.pi) % (2*np.pi)
                length = np.sqrt(dx**2 + dy**2)
                
                ways.append({
                    'id': f"{w['id']}_{i}",
                    'way_id': w['id'],
                    'n1': n1_ref,
                    'n2': n2_ref,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'heading': heading,
                    'length': length,
                    'oneway': w['oneway'],
                    'type': w['type']
                })
                
        out_file = f"map/runtime/{sess}_graph.json"
        with open(out_file, "w") as f:
            json.dump({
                "origin": {"lat": origin[0], "lon": origin[1]},
                "segments": ways
            }, f)
        print(f"  Saved {len(ways)} real OSM segments to {out_file}.")

if __name__ == "__main__":
    build_offline_map()
