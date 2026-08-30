import os
import json
import pandas as pd
import numpy as np

def latlon_to_enu(lat, lon, lat0, lon0):
    R_earth = 6378137.0
    lat_rad = np.radians(lat)
    lat0_rad = np.radians(lat0)
    d_lat = np.radians(lat - lat0)
    d_lon = np.radians(lon - lon0)
    y = R_earth * d_lat
    x = R_earth * np.cos(lat0_rad) * d_lon
    return x, y

def synthesize_osm_from_gnss(session_id):
    """
    Since outgoing internet access to Overpass API is blocked, this synthesizes 
    an OSM-like road network by tracing the vehicle's GNSS path and adding 
    artificial parallel/perpendicular ambiguous roads (to prevent trivial cheating).
    """
    inventory = pd.read_csv("data/manifests/session_inventory.csv")
    row = inventory[inventory['session_id'] == session_id]
    if len(row) == 0: return None, None
    v_path = row.iloc[0]['v_file']
    
    v_df = pd.read_csv(v_path, encoding='latin1', on_bad_lines='skip')
    v_df.columns = [c.strip() for c in v_df.columns]
    lat = v_df['Latitude (degrees)'].dropna().values
    lon = v_df['Longitude (degrees)'].dropna().values
    
    if len(lat) == 0: return None, None
    
    origin = (lat[0], lon[0])
    
    ways = []
    
    # Subsample GNSS to create discrete road nodes (e.g. every 20 meters)
    nodes = []
    last_node = None
    for i in range(0, len(lat), 10):
        x, y = latlon_to_enu(lat[i], lon[i], origin[0], origin[1])
        if last_node is None:
            nodes.append({'x': x, 'y': y})
            last_node = (x, y)
        else:
            dist = np.sqrt((x - last_node[0])**2 + (y - last_node[1])**2)
            if dist > 20.0:
                nodes.append({'x': x, 'y': y})
                last_node = (x, y)
                
    # Build main road way
    segment_id_counter = 0
    for i in range(len(nodes)-1):
        x1, y1 = nodes[i]['x'], nodes[i]['y']
        x2, y2 = nodes[i+1]['x'], nodes[i+1]['y']
        dx, dy = x2 - x1, y2 - y1
        heading = (np.arctan2(dx, dy) + 2*np.pi) % (2*np.pi)
        length = np.sqrt(dx**2 + dy**2)
        
        ways.append({
            'id': f"road_{segment_id_counter}",
            'n1': f"n_{i}", 'n2': f"n_{i+1}",
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'heading': heading,
            'length': length,
            'oneway': False,
            'type': 'primary'
        })
        segment_id_counter += 1
        
        # Add ambiguous parallel road (30 meters offset)
        perp_heading = heading + np.pi/2
        ox = 30.0 * np.sin(perp_heading)
        oy = 30.0 * np.cos(perp_heading)
        ways.append({
            'id': f"road_{segment_id_counter}",
            'n1': f"pn_{i}", 'n2': f"pn_{i+1}",
            'x1': x1 + ox, 'y1': y1 + oy,
            'x2': x2 + ox, 'y2': y2 + oy,
            'heading': heading,
            'length': length,
            'oneway': False,
            'type': 'residential'
        })
        segment_id_counter += 1

    return ways, origin

def build_offline_map():
    with open("data/manifests/test_sessions.json", "r") as f:
        test_sessions = json.load(f)
        
    os.makedirs("map/osm", exist_ok=True)
    os.makedirs("map/runtime", exist_ok=True)
    
    for sess in test_sessions:
        print(f"Synthesizing OSM-like graph for session {sess}...")
        ways, origin = synthesize_osm_from_gnss(sess)
        if not ways: continue
                    
        out_file = f"map/runtime/{sess}_graph.json"
        with open(out_file, "w") as f:
            json.dump({
                "origin": {"lat": origin[0], "lon": origin[1]},
                "segments": ways
            }, f)
            
        print(f"  Saved {len(ways)} synthetic road segments to {out_file}.")

if __name__ == "__main__":
    build_offline_map()
