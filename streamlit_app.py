import streamlit as st
import overpy
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point, LineString
import networkx as nx
import osmnx as ox
import random
import warnings
import gpxpy
import gpxpy.gpx
import pyproj
warnings.filterwarnings('ignore')
import os
import time
from collections import deque
from pathlib import Path

# Configure OSMnx for better Streamlit compatibility
ox.settings.use_cache = True
ox.settings.cache_folder = '/tmp/osmnx_cache'
ox.settings.log_console = False
ox.settings.timeout = 180
ox.settings.overpass_rate_limit = True
ox.settings.overpass_settings = '[out:json][timeout:180]'

@st.cache_data(ttl=3600, show_spinner=False)

def download_network_cached(city_name, center_point=None, radius_meters=5000, network_type='walk'):
    try:
        safe_name = city_name.replace(',', '').replace(' ', '_').lower()
        graph_path = Path("cached_graphs") / f"{safe_name}.graphml"
        if graph_path.exists():
            G = ox.load_graphml(graph_path)
            center_point = CACHED_CITIES[city_name]["center"]
        else:
            center_point = center_point or CACHED_CITIES[city_name]["center"]
            G = ox.graph_from_point(center_point, dist=radius_meters, network_type=network_type)
            G = ox.project_graph(G)
        
        if not nx.is_strongly_connected(G):
            largest_cc = max(nx.strongly_connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        return G, center_point
    except Exception as e:
        st.error(f"Error loading or generating network: {e}")
        return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def find_bars_with_overpy_cached(city_name, center_point, radius_meters=5000, include_restaurants=False):
    """
    Find bars and pubs using Overpy with caching for Streamlit.
    """
    st.info(f"🔍 Searching for bars in {city_name}...")
    
    lat, lon = center_point
    
    # Initialize the Overpass API
    api = overpy.Overpass()

    # Start with just bars and pubs
    amenity_query = '"amenity"~"bar|pub|biergarten|brewery|wine_bar"'
    
    try:
        # Construct Overpass query
        query = f"""
        [out:json][timeout:90];
        (
        node[{amenity_query}](around:{radius_meters},{lat},{lon});
        way[{amenity_query}](around:{radius_meters},{lat},{lon});
        relation[{amenity_query}](around:{radius_meters},{lat},{lon});
        );
        out center;
        """

        result = api.query(query)
        locations = []

        # From nodes
        for node in result.nodes:
            locations.append({
                "osmid": node.id,
                "name": node.tags.get("name", f"Unnamed Bar {len(locations)+1}"),
                "amenity": node.tags.get("amenity", "bar"),
                "lat": float(node.lat),
                "lon": float(node.lon)
            })

        # From ways and relations
        for obj in result.ways + result.relations:
            if hasattr(obj, 'center_lat') and hasattr(obj, 'center_lon'):
                lat = float(obj.center_lat)
                lon = float(obj.center_lon)
            elif hasattr(obj, 'nodes') and obj.nodes:
                lat = sum(float(n.lat) for n in obj.nodes) / len(obj.nodes)
                lon = sum(float(n.lon) for n in obj.nodes) / len(obj.nodes)
            else:
                continue

            locations.append({
                "osmid": obj.id,
                "name": obj.tags.get("name", f"Unnamed Bar {len(locations)+1}"),
                "amenity": obj.tags.get("amenity", "bar"),
                "lat": lat,
                "lon": lon
            })

        # If we don't have enough locations, try including restaurants with alcohol
        if len(locations) < 10 and include_restaurants:
            st.info("Adding restaurants with alcohol to increase venue options...")
            
            # Query for restaurants with alcohol service
            restaurant_query = f"""
            [out:json][timeout:90];
            (
            node["amenity"="restaurant"]["alcohol"="yes"](around:{radius_meters},{lat},{lon});
            way["amenity"="restaurant"]["alcohol"="yes"](around:{radius_meters},{lat},{lon});
            relation["amenity"="restaurant"]["alcohol"="yes"](around:{radius_meters},{lat},{lon});
            );
            out center;
            """
            
            try:
                rest_result = api.query(restaurant_query)
                
                for node in rest_result.nodes:
                    locations.append({
                        "osmid": node.id,
                        "name": node.tags.get("name", f"Restaurant {len(locations)+1}"),
                        "amenity": "restaurant_with_bar",
                        "lat": float(node.lat),
                        "lon": float(node.lon)
                    })
                    
                for obj in rest_result.ways + rest_result.relations:
                    if hasattr(obj, 'center_lat') and hasattr(obj, 'center_lon'):
                        lat = float(obj.center_lat)
                        lon = float(obj.center_lon)
                    elif hasattr(obj, 'nodes') and obj.nodes:
                        lat = sum(float(n.lat) for n in obj.nodes) / len(obj.nodes)
                        lon = sum(float(n.lon) for n in obj.nodes) / len(obj.nodes)
                    else:
                        continue
                    
                    locations.append({
                        "osmid": obj.id,
                        "name": obj.tags.get("name", f"Restaurant {len(locations)+1}"),
                        "amenity": "restaurant_with_bar",
                        "lat": lat,
                        "lon": lon
                    })
            except Exception as e:
                st.warning(f"Could not add restaurants: {e}")

        # Create DataFrame with the results
        df = pd.DataFrame(locations) if locations else pd.DataFrame(columns=["osmid", "name", "amenity", "lat", "lon"])
        
        if df.empty:
            st.warning("No bars found. Try a different city or increasing the search radius.")
            return None
            
        # Convert to GeoDataFrame
        bars_gdf = gpd.GeoDataFrame(
            df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326"
        )
        
        # Remove unwanted venues and reset index
        bars_gdf = bars_gdf[bars_gdf['amenity'] != 'public_bookcase']
        bars_gdf = bars_gdf.reset_index(drop=True)
        
        st.success(f"Found {len(bars_gdf)} venues in {city_name}.")
        return bars_gdf
        
    except Exception as e:
        st.error(f"Error finding bars: {e}")
        
        # Try with a larger radius as fallback
        if radius_meters < 10000:
            st.warning("Trying with larger search radius...")
            return find_bars_with_overpy_cached(city_name, center_point, int(radius_meters * 1.5), include_restaurants)
        else:
            return None

def create_bar_marathon_route(G, bars_gdf, bar_spacing=5.0, num_bars=9, start_bar_idx=None):
    """
    Create a bar marathon route based on the network and bar data.
    Uses a simplified approach for better Streamlit performance.
    """
    st.info("🛣️ Creating bar marathon route...")
    
    if bars_gdf is None or len(bars_gdf) == 0:
        st.error("No bars found. Please try a different city or increase search radius.")
        return None
    
    if G is None:
        st.error("Road network not available. Please try again.")
        return None
    
    # Map bars to network nodes
    try:
        # Project bars to the same CRS as the network
        bars_gdf = bars_gdf.to_crs(G.graph['crs'])
        
        # Find nearest nodes with error handling
        nearest_nodes = []
        for _, row in bars_gdf.iterrows():
            try:
                node = ox.distance.nearest_nodes(G, X=row.geometry.x, Y=row.geometry.y)
                nearest_nodes.append(node)
            except Exception:
                # Use a random node as fallback
                nearest_nodes.append(random.choice(list(G.nodes)))
        
        bars_gdf['nearest_node'] = nearest_nodes
    except Exception as e:
        st.error(f"Error mapping bars to network: {e}")
        return None
    
    # Try multiple starting bars to find the best route
    best_route = None
    best_num_bars = 0
    
    # Target number of bars (adjusted for available data)
    target_num_bars = min(num_bars, len(bars_gdf))
    
    # Choose starting bars
    max_attempts = min(5, len(bars_gdf))
    if start_bar_idx is not None:
        start_bars_to_try = [start_bar_idx]
    else:
        start_bars_to_try = random.sample(range(len(bars_gdf)), max_attempts)
    
    # Progress bar for route creation
    progress_bar = st.progress(0)
    
    for attempt, start_idx in enumerate(start_bars_to_try):
        progress_bar.progress((attempt) / len(start_bars_to_try))
        
        # Get the starting bar and its nearest node
        start_bar = bars_gdf.iloc[start_idx]
        current_node = start_bar['nearest_node']
        
        # List to store selected bar indices and their nodes
        selected_bar_indices = [start_idx]
        selected_bar_nodes = [current_node]
        
        # Continue selecting bars until we have enough
        for _ in range(target_num_bars - 1):
            if len(selected_bar_indices) >= len(bars_gdf):
                break
                
            # Get shortest paths from current node to all other bar nodes
            try:
                distances = nx.single_source_dijkstra_path_length(G, current_node, weight='length')
            except Exception:
                # BFS as fallback
                distances = {}
                queue = deque([(current_node, 0)])
                seen = {current_node}
                
                while queue:
                    node, dist = queue.popleft()
                    distances[node] = dist
                    
                    for neighbor in G.neighbors(node):
                        if neighbor not in seen:
                            seen.add(neighbor)
                            edge_len = G[node][neighbor][0].get('length', 50)
                            queue.append((neighbor, dist + edge_len))
            
            # Find bars at target spacing (with flexible range)
            flexibility = 0.2 + (len(selected_bar_indices) * 0.05)
            min_dist = bar_spacing * (1 - flexibility)
            max_dist = bar_spacing * (1 + flexibility)
            
            candidates = []
            for i, bar in bars_gdf.iterrows():
                if i not in selected_bar_indices and bar['nearest_node'] in distances:
                    dist_km = distances[bar['nearest_node']] / 1000.0
                    if min_dist <= dist_km <= max_dist:
                        candidates.append((i, bar['nearest_node'], dist_km))
            
            # If no bars in ideal range, widen search
            if not candidates:
                min_dist = bar_spacing * 0.5
                max_dist = bar_spacing * 1.5
                
                for i, bar in bars_gdf.iterrows():
                    if i not in selected_bar_indices and bar['nearest_node'] in distances:
                        dist_km = distances[bar['nearest_node']] / 1000.0
                        if min_dist <= dist_km <= max_dist:
                            candidates.append((i, bar['nearest_node'], dist_km))
            
            # If still no candidates, find any reasonably close bar
            if not candidates:
                for i, bar in bars_gdf.iterrows():
                    if i not in selected_bar_indices and bar['nearest_node'] in distances:
                        dist_km = distances[bar['nearest_node']] / 1000.0
                        if dist_km <= 10.0:  # Maximum 10km jump
                            candidates.append((i, bar['nearest_node'], dist_km))
            
            # Sort by how close to target spacing
            candidates.sort(key=lambda x: abs(x[2] - bar_spacing))
            
            # Select best candidate
            if candidates:
                next_bar_idx, next_node, dist = candidates[0]
                selected_bar_indices.append(next_bar_idx)
                selected_bar_nodes.append(next_node)
                current_node = next_node
            else:
                break
        
        # Compute route
        route_nodes = []
        total_length = 0
        bar_to_bar_paths = []
        valid_route = True
        
        for i in range(len(selected_bar_nodes) - 1):
            source = selected_bar_nodes[i]
            target = selected_bar_nodes[i + 1]
            
            try:
                path = nx.shortest_path(G, source, target, weight='length')
                path_length = sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) / 1000.0
                
                bar_to_bar_paths.append({
                    'from': selected_bar_indices[i],
                    'to': selected_bar_indices[i + 1],
                    'path': path,
                    'length_km': path_length
                })
                
                if not route_nodes:
                    route_nodes.extend(path)
                else:
                    route_nodes.extend(path[1:])
                
                total_length += path_length
                
            except nx.NetworkXNoPath:
                valid_route = False
                break
        
        # Only consider valid routes
        if valid_route:
            current_route = {
                'nodes': route_nodes,
                'length': total_length,
                'bar_nodes': selected_bar_nodes,
                'bar_indices': selected_bar_indices,
                'bar_to_bar_paths': bar_to_bar_paths
            }
            
            current_num_bars = len(selected_bar_indices)
            if current_num_bars > best_num_bars or (
                current_num_bars == best_num_bars and 
                (best_route is None or current_route['length'] < best_route['length'])
            ):
                best_route = current_route
                best_num_bars = current_num_bars
    
    progress_bar.progress(1.0)
    
    if best_route:
        selected_bars = bars_gdf.iloc[best_route['bar_indices']]
        st.success(f"Created route with {len(selected_bars)} bars and total length: {best_route['length']:.2f} km")
        return best_route, selected_bars
    else:
        st.error("Failed to create a valid route. Try different parameters.")
        return None, None

def visualize_route_streamlit(G, route, bars_gdf, selected_bars):
    """
    Create an interactive map for Streamlit display
    """
    if route is None:
        st.error("No route available to visualize.")
        return None
    
    st.info("🗺️ Creating interactive map...")
    
    # Create a transformer from the graph's CRS to lat/lon
    transformer = pyproj.Transformer.from_crs(G.graph['crs'], "EPSG:4326", always_xy=True)
    
    # Get the center coordinates
    bar_locs = [G.nodes[node] for node in route['bar_nodes']]
    lats = []
    lons = []
    
    for loc in bar_locs:
        try:
            lon, lat = transformer.transform(loc['x'], loc['y'])
            lats.append(lat)
            lons.append(lon)
        except Exception:
            # Use raw coordinates if transformation fails
            if 'lat' in loc and 'lon' in loc:
                lats.append(loc['lat'])
                lons.append(loc['lon'])
    
    if not lats or not lons:
        # Use the first bar's coordinates as fallback
        center_lat = selected_bars.iloc[0].geometry.y
        center_lon = selected_bars.iloc[0].geometry.x
    else:
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Add the route
    route_points = []
    for node in route['nodes']:
        try:
            x, y = G.nodes[node]['x'], G.nodes[node]['y']
            lon, lat = transformer.transform(x, y)
            route_points.append((lat, lon))
        except Exception:
            continue
    
    if route_points:
        folium.PolyLine(
            route_points,
            color='blue',
            weight=4,
            opacity=0.7,
            tooltip=f"Marathon Route ({route['length']:.2f} km)"
        ).add_to(m)
    
    # Add markers for the bars
    distance_so_far = 0
    prev_bar_loc = None
    
    for i, (node, bar_idx) in enumerate(zip(route['bar_nodes'], route['bar_indices'])):
        bar = bars_gdf.iloc[bar_idx]
        
        if prev_bar_loc is not None:
            try:
                path = nx.shortest_path(G, prev_bar_loc, node, weight='length')
                segment_length = sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) / 1000
                distance_so_far += segment_length
            except Exception:
                segment_length = 5.0  # Default estimate
                distance_so_far += segment_length
        
        # Format popup content
        if i == 0:
            popup_content = f"""
            <b>{bar['name']}</b><br>
            Start of the route<br>
            Distance: 0.00 km
            """
        else:
            popup_content = f"""
            <b>{bar['name']}</b><br>
            Bar #{i+1}<br>
            Distance from start: {distance_so_far:.2f} km
            """
        
        try:
            x, y = G.nodes[node]['x'], G.nodes[node]['y']
            lon, lat = transformer.transform(x, y)
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='red', icon='beer', prefix='fa'),
                tooltip=f"{bar['name']} ({distance_so_far:.2f} km)"
            ).add_to(m)
        except Exception:
            pass
        
        prev_bar_loc = node
    
    # Add other bars as smaller markers
    other_bars_indices = [i for i in range(len(bars_gdf)) if i not in route['bar_indices']]
    if other_bars_indices:
        other_bars = bars_gdf.iloc[other_bars_indices]
        
        marker_cluster = MarkerCluster(name="All Other Bars").add_to(m)
        
        for _, bar in other_bars.iterrows():
            try:
                x, y = G.nodes[bar['nearest_node']]['x'], G.nodes[bar['nearest_node']]['y']
                lon, lat = transformer.transform(x, y)
                
                folium.Marker(
                    location=[lat, lon],
                    popup=bar['name'],
                    icon=folium.Icon(color='green', icon='beer', prefix='fa'),
                    tooltip=bar['name']
                ).add_to(marker_cluster)
            except Exception:
                pass
    
    # Add layer control
    folium.LayerControl().add_to(m)
    return m

def get_route_directions(G, route, bars_gdf):
    """
    Generate step-by-step directions
    """
    if route is None:
        return None
    
    directions = []
    distance_so_far = 0
    prev_bar_loc = None
    
    for i, (node, bar_idx) in enumerate(zip(route['bar_nodes'], route['bar_indices'])):
        bar = bars_gdf.iloc[bar_idx]
        
        if prev_bar_loc is not None:
            path = nx.shortest_path(G, prev_bar_loc, node, weight='length')
            segment_length = sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) / 1000
            distance_so_far += segment_length
            
            # Get street names for the segment
            street_names = []
            prev_node = path[0]
            
            for curr_node in path[1:]:
                edge_data = G[prev_node][curr_node][0]
                street_name = edge_data.get('name', 'Unnamed street')
                
                if isinstance(street_name, list):
                    street_name = ', '.join(street_name)
                
                if not street_names or street_names[-1] != street_name:
                    street_names.append(street_name)
                
                prev_node = curr_node
            
            # Simplify the directions
            simplified_directions = []
            prev_street = None
            
            for street in street_names:
                if street != prev_street:
                    simplified_directions.append(street)
                    prev_street = street
            
            directions.append({
                'segment': f"Bar {i} to Bar {i+1}",
                'from_bar': bars_gdf.iloc[route['bar_indices'][i-1]]['name'],
                'to_bar': bar['name'],
                'distance_km': segment_length,
                'cumulative_distance_km': distance_so_far,
                'directions': " → ".join(simplified_directions) if simplified_directions else "Direct route"
            })
        else:
            directions.append({
                'segment': "Start",
                'from_bar': "N/A",
                'to_bar': bar['name'],
                'distance_km': 0.0,
                'cumulative_distance_km': 0.0,
                'directions': "Starting point"
            })
        
        prev_bar_loc = node
    
    return pd.DataFrame(directions)

def export_to_gpx(G, route, bars_gdf, city_name):
    """
    Export the route to a GPX file
    """
    if route is None:
        return None
    
    # Create a transformer from the graph's CRS to lat/lon
    transformer = pyproj.Transformer.from_crs(G.graph['crs'], "EPSG:4326", always_xy=True)
    
    # Create GPX object
    gpx = gpxpy.gpx.GPX()
    
    # Create track
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx_track.name = f"Bar Marathon in {city_name}"
    gpx_track.description = f"A {route['length']:.2f} km route with {len(route['bar_indices'])} bars"
    gpx.tracks.append(gpx_track)
    
    # Create segment
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    
    # Add track points from route nodes
    for node in route['nodes']:
        try:
            x, y = G.nodes[node]['x'], G.nodes[node]['y']
            lon, lat = transformer.transform(x, y)
            track_point = gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
            gpx_segment.points.append(track_point)
        except Exception:
            continue
    
    # Add waypoints for each bar
    distance_so_far = 0
    prev_bar_loc = None
    
    for i, (node, bar_idx) in enumerate(zip(route['bar_nodes'], route['bar_indices'])):
        bar = bars_gdf.iloc[bar_idx]
        
        if prev_bar_loc is not None:
            try:
                path = nx.shortest_path(G, prev_bar_loc, node, weight='length')
                segment_length = sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) / 1000
                distance_so_far += segment_length
            except Exception:
                distance_so_far += 5.0
        
        try:
            x, y = G.nodes[node]['x'], G.nodes[node]['y']
            lon, lat = transformer.transform(x, y)
            
            waypoint = gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon)
            waypoint.name = f"Bar {i+1}: {bar['name']}"
            waypoint.description = f"Bar #{i+1}: {bar['name']} ({bar['amenity']}) - Distance: {distance_so_far:.2f} km"
            waypoint.symbol = "Bar"
            
            gpx.waypoints.append(waypoint)
        except Exception:
            pass
        
        prev_bar_loc = node
    
    return gpx.to_xml()

import streamlit as st

# Header and app layout setup
st.set_page_config(
    page_title="Bar Marathon Planner",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #4A90E2;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #4A4A4A;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
        }
        .info-text {
            font-size: 1rem;
            color: #4A4A4A;
        }
        .highlight {
            color: #4A90E2;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #7A7A7A;
            font-size: 0.8rem;
        }
    </style>
    <div class="main-header">🍺 Bar Marathon Planner 🏃‍♂️</div>
    <div class="info-text" style="text-align: center;">Create a marathon-length (or any length!) route stopping at bars along the way!</div>
""", unsafe_allow_html=True)

# Sidebar: Select City
st.sidebar.header("🔍 Select Your City")
city_name = st.sidebar.selectbox(
    "Choose a city to load pre-cached data:",
    list(CACHED_CITIES.keys()),
    index=0
)

# Sidebar: Settings
st.sidebar.header("⚙️ Route Settings")
num_bars = st.sidebar.slider("Number of Bars", min_value=3, max_value=15, value=9)
bar_spacing_km = st.sidebar.slider("Target Distance Between Bars (km)", min_value=0.5, max_value=5.0, step=0.1, value=1.0)

# Load graph and venue data from cache
G, center_point = download_network_cached(city_name)
bars_gdf = find_bars_with_overpy_cached(city_name, center_point)

# Create bar marathon route
if G is not None and bars_gdf is not None:
    route, selected_bars = create_bar_marathon_route(G, bars_gdf, bar_spacing=bar_spacing_km, num_bars=num_bars)
    if route:
        # Show directions
        directions_df = get_route_directions(G, route, bars_gdf)
        with st.expander("🗺️ View Directions"):
            st.dataframe(directions_df)

        # Show interactive map
        folium_map = visualize_route_streamlit(G, route, bars_gdf, selected_bars)
        if folium_map:
            st.components.v1.html(folium_map._repr_html_(), height=600, scrolling=True)

        # Download GPX
        gpx_xml = export_to_gpx(G, route, bars_gdf, city_name)
        if gpx_xml:
            st.download_button(
                label="💾 Download GPX",
                data=gpx_xml,
                file_name=f"bar_marathon_{city_name.replace(',', '').replace(' ', '_').lower()}.gpx",
                mime="application/gpx+xml"
            )
    else:
        st.warning("No valid route found. Adjust settings or try another city.")
else:
    st.warning("Unable to load city data. Please try again later.")

# Footer
st.markdown('<p class="footer">Created with Streamlit, OSMnx, and a love for interesting adventures 🍺🌍</p>', 
            unsafe_allow_html=True)

# Import missing function for Folium map display in Streamlit
from streamlit_folium import st_folium

if __name__ == "__main__":
    main()
