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

ox.settings.use_cache = False  # <-- Most memory-safe

print(f"Cache enabled at: {ox.settings.cache_folder}")

class SimplifiedBarMarathonPlanner:
    def __init__(self, city_name, bar_spacing=5.0, num_bars=9,
                center_point=None, radius_meters=5000):
        """
        Initialize the Improved Bar Marathon Planner.

        Parameters:
        -----------
        city_name : str
            Name of the city to create the route in
        bar_spacing : float
            Target spacing between bars in kilometers (default: 5.0)
        num_bars : int
            Number of bars to include in the route (default: 9)
        center_point : tuple of (lat, lon), optional
            Geographic center of the network (default: geocoded from city_name)
        radius_meters : int
            Radius in meters around the center point to define the network and bar area
        """
        self.city_name = city_name
        self.bar_spacing = bar_spacing
        self.num_bars = num_bars
        self.center_point = center_point
        self.radius_meters = radius_meters
        self.G = None
        self.bars_gdf = None
        self.route = None
        self.selected_bars = None
        self.transformer = None
        
    def download_network(self, network_type='walk', retry_count=3):
        """
        Download the street network around a center point or city center.
        
        Parameters:
        -----------
        network_type : str
            Type of network to download ('walk', 'drive', 'bike', etc.)
        retry_count : int
            Number of retries if download fails
        """
        print(f"Downloading {network_type} network around {self.city_name}...")

        for attempt in range(retry_count):
            try:
                if self.center_point is None:
                    self.center_point = ox.geocode(self.city_name)
                    print(f"Center point: {self.center_point}")

                # Increased timeout for large cities
                ox.settings.timeout = 180
                
                # Try to download a connected graph
                self.G = ox.graph_from_point(self.center_point,
                                          dist=self.radius_meters,
                                          network_type=network_type,
                                          simplify=True)
                
                # Ensure the graph is projected to a local CRS for accurate distance calculations
                self.G = ox.project_graph(self.G)
                
                # Check if graph is connected - if not, get the largest connected component
                if not nx.is_strongly_connected(self.G):
                    largest_cc = max(nx.strongly_connected_components(self.G), key=len)
                    self.G = self.G.subgraph(largest_cc).copy()
                    print(f"Using largest connected component with {len(self.G.nodes)} nodes")
                
                # Create the transformer for coordinate conversion
                self.transformer = pyproj.Transformer.from_crs(
                    self.G.graph['crs'], "EPSG:4326", always_xy=True)
                
                print(f"Network downloaded with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
                
                # Only keep named edges if we have enough
                named_edges = [(u, v, k) for u, v, k, data in self.G.edges(keys=True, data=True) 
                              if 'name' in data and data['name']]
                
                if len(named_edges) > len(self.G.edges()) * 0.5:  # If at least 50% of edges are named
                    self.filter_to_named_edges()
                else:
                    print("Keeping all edges including unnamed ones to ensure connectivity")
                
                return self.G

            except Exception as e:
                print(f"Error downloading network (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                    # Maybe try with a different radius
                    if self.radius_meters > 3000:
                        self.radius_meters = int(self.radius_meters * 0.8)
                        print(f"Reducing search radius to {self.radius_meters}m")
                else:
                    print("All attempts failed. Try with a different city or parameters.")
                    return None
        
    def filter_to_named_edges(self):
        """
        Removes edges without a 'name' attribute from the graph to ensure only named streets are used.
        Makes sure the graph remains connected after filtering.
        """
        if self.G is None:
            print("Graph is not loaded.")
            return

        # Count edges before filtering
        initial_edge_count = len(self.G.edges)
        
        # Find edges to remove
        edges_to_remove = []
        for u, v, key, data in self.G.edges(keys=True, data=True):
            if 'name' not in data or not data['name']:
                edges_to_remove.append((u, v, key))

        # Create a temporary graph to test connectivity
        test_G = self.G.copy()
        test_G.remove_edges_from(edges_to_remove)
        
        # Check if the graph would remain connected
        if nx.is_strongly_connected(test_G):
            # Safe to remove edges
            self.G.remove_edges_from(edges_to_remove)
            print(f"Filtered out {len(edges_to_remove)} unnamed edges.")
        else:
            # Not safe to remove all edges - find the largest connected component
            largest_cc = max(nx.strongly_connected_components(test_G), key=len)
            
            # Only keep the edges that are part of the largest component
            self.G = test_G.subgraph(largest_cc).copy()
            print(f"Filtered unnamed edges but kept the largest connected component " 
                  f"({len(self.G.nodes)} nodes, {len(self.G.edges)} edges).")
            print(f"Removed {initial_edge_count - len(self.G.edges)} edges total.")
        
    def find_bars_with_overpy(self, retry_count=3, include_restaurants=False):
        """
        Find bars and pubs in the radius using Overpy.
        
        Parameters:
        -----------
        retry_count : int
            Number of retries if query fails
        include_restaurants : bool
            Whether to include restaurants with alcohol service as fallback
            
        Returns:
        --------
        GeoDataFrame of bars
        """
        print(f"Finding bars near {self.city_name} using Overpy...")

        if self.center_point is None:
            self.center_point = ox.geocode(self.city_name)
        lat, lon = self.center_point
        radius_meters = self.radius_meters

        # Initialize the Overpass API
        api = overpy.Overpass()

        # Start with just bars and pubs
        amenity_query = '"amenity"~"bar|pub|biergarten|brewery|wine_bar"'
        
        for attempt in range(retry_count):
            try:
                # Construct Overpass query using 'around' keyword
                # add a line for "Flat Top Johnny's" as a location in the query
                # add biergarten to the query
                query = f"""
                [out:json][timeout:90];
                (
                node[{amenity_query}](around:{radius_meters},{lat},{lon});
                way[{amenity_query}](around:{radius_meters},{lat},{lon});
                relation[{amenity_query}](around:{radius_meters},{lat},{lon});
                node["name"="Flat Top Johnny's"](around:{radius_meters},{lat},{lon});

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

                # If we don't have enough locations and this is the last attempt, try including restaurants
                if len(locations) < self.num_bars * 2 and attempt == retry_count - 1 and include_restaurants:
                    print(f"Only found {len(locations)} bars/pubs. Adding restaurants with alcohol...")
                    
                    # Query for restaurants with alcohol service
                    restaurant_query = """
                    [out:json][timeout:90];
                    (
                    node["amenity"="restaurant"]["alcohol"="yes"](around:{0},{1},{2});
                    way["amenity"="restaurant"]["alcohol"="yes"](around:{0},{1},{2});
                    relation["amenity"="restaurant"]["alcohol"="yes"](around:{0},{1},{2});
                    );
                    out center;
                    """.format(radius_meters, lat, lon)
                    
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
                        print(f"Error querying for restaurants: {e}")

                # Create DataFrame with the results
                df = pd.DataFrame(locations)
                if df.empty:
                    if attempt < retry_count - 1:
                        print("No bars found. Retrying with larger radius...")
                        self.radius_meters = int(self.radius_meters * 1.5)
                        continue
                    else:
                        print("No bars found after all attempts.")
                        return None

                # Convert to GeoDataFrame
                self.bars_gdf = gpd.GeoDataFrame(
                    df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326"
                )
                # remove public_bookcase from the bars_gdf amenity and reset index
                self.bars_gdf = self.bars_gdf[self.bars_gdf['amenity'] != 'public_bookcase']
                self.bars_gdf = self.bars_gdf[self.bars_gdf['name'] != "Jacque's Cabaret"]
                self.bars_gdf = self.bars_gdf.reset_index(drop=True)
                # Map bars to network nodes
                if self.G is not None:
                    self.bars_gdf = self.bars_gdf.to_crs(self.G.graph['crs'])
                    
                    # Find nearest nodes with error handling
                    nearest_nodes = []
                    for _, row in self.bars_gdf.iterrows():
                        try:
                            node = ox.distance.nearest_nodes(
                                self.G, X=row.geometry.x, Y=row.geometry.y)
                            nearest_nodes.append(node)
                        except Exception as e:
                            print(f"Error finding nearest node for bar {row['name']}: {e}")
                            # Use a random node as fallback (better than crashing)
                            nearest_nodes.append(random.choice(list(self.G.nodes)))
                    
                    self.bars_gdf['nearest_node'] = nearest_nodes

                print(f"Found {len(self.bars_gdf)} bars/pubs within {self.radius_meters}m of center.")
                return self.bars_gdf

            except Exception as e:
                print(f"Error querying Overpass API (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    print("Retrying with increased radius...")
                    self.radius_meters = int(self.radius_meters * 1.5)
                    time.sleep(3)  # Avoid overwhelming the API
                else:
                    print("All attempts failed. Try with a different city or parameters.")
                    return None
    
    def create_simplified_route(self, start_bar_idx=None, max_attempts=5):
        """
        Create a simplified route by selecting bars that are properly spaced.
        
        Parameters:
        -----------
        start_bar_idx : int, optional
            Index of the bar to start the route from. If None, will select a random bar.
        max_attempts : int
            Maximum number of attempts to create a valid route with different starting points
            
        Returns:
        --------
        dict with route information
        """
        if self.bars_gdf is None or len(self.bars_gdf) == 0:
            print("No bars found. Run find_bars_with_overpy() first.")
            return None
        
        if self.G is None:
            print("Road network not loaded. Run download_network() first.")
            return None
        
        print("Creating simplified marathon route with bars...")
        
        # Try multiple starting bars if needed
        best_route = None
        best_num_bars = 0
        
        # If we have few bars, we need to make the most of them
        target_num_bars = min(self.num_bars, len(self.bars_gdf))
        
        # List of starting bars to try - either the specified one or a sample of candidates
        if start_bar_idx is not None:
            start_bars_to_try = [start_bar_idx]
        else:
            # Choose different starting bars in different areas
            num_start_candidates = min(max_attempts, len(self.bars_gdf))
            start_bars_to_try = random.sample(range(len(self.bars_gdf)), num_start_candidates)
        
        for attempt, start_idx in enumerate(start_bars_to_try):
            print(f"Attempt {attempt+1}/{len(start_bars_to_try)}: Starting from bar {start_idx}")
            
            # Get the starting bar and its nearest node
            start_bar = self.bars_gdf.iloc[start_idx]
            current_node = start_bar['nearest_node']
            
            # List to store selected bar indices and their nodes
            selected_bar_indices = [start_idx]
            selected_bar_nodes = [current_node]
            
            # Continue selecting bars until we have enough or run out of candidates
            for _ in range(target_num_bars - 1):
                if len(selected_bar_indices) >= len(self.bars_gdf):
                    break
                    
                # Get shortest paths from current node to all other bar nodes
                try:
                    # Adding weight_method to handle different graph structures
                    distances = nx.single_source_dijkstra_path_length(
                        self.G, current_node, weight='length')
                except Exception as e:
                    print(f"Error computing distances from node {current_node}: {e}")
                    # Try to use an alternative method
                    try:
                        # BFS as fallback to find connected nodes
                        distances = {}
                        queue = deque([(current_node, 0)])
                        seen = {current_node}
                        
                        while queue:
                            node, dist = queue.popleft()
                            distances[node] = dist
                            
                            for neighbor in self.G.neighbors(node):
                                if neighbor not in seen:
                                    seen.add(neighbor)
                                    # Use edge length if available, otherwise default to 50m
                                    edge_len = self.G[node][neighbor][0].get('length', 50)
                                    queue.append((neighbor, dist + edge_len))
                    except Exception as nested_e:
                        print(f"Alternative distance computation also failed: {nested_e}")
                        break
                
                # Find bars that are approximately at target spacing (with flexible range)
                # The flexibility increases as we add more bars
                flexibility = 0.2 + (len(selected_bar_indices) * 0.05)
                min_dist = self.bar_spacing * (1 - flexibility)
                max_dist = self.bar_spacing * (1 + flexibility)
                
                candidates = []
                for i, bar in self.bars_gdf.iterrows():
                    if i not in selected_bar_indices and bar['nearest_node'] in distances:
                        dist_km = distances[bar['nearest_node']] / 1000.0
                        if min_dist <= dist_km <= max_dist:
                            candidates.append((i, bar['nearest_node'], dist_km))
                
                # If no bars in the ideal range, widen the range
                if not candidates:
                    min_dist = self.bar_spacing * 0.5
                    max_dist = self.bar_spacing * 1.5
                    
                    for i, bar in self.bars_gdf.iterrows():
                        if i not in selected_bar_indices and bar['nearest_node'] in distances:
                            dist_km = distances[bar['nearest_node']] / 1000.0
                            if min_dist <= dist_km <= max_dist:
                                candidates.append((i, bar['nearest_node'], dist_km))
                
                # If still no candidates, find any available bar
                if not candidates:
                    for i, bar in self.bars_gdf.iterrows():
                        if i not in selected_bar_indices and bar['nearest_node'] in distances:
                            dist_km = distances[bar['nearest_node']] / 1000.0
                            # Only consider reasonably close bars (within 10km)
                            if dist_km <= 10.0:
                                candidates.append((i, bar['nearest_node'], dist_km))
                
                # Sort by how close the distance is to the target spacing
                candidates.sort(key=lambda x: abs(x[2] - self.bar_spacing))
                
                # If we have candidates, select the best one
                if candidates:
                    next_bar_idx, next_node, dist = candidates[0]
                    selected_bar_indices.append(next_bar_idx)
                    selected_bar_nodes.append(next_node)
                    current_node = next_node
                else:
                    print(f"No more suitable bars found. Route has {len(selected_bar_indices)} bars.")
                    break
            
            # Compute the full route by connecting all selected bars
            route_nodes = []
            total_length = 0
            bar_to_bar_paths = []
            
            # Flag to check if the route is valid
            valid_route = True
            
            for i in range(len(selected_bar_nodes) - 1):
                source = selected_bar_nodes[i]
                target = selected_bar_nodes[i + 1]
                
                try:
                    path = nx.shortest_path(self.G, source, target, weight='length')
                    path_length = sum(self.G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) / 1000.0
                    
                    bar_to_bar_paths.append({
                        'from': selected_bar_indices[i],
                        'to': selected_bar_indices[i + 1],
                        'path': path,
                        'length_km': path_length
                    })
                    
                    # Add to route nodes
                    if not route_nodes:
                        route_nodes.extend(path)
                    else:
                        # Avoid duplicating the connecting node
                        route_nodes.extend(path[1:])
                    
                    total_length += path_length
                    
                except nx.NetworkXNoPath:
                    print(f"No path found between bars {selected_bar_indices[i]} and {selected_bar_indices[i + 1]}")
                    valid_route = False
                    break
            
            # Only consider this route if it's valid
            if valid_route:
                # Store route information
                current_route = {
                    'nodes': route_nodes,
                    'length': total_length,
                    'bar_nodes': selected_bar_nodes,
                    'bar_indices': selected_bar_indices,
                    'bar_to_bar_paths': bar_to_bar_paths
                }
                
                # Check if this route is better than previous ones
                current_num_bars = len(selected_bar_indices)
                if current_num_bars > best_num_bars or (
                    current_num_bars == best_num_bars and 
                    (best_route is None or current_route['length'] < best_route['length'])
                ):
                    best_route = current_route
                    best_num_bars = current_num_bars
                
                print(f"Created route with {current_num_bars} bars and length: {total_length:.2f} km")
                
                # If we've reached the target number of bars, no need to try more
                if current_num_bars >= target_num_bars:
                    break
        
        # Use the best route we found
        if best_route:
            self.route = best_route
            self.selected_bars = self.bars_gdf.iloc[self.route['bar_indices']]
            print(f"Selected best route with {len(self.selected_bars)} bars and total length: {self.route['length']:.2f} km")
            return self.route
        else:
            print("Failed to create a valid route. Try different parameters or city.")
            return None
    
    def visualize_route(self, save_html=None):
        """
        Visualize the route with bars on an interactive map.
        
        Parameters:
        -----------
        save_html : str, optional
            Path to save the HTML map file. If None, won't save.
        
        Returns:
        --------
        folium.Map
            Interactive folium map with the route and bars
        """
        if self.route is None:
            print("No route available. Run create_simplified_route() first.")
            return None
        
        if self.transformer is None:
            # Create a transformer from the graph's CRS to lat/lon
            projected_crs = self.G.graph['crs']
            self.transformer = pyproj.Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
        
        # Transform bar locations for the center of the map
        bar_locs = [self.G.nodes[node] for node in self.route['bar_nodes']]
        lats = []
        lons = []
        
        for loc in bar_locs:
            try:
                lon, lat = self.transformer.transform(loc['x'], loc['y'])
                lats.append(lat)
                lons.append(lon)
            except Exception as e:
                print(f"Error transforming coordinates: {e}")
                # Use raw coordinates if transformation fails
                if 'lat' in loc and 'lon' in loc:
                    lats.append(loc['lat'])
                    lons.append(loc['lon'])
                    
        if not lats or not lons:
            print("Could not determine map center coordinates.")
            # Use the original center point as fallback
            center_lat, center_lon = self.center_point
        else:
            center_lat = np.mean(lats)
            center_lon = np.mean(lons)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        
        # Add the route with properly transformed coordinates
        route_points = []
        for node in self.route['nodes']:
            try:
                # Get node coordinates in projected CRS
                x, y = self.G.nodes[node]['x'], self.G.nodes[node]['y']
                # Transform to lat/lon coordinates
                lon, lat = self.transformer.transform(x, y)
                # Add to route points (folium uses lat, lon order)
                route_points.append((lat, lon))
            except Exception as e:
                print(f"Error adding route point: {e}")
                continue
        
        # Plot the route as a line
        if route_points:
            folium.PolyLine(
                route_points,
                color='blue',
                weight=4,
                opacity=0.7,
                tooltip=f"Marathon Route ({self.route['length']:.2f} km)"
            ).add_to(m)
        
        # Add markers for the bars with distance indicators
        distance_so_far = 0
        prev_bar_loc = None
        
        for i, (node, bar_idx) in enumerate(zip(self.route['bar_nodes'], self.route['bar_indices'])):
            bar = self.bars_gdf.iloc[bar_idx]
            
            if prev_bar_loc is not None:
                try:
                    # Calculate distance from previous bar
                    path = nx.shortest_path(self.G, prev_bar_loc, node, weight='length')
                    segment_length = sum(self.G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) / 1000
                    distance_so_far += segment_length
                except Exception as e:
                    print(f"Error calculating segment length: {e}")
                    # Estimate distance as fallback
                    segment_length = 5.0  # Default estimate
                    distance_so_far += segment_length
            
            # Format the popup content
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
                # Transform node coordinates for the marker
                x, y = self.G.nodes[node]['x'], self.G.nodes[node]['y']
                lon, lat = self.transformer.transform(x, y)
                
                # Add the marker
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_content, max_width=300),
                    icon=folium.Icon(color='red', icon='beer', prefix='fa'),
                    tooltip=f"{bar['name']} ({distance_so_far:.2f} km)"
                ).add_to(m)
            except Exception as e:
                print(f"Error adding bar marker: {e}")
            
            prev_bar_loc = node
        
        # Add all other bars as smaller markers in a cluster
        other_bars_indices = [i for i in range(len(self.bars_gdf)) if i not in self.route['bar_indices']]
        if other_bars_indices:
            other_bars = self.bars_gdf.iloc[other_bars_indices]
            
            marker_cluster = MarkerCluster(name="All Other Bars").add_to(m)
            
            for _, bar in other_bars.iterrows():
                try:
                    # Transform node coordinates for each marker
                    x, y = self.G.nodes[bar['nearest_node']]['x'], self.G.nodes[bar['nearest_node']]['y']
                    lon, lat = self.transformer.transform(x, y)
                    
                    folium.Marker(
                        location=[lat, lon],
                        popup=bar['name'],
                        icon=folium.Icon(color='green', icon='beer', prefix='fa'),
                        tooltip=bar['name']
                    ).add_to(marker_cluster)
                except Exception as e:
                    print(f"Error adding other bar marker: {e}")
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save if requested
        if save_html:
            try:
                m.save(save_html)
                print(f"Map saved to {save_html}")
            except Exception as e:
                print(f"Error saving map: {e}")
        
        return m
    
    
    def get_route_directions(self):
        """
        Generate step-by-step directions for the marathon bar crawl.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with route directions
        """
        if self.route is None:
            print("No route available. Run create_simplified_route() first.")
            return None
        
        directions = []
        distance_so_far = 0
        prev_bar_loc = None
        
        for i, (node, bar_idx) in enumerate(zip(self.route['bar_nodes'], self.route['bar_indices'])):
            bar = self.bars_gdf.iloc[bar_idx]
            
            if prev_bar_loc is not None:
                # Calculate distance from previous bar
                path = nx.shortest_path(self.G, prev_bar_loc, node, weight='length')
                segment_length = sum(self.G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) / 1000
                distance_so_far += segment_length
                
                # Get street names for the segment
                street_names = []
                prev_node = path[0]
                
                for curr_node in path[1:]:
                    edge_data = self.G[prev_node][curr_node][0]
                    street_name = edge_data.get('name', 'Unnamed street')
                    
                    # Convert list of street names to a single string if necessary
                    if isinstance(street_name, list):
                        street_name = ', '.join(street_name)
                    
                    if not street_names or street_names[-1] != street_name:
                        street_names.append(street_name)
                    
                    prev_node = curr_node
                
                # Simplify the directions by combining consecutive segments on the same street
                simplified_directions = []
                prev_street = None
                
                for street in street_names:
                    if street != prev_street:
                        simplified_directions.append(street)
                        prev_street = street
                
                directions.append({
                    'segment': f"Bar {i} to Bar {i+1}",
                    'from_bar': self.bars_gdf.iloc[self.route['bar_indices'][i-1]]['name'],
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
    
    def export_to_gpx(self, filename=None):
        """
        Export the route to a GPX file for use in navigation apps.
        
        Parameters:
        -----------
        filename : str, optional
            Path to save the GPX file. If None, a default name based on the city will be used.
            
        Returns:
        --------
        str
            Path to the saved GPX file
        """
        if self.route is None:
            print("No route available. Run create_simplified_route() first.")
            return None
        
        if self.transformer is None:
            # Create a transformer from the graph's CRS to lat/lon
            projected_crs = self.G.graph['crs']
            self.transformer = pyproj.Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
        
        # Generate default filename if not provided
        if filename is None:
            filename = f"{self.city_name.replace(' ', '_').replace(',', '')}_bar_marathon.gpx"
        
        # Create GPX object
        gpx = gpxpy.gpx.GPX()
        
        # Create track
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx_track.name = f"Bar Marathon in {self.city_name}"
        gpx_track.description = f"A {self.route['length']:.2f} km route with {len(self.selected_bars)} bars"
        gpx.tracks.append(gpx_track)
        
        # Create segment
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        
        # Add track points from route nodes
        for node in self.route['nodes']:
            try:
                # Get node coordinates in projected CRS
                x, y = self.G.nodes[node]['x'], self.G.nodes[node]['y']
                # Transform to lon/lat coordinates (GPX uses lon, lat order)
                lon, lat = self.transformer.transform(x, y)
                
                # Create track point
                track_point = gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
                gpx_segment.points.append(track_point)
            except Exception as e:
                print(f"Error adding route point to GPX: {e}")
                continue
        
        # Add waypoints for each bar
        distance_so_far = 0
        prev_bar_loc = None
        
        for i, (node, bar_idx) in enumerate(zip(self.route['bar_nodes'], self.route['bar_indices'])):
            bar = self.bars_gdf.iloc[bar_idx]
            
            # Calculate cumulative distance for naming
            if prev_bar_loc is not None:
                try:
                    path = nx.shortest_path(self.G, prev_bar_loc, node, weight='length')
                    segment_length = sum(self.G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) / 1000
                    distance_so_far += segment_length
                except Exception as e:
                    print(f"Error calculating distance for waypoint: {e}")
                    # Use an estimate as fallback
                    distance_so_far += 5.0  # Default estimate
            
            try:
                # Transform node coordinates for the waypoint
                x, y = self.G.nodes[node]['x'], self.G.nodes[node]['y']
                lon, lat = self.transformer.transform(x, y)
                
                # Create waypoint with bar information
                waypoint = gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon)
                waypoint.name = f"Bar {i+1}: {bar['name']}"
                waypoint.description = f"Bar #{i+1}: {bar['name']} ({bar['amenity']}) - Distance: {distance_so_far:.2f} km"
                
                # Set waypoint symbol to something that represents a bar
                waypoint.symbol = "Bar"
                
                # Add waypoint to GPX
                gpx.waypoints.append(waypoint)
            except Exception as e:
                print(f"Error adding bar waypoint to GPX: {e}")
            
            prev_bar_loc = node
        
        # Write GPX to file
        try:
            with open(filename, 'w') as f:
                f.write(gpx.to_xml())
            print(f"GPX file saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving GPX file: {e}")
            return None

    def summarize_route(self):
        """
        Generate a summary of the marathon bar crawl route.
        
        Returns:
        --------
        dict
            Dictionary with route summary information
        """
        if self.route is None:
            print("No route available. Run create_simplified_route() first.")
            return None
        
        # Calculate statistics
        num_bars = len(self.selected_bars)
        total_distance = self.route['length']
        bar_distances = []
        
        prev_bar_loc = None
        for node in self.route['bar_nodes']:
            if prev_bar_loc is not None:
                path = nx.shortest_path(self.G, prev_bar_loc, node, weight='length')
                segment_length = sum(self.G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:])) / 1000
                bar_distances.append(segment_length)
            prev_bar_loc = node
        
        summary = {
            'city': self.city_name,
            'total_distance_km': total_distance,
            'num_bars': num_bars,
            'avg_bar_spacing_km': np.mean(bar_distances) if bar_distances else 0,
            'min_bar_spacing_km': min(bar_distances) if bar_distances else 0,
            'max_bar_spacing_km': max(bar_distances) if bar_distances else 0,
            'bars': self.selected_bars[['name', 'amenity']].to_dict('records')
        }
        
        return summary
