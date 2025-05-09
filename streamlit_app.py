import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import os
import sys
import tempfile
import warnings
import time
from collections import deque
warnings.filterwarnings('ignore')

# Import the SimplifiedBarMarathonPlanner class
# Assuming the class is in a file named bar_marathon.py
from bar_marathon import SimplifiedBarMarathonPlanner

# Set page configuration
st.set_page_config(
    page_title="Bar Marathon Planner",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the appearance
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
        color: #4A4A4A;
        margin-top: 1rem;
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
""", unsafe_allow_html=True)

# Define functions for streamlit app
def display_bars_table(bars_gdf):
    """Display a table of bars found in the area"""
    if bars_gdf is not None and not bars_gdf.empty:
        st.write(f"Found {len(bars_gdf)} bars/pubs in the area:")
        
        # Create a simpler DataFrame for display
        display_df = bars_gdf[['name', 'amenity']].copy()
        display_df.columns = ['Bar Name', 'Type']
        
        # Display as a table
        st.dataframe(display_df, use_container_width=True)
    else:
        st.error("No bars found in the selected area.")

def display_route_summary(planner):
    """Display a summary of the created route"""
    if planner.route is not None and planner.selected_bars is not None:
        summary = planner.summarize_route()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Distance", f"{summary['total_distance_km']:.2f} km")
        with col2:
            st.metric("Number of Bars", f"{summary['num_bars']}")
        with col3:
            st.metric("Avg. Spacing", f"{summary['avg_bar_spacing_km']:.2f} km")
        
        st.markdown("### Selected Bars")
        
        # Create a DataFrame for the selected bars
        selected_df = planner.selected_bars[['name', 'amenity']].copy()
        selected_df.index = range(1, len(selected_df) + 1)  # 1-based indexing
        selected_df.columns = ['Bar Name', 'Type']
        
        # Display as a table
        st.dataframe(selected_df, use_container_width=True)

def display_directions(planner):
    """Display turn-by-turn directions for the route"""
    if planner.route is not None:
        directions_df = planner.get_route_directions()
        
        if directions_df is not None and not directions_df.empty:
            st.markdown("### Turn-by-Turn Directions")
            
            # Format the DataFrame for display
            display_df = directions_df.copy()
            display_df.columns = ['Segment', 'From', 'To', 'Distance (km)', 'Total Distance (km)', 'Directions']
            
            # Display as a table
            st.dataframe(display_df, use_container_width=True)

def save_gpx_file(planner):
    """Save the route as a GPX file and provide a download link"""
    if planner.route is not None:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        file_name = f"{planner.city_name.replace(' ', '_').replace(',', '')}_bar_marathon.gpx"
        file_path = os.path.join(temp_dir, file_name)
        
        # Export to GPX
        gpx_file = planner.export_to_gpx(filename=file_path)
        
        if gpx_file:
            # Read the file and provide a download button
            with open(file_path, 'r') as f:
                gpx_data = f.read()
            
            st.download_button(
                label="Download GPX Route",
                data=gpx_data,
                file_name=file_name,
                mime="application/gpx+xml",
                help="Download the route as a GPX file for use in navigation apps"
            )

# Main app
def main():
    st.markdown('<h1 class="main-header">🍺 Bar Marathon Planner 🏃‍♂️</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Create a marathon-length (or any length!) route stopping at bars along the way!</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("## Configure Your Bar Marathon")
        
        # City input
        city_name = st.text_input("City Name", value="Boston, MA, USA", 
                                help="Enter a city name (e.g., 'Boston, MA, USA', 'London, UK')")
        
        # Marathon parameters
        st.markdown("### Marathon Parameters")
        
        marathon_type = st.radio(
            "Marathon Distance Type",
            ["Standard (42.2km)", "Half (21.1km)", "Custom"]
        )
        
        if marathon_type == "Standard (42.2km)":
            total_distance = 42.2
            num_bars = st.slider("Number of Bars", min_value=3, max_value=15, value=9, 
                             help="Choose how many bars to include in your route")
            bar_spacing = total_distance / (num_bars - 1) if num_bars > 1 else total_distance
            st.info(f"Bar spacing: ~{bar_spacing:.2f} km")
            
        elif marathon_type == "Half (21.1km)":
            total_distance = 21.1
            num_bars = st.slider("Number of Bars", min_value=3, max_value=12, value=6, 
                             help="Choose how many bars to include in your route")
            bar_spacing = total_distance / (num_bars - 1) if num_bars > 1 else total_distance
            st.info(f"Bar spacing: ~{bar_spacing:.2f} km")
            
        else:  # Custom
            total_distance = st.number_input("Total Distance (km)", min_value=5.0, max_value=100.0, value=30.0, step=5.0)
            num_bars = st.slider("Number of Bars", min_value=3, max_value=20, value=8, 
                             help="Choose how many bars to include in your route")
            bar_spacing = st.number_input("Target Bar Spacing (km)", min_value=1.0, max_value=20.0, 
                                      value=total_distance / (num_bars - 1) if num_bars > 1 else total_distance, 
                                      step=0.5)
        
        # Advanced options
        with st.expander("Advanced Options"):
            radius_meters = st.slider("Search Radius (meters)", min_value=2000, max_value=10000, value=5000, step=500,
                                  help="Radius around city center to search for bars and create the route")
            network_type = st.selectbox("Network Type", ["walk", "bike", "drive"], index=0,
                                    help="Type of transportation for the route")
        
        # Create route button
        create_button = st.button("Create Bar Marathon", type="primary", use_container_width=True)
    
    # Main area for results
    if create_button:
        with st.spinner("Creating your bar marathon route... This may take a minute."):
            try:
                # Initialize the planner
                planner = SimplifiedBarMarathonPlanner(
                    city_name, 
                    bar_spacing=bar_spacing, 
                    num_bars=num_bars,
                    radius_meters=radius_meters
                )
                
                # Download the network
                with st.status("Downloading street network...") as status:
                    network = planner.download_network(network_type=network_type)
                    if network is None:
                        st.error(f"Failed to download network for {city_name}. Try a different city name.")
                        status.update(label="Network download failed", state="error")
                        return
                    status.update(label="Network downloaded successfully!", state="complete")
                
                # Find bars
                with st.status("Finding bars in the area...") as status:
                    bars = planner.find_bars_with_overpy(include_restaurants=include_restaurants)
                    if bars is None or len(bars) == 0:
                        st.error(f"No bars found in {city_name}. Try increasing the search radius or a different city.")
                        status.update(label="No bars found", state="error")
                        return
                    status.update(label=f"Found {len(bars)} bars!", state="complete")
                
                # Create route
                with st.status("Creating optimal route...") as status:
                    route = planner.create_simplified_route()
                    if route is None:
                        st.error("Failed to create a route. Try different parameters.")
                        status.update(label="Route creation failed", state="error")
                        return
                    status.update(label="Route created successfully!", state="complete")
                
                # Display results - tabs for different views
                tab1, tab2, tab3 = st.tabs(["Map & Summary", "Directions", "Export"])
                
                with tab1:
                    st.markdown('<h2 class="sub-header">Your Bar Marathon Route</h2>', unsafe_allow_html=True)
                    
                    # Map on the left, summary on the right
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### Interactive Map")
                        map_fig = planner.visualize_route()
                        if map_fig:
                            folium_static(map_fig, width=650, height=500)
                    
                    with col2:
                        st.markdown("### Route Summary")
                        display_route_summary(planner)
                
                with tab2:
                    st.markdown('<h2 class="sub-header">Navigation Details</h2>', unsafe_allow_html=True)
                    display_directions(planner)
                    
                    # Display all found bars
                    with st.expander("All Bars in the Area"):
                        display_bars_table(planner.bars_gdf)
                
                with tab3:
                    st.markdown('<h2 class="sub-header">Export Your Route</h2>', unsafe_allow_html=True)
                    
                    st.markdown("""
                    Download your bar marathon route as a GPX file that you can import into:
                    - Navigation apps like Google Maps
                    - Fitness trackers like Strava or Garmin
                    - GPS devices
                    """)
                    
                    save_gpx_file(planner)
                    
                    # Provide copy-paste command for using the planner in code
                    st.markdown("### Use in Python")
                    st.markdown("Copy this code to create the same route in Python:")
                    
                    code = f"""
                    # Import the planner
                    from bar_marathon import SimplifiedBarMarathonPlanner
                    
                    # Create the planner
                    planner = SimplifiedBarMarathonPlanner(
                        city_name="{city_name}", 
                        bar_spacing={bar_spacing}, 
                        num_bars={num_bars},
                        radius_meters={radius_meters}
                    )
                    
                    # Download the network
                    planner.download_network(network_type="{network_type}")
                    
                    # Find bars
                    planner.find_bars_with_overpy(include_restaurants={str(include_restaurants).lower()})
                    
                    # Create and display the route
                    planner.create_simplified_route()
                    map_fig = planner.visualize_route(save_html="bar_marathon_map.html")
                    
                    # Get directions and summary
                    directions = planner.get_route_directions()
                    summary = planner.summarize_route()
                    
                    # Export as GPX
                    planner.export_to_gpx("bar_marathon_route.gpx")
                    """
                    
                    st.code(code, language="python")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try with different parameters or a different city.")
    
    else:
        # Show instructions when the app first loads
        st.markdown("""
        ## 🍻 Welcome to the Bar Marathon Planner!
        
        Create a marathon-length route with stops at bars along the way. Perfect for:
        - Planning a beer marathon with friends
        - Creating a fun pub crawl through a new city
        - Combining exercise with socializing
        
        ### How to use:
        1. Enter a city name in the sidebar
        2. Choose your marathon distance type and number of bars
        3. Adjust advanced options if needed
        4. Click "Create Bar Marathon"
        
        The app will create an optimal route connecting bars with your specified spacing.
        """)
        
        # Add sample route image if available
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/29/A_toast.jpg", 
                caption="Plan your next bar marathon adventure!")

    # Footer
    st.markdown('<p class="footer">Created with Streamlit, OSMnx, and a love for interesting adventures</p>', 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
