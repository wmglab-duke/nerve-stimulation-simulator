#!/usr/bin/env python3
"""
Nerve Stimulation Simulator - Streamlit GUI

Interactive web interface for the nerve stimulation simulator.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Add parent directory to path to import the simulator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nerve_stimulation_simulator import (
    NerveGeometry, FiberPopulation, LaplaceSolver, NerveStimulationSimulator,
    GRID_SIZE, DOMAIN_SIZE, NERVE_CENTER_X, NERVE_CENTER_Y, NERVE_RADIUS,
    FASCICLE_COUNT, FASCICLE_RADIUS, ELECTRODE_COUNT, ELECTRODE_RADIUS,
    FIBER_COUNT_PER_FASCICLE, FIBER_DIAMETER_RANGE, FIBER_THRESHOLD_BASE,
    FIBER_THRESHOLD_SCALING
)

# Page configuration
st.set_page_config(
    page_title="Nerve Stimulation Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_simulator():
    """Initialize the simulator with current parameters."""
    if 'simulator' not in st.session_state:
        with st.spinner("Initializing simulator..."):
            st.session_state.simulator = NerveStimulationSimulator()
    return st.session_state.simulator

def create_parameter_sidebar():
    """Create the parameter control sidebar."""
    st.sidebar.header("🧠 Simulation Parameters")
    
    # Grid parameters
    st.sidebar.subheader("Grid Settings")
    grid_size = st.sidebar.slider("Grid Resolution", 80, 200, 120, 20, 
                                 help="Higher resolution = more accurate but slower")
    
    # Nerve parameters
    st.sidebar.subheader("Nerve Geometry")
    nerve_radius = st.sidebar.slider("Nerve Radius (mm)", 0.2, 0.6, 0.4, 0.05)
    fascicle_count = st.sidebar.slider("Number of Fascicles", 2, 5, 3, 1)
    fascicle_radius = st.sidebar.slider("Fascicle Radius (mm)", 0.05, 0.15, 0.08, 0.01)
    
    # Conductivity parameters
    st.sidebar.subheader("Tissue Conductivity (S/m)")
    epineurium_cond = st.sidebar.number_input("Epineurium", 0.0001, 1.0, 0.1, 0.01, 
                                            format="%.4f", help="Nerve sheath conductivity")
    endoneurium_cond = st.sidebar.number_input("Endoneurium", 0.1, 2.0, 0.5, 0.1, 
                                             format="%.1f", help="Inside fascicles conductivity")
    perineurium_cond = st.sidebar.number_input("Perineurium", 0.00001, 0.1, 0.001, 0.0001, 
                                             format="%.5f", help="Fascicle boundary conductivity")
    outside_cond = st.sidebar.number_input("Outside Nerve", 0.1, 10.0, 0.2, 0.1, 
                                         format="%.1f", help="Surrounding tissue conductivity")
    
    # Electrode parameters
    st.sidebar.subheader("Electrode Settings")
    electrode_radius = st.sidebar.slider("Electrode Radius (mm)", 0.01, 0.05, 0.02, 0.005)
    
    # Electrode amplitudes
    st.sidebar.subheader("Electrode Amplitudes (V)")
    electrode_1 = st.sidebar.slider("Electrode 1", -2.0, 2.0, 0.0, 0.1)
    electrode_2 = st.sidebar.slider("Electrode 2", -2.0, 2.0, 2.0, 0.1)
    electrode_3 = st.sidebar.slider("Electrode 3", -2.0, 2.0, -2.0, 0.1)
    electrode_4 = st.sidebar.slider("Electrode 4", -2.0, 2.0, -1.0, 0.1)
    
    # Fiber parameters
    st.sidebar.subheader("Fiber Settings")
    fiber_count = st.sidebar.slider("Fibers per Fascicle", 10, 100, 30, 10)
    fiber_diameter_min = st.sidebar.slider("Min Fiber Diameter (μm)", 1.0, 20.0, 2.0, 0.5)
    fiber_diameter_max = st.sidebar.slider("Max Fiber Diameter (μm)", 5.0, 25.0, 12.0, 0.5)
    threshold_base = st.sidebar.number_input("Base Threshold (V)", 0.01, 1.0, 0.1, 0.01, 
                                           format="%.3f", help="Base activation threshold")
    
    return {
        'grid_size': grid_size,
        'nerve_radius': nerve_radius,
        'fascicle_count': fascicle_count,
        'fascicle_radius': fascicle_radius,
        'epineurium_cond': epineurium_cond,
        'endoneurium_cond': endoneurium_cond,
        'perineurium_cond': perineurium_cond,
        'outside_cond': outside_cond,
        'electrode_radius': electrode_radius,
        'electrode_amplitudes': [electrode_1, electrode_2, electrode_3, electrode_4],
        'fiber_count': fiber_count,
        'fiber_diameter_range': (fiber_diameter_min, fiber_diameter_max),
        'threshold_base': threshold_base
    }

def update_simulator_parameters(simulator, params):
    """Update simulator parameters if they've changed."""
    # This would require modifying the original simulator to accept parameters
    # For now, we'll use the default parameters and just update electrode amplitudes
    pass

def create_visualization(simulator, electrode_amplitudes):
    """Create the main visualization."""
    # Update simulation with current electrode amplitudes
    potential_field = simulator.update_simulation(electrode_amplitudes)
    
    # Create single figure focused on the nerve
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Set title
    ax.set_title('Nerve Stimulation Simulation', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_aspect('equal')
    
    # Plot potential field
    X, Y = simulator.geometry.X, simulator.geometry.Y
    vmax = max(abs(np.min(potential_field)), abs(np.max(potential_field)))
    contour_levels = np.linspace(-vmax, vmax, 21)
    contour_plot = ax.contourf(X, Y, potential_field, levels=contour_levels, 
                              cmap='RdBu_r', alpha=0.7, extend='both', vmin=-vmax, vmax=vmax)
    
    # Add contour lines
    ax.contour(X, Y, potential_field, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(contour_plot, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Potential (V)', rotation=270, labelpad=20, fontsize=12)
    
    # Draw nerve structure
    draw_nerve_structure(ax, simulator.geometry)
    
    # Plot fibers
    plot_fibers(ax, simulator.fiber_population)
    
    # Zoom in on the nerve area
    nerve_center_x, nerve_center_y = NERVE_CENTER_X, NERVE_CENTER_Y
    nerve_radius = NERVE_RADIUS
    zoom_margin = 0.3  # Add some margin around the nerve
    
    ax.set_xlim(nerve_center_x - nerve_radius - zoom_margin, 
                nerve_center_x + nerve_radius + zoom_margin)
    ax.set_ylim(nerve_center_y - nerve_radius - zoom_margin, 
                nerve_center_y + nerve_radius + zoom_margin)
    
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    return fig

def draw_nerve_structure(ax, geometry):
    """Draw nerve and fascicle boundaries."""
    # Draw nerve boundary
    nerve_circle = patches.Circle(
        (NERVE_CENTER_X, NERVE_CENTER_Y), NERVE_RADIUS,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(nerve_circle)
    
    # Draw fascicle boundaries and perineurium
    for center in geometry.fascicle_centers:
        # Fascicle boundary
        fascicle_circle = patches.Circle(
            center, FASCICLE_RADIUS,
            fill=False, edgecolor='gray', linewidth=1
        )
        ax.add_patch(fascicle_circle)
        
        # Perineurium boundary
        peri_cells = max(1, int(round(0.02 / geometry.dx)))
        perineurium_thickness = peri_cells * geometry.dx
        perineurium_circle = patches.Circle(
            center, FASCICLE_RADIUS + perineurium_thickness,
            fill=False, edgecolor='darkgray', linewidth=2, linestyle='--'
        )
        ax.add_patch(perineurium_circle)
    
    # Draw electrode positions
    electrode_positions = [(0.7, 0.7), (1.3, 0.7), (0.7, 1.3), (1.3, 1.3)]
    for i, pos in enumerate(electrode_positions):
        electrode_circle = patches.Circle(
            pos, ELECTRODE_RADIUS,
            fill=True, color='red', alpha=0.8
        )
        ax.add_patch(electrode_circle)
        ax.text(pos[0], pos[1], f'E{i+1}', ha='center', va='center', 
               color='white', fontsize=8, fontweight='bold')

def plot_fibers(ax, fiber_population):
    """Plot fibers on the specified axis."""
    # Separate active and inactive fibers
    active_fibers = [f for f in fiber_population.fibers if f['active']]
    inactive_fibers = [f for f in fiber_population.fibers if not f['active']]
    
    # Plot inactive fibers as hollow black circles
    if inactive_fibers:
        inactive_positions = np.array([f['position'] for f in inactive_fibers])
        inactive_sizes = [f['diameter'] * 2 for f in inactive_fibers]
        ax.scatter(inactive_positions[:, 0], inactive_positions[:, 1], 
                  c='none', s=inactive_sizes, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Plot active fibers as filled black circles
    if active_fibers:
        active_positions = np.array([f['position'] for f in active_fibers])
        active_sizes = [f['diameter'] * 2 for f in active_fibers]
        ax.scatter(active_positions[:, 0], active_positions[:, 1], 
                  c='black', s=active_sizes, alpha=0.9, edgecolors='black', linewidth=1)

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🧠 Nerve Stimulation Simulator</h1>', 
                unsafe_allow_html=True)
    
    # Create sidebar with parameters
    params = create_parameter_sidebar()
    
    # Initialize simulator
    simulator = initialize_simulator()
    
    # Update simulation
    electrode_amplitudes = params['electrode_amplitudes']
    
    # Create visualization
    with st.spinner("Computing simulation..."):
        fig = create_visualization(simulator, electrode_amplitudes)
    
    # Create two-column layout: plot on left, info on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        # Instructions
        st.subheader("ℹ️ Instructions")
        st.markdown("""
        - **Adjust parameters** in the sidebar to change simulation settings
        - **Electrode amplitudes** control the stimulation strength and polarity
        - **Negative values** (cathodic) activate fibers, **positive values** (anodic) do not
        - **Conductivity values** control how current flows through different tissues
        - **Perineurium** acts as a resistive barrier around fascicles
        - **Fiber activation** depends on potential threshold and fiber diameter
        """)
        
        # Display key statistics
        stats = simulator.fiber_population.get_activation_stats()
        total_fibers = sum(stats[f]['total'] for f in stats)
        total_active = sum(stats[f]['active'] for f in stats)
        activation_rate = (total_active / total_fibers * 100) if total_fibers > 0 else 0
        
        st.subheader("📊 Statistics")
        
        # Show key metrics
        st.metric("Total Fibers", total_fibers)
        st.metric("Active Fibers", total_active)
        st.metric("Activation Rate", f"{activation_rate:.1f}%")
        
        # Detailed statistics
        st.subheader("📈 Detailed Stats")
        stats_data = []
        for fascicle, data in stats.items():
            stats_data.append({
                'Fascicle': fascicle,
                'Active': data['active'],
                'Total': data['total'],
                'Activation %': f"{data['percentage']:.1f}%"
            })
        
        st.dataframe(stats_data, use_container_width=True)
        
        # Potential field info
        potential_field = simulator.solver.compute_total_potential(electrode_amplitudes)
        st.subheader("⚡ Potential Field")
        st.metric("Min Potential", f"{potential_field.min():.3f} V")
        st.metric("Max Potential", f"{potential_field.max():.3f} V")
        st.metric("Range", f"{potential_field.max() - potential_field.min():.3f} V")

if __name__ == "__main__":
    main()
