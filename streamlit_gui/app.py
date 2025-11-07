#!/usr/bin/env python3
"""
Nerve Stimulation Simulator - Streamlit GUI

Interactive web interface for the nerve stimulation simulator.
"""

import streamlit as st
import numpy as np
import pandas as pd
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
        font-size: 1rem !important;
        font-weight: bold !important;
        color: #1f77b4;
        text-align: center !important;
        margin-bottom: 0rem !important;
    }
    h1.main-header {
        font-size: 3rem !important;
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
        font-size: 1.1rem;
    }
    /* Increase font sizes throughout */
    h1:not(.main-header), h2, h3, h4, h5, h6 {
        font-size: 1.5em !important;
    }
    .stMarkdown {
        font-size: 1.3rem;
    }
    .stCaption {
        font-size: 1.2rem;
        color: #000000 !important;
    }
    .stSlider label, .stSelectbox label, .stNumberInput label {
        font-size: 1.2rem;
    }
    .stSubheader {
        font-size: 1.4rem !important;
    }
    p, div, span, label {
        font-size: 1.2rem !important;
    }
    /* Increase sidebar width */
    [data-testid="stSidebar"] {
        min-width: 500px !important;
        max-width: 500px !important;
    }
</style>
""", unsafe_allow_html=True)

def initialize_simulator():
    """Initialize the simulator with current parameters."""
    if 'simulator' not in st.session_state:
        try:
            with st.spinner("Initializing simulator..."):
                st.session_state.simulator = NerveStimulationSimulator()
        except Exception as e:
            st.error(f"Failed to initialize simulator: {e}")
            st.stop()
    return st.session_state.simulator

def create_parameter_sidebar():
    """Create the parameter control sidebar."""
    st.sidebar.header("🧠 Simulation Parameters")
    
    # Electrode type selector
    st.sidebar.subheader("🔌 Electrode Configuration")
    electrode_type = st.sidebar.selectbox(
        "Electrode Type",
        ["Current (4 Discrete)", "Ring (Surrounding)"],
        help="Choose between discrete electrodes or a ring electrode surrounding the nerve"
    )
    
    # Electrode amplitudes (always visible)
    st.sidebar.subheader("⚡ Electrode Amplitudes (V)")
    
    if electrode_type == "Current (4 Discrete)":
        electrode_1 = st.sidebar.slider("Electrode 1", -2.0, 2.0, 0.0, 0.1)
        electrode_2 = st.sidebar.slider("Electrode 2", -2.0, 2.0, 2.0, 0.1)
        electrode_3 = st.sidebar.slider("Electrode 3", -2.0, 2.0, -2.0, 0.1)
        electrode_4 = st.sidebar.slider("Electrode 4", -2.0, 2.0, -1.0, 0.1)
        electrode_amplitudes = [electrode_1, electrode_2, electrode_3, electrode_4]
    else:  # Ring electrode
        ring_amplitude = st.sidebar.slider("Ring Electrode", -2.0, 2.0, -1.0, 0.1)
        electrode_amplitudes = [ring_amplitude]  # Single amplitude for ring
    
    # Advanced parameters (collapsible)
    with st.sidebar.expander("🔧 Advanced Parameters", expanded=False):
        # Grid parameters
        st.subheader("Grid Settings")
        grid_size = st.slider("Grid Resolution", 80, 200, 120, 20, 
                             help="Higher resolution = more accurate but slower")
        
        # Nerve parameters
        st.subheader("Nerve Geometry")
        nerve_radius = st.slider("Nerve Radius (mm)", 0.2, 0.6, 0.4, 0.05)
        fascicle_count = st.slider("Number of Fascicles", 2, 5, 4, 1)
        fascicle_radius = st.slider("Fascicle Radius (mm)", 0.05, 0.15, 0.1, 0.01)
        
        # Conductivity parameters
        st.subheader("Tissue Conductivity (S/m)")
        epineurium_cond = st.number_input("Epineurium", 0.0001, 1.0, 0.1, 0.01, 
                                        format="%.4f", help="Nerve sheath conductivity")
        endoneurium_cond = st.number_input("Endoneurium", 0.1, 2.0, 0.5, 0.1, 
                                         format="%.1f", help="Inside fascicles conductivity")
        perineurium_cond = st.number_input("Perineurium", 0.00001, 0.1, 0.001, 0.0001, 
                                         format="%.5f", help="Fascicle boundary conductivity")
        outside_cond = st.number_input("Outside Nerve", 0.1, 10.0, 0.2, 0.1, 
                                     format="%.1f", help="Surrounding tissue conductivity")
        
        # Electrode parameters
        st.subheader("Electrode Settings")
        electrode_radius = st.slider("Electrode Radius (mm)", 0.01, 0.05, 0.02, 0.005)
        
        # Fiber parameters
        st.subheader("Fiber Settings")
        fiber_count = st.slider("Fibers per Fascicle", 10, 100, 30, 10)
        min_spacing = st.slider("Minimum Spacing (μm)", 0.0, 50.0, 15.0, 0.01, 
                               help="Minimum spacing between fiber centers")
        threshold_base = st.number_input("Base Threshold (V)", 0.1, 5.0, 0.7, 0.1, 
                                       format="%.3f", help="Magnitude of cathodic stimulation needed for activation")
        
        # On-target fiber parameters
        st.subheader("On-Target Fibers")
        on_target_mean = st.slider("Mean Diameter (μm)", 1.0, 16.0, 6.0, 0.5, 
                                  help="Mean diameter for on-target fibers")
        on_target_std = st.slider("Std Deviation (μm)", 0.5, 5.0, 2.0, 0.1, 
                                 help="Standard deviation for on-target fibers")
        
        # Off-target fiber parameters
        st.subheader("Off-Target Fibers")
        off_target_mean = st.slider("Mean Diameter (μm)", 1.0, 16.0, 12.0, 0.5, 
                                   help="Mean diameter for off-target fibers")
        off_target_std = st.slider("Std Deviation (μm)", 0.5, 5.0, 2.0, 0.1, 
                                  help="Standard deviation for off-target fibers")
    
    return {
        'electrode_type': electrode_type,
        'grid_size': grid_size,
        'nerve_radius': nerve_radius,
        'fascicle_count': fascicle_count,
        'fascicle_radius': fascicle_radius,
        'epineurium_cond': epineurium_cond,
        'endoneurium_cond': endoneurium_cond,
        'perineurium_cond': perineurium_cond,
        'outside_cond': outside_cond,
        'electrode_radius': electrode_radius,
        'electrode_amplitudes': electrode_amplitudes,
        'fiber_count': fiber_count,
        'min_spacing': min_spacing,
        'threshold_base': threshold_base,
        'on_target_mean': on_target_mean,
        'on_target_std': on_target_std,
        'off_target_mean': off_target_mean,
        'off_target_std': off_target_std
    }

def update_simulator_parameters(simulator, params):
    """Update simulator parameters if they've changed."""
    # Check if parameters have changed and rebuild if necessary
    current_params = getattr(simulator, '_current_params', {})
    
    # Parameters that require rebuilding the simulation
    rebuild_params = [
        'electrode_type', 'grid_size', 'nerve_radius', 'fascicle_count', 'fascicle_radius',
        'epineurium_cond', 'endoneurium_cond', 'perineurium_cond', 'outside_cond',
        'electrode_radius', 'fiber_count', 'min_spacing', 'threshold_base',
        'on_target_mean', 'on_target_std', 'off_target_mean', 'off_target_std'
    ]
    
    needs_rebuild = any(
        current_params.get(param) != params.get(param) 
        for param in rebuild_params
    )
    
    if needs_rebuild:
        # Store current parameters
        simulator._current_params = params.copy()
        
        # Rebuild the simulation with new parameters
        try:
            with st.spinner("Rebuilding simulation with new parameters..."):
                # Update global parameters (this is a bit hacky but works)
                import nerve_stimulation_simulator as nss
            
            # Update the global constants
            nss.GRID_SIZE = params['grid_size']
            nss.NERVE_RADIUS = params['nerve_radius']
            nss.FASCICLE_COUNT = params['fascicle_count']
            nss.FASCICLE_RADIUS = params['fascicle_radius']
            nss.CONDUCTIVITY_EPINEURIUM = params['epineurium_cond']
            nss.CONDUCTIVITY_ENDONEURIUM = params['endoneurium_cond']
            nss.CONDUCTIVITY_PERINEURIUM = params['perineurium_cond']
            nss.CONDUCTIVITY_OUTSIDE = params['outside_cond']
            nss.ELECTRODE_RADIUS = params['electrode_radius']
            nss.FIBER_COUNT_PER_FASCICLE = params['fiber_count']
            nss.FIBER_THRESHOLD_BASE = params['threshold_base']
            
            # Rebuild the simulator
            simulator.geometry = nss.NerveGeometry(
                grid_size=params['grid_size'],
                domain_size=nss.DOMAIN_SIZE,
                nerve_center=(nss.NERVE_CENTER_X, nss.NERVE_CENTER_Y),
                nerve_radius=params['nerve_radius'],
                fascicle_count=params['fascicle_count'],
                fascicle_radius=params['fascicle_radius']
            )
            simulator.fiber_population = nss.FiberPopulation(
                fascicle_centers=simulator.geometry.fascicle_centers,
                fascicle_radius=params['fascicle_radius'],
                fiber_count_per_fascicle=params['fiber_count'],
                threshold_base=params['threshold_base'],
                threshold_scaling=nss.FIBER_THRESHOLD_SCALING,
                min_spacing=params['min_spacing'],
                on_target_mean=params['on_target_mean'],
                on_target_std=params['on_target_std'],
                off_target_mean=params['off_target_mean'],
                off_target_std=params['off_target_std']
            )
            # Get electrode positions based on type
            if params['electrode_type'] == "Current (4 Discrete)":
                electrode_positions = nss.ELECTRODE_POSITIONS
            else:  # Ring electrode
                # Create ring electrode positions around the nerve
                nerve_center = (nss.NERVE_CENTER_X, nss.NERVE_CENTER_Y)
                ring_radius = nss.NERVE_RADIUS + 0.1  # Ring slightly outside nerve
                num_ring_electrodes = 8  # Reduced number of points for better performance
                electrode_positions = []
                for i in range(num_ring_electrodes):
                    angle = 2 * np.pi * i / num_ring_electrodes
                    x = nerve_center[0] + ring_radius * np.cos(angle)
                    y = nerve_center[1] + ring_radius * np.sin(angle)
                    electrode_positions.append((x, y))
            
            simulator.solver = nss.LaplaceSolver(
                simulator.geometry, 
                electrode_positions, 
                nss.ELECTRODE_RADIUS
            )
        except Exception as e:
            st.error(f"Failed to rebuild simulation: {e}")
            st.warning("Reverting to previous parameters...")
            # Restore previous parameters
            simulator._current_params = current_params

def create_visualization(simulator, electrode_amplitudes, electrode_type="Current (4 Discrete)"):
    """Create the main visualization."""
    try:
        # Update simulation with current electrode amplitudes
        potential_field = simulator.update_simulation(electrode_amplitudes)
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return None
    
    # Create single figure focused on the nerve with higher DPI
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    
    # No title
    ax.set_aspect('equal')
    
    # Remove axis labels and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Plot intensity field with managua colormap
    X, Y = simulator.geometry.X, simulator.geometry.Y
    intensity_field = np.abs(potential_field)  # Use absolute value for intensity
    # Apply sign back to show direction
    signed_intensity = np.sign(potential_field) * intensity_field
    vmax = 1.0  # Fixed intensity bounds
    
    # Use managua colormap directly (flipped)
    cmap = plt.get_cmap('managua').reversed()
    
    contour_levels = np.linspace(-vmax, vmax, 21)
    contour_plot = ax.contourf(X, Y, signed_intensity, levels=contour_levels, 
                              cmap=cmap, alpha=0.7, extend='both', vmin=-vmax, vmax=vmax)
    
    # Add colorbar with zero tick and plus/minus labels
    cbar = plt.colorbar(contour_plot, ax=ax, shrink=0.8, pad=0.02, ticks=[-1, 0, 1])
    cbar.set_ticklabels(['−', '0', '+'])
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Field Intensity', rotation=270, labelpad=20, fontsize=12)
    
    # Draw nerve structure
    draw_nerve_structure(ax, simulator.geometry, electrode_type)
    
    # Plot fibers
    plot_fibers(ax, simulator.fiber_population)
    
    # Zoom in on the nerve area - use geometry values
    nerve_center_x, nerve_center_y = simulator.geometry.nerve_center
    nerve_radius = simulator.geometry.nerve_radius
    zoom_margin = 0.3  # Add some margin around the nerve
    
    ax.set_xlim(nerve_center_x - nerve_radius - zoom_margin, 
                nerve_center_x + nerve_radius + zoom_margin)
    ax.set_ylim(nerve_center_y - nerve_radius - zoom_margin, 
                nerve_center_y + nerve_radius + zoom_margin)
    
    # Grid removed for cleaner visualization
    
    # Add legend with separate entries for borders and fiber states
    import matplotlib.lines as mlines
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Circle
    
    legend_handles = []
    legend_labels = []
    
    # On Target: show two colored border circles (yellow and green)
    on_target_borders = (
        Circle((0, 0), 0.03, fill=False, edgecolor='#f2ef30', linewidth=3),
        Circle((0, 0), 0.03, fill=False, edgecolor='#7fe74e', linewidth=3)
    )
    legend_handles.append(on_target_borders)
    legend_labels.append('On Target')
    
    # Off Target: show two colored border circles (red and blue)
    off_target_borders = (
        Circle((0, 0), 0.03, fill=False, edgecolor='#c60000', linewidth=3),
        Circle((0, 0), 0.03, fill=False, edgecolor='#0053b2', linewidth=3)
    )
    legend_handles.append(off_target_borders)
    legend_labels.append('Off Target')
    
    # Active Fiber: black filled circle
    active_fiber = mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                                 markeredgecolor='black', markersize=8, linestyle='None')
    legend_handles.append(active_fiber)
    legend_labels.append('Active Fiber')
    
    # Inactive Fiber: black hollow circle
    inactive_fiber = mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                                   markeredgecolor='black', markersize=8, markeredgewidth=1, linestyle='None')
    legend_handles.append(inactive_fiber)
    legend_labels.append('Inactive Fiber')
    
    # Create legend with tuple handler for the border pairs
    legend = ax.legend(legend_handles, legend_labels,
                      handler_map={tuple: HandlerTuple(ndivide=None)},
                      loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig

def draw_nerve_structure(ax, geometry, electrode_type="Current (4 Discrete)"):
    """Draw nerve and fascicle boundaries."""
    # Draw nerve boundary - use geometry values to ensure consistency
    nerve_circle = patches.Circle(
        geometry.nerve_center, geometry.nerve_radius,
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(nerve_circle)
    
    # Fascicle colors: yellow, green (on-target), red, blue (off-target)
    fascicle_colors = ['#f2ef30', '#7fe74e', '#c60000', '#0053b2']
    
    # Draw fascicle boundaries with thick colored line between two thin black lines
    fascicle_radius = geometry.fascicle_radius  # Use the actual radius from geometry
    for fascicle_idx, center in enumerate(geometry.fascicle_centers):
        color = fascicle_colors[fascicle_idx % len(fascicle_colors)]
        
        # Outer thin black line
        outer_black = patches.Circle(
            center, fascicle_radius + 0.01,
            fill=False, edgecolor='black', linewidth=1
        )
        ax.add_patch(outer_black)
        
        # Middle thick colored line
        colored_line = patches.Circle(
            center, fascicle_radius,
            fill=False, edgecolor=color, linewidth=3
        )
        ax.add_patch(colored_line)
        
        # Inner thin black line
        inner_black = patches.Circle(
            center, fascicle_radius - 0.01,
            fill=False, edgecolor='black', linewidth=1
        )
        ax.add_patch(inner_black)
    
    # Draw electrode positions based on type
    if electrode_type == "Current (4 Discrete)":
        # Draw discrete electrodes (clockwise order)
        electrode_positions = [(0.7, 0.7), (1.3, 0.7), (1.3, 1.3), (0.7, 1.3)]
        for i, pos in enumerate(electrode_positions):
            electrode_circle = patches.Circle(
                pos, ELECTRODE_RADIUS,
                fill=True, color='red', alpha=0.8
            )
            ax.add_patch(electrode_circle)
            ax.text(pos[0], pos[1], f'E{i+1}', ha='center', va='center', 
                   color='white', fontsize=8, fontweight='bold')
    else:  # Ring electrode
        # Draw ring electrode around the nerve - use geometry values
        nerve_center = geometry.nerve_center
        ring_radius = geometry.nerve_radius + 0.1  # Ring slightly outside nerve
        ring_circle = patches.Circle(
            nerve_center, ring_radius,
            fill=False, edgecolor='red', linewidth=3, alpha=0.8
        )
        ax.add_patch(ring_circle)
        ax.text(nerve_center[0], nerve_center[1] + ring_radius + 0.05, 
               'Ring Electrode', ha='center', va='bottom', 
               color='red', fontsize=10, fontweight='bold')

def plot_fibers(ax, fiber_population):
    """Plot fibers on the specified axis."""
    # All fibers are black regardless of fascicle or activation status
    all_fibers = fiber_population.fibers
    active_fibers = [f for f in all_fibers if f['active']]
    inactive_fibers = [f for f in all_fibers if not f['active']]
    
    # Plot inactive fibers as hollow black circles
    if inactive_fibers:
        positions = np.array([f['position'] for f in inactive_fibers])
        sizes = [f['diameter'] * 2 for f in inactive_fibers]
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c='none', s=sizes, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Plot active fibers as filled black circles
    if active_fibers:
        positions = np.array([f['position'] for f in active_fibers])
        sizes = [f['diameter'] * 2 for f in active_fibers]
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c='black', s=sizes, alpha=0.9, edgecolors='black', linewidth=0.5)

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🧠 Nerve Stimulation Simulator</h1>', 
                unsafe_allow_html=True)

    # Create sidebar with parameters
    params = create_parameter_sidebar()

    # Initialize simulator
    simulator = initialize_simulator()

    # Update simulator parameters if they've changed
    update_simulator_parameters(simulator, params)

    # Update simulation
    electrode_amplitudes = params['electrode_amplitudes']

    # Create visualization
    with st.spinner("Computing simulation..."):
        fig = create_visualization(simulator, electrode_amplitudes, params['electrode_type'])

    if fig is None:
        st.error("Failed to create visualization. Please try adjusting parameters.")
        return

    # Create two-column layout: plot on left, info on right
    col1, col2 = st.columns([2, 1])

    with col1:
        st.pyplot(fig, use_container_width=True)

    with col2:
        # Calculate score: on-target % - off-target %
        stats = simulator.fiber_population.get_activation_stats()

        # On-target fascicles: indices 0 (yellow) and 1 (green)
        # Off-target fascicles: indices 2 (red) and 3 (blue)
        on_target_percentages = []
        off_target_percentages = []

        for fascicle_idx, data in stats.items():
            if isinstance(fascicle_idx, int):
                if fascicle_idx in [0, 1]:  # On-target (yellow, green)
                    on_target_percentages.append(data['percentage'])
                elif fascicle_idx in [2, 3]:  # Off-target (red, blue)
                    off_target_percentages.append(data['percentage'])

        # Calculate averages
        on_target_avg = sum(on_target_percentages) / len(on_target_percentages) if on_target_percentages else 0
        off_target_avg = sum(off_target_percentages) / len(off_target_percentages) if off_target_percentages else 0

        # Score = on_target % - off_target % (clamped to 0-100)
        score = max(0, min(100, on_target_avg - off_target_avg))

        # Display score progress bar
        st.markdown(f"### 📊 Score: {round(score)}")
        st.progress(score / 100.0)
        st.text(f"Off-target activation: score ⬇️")
        st.text(f"On-target activation: score ⬆️")

        # Instructions as dropdown
        with st.expander("ℹ️ Detailed Instructions", expanded=False):
            st.markdown("""
            - **Electrode Type**: Choose between discrete electrodes or ring electrode
            - **Electrode amplitudes** control the stimulation strength and polarity
            - **Negative values** (cathodic) activate fibers, **positive values** (anodic) do not
            - **Ring electrode** surrounds the nerve for uniform stimulation
            - **Discrete electrodes** allow selective stimulation patterns
            - **Conductivity values** control how current flows through different tissues
            - **Perineurium** acts as a resistive barrier around fascicles
            - **Fiber activation** depends on potential threshold and fiber diameter
            - **Larger fibers require more intensity to activate** than smaller fibers
            - **On-target fibers** (yellow/green fascicles) produce therapeutic effects
            - **Off-target fibers** (red/blue fascicles) cause side effects and are larger diameter
            """)

        # Statistics as dropdown
        with st.expander("📈 Statistics", expanded=False):
            # Display key statistics
            stats = simulator.fiber_population.get_activation_stats()
            total_fibers = sum(stats[f]['total'] for f in stats)
            total_active = sum(stats[f]['active'] for f in stats)
            activation_rate = (total_active / total_fibers * 100) if total_fibers > 0 else 0

            # Show key metrics
            st.metric("Total Fibers", total_fibers)
            st.metric("Active Fibers", total_active)
            st.metric("Activation Rate", f"{activation_rate:.1f}%")

            # Detailed statistics
            st.subheader("📈 Detailed Stats")
            # Fascicle colors: yellow, green (on-target), red, blue (off-target)
            fascicle_colors = ['#f2ef30', '#7fe74e', '#c60000', '#0053b2']
            color_names = ['Yellow', 'Green', 'Red', 'Blue']

            stats_data = []
            for fascicle_idx, data in stats.items():
                # Handle both integer and string fascicle identifiers
                if isinstance(fascicle_idx, int):
                    fascicle_num = fascicle_idx  # Already 0-based index
                    fascicle_name = f"Fascicle {fascicle_idx + 1}"
                else:
                    # Try to parse string format (e.g., "Fascicle 1" -> 1)
                    try:
                        fascicle_num = int(str(fascicle_idx).split()[-1]) - 1
                        fascicle_name = str(fascicle_idx)
                    except (ValueError, IndexError):
                        fascicle_num = 0
                        fascicle_name = str(fascicle_idx)

                # Get color based on fascicle index
                color_idx = fascicle_num % len(fascicle_colors)
                color_name = color_names[color_idx]

                stats_data.append({
                    'Fascicle': fascicle_name,
                    'Color': color_name,
                    'Active': data['active'],
                    'Total': data['total'],
                    'Activation %': f"{data['percentage']:.1f}%"
                })

            # Display dataframe with color styling
            df = pd.DataFrame(stats_data)

            # Create styled dataframe with color backgrounds
            def color_fascicle_row(row):
                try:
                    # Extract fascicle number from display name
                    fascicle_name = str(row['Fascicle'])
                    if 'Fascicle' in fascicle_name:
                        fascicle_num = int(fascicle_name.split()[-1]) - 1
                    else:
                        # Try to extract number directly
                        fascicle_num = int(fascicle_name) - 1 if fascicle_name.isdigit() else 0

                    color_idx = fascicle_num % len(fascicle_colors)
                    color_hex = fascicle_colors[color_idx]
                    # Convert hex to RGB for background color (lighter shade)
                    rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                    bg_color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)"
                    return [f'background-color: {bg_color}'] * len(row)
                except (ValueError, IndexError, KeyError, AttributeError):
                    return [''] * len(row)

            styled_df = df.style.apply(color_fascicle_row, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Potential field info
            potential_field = simulator.solver.compute_total_potential(electrode_amplitudes)
            st.subheader("⚡ Potential Field")
            st.metric("Min Potential", f"{potential_field.min():.3f} V")
            st.metric("Max Potential", f"{potential_field.max():.3f} V")
            st.metric("Range", f"{potential_field.max() - potential_field.min():.3f} V")

if __name__ == "__main__":
    main()
