#!/usr/bin/env python3
"""
Nerve Stimulation Simulator - Core Implementation

This script simulates electrical stimulation of nerve fibers using a finite-difference
Laplace solver with lead field approach for real-time visualization of activation patterns.

Physics: ∇ · (σ ∇φ) = 0 (steady-state conduction)
Activation: Only cathodic (negative) stimulation activates fibers
Method: Lead field approach for real-time electrode amplitude updates
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import time

# =============================================================================
# SIMULATION PARAMETERS - Adjust these for different scenarios
# =============================================================================

# Grid and Domain Parameters
GRID_SIZE = 100  # Grid resolution (100x100) - reduced for faster computation
DOMAIN_SIZE = 2.0  # Physical domain size in mm
DX = DOMAIN_SIZE / GRID_SIZE  # Grid spacing

# Nerve Geometry Parameters
NERVE_CENTER_X = 1.0  # Nerve center X position (mm)
NERVE_CENTER_Y = 1.0  # Nerve center Y position (mm)
NERVE_RADIUS = 0.4  # Nerve radius (mm)
FASCICLE_COUNT = 3  # Number of fascicles
FASCICLE_RADIUS = 0.08  # Radius of each fascicle (mm)

# Conductivity Parameters (S/m)
CONDUCTIVITY_EPINEURIUM = 0.01  # Low conductivity - nerve sheath
CONDUCTIVITY_ENDONEURIUM = 0.5  # High conductivity - inside fascicles
CONDUCTIVITY_PERINEURIUM = 0.001  # Very low conductivity - fascicle boundaries
CONDUCTIVITY_OUTSIDE = 1e6  # Perfect conductor - ground boundary

# Electrode Parameters
ELECTRODE_COUNT = 4  # Number of electrodes
ELECTRODE_RADIUS = 0.02  # Electrode radius (mm)
ELECTRODE_POSITIONS = [  # Electrode positions (x, y) in mm
    (0.6, 0.6),   # Electrode 1
    (1.4, 0.6),   # Electrode 2
    (0.6, 1.4),   # Electrode 3
    (1.4, 1.4),   # Electrode 4
]
ELECTRODE_AMPLITUDES = [1.0, -1.0, -1.0, 0.0]  # Current amplitudes (mA)

# Fiber Parameters
FIBER_COUNT_PER_FASCICLE = 30  # Number of fibers per fascicle - reduced for faster computation
FIBER_DIAMETER_RANGE = (10, 10.0)  # Fiber diameter range (μm)
FIBER_THRESHOLD_BASE = 0.1  # Base activation threshold (V)
FIBER_THRESHOLD_SCALING = 8.0  # Diameter scaling factor for threshold

# Simulation Parameters
TOLERANCE = 1e-6  # Convergence tolerance for Laplace solver
MAX_ITERATIONS = 1000  # Maximum iterations for solver
VISUALIZATION_SCALE = 1e3  # Scale factor for potential visualization

# Animation Parameters
ANIMATION_INTERVAL = 100  # Animation update interval (ms)
FRAME_COUNT = 100  # Number of animation frames

# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

class NerveGeometry:
    """Handles nerve geometry and conductivity field generation."""
    
    def __init__(self, grid_size, domain_size, nerve_center, nerve_radius, 
                 fascicle_count, fascicle_radius):
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size
        self.nerve_center = nerve_center
        self.nerve_radius = nerve_radius
        self.fascicle_count = fascicle_count
        self.fascicle_radius = fascicle_radius
        
        # Create coordinate grids
        x = np.linspace(0, domain_size, grid_size)
        y = np.linspace(0, domain_size, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initialize conductivity field
        self.sigma = np.ones((grid_size, grid_size)) * CONDUCTIVITY_OUTSIDE
        
        # Generate fascicle positions
        self.fascicle_centers = self._generate_fascicle_positions()
        
        # Build conductivity field
        self._build_conductivity_field()
        
    def _generate_fascicle_positions(self):
        """Generate fascicle center positions within the nerve."""
        centers = []
        angle_step = 2 * np.pi / self.fascicle_count
        fascicle_distance = self.nerve_radius * 0.6  # Position fascicles inside nerve
        
        for i in range(self.fascicle_count):
            angle = i * angle_step
            x = self.nerve_center[0] + fascicle_distance * np.cos(angle)
            y = self.nerve_center[1] + fascicle_distance * np.sin(angle)
            centers.append((x, y))
        
        return centers
    
    def _build_conductivity_field(self):
        """Build the multi-layer conductivity field."""
        # Start with outside nerve (perfect conductor)
        self.sigma.fill(CONDUCTIVITY_OUTSIDE)
        
        # Add nerve (epineurium)
        nerve_mask = self._get_circle_mask(self.nerve_center, self.nerve_radius)
        self.sigma[nerve_mask] = CONDUCTIVITY_EPINEURIUM
        
        # Add fascicles (endoneurium)
        for center in self.fascicle_centers:
            fascicle_mask = self._get_circle_mask(center, self.fascicle_radius)
            self.sigma[fascicle_mask] = CONDUCTIVITY_ENDONEURIUM
            
            # Add perineurium (thin boundary around fascicle)
            perineurium_mask = self._get_circle_mask(center, self.fascicle_radius + 0.01) & \
                             ~self._get_circle_mask(center, self.fascicle_radius)
            self.sigma[perineurium_mask] = CONDUCTIVITY_PERINEURIUM
    
    def _get_circle_mask(self, center, radius):
        """Get boolean mask for points inside a circle."""
        distance = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2)
        return distance <= radius
    
    def get_conductivity(self):
        """Return the conductivity field."""
        return self.sigma

class FiberPopulation:
    """Handles nerve fiber generation and activation."""
    
    def __init__(self, fascicle_centers, fascicle_radius, fiber_count_per_fascicle,
                 diameter_range, threshold_base, threshold_scaling):
        self.fascicle_centers = fascicle_centers
        self.fascicle_radius = fascicle_radius
        self.fiber_count_per_fascicle = fiber_count_per_fascicle
        self.diameter_range = diameter_range
        self.threshold_base = threshold_base
        self.threshold_scaling = threshold_scaling
        
        # Generate fibers
        self.fibers = self._generate_fibers()
        
    def _generate_fibers(self):
        """Generate fiber population within fascicles."""
        fibers = []
        
        for fascicle_idx, center in enumerate(self.fascicle_centers):
            for _ in range(self.fiber_count_per_fascicle):
                # Random position within fascicle
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, self.fascicle_radius * 0.8)
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                
                # Random diameter
                diameter = np.random.uniform(*self.diameter_range)
                
                # Calculate threshold based on diameter
                threshold = self.threshold_base * (self.threshold_scaling / diameter)
                
                fiber = {
                    'position': (x, y),
                    'diameter': diameter,
                    'threshold': threshold,
                    'fascicle': fascicle_idx,
                    'active': False
                }
                fibers.append(fiber)
        
        return fibers
    
    def update_activation(self, potential_field, grid_size, domain_size):
        """Update fiber activation based on potential field."""
        dx = domain_size / grid_size
        
        for fiber in self.fibers:
            x, y = fiber['position']
            
            # Convert to grid coordinates
            grid_x = int(x / dx)
            grid_y = int(y / dx)
            
            # Clamp to grid bounds
            grid_x = max(0, min(grid_size - 1, grid_x))
            grid_y = max(0, min(grid_size - 1, grid_y))
            
            # Get potential at fiber location
            potential = potential_field[grid_y, grid_x]
            
            # Activation rule: cathodic (negative) stimulation only
            fiber['active'] = potential <= -fiber['threshold']
    
    def get_activation_stats(self):
        """Get activation statistics by fascicle."""
        stats = {}
        for fascicle_idx in range(len(self.fascicle_centers)):
            fascicle_fibers = [f for f in self.fibers if f['fascicle'] == fascicle_idx]
            active_count = sum(1 for f in fascicle_fibers if f['active'])
            total_count = len(fascicle_fibers)
            stats[fascicle_idx] = {
                'active': active_count,
                'total': total_count,
                'percentage': (active_count / total_count * 100) if total_count > 0 else 0
            }
        return stats

class LaplaceSolver:
    """Fast sparse matrix Laplace solver with lead field approach."""
    
    def __init__(self, geometry, electrode_positions, electrode_radius):
        self.geometry = geometry
        self.electrode_positions = electrode_positions
        self.electrode_radius = electrode_radius
        self.grid_size = geometry.grid_size
        self.dx = geometry.dx
        
        # Build the sparse matrix system once
        print("Building sparse matrix system...")
        self.A, self.b_indices = self._build_sparse_system()
        
        # Precompute lead fields for each electrode
        self.lead_fields = self._compute_lead_fields()
        
    def _build_sparse_system(self):
        """Build the sparse matrix system for the Laplace equation."""
        n = self.grid_size
        sigma = self.geometry.get_conductivity()
        
        # Precompute electrode masks for efficiency
        electrode_masks = []
        for pos in self.electrode_positions:
            mask = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(n):
                    if np.sqrt((i*self.dx - pos[0])**2 + (j*self.dx - pos[1])**2) <= ELECTRODE_RADIUS:
                        mask[i, j] = True
            electrode_masks.append(mask)
        
        # Create mapping from 2D grid to 1D vector
        def grid_to_vector(i, j):
            return i * n + j
        
        # Build sparse matrix
        row_indices = []
        col_indices = []
        data = []
        b_indices = []
        
        total_points = n * n
        processed = 0
        
        # Process all grid points (including boundaries)
        for i in range(n):
            for j in range(n):
                idx = grid_to_vector(i, j)
                
                # Check if this is an electrode point
                is_electrode = any(mask[i, j] for mask in electrode_masks)
                
                if is_electrode:
                    # Electrode point: phi = amplitude (will be set in RHS)
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    b_indices.append(idx)
                elif i == 0 or i == n-1 or j == 0 or j == n-1:
                    # Boundary point: phi = 0 (ground boundary)
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    b_indices.append(idx)
                else:
                    # Interior point: finite difference equation
                    # σ∇²φ = 0 becomes: σ(φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1] - 4φ[i,j]) = 0
                    
                    # Center coefficient
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(-4.0 * sigma[i, j])
                    
                    # Neighbor coefficients
                    neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                    for ni, nj in neighbors:
                        if 0 <= ni < n and 0 <= nj < n:
                            row_indices.append(idx)
                            col_indices.append(grid_to_vector(ni, nj))
                            data.append(sigma[i, j])
                
                processed += 1
                if processed % (total_points // 10) == 0:
                    print(f"  Building matrix: {100*processed//total_points}% complete")
        
        print("  Creating sparse matrix...")
        # Create sparse matrix
        A = csc_matrix((data, (row_indices, col_indices)), 
                      shape=(n*n, n*n))
        
        return A, b_indices
    
    def _compute_lead_fields(self):
        """Precompute unit potential fields for each electrode (lead field method)."""
        lead_fields = []
        
        for i, pos in enumerate(self.electrode_positions):
            print(f"Computing lead field for electrode {i+1}...")
            start_time = time.time()
            lead_field = self._solve_laplace_single_electrode(pos, 1.0)
            elapsed = time.time() - start_time
            print(f"  Lead field {i+1} computed in {elapsed:.2f} seconds")
            lead_fields.append(lead_field)
        
        return lead_fields
    
    def _solve_laplace_single_electrode(self, electrode_pos, amplitude):
        """Solve Laplace equation for a single electrode with unit amplitude."""
        n = self.grid_size
        
        # Create RHS vector
        b = np.zeros(n * n)
        
        # Set electrode boundary condition efficiently
        electrode_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if np.sqrt((i*self.dx - electrode_pos[0])**2 + (j*self.dx - electrode_pos[1])**2) <= ELECTRODE_RADIUS:
                    electrode_mask[i, j] = True
        
        # Set boundary conditions
        electrode_indices = np.where(electrode_mask.flatten())[0]
        b[electrode_indices] = amplitude
        
        # Set ground boundary conditions (phi = 0 at boundaries)
        for i in range(n):
            for j in range(n):
                if i == 0 or i == n-1 or j == 0 or j == n-1:
                    if not electrode_mask[i, j]:  # Don't override electrode conditions
                        idx = i * n + j
                        b[idx] = 0.0
        
        # Solve sparse system
        phi_vector = spsolve(self.A, b)
        
        # Reshape to 2D grid
        phi = phi_vector.reshape((n, n))
        
        return phi
    
    def _get_electrode_mask(self, electrode_pos):
        """Get boolean mask for electrode region."""
        distance = np.sqrt((self.geometry.X - electrode_pos[0])**2 + 
                          (self.geometry.Y - electrode_pos[1])**2)
        return distance <= ELECTRODE_RADIUS
    
    def compute_total_potential(self, electrode_amplitudes):
        """Compute total potential field using lead field method."""
        total_potential = np.zeros((self.grid_size, self.grid_size))
        
        for i, amplitude in enumerate(electrode_amplitudes):
            total_potential += amplitude * self.lead_fields[i]
        
        return total_potential

class NerveStimulationSimulator:
    """Main simulator class that coordinates all components."""
    
    def __init__(self):
        print("Initializing Nerve Stimulation Simulator...")
        
        # Initialize geometry
        print("  Creating geometry...")
        self.geometry = NerveGeometry(
            GRID_SIZE, DOMAIN_SIZE, 
            (NERVE_CENTER_X, NERVE_CENTER_Y), NERVE_RADIUS,
            FASCICLE_COUNT, FASCICLE_RADIUS
        )
        print("  Geometry created.")
        
        # Initialize fiber population
        print("  Creating fiber population...")
        self.fiber_population = FiberPopulation(
            self.geometry.fascicle_centers, FASCICLE_RADIUS,
            FIBER_COUNT_PER_FASCICLE, FIBER_DIAMETER_RANGE,
            FIBER_THRESHOLD_BASE, FIBER_THRESHOLD_SCALING
        )
        print("  Fiber population created.")
        
        # Initialize Laplace solver
        print("  Creating Laplace solver...")
        self.solver = LaplaceSolver(
            self.geometry, ELECTRODE_POSITIONS, ELECTRODE_RADIUS
        )
        print("  Laplace solver created.")
        
        # Initialize visualization
        print("  Setting up visualization...")
        self.setup_visualization()
        print("  Visualization setup complete.")
        
        print("Simulator initialized successfully!")
    
    def setup_visualization(self):
        """Setup matplotlib visualization."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Potential field and nerve structure
        self.ax1.set_title('Nerve Cross-Section with Potential Field')
        self.ax1.set_xlabel('X (mm)')
        self.ax1.set_ylabel('Y (mm)')
        self.ax1.set_aspect('equal')
        
        # Right plot: Activation statistics
        self.ax2.set_title('Activation Statistics')
        self.ax2.set_xlabel('Fascicle')
        self.ax2.set_ylabel('Activation Percentage (%)')
        
        # Draw nerve structure
        self._draw_nerve_structure()
        
        plt.tight_layout()
    
    def _draw_nerve_structure(self):
        """Draw nerve and fascicle boundaries."""
        # Draw nerve boundary
        nerve_circle = patches.Circle(
            (NERVE_CENTER_X, NERVE_CENTER_Y), NERVE_RADIUS,
            fill=False, edgecolor='black', linewidth=2
        )
        self.ax1.add_patch(nerve_circle)
        
        # Draw fascicle boundaries
        for center in self.geometry.fascicle_centers:
            fascicle_circle = patches.Circle(
                center, FASCICLE_RADIUS,
                fill=False, edgecolor='gray', linewidth=1
            )
            self.ax1.add_patch(fascicle_circle)
        
        # Draw electrode positions
        for i, pos in enumerate(ELECTRODE_POSITIONS):
            electrode_circle = patches.Circle(
                pos, ELECTRODE_RADIUS,
                fill=True, color='red', alpha=0.8
            )
            self.ax1.add_patch(electrode_circle)
            self.ax1.text(pos[0], pos[1], f'E{i+1}', ha='center', va='center', 
                         color='white', fontsize=8, fontweight='bold')
    
    def update_simulation(self, electrode_amplitudes):
        """Update simulation with new electrode amplitudes."""
        # Compute total potential field
        potential_field = self.solver.compute_total_potential(electrode_amplitudes)
        
        # Update fiber activation
        self.fiber_population.update_activation(
            potential_field, GRID_SIZE, DOMAIN_SIZE
        )
        
        # Update visualization
        self._update_visualization(potential_field)
        
        return potential_field
    
    def _update_visualization(self, potential_field):
        """Update visualization with current state."""
        # Clear the left plot
        self.ax1.clear()
        
        # Redraw nerve structure
        self._draw_nerve_structure()
        
        # Plot potential field with contour lines
        X, Y = self.geometry.X, self.geometry.Y
        
        # Create contour plot of potential field
        contour_levels = np.linspace(np.min(potential_field), np.max(potential_field), 20)
        contour_plot = self.ax1.contourf(X, Y, potential_field, levels=contour_levels, 
                                        cmap='RdBu_r', alpha=0.7, extend='both')
        
        # Add contour lines for better visualization
        contour_lines = self.ax1.contour(X, Y, potential_field, levels=10, 
                                        colors='black', alpha=0.3, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour_plot, ax=self.ax1, shrink=0.8)
        cbar.set_label('Potential (V)', rotation=270, labelpad=20)
        
        # Update fiber positions and colors
        fiber_positions = np.array([f['position'] for f in self.fiber_population.fibers])
        fiber_colors = ['red' if f['active'] else 'black' for f in self.fiber_population.fibers]
        fiber_sizes = [f['diameter'] * 2 for f in self.fiber_population.fibers]  # Size by diameter
        
        self.ax1.scatter(fiber_positions[:, 0], fiber_positions[:, 1], 
                        c=fiber_colors, s=fiber_sizes, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Set plot properties
        self.ax1.set_title('Nerve Cross-Section with Potential Field')
        self.ax1.set_xlabel('X (mm)')
        self.ax1.set_ylabel('Y (mm)')
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        
        # Update statistics
        stats = self.fiber_population.get_activation_stats()
        fascicles = list(stats.keys())
        percentages = [stats[f]['percentage'] for f in fascicles]
        
        self.ax2.clear()
        bars = self.ax2.bar(fascicles, percentages, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
        self.ax2.set_title('Activation Statistics')
        self.ax2.set_xlabel('Fascicle')
        self.ax2.set_ylabel('Activation Percentage (%)')
        self.ax2.set_ylim(0, 100)
        self.ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add total statistics
        total_fibers = sum(stats[f]['total'] for f in fascicles)
        total_active = sum(stats[f]['active'] for f in fascicles)
        total_percentage = (total_active / total_fibers * 100) if total_fibers > 0 else 0
        
        self.ax2.text(0.02, 0.98, f'Total: {total_active}/{total_fibers} fibers ({total_percentage:.1f}%)',
                     transform=self.ax2.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.draw()
    
    def run_simulation(self):
        """Run the main simulation."""
        print("Running simulation...")
        
        # Initial simulation
        potential_field = self.update_simulation(ELECTRODE_AMPLITUDES)
        
        # Print initial statistics
        stats = self.fiber_population.get_activation_stats()
        print("\nInitial Activation Statistics:")
        for fascicle, data in stats.items():
            print(f"Fascicle {fascicle}: {data['active']}/{data['total']} fibers active ({data['percentage']:.1f}%)")
        
        # Show plot
        plt.show()
        
        return potential_field

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the simulation."""
    print("=" * 60)
    print("🧠 Nerve Stimulation Simulator")
    print("=" * 60)
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Domain size: {DOMAIN_SIZE} mm")
    print(f"Nerve radius: {NERVE_RADIUS} mm")
    print(f"Fascicles: {FASCICLE_COUNT}")
    print(f"Electrodes: {ELECTRODE_COUNT}")
    print(f"Fibers per fascicle: {FIBER_COUNT_PER_FASCICLE}")
    print("=" * 60)
    
    try:
        # Create and run simulator
        print("Creating simulator...")
        simulator = NerveStimulationSimulator()
        print("Running simulation...")
        potential_field = simulator.run_simulation()
        
        print("\nSimulation completed!")
        print(f"Potential range: {np.min(potential_field):.3f} to {np.max(potential_field):.3f} V")
        
        # Save plot instead of showing
        plt.savefig('nerve_simulation_result.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'nerve_simulation_result.png'")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
