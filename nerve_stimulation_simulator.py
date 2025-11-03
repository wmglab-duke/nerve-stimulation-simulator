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
GRID_SIZE = 120  # Grid resolution (120x120) - balanced speed vs accuracy
DOMAIN_SIZE = 2.0  # Physical domain size in mm
DX = DOMAIN_SIZE / GRID_SIZE  # Grid spacing

# Nerve Geometry Parameters
NERVE_CENTER_X = 1.0  # Nerve center X position (mm)
NERVE_CENTER_Y = 1.0  # Nerve center Y position (mm)
NERVE_RADIUS = 0.4  # Nerve radius (mm)
FASCICLE_COUNT = 3  # Number of fascicles
FASCICLE_RADIUS = 0.08  # Radius of each fascicle (mm)

# Conductivity Parameters (S/m)
CONDUCTIVITY_EPINEURIUM = 1/6  # Low conductivity - nerve sheath
CONDUCTIVITY_ENDONEURIUM = 1/6  # High conductivity - inside fascicles
CONDUCTIVITY_PERINEURIUM = 0.0008  # Very low conductivity - fascicle boundaries
CONDUCTIVITY_OUTSIDE = 0.1  # Lower conductivity - surrounding tissue

# Electrode Parameters
ELECTRODE_COUNT = 4  # Number of electrodes
ELECTRODE_RADIUS = 0.02  # Electrode radius (mm)
ELECTRODE_POSITIONS = [  # Electrode positions (x, y) in mm - clockwise order
    (0.7, 0.7),   # Electrode 1 (bottom-left)
    (1.3, 0.7),   # Electrode 2 (bottom-right)
    (1.3, 1.3),   # Electrode 3 (top-right)
    (0.7, 1.3),   # Electrode 4 (top-left)
]
ELECTRODE_AMPLITUDES = [0, 2, -2, -1]  # Current amplitudes (mA)

# Ring Electrode Parameters
RING_ELECTRODE_COUNT = 8  # Number of points around the ring (reduced for performance)
RING_ELECTRODE_RADIUS_OFFSET = 0.1  # Distance from nerve edge (mm)

def generate_ring_electrode_positions(nerve_center, nerve_radius, count=RING_ELECTRODE_COUNT, offset=RING_ELECTRODE_RADIUS_OFFSET):
    """Generate positions for ring electrode around the nerve."""
    positions = []
    ring_radius = nerve_radius + offset
    
    for i in range(count):
        angle = 2 * np.pi * i / count
        x = nerve_center[0] + ring_radius * np.cos(angle)
        y = nerve_center[1] + ring_radius * np.sin(angle)
        positions.append((x, y))
    
    return positions

# Fiber Parameters
FIBER_COUNT_PER_FASCICLE = 30  # Number of fibers per fascicle - reduced for faster computation
FIBER_DIAMETER_RANGE = (10, 10.0)  # Fiber diameter range (μm)
FIBER_THRESHOLD_BASE = 0.7  # Base activation threshold (V) - magnitude of cathodic stimulation needed
FIBER_THRESHOLD_SCALING = 4.0  # Diameter scaling factor for threshold

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
        fascicle_distance = self.nerve_radius * 0.5  # Position fascicles at moderate distance from center
        
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
        
        # Add fascicles and perineurium
        for center in self.fascicle_centers:
            # First add perineurium (outer ring) - grid-aware thickness
            peri_cells = max(1, int(round(0.02 / self.dx)))  # At least 1 cell thick
            perineurium_thickness = peri_cells * self.dx
            perineurium_mask = self._get_circle_mask(center, self.fascicle_radius + perineurium_thickness) & \
                             ~self._get_circle_mask(center, self.fascicle_radius)
            self.sigma[perineurium_mask] = CONDUCTIVITY_PERINEURIUM
            
            # Then add fascicle (endoneurium) - this will overwrite perineurium inside fascicle
            fascicle_mask = self._get_circle_mask(center, self.fascicle_radius)
            self.sigma[fascicle_mask] = CONDUCTIVITY_ENDONEURIUM
    
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
                 threshold_base, threshold_scaling, min_spacing=0.1, 
                 on_target_mean=12.0, on_target_std=2.0, off_target_mean=6.0, off_target_std=2.0):
        self.fascicle_centers = fascicle_centers
        self.fascicle_radius = fascicle_radius
        self.fiber_count_per_fascicle = fiber_count_per_fascicle
        self.threshold_base = threshold_base
        self.threshold_scaling = threshold_scaling
        self.min_spacing = min_spacing
        self.on_target_mean = on_target_mean
        self.on_target_std = on_target_std
        self.off_target_mean = off_target_mean
        self.off_target_std = off_target_std
        
        # Generate fibers
        self.fibers = self._generate_fibers()
        
    def _generate_fibers(self):
        """Generate fiber population within fascicles with proper overlap detection."""
        fibers = []
        
        for fascicle_idx, center in enumerate(self.fascicle_centers):
            fascicle_fibers = []
            max_attempts = self.fiber_count_per_fascicle * 200  # Even more attempts
            attempts = 0
            
            # Designate fascicles as on-target or off-target (adjacent pairs)
            # Fascicles 0,1 are on-target, fascicles 2,3 are off-target, etc.
            is_on_target = (fascicle_idx // 2) % 2 == 0
            
            while len(fascicle_fibers) < self.fiber_count_per_fascicle and attempts < max_attempts:
                attempts += 1
                
                # Random position within fascicle
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, self.fascicle_radius * 0.8)
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                
                # Generate diameter based on fascicle type
                if is_on_target:
                    diameter = np.random.normal(self.on_target_mean, self.on_target_std)
                else:
                    diameter = np.random.normal(self.off_target_mean, self.off_target_std)
                
                # Bound diameter between 1 and 16 μm
                diameter = np.clip(diameter, 1.0, 16.0)
                
                # Check for overlap with existing fibers using actual circle sizes
                overlap = False
                for existing_fiber in fascicle_fibers:
                    existing_pos = existing_fiber['position']
                    existing_diameter = existing_fiber['diameter']
                    
                    # Calculate distance between circle centers
                    distance = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                    
                    # Circles overlap if distance < sum of radii + minimum spacing
                    # Convert diameters from μm to mm and min_spacing from μm to mm
                    diameter_mm = diameter / 1000  # Convert μm to mm
                    existing_diameter_mm = existing_diameter / 1000  # Convert μm to mm
                    min_spacing_mm = self.min_spacing / 1000  # Convert μm to mm
                    required_distance = (diameter_mm + existing_diameter_mm) / 2 + min_spacing_mm
                    
                    if distance < required_distance:
                        overlap = True
                        break
                
                # If no overlap, add the fiber
                if not overlap:
                    # Calculate threshold based on diameter
                    threshold = self.threshold_base * (self.threshold_scaling / diameter)
                    
                    fiber = {
                        'position': (x, y),
                        'diameter': diameter,
                        'threshold': threshold,
                        'fascicle': fascicle_idx,
                        'is_on_target': is_on_target,
                        'active': False
                    }
                    fascicle_fibers.append(fiber)
            
            # Add all successfully placed fibers for this fascicle
            fibers.extend(fascicle_fibers)
        
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
            # Activation occurs when potential is more negative than the threshold magnitude
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
        
        # Build the sparse matrix system
        print("Building sparse matrix system...")
        self.A, self.b_indices = self._build_sparse_system()
        
        # Precompute lead fields for each electrode
        self.lead_fields = self._compute_lead_fields()
    
        
    def _build_sparse_system(self):
        """Build the sparse matrix system for the Laplace equation using correct variable-σ Laplacian."""
        sigma = self.geometry.get_conductivity()
        ny, nx = sigma.shape
        N = ny * nx
        dx2 = self.dx * self.dx

        print("  Computing harmonic mean conductivities...")
        # Harmonic means on faces
        # East/West faces (between j and j+1)
        sigE = (2 * sigma[:, 1:] * sigma[:, :-1]) / (sigma[:, 1:] + sigma[:, :-1] + 1e-12)
        sigW = sigE  # same array, used with one-column shift

        # North/South faces (between i and i+1)
        sigS = (2 * sigma[1:, :] * sigma[:-1, :]) / (sigma[1:, :] + sigma[:-1, :] + 1e-12)
        sigN = sigS  # same array, used with one-row shift

        print("  Building matrix structure...")
        # Precompute all indices and data for vectorized assembly
        # Main diagonal
        main = np.zeros_like(sigma, dtype=float)
        main[:, :-1] += sigE / dx2     # to east
        main[:, 1:]  += sigW / dx2     # to west
        main[:-1, :] += sigS / dx2     # to south
        main[1:, :]  += sigN / dx2     # to north
        main = main.ravel()

        # Create all row/col indices and data at once
        rows = []
        cols = []
        data = []
        
        # Main diagonal
        for k in range(N):
            rows.append(k)
            cols.append(k)
            data.append(main[k])
        
        # Off-diagonals - vectorized approach
        # East connections
        for i in range(ny):
            for j in range(nx - 1):
                k = i * nx + j
                ke = i * nx + (j + 1)
                rows.append(k)
                cols.append(ke)
                data.append(-sigE[i, j] / dx2)
        
        # West connections
        for i in range(ny):
            for j in range(1, nx):
                k = i * nx + j
                kw = i * nx + (j - 1)
                rows.append(k)
                cols.append(kw)
                data.append(-sigW[i, j-1] / dx2)
        
        # South connections
        for i in range(ny - 1):
            for j in range(nx):
                k = i * nx + j
                ks = (i + 1) * nx + j
                rows.append(k)
                cols.append(ks)
                data.append(-sigS[i, j] / dx2)
        
        # North connections
        for i in range(1, ny):
            for j in range(nx):
                k = i * nx + j
                kn = (i - 1) * nx + j
                rows.append(k)
                cols.append(kn)
                data.append(-sigN[i-1, j] / dx2)

        print("  Creating sparse matrix...")
        A = csc_matrix((data, (rows, cols)), shape=(N, N))

        print("  Enforcing Dirichlet boundary conditions...")
        # Create Dirichlet mask (outer boundary + electrodes)
        dirichlet_mask = np.zeros((ny, nx), dtype=bool)
        
        # Outer boundaries (ground = 0 V)
        dirichlet_mask[0, :] = True
        dirichlet_mask[-1, :] = True
        dirichlet_mask[:, 0] = True
        dirichlet_mask[:, -1] = True
        
        # All electrode pixels are Dirichlet too (we'll set their value in b per solve)
        X, Y = self.geometry.X, self.geometry.Y
        for pos in self.electrode_positions:
            electrode_mask = (X - pos[0])**2 + (Y - pos[1])**2 <= ELECTRODE_RADIUS**2
            dirichlet_mask |= electrode_mask

        dir_idx = np.where(dirichlet_mask.ravel())[0]

        # Enforce Dirichlet by setting rows to identity; don't touch columns
        A = A.tolil()
        for k in dir_idx:
            A.rows[k] = [k]
            A.data[k] = [1.0]
        A = A.tocsc()
        
        # Store for debugging
        self._dirichlet_indices = dir_idx
        
        return A, dir_idx
    
    def _compute_lead_fields(self):
        """Precompute unit potential fields for each electrode (lead field method)."""
        lead_fields = []
        
        # Debug: Print Dirichlet and electrode info
        print(f"Dirichlet count (rows forced to identity): {len(self._dirichlet_indices)}")
        for i, pos in enumerate(self.electrode_positions):
            mask = (self.geometry.X - pos[0])**2 + (self.geometry.Y - pos[1])**2 <= ELECTRODE_RADIUS**2
            print(f"Electrode {i+1}: pixels={mask.sum()}")
        
        for i, pos in enumerate(self.electrode_positions):
            print(f"Computing lead field for electrode {i+1}...")
            start_time = time.time()
            lead_field = self._solve_laplace_single_electrode(pos, 1.0)
            elapsed = time.time() - start_time
            print(f"  Lead field {i+1} computed in {elapsed:.2f} seconds")
            print(f"  Lead field {i+1} min/max: {lead_field.min():.6f} / {lead_field.max():.6f}")
            lead_fields.append(lead_field)
        
        return lead_fields
    
    def _solve_laplace_single_electrode(self, electrode_pos, amplitude):
        """Solve Laplace equation for a single electrode with unit amplitude."""
        # Create RHS vector (all zeros by default)
        b = np.zeros(self.A.shape[0])
        
        # Set electrode boundary condition efficiently (vectorized)
        X, Y = self.geometry.X, self.geometry.Y
        electrode_mask = (X - electrode_pos[0])**2 + (Y - electrode_pos[1])**2 <= ELECTRODE_RADIUS**2
        elec_idx = np.where(electrode_mask.flatten())[0]
        
        # Debug safety: make sure we have some nodes!
        assert elec_idx.size > 0, f"No grid nodes fell inside electrode at {electrode_pos}. " \
                                  f"Increase ELECTRODE_RADIUS or grid resolution."
        
        # Set electrode potential (ground nodes remain 0.0)
        b[elec_idx] = amplitude
        
        # Debug: Check RHS
        print(f"  RHS non-zero entries: {np.count_nonzero(b)}")
        print(f"  RHS range: {b.min():.6f} to {b.max():.6f}")
        
        # Solve sparse system
        phi_vector = spsolve(self.A, b)
        
        # Debug: Check solution
        print(f"  Solution range: {phi_vector.min():.6f} to {phi_vector.max():.6f}")
        print(f"  Solution non-zero entries: {np.count_nonzero(phi_vector)}")
        
        # Reshape to 2D grid
        phi = phi_vector.reshape(self.geometry.get_conductivity().shape)
        
        return phi
    
    def _get_electrode_mask(self, electrode_pos):
        """Get boolean mask for electrode region."""
        distance = np.sqrt((self.geometry.X - electrode_pos[0])**2 + 
                          (self.geometry.Y - electrode_pos[1])**2)
        return distance <= ELECTRODE_RADIUS
    
    def compute_total_potential(self, electrode_amplitudes):
        """Compute total potential field using lead field method."""
        total_potential = np.zeros((self.grid_size, self.grid_size))
        
        # Handle both discrete and ring electrode configurations
        if len(electrode_amplitudes) == 1 and len(self.lead_fields) > 1:
            # Ring electrode: single amplitude applied to all ring electrodes
            amplitude = electrode_amplitudes[0]
            for lead_field in self.lead_fields:
                total_potential += amplitude * lead_field
        else:
            # Discrete electrodes: each electrode has its own amplitude
            for i, amplitude in enumerate(electrode_amplitudes):
                if i < len(self.lead_fields):
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
            fascicle_centers=self.geometry.fascicle_centers,
            fascicle_radius=FASCICLE_RADIUS,
            fiber_count_per_fascicle=FIBER_COUNT_PER_FASCICLE,
            threshold_base=FIBER_THRESHOLD_BASE,
            threshold_scaling=FIBER_THRESHOLD_SCALING,
            min_spacing=0.1,  # Default spacing
            on_target_mean=6.0,   # Default on-target mean (smaller)
            on_target_std=2.0,    # Default on-target std
            off_target_mean=12.0, # Default off-target mean (larger)
            off_target_std=2.0    # Default off-target std
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
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top left: Intensity field and nerve structure
        self.ax1.set_title('Intensity Field with Fiber Activation')
        self.ax1.set_aspect('equal')
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax1.set_xlabel('')
        self.ax1.set_ylabel('')
        
        # Top right: Conductivity field
        self.ax2.set_title('Conductivity Field')
        self.ax2.set_aspect('equal')
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax2.set_xlabel('')
        self.ax2.set_ylabel('')
        
        # Bottom left: Activation statistics
        self.ax3.set_title('Activation Statistics by Fascicle')
        self.ax3.set_xlabel('Fascicle')
        self.ax3.set_ylabel('Activation Percentage (%)')
        
        # Bottom right: Potential field cross-section
        self.ax4.set_title('Potential Field Cross-Section')
        self.ax4.set_xlabel('Distance (mm)')
        self.ax4.set_ylabel('Potential (V)')
        
        # Draw nerve structure on both top plots
        self._draw_nerve_structure("discrete")
        self._draw_nerve_structure_conductivity("discrete")
        
        plt.tight_layout()
    
    def _draw_nerve_structure(self, electrode_type="discrete"):
        """Draw nerve and fascicle boundaries."""
        # Draw nerve boundary
        nerve_circle = patches.Circle(
            (NERVE_CENTER_X, NERVE_CENTER_Y), NERVE_RADIUS,
            fill=False, edgecolor='black', linewidth=2
        )
        self.ax1.add_patch(nerve_circle)
        
        # Draw fascicle boundaries with thick colored line between two thin black lines
        fascicle_colors = ['#0053b2', '#7fe74e', '#c60000', '#f2ef30']  # blue, green, red, yellow
        
        for fascicle_idx, center in enumerate(self.geometry.fascicle_centers):
            color = fascicle_colors[fascicle_idx % len(fascicle_colors)]
            
            # Outer thin black line
            outer_black = patches.Circle(
                center, FASCICLE_RADIUS + 0.01,
                fill=False, edgecolor='black', linewidth=1
            )
            self.ax1.add_patch(outer_black)
            
            # Middle thick colored line
            colored_line = patches.Circle(
                center, FASCICLE_RADIUS,
                fill=False, edgecolor=color, linewidth=3
            )
            self.ax1.add_patch(colored_line)
            
            # Inner thin black line
            inner_black = patches.Circle(
                center, FASCICLE_RADIUS - 0.01,
                fill=False, edgecolor='black', linewidth=1
            )
            self.ax1.add_patch(inner_black)
        
        # Draw electrode positions based on type
        if electrode_type == "discrete":
            # Draw discrete electrodes
            for i, pos in enumerate(ELECTRODE_POSITIONS):
                electrode_circle = patches.Circle(
                    pos, ELECTRODE_RADIUS,
                    fill=True, color='red', alpha=0.8
                )
                self.ax1.add_patch(electrode_circle)
                self.ax1.text(pos[0], pos[1], f'E{i+1}', ha='center', va='center', 
                             color='white', fontsize=8, fontweight='bold')
        else:  # ring electrode
            # Draw ring electrode
            ring_positions = generate_ring_electrode_positions(
                (NERVE_CENTER_X, NERVE_CENTER_Y), NERVE_RADIUS
            )
            for i, pos in enumerate(ring_positions):
                electrode_circle = patches.Circle(
                    pos, ELECTRODE_RADIUS,
                    fill=True, color='red', alpha=0.6
                )
                self.ax1.add_patch(electrode_circle)
    
    def _draw_nerve_structure_conductivity(self, electrode_type="discrete"):
        """Draw nerve and fascicle boundaries for conductivity plot."""
        # Draw nerve boundary
        nerve_circle = patches.Circle(
            (NERVE_CENTER_X, NERVE_CENTER_Y), NERVE_RADIUS,
            fill=False, edgecolor='black', linewidth=2
        )
        self.ax2.add_patch(nerve_circle)
        
        # Draw fascicle boundaries with thick colored line between two thin black lines
        fascicle_colors = ['#0053b2', '#7fe74e', '#c60000', '#f2ef30']  # blue, green, red, yellow
        
        for fascicle_idx, center in enumerate(self.geometry.fascicle_centers):
            color = fascicle_colors[fascicle_idx % len(fascicle_colors)]
            
            # Outer thin black line
            outer_black = patches.Circle(
                center, FASCICLE_RADIUS + 0.01,
                fill=False, edgecolor='black', linewidth=1
            )
            self.ax2.add_patch(outer_black)
            
            # Middle thick colored line
            colored_line = patches.Circle(
                center, FASCICLE_RADIUS,
                fill=False, edgecolor=color, linewidth=3
            )
            self.ax2.add_patch(colored_line)
            
            # Inner thin black line
            inner_black = patches.Circle(
                center, FASCICLE_RADIUS - 0.01,
                fill=False, edgecolor='black', linewidth=1
            )
            self.ax2.add_patch(inner_black)
        
        # Draw electrode positions based on type
        if electrode_type == "discrete":
            # Draw discrete electrodes
            for i, pos in enumerate(ELECTRODE_POSITIONS):
                electrode_circle = patches.Circle(
                    pos, ELECTRODE_RADIUS,
                    fill=True, color='red', alpha=0.8
                )
                self.ax2.add_patch(electrode_circle)
                self.ax2.text(pos[0], pos[1], f'E{i+1}', ha='center', va='center', 
                             color='white', fontsize=8, fontweight='bold')
        else:  # ring electrode
            # Draw ring electrode
            ring_positions = generate_ring_electrode_positions(
                (NERVE_CENTER_X, NERVE_CENTER_Y), NERVE_RADIUS
            )
            for i, pos in enumerate(ring_positions):
                electrode_circle = patches.Circle(
                    pos, ELECTRODE_RADIUS,
                    fill=True, color='red', alpha=0.6
                )
                self.ax2.add_patch(electrode_circle)
    
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
        # Clear all plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Set up plot properties
        X, Y = self.geometry.X, self.geometry.Y
        
        # Panel 1: Intensity field with fiber activation
        self.ax1.set_title('Intensity Field with Fiber Activation')
        self.ax1.set_aspect('equal')
        
        # Remove axis labels and ticks for cleaner look
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax1.set_xlabel('')
        self.ax1.set_ylabel('')
        
        # Create contour plot of intensity field with managua colormap
        intensity_field = np.abs(potential_field)
        # Apply sign back to show direction
        signed_intensity = np.sign(potential_field) * intensity_field
        vmax = 1.0  # Fixed intensity bounds
        
        # Use managua colormap directly (flipped)
        cmap = plt.get_cmap('managua').reversed()
        
        contour_levels = np.linspace(-vmax, vmax, 21)
        contour_plot = self.ax1.contourf(X, Y, signed_intensity, levels=contour_levels, 
                                        cmap=cmap, alpha=0.7, extend='both', vmin=-vmax, vmax=vmax)
        
        # Add colorbar with zero tick and plus/minus labels
        cbar1 = plt.colorbar(contour_plot, ax=self.ax1, shrink=0.8, ticks=[-1, 0, 1])
        cbar1.set_ticklabels(['-', '0', '+'])
        cbar1.ax.tick_params(labelsize=14)
        cbar1.set_label('Intensity', rotation=270, labelpad=20)
        
        # Draw nerve structure
        self._draw_nerve_structure("discrete")
        
        # Update fiber positions and colors
        self._plot_fibers(self.ax1)
        
        # Add legend with separate entries for borders and fiber states
        import matplotlib.lines as mlines
        from matplotlib.legend_handler import HandlerTuple
        from matplotlib.patches import Circle
        
        legend_handles = []
        legend_labels = []
        
        # On Target: show two colored border circles (blue and green)
        on_target_borders = (
            Circle((0, 0), 0.03, fill=False, edgecolor='#0053b2', linewidth=3),
            Circle((0, 0), 0.03, fill=False, edgecolor='#7fe74e', linewidth=3)
        )
        legend_handles.append(on_target_borders)
        legend_labels.append('On Target')
        
        # Off Target: show two colored border circles (red and yellow)
        off_target_borders = (
            Circle((0, 0), 0.03, fill=False, edgecolor='#c60000', linewidth=3),
            Circle((0, 0), 0.03, fill=False, edgecolor='#f2ef30', linewidth=3)
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
        self.ax1.legend(legend_handles, legend_labels,
                       handler_map={tuple: HandlerTuple(ndivide=None)},
                       loc='upper right', fontsize=8)
        
        # Panel 2: Conductivity field
        self.ax2.set_title('Conductivity Field')
        self.ax2.set_aspect('equal')
        
        # Remove axis labels and ticks for cleaner look
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax2.set_xlabel('')
        self.ax2.set_ylabel('')
        
        # Plot conductivity field
        sigma = self.geometry.get_conductivity()
        
        # Debug: Print conductivity values
        print(f"Conductivity values:")
        print(f"  Epineurium: {CONDUCTIVITY_EPINEURIUM} S/m")
        print(f"  Endoneurium: {CONDUCTIVITY_ENDONEURIUM} S/m") 
        print(f"  Perineurium: {CONDUCTIVITY_PERINEURIUM} S/m")
        print(f"  Outside: {CONDUCTIVITY_OUTSIDE} S/m")
        print(f"  Actual sigma range: {np.min(sigma):.2e} to {np.max(sigma):.2e} S/m")
        
        # Use log scale for better visualization of conductivity differences
        sigma_log = np.log10(sigma + 1e-10)  # Add small value to avoid log(0)
        conductivity_plot = self.ax2.imshow(sigma_log, extent=[0, DOMAIN_SIZE, 0, DOMAIN_SIZE], 
                                          cmap='viridis', alpha=0.8, origin='lower')
        
        # Add colorbar for conductivity (log scale)
        cbar2 = plt.colorbar(conductivity_plot, ax=self.ax2, shrink=0.8)
        cbar2.set_label('Log10(Conductivity) (S/m)', rotation=270, labelpad=20)
        
        # Draw nerve structure
        self._draw_nerve_structure_conductivity("discrete")
        
        # Panel 3: Activation statistics
        self.ax3.set_title('Activation Statistics by Fascicle')
        self.ax3.set_xlabel('Fascicle')
        self.ax3.set_ylabel('Activation Percentage (%)')
        
        stats = self.fiber_population.get_activation_stats()
        fascicles = list(stats.keys())
        percentages = [stats[f]['percentage'] for f in fascicles]
        
        bars = self.ax3.bar(fascicles, percentages, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
        self.ax3.set_ylim(0, 100)
        self.ax3.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add total statistics
        total_fibers = sum(stats[f]['total'] for f in fascicles)
        total_active = sum(stats[f]['active'] for f in fascicles)
        total_percentage = (total_active / total_fibers * 100) if total_fibers > 0 else 0
        
        self.ax3.text(0.02, 0.98, f'Total: {total_active}/{total_fibers} fibers ({total_percentage:.1f}%)',
                     transform=self.ax3.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Panel 4: Potential field cross-section
        self.ax4.set_title('Potential Field Cross-Section')
        self.ax4.set_xlabel('Distance (mm)')
        self.ax4.set_ylabel('Potential (V)')
        
        # Plot cross-section through center
        center_y = GRID_SIZE // 2
        cross_section = potential_field[center_y, :]
        x_coords = np.linspace(0, DOMAIN_SIZE, GRID_SIZE)
        
        self.ax4.plot(x_coords, cross_section, 'b-', linewidth=2, label='Potential')
        self.ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero potential')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend()
        
        plt.tight_layout()
        plt.draw()
    
    def _plot_fibers(self, ax):
        """Plot fibers on the specified axis."""
        # All fibers are black regardless of fascicle or activation status
        all_fibers = self.fiber_population.fibers
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
