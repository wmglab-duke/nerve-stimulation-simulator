#!/usr/bin/env python3
"""Test script to debug fiber generation."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nerve_stimulation_simulator import NerveGeometry, FiberPopulation

# Test parameters
grid_size = 120
domain_size = 2.0
nerve_center = (1.0, 1.0)
nerve_radius = 0.4
fascicle_count = 3
fascicle_radius = 0.08

try:
    print("Creating geometry...")
    geometry = NerveGeometry(grid_size, domain_size, nerve_center, nerve_radius, fascicle_count, fascicle_radius)

    print("Creating fiber population...")
    fiber_pop = FiberPopulation(
        fascicle_centers=geometry.fascicle_centers,
        fascicle_radius=fascicle_radius,
        fiber_count_per_fascicle=10,
        diameter_range=(2.0, 12.0),
        threshold_base=0.1,
        threshold_scaling=8.0,
        min_spacing=0.01  # 0.01 μm
    )

    print(f"Total fibers generated: {len(fiber_pop.fibers)}")
    for i, fascicle in enumerate(geometry.fascicle_centers):
        fascicle_fibers = [f for f in fiber_pop.fibers if f['fascicle'] == i]
        print(f"Fascicle {i}: {len(fascicle_fibers)} fibers")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
