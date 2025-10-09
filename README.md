# 🧠 Nerve Stimulation Simulator

A Python-based simulation tool for modeling electrical nerve stimulation with real-time visualization and interactive controls.

## 📁 Project Structure

```
nerve_gui/
├── nerve_stimulation_simulator.py    # Core simulation engine
├── launch_gui.py                     # Launch script for Streamlit GUI
├── streamlit_gui/                    # Streamlit web interface
│   ├── app.py                       # Main GUI application
│   └── requirements.txt             # GUI dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### Option 1: Command Line Interface
```bash
python nerve_stimulation_simulator.py
```

### Option 2: Web GUI (Recommended)
```bash
python launch_gui.py
```

The web interface will open at `http://localhost:8501` in your default browser.

## 🧪 Features

### Core Simulation
- **Physics-based modeling**: Steady-state conduction equation `∇ · (σ ∇φ) = 0`
- **Multi-layer conductivity**: Epineurium, endoneurium, perineurium, and surrounding tissue
- **Lead field method**: Real-time parameter updates without recomputing the system matrix
- **Fiber activation model**: Threshold-based activation with diameter-dependent thresholds

### Visualization
- **4-panel display**:
  - Potential field with fiber activation
  - Conductivity field visualization
  - Activation statistics by fascicle
  - Potential field cross-section
- **Interactive controls**: Real-time parameter adjustment
- **Fiber visualization**: Active (filled) vs inactive (hollow) fibers

### Web Interface
- **Real-time controls**: Adjust all parameters via sliders and inputs
- **Live updates**: See results instantly as you change parameters
- **Statistics dashboard**: Detailed activation metrics
- **Responsive design**: Works on desktop and mobile devices

## ⚙️ Parameters

### Grid Settings
- **Grid Resolution**: 80-200 (higher = more accurate, slower)
- **Domain Size**: 2.0 mm × 2.0 mm

### Nerve Geometry
- **Nerve Radius**: 0.2-0.6 mm
- **Fascicle Count**: 2-5 fascicles
- **Fascicle Radius**: 0.05-0.15 mm

### Tissue Conductivity (S/m)
- **Epineurium**: 0.0001-1.0 (nerve sheath)
- **Endoneurium**: 0.1-2.0 (inside fascicles)
- **Perineurium**: 0.00001-0.1 (fascicle boundary)
- **Outside Nerve**: 0.1-10.0 (surrounding tissue)

### Electrode Settings
- **Electrode Radius**: 0.01-0.05 mm
- **Electrode Amplitudes**: -2.0 to +2.0 V
- **Electrode Positions**: 4 electrodes around the nerve

### Fiber Settings
- **Fibers per Fascicle**: 10-100
- **Fiber Diameter**: 1.0-25.0 μm
- **Activation Threshold**: 0.01-1.0 V

## 🔬 How It Works

1. **Geometry Setup**: Creates nerve cross-section with fascicles and perineurium
2. **Conductivity Field**: Assigns different conductivities to tissue layers
3. **Lead Field Computation**: Precomputes unit potential fields for each electrode
4. **Real-time Simulation**: Scales and sums lead fields based on electrode amplitudes
5. **Fiber Activation**: Determines which fibers activate based on potential thresholds
6. **Visualization**: Updates plots and statistics in real-time

## 📊 Understanding the Results

### Potential Field
- **Red regions**: Positive potential (anodic)
- **Blue regions**: Negative potential (cathodic)
- **White regions**: Zero potential
- **Contour lines**: Show potential gradients

### Fiber Activation
- **Filled black circles**: Activated fibers
- **Hollow black circles**: Non-activated fibers
- **Size**: Proportional to fiber diameter

### Statistics
- **Activation percentage**: Percentage of fibers activated in each fascicle
- **Total counts**: Absolute numbers of active/total fibers
- **Potential range**: Min/max potential values in the field

## 🛠️ Dependencies

### Core Simulation
- `numpy` >= 1.21.0
- `matplotlib` >= 3.5.0
- `scipy` >= 1.7.0

### Web GUI
- `streamlit` >= 1.28.0

## 🔧 Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install numpy matplotlib scipy streamlit
   ```
3. **Run the simulation**:
   ```bash
   python launch_gui.py
   ```

## 📝 Usage Tips

- **Start with default parameters** to see the basic simulation
- **Adjust electrode amplitudes** to see different activation patterns
- **Increase grid resolution** for more accurate results (but slower)
- **Modify conductivity values** to simulate different tissue types
- **Use negative electrode values** for cathodic stimulation (activates fibers)
- **Use positive electrode values** for anodic stimulation (doesn't activate fibers)

## 🎯 Applications

- **Research**: Study nerve stimulation mechanisms
- **Education**: Learn about electrical stimulation principles
- **Device Design**: Optimize electrode configurations
- **Clinical Simulation**: Model different stimulation scenarios

## 📚 Technical Details

The simulator uses:
- **Finite difference method** for solving the Laplace equation
- **Sparse matrix solver** for efficient computation
- **Harmonic mean conductivities** at tissue interfaces
- **Dirichlet boundary conditions** for electrodes and ground
- **Lead field approach** for real-time parameter updates

## 🤝 Contributing

Feel free to modify parameters, add features, or improve the visualization. The code is designed to be modular and extensible.

## 📄 License

This project is open source and available under the MIT License.
