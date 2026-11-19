# Destination Imagination: Conveyor Belt Shape Sorter

Computer vision shape detection and sorting pipeline built for a **Raspberry Pi Zero** (with **PiWiHub** and **Pi Zero Camera**) and testable directly on **macOS**.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full system architecture, camera placement guide, and phased roadmap.

## Quick Start (macOS)

1. **Generate synthetic test shapes & calibrate the model:**
   ```bash
   python3 scripts/generate_test_shapes.py
   ```
   This generates 75 synthetic side-profile images in `data/samples/{cube,sphere,pyramid}/`, calibrates `config/shapes_model.json`, and saves an annotated preview image at `data/preview_hud_grid.png`.

2. **Run the live simulated conveyor belt (opens debug HUD window, press `q` to quit):**
   ```bash
   PYTHONPATH=src python3 -m sorter.main --mode simulate
   ```
   *(Add `--headless` to run without a GUI window.)*

3. **Test live with your Mac webcam:**
   ```bash
   PYTHONPATH=src python3 -m sorter.main --mode live
   ```
   *(Note: If holding dark objects against a bright background, set `invert_threshold: true` in `config/settings.yaml`.)*

4. **Collect new shape samples & recalibrate:**
   ```bash
   # Collect snapshots automatically as objects pass through the inspection zone:
   PYTHONPATH=src python3 -m sorter.main --mode collect --shape cylinder

   # Recalibrate the shape model from all folders in data/samples/:
   PYTHONPATH=src python3 -m sorter.main --mode calibrate
   ```

5. **Run automated unit and accuracy tests:**
   ```bash
   pytest -v
   ```
