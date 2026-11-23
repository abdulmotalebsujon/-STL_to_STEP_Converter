name: Build Windows EXE

on:
  workflow_dispatch:
  push:
    branches: [ main ]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install PyQt5 trimesh numpy scipy breptile pyinstaller

      - name: Build EXE
        run: |
          pyinstaller --onefile --windowed --name "STL_to_STEP_Converter" `
            --hidden-import=PyQt5.sip `
            --hidden-import=scipy.spatial.transform._rotation_groups `
            --hidden-import=trimesh.exchange.stl `
            --collect-all PyQt5 `
            --collect-all trimesh `
            --collect-all breptile `
            --collect-all build123d `
            --collect-all OCP `
            main.py

      - name: Upload EXE artifact
        uses: actions/upload-artifact@v4
        with:
          name: STL_to_STEP_Converter
          path: dist/STL_to_STEP_Converter.exe
