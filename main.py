import os
import sys
import shutil
import tempfile
import traceback

def _configure_frozen_environment() -> None:
    """Ensure OpenCASCADE (OCP) and Qt plugins resolve inside a PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        ocp_res = os.path.join(base_path, "OCP", "resources")
        if os.path.isdir(ocp_res):
            os.environ.setdefault("CASROOT", ocp_res)


_configure_frozen_environment()

import numpy as np
import trimesh
from breptile import convert as breptile_convert
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


FOOTER_TEXT = (
    "Made by:\n"
    "Md . Abdul Motaleb (Sujon)\n"
    "Asst. Manager-IT\n"
    "HATIL Complex Ltd.\n"
    "Cell: 01924797293"
)


class ConversionWorker(QThread):
    """Background worker that runs Trimesh repairs and breptile STEP conversion."""

    progress_changed = pyqtSignal(int, str)
    conversion_succeeded = pyqtSignal(str, str, dict)  # temp_dir, temp_step_path, report
    conversion_failed = pyqtSignal(str)

    def __init__(
        self,
        stl_path: str,
        repair_holes: bool,
        fix_normals: bool,
        remove_degenerate: bool,
        merge_vertices: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.stl_path = stl_path
        self.repair_holes = repair_holes
        self.fix_normals = fix_normals
        self.remove_degenerate = remove_degenerate
        self.merge_vertices = merge_vertices

    def _load_raw_mesh(self, path: str) -> trimesh.Trimesh:
        """Load STL without automatic processing so unchecked options stay disabled."""
        loaded = trimesh.load(path, process=False, force="mesh")

        if isinstance(loaded, trimesh.Scene):
            geometries = [
                g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)
            ]
            if not geometries:
                raise ValueError("The selected STL file does not contain any valid 3D mesh geometry.")
            mesh = trimesh.util.concatenate(geometries)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise ValueError(f"Unsupported mesh type loaded from file: {type(loaded).__name__}")

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("The selected STL mesh is empty (0 vertices or 0 faces).")

        # Ensure numpy arrays are contiguous float64/int64 for repair routines
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        mesh.faces = np.asarray(mesh.faces, dtype=np.int64)
        return mesh

    def run(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="stl2step_")
        try:
            any_repairs = any(
                [
                    self.merge_vertices,
                    self.remove_degenerate,
                    self.repair_holes,
                    self.fix_normals,
                ]
            )

            self.progress_changed.emit(15, "Loading STL mesh...")
            mesh = self._load_raw_mesh(self.stl_path)
            initial_faces = len(mesh.faces)

            if any_repairs:
                step_num = 0
                applied = []

                if self.merge_vertices:
                    step_num += 1
                    self.progress_changed.emit(25, "Merging duplicate vertices...")
                    mesh.merge_vertices()
                    applied.append("Merge Vertices")

                if self.remove_degenerate:
                    step_num += 1
                    self.progress_changed.emit(35, "Removing degenerate faces...")
                    if hasattr(mesh, "nondegenerate_faces"):
                        mesh.update_faces(mesh.nondegenerate_faces())
                    elif hasattr(mesh, "remove_degenerate_faces"):
                        mesh.remove_degenerate_faces()
                    if len(mesh.faces) == 0:
                        raise ValueError(
                            "All faces were removed as degenerate. The mesh has no valid triangles."
                        )
                    applied.append("Remove Degenerate Faces")

                if self.repair_holes:
                    step_num += 1
                    self.progress_changed.emit(45, "Repairing mesh holes...")
                    trimesh.repair.fill_holes(mesh)
                    applied.append("Repair Mesh Holes")

                if self.fix_normals:
                    step_num += 1
                    self.progress_changed.emit(55, "Fixing face normals...")
                    trimesh.repair.fix_normals(mesh)
                    applied.append("Fix Normals")

                source_stl_for_step = os.path.join(temp_dir, "processed_input.stl")
                mesh.export(source_stl_for_step)
            else:
                # All options unchecked: convert the raw STL untouched
                self.progress_changed.emit(40, "Skipping mesh repairs (raw mesh mode)...")
                source_stl_for_step = self.stl_path
                applied = ["None (Raw Mesh)"]

            temp_step_path = os.path.join(temp_dir, "converted_output.step")
            self.progress_changed.emit(
                70, "Converting mesh to STEP B-Rep geometry (this may take a moment)..."
            )

            used_mode = "auto"
            try:
                report = breptile_convert(
                    source_stl_for_step,
                    temp_step_path,
                    mode="auto",
                    schema="AP214",
                    force=True,
                )
            except Exception as auto_err:
                # If analytic fitting cannot handle an unrepaired/non-manifold raw mesh,
                # fall back to tessellated B-Rep STEP export so valid STEP output is produced.
                self.progress_changed.emit(
                    82, "Analytic fitting fallback: generating tessellated STEP B-Rep..."
                )
                used_mode = "tessellated"
                try:
                    report = breptile_convert(
                        source_stl_for_step,
                        temp_step_path,
                        mode="tessellated",
                        schema="AP214",
                        force=True,
                    )
                except Exception as fallback_err:
                    raise RuntimeError(
                        f"STEP conversion failed.\nPrimary error: {auto_err}\nFallback error: {fallback_err}"
                    ) from fallback_err

            if not os.path.isfile(temp_step_path) or os.path.getsize(temp_step_path) == 0:
                raise RuntimeError("Conversion completed without producing a valid STEP file.")

            with open(temp_step_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(256)
            if "ISO-10303-21" not in header:
                raise RuntimeError("Generated file does not have a valid ISO-10303-21 STEP header.")

            summary = {
                "initial_faces": initial_faces,
                "final_faces": len(mesh.faces),
                "watertight": bool(mesh.is_watertight),
                "repairs_applied": applied,
                "mode": used_mode,
                "breptile_report": report if isinstance(report, dict) else {},
            }

            self.progress_changed.emit(95, "STEP file ready. Choose save location...")
            self.conversion_succeeded.emit(temp_dir, temp_step_path, summary)

        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            detail = f"{exc}\n\n{traceback.format_exc(limit=3)}"
            self.conversion_failed.emit(detail)


class StlToStepWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_stl_path: str = ""
        self.worker: ConversionWorker | None = None

        self.setWindowTitle("STL to STEP Converter")
        self.setMinimumSize(620, 540)
        self.resize(660, 580)

        self._init_ui()
        self._apply_styles()

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 22, 24, 16)
        main_layout.setSpacing(14)

        # Header
        title_label = QLabel("STL to STEP Converter")
        title_label.setObjectName("HeaderTitle")
        subtitle_label = QLabel(
            "Convert 3D STL meshes to CAD-ready STEP (.step / .stp) files with optional mesh repair."
        )
        subtitle_label.setObjectName("HeaderSubtitle")
        subtitle_label.setWordWrap(True)
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        # Input File Selection Section
        file_group = QGroupBox("Input STL File")
        file_layout = QHBoxLayout(file_group)
        file_layout.setContentsMargins(14, 16, 14, 14)
        file_layout.setSpacing(12)

        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setObjectName("FilePathLabel")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.browse_button = QPushButton("Browse")
        self.browse_button.setObjectName("BrowseButton")
        self.browse_button.setFixedWidth(110)
        self.browse_button.setCursor(Qt.PointingHandCursor)
        self.browse_button.clicked.connect(self.browse_stl_file)

        file_layout.addWidget(self.file_path_label, stretch=1)
        file_layout.addWidget(self.browse_button)
        main_layout.addWidget(file_group)

        # Correction Options Section
        options_group = QGroupBox("Correction Options")
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(14, 16, 14, 14)
        options_layout.setSpacing(10)

        self.chk_repair_holes = QCheckBox("Repair Mesh Holes")
        self.chk_repair_holes.setChecked(True)

        self.chk_fix_normals = QCheckBox("Fix Normals")
        self.chk_fix_normals.setChecked(True)

        self.chk_remove_degenerate = QCheckBox("Remove Degenerate Faces")
        self.chk_remove_degenerate.setChecked(False)

        self.chk_merge_vertices = QCheckBox("Merge Vertices")
        self.chk_merge_vertices.setChecked(False)

        for chk in (
            self.chk_repair_holes,
            self.chk_fix_normals,
            self.chk_remove_degenerate,
            self.chk_merge_vertices,
        ):
            chk.setCursor(Qt.PointingHandCursor)
            options_layout.addWidget(chk)

        main_layout.addWidget(options_group)

        # Convert Action Button
        self.convert_button = QPushButton("Convert")
        self.convert_button.setObjectName("ConvertButton")
        self.convert_button.setMinimumHeight(48)
        self.convert_button.setCursor(Qt.PointingHandCursor)
        self.convert_button.clicked.connect(self.start_conversion)
        main_layout.addWidget(self.convert_button)

        # Progress & Status Section
        status_layout = QVBoxLayout()
        status_layout.setSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(22)

        self.status_label = QLabel("Ready. Select an STL file to begin.")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)

        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(self.status_label)
        main_layout.addLayout(status_layout)

        main_layout.addStretch(1)

        # Divider + Fixed Branding Footer
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        divider.setObjectName("FooterDivider")
        main_layout.addWidget(divider)

        self.footer_label = QLabel(FOOTER_TEXT)
        self.footer_label.setObjectName("FooterLabel")
        self.footer_label.setAlignment(Qt.AlignCenter)
        footer_font = QFont("Segoe UI", 9)
        footer_font.setWeight(QFont.Medium)
        self.footer_label.setFont(footer_font)
        main_layout.addWidget(self.footer_label)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f4f6f9;
            }
            QLabel#HeaderTitle {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 20px;
                font-weight: 700;
                color: #1e293b;
            }
            QLabel#HeaderSubtitle {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12px;
                color: #64748b;
                margin-bottom: 4px;
            }
            QGroupBox {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
                font-weight: 600;
                color: #334155;
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background-color: #ffffff;
            }
            QLabel#FilePathLabel {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12px;
                color: #334155;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 5px;
                padding: 8px 10px;
            }
            QCheckBox {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
                color: #1e293b;
                spacing: 8px;
            }
            QPushButton#BrowseButton {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 13px;
                font-weight: 600;
                color: #1e293b;
                background-color: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 14px;
            }
            QPushButton#BrowseButton:hover {
                background-color: #cbd5e1;
            }
            QPushButton#BrowseButton:disabled {
                color: #94a3b8;
                background-color: #f1f5f9;
            }
            QPushButton#ConvertButton {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 15px;
                font-weight: 700;
                color: #ffffff;
                background-color: #2563eb;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton#ConvertButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton#ConvertButton:pressed {
                background-color: #1e40af;
            }
            QPushButton#ConvertButton:disabled {
                background-color: #93c5fd;
            }
            QProgressBar {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                background-color: #ffffff;
                text-align: center;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                color: #1e293b;
            }
            QProgressBar::chunk {
                background-color: #2563eb;
                border-radius: 5px;
            }
            QLabel#StatusLabel {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12px;
                color: #475569;
            }
            QFrame#FooterDivider {
                color: #cbd5e1;
                margin-top: 6px;
            }
            QLabel#FooterLabel {
                color: #334155;
                background-color: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 10px 14px;
                line-height: 1.35;
            }
            """
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.browse_button.setEnabled(enabled)
        self.convert_button.setEnabled(enabled)
        self.chk_repair_holes.setEnabled(enabled)
        self.chk_fix_normals.setEnabled(enabled)
        self.chk_remove_degenerate.setEnabled(enabled)
        self.chk_merge_vertices.setEnabled(enabled)

    def browse_stl_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input STL File",
            os.path.dirname(self.selected_stl_path) if self.selected_stl_path else "",
            "STL Files (*.stl);;All Files (*)",
        )
        if file_path:
            self.selected_stl_path = file_path
            self.file_path_label.setText(file_path)
            self.progress_bar.setValue(0)
            self.status_label.setText("STL file loaded. Configure options and click Convert.")

    def start_conversion(self) -> None:
        if not self.selected_stl_path or not os.path.isfile(self.selected_stl_path):
            QMessageBox.warning(
                self,
                "No Input File",
                "Please select a valid .stl file using the Browse button first.",
            )
            return

        if self.worker is not None and self.worker.isRunning():
            return

        self._set_controls_enabled(False)
        self.progress_bar.setValue(5)
        self.status_label.setText("Starting conversion...")

        self.worker = ConversionWorker(
            stl_path=self.selected_stl_path,
            repair_holes=self.chk_repair_holes.isChecked(),
            fix_normals=self.chk_fix_normals.isChecked(),
            remove_degenerate=self.chk_remove_degenerate.isChecked(),
            merge_vertices=self.chk_merge_vertices.isChecked(),
            parent=self,
        )
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.conversion_succeeded.connect(self._on_conversion_succeeded)
        self.worker.conversion_failed.connect(self._on_conversion_failed)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _on_progress_changed(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def _on_conversion_succeeded(self, temp_dir: str, temp_step_path: str, summary: dict) -> None:
        try:
            base_name = os.path.splitext(os.path.basename(self.selected_stl_path))[0]
            default_dir = os.path.dirname(self.selected_stl_path)
            suggested_path = os.path.join(default_dir, f"{base_name}.step")

            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Converted STEP File",
                suggested_path,
                "STEP Files (*.step *.stp);;All Files (*)",
            )

            if not save_path:
                self.progress_bar.setValue(0)
                self.status_label.setText("Save cancelled by user.")
                return

            if not save_path.lower().endswith((".step", ".stp")):
                save_path += ".step"

            shutil.copy2(temp_step_path, save_path)

            self.progress_bar.setValue(100)
            self.status_label.setText(f"Success! Saved STEP file to: {save_path}")

            repairs_str = ", ".join(summary.get("repairs_applied", []))
            QMessageBox.information(
                self,
                "Conversion Successful",
                (
                    f"STEP file saved successfully!\n\n"
                    f"Output File:\n{save_path}\n\n"
                    f"Repairs Applied: {repairs_str}\n"
                    f"Mesh Faces: {summary.get('initial_faces')} -> {summary.get('final_faces')}\n"
                    f"Watertight: {'Yes' if summary.get('watertight') else 'No'}\n"
                    f"Conversion Mode: {summary.get('mode', 'auto')}"
                ),
            )
        except Exception as save_err:
            self.progress_bar.setValue(0)
            self.status_label.setText("Failed to save STEP file.")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Conversion succeeded, but saving the output file failed:\n\n{save_err}",
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self._set_controls_enabled(True)

    def _on_conversion_failed(self, error_message: str) -> None:
        self.progress_bar.setValue(0)
        self.status_label.setText("Conversion failed. See error dialog for details.")
        self._set_controls_enabled(True)
        QMessageBox.critical(
            self,
            "Conversion Error",
            f"An error occurred while converting the STL file:\n\n{error_message}",
        )

    def _cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Conversion In Progress",
                "A conversion is currently running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self.worker.terminate()
            self.worker.wait(2000)
        event.accept()


def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("STL to STEP Converter")

    window = StlToStepWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
