import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from sklearn.linear_model import Ridge, HuberRegressor, Lasso
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
                             QPushButton, QDialog, QDialogButtonBox, QMessageBox)


class DraggablePoint:
    lock = None

    def __init__(self, parent, point):
        self.cidmotion = None
        self.cidrelease = None
        self.cidpress = None
        self.parent = parent
        self.point = point
        self.press = None
        self.background = None
        self.point.set_color('blue')
        self.point.set_picker(5)
        self.connect()

    def connect(self):
        self.cidpress = self.point.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes or DraggablePoint.lock is not None:
            return

        contains, attrd = self.point.contains(event)
        if not contains:
            return

        self.press = (self.point.center[0], self.point.center[1]), (event.xdata, event.ydata)
        DraggablePoint.lock = self

        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(axes.bbox)

        axes.draw_artist(self.point)
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePoint.lock is not self or event.inaxes is None:
            return

        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (x0 + dx, y0 + dy)

        canvas = self.point.figure.canvas
        axes = self.point.axes
        canvas.restore_region(self.background)
        axes.draw_artist(self.point)
        canvas.blit(axes.bbox)

    def on_release(self, event):
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None
        self.point.set_animated(False)
        self.background = None

        self.parent.update_point_position(self)
        self.point.figure.canvas.draw()

    def disconnect(self):
        if self.point.figure:
            self.point.figure.canvas.mpl_disconnect(self.cidpress)
            self.point.figure.canvas.mpl_disconnect(self.cidrelease)
            self.point.figure.canvas.mpl_disconnect(self.cidmotion)


class RandomPointsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Random Points")
        self.setGeometry(200, 200, 450, 350)
        self.setStyleSheet("""
            QDialog {
                background-color: #dbdad5;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #333;
                font-size: 12px;
                font-weight: 500;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        layout.addWidget(QLabel("Function (use x as variable):"))
        self.function_edit = QLineEdit()
        self.function_edit.setPlaceholderText("e.g., x**2, sin(x), exp(x/3)")
        self.function_edit.setText("sin(x)")
        layout.addWidget(self.function_edit)

        layout.addWidget(QLabel("Number of Points:"))
        self.num_points = QSpinBox()
        self.num_points.setRange(5, 1000)
        self.num_points.setValue(50)
        layout.addWidget(self.num_points)

        layout.addWidget(QLabel("Noise Level:"))
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 10.0)
        self.noise_spin.setValue(1.0)
        self.noise_spin.setSingleStep(0.1)
        layout.addWidget(self.noise_spin)

        layout.addWidget(QLabel("X Range:"))
        hbox = QHBoxLayout()
        hbox.setSpacing(10)
        self.xmin = QDoubleSpinBox()
        self.xmin.setRange(-10, -1)
        self.xmin.setValue(-10)
        self.xmax = QDoubleSpinBox()
        self.xmax.setRange(1, 10)
        self.xmax.setValue(10)
        hbox.addWidget(QLabel("Min:"))
        hbox.addWidget(self.xmin)
        hbox.addWidget(QLabel("Max:"))
        hbox.addWidget(self.xmax)
        layout.addLayout(hbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QPushButton {
                min-width: 80px;
                padding: 6px;
            }
            QPushButton:first-child {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:last-child {
                background-color: #f44336;
                color: white;
            }
        """)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_parameters(self):
        return {
            "function": self.function_edit.text(),
            "num_points": self.num_points.value(),
            "noise_level": self.noise_spin.value(),
            "x_range": (self.xmin.value(), self.xmax.value())
        }


class RegressionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Regression Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QWidget {
                background-color: #dbdad5;
            }
            QLabel {
                color: #333;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: 500;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#danger {
                background-color: #f44336;
            }
            QPushButton#danger:hover {
                background-color: #d32f2f;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
                min-width: 80px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #ddd;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #ddd;
                selection-background-color: #e0e0e0;  
                selection-color: black;               
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #f0f0f0;  
                color: black;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        self.figure = Figure(figsize=(10, 7), facecolor='#f5f5f5')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
        self.ax.set_facecolor('#f9f9f9')
        self.ax.axhline(0, color='#333333', linewidth=0.8)
        self.ax.axvline(0, color='#333333', linewidth=0.8)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')
        self.ax.set_title("Interactive Regression Tool", fontsize=14, fontweight='bold', pad=20)
        self.ax.set_xlabel("X", fontsize=10)
        self.ax.set_ylabel("Y", fontsize=10)
        self.ax.tick_params(colors='#555555', labelsize=8)

        self.points = []
        self.draggable_points = []
        self.regression_line = None

        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(0, 0, 0, 0)

        control_layout.addWidget(QLabel("Polynomial Degree:"))
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 100)
        self.degree_spin.setValue(1)
        control_layout.addWidget(self.degree_spin)

        control_layout.addWidget(QLabel("Loss Function:"))
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["SSE (Sum of Squared Errors)",
                                  "RMS (Root Mean Square)",
                                  "Ridge (L2 Regularization)",
                                  "Lasso (L1 Regularization)",
                                  "Huber (Robust Regression)"])
        control_layout.addWidget(self.loss_combo)

        control_layout.addWidget(QLabel("Regularization (λ):"))
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.0, 100.0)
        self.lambda_spin.setValue(0.1)
        self.lambda_spin.setSingleStep(0.1)
        control_layout.addWidget(self.lambda_spin)

        self.calc_button = QPushButton("Regression")
        self.calc_button.clicked.connect(self.calculate_regression)
        self.calc_button.setStyleSheet("background-color: #2196F3;")
        control_layout.addWidget(self.calc_button)

        button_group = QWidget()
        button_layout = QHBoxLayout(button_group)
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(0, 0, 0, 0)

        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self.clear_points)
        clear_button.setObjectName("danger")
        button_layout.addWidget(clear_button)

        random_button = QPushButton("Random")
        random_button.clicked.connect(self.generate_random_points)
        button_layout.addWidget(random_button)

        control_layout.addWidget(button_group)
        layout.addWidget(control_panel)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Click to add points | Right-click to delete")

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 3:
            self.delete_point(event)
            return

        if event.button == 1:
            self.add_point(event.xdata, event.ydata)

    def add_point(self, x, y):
        point = Circle((x, y), 0.25, color='#E53935', alpha=0.9,
                        linewidth=0.5, picker=5)
        self.ax.add_patch(point)

        draggable = DraggablePoint(self, point)
        self.draggable_points.append(draggable)
        self.points.append((x, y))

        self.canvas.draw()
        self.status_bar.showMessage(f"Added point at ({x:.2f}, {y:.2f})")

    def update_point_position(self, draggable):
        index = self.draggable_points.index(draggable)
        x, y = draggable.point.center
        self.points[index] = (x, y)
        self.status_bar.showMessage(f"Moved point to ({x:.2f}, {y:.2f})")

    def delete_point(self, event):
        for i, draggable in enumerate(self.draggable_points[:]):
            if draggable.point.contains(event)[0]:
                x, y = draggable.point.center
                draggable.point.remove()
                draggable.disconnect()
                self.draggable_points.remove(draggable)
                del self.points[i]
                self.canvas.draw()
                self.status_bar.showMessage(f"Deleted point at ({x:.2f}, {y:.2f})")
                break

    def clear_points(self):
        for draggable in self.draggable_points[:]:
            draggable.point.remove()
            if draggable.point.figure:
                draggable.disconnect()

        self.points = []
        self.draggable_points = []

        if self.regression_line:
            self.regression_line.remove()
            self.regression_line = None

        self.canvas.draw()
        self.status_bar.showMessage("Cleared all points")

    def generate_random_points(self):
        dialog = RandomPointsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_parameters()
            self.create_random_points(params)

    def create_random_points(self, params):

        x_min, x_max = params["x_range"]
        n = params["num_points"]
        x = np.linspace(x_min, x_max, n)

        try:
            func_str = params["function"]
            replacements = {
                "sin": "np.sin", "cos": "np.cos", "tan": "np.tan",
                "exp": "np.exp", "log": "np.log", "sqrt": "np.sqrt",
                "abs": "np.abs", "arcsin": "np.arcsin", "arccos": "np.arccos",
                "arctan": "np.arctan", "sinh": "np.sinh", "cosh": "np.cosh",
                "tanh": "np.tanh"
            }

            for old, new in replacements.items():
                func_str = func_str.replace(old, new)

            y = eval(func_str, {"np": np, "x": x})
            previous_x = x.copy()

            noise_level = params["noise_level"]
            if noise_level > 0:
                y += noise_level * np.random.randn(n)


            for x_val, y_val in zip(previous_x, y):
                self.add_point(x_val, y_val)

            self.status_bar.showMessage(f"Generated {n} random points with function: {params['function']}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to evaluate function:\n{str(e)}")

    def calculate_regression(self):
        if not self.points:
            QMessageBox.warning(self, "No Points", "Please add some points first!")
            return
        degree = self.degree_spin.value()
        loss_function = self.loss_combo.currentText()
        lambda_val = self.lambda_spin.value()

        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])

        try:
            X = np.vander(x, degree + 1, increasing=True)

            if "SSE" in loss_function:
                w = np.linalg.lstsq(X, y, rcond=None)[0]
            elif "Ridge" in loss_function:
                model = Ridge(alpha=lambda_val, fit_intercept=True)
                model.fit(X, y)
                w = model.coef_
                bias = model.intercept_

            elif "Lasso" in loss_function:
                model = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=100000)
                model.fit(X, y)
                w = model.coef_
            elif "Huber" in loss_function:
                model = HuberRegressor(alpha=lambda_val, fit_intercept=False)
                model.fit(X, y)
                w = model.coef_
            elif "RMS" in loss_function:
                w = np.linalg.lstsq(X, y, rcond=None)[0]

            x_curve = np.linspace(min(x) - 1, max(x) + 1, 200)
            X_curve = np.vander(x_curve, degree + 1, increasing=True)
            y_curve = X_curve @ w

            y_pred = X @ w
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            if self.regression_line:
                self.regression_line.remove()

            self.regression_line, = self.ax.plot(
                x_curve, y_curve, color='#1E88E5', linewidth=2.5, alpha=0.9,
                label=f"{loss_function.split(' ')[0]} (deg={degree}, R²={r_squared:.3f})"
            )

            self.ax.legend(loc='upper right', fontsize=9)
            self.canvas.draw()

            self.status_bar.showMessage(
                f"Calculated {loss_function.split(' ')[0]} regression (degree {degree}) "
                f"with R² = {r_squared:.3f}"
            )

        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", f"Regression failed: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont()
    font.setFamily("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)

    window = RegressionApp()
    window.show()
    sys.exit(app.exec_())