import io
import sys
import numpy as np
import sympy as sp
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QDoubleSpinBox,
    QSpinBox, QSplitter, QCheckBox, QFrame,
)


STYLESHEET = """
QMainWindow {
    background-color: #dbdad5;
    color: #333333;
}

QWidget {
    background-color: #dbdad5;
    color: #333333;
    border: none;
}

QGroupBox {
    background-color: #f0f0f0;
    border: 1px solid #c0c0c0;
    border-radius: 8px;
    margin-top: 1.5ex;
    padding-top: 15px;
    padding-bottom: 15px;
    padding-left: 15px;
    padding-right: 15px;
    font-weight: bold;
    font-size: 12pt;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
    color: #555555;
}

QLabel {
    background-color: #f0f0f0;
    color: #555555;
    font-size: 10pt;
}

QLineEdit {
    background-color: #ffffff;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    padding: 8px;
    color: #333333;
    font-size: 11pt;
    selection-background-color: #b0d0ff;
}

QLineEdit:focus {
    border: 1px solid #5a7bff;
}

QPushButton {
    color: white;
    border: none;
    border-radius: 4px;
    padding: 10px 15px;
    font-weight: bold;
    font-size: 11pt;
    min-width: 100px;
}

QPushButton#generateBtn {
    background-color: #4caf50;
}

QPushButton#generateBtn:hover {
    background-color: #66bb6a;
}

QPushButton#generateBtn:pressed {
    background-color: #388e3c;
}

QPushButton#regressBtn {
    background-color: #2196f3;
}

QPushButton#regressBtn:hover {
    background-color: #42a5f5;
}

QPushButton#regressBtn:pressed {
    background-color: #0b79d0;
}

QPushButton:disabled {
    background-color: #a0a0a0;
    color: #e0e0e0;
}

QDoubleSpinBox, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    padding: 8px;
    color: #333333;
    font-size: 11pt;
}

QDoubleSpinBox::up-button, QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #c0c0c0;
}

QDoubleSpinBox::down-button, QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid #c0c0c0;
}

QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
    image: url(none);
    width: 10px;
    height: 10px;
}

QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
    image: url(none);
    width: 10px;
    height: 10px;
}

QCheckBox {
    color: #555555;
    font-size: 10pt;
    spacing: 8px;
    background-color: #f0f0f0;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #c0c0c0;
    border-radius: 4px;
    background-color: #ffffff;
}

QCheckBox::indicator:checked {
    background-color: #2196f3;
    image: url(none);
    border: 1px solid #c0c0c0;
}

QTabWidget::pane {
    border: 5px solid #c0c0c0;
}

QTabBar::tab {
    background-color: #f0f0f0;
    color: #555555;
    padding: 10px 20px;
    border: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    font-size: 10pt;
}

QTabBar::tab:selected {
    background-color: #2196f3;
    color: white;
    font-weight: bold;
}

QSplitter::handle {
    background-color: #c0c0c0;
    width: 3px;
}

QStatusBar {
    background-color: #f0f0f0;
    color: #555555;
    border-top: 1px solid #c0c0c0;
    font-size: 9pt;
}
"""

class LatexLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def set_latex(self, latex_str):
        fig = Figure(facecolor='none', dpi=150)
        fig.set_canvas(FigureCanvasAgg(fig))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

        text_color = self.palette().color(QPalette.WindowText)
        color_tuple = (text_color.red() / 255, text_color.green() / 255, text_color.blue() / 255)

        t = ax.text(0.5, 0.5, f"${latex_str}$",
                    ha='center', va='center',
                    size=8,
                    color=color_tuple,
                    usetex=False)

        fig.canvas.draw()
        bbox = t.get_window_extent()
        fig.set_size_inches(bbox.width / fig.dpi, bbox.height / fig.dpi)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=fig.dpi,
                    bbox_inches='tight', pad_inches=0.0,
                    transparent=True)
        self.setPixmap(QPixmap())
        self.pixmap().loadFromData(buf.getvalue(), 'PNG')

        buf.close()

class SurfaceRegressionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.y_grid = None
        self.x_grid = None
        self.lasso_model = None
        self.ols_model = None
        self.z_true = None
        self.z_noisy = None
        self.y = None
        self.x = None
        self.status_bar = None
        self.generate_btn = None
        self.regress_btn = None
        self.show_ols_check = None
        self.show_lasso_check = None
        self.show_true_surface_check = None
        self.alpha_spin = None
        self.show_points_check = None
        self.degree_spin = None
        self.noise_spin = None
        self.points_spin = None
        self.function_input = None
        self.function_preview = None
        self.canvas = None
        self.figure = None
        self.plot_widget = None
        self.setWindowTitle("3D Surface Regression")
        self.setGeometry(100, 100, 1400, 850)

        self.setStyleSheet(STYLESHEET)

        app_font = QFont("Segoe UI", 9)
        QApplication.setFont(app_font)

        self.function_str = "sin(x) + cos(y)"
        self.noise_variance = 0.1
        self.num_points = 1000
        self.poly_degree = 10
        self.lasso_alpha = 0.1
        self.x_range = (-3, 3)
        self.y_range = (-3, 3)

        self.init_ui()

        self.generate_data()
        self.update_plot()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        control_panel = QWidget()
        control_panel.setMinimumWidth(300)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        splitter.addWidget(control_panel)

        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self.plot_widget)

        splitter.setSizes([350, 1050])

        self.figure = Figure(figsize=(10, 8), dpi=100, facecolor="#f0f0f0")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #f0f0f0;")
        plot_layout.addWidget(self.canvas)

        header = QLabel("3D Surface Regression")
        header_font = QFont("Segoe UI", 18, QFont.Bold)
        header.setFont(header_font)
        header.setStyleSheet("color: #2196f3; padding: 10px 0;")
        header.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(header)

        function_group = QGroupBox("Function Definition")
        function_layout = QVBoxLayout(function_group)
        function_layout.setSpacing(10)

        self.function_input = QLineEdit(self.function_str)
        self.function_input.setPlaceholderText("Enter function of x and y (e.g., sin(x)*cos(y))")
        function_layout.addWidget(QLabel("f(x, y):"))
        function_layout.addWidget(self.function_input)

        self.function_preview = LatexLabel()
        self.function_preview.setAlignment(Qt.AlignCenter)
        self.function_preview.setStyleSheet("font-size: 8pt; color: #555555;")
        function_layout.addWidget(QLabel("Function Preview:"))
        function_layout.addWidget(self.function_preview)
        self.update_function_preview()

        noise_group = QGroupBox("Noise Settings")
        noise_layout = QVBoxLayout(noise_group)
        noise_layout.setSpacing(10)

        noise_layout.addWidget(QLabel("Noise Variance:"))
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 10.0)
        self.noise_spin.setSingleStep(0.05)
        self.noise_spin.setValue(self.noise_variance)
        noise_layout.addWidget(self.noise_spin)

        noise_layout.addWidget(QLabel("Number of Points:"))
        self.points_spin = QSpinBox()
        self.points_spin.setRange(10, 2000)
        self.points_spin.setValue(self.num_points)
        noise_layout.addWidget(self.points_spin)

        regression_group = QGroupBox("Regression Settings")
        regression_layout = QVBoxLayout(regression_group)
        regression_layout.setSpacing(10)

        regression_layout.addWidget(QLabel("Polynomial Degree:"))
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 40)
        self.degree_spin.setValue(self.poly_degree)
        regression_layout.addWidget(self.degree_spin)

        regression_layout.addWidget(QLabel("Lasso Alpha:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 10.0)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setValue(self.lasso_alpha)
        regression_layout.addWidget(self.alpha_spin)

        vis_group = QGroupBox("Visualization Settings")
        vis_layout = QVBoxLayout(vis_group)
        vis_layout.setSpacing(10)

        self.show_points_check = QCheckBox("Show Points", self)
        self.show_points_check.setChecked(True)
        vis_layout.addWidget(self.show_points_check)

        self.show_true_surface_check = QCheckBox("Show True Surface", self)
        self.show_true_surface_check.setChecked(True)
        vis_layout.addWidget(self.show_true_surface_check)

        self.show_ols_check = QCheckBox("Show OLS Regression", self)
        self.show_ols_check.setChecked(True)
        vis_layout.addWidget(self.show_ols_check)

        self.show_lasso_check = QCheckBox("Show Lasso Regression", self)
        self.show_lasso_check.setChecked(True)
        vis_layout.addWidget(self.show_lasso_check)

        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.generate_btn = QPushButton("Generate Data")
        self.generate_btn.setObjectName("generateBtn")  # For CSS targeting
        self.generate_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.generate_btn.setMinimumHeight(40)

        self.regress_btn = QPushButton("Run Regression")
        self.regress_btn.setObjectName("regressBtn")  # For CSS targeting
        self.regress_btn.setIcon(QIcon.fromTheme("system-run"))
        self.regress_btn.setMinimumHeight(40)

        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.regress_btn)

        control_layout.addWidget(function_group)
        control_layout.addWidget(noise_group)
        control_layout.addWidget(regression_group)
        control_layout.addWidget(vis_group)
        control_layout.addWidget(button_frame)
        control_layout.addStretch()

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        self.function_input.textChanged.connect(self.update_function_preview)
        self.generate_btn.clicked.connect(self.generate_data)
        self.regress_btn.clicked.connect(self.run_regression)
        self.show_points_check.stateChanged.connect(self.update_plot)
        self.show_true_surface_check.stateChanged.connect(self.update_plot)
        self.show_ols_check.stateChanged.connect(self.update_plot)
        self.show_lasso_check.stateChanged.connect(self.update_plot)

    def update_function_preview(self):
        func_str = self.function_input.text() or self.function_str
        try:
            _, y = sp.symbols('x y')
            expr = sp.sympify(func_str)
            latex_str = sp.latex(expr)
            self.function_preview.set_latex(f"f(x, y) = {latex_str}")
        except Exception:
            self.function_preview.setText("Invalid function expression")

    def safe_eval(self, x, y):
        func_str = self.function_input.text() or self.function_str
        try:
            safe_env = {
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "log10": np.log10,
                "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi,
                "e": np.e, "x": x, "y": y
            }
            return eval(func_str, {"__builtins__": None}, safe_env)
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}")
            return np.sin(x) * np.cos(y) 

    def generate_data(self):
        self.function_str = self.function_input.text() or self.function_str
        self.noise_variance = self.noise_spin.value()
        self.num_points = self.points_spin.value()

        np.random.seed(42) 
        self.x = np.random.uniform(self.x_range[0], self.x_range[1], self.num_points)
        self.y = np.random.uniform(self.y_range[0], self.y_range[1], self.num_points)

        self.z_true = self.safe_eval(self.x, self.y)

        noise = np.random.normal(0, np.sqrt(self.noise_variance), self.num_points)
        self.z_noisy = self.z_true + noise

        self.ols_model = None
        self.lasso_model = None

        self.status_bar.showMessage(f"Generated {self.num_points} points with noise variance {self.noise_variance:.2f}")
        self.update_plot()

    def run_regression(self):
        self.poly_degree = self.degree_spin.value()
        self.lasso_alpha = self.alpha_spin.value()

        X = np.column_stack((self.x, self.y))

        poly = PolynomialFeatures(degree=self.poly_degree)
        X_poly = poly.fit_transform(X)

        self.ols_model = LinearRegression()
        self.ols_model.fit(X_poly, self.z_noisy)

        self.lasso_model = Lasso(alpha=self.lasso_alpha, max_iter=10000)
        self.lasso_model.fit(X_poly, self.z_noisy)

        grid_res = 30
        self.x_grid, self.y_grid = np.meshgrid(
            np.linspace(self.x_range[0], self.x_range[1], grid_res),
            np.linspace(self.y_range[0], self.y_range[1], grid_res)
        )

        self.z_true_surf = self.safe_eval(self.x_grid, self.y_grid)

        xy_grid = np.column_stack((self.x_grid.ravel(), self.y_grid.ravel()))
        xy_grid_poly = poly.transform(xy_grid)

        if self.ols_model:
            self.z_ols_surf = self.ols_model.predict(xy_grid_poly).reshape(self.x_grid.shape)

        if self.lasso_model:
            self.z_lasso_surf = self.lasso_model.predict(xy_grid_poly).reshape(self.x_grid.shape)

        self.status_bar.showMessage(
            f"Regression complete: Degree {self.poly_degree}, Lasso α={self.lasso_alpha:.3f}"
        )
        self.update_plot()

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')

        ax.set_facecolor('#f0f0f0')
        self.figure.set_facecolor('#f0f0f0')
        ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
        ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))

        if self.show_points_check.isChecked():
            ax.scatter(self.x, self.y, self.z_noisy, c='#ff5252', s=30, alpha=0.8, label='Noisy Points')

        if self.show_true_surface_check.isChecked() and hasattr(self, 'z_true_surf'):
            ax.plot_surface(self.x_grid, self.y_grid, self.z_true_surf,
                            alpha=0.3, color='#2196f3', edgecolor='none', label='True Surface')

        if (self.show_ols_check.isChecked() and
                self.ols_model and
                hasattr(self, 'z_ols_surf')):
            ax.plot_surface(self.x_grid, self.y_grid, self.z_ols_surf,
                            alpha=0.5, color='#4caf50', edgecolor='none', label='OLS Regression')

        if (self.show_lasso_check.isChecked() and
                self.lasso_model and
                hasattr(self, 'z_lasso_surf')):
            ax.plot_surface(self.x_grid, self.y_grid, self.z_lasso_surf,
                            alpha=0.5, color='#ff9800', edgecolor='none', label='Lasso Regression')

        ax.set_xlabel('X', fontsize=10, color='#555555')
        ax.set_ylabel('Y', fontsize=10, color='#555555')
        ax.set_zlabel('Z', fontsize=10, color='#555555')
        ax.set_title('3D Surface Regression', fontsize=12, color='#333333')

        ax.tick_params(axis='x', colors='#555555')
        ax.tick_params(axis='y', colors='#555555')
        ax.tick_params(axis='z', colors='#555555')

        ax.legend(fontsize=9)

        ax.view_init(elev=30, azim=45)

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    light_palette = QPalette()
    light_palette.setColor(QPalette.Window, QColor(219, 218, 213))  # #dbdad5
    light_palette.setColor(QPalette.WindowText, QColor(51, 51, 51))
    light_palette.setColor(QPalette.Base, QColor(240, 240, 240))
    light_palette.setColor(QPalette.AlternateBase, QColor(230, 230, 230))
    light_palette.setColor(QPalette.ToolTipBase, QColor(51, 51, 51))
    light_palette.setColor(QPalette.ToolTipText, QColor(51, 51, 51))
    light_palette.setColor(QPalette.Text, QColor(51, 51, 51))
    light_palette.setColor(QPalette.Button, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ButtonText, QColor(51, 51, 51))
    light_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    light_palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
    light_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(light_palette)

    window = SurfaceRegressionApp()
    window.show()
    sys.exit(app.exec_())