from dataclasses import dataclass
from typing import List, Dict, Tuple
from io import StringIO, TextIOWrapper
import matplotlib.pyplot as plt
import numpy as np
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
import pandas as pd

from sklearn import manifold
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

from factor_analyzer import FactorAnalyzer
import math
import random
import copy
from scipy.stats import spearmanr
from scipy.spatial import procrustes
import os
import sys
from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt, QFile, QIODevice, QObject, QRect, QSaveFile, QSize, Signal
from PySide6.QtWidgets import QApplication, QButtonGroup, QDialog,\
	QDialogButtonBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, \
	QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,  \
	QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpacerItem, \
	QSpinBox, QStatusBar, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QToolBar, \
	QVBoxLayout, QWidget
from PySide6.QtGui import QAction, QColor, QFont, QIcon,  QPalette, QKeySequence, QMouseEvent, QWheelEvent
from PySide6.QtUiTools import QUiLoader
# os.environ["QT_API"] = "pyside6"
# from qtpy.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPlainTextEdit, QScrollArea
# from qtpy.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# end of imports

class VerbositySignal(QObject):
	signal = Signal(bool)


class MyTextEditWrapper(TextIOWrapper):
	def __init__(self, text_edit):
		self.text_edit = text_edit
		super().__init__(StringIO())

	def write(self, text):
		self.text_edit.insertPlainText(text)

	def flush(self):
		pass


class ChoseOptionDialog(QDialog):
	def __init__(self, title, options_title, options, parent=None):
		super().__init__(parent)

		self.setWindowTitle(title)
		self.setFixedWidth(300)
		self.group = QVBoxLayout()
		self.button_group = QHBoxLayout()
		self.main_layout = QVBoxLayout()

		self.selected_option = None
		self.options = options

		# Create a label for the radio button group
		label = QLabel(options_title)
		self.group.addWidget(label)

		for option in options:
			rb = QRadioButton(option)
			rb.toggled.connect(self.updateSelectedOption)
			self.group.addWidget(rb)

		ok_button = QPushButton("OK")
		ok_button.clicked.connect(self.accept)
		cancel_button = QPushButton("Cancel")
		cancel_button.clicked.connect(self.reject)
		self.button_group.addWidget(ok_button)
		self.button_group.addWidget(cancel_button)

		self.main_layout.addLayout(self.group)
		self.main_layout.addLayout(self.button_group)
		self.setLayout(self.main_layout)

	def updateSelectedOption(self, checked):
		if checked:
			sender = self.sender()
			self.selected_option = self.options.index(sender.text())


class MatrixDialog(QDialog):
	def __init__(self, title, label, column_labels, row_labels, parent=None):
		super().__init__(parent)

		self.column_labels = column_labels
		self.row_labels = row_labels

		self.setWindowTitle(title)
		self.init_ui(label)

	def init_ui(self, label):
		layout = QVBoxLayout()

		instruction_label = QLabel(label)
		layout.addWidget(instruction_label)

		grid_layout = QGridLayout()

		self.spin_boxes = []

		for j, col_label in enumerate(self.column_labels):
			label = QLabel(col_label)
			grid_layout.addWidget(label, 0, j+1, QtCore.Qt.AlignmentFlag.AlignHCenter)

		for i, row_label_text in enumerate(self.row_labels):
			row_label = QLabel(row_label_text)
			grid_layout.addWidget(row_label, i+1, 0, QtCore.Qt.AlignmentFlag.AlignRight)

			row_spin_boxes = []
			for j in range(len(self.column_labels)):
				spin_box = QDoubleSpinBox()
				spin_box.setRange(-1000.0, 1000.0)
				spin_box.setValue(0.0)
				grid_layout.addWidget(spin_box, i+1, j+1)
				row_spin_boxes.append(spin_box)
			self.spin_boxes.append(row_spin_boxes)

		layout.addLayout(grid_layout)

		button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		button_box.accepted.connect(self.accept)
		button_box.rejected.connect(self.reject)

		layout.addWidget(button_box)
		self.setLayout(layout)

	def get_matrix(self):
		matrix = []
		for i in range(len(self.row_labels)):
			row = []
			for j in range(len(self.column_labels)):
				row.append(self.spin_boxes[i][j].value())
			matrix.append(row)
		return matrix


class ModifyItemsDialog(QtWidgets.QDialog):
	def __init__(self, title, items, default_values=None, parent=None):
		super().__init__(parent)
		self.setWindowTitle(title)
		self.setFixedWidth(225)
		self.items = items
		self.checkboxes = []
		self.layout = QtWidgets.QVBoxLayout()
		for item, default_value in zip(items, default_values or []):
			checkbox = QtWidgets.QCheckBox(item)
			self.checkboxes.append(checkbox)
			checkbox.setChecked(default_value)
			self.layout.addWidget(checkbox)
		self.checkbox_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
		self.checkbox_box.accepted.connect(self.accept)
		self.checkbox_box.rejected.connect(self.reject)

		checkbox_layout = QtWidgets.QHBoxLayout()
		checkbox_layout.addWidget(self.checkbox_box)
		checkbox_layout.setAlignment(QtCore.Qt.AlignHCenter)
		self.checkbox_box.layout().setContentsMargins(0, 0, 0, 0)
		self.checkbox_box.layout().setAlignment(QtCore.Qt.AlignLeft)
		self.layout.addLayout(checkbox_layout)
		self.setLayout(self.layout)

	def selected_items(self):
		selected = [self.items[i] for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]
		return selected


class ModifyTextDialog(QtWidgets.QDialog):
	def __init__(self, title, items, default_values=None, parent=None):
		super().__init__(parent)
		self.setWindowTitle(title)
		self.setFixedWidth(325)
		self.items = items
		self.spinboxes = []
		self.layout = QtWidgets.QVBoxLayout()
		for item, default_value in zip(items, default_values or []):
			hbox = QtWidgets.QHBoxLayout()
			label = QtWidgets.QLabel(item)
			spinbox = QtWidgets.QSpinBox()
			spinbox.setMinimum(0)
			spinbox.setMaximum(100)
			spinbox.setValue(default_value)
			spinbox.setAlignment(Qt.AlignRight)
			hbox.addWidget(label)
			hbox.addWidget(spinbox)

			self.spinboxes.append(spinbox)

			self.layout.addLayout(hbox)
		self.checkbox_box = QtWidgets.QDialogButtonBox(
			QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
		self.checkbox_box.accepted.connect(self.accept)
		self.checkbox_box.rejected.connect(self.reject)

		checkbox_layout = QtWidgets.QHBoxLayout()
		checkbox_layout.addWidget(self.checkbox_box)
		checkbox_layout.setAlignment(QtCore.Qt.AlignHCenter)
		self.checkbox_box.layout().setContentsMargins(0, 0, 0, 0)
		self.checkbox_box.layout().setAlignment(QtCore.Qt.AlignLeft)
		self.layout.addLayout(checkbox_layout)
		self.setLayout(self.layout)

	def selected_items(self):
		selected = [(self.items[i], self.spinboxes[i].value()) for i in range(len(self.items))]
		print(f"DEBUG -- {self.items = } {selected = }")
		return selected


class ModifyValuesDialog(QtWidgets.QDialog):
	def __init__(self, title, items, integers, default_values=None, parent=None):
		super().__init__(parent)
		self.setWindowTitle(title)
		self.setFixedWidth(325)
		self.items = items
		self.spinboxes = []
		self.layout = QtWidgets.QVBoxLayout()
		for item, default_value in zip(items, default_values or []):
			hbox = QtWidgets.QHBoxLayout()
			label = QtWidgets.QLabel(item)
			if integers:
				spinbox = QtWidgets.QSpinBox()
			else:
				spinbox = QtWidgets.QDoubleSpinBox()
			spinbox.setMinimum(0)
			spinbox.setMaximum(100)
			spinbox.setValue(default_value)
			# spinbox.setGeometry(10,20,51,22)
			spinbox.setAlignment(Qt.AlignRight)
			hbox.addWidget(spinbox)
			hbox.addWidget(label)

			self.spinboxes.append(spinbox)

			self.layout.addLayout(hbox)
		self.checkbox_box = QtWidgets.QDialogButtonBox(
			QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
		self.checkbox_box.accepted.connect(self.accept)
		self.checkbox_box.rejected.connect(self.reject)

		checkbox_layout = QtWidgets.QHBoxLayout()
		checkbox_layout.addWidget(self.checkbox_box)
		checkbox_layout.setAlignment(QtCore.Qt.AlignHCenter)
		self.checkbox_box.layout().setContentsMargins(0, 0, 0, 0)
		self.checkbox_box.layout().setAlignment(QtCore.Qt.AlignLeft)
		self.layout.addLayout(checkbox_layout)
		self.setLayout(self.layout)

	def selected_items(self):
		selected = [(self.items[i], self.spinboxes[i].value()) for i in range(len(self.items))]
		print(f"DEBUG -- {self.items = } {selected = }")
		return selected


class MoveDialog(QDialog):
	def __init__(self, title, value_title, options, parent=None):
		super().__init__(parent)

		self.setWindowTitle(title)
		self.setFixedWidth(400)
		self.group = QVBoxLayout()
		self.input_group = QHBoxLayout()
		self.button_group = QHBoxLayout()
		self.main_layout = QVBoxLayout()

		self.selected_option = None
		self.options = options
		self.decimal_input = QDoubleSpinBox()
		self.decimal_input.setRange(-sys.float_info.max, sys.float_info.max)

		for option in options:
			rb = QRadioButton(option)
			rb.toggled.connect(self.updateSelectedOption)
			self.group.addWidget(rb)

		self.input_group.addWidget(QLabel(value_title))
		self.input_group.addWidget(self.decimal_input)
		self.group.addLayout(self.input_group)

		ok_button = QPushButton("OK")
		ok_button.clicked.connect(self.accept)
		cancel_button = QPushButton("Cancel")
		cancel_button.clicked.connect(self.reject)
		self.button_group.addWidget(ok_button)
		self.button_group.addWidget(cancel_button)
		self.group.addLayout(self.button_group)

		self.main_layout.addLayout(self.group)
		self.setLayout(self.main_layout)

	def updateSelectedOption(self, checked):
		if checked:
			sender = self.sender()
			self.selected_option = self.options.index(sender.text())

	def getSelectedOption(self):
		return self.selected_option

	def getDecimalValue(self):
		return self.decimal_input.value()


class SelectItemsDialog(QtWidgets.QDialog):
	def __init__(self, title, items, parent=None):
		super().__init__(parent)
		self.setWindowTitle(title)
		self.setFixedWidth(225)
		self.items = items
		self.checkboxes = []
		self.layout = QtWidgets.QVBoxLayout()
		for item in items:
			checkbox = QtWidgets.QCheckBox(item)
			self.checkboxes.append(checkbox)
			self.layout.addWidget(checkbox)
		self.checkbox_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
		self.checkbox_box.accepted.connect(self.accept)
		self.checkbox_box.rejected.connect(self.reject)

		checkbox_layout = QtWidgets.QHBoxLayout()
		checkbox_layout.addWidget(self.checkbox_box)
		checkbox_layout.setAlignment(QtCore.Qt.AlignHCenter)
		self.checkbox_box.layout().setContentsMargins(0, 0, 0, 0)
		self.checkbox_box.layout().setAlignment(QtCore.Qt.AlignLeft)
		self.layout.addLayout(checkbox_layout)
		self.setLayout(self.layout)

	def selected_items(self):
		selected = [self.items[i] for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]
		return selected


class PairofPointsDialog(QtWidgets.QDialog):
	def __init__(self, title, items, parent=None):
		super().__init__(parent)
		self.setWindowTitle(title)
		self.setFixedWidth(225)
		self.items = items
		self.checkboxes = []
		self.layout = QtWidgets.QVBoxLayout()
		for item in items:
			checkbox = QtWidgets.QCheckBox(item)
			self.checkboxes.append(checkbox)
			self.layout.addWidget(checkbox)
		self.checkbox_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
		self.checkbox_box.accepted.connect(self.accept)
		self.checkbox_box.rejected.connect(self.reject)

		checkbox_layout = QtWidgets.QHBoxLayout()
		checkbox_layout.addWidget(self.checkbox_box)
		checkbox_layout.setAlignment(QtCore.Qt.AlignHCenter)
		self.checkbox_box.layout().setContentsMargins(0, 0, 0, 0)
		self.checkbox_box.layout().setAlignment(QtCore.Qt.AlignLeft)
		self.layout.addLayout(checkbox_layout)
		self.setLayout(self.layout)

	def selected_items(self):
		selected = [self.items[i] for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]
		if len(selected) != 2:
			raise ValueError("Exactly 2 points must be selected")
		return selected


class SetNamesDialog(QDialog):
	def __init__(self, title, label, default_names, max_chars, parent=None):
		super().__init__(parent)

		self.setWindowTitle(title)
		self.setMinimumSize(400, 200)

		# Create a layout for the dialog
		layout = QVBoxLayout(self)

		# Create a label to instruct the user
		instruction_label = QLabel(label, self)
		layout.addWidget(instruction_label)

		# Create a scroll area for the line edits
		scroll_area = QScrollArea(self)
		scroll_area.setWidgetResizable(True)
		layout.addWidget(scroll_area)

		# Create a container widget for the line edits
		container = QWidget()
		scroll_area.setWidget(container)
		line_edit_layout = QVBoxLayout(container)

		# Create a list of line edits to allow the user to input multiple names
		self.line_edits = []
		for default_name in default_names:
			line_edit = QLineEdit(self)
			line_edit.setText(default_name)
			line_edit.setMaxLength(max_chars)
			line_edit_layout.addWidget(line_edit)
			self.line_edits.append(line_edit)

		# Create a button box with OK and Cancel buttons
		button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
		button_box.accepted.connect(self.validate_and_accept)
		button_box.rejected.connect(self.reject)

		# Add the button box to the layout
		layout.addWidget(button_box)

		# Set the dialog to return the value of the line edits when accepted
		self.setResult(0)

	def getNames(self):
		# Retrieve the text of the line edits when the dialog is accepted
		return [line_edit.text() for line_edit in self.line_edits]

	def validate_and_accept(self):
		# print(f"DEBUG -- at top of validate_and_accept")
		names = self.getNames()
		# print(f"DEBUG -- {names = }")
		# set_names = set(names)
		# print(f"DEBUG -- {set_names = }")
		if len(names) != len(set(names)):
			QMessageBox.warning(
				self,
				"Duplicate Names/labels",
				"All names/labels must be distinct. Please correct the duplicate names."
			)
		else:
			self.accept()


class SetValueDialog(QDialog):
	def __init__(self, title, label, min, max, an_integer, default_value, parent=None):
		super().__init__(parent)

		self.setWindowTitle(title)
		self.setFixedSize(325, 125)

		# Create a layout for the dialog
		layout = QVBoxLayout(self)

		# Create a label and spin box to allow the user to set a value
		label = QLabel(label, self)
		if an_integer:       # == True
			self.spin_box = QSpinBox(self)
		else:
			self.spin_box = QDoubleSpinBox(self)
		self.spin_box.setFixedWidth(100)
		self.spin_box.setAlignment(Qt.AlignHCenter)
		self.spin_box.setMinimum(min)
		self.spin_box.setMaximum(max)
		self.spin_box.setValue(default_value)

		# Add the label and spin box to the layout
		layout.addWidget(label)
		layout.addWidget(self.spin_box)
		layout.setAlignment(Qt.AlignHCenter)

		# Create a button box with OK and Cancel buttons
		button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
		button_box.accepted.connect(self.accept)
		button_box.rejected.connect(self.reject)

		# Add the button box to the layout
		layout.addWidget(button_box)

		# Set the dialog to return the value of the spin box when accepted
		self.setResult(0)

	def getValue(self):
		# Retrieve the value of the spin box when the dialog is accepted
		return self.spin_box.value()


class Status(QMainWindow):
	"""Main Window."""
	def __init__(self, parent=None):
		"""Initializer."""
		super().__init__()
		qss = "QMainWindow {Background: lightblue;}"
		self.setStyleSheet(qss)
		self.basedir = os.path.dirname(__file__)
		self.verbosity_alternative = "Verbose"
		self.max_cols = 10

		# lines that follow come from old init of old class Status
		#
		self.replies_log: List[str] = []  # record of all replies from user
		self.input_file_handle: str = ""
		self.undo_stack = [0]
		# initialize undo_stack with any object to avoid warning when appending more objects
		self.undo_stack_source: List[str] = ["Initialize"]
		#
		# self.show_bisector = False
		self.width: int = 0  # had been 8 in other class
		self.decimals: int = 0  # had been 2 in other class
		self.refs_a: int = -1
		self.refs_b: int = -1
		self.value_type: str = "Unknown"
		self.connector_bisector_cross_x: int = 0
		self.connector_bisector_cross_y: int = 0
		#
		self.active = Configuration()
		#
		self.target = Configuration()
		#
		#
		# list of commands
		#
		self.commands = (
			"Alike", "Base", "Battleground", "Bisector", "Center", "Cluster", "Compare",
			"Configuration", "Contest", "Convertibles", "Core supporters",
			"Correlations", "Create", "Deactivate", "Differences", "Directions", "Distances",
			"Evaluations", "Exit", "Factor", "Grouped", "History", "Individual",
			"Invert", "Joint", "Likely supporters", "Line of Sight", "MDS",
			"Move", "Paired", "Plane", "Principal Components", "Print configuration",
			"Print target", "Print grouped data", "Print correlations", "Print similarities",
			"Print evaluations", "Ranks",
			"Reference points", "Rescale", "Rotate", "Sample designer",
			"Save configuration", "Save target", "Segment", "Settings",
			"Scores", "Scree", "Shepard", "Similarities", "Status",
			"Stress", "Target", "Terse", "Undo", "Varimax", "Vectors", "Verbose",
			"View configuration", "View target", "View grouped data", "View correlations",
			"View similarities", "View evaluations"
		)
		self.setWindowTitle("Spaces")
		#
		self.setGeometry(100, 100, 800, 600)
		#
		self.tab_widget = QTabWidget()
		self.text_edit = QTextEdit()
		self.tab_widget.setTabPosition(QTabWidget.South)
		#
		self.plot_widget = QScrollArea()
		self.tab_widget.addTab(self.plot_widget, "Plot")
		#
		self.output_widget = QPlainTextEdit()
		# self.output_widget = QScrollArea()
		self.tab_widget.addTab(self.output_widget, "Output")
		#
		self.gallery_widget = QVBoxLayout()
		self.gallery_tab = QWidget()
		self.gallery_tab.setLayout(self.gallery_widget)

		self.gallery_scroll_area = QScrollArea()
		self.gallery_scroll_area.setWidgetResizable(True)
		self.gallery_scroll_area.setWidget(self.gallery_tab)
		self.tab_widget.addTab(self.gallery_scroll_area, "Gallery")
		#
		self.log_widget = QPlainTextEdit()
		self.tab_widget.addTab(self.log_widget, "Log")
		#
		self.tab_widget.addTab(self.text_edit, "Temporary sysout")
		#
		self.setCentralWidget(self.tab_widget)

		# Set the Qt Style Sheet for the QTabWidget
		# qss
		self.tab_widget.tabBar().setStyleSheet("""
			QTabBar::tab:hover {
			font-weight: bold;
			}
		""")
		# self.tab_widget.setStyleSheet("QTabWidget::tab:hover {font-weight: bold;}")  # bold when hovering
		# self.tab_widget.setStyleSheet("QTabWidget: {Background: orange;}")
		# self.tab_widget.setStyleSheet("QTabWidget::pane {border: 1px solid #C2C7CB;}")
		self.build_traffic_dict()
		self.create_actions()
		self.create_menu_bar()
		self.create_status_bar()
		self.create_tool_bar()



	def add_plot(self, fig):
		# Add the plot to the Plot tab (replace the current plot)
		canvas_plot = FigureCanvas(fig)
		width, height = int(fig.get_size_inches()[0] * fig.dpi), int(fig.get_size_inches()[1] * fig.dpi)
		canvas_plot.setFixedSize(width, height)
		canvas_plot.draw()
		self.plot_widget.setWidget(canvas_plot)

		# Add the plot to the Gallery tab (append to existing plots)
		canvas_gallery = FigureCanvas(fig)
		canvas_gallery.setFixedSize(width, height)
		canvas_gallery.draw()
		self.gallery_widget.addWidget(canvas_gallery)

		# Add a spacer at the end to push plots to the top
		spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
		self.gallery_widget.addItem(spacer)

		plt.close(fig)

	def conf_output(self):
		title = "Configuration"
		rows = 1 + len(self.active.point_names)
		cols = len(self.active.dim_names) + 2
		table = []

		# Add header row to table
		header = [""] * 2 + self.active.dim_names[2:]
		table.append(header)

		# Add data rows to table
		for i in range(len(self.active.point_names)):
			row = []
			row.append(self.active.point_names[i])
			row.append(self.active.point_labels[i][:4])
			row += list(self.active.point_coords.loc[i, self.active.dim_names[2:]])
			table.append(row)

		# Print table to debug
		print(f"DEBUG table = {table}")

		# Write table to output and log tabs
		output_text_edit = self.output_widget.widget(1)
		output_text_edit.clear()
		output_text_edit.appendPlainText(f"\n{title}\n")
		output_text_edit.appendPlainText(tabulate(table, headers="firstrow", tablefmt="grid"))

		log_text_edit = self.output_widget.widget(3)
		log_text_edit.moveCursor(QtGui.QTextCursor.End)
		log_text_edit.insertPlainText(f"\n{title}\n")
		log_text_edit.insertPlainText(tabulate(table, headers="firstrow", tablefmt="grid"))

	def add_output(self, title):
		output_text_edit = QtWidgets.QTextEdit()
		output_text_edit.setReadOnly(True)
		output_text_edit.setStyleSheet("QTextEdit { background-color : black; color : white; }")
		self.output_widget.addTab(output_text_edit, title)
		self.output_widget.setCurrentIndex(self.output_widget.count() - 1)
		self.output_tabs.append(output_text_edit)
		if self.output_widget.count() > 10:
			self.output_widget.removeTab(0)
			self.output_tabs.pop(0)

	def create_menu_bar(self):
		#
		spaces_menu = self.menuBar()
		# Creating menus using a QMenu object

		# File menu
		file_menu = QMenu("File", self)
		spaces_menu.addMenu(file_menu)
		new_menu = file_menu.addMenu("New")
		new_menu.addAction(self.new_configuration_action)
		new_menu.addAction(self.new_grouped_action)
		new_menu.addSeparator()
		new_menu.addAction(self.new_similarities_action)
		new_menu.addAction(self.new_correlations_action)
		new_menu.addSeparator()
		new_menu.addAction(self.new_evaluations_action)

		#
		open_menu = file_menu.addMenu("Open")
		open_menu.addAction(self.open_configuration_action)
		open_menu.addAction(self.open_target_action)
		open_menu.addAction(self.open_grouped_action)
		open_menu.addSeparator()
		open_menu.addAction(self.open_similarities_action)
		open_menu.addAction(self.open_correlations_action)
		open_menu.addSeparator()
		open_menu.addAction(self.open_evaluations_action)
		open_menu.addAction(self.open_individuals_action)
		#
		save_menu = file_menu.addMenu("Save")
		save_menu.addAction(self.save_configuration_action)
		save_menu.addAction(self.save_target_action)
		#
		file_menu.addAction(self.deactivate_action)
		file_menu.addSeparator()
		#
		settings_menu = file_menu.addMenu(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_settings_icon.jpg")), "Settings")
		settings_menu.addAction(self.settings_plot_settings_action)
		settings_menu.addAction(self.settings_segment_sizing_action)
		settings_menu.addAction(self.settings_display_sizing_action)
		settings_menu.addAction(self.settings_vector_sizing_action)
		settings_menu.addAction(self.settings_layout_options_action)
		#
		print_menu = file_menu.addMenu(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_fileprint.png")), "Print")
		#file_menu.addAction(self.print_action)
		print_menu.addAction(self.print_configuration_action)
		print_menu.addAction(self.print_target_action)
		print_menu.addSeparator()
		print_menu.addAction(self.print_grouped_data_action)
		print_menu.addSeparator()
		print_menu.addAction(self.print_correlations_action)
		print_menu.addAction(self.print_similarities_action)
		print_menu.addSeparator()
		print_menu.addAction(self.print_evaluations_action)
		#
		file_menu.addAction(self.exit_action)

		# Edit menu
		edit_menu = spaces_menu.addMenu("Edit")
		edit_menu.addAction(self.undo_action)
		edit_menu.addAction(self.redo_action)

		# View menu
		view_menu = spaces_menu.addMenu("View")
		view_menu.addAction(self.view_configuration_action)
		view_menu.addAction(self.view_target_action)
		view_menu.addAction(self.view_grouped_action)
		view_menu.addAction(self.view_similarities_action)
		view_menu.addAction(self.view_correlations_action)
		view_menu.addAction(self.history_action)

		# Transform menu
		transform_menu = spaces_menu.addMenu("Transform")
		transform_menu.addAction(self.center_action)
		transform_menu.addAction(self.move_action)
		transform_menu.addAction(self.invert_action)
		transform_menu.addAction(self.rescale_action)
		transform_menu.addAction(self.rotate_action)
		transform_menu.addAction(self.compare_action)
		transform_menu.addAction(self.varimax_action)

		# Associations menu
		associations_menu = spaces_menu.addMenu("Associations")
		associations_menu.addAction(self.associations_correlations_action)
		associations_menu.addAction(self.associations_similarities_action)
		associations_menu.addAction(self.associations_los_action)
		associations_menu.addSeparator()
		associations_menu.addAction(self.associations_paired_action)
		associations_menu.addAction(self.associations_alike_action)
		associations_menu.addAction(self.associations_cluster_action)
		associations_menu.addAction(self.associations_differences_action)
		associations_menu.addAction(self.associations_distances_action)
		associations_menu.addAction(self.associations_ranks_action)
		associations_menu.addAction(self.associations_scree_action)
		associations_menu.addAction(self.associations_shepard_action)
		associations_menu.addAction(self.associations_stress_action)

		# Model menu
		model_menu = spaces_menu.addMenu("Model")
		model_menu.addAction(self.principal_components_action)
		model_menu.addAction(self.factor_analysis_action)
		mds_menu = model_menu.addMenu("Multidimensional scaling")
		mds_menu.addAction(self.mds_non_metric_action)
		mds_menu.addAction(self.mds_metric_action)
		model_menu.addSeparator()
		model_menu.addAction(self.vectors_action)
		model_menu.addAction(self.directions_action)

		# Respondents menu
		respondents_menu = spaces_menu.addMenu("Respondents")
		respondents_menu.addAction(self.evaluations_action)
		respondents_menu.addAction(self.sample_designer_action)
		respondents_menu.addAction(self.scores_action)
		respondents_menu.addAction(self.joint_action)
		respondents_menu.addSeparator()
		respondents_menu.addAction(self.reference_action)
		respondents_menu.addAction(self.contest_action)
		respondents_menu.addAction(self.segments_action)
		respondents_menu.addAction(self.core_action)
		respondents_menu.addAction(self.base_action)
		respondents_menu.addAction(self.like_action)
		respondents_menu.addAction(self.battleground_action)
		respondents_menu.addAction(self.convertible_action)
		respondents_menu.addAction(self.first_action)
		respondents_menu.addAction(self.second_action)
		# Help menu
		help_menu = spaces_menu.addMenu("Help")   # QIcon(":help-content.svg"),
		help_menu.addAction(self.help_content_action)
		help_menu.addAction(self.help_status_action)
		help_menu.addAction(self.help_verbosity_action)
		help_menu.addAction(self.about_action)

	def create_status_bar(self):

		# Create a status bar
		# self.spaces_statusbar = QtWidgets.QStatusBar()
		self.spaces_statusbar = QStatusBar()

		# Set the status bar to the main window
		self.setStatusBar(self.spaces_statusbar)

		# Show a message in the status bar
		self.spaces_statusbar.showMessage("Awaiting your command!", 40000)  # (Message, Timeout)
	#
	# In traffic_control items are in the menu order BUT one entry can handle multiple menu items
	#     thus there are fewer match cases than menu items
	#

	def build_traffic_dict(self):
		self.traffic_dict = {
			"new_configuration": lambda: self.create_command(),
			"new_grouped": lambda: self.grouped_command(),
			"new_similarities": lambda: self.similarities_command(),
			"new_correlations": lambda: self.correlations_command(),
			"new_evaluations": lambda: self.evaluations_command(),
			"open_configuration": lambda: self.configuration_command(),
			"open_target": lambda: self.target_command(),
			"open_grouped": lambda: self.grouped_command(),
			"open_similarities": lambda: self.similarities_command(),
			"open_correlations": lambda: self.correlations_command(),
			"open_evaluations": lambda: self.evaluations_command(),
			"open_individuals": lambda: self.individuals_command(),
			"save_configuration": lambda: self.save_configuration_command(),
			"save_target": lambda: self.save_target_command(),
			"deactivate": lambda: self.deactivate_command(),
			"settings_plot": lambda: self.settings_command("plot"),
			"settings_segment": lambda: self.settings_command("segment"),
			"settings_display": lambda: self.settings_command("display"),
			"settings_vector": lambda: self.settings_command("vectors"),
			"settings_layout": lambda: self.settings_command("layout"),
			"print_configuration": lambda: self.print_configuration_command(),
			"print_target": lambda: self.print_target_command(),
			"print_grouped_data": lambda: self.print_grouped_data_command(),
			"print_correlations": lambda: self.print_correlations_command(),
			"print_similarities": lambda: self.print_similarities_command(),
			"print_evaluations": lambda: self.print_evaluations_command(),
			"exit": lambda: self.done_command(),
			"undo": lambda: self.undo_command(),
			"redo": lambda: self.redo_command(),
			"view_configuration": lambda: self.view_configuration_command(),
			"view_target": lambda: self.view_target_command(),
			"view_grouped": lambda: self.view_grouped_command(),
			"view_similarities": lambda: self.view_similarities_command(),
			"view_correlations": lambda: self.view_correlations_command(),
			"history": lambda: self.history_command(),
			"center": lambda: self.center_command(),
			"move": lambda: self.move_command(),
			"invert": lambda: self.invert_command(),
			"rescale": lambda: self.rescale_command(),
			"rotate": lambda: self.rotate_command(),
			"compare": lambda: self.compare_command(),
			"varimax": lambda: self.varimax_command(),
			"correlations": lambda: self.correlations_command(),
			"similarities": lambda: self.similarities_command(),
			"paired": lambda: self.paired_command(),
			"line_of_sight": lambda: self.los_command(),
			"alike": lambda: self.alike_command(),
			"cluster": lambda: self.cluster_command(),
			"differences": lambda: self.differences_command(),
			"distances": lambda: self.distances_command(),
			"ranks": lambda: self.ranks_command(),
			"scree": lambda: self.scree_command(),
			"shepard": lambda: self.shepard_command(),
			"stress": lambda: self.stress_command(),
			"principal": lambda: self.principal_command(),
			"factor_analysis": lambda: self.factor_command(),
			"mds_non_metric": lambda:  self.mds_command(False),
			"mds_metric": lambda:  self.mds_command(True),
			"vectors": lambda: self.vectors_command(),
			"directions": lambda: self.directions_command(),
			"evaluations": lambda: self.evaluations_command(),
			"sample_designer": lambda: self.sample_designer_command(),
			"scores": lambda: self.scores_command(),
			"joint": lambda: self.joint_command(),
			"reference_points": lambda: self.reference_command(),
			"contest": lambda: self.contest_command(),
			"segments": lambda: self.segments_command(),
			"core": lambda: self.core_command(),
			"base": lambda: self.base_command(),
			"likely": lambda: self.likely_command(),
			"convertible": lambda: self.convertible_command(),
			"battleground": lambda: self.battleground_command(),
			"first": lambda: self.first_dim_command(),
			"second": lambda: self.second_dim_command(),
			"help content": lambda: self.help_command(),
			"status": lambda: self.status_command(),
			"about": lambda: self.about_command(),
			"fask": lambda: self.factor_command_sk(),
			"terse": lambda: self.terse_command(),
			"verbose": lambda: self.verbose_command(),
		}
		# print(f"DEBUG --- {len(self.traffic_dict) = }")

	def traffic_control(self, next_command):
		if next_command in self.traffic_dict:
			self.traffic_dict[next_command]()
		else:
			print("Key not found in the dictionary")

	def create_tool_bar(self):
		# Using a title
		spaces_toolbar = QToolBar("Spaces toolbar")
		spaces_toolbar.setIconSize(QSize(20, 20))
		self.addToolBar(spaces_toolbar)
		#
		new_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_filenew.png")), "New", self)
		new_button_action.setStatusTip("New")
		new_button_action.triggered.connect(lambda: self.traffic_control("new_configuration"))
		spaces_toolbar.addAction(new_button_action)

		open_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_fileopen.png")), "Open", self)
		open_button_action.setStatusTip("Open")
		open_button_action.triggered.connect(lambda: self.traffic_control("open_configuration"))
		spaces_toolbar.addAction(open_button_action)

		# open_evaluations_button_action = QAction(QIcon(os.path.join(self.basedir,
		# 	"Spaces_icons/spaces_evaluations_icon.jpg")), "Evaluations", self)
		# open_evaluations_button_action.setStatusTip("Evaluations")
		# open_evaluations_button_action.triggered.connect(lambda: self.traffic_control("open_evaluations"))
		# spaces_toolbar.addAction(open_evaluations_button_action)

		save_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_filesave.png")), "Save", self)
		save_button_action.setStatusTip("Save")
		save_button_action.triggered.connect(lambda: self.traffic_control("save_configuration"))
		spaces_toolbar.addAction(save_button_action)

		undo_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_undo_icon.jpg")), "Undo", self)
		undo_button_action.setStatusTip("Undo")
		undo_button_action.triggered.connect(lambda: self.traffic_control("undo"))
		spaces_toolbar.addAction(undo_button_action)

		redo_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_redo_icon.jpg")), "Redo", self)
		redo_button_action.setStatusTip("Redo")
		# add trigger once enabled
		redo_button_action.setEnabled(False)
		spaces_toolbar.addAction(redo_button_action)

		# printer_button_action = QAction(QIcon(os.path.join(self.basedir,
		# 	"Spaces_icons/spaces_fileprint.png")), "Print", self)
		# printer_button_action.setStatusTip("Print")
		# printer_button_action.triggered.connect(lambda: self.traffic_control("print"))
		# printer_button_action.setEnabled(False)
		# spaces_toolbar.addAction(printer_button_action)

		spaces_toolbar.addSeparator()

		center_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_center_icon.jpg")), "Center", self)
		center_button_action.setStatusTip("Center")
		center_button_action.triggered.connect(lambda: self.traffic_control("center"))
		spaces_toolbar.addAction(center_button_action)

		move_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_move_icon.png")), "Move", self)
		move_button_action.setStatusTip("Move")
		move_button_action.triggered.connect(lambda: self.traffic_control("move"))
		spaces_toolbar.addAction(move_button_action)

		invert_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_invert_icon.jpg")), "Invert", self)
		invert_button_action.setStatusTip("Invert")
		invert_button_action.triggered.connect(lambda: self.traffic_control("invert"))
		spaces_toolbar.addAction(invert_button_action)

		rescale_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_rescale_icon.jpg")), "Rescale", self)
		rescale_button_action.setStatusTip("Rescale")
		rescale_button_action.triggered.connect(lambda: self.traffic_control("rescale"))
		spaces_toolbar.addAction(rescale_button_action)

		rotate_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_rotate_icon.jpg")), "Rotate", self)
		rotate_button_action.setStatusTip("Rotate")
		rotate_button_action.triggered.connect(lambda: self.traffic_control("rotate"))
		spaces_toolbar.addAction(rotate_button_action)

		associations_scree_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_scree_icon.jpg")), "Scree diagram", self)
		associations_scree_button_action.setStatusTip("Scree diagram")
		associations_scree_button_action.triggered.connect(lambda: self.traffic_control("scree"))
		spaces_toolbar.addAction(associations_scree_button_action)

		associations_shepard_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_shepard_icon.jpg")), "Shepard diagram", self)
		associations_shepard_button_action.setStatusTip("Shepard diagram")
		associations_shepard_button_action.triggered.connect(lambda: self.traffic_control("shepard"))
		spaces_toolbar.addAction(associations_shepard_button_action)

		spaces_toolbar.addSeparator()

		mds_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_MDS_icon.jpg")), "Non-metric MultiDimensional Scaling", self)
		mds_button_action.setStatusTip("Non_metric MultiDimensional Scaling")
		mds_button_action.triggered.connect(lambda: self.traffic_control("mds_non_metric"))
		spaces_toolbar.addAction(mds_button_action)

		factor_analysis_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_factor_analysis_icon.jpg")), "Factor Analysis", self)
		factor_analysis_button_action.setStatusTip("Factor analysis")
		factor_analysis_button_action.triggered.connect(lambda: self.traffic_control("factor_analysis"))
		spaces_toolbar.addAction(factor_analysis_button_action)

		principal_components_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_pca_icon.jpg")), "Principal Components Analysis", self)
		principal_components_button_action.setStatusTip("Principal Components Analysis")
		principal_components_button_action.triggered.connect(lambda: self.traffic_control("principal"))
		spaces_toolbar.addAction(principal_components_button_action)

		spaces_toolbar.addSeparator()

		reference_points_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_reference_icon.jpg")), "Reference points", self)
		reference_points_button_action.setStatusTip("Reference points")
		reference_points_button_action.triggered.connect(lambda: self.traffic_control("reference_points"))
		spaces_toolbar.addAction(reference_points_button_action)

		contest_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_contest_icon.jpg")), "Contest", self)
		contest_button_action.setStatusTip("Contest")
		contest_button_action.triggered.connect(lambda: self.traffic_control("contest"))
		spaces_toolbar.addAction(contest_button_action)

		segments_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_segments_icon.jpg")), "Segments", self)
		segments_button_action.setStatusTip("Segments")
		segments_button_action.triggered.connect(lambda: self.traffic_control("segments"))
		spaces_toolbar.addAction(segments_button_action)

		core_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_core_icon.jpg")), "Core supporters", self)
		core_button_action.setStatusTip("Core supporters")
		core_button_action.triggered.connect(lambda: self.traffic_control("core"))
		spaces_toolbar.addAction(core_button_action)

		base_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_base_icon.jpg")), "Base supporters", self)
		base_button_action.setStatusTip("Base supporters")
		base_button_action.triggered.connect(lambda: self.traffic_control("base"))
		spaces_toolbar.addAction(base_button_action)

		likely_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_likely_icon.jpg")), "Likely supporters", self)
		likely_button_action.setStatusTip("Likely supporters")
		likely_button_action.triggered.connect(lambda: self.traffic_control("likely"))
		spaces_toolbar.addAction(likely_button_action)

		battleground_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_battleground_icon.jpg")), "Battleground", self)
		battleground_button_action.setStatusTip("Battleground")
		battleground_button_action.triggered.connect(lambda: self.traffic_control("battleground"))
		spaces_toolbar.addAction(battleground_button_action)

		convertible_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_convertible_icon.jpg")), "Convertible", self)
		convertible_button_action.setStatusTip("Convertible")
		convertible_button_action.triggered.connect(lambda: self.traffic_control("convertible"))
		spaces_toolbar.addAction(convertible_button_action)

		first_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_first_dim_icon.jpg")), "Focused on first dimension", self)
		first_button_action.setStatusTip("Focused on first dimension")
		first_button_action.triggered.connect(lambda: self.traffic_control("first"))
		spaces_toolbar.addAction(first_button_action)

		second_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_second_dim_icon.jpg")), "Focused on second dimension", self)
		second_button_action.setStatusTip("Focused on second dimension")
		second_button_action.triggered.connect(lambda: self.traffic_control("second"))
		spaces_toolbar.addAction(second_button_action)

		help_content_button_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_help_icon.jpg")), "Help content", self)
		help_content_button_action.setStatusTip("Help content")
		help_content_button_action.triggered.connect(lambda: self.traffic_control("help content"))
		spaces_toolbar.addAction(help_content_button_action)

	def create_actions(self):
		# Creating action using the first constructor
		self.new_action = QAction(self)
		self.new_action.setText("New")
		self.new_configuration_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_filenew.png")), "Configuration", self)
		self.new_grouped_action = QAction("Grouped data", self)
		self.new_similarities_action = QAction("Similarities", self)
		self.new_correlations_action = QAction("Correlations", self)
		self.new_evaluations_action = QAction("Evaluations", self)
		#
		self.open_action = QAction("Open", self)
		self.open_configuration_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_fileopen.png")), "Configuration", self)
		self.open_target_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_target_icon.jpg")), "Target", self)
		self.open_grouped_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_grouped_data_icon.jpg")), "Grouped data", self)
		self.open_similarities_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_similarities_icon.jpg")), "Similarities", self)
		self.open_correlations_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_correlations_icon.jpg")), "Correlations", self)
		self.open_evaluations_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_evaluations_icon.jpg")), "Evaluations", self)
		self.open_individuals_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_individuals_icon.jpg")), "Individuals", self)
		#
		self.save_action = QAction("Save", self)
		self.save_configuration_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_filesave.png")), "Configuration", self)
		self.save_target_action = QAction("Target", self)
		self.deactivate_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_deactivate_icon.jpg")), "Deactivate", self)
		# self.settings_action = QAction(QIcon(os.path.join(self.basedir,
		# 	"Spaces_icons/spaces_settings_icon.jpg")), "Settings", self)
		self.settings_plot_settings_action = QAction("Plot settings", self)
		self.settings_segment_sizing_action = QAction("Segment sizing", self)
		self.settings_display_sizing_action = QAction("Display sizing", self)
		self.settings_vector_sizing_action = QAction("Vector sizing", self)
		self.settings_layout_options_action = QAction("Layout options", self)
		#
		self.print_configuration_action = QAction("Configuration", self)
		self.print_target_action = QAction("Target", self)
		self.print_grouped_data_action = QAction("Grouped data", self)
		self.print_correlations_action = QAction("Correlations", self)
		self.print_similarities_action = QAction("Similarities", self)
		self.print_evaluations_action = QAction("Evaluations", self)
		#
		self.exit_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_exit_icon.jpg")), "Exit", self)
		#
		self.undo_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_undo_icon.jpg")), "Undo", self)
		self.redo_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_redo_icon.jpg")), "Redo", self)
		#
		self.view_configuration_action = QAction("Configuration", self)
		self.view_target_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_view_target_icon.jfif")), "Target", self)
		self.view_grouped_action = QAction("Grouped data", self)
		self.view_similarities_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_black_icon.jpg")), "Similarities", self)
		self.view_correlations_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_r_red_icon.jpg")), "Correlations", self)
		self.history_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_history_icon.jpg")), "History", self)
		#
		self.center_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_center_icon.jpg")), "Center", self)
		self.move_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_move_icon.png")), "Move", self)
		self.invert_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_invert_icon.jpg")), "Invert", self)
		self.rescale_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_rescale_icon.jpg")), "Rescale", self)
		self.rotate_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_rotate_icon.jpg")), "Rotate", self)
		self.compare_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_compare_icon.jpg")), "Compare", self)
		self.varimax_action = QAction("Varimax", self)
		#
		self.associations_correlations_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_correlations_icon.jpg")), "Correlations", self)
		self.associations_similarities_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_similarities_icon.jpg")), "Dis/similarities", self)
		self.associations_los_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_los_icon.jpg")), "Line of sight", self)
		self.associations_paired_action = QAction("Paired", self)
		self.associations_alike_action = QAction("Alike", self)
		self.associations_cluster_action = QAction("Cluster", self)
		self.associations_differences_action = QAction("Differences", self)
		self.associations_distances_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_distances_icon.jpg")), "Distances", self)
		self.associations_ranks_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_associations_ranks_icon.jpg")), "Ranks", self)
		self.associations_scree_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_scree_icon.jpg")), "Scree diagram", self)
		self.associations_shepard_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_shepard_icon.jpg")), "Shepard diagram", self)
		self.associations_stress_action = QAction("Contribution to stress", self)
		#
		self.principal_components_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_pca_icon.jpg")), "Principal Components", self)
		self.factor_analysis_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_factor_analysis_icon.jpg")), "Factor Analysis", self)
		self.mds_non_metric_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_MDS_icon.jpg")), "Non-metric", self)
		self.mds_metric_action = QAction("Metric", self)
		self.vectors_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_vectors_icon.jfif")), "Vectors", self)
		self.directions_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_directions_icon.jfif")), "Directions", self)
		#
		self.evaluations_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_evaluations_icon.jpg")), "Evaluations", self)
		self.sample_designer_action = QAction("Sample designer", self)
		self.scores_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_scores_icon.jpg")), "Score", self)
		self.joint_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_joint_icon.jpg")), "Joint", self)
		self.reference_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_reference_icon.jpg")), "Reference points")
		self.contest_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_contest_icon.jpg")), "Contest", self)
		self.segments_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_segments_icon.jpg")), "Segments", self)
		self.core_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_core_icon.jpg")), "Core supporters", self)
		self.base_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_base_icon.jpg")), "Base supporters", self)
		self.like_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_likely_icon.jpg")), "Likely supporters", self)
		self.battleground_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_battleground_icon.jpg")), "Battleground", self)
		self.convertible_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_convertible_icon.jpg")), "Convertible", self)
		self.first_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_first_dim_icon.jpg")), "Focused on first dimension", self)
		self.second_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_second_dim_icon.jpg")), "Focused on second dimension", self)
		#
		self.help_content_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_help_icon.jpg")), "Help Content", self)
		self.help_status_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_status_icon.jpg")),
			"Status", self)
		self.help_verbosity_action = QAction(self.verbosity_alternative, self)
		self.about_action = QAction(QIcon(os.path.join(self.basedir,
			"Spaces_icons/spaces_about_icon.jpg")), "About", self)

		# triggers
		self.new_configuration_action.triggered.connect(lambda: self.traffic_control("new_configuration"))
		self.new_grouped_action.setEnabled(False)
		self.new_similarities_action.setEnabled(False)
		self.new_correlations_action.setEnabled(False)
		self.new_evaluations_action.setEnabled(False)
		#
		self.open_configuration_action.triggered.connect(lambda: self.traffic_control("open_configuration"))
		self.open_target_action.triggered.connect(lambda: self.traffic_control("open_target"))
		self.open_grouped_action.triggered.connect(lambda: self.traffic_control("open_grouped"))
		self.open_similarities_action.triggered.connect(lambda: self.traffic_control("open_similarities"))
		self.open_correlations_action.triggered.connect(lambda: self.traffic_control("open_correlations"))
		self.open_evaluations_action.triggered.connect(lambda: self.traffic_control("open_evaluations"))
		self.open_individuals_action.triggered.connect(lambda: self.traffic_control("open_individuals"))
		#
		self.save_configuration_action.triggered.connect(lambda: self.traffic_control("save_configuration"))
		self.save_target_action.triggered.connect(lambda: self.traffic_control("save_target"))
		self.deactivate_action.triggered.connect(lambda: self.traffic_control("deactivate"))
		#
		self.settings_plot_settings_action.triggered.connect(
			lambda: self.traffic_control("settings_plot"))
		self.settings_segment_sizing_action.triggered.connect(
			lambda: self.traffic_control("settings_segment"))
		self.settings_display_sizing_action.triggered.connect(
			lambda: self.traffic_control("settings_display"))
		self.settings_vector_sizing_action.triggered.connect(
			lambda: self.traffic_control("settings_vector"))
		self.settings_layout_options_action.triggered.connect(
			lambda: self.traffic_control("settings_layout"))
		#
		self.print_configuration_action.triggered.connect(lambda: self.traffic_control("print_configuration"))
		self.print_target_action.triggered.connect(lambda: self.traffic_control("print_target"))
		self.print_grouped_data_action.triggered.connect(lambda: self.traffic_control("print_grouped_data"))
		self.print_correlations_action.triggered.connect(lambda: self.traffic_control("print_correlations"))
		self.print_similarities_action.triggered.connect(lambda: self.traffic_control("print_similarities"))
		self.print_evaluations_action.triggered.connect(lambda: self.traffic_control("print_evaluations"))
		#
		self.exit_action.triggered.connect(lambda: self.traffic_control("exit"))
		#
		self.undo_action.triggered.connect(lambda: self.traffic_control("undo"))
		self.redo_action.setEnabled(False)
		#
		self.view_configuration_action.triggered.connect(lambda: self.traffic_control("view_configuration"))
		self.view_target_action.triggered.connect(lambda: self.traffic_control("view_target"))
		self.view_grouped_action.triggered.connect(lambda: self.traffic_control("view_grouped"))
		self.view_similarities_action.triggered.connect(lambda: self.traffic_control("view_similarities"))
		self.view_correlations_action.triggered.connect(lambda: self.traffic_control("view_correlations"))
		self.history_action.triggered.connect(lambda: self.traffic_control("history"))
		#
		self.center_action.triggered.connect(lambda: self.traffic_control("center"))
		self.move_action.triggered.connect(lambda: self.traffic_control("move"))
		self.invert_action.triggered.connect(lambda: self.traffic_control("invert"))
		self.rescale_action.triggered.connect(lambda: self.traffic_control("rescale"))
		self.rotate_action.triggered.connect(lambda: self.traffic_control("rotate"))
		self.compare_action.triggered.connect(lambda: self.traffic_control("compare"))
		self.varimax_action.triggered.connect(lambda: self.traffic_control("varimax"))
		# self.varimax_action.setEnabled(False)
		#
		self.associations_correlations_action.triggered.connect(lambda: self.traffic_control("correlations"))
		self.associations_similarities_action.triggered.connect(lambda: self.traffic_control("similarities"))
		self.associations_paired_action.triggered.connect(lambda: self.traffic_control("paired"))
		self.associations_los_action.triggered.connect(lambda: self.traffic_control("line_of_sight"))
		self.associations_alike_action.triggered.connect(lambda: self.traffic_control("alike"))
		# self.associations_cluster_action.triggered.connect(lambda: self.traffic_control("cluster"))
		self.associations_cluster_action.setEnabled(False)
		# self.associations_differences_action.triggered.connect(lambda: self.traffic_control("differences"))
		self.associations_differences_action.setEnabled(False)
		self.associations_distances_action.triggered.connect(lambda: self.traffic_control("distances"))
		self.associations_ranks_action.triggered.connect(lambda: self.traffic_control("ranks"))
		self.associations_scree_action.triggered.connect(lambda: self.traffic_control("scree"))
		self.associations_shepard_action.triggered.connect(lambda: self.traffic_control("shepard"))
		self.associations_stress_action.triggered.connect(lambda: self.traffic_control("stress"))
		#
		self.principal_components_action.triggered.connect(lambda: self.traffic_control("principal"))
		self.factor_analysis_action.triggered.connect(lambda: self.traffic_control("factor_analysis"))
		self.mds_non_metric_action.triggered.connect(lambda: self.traffic_control("mds_non_metric"))
		self.mds_metric_action.triggered.connect(lambda: self.traffic_control("mds_metric"))
		self.vectors_action.triggered.connect(lambda: self.traffic_control("vectors"))
		self.directions_action.triggered.connect(lambda: self.traffic_control("directions"))
		#
		self.evaluations_action.triggered.connect(lambda: self.traffic_control("evaluations"))
		self.sample_designer_action.triggered.connect(lambda: self.traffic_control("sample_designer"))
		# self.scores_action.triggered.connect(lambda: self.traffic_control("scores"))
		self.scores_action.setEnabled(False)
		self.joint_action.triggered.connect(lambda: self.traffic_control("joint"))
		self.reference_action.triggered.connect(lambda: self.traffic_control("reference_points"))
		self.contest_action.triggered.connect(lambda: self.traffic_control("contest"))
		self.segments_action.triggered.connect(lambda: self.traffic_control("segments"))
		self.battleground_action.triggered.connect(lambda: self.traffic_control("battleground"))
		self.like_action.triggered.connect(lambda: self.traffic_control("likely"))
		self.convertible_action.triggered.connect(lambda: self.traffic_control("convertible"))
		self.core_action.triggered.connect(lambda: self.traffic_control("core"))
		self.base_action.triggered.connect(lambda: self.traffic_control("base"))
		self.first_action.triggered.connect(lambda: self.traffic_control("first"))
		self.second_action.triggered.connect(lambda: self.traffic_control("second"))
		#
		self.help_content_action.triggered.connect(lambda: self.traffic_control("help content"))
		self.help_status_action.triggered.connect(lambda: self.traffic_control("status"))
		self.help_verbosity_action.triggered.connect(lambda: self.traffic_control(self.verbosity_alternative.lower()))
		self.about_action.triggered.connect(lambda: self.traffic_control("about"))

		# Connect the signal to the slot
		self.verbosity_signal = VerbositySignal()
		self.verbosity_signal.signal.connect(self.toggle_verbosity)

	def toggle_verbosity(self, value):
		new_name = 'Terse' if value else 'Verbose'
		self.help_verbosity_action.setText(new_name)


# class Status had been here

	# def __init__(self, status_gui):

	# the lines commented out here are moved up to the bottom of the init of old class Window
	#
	# self.active = Configuration()
	#
	# -----------------------------------------------------------------------------

	def about_command(self):
		""" The About command
		"""
		#
		# Record use of About command
		#
		self.start("About")
		#
		# Explain what command does (if needed)
		#
		#
		# Handle improper order of commands
		#
		# Set variables needed
		#

		QMessageBox.about(
			self,
			"About Spaces",
			"Spaces was developed by Ed Schneider." +
			"\nIt is based on programs he developed in the 1970s as a graduate student at" +
			"\nThe University of Michigan and while consulting on the Obama 2008 campaign."
		)
		#
		self.complete("About")
		#
		return None

	# -----------------------------------------------------------------------------

	def alike_command(self):
		""" The Alike command - creates a plot with a line joining points with high similarity.
		The Alike command is used to identify pairs of points with high similarity.
		The user will be shown a histogram of similarities and will be asked for a cutoff
		value.  A plot of the configuration with be created with pairs of points with
		a similarity above (or if dissimilarities, below) the cutoff having a line
		joining the points.
		"""
		#
		# Record use of Alike command
		#
		self.start("Alike")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Alike")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Alike")
		# print("DEBUG --   in alike - {problem_detected=}")
		#
		if problem_detected:
			self.incomplete("Alike")
			return
		#
		# Set variables needed
		#
		cut = "Null"
		num_bins = 10
		#
		# Begin process
		#
		# Display histogram of similarities to assist choice of cut_point
		#
		fig = self.active.plot_cutoff(num_bins)
		self.add_plot(fig)
		self.show()
		#
		# Get user defined cut_point
		#

		title = "Set cutoff level"
		label = f"If similarities, minimum similarity to consider points alike\nIf dis/similarities, maximum dis/similarity"
		min = 0
		max = 10000
		default = 0
		an_integer = True

		app = QMainWindow()

		# Create an instance of the SetValueDialog class
		dialog = SetValueDialog(title, label, min, max, an_integer, default)

		# Show the dialog and retrieve the selected value
		result = dialog.exec()
		if result == QDialog.Accepted:
			cut_point = dialog.getValue()
			print(f"DEBUG -- Selected value: {cut_point}")
		else:
			print("DEBUG -- Dialog canceled or closed")
			self.incomplete("Alike")
			return

		if cut_point == 0:
			self.incomplete("Alike")
			return
		#
		# Echo cutoff
		#
		print("\tCut point: ", cut_point)
		#
		self.active.print_most_similar(cut_point)
		#
		# Detect when no similarities meet cutoff criteria
		#
		if len(self.active.a_x_alike) == 0:
			self.active.error("No similarity satisfies cutoff criteria.",
							  "Change cutoff criteria.")
			# self.active.suggest("Change cutoff criteria.")
			self.incomplete("Alike")
			return None
		#
		# Display plot with lines between pairs having similarity greater than
		# (or below) cutoff
		#
		fig = self.active.plot_alike()
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Alike")
		#
		return None

# -----------------------------------------------------------------------------

	def base_command(self):
		"""The Base command identifies regions defining base supporters of both reference points
		"""
		#
		# Record use of Base command
		#
		self.start("Base")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Base")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Base")
		#
		if problem_detected:
			self.incomplete("Base")
			return
		#
		# Set variables needed
		#
		reply = ""
		#
		# Begin process
		#
		# Determine whether segments of individuals have been established
		# If yes, ask user whether to show left, right, both, or neither
		#

		if self.active.have_segments():
			app = QMainWindow()
			title = "Base segments"
			options_title = "Segments in which to show individual points"
			options = ["Left", "Right", "Both", "Neither"]
			dialog = ChoseOptionDialog(title, options_title, options)
			result = dialog.exec()

			if result == QDialog.Accepted:
				selected_option = dialog.selected_option         #   + 1
				match selected_option:
					case 0:
						reply = "left"
					case 1:
						reply = "righ"
					case 2:
						reply = "both"
					case 3:
						reply = "neit"
					case _:
						print(f"DEBUG -- result is blank")
						pass
				# print(f"DEBUG -- at accepted {selected_option = } {result = } {reply = }")
			else:
				# print(f"DEBUG -- at else {result = }")
				pass
		#
		# Display plot
		#
		fig = self.active.plot_base(reply)
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Base")
		#
		return None

	# --------------------------------------------------------------------------------------------

	def bisector_command(self):
		""" bisector function - creates a perpendicular bisector to a line joining reference points.
		"""
		# def __init__(self.show_bisector):
		# 	self.show_bisector = False
		#
		# Record use of Bisector command
		#
		self.start("Bisector")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Bisector")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Bisector")
		#
		if problem_detected:
			self.incomplete("Bisector")
			return
		#
		# Display configuration in printed form
		#
		# self.active.print_active_function()
		#
		# self.active.set_line_case()   ????????????????????????????????????????????????
		#
		# Show plot of active configuration
		#
		self.active.show_bisector = True
		#
		self.active.max_and_min("Bisector")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		self.complete("Bisector")
		return

	# ----------------------------------------------------------------------------

	def center_command(self):
		""" The Center command shifts points to be centered around the origin.
			This is useful when coordinates are Lat long degrees.
		"""
		#
		# Record use of Center command
		#
		# print("DEBUG -- in Center just before calling start")
		self.start("Center")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Center")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Center")
		#
		if problem_detected:
			self.incomplete("Center")
			return
		#
		# Begin process
		#
		# Calculate average on each dimension and subtract from each coord
		#
		self.active.center()
		#
		# Display configuration in printed form
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Center")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		self.complete("Center")
		#
		return

# ----------------------------------------------------------------------------
	def cluster_command(self):
		""" The Cluster command - is not yet implemented
		"""
		#
		# Record use of Cluster command
		#
		self.start("Cluster")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Cluster")
		#
		# Handle improper order of commands
		#
		#problem_detected = self.dependencies("Cluster")
		#
		# if problem_detected:
		#	self.incomplete()
		#	return
		#
		# To be continued>>>>>>>>>>>>>>>>>>>>>>>>>>
		#
		# print("\n\tThe Cluster command has not yet been implemented")
		#
		QMessageBox.information(
			self,
			f"Command not implemented.",
			f"The Cluster command has not yet been implemented."
		)
		self.incomplete("Cluster")
		#
		return
		#
		# self.active.have_clusters = "Yes"
		self.complete("Cluster")
		return

	# ----------------------------------------------------------------------------------------------------------

	def compare_command(self):
		""" compare command - has not yet been implemented. It will be used to perform
			target rotation to orient the target rotation to the closest approximation
			of the active configuration.  It assumes the target has the same number of
			points and dimensions as the active configuration.
		"""
		#
		# Potentially (alternatively) the role of the target may be the configuration
		# read in and the active is rotated to match it.
		#
		# Record use of Compare command
		#
		self.start("Compare")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Compare")
		#
		# Handle improper order of commands
		#
		problem_detected = False
		problem_detected = self.dependencies("Compare")
		#
		if problem_detected:
			self.incomplete("Compare")
			return
		#
		# Make sure target and active have same number of dimensions and points
		#
		print(f"DEBUG -- {self.active.ndim = }")
		print(f"DEBUG -- {self.target.ndim = }")
		print(f"DEBUG -- {self.active.npoint = }")
		print(f"DEBUG -- {self.target.npoint = }")
		if not self.active.ndim == self.target.ndim:
			self.active.error(
				"Number of dimensions in target and active do not match.",
				"Choose configurations with the same number of dimensions."
			)
			self.incomplete("Compare")
			return

		if not self.active.npoint == self.target.npoint:
			self.active.error(
				"Number of points in target and active do not match.",
				"Choose configurations with the same \nnumber of points."
			)
			self.incomplete("Compare")
			return

		if self.active.ndim > 2:
			self.active.error(
				"Number of dimensions is larger than two.",
				"Current version of Compare command is \nlimited to two dimensions."
			)
		#
		# Create numpy arrays from point coords
		#
		active_in = np.array(self.active.point_coords)
		target_in = np.array(self.target.point_coords)
		#
		print(f"{active_in = }")
		print(f"{target_in = }")
		#
		# Rotate Active
		#
		active_out, target_out, disparity = procrustes(active_in, target_in)
		#
		print(f"Disparity = {disparity:8.4f}")
		# print(f"{active_out = }")
		# print(f"{target_out = }")
		self.active.point_coords = pd.DataFrame(active_out, columns=self.active.dim_labels, index=self.active.point_names)
		self.target.point_coords = pd.DataFrame(target_out, columns=self.active.dim_labels, index=self.active.point_names)

		print(f"Active configuration:")
		print(self.active.point_coords)
		print(f"Target configuration:")
		print(self.target.point_coords)

		#
		# Plot with a point "T" for target and "R" for rotated active and a line
		# between them
		#
		self.active.max_and_min("Compare")
		#
		fig = self.active.plot_compare(self.target.point_coords)
		self.add_plot(fig)
		self.show()
		#

		#
		self.complete("Compare")
		#
		return

	# ---------------------------------------------------------------------------

	def complete(self, command):

		# changes last exit code from in process to success
		#
		self.active.command_exit_code[-1] = 0
		self.spaces_statusbar.showMessage(f"Completed {command} command")
		#

		print(f"DEBUG -- in complete() {self.active.commands_used = }")
		print(f"DEBUG -- in complete() {self.active.command_exit_code = }")

	# ------------------------------------------------------------------------------------------------------

	def configuration_command(self):
		""" The Configuration command reads in a configuration to be used as the active
			configuration.
		"""
		# print("DEBUG -- At top of configuration command")
		#
		# Record use of Configuration command
		#
		self.start("Configuration")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Configuration")
		#
		# Set variables needed
		#
		self.active.hor_dim = 0
		self.active.vert_dim = 1
		self.active.ndim = 0
		self.active.npoint = 0
		self.active.dim_labels.clear()
		self.active.dim_names.clear()
		self.active.point_labels.clear()
		self.active.point_names.clear()
		self.active.point_coords = pd.DataFrame()
		self.active.bisector.case = "Unknown"
		self.active.distances.clear()
		self.active.show_bisector = False
		self.active.show_connector = False
		#
		# Get file name from dialog and handle nonexistent file names
		#
		ui_file = QFileDialog.getOpenFileName(caption="Open configuration", filter="*.txt")
		file = ui_file[0]
		#
		# Read configuration
		#
		problem_reading_file = self.active.read_configuration_function(file)

		if problem_reading_file:
			self.incomplete("Configuration")
			return
		#
		# Describe configuration read
		#
		print("\n\tConfiguration has", self.active.ndim, "dimensions and", self.active.npoint, "points\n")
		#
		# Display configuration in printed form
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Configuration")
		#
		# print("DEBUG -- just before call to plot_configuration")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		# print("DEBUG -- just after call to plot_configuration")
		#
		# Calculate inter point distances
		#
		self.active.inter_point_distances()
		#
		# text = "Hello world!\nCan we put more here?\nI wonder where it will  go?  "
		# print(f"DEBUG -- about to call add_output {text = }")
		# self.add_output(text)
		# print(f"DEBUG -- just called add_output {text = }")
		# print(f"DEBUG {self.active.point_coords = }")
		# self.conf_output()

		self.complete("Configuration")
		#
		return

		# ---------------------------------------------------------------------------

	def contest_command(self):
		""" contest command - identifies regions defined by reference points
		"""
		#
		# Record use of Contest command
		#
		self.start("Contest")
		#
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Contest")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Contest")
		#
		if problem_detected:
			self.incomplete("Contest")
			return
		#
		self.active.show_bisector = True
		#
		fig = self.active.plot_contest()
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.active.show_bisector = False
		#
		self.complete("Contest")
		#
		return None

	# -------------------------------------------------------------------------------------------------------------

	def convertible_command(self):
		""" The convertible supporters command identifies regions of opponent supporters that may be convertible
		"""
		#
		# Record use of Convertible command
		#
		self.start("Convertible")
		#
		# Explain what command does
		#
		self.active.explain("Convertible")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Convertible")
		#
		if problem_detected:
			self.incomplete("Convertible")
			return
		#
		# Set variables needed
		#
		reply = ""
		#
		# Determine whether segments of individuals have been established
		# If yes, ask user whether to show left, right, both, or settled
		#
		if self.active.have_segments():
			app = QMainWindow()
			title = "Convertible segments"
			options_title = "Segments in which to show individual points"
			options = ["Left", "Right", "Both", "Settled"]
			dialog = ChoseOptionDialog(title, options_title, options)
			result = dialog.exec()

			if result == QDialog.Accepted:
				selected_option = dialog.selected_option        # + 1
				match selected_option:
					case 0:
						reply = "left"
					case 1:
						reply = "righ"
					case 2:
						reply = "both"
					case 3:
						reply = "sett"
					case _:
						pass
				# print(f"DEBUG -- at accepted {selected_option = } {result = } {reply = }")
			else:
				# print(f"DEBUG -- at else {result = }")
				pass
		#
		fig = self.active.plot_convertible(reply)
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Convertible")
		#
		return

	# -----------------------------------------------------------------------------------------------------------

	def core_command(self):
		""" The core supporters command identifies regions defining core supporters of both reference points
		"""
		#
		# Record use of Core command
		#
		self.start("Core supporters")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Core")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Core supporters")
		#
		if problem_detected:
			self.incomplete("Core supporters")
			return
		#
		# Set variables needed
		#
		reply = ""
		#
		# Determine whether segments of individuals have been established
		# If yes, ask user whether to show left, right, both, or neither
		#
		if self.active.have_segments():
			app = QMainWindow()
			title = "Core segments"
			options_title = "Segments in which to show individual points"
			options = ["Left", "Right", "Both", "Neither"]
			dialog = ChoseOptionDialog(title, options_title, options)
			result = dialog.exec()

			if result == QDialog.Accepted:
				selected_option = dialog.selected_option          # + 1
				match selected_option:
					case 0:
						reply = "left"
					case 1:
						reply = "righ"
					case 2:
						reply = "both"
					case 3:
						reply = "neit"
					case _:
						pass
				# print(f"DEBUG -- at accepted {result = }")
			else:
				# print(f"DEBUG -- at else {result = }")
				pass
		#
		# Show plot of active configuration with care supporters shaded
		#
		fig = self.active.plot_core(reply)
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Core supporters")
		#
		return None

	# -------------------------------------------------------------------------------------------

	def correlations_command(self):
		""" The Correlations command computes correlations among evaluations.
			These can be used as similarities for multidimensional
			scaling
		"""
		# print("DEBUG -- at top of correlations_command")
		#
		# Record use of Correlations command
		#
		self.start("Correlations")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Correlations")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Correlations")
		#
		if problem_detected:
			self.incomplete("Correlations")
			return
		#
		# Set variables needed
		#
		# self.correlations.clear()   ##### needs to be updated
		# self.nreferent = 0
		# self.item_labels.clear()
		# self.item_names.clear()
		# self.value_type = "Unknown"
		self.width = 8
		self.decimals = 2
		#file = ""
		#
		# Get name of file from user and handle nonexistent file names
		#
		ui_file = QFileDialog.getOpenFileName(caption="Open correlations", filter="*.txt")
		file = ui_file[0]
		if file == "":
			self.active.error("No file selected",
							  "To establish correlations select file in dialog.")
			self.incomplete("Correlations")
			return
		#
		#
		# Call read-Lower_triangular to read file with correlations
		#
		(problem_reading_file, self.active.correlations) = self.active.read_lower_triangular(file)
		#
		# Handle any problem reading file
		#
		if problem_reading_file == "Yes":
			self.active.have_correlations = "No"
			self.active.have_similarities = "No"
			self.value_type = "Unknown"
			self.incomplete("Correlations")
			return
		#
		# Print matrix type and number of items
		#
		# print("\n\tThe correlation matrix has", self.nreferent, "items")    needs to be updated
		#
		# Call print_lower_triangle	to print matrix
		#
		# print_lower_triangle_function(                             needs to be updated
		#	decimals, self.item_labels, self.item_names,
		#	self.nreferent, self.correlations, width)
		# WARNING - DO NOT ALPHABETIZE ARGUMENT LIST - order determined in reusable function
		#
		# self.active.correlations = self.active.evaluations.corr()
		#
		# Print matrix type and number of items
		#
		# print("\n\tThe", self.active.value_type, "matrix has", self.active.nreferent, "items") ???needed?
		#
		# Call print_lower_triangle	to print correlations
		#
		self.active.print_lower_triangle(
			self.decimals, self.active.item_labels, self.active.item_names, self.active.nreferent,
			self.active.correlations, self.width)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Correlations")
		#
		return

	# -------------------------------------------------------------------------------------

	def create_command(self):
		""" The Create command is used to build the active configuration.
		"""
		#
		# Record use of Create command
		#
		self.start("Create")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Create")
		#
		# Set variables neede
		#
		need_names = True
		self.active.hor_dim = 0
		self.active.vert_dim = 1
		self.active.ndim = 0
		self.active.npoint = 0
		self.active.dim_labels.clear()
		self.active.dim_names.clear()
		self.active.point_labels.clear()
		self.active.point_names.clear()
		self.active.point_coords = pd.DataFrame()
		file = ""
		#
		settings_app = QMainWindow()
		title = "Set shape of new configuration"
		items = [
			"Set number of dimensions",
			"Set number of points"
		]
		if self.active.ndim == 0:
			self.active.ndim = 1
		if self.active.npoint == 0:
			self.active.npoint = 1
		default_values = [
			self.active.ndim,
			self.active.npoint
		]
		integers = True
		#
		dialog = ModifyValuesDialog(title, items, integers, default_values=default_values)

		# Show the dialog and retrieve the selected value

		dialog.selected_items()

		result = dialog.exec()

		print(f"DEBUG -- {dialog.selected_items = }")

		if result == QDialog.Accepted:
			value = dialog.selected_items()
			print(f"Selected value: {value}")
			print(f"DEBUG -- {value[0][1] = }")
			print(f"DEBUG -- {value[1][1] = }")
			self.active.ndim = value[0][1]
			self.active.npoint = value[1][1]
		else:
			# print("Dialog canceled or closed")
			self.incomplete("Create")
			return

		print(f"DEBUG -- {self.active.ndim = }")
		print(f"DEBUG -- {self.active.npoint = }")
		self.active.range_dims = range(self.active.ndim)
		self.active.range_points = range(self.active.npoint)

		#
		# Get dimension names
		#
		# settings_app = QMainWindow()
		# title = "Dimension names"
		# items = []
		# for each_dim in range(self.active.ndim):
		# 	items.append(str(each_dim+1))
		#
		# default_values = [
		# 	int(self.active.tolerance * 100),
		# 	int(self.active.core_tolerance * 100)
		# ]
		#
		# dialog = ModifyValuesDialog(title, items, default_values=default_values)
		#
		# # Show the dialog and retrieve the selected value
		#
		# dialog.selected_items()
		#
		# result = dialog.exec()
		#
		# print(f"DEBUG -- {dialog.selected_items = }")
		#
		# if result == QDialog.Accepted:
		# 	value = dialog.selected_items()
		# 	print(f"Selected value: {value}")
		# 	print(f"DEBUG -- {value[0][1] = }")
		# 	print(f"DEBUG -- {value[1][1] = }")
		# 	self.active.tolerance = value[0][1] / 100.0
		# 	self.active.core_tolerance = value[1][1] / 100.0
		# else:
		# 	print("Dialog canceled or closed")
		#
		# print(f"DEBUG -- {self.active.tolerance = }")
		# print(f"DEBUG -- {self.active.core_tolerance =}")

		#
		# Get dimension names
		#
		max_chars = 32
		default_names = []
		for n in self.active.range_dims:
			default_names.append("Dimension "+str(n+1))
		#
		dialog = SetNamesDialog(
			"Enter dimension names",
			"Type dimension name, use tab to advance to next dimension:",
			default_names,
			max_chars
		)
		if dialog.exec() == QDialog.Accepted:
			self.active.dim_names = dialog.getNames()
		else:
			# print("Dialog canceled or closed")
			self.incomplete("Create")
			return

		print(f"DEBUG -- {self.active.dim_names = }")
		#
		# Get dimension labels
		#
		default_labels = []
		max_chars = 4
		for n in self.active.range_dims:
			default_labels.append(self.active.dim_names[n][0:4])
		#
		dialog = SetNamesDialog(
			"Enter dimension labels, maximum four characters",
			"Type labels, use tab to advance to next label:",
			default_labels,
			max_chars
		)
		if dialog.exec() == QDialog.Accepted:
			self.active.dim_labels = dialog.getNames()
		else:
			# print("Dialog canceled or closed")
			self.incomplete("Create")
			return

		print(f"DEBUG -- {self.active.dim_labels = }")
		#
		# Get point names
		#
		max_chars = 32
		default_names = []
		for n in self.active.range_points:
			default_names.append("Point " + str(n + 1))
		#
		dialog = SetNamesDialog(
			"Enter point names",
			"Type names, use tab to advance to next name:",
			default_names,
			max_chars
		)
		if dialog.exec() == QDialog.Accepted:
			self.active.point_names = dialog.getNames()
		else:
			# print("Dialog canceled or closed")
			self.incomplete("Create")
			return

		print(f"DEBUG -- {self.active.point_names = }")
		#
		# Get point labels
		#
		default_labels = []
		max_chars = 4
		for n in self.active.range_points:
			default_labels.append(self.active.point_names[n][0:4])
		#
		dialog = SetNamesDialog(
			"Enter points labels, maximum four characters",
			"Type labels, use tab to advance to next label:",
			default_labels,
			max_chars
		)
		if dialog.exec() == QDialog.Accepted:
			self.active.point_labels = dialog.getNames()
		else:
			# print("Dialog canceled or closed")
			self.incomplete("Create")
			return

		print(f"DEBUG -- {self.active.point_labels = }")
		#
		# Get point coordinates
		#
		# Select method to get coordinates
		#
		app = QMainWindow()
		title = "Method to specify coordinates"
		options_title = "Select method"
		options = ["Random", "Ordered", "Enter values"]
		dialog = ChoseOptionDialog(title, options_title, options)
		result = dialog.exec()
		print(f"DEBUG -- {result = }")
		print(f"DEBUG -- {dialog.selected_option = }")
		if result == QDialog.Accepted:
			selected_option = dialog.selected_option        # + 1
			print(f"DEBUG -- inside if {result = }")
			print(f"DEBUG -- inside if  {dialog.selected_option = }")
			match selected_option:
				case 0:
					reply = "random"
				case 1:
					reply = "ordered"
				case 2:
					reply = "enter values"
				case _:
					print(f"DEBUG -- result is zero")
					print("Canceled")
					self.incomplete("Create")
					return
			# print(f"DEBUG -- at accepted {selected_option = } {result = } {reply = }")
		else:
			# print(f"DEBUG -- at else!!!!!!!! {result = }")
			self.incomplete("Create")
			return

		if reply == "random":
			# print(f"DEBUG -- at random")
			all_point_coords = []
			for each_point in self.active.range_points:
				a_point_coords = []
				for each_dim in self.active.range_dims:
					coord = np.random.uniform(-1.5, 1.5)
					a_point_coords.append(coord)

				all_point_coords.append(a_point_coords)

			print(f"DEBUG -- {self.active.point_coords = }")

			self.active.point_coords = pd.DataFrame(
				all_point_coords,
				index=self.active.point_names,
				columns=self.active.dim_labels
			)

		elif reply == "ordered":
			# print(f"DEBUG -- at ordered")
			print("Create ordered configuration")
			all_point_coords = []
			my_next = 1
			for each_point in self.active.range_points:
				a_point_coords = []
				for each_dim in self.active.range_dims:
					if each_point % self.active.ndim == each_dim:
						coord = float(my_next)
						if each_dim == self.active.ndim - 1:
							my_next += 1
					else:
						coord = 0.0
					a_point_coords.append(coord)
					print(f"DEBUG -- inner- {each_dim = } {each_point = } {a_point_coords = }")
				print(f"DEBUG -- outter - {each_dim = } {each_point = } {a_point_coords = }")
				all_point_coords.append(a_point_coords)

				print(f"DEBUG -- {all_point_coords = }")
			self.active.point_coords = pd.DataFrame(
				all_point_coords,
				columns=self.active.dim_names,
				index=self.active.point_names
			)

		elif reply == "enter values":
			# print(f"DEBUG -- at enter values")

			title = "Set point coordinates"
			label = f"Enter coordinate for each point on each dimension\n Use tab to advance to next coordinate"
			column_labels = self.active.dim_names
			row_labels = self.active.point_names
			#min = -10000
			#max = 10000

			app = QMainWindow()
			#
			# Create an instance of the MatrixDialog class
			#
			matrix_dialog = MatrixDialog(title, label, column_labels, row_labels)
			result = matrix_dialog.exec()
			# Show the dialog to get values
			if result == QDialog.Accepted:
				matrix = matrix_dialog.get_matrix()
				print("Matrix:")
				for row in matrix:
					print(row)
			else:
				print("Canceled")
				self.incomplete("Create")
				return
			#
			self.active.point_coords = pd.DataFrame(
				matrix, columns=self.active.dim_names, index=self.active.point_names)
		else:
			print("Canceled")
			self.incomplete("Create")
			return

		# print("\nEnter dimension names. Use done to end.")
		# while need_names:
		# 	try:
		# 		reply = self.input_source_function("Dimension name: ")
		# 		reply = reply.strip()
		# 		if len(reply) == 0:
		# 			self.active.error("Empty response.",
		# 							  "Name of dimension expected")
		# 			self.incomplete("Create")
		# 			return
		# 	#
		# 	# Handle non-recognized input
		# 	#
		# 	except (IOError, KeyboardInterrupt):
		# 		self.active.error("Invalid input.",
		# 						  "")
		# 		self.incomplete("Create")
		# 		return
		# 	#
		# 	if reply not in ("done", "Done", "DONE"):
		# 		self.active.dim_names.append(reply)
		# 	else:
		# 		need_names = False
		# #
		# self.active.ndim = len(self.active.dim_names)
		# #
		# print("\nYou have entered names for ", self.active.ndim, " dimensions")
		# self.active.range_dims = range(self.active.ndim)
		# for each_dim in self.active.range_dims:
		# 	try:
		# 		reply = self.input_source_function(
		# 			str("Enter <4 character label for " + self.active.dim_names[each_dim] + ": "))
		# 		reply = reply.lower()
		# 		reply = reply.strip()
		# 		if len(reply) == 0:
		# 			self.active.error("Empty response.",
		# 							  "Label expected")
		# 			self.incomplete("Create")
		# 			return
		# 	#
		# 	# Handle non-recognized input
		# 	#
		# 	except (IOError, KeyboardInterrupt):
		# 		self.active.error("Invalid input.",
		# 						  "Label expected")
		# 		self.incomplete("Create")
		# 		return
		# 	#
		# 	self.active.dim_labels.append(reply)
		# #
		# need_names = True
		# reply = ""
		# print("\nEnter point names. Use done to end.")
		# while need_names:
		# 	try:
		# 		reply = self.input_source_function("Point name: ")
		# 		reply = reply.strip()
		# 		if len(reply) == 0:
		# 			self.active.error("Empty response.",
		# 							  "Point name expected")
		# 			self.incomplete('Create')
		# 			return
		# 	#
		# 	# Handle non-recognized input
		# 	#
		# 	except (IOError, KeyboardInterrupt):
		# 		self.active.error("Invalid input.",
		# 						  "Point name expected")
		# 		self.incomplete("Create")
		# 		return
		# 	if reply != "done":
		# 		self.active.point_names.append(reply)
		# 	else:
		# 		need_names = False
		# #
		# self.active.npoint = len(self.active.point_names)
		# print("\nYou have entered names for ", self.active.npoint, " points")
		# self.active.range_points = range(self.active.npoint)
		# for each_point in self.active.range_points:
		# 	try:
		# 		reply = self.input_source_function(
		# 			str("Enter <4 character label for " + self.active.point_names[each_point] + ": "))
		# 		if len(reply) == 0:
		# 			self.active.error("Empty response.",
		# 							  "Point label expected")
		# 			self.incomplete("Create")
		# 			return
		# 		if len(reply) > 4:
		# 			self.active.error("Labels must be four characters or less.",
		# 							  "Enter shorter label")
		# 			self.incomplete("Create")
		# 			return
		# 	#
		# 	# Handle non-recognized input
		# 	#
		# 	except (IOError, KeyboardInterrupt):
		# 		self.active.error("Invalid input.",
		# 						  "Point label expected")
		# 		self.incomplete("Create")
		# 		return
		# 	self.active.point_labels.append(reply)
		# #
		# print("\nHow do you want to create or supply coordinates: ")
		# try:
		# 	reply = self.input_source_function("Random/Ordered/User? ")
		# 	reply = reply.lower()
		# 	reply = reply.strip()
		# 	if len(reply) == 0:
		# 		self.active.error("Empty response.",
		# 						  "")
		# 		self.incomplete("Create")
		# 		return
		# #
		# # Handle non-recognized input
		# #
		# except (IOError, KeyboardInterrupt):
		# 	self.active.error("Invalid input.",
		# 					  "")
		# 	self.incomplete("Create")
		# 	return
		# a_point_coords = []
		# if reply[0:4] == "rand":
		# 	a_point_coords = []
		# 	for each_point in self.active.range_points:
		#
		# 		for each_dim in self.active.range_dims:
		# 			coord = np.random.uniform(-1.5, 1.5)
		# 			# self.active.point_coords.iloc[each_point][each_dim] = coord
		# 			a_point_coords.append(coord)
		# 			self.active.point_coords.concat(a_point_coords)
		# 	# self.active.point_coords.append(a_point_coords)
		# 	#
		# 	#   Read in POINTS
		# 	#
		# 	#	[       part of the pandas call below
		# 	#		[
		# 	#			float(p) for p in
		# 	##			file_handle.readline().split()
		# 	#		]
		# 	###		for i in range(expected_points)
		# 	#	],
		# 	self.active.point_coords = pd.DataFrame(
		# 		index=self.active.point_names,
		# 		columns=self.active.dim_labels
		# 	)
		# #
		# elif reply[0:4] == "orde":
		# 	print("Create ordered configuration")
		# 	my_next = 1
		# 	for each_point in self.active.range_points:
		# 		# print("For ", self.active.point_names[each_point], ": ")
		# 		a_point_coords = []
		# 		for each_dim in self.active.range_dims:
		# 			if each_point % self.active.ndim == each_dim:
		# 				try:
		# 					coord = float(my_next)
		# 					if each_dim == self.active.ndim - 1:
		# 						my_next += 1
		# 				except ValueError:
		# 					self.active.error("Unexpected input",
		# 									  "Number expected")
		# 					self.incomplete("Create")
		# 					return
		# 			else:
		# 				coord = 0.0
		# 			a_point_coords.append(coord)
		# 		self.active.point_coords.concat(a_point_coords)
		# #
		# elif reply[0:4] == "user":
		# 	for each_point in self.active.range_points:
		# 		print("For ", self.active.point_names[each_point], ": ")
		# 		a_point_coords = []
		# 		for each_dim in self.active.range_dims:
		# 			reply = self.input_source_function(
		# 				str("Coordinate for " + self.active.dim_names[each_dim] + ": "))
		# 			try:
		# 				coord = float(reply)
		# 			except ValueError:
		# 				self.active.error("Unexpected input",
		# 								  "Number expected")
		# 				self.incomplete("Create")
		# 				return
		# 			a_point_coords.append(coord)
		# 		self.active.point_coords.concat(a_point_coords)
		# else:
		# 	self.active.error("Unrecognized response:",
		# 					  "Check spelling")
		# 	# self.active.suggest("Check spelling.")
		# 	self.incomplete("Create")
		# 	return
		#
		# Display configuration in printed form
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Create")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		self.complete("Create")
		#
		return

	# -----------------------------------------------------------------------------------------------

	def deactivate_command(self):
		""" The Deactivate command is used to abandon the active
			configuration, or existing similarities or correlations.
		"""
		#
		# Record use of Deactivate command
		#
		self.start("Deactivate")
		#
		# If requested explain what command does
		#
		self.active.explain("Deactivate")
		#
		# Check whether there is anything to be deactivated and if not issue error message and return
		#
		if (not self.active.have_active_configuration()) \
			and (not self.active.have_grouped_data()) \
			and (not self.active.have_similarities()) \
			and (not self.active.have_reference_points()) \
			and (not self.active.have_correlations()) \
			and (not self.active.have_evaluations()) \
			and (not self.active.have_individual_data()):
			self.active.error(
				"Nothing has been established.",
				"Something must be established before anything can be deactivated."
			)
			self.incomplete("Deactivate")
			return

		deactivate_app = QMainWindow()

		title = "Select items to deactivate"
		items = []

		if self.active.have_active_configuration():
			items.append("Active configuration")
		if self.target.have_target_configuration():
			items.append("Target")
		if self.active.have_grouped_data():
			items.append("Grouped data")
		if self.active.have_similarities():
			items.append("Similarities")
		if self.active.have_reference_points():
			items.append("Reference points")
		if self.active.have_correlations():
			items.append("Correlations")
		if self.active.have_evaluations():
			items.append("Evaluations")
		if self.active.have_individual_data():
			items.append("Individual data")

		dialog = SelectItemsDialog(title, items)

		if dialog.exec() == QtWidgets.QDialog.Accepted:
			try:
				selected_items = dialog.selected_items()
			except ValueError as error:
				QtWidgets.QMessageBox.warning(dialog, "Item:", str(error))
		else:
			self.incomplete("Deactivate")
			return

		del dialog

		items_indexes = [
			j for i in range(len(selected_items))
			for j in range(len(items))
			if selected_items[i] == items[j]
		]
		# print(f"DEBUG -- {selected_items = }")
		# print(f"DEBUG -- {items_indexes = }")
		#
		for each_item in range(len(items)):
			for checked_item in items_indexes:
				if items[checked_item] == "Active configuration":
					self.active = Configuration()
					print("\n\tActive configuration and dependent information have been abandoned.")
				if items[checked_item] == "Target":
					self.target = Configuration()
					print("\n\tTarget configuration and dependent information have been abandoned.")
				if items[checked_item] == "Grouped data":
					print(f"DEBUG -- time to deactivate grouped data")
					self.active.dim_labels_grpd.clear()
					self.active.dim_names_grpd.clear()
					self.active.point_labels_grpd.clear()
					self.active.point_names_grpd.clear()
					self.active.point_coords_grpd = pd.DataFrame()
					print("\n\tGrouped data have been abandoned.")
				if items[checked_item] == "Similarities":
					print(f"DEBUG -- time to deactivate similarities")
					self.active.similarities.clear()
					self.active.similarities_as_dict = dict()
					self.active.similarities_as_list.clear()
					self.active.similarities_as_square.clear()
					self.active.sorted_similarities = dict()
					print("\n\tSimilarities have been abandoned.")
				if items[checked_item] == "Reference points":
					print(f"DEBUG -- time to deactivate reference points")
					self.refs_a = -1
					self.refs_b = -1
					self.active.bisector = Line()
					self.active.west = Line()
					self.active.east = Line()
					self.active.connector = Line()
					self.n_likely_left = 0
					self.n_likely_right = 0
					self.seg = pd.DataFrame()
					print("\n\tReference points have been abandoned.")
				if items[checked_item] == "Correlations":
					print(f"DEBUG -- time to deactivate correlations")
					self.active.correlations = pd.DataFrame()
					print("\n\tCorrelations have been abandoned.")
				if items[checked_item] == "Evaluations":
					print(f"DEBUG -- time to deactivate evaluations")
					self.active.evaluations = pd.DataFrame()
					print("\n\tEvaluations have been abandoned.")
				if items[checked_item] == "Individual data":
					print(f"DEBUG -- time to deactivate individual datas")
					self.active.ind_vars = pd.DataFrame()
					print("\n\tIndividual data have been abandoned.")

			#
		self.set_focus_on_tab(4)
		#
		self.complete("Deactivate")
		#
		return

	# -------------------------------------------------------------------------

	def dependencies(self, command):
		#
		# print(f"At top of dependencies *******************************************")
		problem_detected = False
		n_problems = 0
		#
		lower_cmd = command.lower()
		#
		if lower_cmd in [
			"alike", "base", "bisector", "center", "cluster", "compare", "contest",
			"convertible", "core", "differences", "directions", "distances", "first dimension",
			"grouped data", "invert", "joint", "likely supporters", "battleground",
			"move", "paired", "plane", "plot", "print configuration", "ranks", "reference", "rescale",
			"rotate", "save configuration", "scores", "second dimension", "segments",
			"shepard", "stress", "varimax", "vectors", "view configuration",
			"view grouped data"
		]:
			problem_detected = self.active.needs_active(command)
			# print(f"DEBUG -- Just after test for calling needs active  {problem_detected = } ******************")
			if problem_detected:
				n_problems += 1
		if lower_cmd in [
			"alike", "mds", "paired", "print similarities", "ranks", "scree",
			"shepard", "stress", "view similarities"
		]:
			problem_detected = self.active.needs_similarities(command)
			if problem_detected:
				n_problems += 1
		if lower_cmd in [
			"base", "bisector", "contest", "convertible", "core", "first dimension",
			"likely supporters", "battleground", "second dimension", "segments"
		]:
			problem_detected = self.active.needs_reference_points(command)
			if problem_detected:
				n_problems += 1
		if lower_cmd in ["print correlations", "view correlations"]:
			problem_detected = self.active.needs_correlations(command)
			if problem_detected:
				n_problems += 1
		if lower_cmd in ["joint", "segments"]:
			problem_detected = self.active.needs_individual_data(command)
			if problem_detected:
				n_problems += 1
		if lower_cmd in ["ranks", "shepard", "stress"]:
			problem_detected = self.active.needs_distances(command)
			if problem_detected:
				n_problems += 1
		if lower_cmd in ["shepard", "stress"]:
			problem_detected = self.active.needs_ranks(command)
			if problem_detected:
				n_problems += 1
		if lower_cmd in ["factor", "principal components", "print evaluations", "line of sight"]:
			problem_detected = self.active.needs_evaluations(command)
			if problem_detected:
				n_problems += 1
		if lower_cmd in ["compare", "print target", "save target", "view target"]:
			problem_detected = self.target.needs_target(command)
			print(f"DEBUG -- {problem_detected = }")
			if problem_detected:
				n_problems += 1
		if lower_cmd in ["print grouped data", "view grouped data"]:
			problem_detected = self.active.needs_grouped_data(command)
			if problem_detected:
				n_problems += 1
		if lower_cmd in "alike":
			if not (self.active.nreferent == self.active.npoint):
				self.active.error("Number of points in active configuration differs from number of stimuli in similarities.",
						"Use Configuration or Similarities command.")
				problem_detected = True
				if problem_detected:
					n_problems += 1
		#
		# print(f"DEBUG -- At end of dependencies {n_problems =} {problem_detected =} ***********************")
		if n_problems > 0:
			return True
		else:
			return False

	# ---------------------------------------------------------------------------------

	def differences_command(self):
		""" The Differences command has not yet been implemented.
		"""
		#
		# Record use of Differences command
		#
		self.start("Differences")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Differences")
		#
		# Handle improper order of commands#
		#
		# problem_detected = self.dependencies("Differences")
		#
		# if problem_detected:
		#	self.incomplete()
		#	return
		#
		# To be continued<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		#
		# print("\n\tThe Differences command has not yet been implemented")
		QMessageBox.information(
			None,
			f"Command not implemented.",
			f"The Differences command has not yet been implemented."
		)
		self.active.have_differences = "No"
		self.incomplete("Diferences")
		return
		# Set variables needed
		#
		# See the stress command -- is this still needed or should stress
		# be replaced with this  ????????????????
		# Differences function
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Differences")
		#
		return

	# ------------------------------------------------------------------------------------------------------

	def directions_command(self):
		""" The Directions command plots the active configuration using unit length vectors.
		"""
		# print("DEBUG -- At top of directions command")
		#
		# Record use of Directions command
		#
		self.start("Directions")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Directions")
		#
		# Set variables needed
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Directions")
		#
		if problem_detected:
			self.incomplete("Directions")
			return
		#
		# Display configuration in printed form
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Directions")
		#
		# print("DEBUG -- just before call to plot_directions")
		if self.active.ndim > 1:
			fig = self.active.plot_directions()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		# print("DEBUG -- just after call to plot_directions")
		#

		#
		# text = "Hello world!\nCan we put more here?\nI wonder where it will  go?  "
		# print(f"DEBUG -- about to call add_output {text = }")
		# self.add_output(text)
		# print(f"DEBUG -- just called add_output {text = }")
		# print(f"DEBUG {self.active.point_coords = }")
		# self.conf_output()

		self.complete("Directions")
		#
		return
	# ------------------------------------------------------------------------------

	def distances_command(self):
		""" Distances command - displays a matrix of inter-point distances.
		"""
		#
		# Record use of Distances command
		#
		self.start("Distances")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Distances")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Distances")
		#
		if problem_detected:
			self.incomplete("Distances")
			return
		#
		# Set variables needed
		#
		self.active.inter_point_distances()
		# print(f"DEBUG -- {self.active.distances = }")
		# print(f"DEBUG -- {self.active.distances_as_dict = }")
		# print(f"DEBUG -- {self.active.distances_as_list = }")
		# print(f"DEBUG -- {self.active.sorted_distances = }")
		#
		width = 8
		decimals = 2
		self.active.nreferent = self.active.npoint
		#
		self.active.print_lower_triangle(
			decimals, self.active.point_labels,
			self.active.point_names, self.active.nreferent, self.active.distances, width)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Distances")
		#
		return

	# ---------------------------------------------------------------------------

	def done_command(self):

		quit()

		return

	# ---------------------------------------------------------------------------

	def evaluations_command(self):
		""" The Evaluations command reads in feeling thermometer data from csv file
		"""
		#
		# Record use of Evaluations command
		#
		# global file
		self.start("Evaluations")
		#
		# Explain what command does
		#
		self.active.explain("Evaluations")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Evaluations")
		#
		if problem_detected:
			self.incomplete("Evaluations")
			return
		#
		# Set variables needed
		#
		self.active.evaluations = pd.DataFrame()
		self.active.point_coords = pd.DataFrame()
		all_names = []
		#
		# Begin UI
		#
		ui_file = QFileDialog.getOpenFileName(caption="Open evaluations", filter="*.csv")
		file = ui_file[0]

		#
		# print(f"DEBUG -- {file = }")

		if len(file) == 0:
			self.active.error("Empty response.",
							"To establish evaluations select file in dialog.")
			self.incomplete("Evaluations")
			return

		#
		self.active.evaluations = pd.read_csv(file)
		#
		self.active.evaluate()
		#
		print("\nEvaluations: \n", self.active.evaluations)
		#
		self.active.avg_eval = self.active.evaluations.mean()
		self.active.avg_eval.sort_values(inplace=True)
		#
		fig = self.active.plot_eval()
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Evaluations")
		#
		return

	# ---------------------------------------------------------------------------

	def factor_command_sk(self):
		""" The Factor command calculates latent variables to explain the
		variation in evaluations.
		"""
		#
		# Record use of Factor command
		#
		self.start("Factor")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Factor")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Factor")
		#
		if problem_detected:
			self.incomplete("Factor")
			return
		#
		# Set variables needed
		#
		components = pd.DataFrame
		self.active.covar = pd.DataFrame()
		trans = pd.DataFrame()
		x_trans = pd.DataFrame()
		x_new = pd.DataFrame()
		transpose = pd.DataFrame()
		self.active.ndim = 0
		self.active.npoint = 0
		self.active.range_dims = range(self.active.ndim)
		self.active.range_points = range(self.active.npoint)
		self.active.dim_names = []
		self.active.dim_labels = []
		self.active.point_names = []
		self.active.point_labels = []
		self.active.point_coords = pd.DataFrame()
		self.active.distances = []
		#
		# Perform factor analysis
		#
		X = self.active.evaluations
		scaler = StandardScaler()
		print("\n\nscaler: \n", scaler)
		scaler.fit(X)
		print("\nScaler.fit(X): \n", scaler.fit(X))
		StandardScaler()
		print("\nStandardScaler(): \n", StandardScaler())
		print("\nscaler.mean_: \n", scaler.mean_)
		scaler.transform(X)
		print("\nScaler.transform(X): ", scaler.transform(X))
		#
		# X, _ = load_digits(return_X_y=True)
		transformer = FactorAnalysis(n_components=2, svd_method="lapack", copy=True, rotation="varimax", random_state=0)
		print("\n\tTransformer: ", transformer)
		# x_fit = transformer.fit(X)
		# print("\nX-fit: \n", x_fit)
		X_transformed = transformer.fit_transform(X)
		print("\nX_transformed.shape: ", X_transformed.shape)
		# print("\ncomponents_\n", transformer.components_)
		pd.set_option('display.max_columns', None)
		pd.set_option('display.precision', 2)
		pd.set_option('display.max_colwidth', 300)
		components = pd.DataFrame(transformer.components_,
					index=transformer.get_feature_names_out(), columns=self.active.item_names)
		# print("\nComponents: \n", components)

		# X_transformed.columns = transformer.get_feature_names_out() badddd
		x_trans = pd.DataFrame(X_transformed, columns=transformer.get_feature_names_out())
		print("\nX_Trans: \n", x_trans)

		print("\ntransformer.get_params(): \n", transformer.get_params())
		print("\ntransformer.get_feature_names_out(): ", transformer.get_feature_names_out())
		#  print("Get_covariance: ", transformer.get_covariance())
		self.active.covar = pd.DataFrame(transformer.get_covariance(),
					columns=self.active.item_names, index=self.active.item_names)

		print("\nCovariance: \n", self.active.covar)
		print("\nPrecision: ", transformer.get_precision())
		print("\nScore: \n", transformer.score(X))
		print("\nScore_samples: \n", transformer.score_samples(X))
		print("\n\tComponents_: \n", transformer.components_)
		print("\n\tn_features_in_: ", transformer.n_features_in_)
		print("\n\tnoise_variance_: \n", transformer.noise_variance_)
		print("\n\tmean_: \n", transformer.mean_)
		x_new = pd.DataFrame(transformer.transform(X),
					columns=transformer.get_feature_names_out())
		print("\nX_new: \n", x_new)

		transpose = components.transpose()
		print("\nTranspose: \n", transpose)
		trans = pd.DataFrame(transpose)
		print(f"\nDEBUG -- {trans = }\n")

		self.active.hor_dim = 0
		self.active.vert_dim = 1
		self.active.ndim = len(trans.columns)
		self.active.npoint = len(trans.index)
		print(f"DEBUG -- {self.active.ndim = }")
		print(f"DEBUG -- {self.active.npoint = }")
		self.active.range_dims = range(self.active.ndim)
		print(f"\nDEBUG -- {self.active.range_dims = }")
		self.active.range_points = range(self.active.npoint)
		print(f"DEBUG -- {trans.columns = }")
		print(f"DEBUG -- {trans.columns[0] = }")
		for each_dim in self.active.range_dims:
			# print(f"DEBUG -- {each_dim = }")
			self.active.dim_names.append(trans.columns[each_dim])
			self.active.dim_labels.append("FA"+str(each_dim))
		# print(f"DEBUG -- {self.active.dim_names = }")
		# print(f"DEBUG -- {self.active.dim_labels = }")
		for each_point in self.active.range_points:
			# print(f"DEBUG -- {each_point = }")
			self.active.point_names.append(trans.index[each_point])
			self.active.point_labels.append(self.active.point_names[each_point][0:4])
		self.active.rival_a = -1
		self.active.rival_b = -1
		self.active.bisector.case = "Unknown"
		self.active.bisector.direction = "Unknown"
		self.active.show_bisector = False
		self.active.show_connector = False
		self.active.distances.clear()
		self.active.point_coords = pd.DataFrame(trans)
		print("\nPoint_coords: \n", self.active.point_coords)
		#
		self.active.max_and_min("Factor")
		#
		# print("DEBUG -- just before call to plot_configuration")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		# Display scree diagram showing eigenvalues by dimensionality
		#
		# Get the eigenvector and the eigenvalues
		# ev, v = x_trans.get_eigenvalues()
		# ev, v = X_transformed.get_eigenvalues()
		# print(ev,v)

		# Plotting the scree-plot
		# xvals = range(1, x_trans.shape[1]+1)
		# plt.scatter(xvals, ev)
		# plt.plot(xvals, ev)
		# plt.title('Scree Plot')
		# plt.xlabel('Factors')
		# plt.ylabel('Eigenvalue')
		# plt.grid()
		# plt.show()
		#
		# Ask user how many dimensions to be retained
		#
		#
		# Display configuration with vectors from origin to each point
		#
		self.complete("Factor")
		#
		return
	# ---------------------------------------------------------------------------

	def factor_command(self):
		""" The Factor command calculates latent variables to explain the
		variation in evaluations.
		"""
		#
		# Record use of Factor command
		#
		self.start("Factor")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Factor")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Factor")
		#
		if problem_detected:
			self.incomplete("Factor")
			return
		#
		# Set variables needed
		#
		self.active.fa = pd.DataFrame()
		self.active.loadings = pd.DataFrame()
		self.active.eigen = pd.DataFrame()
		self.active.eigen_common = pd.DataFrame()
		self.active.commonalities = pd.DataFrame()
		self.active.factor_variance = pd.DataFrame()
		self.active.uniquenesses = pd.DataFrame()
		self.active.factor_scores = pd.DataFrame()
		#
		self.active.ndim = 0
		self.active.npoint = 0
		self.active.range_dims = range(self.active.ndim)
		self.active.range_points = range(self.active.npoint)
		self.active.dim_names = []
		self.active.dim_labels = []
		self.active.point_names = []
		self.active.point_labels = []
		self.active.point_coords = pd.DataFrame()
		self.active.distances = []
		#
		title = "Factor analysis"
		label = "Number of factors to extract:"
		default = 1
		min = 1
		max = self.active.nreferent
		an_integer = True

		app = QMainWindow()

		# Create an instance of the SetValueDialog class
		dialog = SetValueDialog(title, label, min, max, an_integer, default)

		# Show the dialog and retrieve the selected value
		result = dialog.exec()
		if result == QDialog.Accepted:
			ext_fact = dialog.getValue()
			# print(f"DEBUG -- Selected value: {ext_fact}")
		else:
			# print("DEBUG -- Dialog canceled or closed")
			self.incomplete("Factor")
			return

		if ext_fact == 0:
			self.incomplete("Factor")
			return

		#
		self.active.ndim = int(ext_fact)
		#
		if self.active.ndim == 2:
			self.active.hor_dim = 0
			self.active.vert_dim = 1
		#
		# factors and scores performs the actual factor analysis
		#
		# print(f"DEBUG -- about to call factor_and_scores {self.active.item_names = }")
		self.active.factors_and_scores()
		#
		# print(f"DEBUG -- in factor_command just after call to factors_and_scores")
		print("\nLoadings: \n", self.active.loadings)
		print("\nPoint_coords: \n", self.active.point_coords)
		self.active.show_respondent_points = False
		self.active.max_and_min("Factor")
		#
		# print(f"DEBUG - in factor_command about to call plot_vectors")
		if self.active.ndim > 1:
			fig = self.active.plot_vectors()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		print("\nEigenvalues: \n", self.active.eigen)
		print("\nCommon Factor Eigenvalues: \n", self.active.eigen_common)
		# Plotting the scree-plot
		#
		fig, ax = plt.subplots()
		#
		xvals = range(1, self.active.evaluations.shape[1]+1)
		ax.scatter(xvals, self.active.eigen)
		ax.plot(xvals, self.active.eigen)
		ax.set_title('Scree Plot')
		ax.set_xlabel('Factors')
		ax.set_ylabel('Eigenvalue')
		ax.grid()
		fig.show()

		print("\nCommonalities: \n", self.active.commonalities)
		print("\nFactor Variance:")
		print("\n\tNote: Variance is sum of squared loadings \n")
		print(self.active.factor_variance)
		print("\nUniquenesses: \n")
		print(self.active.uniquenesses)
		print("\nFactor Scores: \n")
		print(self.active.factor_scores)
		#
		# write out the factor scores
		#
		# print(f"DEBUG -- {self.active.point_labels = }")
		#
		self.active.factor_scores.to_csv("factor_scores.csv")
		#
		self.complete("Factor")
		#
		return

	# ---------------------------------------------------------------------------

	def fail(self):

		# changes last exit code from in process to 1 indicating command failed
		#
		self.active.command_exit_code[-1] = 1
		#
		# eliminate last entry to undo stack
		#
		if not len(self.undo_stack) == 1:
			del self.undo_stack[-1]
			del self.undo_stack_source[-1]
		print(f"DEBUG -- {self.active.command_exit_code = }")
		print(f"DEBUG -- {self.undo_stack = }")
		print(f"DEBUG -- {self.undo_stack_source = }")

		return

	# -----------------------------------------------------------------------------

	def first_dim_command(self):
		""" The first_dim command - identifies regions defined by the first dimension
		"""
		#
		# Record use of First dimension command
		#
		self.start("First dimension")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("First dimension")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("First dimension")
		#
		if problem_detected:
			self.incomplete("First dimension")
			return
		#
		# Set variables needed
		#
		reply = ""

		#
		# Determine whether segments of individuals have been established
		# If yes, ask user whether to show left, right, both, or neither
		#

		if self.active.have_segments():
			app = QMainWindow()
			title = "First dimension segments"
			options_title = "Segments in which to show individual points"
			options = ["Left", "Right"]
			dialog = ChoseOptionDialog(title, options_title, options)
			result = dialog.exec()

			if result == QDialog.Accepted:
				selected_option = dialog.selected_option         # + 1
				match selected_option:
					case 0:
						reply = "left"
					case 1:
						reply = "righ"
					case _:
						pass
				# print(f"DEBUG -- at accepted {selected_option = } {result = } {reply = }")
			else:
				# print(f"DEBUG -- at else {result = }")
				pass

		#
		# print("DEBUG -- in first_dim about to call plot_first")
		fig = self.active.plot_first(reply)
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("First dimension")
		#
		return None
	#
	# -----------------------------------------------------------------------------

	def grouped_command(self):
		""" The Grouped command reads a file with coordinates for a set of groups.
			The number of dimensions must be the same as the active configuration.
		"""
		#
		# Record use of Grouped command
		#
		self.start("Grouped")
		#
		# Explain what command does
		#
		self.active.explain("Grouped")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Grouped")
		#
		if problem_detected:
			self.incomplete("Grouped")
			return
		#
		# Set variables needed to get grouped data
		#
		self.active.file_handle = ""
		self.active.ndim_grpd = 0
		self.active.dim_labels_grpd.clear()
		self.active.dim_names_grpd.clear()
		self.active.point_labels_grpd.clear()
		self.active.point_codes_grpd.clear()
		self.active.point_names_grpd.clear()
		self.active.point_coords_grpd = pd.DataFrame()
		#
		ui_file = QFileDialog.getOpenFileName(caption="Open grouped data", filter="*.txt")
		file = ui_file[0]

		#
		# Ask user whether to show reference points
		#
		if self.active.have_reference_points():
			try:
				line = self.input_source_function(
					"\n\tShow reference points? Yes/No ")
				line = line.strip()
				if len(line) == 0:
					self.active.error("Empty response.",
									"")
					self.incomplete("Grouped")
					return
				lower = line.lower()
				use_refs = lower.strip()
			except IOError:
				self.active.error("Unrecognized input.",
								"")
				self.incomplete("Grouped")
				return
			# print(f"DEBUG -- in grouped command {use_refs = }")
			if use_refs == "yes":
				self.active.show_reference_points = True
				if self.active.have_bisector_info():
					#
					# Ask user whether to show bisector
					#
					try:
						line = self.input_source_function("\n\tShow bisector? Yes/No ")
						line = line.strip()
						if len(line) == 0:
							self.active.error("Empty response.",
											"")
							self.incomplete("Grouped")
							return
						lower = line.lower()
						use_bi = lower.strip()
					except IOError:
						self.active.error("Unrecognized input.",
										"")
						self.incomplete("Grouped")
						return
					if use_bi == "yes":
						self.active.show_bisector = True
					elif not use_bi == "yes":
						self.active.show_reference_points = False
						self.active.show_bisector = False
			elif not use_refs == "yes":
				# print(f"DEBUG -- in grouped command if not yes on {use_refs = }")
				self.active.show_reference_points = False
				self.active.show_bisector = False
		#
		problem_reading_file = self.active.read_grouped_data(file)
		#
		if problem_reading_file:
			self.active.error("Problem reading grouped file.",
							"Review file name and contents")
			return
		#
		# Print grouped
		#
		self.active.print_grouped_function()
		#
		# Show plot of grouped points
		#
		fig = self.active.plot_grouped()
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Grouped")
		#
		return

	# ---------------------------------------------------------------------------------------

	def have_previous_active(self):
		#
		if len(self.undo_stack) == 1:
			return False
		else:
			return True

	# ---------------------------------------------------------------------------------

	def help_command(self):
		""" The Help command has not yet been implemented.
		"""
		# print(f"DEBUG -- at top of Help command")
		#
		# Record use of Help command
		#
		self.start("Help")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Help")
		#
		# Handle improper order of commands#
		#

		#
		# print("\n\tThe Help command has not yet been implemented")
		QMessageBox.information(
			None,
			f"Command not implemented.",
			f"The Help command has not yet been implemented."
		)

		self.incomplete("Help")
		return
		# Set variables needed
		#
		# See the stress command -- is this still needed or should stress
		# be replaced with this  ????????????????
		# Differences function
		#
		self.complete("Help")
		#
		return

	# -------------------------------------------------------------------------------------

	def history_command(self):

	#
	# Record use of History command
	#
		self.start("History")
	#
	# Explain what command does (if necessary)
	#
		self.active.explain("History")
	#
	# Handle improper order of commands - none required
	#
	# Set variables needed
	#
		line = "\n\t"
		print("\n\tCommands used")
		range_commands_used = range(1, len(self.active.commands_used))
		for i in range_commands_used:
			line = line + self.active.commands_used[i] + "\t"
			if self.active.command_exit_code[i] == 0:
				line = line + "Completed successfully"
			elif i < (len(self.active.commands_used)-1):
				line = line + "Failed"
			else:
				line = line + "In process"
			print(line)
			line = "\t"
	#
		self.set_focus_on_tab(4)

		table = QTableWidget(len(self.active.commands_used), 2)
		for i in range(len(self.active.commands_used)):
			table.setItem(i, 0, QTableWidgetItem(self.active.commands_used[i]))
			status = self.active.command_exit_code[i]
			if status == 1:
				status_str = "Failed"
			elif status == 0:
				status_str = "Completed successfully"
			else:
				status_str = "In process"
			table.setItem(i, 1, QTableWidgetItem(status_str))

		# Write the table to the "output" tab, tab index 1, replacing anything now on that tab.

		output_layout = QVBoxLayout()
		# output_tab_index = self.tab_widget.indexOf(self.tab_widget.findChild(QWidget, "Output"))
		output_tab_index = 1
		if self.tab_widget.widget(output_tab_index) is not None:
			self.tab_widget.widget(output_tab_index).clear()
		output_layout.addWidget(table)
		# self.tab_widget.widget(output_tab_index).addWidget(table)

		log_layout = QVBoxLayout()
		# log_tab_index = self.tab_widget.indexOf(self.tab_widget.findChild(QWidget, "Log"))
		log_tab_index = 3
		spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
		# self.tab_widget.widget(log_tab_index).addItem(spacer)
		log_layout.addWidget(table)

	# self.editor = QPlainTextEdit()
	# self.editor.textChanged.connect(self.update_styles)
	#
	# layout = QVBoxLayout()
	# layout.addWidget(self.editor)

	# set status indicators
	#
		self.complete("History")
	#
		return

	# ---------------------------------------------------------------------------

	def incomplete(self, command):

		# changes last exit code from in process to 1 indicating command failed
		#
		self.active.command_exit_code[-1] = 1
		self.spaces_statusbar.showMessage(f"Unable to complete {command} command")
		#
		# eliminate last entry to undo stack
		#
		if not len(self.undo_stack) == 1:
			del self.undo_stack[-1]
			del self.undo_stack_source[-1]
		#
		print(f"DEBUG -- {self.active.commands_used = }")
		print(f"DEBUG -- {self.active.command_exit_code = }")
		print(f"DEBUG -- {self.undo_stack = }")
		print(f"DEBUG -- {self.undo_stack_source = }")

		return

	# ---------------------------------------------------------------------------

	def individuals_command(self):
		""" The Individuals command is used to establish scores and filters for individuals.
		"""
		#
		# Record use of Individuals command
		#
		self.start("Individuals")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Individuals")
		#
		# Handle improper order of commands - none required
		#
		# Set variables needed
		#
		cols = []
		# self.active.dim1 = []
		# self.active.dim2 = []
		self.active.hor_axis_name = "Unknown"
		self.active.n_individ = 0
		self.active.var_names.clear()
		self.active.vert_axis_name = "Unknown"
		file = ""
		self.active.ind_vars = pd.DataFrame()
		#
		var_docs = []
		problem_reading_file = False

		# Get file with individual data from user
		#
		# Get file name from user and handle nonexistent file names
		#
		ui_file = QFileDialog.getOpenFileName(caption="Open individual data", filter="*.csv")
		file = ui_file[0]
		#
		self.active.ind_vars = pd.read_csv(file)
		self.active.n_individ = self.active.ind_vars.shape[0]

		# print(f"DEBUG -- {self.active.ind_vars = }")
		#
		# Inform user file contains so many individuals
		#
		print("\n\tThe file contains ", self.active.n_individ, " individuals.")
		self.active.range_n_individ = range(self.active.n_individ)
		print("\n\tFor each individual: ")
		cols = self.active.ind_vars.columns
		for i in range(len(cols)):
			print("\t\t", cols[i])
			self.active.var_names.append(cols[i])
		#
		# print(f"DEBUG -- {self.active.var_names = }")
		#
		self.active.show_respondent_points = True
		#
		# self.active.bisector_function(self.active.rival_a, self.active.rival_b)
		#
		# self.active.set_direction_flags() - already called by bisector_function
		#
		# self.active.set_line_case()
		#
		# Select variables to define axes - currently hard wired to vars 2 and 3
		#
		self.active.hor_axis_name = self.active.ind_vars.columns[1]
		self.active.vert_axis_name = self.active.ind_vars.columns[2]
		#
		self.active.dim1 = self.active.ind_vars[self.active.hor_axis_name]
		self.active.dim2 = self.active.ind_vars[self.active.vert_axis_name]
		#
		self.active.max_and_min("Individuals")
		# self.active.bisector_function(self.active.rival_a, self.active.rival_b)

		# print(f"DEBUG - inside indi- {self.active.hor_max = }")
		if self.active.have_reference_points():
			self.active.ends_of_bisector_function()
			self.active.set_line_case()
			self.active.assign_to_segments()
		# Begin the building of the plot - why plot, why hot just read ??????????????????????????
		# print(f"DEBUG - just after assign???? - {self.active.hor_max = }")
		self.active.plot_individuals()
		#
		self.set_focus_on_tab(0)
		#
		# set status indicators
		#
		problem_reading_file = False
		#
		self.complete("Individuals")
		#
		return problem_reading_file

	# ------------------------------------------------------------------------------------------------------

	def invert_command(self):
		""" The Invert command is used to invert dimensions
		"""
		#
		# Record use of Invert command
		#
		self.start("Invert")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Invert")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Invert")
		#
		if problem_detected:
			self.incomplete("Invert")
			return
		#
		# Ask user whether to invert a dimension by displaying label and name
		#
		invert_app = QMainWindow()
		title = "Select dimensions to invert"
		dims = self.active.dim_names

		dialog = SelectItemsDialog(title, dims)

		if dialog.exec() == QtWidgets.QDialog.Accepted:
			try:
				selected_items = dialog.selected_items()
			except ValueError as error:
				QtWidgets.QMessageBox.warning(dialog, "Dimensions", str(error))
		else:
			print("Canceled")

		del dialog

		dims_indexes = [
			j for i in range(len(selected_items))
			for j in self.active.range_dims
			if selected_items[i] == self.active.dim_names[j]
		]
		# print(f"DEBUG -- {selected_dims = }")
		# print(f"DEBUG -- {dims_indexes = }")
		#
		# Multiply all points on selected dimensions to be inverted by -1
		#
		for each_dim in self.active.range_dims:
			for checked_dim in dims_indexes:
				if each_dim == checked_dim:
					self.active.invert(checked_dim)
		#
		# Multiply all points on dimension to be inverted by -1
		#
		# self.active.invert(invert_dim)
		#
		# Print inverted active configuration
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Invert")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		# Update active configuration by returning new values
		#
		self.complete("Invert")
		#
		return

	# ---------------------------------------------------------------------------------------------------

	def joint_command(self):
		""" The Joint command is used to create a plot with the reference points and individuals.
		"""
		#
		# Record use of Joint command
		#
		self.start("Joint")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Joint")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Joint")
		#
		if problem_detected:
			self.incomplete("Joint")
			return
		#
		# Set variables needed
		#
		self.active.show_just_reference_points = "No"
		self.active.show_bisector = False
		self.active.var_labels.clear()
		self.active.var_names.clear()
		self.active.nvar = 0
		#
		# Begin process
		#
		# if self.active.have_reference_points():
		# 	try:
		# 		line = self.input_source_function("\n\tShow only reference points? Yes/No ")
		# 		line = line.strip()
		# 		lower = line.lower()
		# 		just_reference_points = lower.strip()
		# 	except (IOError, KeyboardInterrupt):
		# 		self.active.error("Unrecognized input.")
		# 		self.incomplete()
		# 		return
		# 	if len(line) == 0:
		# 		self.active.error("Empty response.")
		# 		self.incomplete()
		# 		return
		# 	#
		# 	if self.active.have_bisector_info():
		# 		try:
		# 			line = self.input_source_function("\n\tShow bisector? Yes/No ")
		# 			line = line.strip()
		# 			line = line.lower()
		# 			if len(line) == 0:
		# 				self.active.error("Empty response.")
		# 				self.incomplete()
		# 				return
		# 		#
		# 		# Handle non-recognized input
		# 		#
		# 		except (IOError, KeyboardInterrupt):
		# 			self.active.error("Invalid input.")
		# 			self.incomplete()
		# 			return
		# 		#
		# 		if line == "yes":
		# 			self.active.show_bisector = "Yes"
		# 		if line == "no":
		# 			self.active.show_bisector = "No"
		# 	#
		# 	# Add code to plot bisector if requested ????
		# 	#
		# 	if just_reference_points == "yes":
		# 		self.active.show_just_reference_points = "Yes"
		#
		self.active.max_and_min("Joint")
		fig = self.active.plot_joint()
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Joint")
		#
		return

	# --------------------------------------------------------------------------------

	def likely_command(self):
		""" The Likely supporters command identifies regions defining likely supporters of both reference points
		"""
		#
		# Record use of Likely supporters command
		#
		self.start("Likely supporters")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Likely supporters")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Likely supporters")
		#
		if problem_detected:
			self.incomplete("Likely")
			return

		reply = ""
		#
		# Determine whether segments of individuals have been established
		# If yes, ask user whether to show left, right, both, or neither
		#

		if self.active.have_segments():
			app = QMainWindow()
			title = "Likely segments"
			options_title = "Segments in which to show individual points"
			options = ["Left", "Right", "Both"]
			dialog = ChoseOptionDialog(title, options_title, options)
			result = dialog.exec()

			# print(f"DEBUG - {result = }")

			if result == QDialog.Accepted:
				selected_option = dialog.selected_option           # + 1
				match selected_option:
					case 0:
						reply = "left"
					case 1:
						reply = "righ"
					case 2:
						reply = "both"
					case _:
						# print(f"DEBUG -- result is zero")
						pass
				# print(f"DEBUG -- at accepted {result = }")
			else:
				# print(f"DEBUG -- at else {result = }")
				pass
		# print(f"DEBUG -- {reply = }")
		#
		self.active.max_and_min("Likely")
		#
		fig = self.active.plot_likely(reply)
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Likely")
		#
		return None

	# --------------------------------------------------------------------------------------------------

	def los_command(self):
		"""The Line of Sight command computes the line of sight measure of association
		"""
		#
		# Record use of Line of Sight command
		#
		self.start("Line of Sight")

		print(f"DEBUG-- at top of Line of Sight command")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Line of Sight")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Line of Sight")
		#
		if problem_detected:
			self.incomplete("Line of Sight")
			return
		#
		# Set variables needed
		width = 8
		decimals = 1
		#
		self.active.los()
		#
		self.active.print_lower_triangle(decimals, self.active.item_labels, self.active.item_names,
									self.active.nreferent, self.active.similarities, width)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Line of Sight")
		#
		return

	# -----------------------------------------------------------------------------------

	def battleground_command(self):
		""" The Battleground command creates a plot with a lines
			delineating area with battleground individuals.
		"""
		#
		# Record use of Battleground command
		#
		self.start("Battleground")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Battleground")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Battleground")
		#
		if problem_detected:
			self.incomplete("Battleground")
			return
		#
		# Set variables needed
		#
		reply = ""
		#
		# Determine whether segments of individuals have been established
		# If yes, ask user whether to show battleground, settled, both, or neither
		#
		if self.active.have_segments():
			app = QMainWindow()
			title = "Battleground segments"
			options_title = "Segments in which to show individual points"
			options = ["Battleground", "Settled"]
			dialog = ChoseOptionDialog(title, options_title, options)
			result = dialog.exec()

			# print(f"DEBUG - {result = }")

			if result == QDialog.Accepted:
				selected_option = dialog.selected_option           # + 1
				match selected_option:
					case 0:
						reply = "batt"
					case 1:
						reply = "sett"
					case _:
						# print(f"DEBUG -- result is zero")
						pass
				# print(f"DEBUG -- at accepted {result = }")
			else:
				# print(f"DEBUG -- at else {result = }")
				pass
		#
		# Display active configuration showing battleground region
		#
		# print(f"DEBUG -- {reply = }")
		fig = self.active.plot_battleground(reply)
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Battleground")
		#
		return None

	# ----------------------------------------------------------------------------------------------

	def mds_command(self, metric_switch):
		"""The MDS command performs non-metric multidimensional scaling on the active configuration.
			An initial configuration and similarities have to have been established.
		"""
		#
		# Record use of MDS command
		#
		self.start("MDS")
		print(f"DEBUG -- {metric_switch = }")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("MDS")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("MDS")
		#
		if problem_detected:
			self.incomplete("MDS")
			return
		#
		# Define needed variables
		#
		self.active.best_stress = -1
		self.active.use_metric = metric_switch
		#

		#
		# Get degrees to rotate configuration
		#
		title = "Components to extract"
		label = "Set number of components to extract"
		min = 1
		max = 10
		default = 2
		an_integer = True

		app = QMainWindow()

		# Create an instance of the SetValueDialog class
		dialog = SetValueDialog(title, label, min, max, an_integer, default)

		# Show the dialog and retrieve the selected value
		result = dialog.exec()
		if result == QDialog.Accepted:
			self.active.n_comp = dialog.getValue()
			print(f"DEBUG -- Selected value: {self.active.n_comp}")
			print(f"DEBUG -- {self.active.npoint = }")
		else:
			# print("DEBUG -- Dialog canceled or closed")
			self.incomplete("MDS")
			return

		if self.active.n_comp == 0:
			self.incomplete("MDS")
			return
		#
		print(f"DEBUG -- {self.active.n_comp = }")
		if self.active.ndim == 0:
			self.active.ndim = self.active.n_comp
		print(f"DEBUG -- in mds_command {self.active.point_names = }")
		self.active.mds()
		#
		# Show active configuration
		#

		self.active.print_active_function()
		#
		self.active.max_and_min("MDS")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		#
		# write out the mds scores
		#

		#
		# self.active.mds_scores.to_csv("mds_scores.csv")
		self.complete("MDS")
		#
		return

	#

	# ------------------------------------------------------------------------------------

	def move_command(self):
		""" The Move command allows the used to add
			or subtract a constant form all points on one or more dimensions.
		"""
		#
		# Record use of Move command
		#
		self.start("Move")
		#
		# Explain what command does
		#
		self.active.explain("Move")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Move")
		#
		if problem_detected:
			self.incomplete("Move")
			return
		#
		self.active.bisector.case = "Unknown"
		#
		# Get parameters of configuration
		#
		# Ask user which dimension they want to move
		#

		app = QMainWindow()
		title = "Select dimension and value for move"
		value_title = "Value to add to all points on this dimension"
		options = self.active.dim_names
		dialog = MoveDialog(title, value_title, options)
		result = dialog.exec()

		# print(f"DEBUG -- {dialog.ok_button = }")

		if result == QDialog.Accepted:
			selected_option = dialog.selected_option
			decimal_value = dialog.getDecimalValue()

			# print(f"Accepted: Selected option: {selected_option}, Decimal value: {decimal_value}")
		elif result == QDialog.Rejected:
			# print(f"Rejected")
			self.incomplete("Move")
			return

		if decimal_value == 0.0:
			self.incomplete("Move")
			return

		self.active.move(selected_option, decimal_value)
		#
		# Update active configuration by returning new values
		#
		# Print moved active configuration
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Move")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		# Update active configuration by returning new values
		#
		self.complete("Move")
		#
		return

	# -----------------------------------------------------------------------------

	def paired_command(self):
		""" paired function - The paired command is used to get the interpoint distance and/or similarity
			for pairs of points.
		"""
		#
		# Record use of Paired command
		#
		self.start("Paired")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Paired")
		#
		# Handle improper order of commands
		#
		# print(f"DEBUG -- {self.active.have_similarities() = }")
		# print(f"DEBUG -- {self.active.have_distances() = }")

		problem_detected = self.dependencies("Paired")
		#
		if problem_detected:
			self.incomplete("Paired")
			return
		#
		# Set variables needed
		#
		first_index = -1
		second_index = -1
		#

		pair_app = QMainWindow()

		title = "Select pair of points"
		items = self.active.point_names

		dialog = PairofPointsDialog(title, items)

		while True:
			if dialog.exec() == QtWidgets.QDialog.Accepted:
				try:
					selected_items = dialog.selected_items()
					break
				except ValueError as error:
					QtWidgets.QMessageBox.warning(dialog, "Points", str(error))
			else:
				print("Canceled")
				break

		del dialog

		pair_indexes = [
			j for i in range(2)
			for j in self.active.range_points
			if selected_items[i] == self.active.point_names[j]
		]
		# print(f"DEBUG -- {selected_items = }")
		# print(f"DEBUG -- {refs_indexes = }")
		first_index = pair_indexes[0]
		second_index = pair_indexes[1]

		#
		# Make sure first_index is less than second_index, if not invert
		#
		if first_index > second_index:
			first_index, second_index = second_index, first_index
		#
		print("\n\tFirst point: ", self.active.point_labels[first_index], self.active.point_names[first_index])
		print("\tSecond point: ", self.active.point_labels[second_index], self.active.point_names[second_index])
		key = str(self.active.point_labels[first_index] + "_" + self.active.point_labels[second_index])
		#
		print("\tSimilarity: ", self.active.similarities_as_square[first_index][second_index])
		print("\tDistance: ", self.active.distances_as_dict[key])
		# print(f"DEBUG -- {self.active.distances_as_list = }")
		# print(f"DEBUG -- {self.active.distances_as_list[2][4] = }")
		# print("\tDistance: ", self.active.distances_as_list[first_index][second_index])
		self.set_focus_on_tab(4)
		#
		self.complete("Paired")
		#
		return None

	# ----------------------------------------------------------------------------

	def plane_command(self):
		""" The Plane command allows the user to specify which dimensions
			to use for the horizontal and vertical axes.
			In GUI this can be done only in settings
		"""
		#
		# Record use of Plane command
		#
		self.start("Plane")
		# Explain what command does (if necessary)
		#
		self.active.explain("Plane")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Plane")
		#
		if problem_detected:
			self.incomplete("Plane")
			return
		#
		# Set variables needed
		#
		hor_orig = self.active.hor_dim
		self.active.hor_dim = -1
		vert_orig = self.active.vert_dim
		self.active.vert_dim = -1
		#
		# Show dimensions in active configuration
		#
		print("\n\tDimensions in active configuration: ")
		for index, each_dim in enumerate(self.active.dim_labels):
			print("\t\t", each_dim, self.active.dim_names[index])
		#
		# Ask user which dimensions should define plane to be used in plots
		#
		# Establish horizontal dimension
		#

		ui_file_name = "establish_plane.ui"    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		ui_file = QFile(ui_file_name)
		if not ui_file.open(QIODevice.ReadOnly):
			print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
			sys.exit(-1)
		loader = QUiLoader()
		establish_plane_file_dialog = loader.load(ui_file)
		ui_file.close()
		if not establish_plane_file_dialog:
			print(loader.errorString())
			sys.exit(-1)
		establish_plane_file_dialog.show()
		establish_plane_file_dialog.exec()
		#
		self.active.hor_dim = (establish_plane_file_dialog.spinBox.value() - 1)
		self.active.vert_dim = (establish_plane_file_dialog.spinBox_2.value() - 1)

		# try:
		# 	hor = self.input_source_function("\n\tEnter label of horizontal dimension: ")
		# 	hor = hor.strip()
		# 	if len(hor) == 0:
		# 		self.active.error("Empty response.")
		# 		self.incomplete()
		# 		return
		# #
		# # Handle non-recognized input
		# #
		# except (IOError, KeyboardInterrupt):
		# 	self.active.error("Unexpected input.")
		# 	self.incomplete()
		# 	return
		# #
		# for index, each_dim in enumerate(self.active.dim_labels):
		# 	if hor == each_dim:
		# 		self.active.hor_dim = index
		# #
		# if self.active.hor_dim == -1:
		# 	self.active.error(f" {hor} is not a dimension label")
		# 	self.active.suggest("Case sensitivity may be the issue.")
		# 	#
		# 	self.active.hor_dim = hor_orig
		# 	self.active.vert_dim = vert_orig
		# 	self.incomplete()
		# 	return
		# #
		# # Establish vertical dimension
		# #
		# try:
		# 	vert = self.input_source_function("\n\tEnter label of vertical dimension: ")
		# 	vert = vert.strip()
		# 	if len(vert) == 0:
		# 		self.active.error("Empty response.")
		# 		self.incomplete()
		# 		return
		# #
		# # Handle non-recognized input
		# #
		# except (IOError, KeyboardInterrupt):
		# 	self.active.error("Unexpected input.")
		# 	self.incomplete()
		# 	return
		# #
		# for index, each_dim in enumerate(self.active.dim_labels):
		# 	if vert == each_dim:
		# 		self.active.vert_dim = index
		# #
		# if self.active.vert_dim == -1:
		# 	self.active.error(f" {hor} is not a dimension label")
		# 	self.active.suggest("Case sensitivity may be the issue.")
		# 	#
		# 	self.active.hor_dim = hor_orig
		# 	self.active.vert_dim = vert_orig
		# 	self.incomplete()
		# 	return
		#
		print("\n\tPlots will show a plane with: ")
		print("\t\t", "Horizontal dimension: ", self.active.dim_labels[self.active.hor_dim],
			self.active.dim_names[self.active.hor_dim].strip())
		print("\t\t", "Vertical dimension: ", self.active.dim_labels[self.active.vert_dim],
			self.active.dim_names[self.active.vert_dim].strip())
		#
		self.active.max_and_min("Plane")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		self.complete("Plane")
		#
		return

	# ---------------------------------------------------------------------------
	#
	# def plot_command(self):
	# 	""" The Plot command is used to obtain a plot of the active configuration.
	# 	"""
	# 	#
	# 	# Record use of Plot command
	# 	#
	# 	self.start("Plot")
	# 	#
	# 	# Explain what command does (if necessary)
	# 	#
	# 	self.active.explain("Plot")
	# 	#
	# 	# Handle improper order of commands
	# 	#
	# 	problem_detected = self.dependencies("Plot")
	# 	#
	# 	if problem_detected:
	# 		self.incomplete("Plot")
	# 		return
	# 	#
	# 	# Display configuration in printed form
	# 	#
	# 	self.active.print_active_function()
	# 	#
	# 	# Show plot of active configuration using plane defined in plane
	# 	#
	# 	self.active.max_and_min("Plot")
	# 	if self.active.ndim > 1:
	# 		fig = self.active.plot_configuration()
	# 		self.add_plot(fig)
	# 		self.show()
	# 		self.set_focus_on_tab(0)
	# 	#
	# 	self.complete("Plot")
	# 	#
	# 	return None

	# ------------------------------------------------------------------------------------------------------

	def principal_command(self):
		""" The Principal components command
		"""
		#
		# Record use of Principal components command
		#
		self.start("Principal components")
		#
		# Explain what command does ( if necessary)
		#
		self.active.explain("Principal components")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Principal components")
		#
		if problem_detected:
			self.incomplete("Principal components")
			return
		#
		self.active.pca_covar = pd.DataFrame()
		trans = pd.DataFrame()
		self.active.ndim = 0
		self.active.npoint = 0
		self.active.range_dims = range(self.active.ndim)
		self.active.range_points = range(self.active.npoint)
		self.active.dim_names = []
		self.active.dim_labels = []
		self.active.point_names = []
		self.active.point_labels = []
		#
		# Perform principal components analysis
		#
		X_pca = self.active.evaluations

		pca_transformer = PCA(n_components=2, copy=True, random_state=0)
		print("\n\t", pca_transformer)
		X_pca_transformed = pca_transformer.fit_transform(X_pca)
		print("X_pca_transformed.shape: ", X_pca_transformed.shape)

		pd.set_option('display.max_columns', None)
		pd.set_option('display.precision', 2)
		pd.set_option('display.max_colwidth', 300)
		components = pd.DataFrame(pca_transformer.components_,
			index=pca_transformer.get_feature_names_out(), columns=self.active.item_names)

		x_pca_trans = pd.DataFrame(X_pca_transformed, columns=pca_transformer.get_feature_names_out())
		print("X_pca_Trans: \n", x_pca_trans)
		print("pca_transformer.get_params(): \n", pca_transformer.get_params())
		print("pca_transformer.get_feature_names_out(): ", pca_transformer.get_feature_names_out())
		#  print("Get_covariance: ", transformer.get_covariance())
		self.active.pca_covar = pd.DataFrame(pca_transformer.get_covariance(),
					columns=self.active.item_names, index=self.active.item_names)

		print("PCA Covariance: \n", self.active.pca_covar)

		transpose = components.transpose()
		print("\nTranspose: \n", transpose)
		trans = pd.DataFrame(transpose)
		# print(f"\nDEBUG -- {trans = }\n")
		self.active.hor_dim = 0
		self.active.vert_dim = 1
		self.active.ndim = len(trans.columns)
		self.active.npoint = len(trans.index)
		# print(f"DEBUG -- {self.active.ndim = }")
		# print(f"DEBUG -- {self.active.npoint = }")
		self.active.range_dims = range(self.active.ndim)
		# print(f"\nDEBUG -- {self.active.range_dims = }")
		self.active.range_points = range(self.active.npoint)
		# print(f"DEBUG -- {trans.columns = }")
		# print(f"DEBUG -- {trans.columns[0] = }")
		for each_dim in self.active.range_dims:
			# print(f"DEBUG -- {each_dim = }")
			self.active.dim_names.append(trans.columns[each_dim])
			self.active.dim_labels.append("CO"+str(each_dim))
		# print(f"DEBUG -- {self.active.dim_names = }")
		# print(f"DEBUG -- {self.active.dim_labels = }")
		for each_point in self.active.range_points:
			# print(f"DEBUG -- {each_point = }")
			self.active.point_names.append(trans.index[each_point])
			self.active.point_labels.append(self.active.point_names[each_point][0:4])
		self.active.bisector.case = "Unknown"
		self.active.distances.clear()
		self.active.point_coords = pd.DataFrame(trans)
		print("\nPoint_coords: \n", self.active.point_coords)
		#
		self.active.max_and_min("Principal components")
		#
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		# Display scree diagram showing eigenvalues by dimensionality
		#
		# Ask user how many dimensions to be retained
		#
		#
		# Display configuration with vectors from origin to each point
		#
		self.complete("Principal components")
		#
		return
	# ---------------------------------------------------------------------------

	def print_configuration_command(self):
		"""The Print configuration command is used to print a copy of the active
			configuration.
		"""
		#
		# Record use of Print configuration command
		#
		self.start("Print configuration")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Print configuration")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Print configuration")
		#
		if problem_detected:
			self.incomplete("Print configuration")
			return
		#
		# Print the file
		#
		problem_writing_file = self.active.write(file_name)

		#
		if problem_writing_file:
			self.active.error("Problem writing file.",
				"Check whether file already exists")
			# self.active.suggest("Check whether file already exists")
			self.incomplete("Print configuration")
			return None
		#
		self.complete("Print configuration")
		#
		return None

	# ---------------------------------------------------------------------------

	def print_correlations_command(self):
		"""The Print correlations command is used to print correlations.
		"""
		#
		# Record use of Print correlations command
		#
		self.start("Print correlations")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Print correlations")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Print correlations")
		#
		if problem_detected:
			self.incomplete("Print correlations")
			return
		#
		# Print the file
		#
		problem_writing_file = self.active.write(file_name)

		#
		if problem_writing_file:
			self.active.error("Problem writing file.",
							"Check whether file already exists")
			# self.active.suggest("Check whether file already exists")
			self.incomplete("Print correlations")
			return None
		#
		self.complete("Print correlations")
		#
		return None

	# ---------------------------------------------------------------------------

	def print_evaluations_command(self):
		"""The Print evaluations command is used to print evaluations.
		"""
		#
		# Record use of Print evaluations command
		#
		self.start("Print evaluations")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Print evaluations")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Print evaluations")
		#
		if problem_detected:
			self.incomplete("Print evaluations")
			return
		#
		# Print the file
		#
		problem_writing_file = self.active.write(file_name)

		#
		if problem_writing_file:
			self.active.error("Problem writing file.",
							"Check whether file already exists")
			# self.active.suggest("Check whether file already exists")
			self.incomplete("Print evaluations")
			return None
		#
		self.complete("Print evaluations")
		#
		return None

	# ---------------------------------------------------------------------------

	def print_grouped_data_command(self):
		"""The Print grouped data command is used to print grouped data.
		"""
		#
		# Record use of Print grouped data command
		#
		self.start("Print grouped data")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Print grouped data")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Print grouped data")
		#
		if problem_detected:
			self.incomplete("Print grouped data")
			return
		#
		# Print the file
		#
		problem_writing_file = self.active.write(file_name)

		#
		if problem_writing_file:
			self.active.error("Problem writing file.",
							"Check whether file already exists")
			# self.active.suggest("Check whether file already exists")
			self.incomplete("Print grouped data")
			return None
		#
		self.complete("Print grouped data")
		#
		return None

	# ---------------------------------------------------------------------------

	def print_similarities_command(self):
		"""The Print similarities command is used to print similarities.
		"""
		#
		# Record use of Print similarities command
		#
		self.start("Print similarities")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Print similarities")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Print similarities")
		#
		if problem_detected:
			self.incomplete("Print similarities")
			return
		#
		# Print the file
		#
		problem_writing_file = self.active.write(file_name)

		#
		if problem_writing_file:
			self.active.error("Problem writing file.",
							"Check whether file already exists")
			# self.active.suggest("Check whether file already exists")
			self.incomplete("Print similarities")
			return None
		#
		self.complete("Print similarities")
		#
		return None

	# ---------------------------------------------------------------------------

	def print_target_command(self):
		"""The Print target command is used to print target .
		"""
		#
		# Record use of Print target command
		#
		self.start("Print target")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Print target")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Print target")
		#
		if problem_detected:
			self.incomplete("Print target")
			return
		#
		# Print the file
		#
		problem_writing_file = self.active.write(file_name)

		#
		if problem_writing_file:
			self.active.error("Problem writing file.",
							"Check whether file already exists")
			# self.active.suggest("Check whether file already exists")
			self.incomplete("Print target")
			return None
		#
		self.complete("Print target")
		#
		return None
	# ---------------------------------------------------------------------------

	def ranks_command(self):
		""" The Ranks command computes the ranks of the similarities and distances.
			It creates a plot showing the rank of the similarity against the rank of the
			corresponding distance which shows how well or poorly the points represent
			the similarities.
		"""
		#
		# Record use of Ranks command
		#
		self.start("Ranks")
		#
		# Explain what command does (if necessary
		#
		self.active.explain("Ranks")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Ranks")
		#
		if problem_detected:
			self.incomplete("Ranks")
			return
		#
		fig = self.active.rank()
		#
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Ranks")
		#
		return

	# ---------------------------------------------------------------------------------------------------------

	def redo_command(self):
		""" The Redo command has yet to be implemented.
		"""
		#
		# Record use of Redo command
		#
		self.start("Redo")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Redo")
		#
		# Handle improper order of commands
		#
		#problem_detected = self.dependencies("Redo")
		#
		# if problem_detected:
		#	self.incomplete("Redo")
		#	return
		#
		# To be continued <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		#
		# print("\n\tThe Redo command has not yet been implemented")
		QMessageBox.information(
			None,
			f"Command not implemented.",
			f"The Redo command has not yet been implemented."
		)

		self.incomplete("Redo")
		return
		#
		self.set_focus_on_tab(4)
		self.complete("Redo")
		#
		return
	#
	# ------------------------------------------------------------------------------------------------------------

	def reference_command(self):
		""" The Reference command is used to establish a pair of
			points to be used as reference points.
		"""
		#
		# Record use of Reference command
		#
		self.start("Reference")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Reference")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Reference")
		#
		if problem_detected:
			self.incomplete("Reference")
			return
		#
		# Set Variables needed
		#
		# refs_a = -1
		# refs_b = -1
		# refs_indexes = []
		#
		self.connector_bisector_cross_x = 0
		self.connector_bisector_cross_y = 0
		self.active.bisector.end_x = 0.0
		self.active.bisector.end_y = 0.0
		self.active.bisector.slope = 0
		self.active.bisector.start_x = 0.0
		self.active.bisector.start_y = 0.0
		self.active.bisector.intercept = 0.0
		self.active.bisector.direction = "Unknown"
		self.active.bisector.case = "Unknown"
		#
		self.active.print_active_function()
		#
		# Establish first reference point
		#
		# print("DEBUG -- just before getting reference points")

		# refs_app = QtWidgets.QApplication(sys.argv)

		refs_app = QMainWindow()

		title = "Select reference points"
		items = self.active.point_names

		dialog = PairofPointsDialog(title, items)

		while True:
			if dialog.exec() == QtWidgets.QDialog.Accepted:
				try:
					selected_items = dialog.selected_items()
					# print("Reference points:", selected_items)
					break
				except ValueError as error:
					QtWidgets.QMessageBox.warning(dialog, "Reference points", str(error))
			else:
				print("Canceled")
				break

		del dialog

		refs_indexes = [
			j for i in range(2)
			for j in self.active.range_points
			if selected_items[i] == self.active.point_names[j]
		]
		# print(f"DEBUG -- {selected_items = }")
		# print(f"DEBUG -- {refs_indexes = }")
		refs_a = refs_indexes[0]
		refs_b = refs_indexes[1]

		self.active.rival_a = refs_a
		self.active.rival_b = refs_b
		#
		print("\n\tReference points: ")
		print("\t\t", self.active.point_labels[refs_a], self.active.point_names[refs_a])
		print("\t\t", self.active.point_labels[refs_b], self.active.point_names[refs_b])
		#
		# Set connector and bisector direction
		#
		self.active.set_direction_flags()
		#
		# Determine attributes of bisector
		#
		self.active.bisector_function(self.active.rival_a, self.active.rival_b)
		# print(f"DEBUG - after call to bisect in ref - {self.active.bisector.start_y = }")
		#
		self.active.set_line_case()
		#
		# Determine dividers in each dimension
		#
		self.active.dividers()
		#
		self.active.show_bisector = True
		#
		self.active.ends_of_bisector_function()
		#
		# print(f"DEBUG - at call to assign in ref - {self.active.bisector.start_y = }")
		# print(f"DEBUG -- {self.active.have_individual_data() = }")
		if self.active.have_individual_data():
			# if not self.active.have_segments():
			self.active.assign_to_segments()
		#
		self.active.max_and_min("Reference")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		# print(f"DEBUG in reference_command {self.active.dim1_div = }")
		# print(f"DEBUG in reference_command {self.active.dim2_div = }")
		#
		# ############################################add show plot with connector ???????????????????????????
		#
		self.complete("Reference")
		#
		return

	# ---------------------------------------------------------------------------------------------------

	def rescale_command(self):
		""" The Rescale command is used to rescale one or more dimensions.
		"""
		#
		# Record use of Rescale command
		#
		self.start("Rescale")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Rescale")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Rescale")
		#
		if problem_detected:
			self.incomplete("Rescale")
			return
		#
		# Set needed variables
		#
		self.active.bisector.case = "Unknown"
		#
		# Ask user which dimension they want to rescale
		#
		invert_app = QMainWindow()
		title = "Select dimensions to rescale"
		dims = self.active.dim_names

		dialog = SelectItemsDialog(title, dims)

		if dialog.exec() == QtWidgets.QDialog.Accepted:
			try:
				selected_items = dialog.selected_items()
			except ValueError as error:
				QtWidgets.QMessageBox.warning(dialog, "Dimensions", str(error))
		else:
			print("Canceled")

		if len(selected_items) == 0:
			self.incomplete("Rescale")
			return

		del dialog

		dims_indexes = [
			j for i in range(len(selected_items))
			for j in self.active.range_dims
			if selected_items[i] == self.active.dim_names[j]
		]
		# print(f"DEBUG -- {selected_items = }")

		title = "Rescale configuration"
		label = "Amount by which to multiple every point \non selected dimensions"
		min = -9999.9
		max = 9999.9
		an_integer = False
		default = 0.0

		app = QMainWindow()

		# Create an instance of the SetValueDialog class
		dialog = SetValueDialog(title, label, min, max, an_integer, default)

		# Show the dialog and retrieve the selected value
		result = dialog.exec()
		if result == QDialog.Accepted:
			value = dialog.getValue()
		# print(f"DEBUG -- Selected value: {value}")
		else:
			# print("DEBUG -- Dialog canceled or closed")
			self.incomplete("Rescale")
			return

		if value == 0:
			self.incomplete("Rescale")
			return

		#

		for each_dim in self.active.range_dims:
			if self.active.dim_names[each_dim] in selected_items:
				self.active.rescale(each_dim, value)

		#
		# Print rescale active configuration
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Rescale")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		self.complete("Rescale")
		#
		return

	# ---------------------------------------------------------------------------------------------------------

	def rotate_command(self):
		""" The Rotate command is used to rotate the active configuration.
		"""
		# print("DEBUG -- at top of rotate command")
		#
		# Record use of Rotate command
		#
		self.start("Rotate")
		#
		# Define needed variables
		#
		self.active.bisector.case = "Unknown"
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Rotate")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Rotate")
		#
		if problem_detected:
			self.incomplete("Rotate")
			return
		#
		# Get degrees to rotate configuration
		#
		title = "Degree to rotate configuration"
		label = f"Positive is counter-clockwise\nNegative is clockwise"
		min = -360
		max = 360
		default = 0
		an_integer = True

		app = QMainWindow()

		# Create an instance of the SetValueDialog class
		dialog = SetValueDialog(title, label, min, max, an_integer, default)

		# Show the dialog and retrieve the selected value
		result = dialog.exec()
		if result == QDialog.Accepted:
			deg = dialog.getValue()
			# print(f"DEBUG -- Selected value: {deg}")
		else:
			# print("DEBUG -- Dialog canceled or closed")
			self.incomplete("Rotate")
			return

		if deg == 0:
			self.incomplete("Rotate")
			return

		#
		# Convert degrees to radians
		#
		radians = math.radians(float(deg))
		#
		# Rotate current plane of active configuration by user supplied value transformed to radians
		#
		self.active.rotate(radians)
		#
		# Print rotated active configuration
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Rotate")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		self.complete("Rotate")
		#
		return

	# ---------------------------------------------------------------------------

	def sample_designer_command(self):
		"""The Sample designer command is used to create a sample design.
		"""
		#
		# Record use of Sample designer command
		#
		self.start("Sample designer")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Sample designer")
		#
		# Handle improper order of commands
		#
		# Get parameters needed to design sample repetitions -
		# 	number of respondents, number of repetitions, probability of inclusions
		#
		# Open dialog????????????????????
		#
		file_name, _ = QFileDialog.getSaveFileName(caption="Sample designer")
		if len(file_name) == 0:
			self.active.error("Empty response.",
				"")
			self.incomplete("Sample designer")
			return None
		#
		# Save the file with the selected file name
		#
		# Write file
		#
		problem_writing_file = self.active.write(file_name)

		#
		if problem_writing_file:
			self.active.error("Problem writing file.",
				"Check whether file already exists")
			# self.active.suggest("Check whether file already exists")
			self.incomplete("Sample designer")
			return None
		#
		# Let user know active configuration has been written to the file
		#
		print("\n\tThe active configuration has be written to: ", file_name)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Sample designer")
		#
		return None

	# ---------------------------------------------------------------------------

	def save_configuration_command(self):
		"""The Save configuration command is used to write a copy of the active
			configuration to a file.
		"""
		#
		# Record use of Write command
		#
		self.start("Save configuration")
		#
		# Explain what command does (if needed)
		#
		self.active.explain("Save configuration")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Save configuration")
		#
		if problem_detected:
			self.incomplete("Save configuration")
			return
		#
		# Get parameters needed to write file
		#
		# Get file name from user and handle problematic file names
		#
		# Open the file save dialog
		#
		file_name, _ = QFileDialog.getSaveFileName(caption="Save configuration")
		if len(file_name) == 0:
			self.active.error("Empty response.",
				"")
			self.incomplete("Save configuration")
			return None
		#
		# Save the file with the selected file name
		#
		# Write file
		#
		problem_writing_file = self.active.write(file_name)

		#
		if problem_writing_file:
			self.active.error("Problem writing file.",
				"Check whether file already exists")
			# self.active.suggest("Check whether file already exists")
			self.incomplete("Save configuration")
			return None
		#
		# Let user know active configuration has been written to the file
		#
		print("\n\tThe active configuration has be written to: ", file_name)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Save configuration")
		#
		return None

	# ---------------------------------------------------------------------------

	def save_target_command(self):
		"""The Save target command is used to write a copy of the target
			configuration to a file.
		"""
		#
		# Record use of Save target command
		#
		self.start("Save target")
		#
		# Explain what command does (if needed)
		#
		self.target.explain("Save target")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Save target")
		#
		if problem_detected:
			self.incomplete("Save target")
			return
		#
		# Get parameters needed to write file
		#
		# Get file name from user and handle problematic file names
		#
		# Open the file save dialog
		#
		file_name, _ = QFileDialog.getSaveFileName(caption="Save target")
		if len(file_name) == 0:
			self.target.error("Empty response.",
				"")
			self.incomplete("Save target")
			return None
		#
		# Save the file with the selected file name
		#
		# Write file
		#
		problem_writing_file = self.target.write(file_name)

		#
		if problem_writing_file:
			self.active.error("Problem writing file.",
				"Check whether file already exists")
			# self.active.suggest("Check whether file already exists")
			self.incomplete("Save target")
			return None
		#
		# Let user know target configuration has been written to the file
		#
		print("\n\tThe target configuration has be written to: ", file_name)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Save target")
		#
		return None
	# ----------------------------------------------------------------------------------------------------------

	def scores_command(self):
		""" The Scores command has not yet been implemented.
		"""
		#
		# Record use of Scores command
		#
		self.start("Scores")
		#
		# Define needed variables
		#
		file = ""
		#
		# Explain what command does
		#
		self.active.explain("Scores")
		#
		# Handle improper order of commands
		#
		# problem_detected = self.dependencies("Scores")
		#
		# if problem_detected:
		#	self.incomplete()
		#	return
		#
		# To be continued
		#
		# print("\n\tThe Scores command has not yet been implemented.")
		QMessageBox.information(
			None,
			f"Command not implemented.",
			f"The Scores command has not yet been implemented."
		)
		self.incomplete("Scores")
		return
		# try:
		# (flags, file) = self.active.input_source_function("\n\tEnter a file name: ", flags)
		# file = file.strip()
		# fhand = open(file)
		# except FileNotFoundError:
		# print("\n\tERROR: File can not be opened:", file)
		# return flags
		#
		# nmscores - based on Fortran program writen by George Rabinovitz
		#
		# maxn - number of points
		# xn - Copy of maxn (see line 39)
		# xmaxn - Copy of xn (see line 56)
		# data - data vector for a single case, will contain maxn elements
		# r - number of dimensions
		# ntot - Total number of cases
		# n - number of valid cases scored
		# nd - the number of distinct scores offered by the subject less 1
		# ndp - the number of pais discriminated by the subject
		# xnd and xndp are initialized at the rnd??? end of the routine
		# nkey[8] - number of degenerate cases
		# nkey[9] - number of cases with all missing data
		# stress - Measure of fit
		# centrd - Centroid of dimension
		# ubnd - Upper bound of dimension
		# lbnd - Lower bound of dimension
		# bnd - Standard deviation?????? of dimension
		# xbnd ??????
		# xbound ???????

		stress = 0  # temporary <<<<<<<<<<<<<<<<<<<
		#
		#
		# name: name of points (max=20)
		name = []
		# y: coordinates of points (max 20x4)
		y = []
		# p: data vector for a single case. Will have maxn elements
		p = []

		nines = 9.0
		md = 97
		strmin = .001
		scrit = .99
		kcrit = 4
		sstart = .4
		magmin = 0.00001
		mxiter = 25
		sumssq = 0.0
		nkey = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		xbound = 6.0
		centrd = [0.0, 0.0, 0.0, 0.0]
		bnd = [0.0, 0.0, 0.0, 0.0]
		ubnd = [0.0, 0.0, 0.0, 0.0]
		lbnd = [0.0, 0.0, 0.0, 0.0]
		achkmn = .1
		xmstr = .6
		gmagen = .001
		#
		# Get active configuration
		#
		maxn = self.active.npoint
		r = self.active.ndim
		range_points = range(maxn)
		range_dims = range(r)
		#
		for each_point in range_points:
			name.append(self.active.point_names[each_point])
			y_temp = []
			for each_dim in range_dims:
				y_temp.append(self.active.point_coords.iloc[each_point][each_dim])
			y.append(y_temp)

		#
		# Find centroid and standard deviation of active configuration.
		# Then establish boundaries for scoring individuals.
		#
		for each_point in range_points:
			for each_dim in range_dims:
				centrd[each_dim] = centrd[each_dim] + y[each_point][each_dim]
				bnd[each_dim] += y[each_point][each_dim] * y[each_point][each_dim]
		xn = maxn
		print("????")
		for each_dim in range_dims:
			centrd[each_dim] = centrd[each_dim] / xn
			bnd[each_dim] = np.sqrt(bnd[each_dim] / xn - centrd[each_dim] * centrd[each_dim])
			xbnd = xbound * bnd[each_dim]
			lbnd[each_dim] = centrd[each_dim] - xbnd
			ubnd[each_dim] = centrd[each_dim] + xbnd
			width = 10
			decimals = 5
			print(
				"Dimension: ", each_dim + 1,
				"    Centroid: " + f"{centrd[each_dim]:{width}.{decimals}f}",
				"    Bound: " + f"{bnd[each_dim]:{width}.{decimals}f}",
				"    Lower bound: " + f"{lbnd[each_dim]:{width}.{decimals}f}",
				"    Upper bound: " + f"{ubnd[each_dim]:{width}.{decimals}f}", )
		xtemp = 0.0
		for each_dim in range_dims:
			if nines < ubnd[each_dim]:
				xtemp = ubnd[each_dim]
		if nines > xtemp:
			nines = 10. * nines + nines
		# ???????????
		print(" Missing data will be scored at: ", nines)
		xmaxn = xn
		#
		# initialize ssort
		#
		kk = scores_ssort_function(maxn)
		# ???????????????????   call ssort(maxn)

		#
		# ***** Start Population loop
		#
		# Read in case, and initialize population
		#
		# simulating reading in a case
		#
		data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
		p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

		#
		# Sort data from low to high
		#
		# ??????????????????????????? call sort
		scores_sort_function(maxn, kk)

		#
		# Set n equal to the number of valid cases.
		# If there are no valid cases go to ??????700
		#
		n = maxn
		range_scores_less_1 = range(1, maxn)
		for each_score in range_scores_less_1:
			if data[each_score].lt.md:
				n += -1
		# ####### GO TO 700????????????

		#
		# Calculate summary stats I/O and finish
		#  same as 900
		n = 0
		range_keys = range(8)
		for each_key in range_keys:
			n += nkey[each_key]
		xn = n
		ntot = n + nkey[8] + nkey[9]
		#
		print(
			"\tSummary:",
			"\n\tN cases: ", ntot,
			"\n\tN scored: ", n,
			"\n\n\tDegenerate: ", nkey[8],
			"\n\tN with all missing data: ", nkey[9])
		print("\n\tKeys: ", nkey)
		print("\n\t\tStress: ", stress)

		#
		# set status indicators
		#
		self.active.have_scores = "Yes"
		#
		# To be continued<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		#
		self.complete("Scores")
		#
		return

	# ---------------------------------------------------temp

	def scores_ssort_function(nn):
		kk = 4
		while kk < nn:
			kk = kk * 2
		kk = (kk / 2) - 1
		return kk

	# ---------------------------------------------------temp

	def scores_sort_function(nn, kk):
		l = kk
		max = nn - l
		range_max = range(max)
		for each_score in range_max:
			k = each_score

		return

	# ----------------------------------------------------------------------------------------------------------

	def scree_command(self):
		""" The Scree command creates diagram showing stress vs. dimensionality.
		"""
		#
		# Record use of Scree command
		#
		self.start("Scree")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Scree")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Scree")
		#
		if problem_detected:
			self.incomplete("Scree")
			return
		#
		# Set variables needed
		#
		self.active.min_stress.clear()
		x_coords = []
		y_coords = []
		self.active.dim_names.clear()
		self.active.dim_labels.clear()
		#
		app = QMainWindow()
		title = "MDS model"
		options_title = "Model to use"
		options = ["Non-metric", "Metric"]
		dialog = ChoseOptionDialog(title, options_title, options)
		result = dialog.exec()

		if result == QDialog.Accepted:
			selected_option = dialog.selected_option       # + 1
			match selected_option:
				case 0:
					self.active.use_metric = False
				case 1:
					self.active.use_metric = True
				case _:
					# print(f"DEBUG -- result is blank")
					self.incomplete("Scree")
					return
			# print(f"DEBUG -- at accepted {selected_option = } {result = } {self.active.use_metric = }")
		else:
			# print(f"Rejected")
			self.incomplete("Scree")
			return

		#
		# Perform repeated multidimensional scaling to create Scree diagram
		#
		self.active.scree()
		#
		fig = self.active.plot_scree()
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Scree")
		#
		return

	# -----------------------------------------------------------------------------

	def second_dim_command(self):
		""" The Second dimension command identifies regions defined by the second dimension
		"""
		#
		# Record use of Second dimension command
		#
		self.start("Second dimension")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Second dimension")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Second dimension")
		#
		if problem_detected:
			self.incomplete("Second dimension")
			return
		#
		# Set variables needed
		#
		reply = ""
		#
		# Determine whether segments of individuals have been established
		# If yes, ask user whether to show left, right, both, or neither
		#

		if self.active.have_segments():
			app = QMainWindow()
			title = "Second dimension segments"
			options_title = "Segments in which to show individual points"
			options = ["Upper", "Lower"]
			dialog = ChoseOptionDialog(title, options_title, options)
			result = dialog.exec()

			if result == QDialog.Accepted:
				selected_option = dialog.selected_option        # + 1
				match selected_option:
					case 0:
						reply = "uppe"
					case 1:
						reply = "lowe"
					case _:
						# print(f"DEBUG -- result is zero")
						pass
				# print(f"DEBUG -- at accepted {selected_option = } {result = } {reply = }")
			else:
				# print(f"DEBUG -- at else {result = }")
				pass

		#
		fig = self.active.plot_second(reply)
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Second dimension")
		#
		return None

	# --------------------------------------------------------------------------------------------------------

	def segments_command(self):
		""" The Segments command identifies regions defined by the individual scores
		"""
		#
		# Record use of Segments command
		#
		self.start("Segments")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Segments")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Segments")
		#
		if problem_detected:
			self.incomplete("Segments")
			return
		#
		# Set variables needed
		#
		width = 4
		decimals = 1
		#
		# Assign segment if needed
		#
		if not self.active.have_segments():
			self.active.assign_to_segments()
		#
		self.active.print_segments(width, decimals)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Segments")
		#
		return

	# -----------------------------------------------------------------------------------

	def set_focus_on_tab(self, index):
		# print(f"DEBUG - at set focus")
		self.tab_widget.setCurrentIndex(index)
		self.tab_widget.update()

	# -----------------------------------------------------------------------------------

	def settings_command(self, settings_group):

		print(f"DEBUG -- at top of settings_command {settings_group = }")
		#
		# Record use of Settings command
		#
		self.start("Settings")
		#
		# Explain what the command does (if necessary)
		#
		self.active.explain("Settings")
		#
		if settings_group == "plot":
			settings_app = QMainWindow()
			title = "Show if available"
			items = [
				"Reference points",
				"Bisector",
				"Connector",
				"Only reference points in Joint plots"
			]
			default_values = [
				self.active.show_reference_points,
				self.active.show_bisector,
				self.active.show_connector,
				self.active.show_just_reference_points
			]

			dialog = ModifyItemsDialog(title, items, default_values=default_values)

			if dialog.exec() == QtWidgets.QDialog.Accepted:
				try:
					selected_items = dialog.selected_items()
				except ValueError as error:
					QtWidgets.QMessageBox.warning(dialog, "Features", str(error))
			else:
				print("Canceled")
				self.incomplete("Settings")
				return None

			del dialog

			# print(f"DEBUG -- {selected_items = }")

			features_indexes = [
				j for i in range(len(selected_items))
				for j in range(len(items))
				if selected_items[i] == items[j]
			]
			# print(f"DEBUG -- {features_indexes = }")
			if 0 in features_indexes:
				self.active.show_reference_points = True
			else:
				self.active.show_reference_points = False
			if 1 in features_indexes:
				self.active.show_bisector = True
			else:
				self.active.show_bisector = False
			if 2 in features_indexes:
				self.active.show_connector = True
			else:
				self.active.show_connector = False
			if 3 in features_indexes:
				self.active.show_just_reference_points = True
			else:
				self.active.show_just_reference_points = False

			print(f"DEBUG -- {self.active.show_reference_points = }")
			print(f"DEBUG -- {self.active.show_bisector = }")
			print(f"DEBUG -- {self.active.show_connector = }")
			print(f"DEBUG -- {self.active.show_just_reference_points = }")

		elif settings_group == "segment":
			settings_app = QMainWindow()
			title = "Segment sizing"
			items = [
				"Define battleground sector using percent \nof connector on each side of bisector ",
				"Define core sector around reference \npoint using percent of connector            "
			]
			default_values = [
				int(self.active.tolerance * 100),
				int(self.active.core_tolerance * 100)
			]
			integers = True

			dialog = ModifyValuesDialog(
				title, items, integers, default_values=default_values)

			# Show the dialog and retrieve the selected value

			dialog.selected_items()

			result = dialog.exec()

			print(f"DEBUG -- {dialog.selected_items = }")

			if result == QDialog.Accepted:
				value = dialog.selected_items()
				print(f"Selected value: {value}")
				print(f"DEBUG -- {value[0][1] = }")
				print(f"DEBUG -- {value[1][1] = }")
				self.active.tolerance = value[0][1] / 100.0
				self.active.core_tolerance = value[1][1] / 100.0
			else:
				print("Dialog canceled or closed")
				self.incomplete("Settings")
				return None

			print(f"DEBUG -- {self.active.tolerance = }")
			print(f"DEBUG -- {self.active.core_tolerance =}")
			#app.exec()

		elif settings_group == "display":
			settings_app = QMainWindow()
			title = "Display adjustments"
			items = [
				"Extend axis by adding percent of axis maxima \nto keep points from falling on the edge of plots",
				"Improve visibility by displacing labelling off\n point by percent of axis maxima                        ",
				"Size in points of the dots representing people\n in plots                                               ",
			]
			integers = True
			default_values = [
				int(self.active.axis_extra * 100),
				int(self.active.displacement * 100),
				self.active.point_size,
			]

			dialog = ModifyValuesDialog(title, items, integers, default_values=default_values)

			# Show the dialog and retrieve the selected value

			dialog.selected_items()

			result = dialog.exec()

			print(f"DEBUG -- {dialog.selected_items = }")

			if result == QDialog.Accepted:
				value = dialog.selected_items()
				print(f"Selected value: {value}")
				print(f"DEBUG -- {value[0][1] = }")
				print(f"DEBUG -- {value[1][1] = }")
				self.active.axis_extra = value[0][1] / 100.0
				self.active.displacement = value[1][1] / 100.0
				self.active.point_size = value[2][1]
			else:
				print("Dialog canceled or closed")
				self.incomplete("Settings")
				return None

			print(f"DEBUG -- {self.active.axis_extra = }")
			print(f"DEBUG -- {self.active.displacement = }")
			print(f"DEBUG -- {self.active.point_size = }")

		elif settings_group == "vectors":
			settings_app = QMainWindow()
			title = "Vector size"
			items = [
				"Vector head size in inches",
				"Vector thickness in inches"
			]
			integers = False
			default_values = [
				self.active.vector_head_width,
				self.active.vector_width
			]

			dialog = ModifyValuesDialog(title, items, integers, default_values=default_values)

			# Show the dialog and retrieve the selected value

			dialog.selected_items()

			result = dialog.exec()

			print(f"DEBUG -- {dialog.selected_items = }")

			if result == QDialog.Accepted:
				value = dialog.selected_items()
				print(f"Selected value: {value}")
				print(f"DEBUG -- {value[0][1] = }")
				print(f"DEBUG -- {value[1][1] = }")
				self.active.vector_head_width = value[0][1]
				self.active.vector_width = value[1][1]
			else:
				print("Dialog canceled or closed")
				self.incomplete("Settings")
				return None

			print(f"DEBUG -- {self.active.vector_head_width = }")
			print(f"DEBUG -- {self.active.vector_width = }")

		elif settings_group == "layout":
			settings_app = QMainWindow()
			title = "Layout options"
			items = [
				"Maximum number of columns per page",
				"Field width",
				"Decimal points                                               "
			]
			integers = True
			default_values = [
				self.max_cols,
				self.width,
				self.decimals
			]

			dialog = ModifyValuesDialog(title, items, integers, default_values=default_values)

			# Show the dialog and retrieve the selected value

			dialog.selected_items()

			result = dialog.exec()

			print(f"DEBUG -- {dialog.selected_items = }")

			if result == QDialog.Accepted:
				value = dialog.selected_items()
				print(f"Selected value: {value}")
				print(f"DEBUG -- {value[0][1] = }")
				print(f"DEBUG -- {value[1][1] = }")
				self.max_cols = value[0][1]
				self.width = value[1][1]
				self.decimals = value[2][1]
			else:
				print("Dialog canceled or closed")
				self.incomplete("Settings")
				return None

			print(f"DEBUG -- {self.max_cols = }")
			print(f"DEBUG -- {self.width = }")
			print(f"DEBUG -- {self.decimals = }")
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Settings")
		#
		return None
	# -----------------------------------------------------------------------------------------

	def shepard_command(self):
		""" The Shepard command creates shepard diagram - rank of distance against rank or similarity
		"""
		#
		# Record use of Shepard command
		#
		self.start("Shepard")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Shepard")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Shepard")
		#
		if problem_detected:
			self.incomplete("Shepard")
			return
		#
		# Define needed variables
		#
		axis = ""
		#
		# Ask user which axis to use to display similarities
		#
		app = QMainWindow()
		title = "Shepard diagram"
		options_title = "Show similarity on:"
		options = ["X-axis (horizontal)", "Y-axis (vertical)"]
		dialog = ChoseOptionDialog(title, options_title, options)
		result = dialog.exec()

		if result == QDialog.Accepted:
			selected_option = dialog.selected_option       # + 1
			match selected_option:
				case 0:
					axis = "X"
				case 1:
					axis = "Y"
				case _:
					# print(f"DEBUG -- result is blank")
					self.incomplete("Shepard")
					return
			# print(f"DEBUG -- at accepted {selected_option = } {result = } {self.active.use_metric = }")
		else:
			# print(f"Rejected")
			self.incomplete("Shepard")
			return

		fig = self.active.plot_shep(axis)
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Shepard")
		#
		return None

	# ---------------------------------------------------------------------------------------------------------------

	def similarities_command(self):
		""" The Similarities command is used to establish similarities
			between the points.
		"""
		#
		# Record use of Similarities command
		#
		self.start("Similarities")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Similarities")
		#
		# Set variables needed to read in, store, and print similarities
		#
		self.active.nreferent = 0
		self.active.item_labels.clear()
		self.active.item_names.clear()
		self.active.similarities.clear()
		self.active.similarities_as_dict = dict()
		self.active.similarities_as_list.clear()
		self.active.similarities_as_square.clear()
		self.active.sorted_similarities = dict()
		self.active.a_item.clear()
		self.active.b_item.clear()
		self.active.zipped.clear()
		self.active.value_type = "Unknown"
		width = 8
		decimals = 2
		#

		title = "Similarity measure"
		options_title = "Treat values as"
		options = ["Similarities", "Dis/similarities"]
		app = QMainWindow()
		dialog = ChoseOptionDialog(title, options_title, options)
		result = dialog.exec()

		if result == QDialog.Accepted:
			selected_option = dialog.selected_option                     # + 1
			match selected_option:
				case 0:
					self.active.value_type = "similarities"
				case 1:
					self.active.value_type = "dissimilarities"
				case _:
					# print(f"DEBUG -- result is zero")
					self.incomplete("Similarities")
					return
					# pass
			# print(f"DEBUG -- at accepted {selected_option = } {result = } {self.active.value_type = }")
		else:
			# print(f"Rejected")
			self.incomplete("Similarities")
			return

		# print(f"DEBUG -- {self.active.value_type = }")

		#
		# Ask user for file name containing similarities and handle nonexistent file names
		#
		ui_file = QFileDialog.getOpenFileName(caption="Open similarities", filter="*.txt")
		file = ui_file[0]
		if file == "":
			self.active.error("No file selected",
				"To establish similarities select file in dialog.")
			self.incomplete("Similarities")
			return

		#																	self.value_type =
		# Call read_lower_triangular to read in similarity matrix
		#
		#problem_reading_file = self.active.read_lower_triangular(file_name)
		problem_reading_file = self.active.read_lower_triangular(file)
		#
		# Handle problem reading file
		#
		if problem_reading_file is True:
			self.value_type = "Unknown"
			self.incomplete("Similarities")
			return
		#
		# print(f"DEBUG -- about to check against active configuration")
		if self.active.have_active_configuration():
			if not (self.active.npoint == self.active.nreferent):
				self.similarities = []
				self.similarities_as_dict = dict()
				self.similarities_as_list = []
				self.similarities_as_square = []
				self.value_type = "Unknown"
				self.incomplete("Similarities")
				return

		self.active.similarities = self.active.values
		#
		self.active.npoint = self.active.nreferent
		#
		if not self.active.have_active_configuration():
			self.active.point_labels = self.active.item_labels
			self.active.point_names = self.active.item_names
		#
		# Print matrix type and number of items
		#
		print("\n\tThe", self.active.value_type, "matrix has", self.active.nreferent, "items")
		#
		# Call print_lower_triangle	to print similarities
		#
		self.active.print_lower_triangle(
			decimals, self.active.item_labels, self.active.item_names, self.active.nreferent, self.active.similarities, width)

		# WARNING - DO NOT ALPHABETIZE ARGUMENT LIST, Order needed because this uses reusable function
		#
		self.active.duplicate_similarities()
		#
		# set status indicator to indicate that similarities have been established
		#
		# Return the status indicator, value values are dis/similarities,
		# the number of points, the label and name for each point and the similarities as a list
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Similarities")
		#
		return
	# ----------------------------------------------------------------------------------

	def start(self, command):
		#testing commit by editing this file
		#
		# Update status bar
		#
		self.spaces_statusbar.showMessage(f"Starting {command} command")
		#
		# Record use of named command
		#
		self.active.commands_used.append(command)
		self.active.command_exit_code.append(-1)			# -1 indicates command is in process
		#
		# self.spaces_win.statusbar.showMessage(f"{command} started")
		# print(f"DEBUG -- {self.active.commands_used = }")
		# print(f"DEBUG -- {self.active.command_exit_code = }")
		#
		# Create previous for undo
		#
		passive_commands = (
			"About", "Base", "Battleground", "Bisector", "Contest", "Convertibles", "Core supporters",
			"Deactivate", "Differences", "Distances", "Exit", "Help", "History",
			"Joint", "Likely supporters",
			"Paired", "Ranks", "Sample designer", "Save configuration",
			"Save target", "Shepard", "Status",
			"Stress", "Terse", "Undo", "Verbose", "View configuration", "View grouped data",
			"View correlations", "View similarities", "View target")
		# if self.active.have_active_configuration() \
			# and command not in passive_commands:
		if command not in passive_commands:
			self.undo_stack.append(copy.deepcopy(self.active))
			self.undo_stack_source.append(command)
		# range_undo = range(len(self.undo_stack))
		# for each_object in range_undo:
			# print(f"DEBUG -- {each_object = } {self.undo_stack[each_object] = }")

# -----------------------------------------------------------------------------------

	def status_command(self):
		"""The status command displays all the indicators describing the current status.
		"""
		#
		# Record use of Status command
		#
		self.start("Status")
		#
		# Explain what the command does (if necessary)
		#
		self.active.explain("Status")
		#
		print("\n\tWhether various aspects have been established")
		print("\t\t have_active_configuration: ", self.active.have_active_configuration())
		print("\t\t have_bisector_info: ", self.active.have_bisector_info())
		# print("\t\t have_boundaries_info: ", self.active.have_boundaries_info).......delete??????????
		print("\t\t have_clusters: ", self.active.have_clusters)
		print("\t\t have_correlations: ", self.active.have_correlations())
		print("\t\t have_differences: ", self.active.have_differences)
		print("\t\t have_distances: ", self.active.have_distances())
		print("\t\t have_evaluations: ", self.active.have_evaluations())
		print("\t\t have_factors: ", self.active.have_factors)
		print("\t\t have_grouped_data: ", self.active.have_grouped_data())
		print("\t\t have_individual_data: ", self.active.have_individual_data())
		print("\t\t have_MDS_results: ", self.active.have_mds_results())
		print("\t\t have_previous_active: ", self.have_previous_active())
		print("\t\t have_ranks: ", self.active.have_ranks())
		print("\t\t have_reference_points: ", self.active.have_reference_points())
		print("\t\t have_scores: ", self.active.have_scores)
		print("\t\t have_segments: ", self.active.have_segments())
		print("\t\t have_similarities: ", self.active.have_similarities())
		print("\t\t have_target_configuration: ", self.active.have_target_configuration())
		print("\t\t include_explanation: ", self.active.include_explanation())
		#
		print("\n\tWhether various aspects will be included in plots ")
		print("\t\t show_bisector: ", self.active.show_bisector)
		# print("\t\t show_groups: ", self.active.show_groups)............delete?
		print("\t\t show_just_reference_points: ", self.active.show_just_reference_points)
		print("\t\t show_respondent_points: ", self.active.show_respondent_points)
		#
		print("\n\tParameter settings")
		print("\t\t axis_extra: ", self.active.axis_extra)
		print("\t\t displacement: ", self.active.displacement)
		print("\t\t tolerance: ", self.active.tolerance)
		print("\t\t core radius tolerance: ", self.active.core_tolerance)
		print("\t\t size in points of dots for individuals", self.active.point_size)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Status")
		#
		return None

	# ---------------------------------------------------------------------------------------------------

	def stress_command(self):
		""" The Stress command assesses point contribution to lack of fit.
			The Stress command is used to identify points with high contribution to the lack
			of fit. The user will be shown all the pairs that include a point in a plot
			showing rank of similarity and rank of distance between the points.
		"""
		# print("DEBUG -- at top of stress_command")
		#
		# Record use of Stress command
		#
		self.start("Stress")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Stress")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Stress")
		#
		if problem_detected:
			self.incomplete("Stress")
			return
		#
		# Set variables needed
		#
		point_index = -1
		#
		print("\n\tThe following pairs contribute the most to Stress: \n")
		#
		# Square the distances so that negative distances are as important as
		# positive. This is needed for the Stress command
		#
		self.active.df["Squared_Difference"] = self.active.df["AB_Rank_Difference"] * self.active.df["AB_Rank_Difference"]
		self.active.df["Pct_of_Stress"] = np.sqrt(self.active.df["Squared_Difference"])
		total_stress = np.sum(self.active.df["Pct_of_Stress"])
		self.active.df["Pct_of_Stress"] = (self.active.df["Pct_of_Stress"] / total_stress) * 100
		sorted_dyads = self.active.df.sort_values(by='Squared_Difference', ascending=False)
		worst_fit = sorted_dyads[["A", "B", "Similarity_Rank", "Distance_Rank", "Pct_of_Stress"]]
		print(worst_fit.head(n=20))

		# self.active.df.head(20).style.format({"A", "B", "Similarity_Rank", "Distance_Rank", "Pct_of_Stress": "{:20, 6.2f}"})

		a_point = "None"
		#
		self.active.print_active_function()
		#
		title = "Contribution to stress"
		label = "Select point"
		items = self.active.point_names
		selection = 0
		selected_name, ok = QInputDialog.getItem(
			self,
			title,
			label,
			items,
			current=selection,
			editable=False
		)
		#
		# get index of point
		#
		for each_point in self.active.range_points:
			if self.active.point_names[each_point] == selected_name:
				index_selected_name = each_point
				break
			else:
				continue

		# ui_file_name = "temp_stress.ui"   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<archived

		point_index = index_selected_name
		point = self.active.item_labels[point_index]

		# print(f"DEBUG -- {point = }")

		# print(f"DEBUG -- {file = }")

		# print("DEBUG -- within temp_stress_item before call to command")

		#
		# self.active.plot_stress_by_point(point, point_index)

		fig = self.active.plot_stress_by_point(point, index_selected_name)
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		#
		self.complete("Stress")
		#
		return None

	# ---------------------------------------------------------------------------

	def target_command(self):
		"""The Target command establishes a target configuration.
		"""
		#
		# Record use of Target command
		#
		# print("DEBUG -- in Target just before calling start")
		self.start("Target")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Target")
		#
		# Set variables needed
		#
		# self.target = Configuration()
		#
		self.target.hor_dim = 0
		self.target.vert_dim = 1
		self.target.ndim = 0
		self.target.npoint = 0
		self.target.dim_labels.clear()
		self.target.dim_names.clear()
		self.target.point_labels.clear()
		self.target.point_names.clear()
		# self.target.dim_labels = []
		# self.target.dim_names = []
		# self.target.point_labels = []
		# self.target.point_names = []
		self.target.point_coords = pd.DataFrame()

		ui_file = QFileDialog.getOpenFileName(caption="Open target configuration", filter="*.txt")
		file = ui_file[0]
		#
		# Get file name from dialog and handle nonexistent file names
		#
		# Read configuration
		#
		problem_reading_file = self.target.read_configuration_function(file)

		if problem_reading_file:
			self.incomplete("Target")
			return

		#
		# Describe configuration read
		#
		print("\n\tTarget configuration has", self.target.ndim, "dimensions and", self.target.npoint, "points\n")
		#
		# Display configuration in printed form
		#
		self.target.print_active_function()
		#
		# Show plot of active configuration
		#
		self.target.max_and_min("Target")
		#
		# print("DEBUG -- just before call to plot_configuration")
		fig = self.target.plot_configuration()
		self.add_plot(fig)
		self.show()
		self.set_focus_on_tab(0)
		# print("DEBUG -- just after call to plot_configuration")
		#
		self.complete("Target")
		#
		return

	# ---------------------------------------------------------------------------

	def terse_command(self):
		"""The Terse command toggles the include_explanation indicator to False.
		"""
		#
		# Record use of Terse command
		#
		self.start("Terse")
		#
		print("\n\tCommands will not include explanations.")

		#
		verbosity_toggle = False
		self.verbosity_alternative = "Verbose"
		#
		# Emit the signal with the boolean value
		#
		self.verbosity_signal.signal.emit(verbosity_toggle)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Terse")
		#
		return
# ---------------------------------------------------------------------------------

	def undo_command(self):

		self.start("Undo")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Undo")
		#
		passive_but_failable = (
			"Base", "Bisector", "Contest", "Convertibles", "Core supporters",
			"Deactivate", "Differences", "Distances",
			"Joint", "Likely supporters", "Battleground",
			"Paired", "Plot", "Ranks", "Save configuration",
			"Save target", "Shepard",
			"Stress", "Undo")

		can_not_fail = ("Done", "History", "Status", "Stop", "Terse", "Verbose")

		if self.have_previous_active():
			print("\n\tUndoing ", self.undo_stack_source[-1])
			self.active = self.undo_stack[-1]
			#
			if self.active.have_active_configuration():
				self.active.print_active_function()
				#
				self.active.max_and_min("Undo")
				if self.active.ndim > 1:
					fig = self.active.plot_configuration()
					self.add_plot(fig)
					self.show()
			del self.undo_stack[-1]
			del self.undo_stack_source[-1]
		else:
			self.active.error("No previous configuration",
				"Establish an active configuration before using Undo")
			self.incomplete("Undo")
			return
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Undo")
		#
		return

	# ---------------------------------------------------------------------------------------------------------

	def varimax_command(self):
		""" The Varimax command performs a varimax rotation on the active configuration.
		"""
		#
		# Record use of Varimax command
		#
		self.start("Varimax")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Varimax")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Varimax")
		#
		if problem_detected:
			self.incomplete("Varimax")
			return
		#
		if len(self.active.item_names) == 0:
			self.active.item_names = self.active.point_names
		#
		print(f"DEBUG --before -- {self.active.point_coords = } {self.active.ndim = } {self.active.npoint = } {self.active.nreferent = }")
		to_be_rotated = np.array(self.active.point_coords)
		#
		rotated = self.active.varimax_function(to_be_rotated, gamma=1.0, q=20, tol=1e-6)
		print(f"DEBUG -- in process {rotated = }")
		print(f"DEBUG -- {self.active.dim_labels = } {self.active.point_names = } {self.active.item_names = }")
		#
		self.active.point_coords = pd.DataFrame(
			rotated, columns=self.active.dim_labels, index=self.active.item_names)

		print(f"DEBUG -- after -- {self.active.point_coords = } {self.active.ndim = } {self.active.npoint = } {self.active.nreferent = }")
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Varimax")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		self.complete("Varimax")
		#
		return

		# ------------------------------------------------------------------------------------------------------

	def vectors_command(self):
		""" The Vectors command plots the active configuration using vectors.
		"""
		# print("DEBUG -- At top of vectors command")
		#
		# Record use of Vector command
		#
		self.start("Vectors")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("Vectors")
		#
		# Set variables needed
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("Vectors")
		#
		if problem_detected:
			self.incomplete("Vectors")
			return
		#
		# Display configuration in printed form
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("Vectors")
		#
		# print("DEBUG -- just before call to plot_vectors")
		if self.active.ndim > 1:
			fig = self.active.plot_vectors()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		# print("DEBUG -- just after call to plot_vectors")
		#

		#
		# text = "Hello world!\nCan we put more here?\nI wonder where it will  go?  "
		# print(f"DEBUG -- about to call add_output {text = }")
		# self.add_output(text)
		# print(f"DEBUG -- just called add_output {text = }")
		# print(f"DEBUG {self.active.point_coords = }")
		# self.conf_output()

		self.complete("Vectors")
		#
		return

	# ------------------------------------------------------------------------------------------

	def verbose_command(self):

		#
		# Record use of Verbose command
		#
		# print("DEBUG -- in Verbose just before calling start")
		self.start("Verbose")
		#
		# Explain what command does
		#
		print("Commands WILL include explanations.")
		#
		verbosity_toggle = True
		self.verbosity_alternative = "Terse"
		#
		# Emit the signal with the boolean value
		#
		self.verbosity_signal.signal.emit(verbosity_toggle)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("Verbose")
		# print(f"DEBUG -- at end of verbose_command {self.active.commands_used = }")
		# print(f"DEBUG -- at end of verbose_command {self.active.command_exit_code = }")
		# print(f"DEBUG -- at end of verbose_command {self.active.include_explanation() = }")
		#
		return

	# ------------------------------------------------------------------------------------------------------

	def view_configuration_command(self):
		""" The View configuration command displays the active configuration .
		"""
		# print("DEBUG -- At top of view configuration command")
		#
		# Record use of Configuration command
		#
		self.start("View configuration")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("View configuration")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("View configuration")
		#
		if problem_detected:
			self.incomplete("View configuration")
			return
		#
		#
		# Describe configuration read
		# print(f"DEBUG -- {self.active.ndim = }  {self.active.npoint =}")
		#
		print("\n\tConfiguration has", self.active.ndim, "dimensions and", self.active.npoint, "points\n")
		#
		# Display configuration in printed form
		#
		self.active.print_active_function()
		#
		# Show plot of active configuration
		#
		self.active.max_and_min("View configuration")
		#
		# print("DEBUG -- just before call to plot_configuration")
		if self.active.ndim > 1:
			fig = self.active.plot_configuration()
			self.add_plot(fig)
			self.show()
			self.set_focus_on_tab(0)
		#
		self.complete("View configuration")
		#
		return

	# ------------------------------------------------------------------------------------------------------

	def view_target_command(self):
		""" The View target command displays the target configuration .
		"""
		# print("DEBUG -- At top of view configuration command")
		#
		# Record use of Configuration command
		#
		self.start("View target")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("View target")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("View target")
		#
		if problem_detected:
			self.incomplete("View target")
			return
		#
		#
		# Describe configuration read
		# print(f"DEBUG -- {self.active.ndim = }  {self.active.npoint =}")
		#
		print("\n\tTarget configuration has", self.target.ndim, "dimensions and", self.target.npoint, "points\n")
		#
		# Display configuration in printed form
		#
		self.target.print_active_function()
		#
		# Show plot of active configuration
		#
		self.target.max_and_min("View target")
		#
		# print("DEBUG -- just before call to plot_configuration")
		self.target.plot_configuration()
		#
		self.set_focus_on_tab(0)
		#
		self.complete("View target")
		#
		return

	# ------------------------------------------------------------------------------------------------------

	def view_correlations_command(self):
		""" The View correlations command displays the active correlations .
		"""
		#
		# print("DEBUG -- At top of view correlations command")
		#
		# Record use of View correlations command
		#
		self.start("View correlations")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("View correlations")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("View correlations")
		#
		if problem_detected:
			self.incomplete("View correlations")
			return
		#
		# Display correlations in printed form
		#
		# self.active.print_correlations_function()
		self.active.print_lower_triangle(self.decimals, self.active.item_labels, self.active.item_names,
			self.active.nreferent, self.active.correlations, self.width)
		#
		self.set_focus_on_tab(4)
		#
		self.complete("View correlations")
		#
		return
	#
	# -----------------------------------------------------------------------------

	def view_grouped_command(self):
		#
		# Record use of View grouped command
		#
		self.start("View grouped data")
		#
		# Explain what command does
		#
		self.active.explain("View grouped data")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("View grouped data")
		#
		if problem_detected:
			self.incomplete("View grouped data")
			return
		#
		# Print grouped
		#
		print(f"\n\tGroups were defined using {self.active.grouping_var} variable.\n")
		#
		self.active.print_grouped_function()
		#
		# Show plot of grouped points
		#
		self.active.plot_grouped()
		#
		self.set_focus_on_tab(0)
		#
		self.complete("View grouped data")
		#
		return

	# ---------------------------------------------------------------------------------------------------------------

	def view_similarities_command(self):
		""" The View similarities command is used to display similarities
			between the points.
		"""
		#
		# Record use of Similarities command
		#
		self.start("View similarities")
		#
		# Explain what command does (if necessary)
		#
		self.active.explain("View similarities")
		#
		# Handle improper order of commands
		#
		problem_detected = self.dependencies("View similarities")
		#
		if problem_detected:
			self.incomplete("View similarities")
			return
		#
		#
		# Set variables needed to read in, store, and print similarities
		#
		width = 8
		decimals = 2
		#
		# Print matrix type and number of items
		#
		print("\n\tThe", self.active.value_type, "matrix has", self.active.nreferent, "items")
		#
		# Call print_lower_triangle	to print similarities
		#
		self.active.print_lower_triangle(
			decimals, self.active.item_labels, self.active.item_names, self.active.nreferent, self.active.similarities, width)

		# WARNING - DO NOT ALPHABETIZE ARGUMENT LIST, Order needed because this uses reusable function
		#
		self.active.duplicate_similarities()
		#
		# set status indicator to indicate that similarities have been established
		#
		# Return the status indicator, value values are dis/similarities,
		# the number of points, the label and name for each point and the similarities as a list
		#
		self.set_focus_on_tab(4)
		#
		self.complete("View similarities")
		#
		return

# -------------------------------------------------------------------------------------


class Line:
	def __init__(self):

		self.length: float = 0.0
		#
		self.slope: float = 0.0
		self.SLOPE_FUDGE: float = .00000000000000000000001
		self.SLOPE_TEST: float = -1.0000000000000001e+23
		self.intercept: float = 0.0
		self.direction = "Unknown"
		#
		self.goes_through_top = "Unknown"
		self.goes_through_bottom = "Unknown"
		self.goes_through_left_side = "Unknown"
		self.goes_through_right_side = "Unknown"
		self.case = "Unknown"
		#
		self.start_x: float = 0.0
		self.start_y: float = 0.0
		self.end_x: float = 0.0
		self.end_y: float = 0.0
		self.y_at_max_x: float = 0.0
		self.y_at_min_x: float = 0.0
		self.x_at_max_y: float = 0.0
		self.x_at_min_y: float = 0.0


class Polygon:

	def __init(self):

		self.vertices: List[float] = []
		self.x: List[float] = []
		self.y: List[float] = []
		self.color = "Black"


class Configuration:
	"""
	class Configuration - defines the characteristics of a configuration
	"""
	# -----------------------------------------------------------------------------------------

	def __init__(self):

		self.bisector = Line()
		self.west = Line()
		self.east = Line()
		self.connector = Line()
		# self.vertical = Line()
		# self.horizontal = Line()
		# self.party_wars = Line()
		# self.culture_wars = line()
		self.base_left = Polygon()
		self.base_right = Polygon()
		self.convertible_to_left = Polygon()
		self.convertible_to_right = Polygon()
		self.likely_left = Polygon()
		self.likely_right = Polygon()
		self.battleground = Polygon()
		self.first_left = Polygon()
		self.first_right = Polygon()
		self.second_up = Polygon()
		self.second_down = Polygon()
		self.commands_used: List[str] = ["Initialize"]  # Seeding commands_used and command_exit_code needed to avoid problem when active is deactivated
		self.command_exit_code: List[int] = [0]
		self.dim1 = pd.DataFrame()
		self.dim2 = pd.DataFrame()
		self.avg_eval = pd.DataFrame()
		self.evaluations = pd.DataFrame()

		# from Refs class:
		# Refs class contains several variables which are totally dependent on which items are selected as referents
		self.rival_a: int = -1 	# Index of the first reference point in point_coord, points_labels, points_names
		self.rival_b: int = -1  # Index of the second reference point in point_coord, points_labels, points_names
		self.connector_bisector_cross_x: float = 0.0		# connector_bisector_crossing_x
		self.connector_bisector_cross_y: float = 0.0		# connector_bisector_crossing_y
		#
		# segments
		#
		self.seg = pd.DataFrame()
		#
		self.core_radius: float = 0.0
		# self.core_tolerance: float = 0.3
		self.core_tolerance: float = 0.2
		self.core_pcts: List[float] = []
		self.base_right_x: List[float] = []
		self.base_right_y: List[float] = []
		self.base_left_x: List[float] = []
		self.base_left_y: List[float] = []
		self.base_pcts: List[float] = []
		self.convertible_to_left_x: List[float] = []
		self.convertible_to_left_y: List[float] = []
		self.convertible_to_right_x: List[float] = []
		self.convertible_to_right_y: List[float] = []
		self.conv_pcts: List[float] = []
		self.likely_right_x: List[float] = []
		self.likely_right_y: List[float] = []
		self.likely_left_x: List[float] = []
		self.likely_left_y: List[float] = []
		self.like_pcts: list = []
		self.battleground_pcts: list = []
		self.dim1_pcts: list = []
		self.dim2_pcts: list = []
		self.dim1_div: float = 0.0
		self.dim2_div: float = 0.0
		self.tolerance: float = .25 	# Defines a battleground sector as a percent of connector on each side of bisector
		self.vector_head_width: float = .05
		self.vector_width: float = .01
		#
		# ----------------------------------------
		# from flags class - in the process of replacing these status indicators with tests of condition
		#
		self.have_clusters = False      # needs to be updated
		self.have_differences = False   # #########################
		self.have_factors = False      # needs to be updated
		self.have_scores = False
		#
		self.show_bisector = False
		self.show_connector = False
		self.show_just_reference_points = False
		self.show_reference_points = False   #  Not clear if this is ever used
		self.show_respondent_points = False
		# -----------------------------------------
		# from Conf class  - defines the characteristics of a configuration
		#
		self.ndim: int = 0
		self.npoint: int = 0
		self.npoints: int = 0
		self.dim_labels: List[str] = []
		self.dim_names: List[str] = []
		self.point_coords = pd.DataFrame()
		self.point_labels: List[str] = []
		self.point_names: List[str] = []
		self.range_dims = []
		self.range_points = []
		# ----------------------------------------
		# from Groups
		#
		# Groups class defines the characteristics of a set of groups
		self.dim_labels_grpd: List[str] = []
		self.dim_names_grpd: List[str] = []
		self.file_handle: str = ""		# the handle of the grouped configuration file
		self.grouping_var: str = ""
		self.ndim_grpd: int = 0
		self.npoint_grpd: int = 0
		self.point_codes_grpd: List = []
		self.point_coords_grpd = pd.DataFrame()
		self.point_labels_grpd: List[str] = []
		self.point_names_grpd: List[str] = []
		self.range_dims_grpd = []
		self.range_points_grpd = []

		# -----------------------------------------------
		# from  Parts class
		# The Parts class defines which parts are to be included in the next plot
		#
		self.hor_max: float = 0.0  # the horizontal maximum for plot (once all_hor_max)
		self.hor_min: float = 0.0  # the horizontal minimum for plot (once all_hor_min)
		self.vert_max: float = 0.0  # the vertical maximum for plot  (once all_vert_max)
		self.vert_min: float = 0.0  # the vertical minimum for plot  (once all_vert_min
		self.axis_extra: float = 0.1  # Addition to axis to keep points from falling on the edge of plots as % of abs (maximum coordinate)
		self.displacement: float = .04  # factor of largest axis length to move labeling off point to improve visibility
		self.hor_dim: int = 0  # Index of dimension to be shown on horizontal axis
		self.vert_dim: int = 1  # Index of dimension to be shown on vertical axis
		self.move_label: float = 0.0  # the amount labels will be moved to improve visibility   (???????)

		# ---------------------------------------------------------------------
		# from Dyad class:
		# The Dyad class contains measures about pairs of points

		self.a_item: List[int] = []		# the label of the first item in the dyad
		self.b_item: List[int] = []		# the label of the second item in the dyad
		self.correlations: list = []
		self.distances: List[float] = []
		self.distances_as_dict: Dict = dict()
		self.distances_as_list: List[float] = []
		self.item_labels: List = []
		self.item_names: List = []
		self.n_evaluations: int = 0
		self.n_evaluators: int = 0
		self.n_pairs: int = 0
		self.ndyad: int = 0
		self.nreferent: int = 0
		self.df = pd.DataFrame()  # Pandas data frame used for advanced computations on dyads
		self.ev = pd.DataFrame()  # Pandas data framer used for evaluations
		self.range_items = []
		self.range_items_less_one = []
		self.range_evaluators = []
		self.range_similarities = []
		self.range_pairs = []
		self.selection: str = "" 	# defines the columns in df to be displayed
		self.similarities = []
		self.similarities_as_dict = dict()
		self.similarities_as_list = []
		self.similarities_as_square = []
		self.sorted_distances = dict()
		self.sorted_distances_in_numpy = []
		self.sorted_similarities = dict()  # similarities dictionary sorted from smallest to largest
		self.zipped: List = []				# the similarities sorted by value with a and b labels
		self.a_x_alike: List = []
		self.a_y_alike: List = []
		self.b_x_alike: List = []
		self.b_y_alike: List = []
		self.best_stress: int = -1
		self.n_comp: int = 0
		self.use_metric = False
		self.min_stress: List = []

		#
		# -------------------------------------------------
		# from People class
		# The People class contains a set of variable about individual respondents/voters/people/customers
		# dim1 - 						the values for people on the variable to be displayed on horizontal axis
		# 									in plot of individuals or joint plot
		# self.dim1 = []
		# dim2 - 						the values for people on the variable to be displayed on vertical
		# 								 	axis in plot of individuals or joint plot
		# self.dim2 = []

		self.hor_axis_name: str = "Unknown"  # the name to be used on the horizontal axis in plot of individuals or joint plot
		self. hor_var: int = -1  # the index of the variable to be used on the horizontal axis in plot of individuals or joint plot
		self.ind_vars = pd.DataFrame()
		self.n_individ: int = 0
		self.nvar: int = 0		# the number of variables about the people
		self.point_size: int = 15		# the size of the dots representing people in scatterplots
		self.range_n_individ = []  # the range of individuals
		self.range_nvar = []		# the range of variables in individual data file
		self.respno: List = []
		self.var_labels: List[str] = []		# the labels of the variables about the people
		self.var_names: List[str] = []		# the names of the variables about the people
		self.vert_axis_name: str = "Unknown"  # the name to be used on the vertical axis in plot of individuals or joint plot
		self.vert_var: int = -1  # the index of the variable to be used on the vertical axis in plot of individuals or joint plot

		return

	# ----------------------------------------------------------------------------------

	def assign_to_segments(self):

		#
		# Every case is assigned to all (each) types of segments
		# Segment types are not mutually exclusive
		# Within segment type segments are mutually exclusive
		# The total for each type of segment should be the same
		#
		# Set variables needed
		# n_core_1 = 0
		# n_core_2 = 0
		# n_core_3 = 0
		#
		# self.ends_of_bisector_function()
		# print(f"DEBUG - at top of assign {self.bisector.direction = }")
		# print(f"DEBUG - at top of assign {self.bisector.start_y = }")
		# print(f"DEBUG - at top of assign {self.east.start_y = }")
		# print(f"DEBUG - at top of assign {self.west.start_y = }")
		#
		self.seg = pd.DataFrame(columns=[
			"Dim1_score", "Dim2_score",
			"Base", "Convertible", "Core", "Likely", "Battle_ground", "Only_Dim1", "Only_Dim2"])
		pd.set_option('display.max_columns', 10)
		#
		# self.seg.pop()
		#"Dim2_score",
		#	"Dim1_score", "Base", "Convertible", "Core", "Likely", "Battle_ground", "Only_Dim1", "Only_Dim2")
		# print(f"DEBUG in assign - \n {self.seg = }")
		self.seg["Dim1_score"] = self.dim1
		self.seg["Dim2_score"] = self.dim2
		# print(f"DEBUG2 in assign - \n {self.seg = }")
		#
		# Determine base left and base right  segments -------------------------------------
		#
		for each_indiv in self.range_n_individ:
			west_connector_cross_x = (
					self.seg.loc[each_indiv, "Dim2_score"]
					- self.west.intercept) \
					/ self.west.slope
			east_connector_cross_x = (
					self.seg.loc[each_indiv, "Dim2_score"]
					- self.east.intercept) \
					/ self.east.slope
			if self.bisector.direction == "Flat":
				if self.seg.loc[each_indiv, "Dim2_score"] < self.west.start_y:
					self.seg.loc[each_indiv, "Base"] = 1
				elif self.east.start_y > self.seg.loc[each_indiv, "Dim2_score"] > self.west.start_y:
					self.seg.loc[each_indiv, "Base"] = 2
				else:
					self.seg.loc[each_indiv, "Base"] = 3
			#
			if self.bisector.direction == "Vertical":
				if self.seg.loc[each_indiv, "Dim1_score"] < self.west.start_x:
					self.seg.loc[each_indiv, "Base"] = 1
				elif self.seg.loc[each_indiv, "Dim1_score"] > self.east.start_x:
					self.seg.loc[each_indiv, "Base"] = 3
				else:
					self.seg.loc[each_indiv, "Base"] = 2
			#
			elif self.bisector.direction == "Upward slope":
				if self.seg.loc[each_indiv, "Dim1_score"] < west_connector_cross_x:
					self.seg.loc[each_indiv, "Base"] = 1
				elif east_connector_cross_x > self.seg.loc[each_indiv, "Dim1_score"] > west_connector_cross_x:
					self.seg.loc[each_indiv, "Base"] = 2
				else:
					self.seg.loc[each_indiv, "Base"] = 3
			#
			elif self.bisector.direction == "Downward slope":
				if self.seg.loc[each_indiv, "Dim1_score"] < west_connector_cross_x:  # switched side
					self.seg.loc[each_indiv, "Base"] = 1
				elif east_connector_cross_x > self.seg.loc[each_indiv, "Dim1_score"] > west_connector_cross_x:  # switched side twice
					self.seg.loc[each_indiv, "Base"] = 2
				else:
					self.seg.loc[each_indiv, "Base"] = 3
		#
		# Determine convertible_to_left and convertible_to_right  segments --------------------------
		#
		for each_indiv in self.range_n_individ:
			bisector_x = (
				self.seg.loc[each_indiv, "Dim2_score"]
				- self.bisector.intercept) \
				/ self.bisector.slope
			west_connector_cross_x = (
				self.seg.loc[each_indiv, "Dim2_score"]
				- self.west.intercept) \
				/ self.west.slope
			east_connector_cross_x = (
				self.seg.loc[each_indiv, "Dim2_score"]
				- self.east.intercept) \
				/ self.east.slope
			#
			if self.bisector.direction == "Flat":
				if self.bisector.start_y < self.seg.loc[each_indiv, "Dim2_score"] < self.east.start_y:
					self.seg.loc[each_indiv, "Convertible"] = 1

				elif self.bisector.start_y > self.seg.loc[each_indiv, "Dim2_score"] > self.west.start_y:
					self.seg.loc[each_indiv, "Convertible"] = 2
					# if each_indiv < 100:
						# print(f"DEBUG -- conv Dim2: {self.seg.loc[each_indiv, 'Dim2_score']}")

				else:
					self.seg.loc[each_indiv, "Convertible"] = 3
			elif self.bisector.direction == "Vertical":
				if self.bisector.start_x < self.seg.loc[each_indiv, "Dim1_score"] < self.east.start_x:
					self.seg.loc[each_indiv, "Convertible"] = 1
				elif self.bisector.start_x > self.seg.loc[each_indiv, "Dim1_score"] > self.west.start_x:
					self.seg.loc[each_indiv, "Convertible"] = 2
				else:
					self.seg.loc[each_indiv, "Convertible"] = 3
			#
			elif self.bisector.direction == "Upward slope":
				if bisector_x < self.seg.loc[each_indiv, "Dim1_score"] < east_connector_cross_x:
					self.seg.loc[each_indiv, "Convertible"] = 1
				elif bisector_x > self.seg.loc[each_indiv, "Dim1_score"] > west_connector_cross_x:
					self.seg.loc[each_indiv, "Convertible"] = 2
				else:
					self.seg.loc[each_indiv, "Convertible"] = 3
			elif self.bisector.direction == "Downward slope":
				if bisector_x < self.seg.loc[each_indiv, "Dim1_score"] < east_connector_cross_x:  # switched side
					self.seg.loc[each_indiv, "Convertible"] = 1
				elif bisector_x > self.seg.loc[each_indiv, "Dim1_score"] > west_connector_cross_x:  # switched side
					self.seg.loc[each_indiv, "Convertible"] = 2
				else:
					self.seg.loc[each_indiv, "Convertible"] = 3

		#
		# Determine Core left and core right  segments -------------------------------------------
		#
		if self.bisector.direction == "Flat":
			# here if the bisector is flat
			if self.point_coords.iloc[self.rival_a][self.vert_dim] \
					> self.point_coords.iloc[self.rival_b][self.vert_dim]:
				# here if rival a is higher vertically than rival b
				left_x = self.point_coords.iloc[self.rival_a][self.hor_dim]
				left_y = self.point_coords.iloc[self.rival_a][self.vert_dim]
				right_x = self.point_coords.iloc[self.rival_b][self.hor_dim]
				right_y = self.point_coords.iloc[self.rival_b][self.vert_dim]
			else:
				# here if rival a is NOT higher vertically than rival b
				left_x = self.point_coords.iloc[self.rival_b][self.hor_dim]
				left_y = self.point_coords.iloc[self.rival_b][self.vert_dim]
				right_x = self.point_coords.iloc[self.rival_a][self.hor_dim]
				right_y = self.point_coords.iloc[self.rival_a][self.vert_dim]
		# here if bisector is NOT Flat
		elif self.point_coords.iloc[self.rival_a][self.hor_dim] \
				< self.point_coords.iloc[self.rival_b][self.hor_dim]:
			# here if rival a is more westward than rival b
			left_x = self.point_coords.iloc[self.rival_a][self.hor_dim]
			left_y = self.point_coords.iloc[self.rival_a][self.vert_dim]
			right_x = self.point_coords.iloc[self.rival_b][self.hor_dim]
			right_y = self.point_coords.iloc[self.rival_b][self.vert_dim]
		else:
			# here is rival a is NOT more westward than rival b
			left_x = self.point_coords.iloc[self.rival_b][self.hor_dim]
			left_y = self.point_coords.iloc[self.rival_b][self.vert_dim]
			right_x = self.point_coords.iloc[self.rival_a][self.hor_dim]
			right_y = self.point_coords.iloc[self.rival_a][self.vert_dim]
		#
		for each_indiv in self.range_n_individ:
			target_x = self.seg.loc[each_indiv, "Dim1_score"]
			target_y = self.seg.loc[each_indiv, "Dim2_score"]
			dist_to_left = self.distance_between_points(
				left_x, left_y, target_x, target_y)
			dist_to_right = self.distance_between_points(
				right_x, right_y, target_x, target_y)
			#
			if dist_to_left < self.core_radius:
				self.seg.loc[each_indiv, "Core"] = 1
				# n_core_1 += 1
			elif dist_to_right < self.core_radius:
				self.seg.loc[each_indiv, "Core"] = 3
				# n_core_3 += 1
			else:
				self.seg.loc[each_indiv, "Core"] = 2
				# n_core_2 += 1
	#
	# Determine battleground and settled segments ----------------------------------------------
	#
		for each_indiv in self.range_n_individ:
			west_connector_cross_x = (
				self.seg.loc[each_indiv, "Dim2_score"]
				- self.west.intercept) \
				/ self.west.slope
			east_connector_cross_x = (
				self.seg.loc[each_indiv, "Dim2_score"]
				- self.east.intercept) \
				/ self.east.slope
			if self.bisector.direction == "Flat":
				if self.east.intercept < self.seg.loc[each_indiv, "Dim2_score"] < self.west.intercept:
					self.seg.loc[each_indiv, "Battle_ground"] = 1
				else:
					self.seg.loc[each_indiv, "Battle_ground"] = 2
			elif self.bisector.direction == "Vertical":
				if self.east.start_x > self.seg.loc[each_indiv, "Dim1_score"] > self.west.start_x:
					self.seg.loc[each_indiv, "Battle_ground"] = 1
				else:
					self.seg.loc[each_indiv, "Battle_ground"] = 2
			elif self.bisector.direction == "Upward slope":
				if east_connector_cross_x > self.seg.loc[each_indiv, "Dim1_score"] > west_connector_cross_x:
					self.seg.loc[each_indiv, "Battle_ground"] = 1
				else:
					self.seg.loc[each_indiv, "Battle_ground"] = 2
			elif self.bisector.direction == "Downward slope":
				if east_connector_cross_x > self.seg.loc[each_indiv, "Dim1_score"] > west_connector_cross_x:  # switched side twice
					self.seg.loc[each_indiv, "Battle_ground"] = 1
				else:
					self.seg.loc[each_indiv, "Battle_ground"] = 2
	#
	# Determine Only Dim1 and Dim2 segments ----------------------------------------------
	#
		for each_indiv in self.range_n_individ:
			# Dim1
			if self.seg.loc[each_indiv, "Dim1_score"] < self.dim1_div:
				self.seg.loc[each_indiv, "Only_Dim1"] = 1
			else:
				self.seg.loc[each_indiv, "Only_Dim1"] = 2
			# Dim2
			if self.dim2.loc[each_indiv] > self.dim2_div:
				self.seg.loc[each_indiv, "Only_Dim2"] = 1
			else:
				self.seg.loc[each_indiv, "Only_Dim2"] = 2
		#
		# Determine Likely  segments ----------------------------------------
		#
		if self.bisector.direction == "Flat":
			for each_indiv in self.range_n_individ:
				if self.seg.loc[each_indiv, "Dim2_score"] \
						< self.bisector.intercept:
					self.seg.loc[each_indiv, "Likely"] = 1
				else:
					self.seg.loc[each_indiv, "Likely"] = 2
		else:
			for each_indiv in self.range_n_individ:
				if self.seg.loc[each_indiv, "Dim1_score"] \
						< (self.seg.loc[each_indiv, "Dim2_score"] - self.bisector.intercept) \
						/ self.bisector.slope:
					# likely left
					self.seg.loc[each_indiv, "Likely"] = 1
				else:
					self.seg.loc[each_indiv, "Likely"] = 2

		#
		# gets percent for codes appearing in series
		# will not return 0.0 if code does not appear in series
		#
		self.base_pcts = self.seg.value_counts("Base", normalize=True, sort=False)*100
		self.conv_pcts = self.seg.value_counts("Convertible", normalize=True, sort=False)*100
		self.core_pcts = self.seg.value_counts("Core", normalize=True, sort=False)*100
		self.like_pcts = self.seg.value_counts("Likely", normalize=True, sort=False)*100
		self.battleground_pcts = self.seg.value_counts("Battle_ground", normalize=True, sort=False)*100
		self.dim1_pcts = self.seg.value_counts("Only_Dim1", normalize=True, sort=False)*100
		self.dim2_pcts = self.seg.value_counts("Only_Dim2", normalize=True, sort=False)*100

		# print(f"DEBUG -- {self.core_pcts = }")
		#
		# Needed to capture any code that does not appear in pcts
		# ensures percent is 0.0 for any missing code
		#
		for i in range(1, 3):
			self.base_pcts.loc[i] = self.base_pcts.get(i, 0.0)
			self.conv_pcts.loc[i] = self.conv_pcts.get(i, 0.0)
			self.core_pcts.loc[i] = self.core_pcts.get(i, 0.0)

		for i in range(1, 2):
			self.like_pcts.loc[i] = self.like_pcts.get(i, 0.0)
			self.battleground_pcts.loc[i] = self.battleground_pcts.get(i, 0.0)
			self.dim1_pcts.loc[i] = self.dim1_pcts.get(i, 0.0)
			self.dim2_pcts.loc[i] = self.dim2_pcts.get(i, 0.0)

		# print(f"DEBUG -- {self.core_pcts = }")
		#
		# Ensures order of pcts
		#
		self.base_pcts.sort_index(inplace=True)
		self.conv_pcts.sort_index(inplace=True)
		self.core_pcts.sort_index(inplace=True)
		self.like_pcts.sort_index(inplace=True)
		self.battleground_pcts.sort_index(inplace=True)
		self.dim1_pcts.sort_index(inplace=True)
		self.dim2_pcts.sort_index(inplace=True)

		# print(f"DEBUG -- {self.core_pcts = }")

		return

	# ----------------------------------------------------------------------------------

	def bisector_function(self, rival_a, rival_b):
		#
		#  Determine midpoint of connector (line between reference points)
		#
		self.connector_bisector_cross_x = (
			self.point_coords.iloc[rival_a][self.hor_dim]
			+ self.point_coords.iloc[rival_b][self.hor_dim]
			) / 2
		self.connector_bisector_cross_y = (
			self.point_coords.iloc[rival_a][self.vert_dim]
			+ self.point_coords.iloc[rival_b][self.vert_dim]
			) / 2
		#
		# Determine length of connector
		#
		sumofsqs = (
			self.point_coords.iloc[rival_a][self.hor_dim]
			- self.point_coords.iloc[rival_b][self.hor_dim]) \
			* (
			self.point_coords.iloc[rival_a][self.hor_dim]
			- self.point_coords.iloc[rival_b][self.hor_dim]) \
			+ (
			self.point_coords.iloc[rival_a][self.vert_dim]
			- self.point_coords.iloc[rival_b][self.vert_dim]) \
			* (
			self.point_coords.iloc[rival_a][self.vert_dim]
			- self.point_coords.iloc[rival_b][self.vert_dim])
		self.connector.length = math.sqrt(sumofsqs)
		self.core_radius = self.connector.length * self.core_tolerance
		#
		# Determine slope and intercept of line joining reference points
		#
		# To avoid dividing by zero use FUDGE factor to minimally move from zero
		#
		if self.connector.direction == "Vertical":
			self.connector.slope = self.connector.SLOPE_FUDGE
		else:
			self.connector.slope = \
				(
					(
						self.point_coords.iloc[rival_b][self.vert_dim]
						- self.point_coords.iloc[rival_a][self.vert_dim]
					)
					/
					(
						self.point_coords.iloc[rival_b][self.hor_dim]
						- self.point_coords.iloc[rival_a][self.hor_dim]
						+ self.connector.SLOPE_FUDGE
					)
				)
		self.connector.intercept = self.connector_bisector_cross_y \
			- (self.connector.slope * self.connector_bisector_cross_x)

		# trial removal of next statement which did NOT exist in SPACES.py ???????????

		# self.bisector.intercept = self.connector_bisector_cross_y
		#
		#  Determine slope and intercept of perpendicular bisector
		#
		self.connector.slope = self.connector.slope + self.connector.SLOPE_FUDGE
		if self.connector.direction == "Flat":
			self.bisector.slope = (-1) * (1 / self.connector.slope)
			self.bisector.intercept = self.connector_bisector_cross_y \
				- (self.bisector.slope * self.connector_bisector_cross_x)
		elif self.connector.direction == "Vertical":
			self.bisector.intercept = self.connector_bisector_cross_y
			self.bisector.slope = (- 1) * (1 / self.connector.slope)
		else:
			self.bisector.slope = (- 1) * (1 / self.connector.slope)
			self.bisector.intercept = self.connector_bisector_cross_y - \
				(self.bisector.slope * self.connector_bisector_cross_x)
		#
		# Set connector and bisector direction
		#
		#
		self.set_direction_flags()
		#
		# print(f"DEBUG - at bottom of bisect - {self.bisector.start_y = }")
		# print(f"DEBUG - at bottom of bisect - {self.connector_bisector_cross_y = }")
	# ---------------------------------------------------------------------------

	def center(self):

		# dim_avg.clear()
		#
		# Adjust each coordinate by removing dimension mean from each coordinate
		#
		dim_avg = self.point_coords.mean()
		#
		for index_dim, each_dim in enumerate(dim_avg):
			for index_point, each_point in enumerate(self.point_labels):
				self.point_coords.iloc[index_point][index_dim]\
					= self.point_coords.iloc[index_point][index_dim]\
					- dim_avg[index_dim]
# -----------------------------------------------------------------------------------------------------------

	def choose_a_side_function(self):
		"""The choose a side function - randomly chooses a side.
		\nIf a line goes through a corner this randomly assigns it as going through one side.
		\nArguments -
		\nNone
		\nReturned variables -
		\nside_1:  "Yes" or "No" indicating the line will be considered	as going through side_1
		\nside_2: "Yes" or "No" indicating whether the line will be considered as going through side_2
		"""

		#
		# randomly set first to be true or false
		#
		first = random.choice([True, False])
		#
		if first:
			side_1 = "Yes"
			side_2 = "No"
		else:
			side_1 = "No"
			side_2 = "Yes"

		return side_1, side_2


# -------------------------------------------------------------------------------------------------

	def distance_between_points(self, point_1_x, point_1_y, point_2_x, point_2_y):
		""" Distance between points function - calculates distance between two points.
		"""

		diffs = []
		sqs = []
		sumofsqs = 0
		#
		# Calculate the distance between pair of points
		#
		sq_x = (point_1_x - point_2_x) * (point_1_x - point_2_x)
		sq_y = (point_1_y - point_2_y) * (point_1_y - point_2_y)
		sumofsqs = sq_x + sq_y
		# Take the sq root of the sum of squares
		dist_pts = math.sqrt(sumofsqs)
		#
		return dist_pts
# ---------------------------------------------------------------------------------------

	def dividers(self):
		### the point on each dimension that separates rivals on that dimension
		###
		# print(f"DEBUG -- A ({self.point_coords.iloc[self.rival_a, 0]}, {self.point_coords.iloc[self.rival_a, 1]})")
		# print(f"DEBUG -- B ({self.point_coords.iloc[self.rival_b, 0]}, {self.point_coords.iloc[self.rival_b, 1]})")
		diff_dim1 = self.point_coords.iloc[self.rival_b, 0] - self.point_coords.iloc[self.rival_a, 0]
		diff_dim2 = self.point_coords.iloc[self.rival_b, 1] - self.point_coords.iloc[self.rival_a, 1]
		sq_dim1 = diff_dim1 ** 2
		sq_dim2 = diff_dim2 ** 2
		half_dim1 = math.sqrt(sq_dim1) / 2
		half_dim2 = math.sqrt(sq_dim2) / 2
		min_dim1 = min(
			self.point_coords.iloc[self.rival_a, 0],
			self.point_coords.iloc[self.rival_b, 0])
		min_dim2 = min(
			self.point_coords.iloc[self.rival_a, 1],
			self.point_coords.iloc[self.rival_b, 1])
		self.dim1_div = min_dim1 + half_dim1
		self.dim2_div = min_dim2 + half_dim2

		return
# ------------------------------------------------------------------------------------------------

	def duplicate_similarities(self):
		#
		# Create similarities as a list and a dictionary keyed by a dyad

		from_points = range(1, self.nreferent)
		for an_item in from_points:
			to_points = range(an_item)
			for another_item in to_points:
				self.similarities_as_list.append(self.similarities[an_item - 1][another_item])
				new_key = str(self.item_labels[another_item] + "_" + self.item_labels[an_item])
				self.a_item.append(self.item_labels[another_item])
				self.b_item.append(self.item_labels[an_item])
				self.similarities_as_dict[new_key] = self.similarities[an_item - 1][another_item]
		#
		self.ndyad = int((self.nreferent * (self.nreferent - 1) / 2))
		self.range_similarities = range(self.ndyad)
		sorted_similarities = dict(sorted(self.similarities_as_dict.items(), key=lambda x: x[1]))
		self.zipped = sorted(zip(self.similarities_as_list, self.a_item, self.b_item))
		#
		# create similarities-as-square a full, upper and lower, matrix,
		#
		self.range_items = range(len(self.item_labels))
		for each_item in self.range_items:
			self.similarities_as_square.append([])
			for other_item in self.range_items:
				if each_item == other_item:
					# self.similarities_as_square[each_item][other_item] = 0.0
					self.similarities_as_square[other_item].append(0.0)
				elif each_item < other_item:
					index = str(self.item_labels[each_item] + "_" + self.item_labels[other_item])
					self.similarities_as_square[each_item].append(self.similarities_as_dict[index])
				else:
					index = str(self.item_labels[other_item] + "_" + self.item_labels[each_item])
					self.similarities_as_square[each_item].append(self.similarities_as_dict[index])
		# sim_in_numpy = np.array(self.zipped)
		# temp = sim_in_numpy.argsort(axis=0)
		# ranks = temp.argsort(axis=0)

# ------------------------------------------------------------------------------------------------------

	def ends_of_bisector_function(self):
		""" ends of bisector function - determines the coordinates of the endpoints of the bisector.
		"""
		#
		# Initialize variables needed
		#
		self.bisector.goes_through_top = "No"
		self.bisector.goes_through_bottom = "No"
		self.bisector.goes_through_right_side = "No"
		self.bisector.goes_through_left_side = "No"
		self.bisector.start_x = 0.0
		self.bisector.start_y = 0.0
		self.bisector.end_x = 0.0
		self.bisector.end_y = 0.0
		# bisector passes through top or bottom
		#   y at max x
		self.bisector.y_at_max_x = (self.bisector.slope * self.hor_max) + self.bisector.intercept
		#   y at min x
		self.bisector.y_at_min_x = (self.bisector.slope * self.hor_min) + self.bisector.intercept
		#
		# bisector passes through right or left side
		if self.bisector.slope == 0.0:
			self.bisector.slope = self.bisector.SLOPE_FUDGE
		#    x at max y
		self.bisector.x_at_max_y = (self.vert_max - self.bisector.intercept) / self.bisector.slope
		# 	x at min y
		self.bisector.x_at_min_y = (self.vert_min - self.bisector.intercept) / self.bisector.slope
		#
		# Add bisector to plot
		#
		# Determine whether bisector goes through edge - top/bottom, right/left
		#
		if self.bisector.direction == "Flat":
			self.bisector.goes_through_right_side = "Yes"
			self.bisector.goes_through_left_side = "Yes"
		#
		else:
			if self.hor_min < self.bisector.x_at_max_y < self.hor_max:
				self.bisector.goes_through_top = "Yes"
			if self.hor_min < self.bisector.x_at_min_y < self.hor_max:
				self.bisector.goes_through_bottom = "Yes"
			if self.vert_min < self.bisector.y_at_max_x < self.vert_max:
				self.bisector.goes_through_right_side = "Yes"
			if self.vert_min < self.bisector.y_at_min_x < self.vert_max:
				self.bisector.goes_through_left_side = "Yes"
		#
		# Handle lines going through corners
		#
		# upper right
		if self.bisector.x_at_max_y == self.hor_max \
			and self.bisector.y_at_max_x == self.vert_max:
			(
				self.bisector.goes_through_top,
				self.bisector.goes_through_right_side
			) = self.choose_a_side_function()
		# upper left
		if self.bisector.x_at_max_y == self.hor_min \
			and self.bisector.y_at_min_x == self.vert_max:
			(
				self.bisector.goes_through_top,
				self.bisector.goes_through_left_side
			) = self.choose_a_side_function()
		# lower right
		if self.bisector.x_at_min_y == self.hor_max \
			and self.bisector.y_at_max_x == self.vert_min:

			(
				self.bisector.goes_through_bottom,
				self.bisector.goes_through_right_side
			) = self.choose_a_side_function()
		# lower left
		if self.bisector.x_at_min_y == self.hor_min \
			and self.bisector.y_at_min_x == self.vert_min:
			(
				self.bisector.goes_through_bottom,
				self.bisector.goes_through_left_side
			) = self.choose_a_side_function()
		#
		self.set_bisector_case()
		#
		match self.bisector.case:
			#
			# Bisector Case 0a Bisector slope is zero from Left side to Right side
			#
			case "0a":
				self.bisector.start_x = self.hor_min
				self.bisector.start_y = self.connector_bisector_cross_y
				self.bisector.end_x = self.hor_max
				self.bisector.end_y = self.connector_bisector_cross_y
			#
			# Bisector 1 Case 0b Connector slope is zero - from top to bottom
			#
			case "0b":
				self.bisector.start_x = self.connector_bisector_cross_x
				self.bisector.start_y = self.vert_max
				self.bisector.end_x = self.connector_bisector_cross_x
				self.bisector.end_y = self.vert_min
			#
			# Bisector Case Ia Positive slope from Left side to Right side and min_y > vert_min
			#
			case "Ia":
				self.bisector.start_x = self.hor_min
				self.bisector.start_y = self.bisector.y_at_min_x
				self.bisector.end_x = self.hor_max
				self.bisector.end_y = self.bisector.y_at_max_x
		#
		# Bisector Case IIa Positive slope from Left side to Top and max_y == vert_max
		#
			case "IIa":
				self.bisector.start_x = self.hor_min
				self.bisector.start_y = self.bisector.y_at_min_x
				self.bisector.end_x = self.bisector.x_at_max_y
				self.bisector.end_y = self.vert_max
		#
		# Bisector Case IIIa Positive slope from Bottom to Right side
		#
			case "IIIa":
				self.bisector.start_x = self.bisector.x_at_min_y
				self.bisector.start_y = self.vert_min
				self.bisector.end_x = self.hor_max
				self.bisector.end_y = self.bisector.y_at_max_x
		#
		# Bisector Case IVa Positive slope from Bottom to Top and min_x < hor_min
		#
			case "IVa":
				self.bisector.start_x = self.bisector.x_at_min_y
				self.bisector.start_y = self.vert_min
				self.bisector.end_x = self.bisector.x_at_max_y
				self.bisector.end_y = self.vert_max
		#
		# Bisector Case Ib Negative slope from Left side to Right side
		#
			case "Ib":
				self.bisector.start_x = self.hor_min
				self.bisector.start_y = self.bisector.y_at_min_x
				self.bisector.end_x = self.hor_max
				self.bisector.end_y = self.bisector.y_at_max_x
		#
		# Bisector Case IIb Negative slope from Left side to Bottom and min_y == vert_min
		#
			case "IIb":
				self.bisector.start_x = self.hor_min
				self.bisector.start_y = self.bisector.y_at_min_x
				self.bisector.end_x = self.bisector.x_at_min_y
				self.bisector.end_y = self.vert_min
		#
		# Bisector Case IIIb Negative slope from Top to Right side and min_x < hor_min ?????????
		#
			case "IIIb":
				self.bisector.start_x = self.bisector.x_at_max_y
				self.bisector.start_y = self.vert_max
				self.bisector.end_x = self.hor_max
				self.bisector.end_y = self.bisector.y_at_max_x
		#
		# Bisector Case IVb Negative slope from Bottom to Top and max_y > vert_max  ????????????????????????
		#
			case "IVb":
				self.bisector.start_x = self.bisector.x_at_max_y
				self.bisector.start_y = self.vert_max
				self.bisector.end_x = self.bisector.x_at_min_y
				self.bisector.end_y = self.vert_min
		#
		# Ready to add bisector to plot
		#
		# Correct outliers
		#
		if self.bisector.start_x > self.hor_max:
			self.bisector.start_x = self.hor_max
		if self.bisector.start_y > self.vert_max:
			self.bisector.start_y = self.vert_max
		if self.bisector.end_x < self.hor_min:
			self.bisector.end_x = self.hor_min
		if self.bisector.end_y < self.vert_min:
			self.bisector.end_y = self.vert_min
		#
		# Ready to add bisector to plot
		#
		return
	# ------------------------------------------------------------------------------

	def evaluate(self):

		#
		(self.n_individ, self.nreferent) = self.evaluations.shape
		self.npoints = self.nreferent
		self.range_points = range(self.npoints)
		self.range_items = range(self.nreferent)
		self.item_names = self.evaluations.columns.tolist()
		self.point_names = self.item_names
		self.range_similarities = range(len(self.item_names))

	# ---------------------------------------------------------------------------

	def explain(self, command):
		#
		if self.include_explanation():
			match command:
				case "Alike":
					if self.value_type == "similarities":
						include = "above"
					else:
						include = "below"
					print(
						"\n\tThe Alike command can be used to place lines between points with high similarity." +
						"\n\n\tThe user will be shown a histogram of similarities and will be asked for a cutoff value." +
						f"\n\tOnly pairs of points with a similarity {include} the cutoff will have a line joining the points."
					)
				case "Base":
					print(
						"\n\tThe Base command identifies regions closer to the reference points than the battleground region." +
						"\n\tIndividuals in these areas prefer these candidates."
					)
				case "Bisector":
					print(
						"\n\tThe Bisector command is used to create a plot of the active configuration." +
						"\n\tA line, the bisector, will be added to the plot to divide the space between the two reference points."
					)
				case "Center":
					print(
						"\n\tThe Center command is used to shift points to be centered around the origin." +
						"\n\tThis is achieved by subtracting the mean of each dimension from each coordinate" +
						"\n\tThis is especially useful when the coordinates are latitudes and longitudes."
					)
				case "Cluster":
					print(
						"\n\tThe Cluster command is used to assign points to clusters." +
						"\n\tThe assignments can be read from a file or determined by a clustering algorithm."
					)
				case "Compare":
					print(
						"\n\tThe Compare command is used to compare the target configuration " +
						"with the active configuration." +
						"\n\tThe target configuration will be centered to facilitate comparison."
						"\n\tThe active configuration will be rotated and transformed to minimize the differences " +
						"with the target configuration." +
						"\n\tA measure of the difference, disparity, will be computed"
						"\n\tThe result will be plotted with a line connecting corresponding points." +
						"\n\t\tThe line will be labeled with the label for the point." +
						"\n\t\tThe line will have zero length when the configurations match perfectly. " +
						"\n\t\tThe point from the rotated configuration with be labeled with an R. " +
						"\n\t\tThe point from the target congratulation will be labeled with a T."
					)
				case "Configuration":
					print(
						"\n\tThe configuration command reads in a configuration file." +
						"\n\tThe file must be formatted correctly." +
						"\n\tThe first line should be Configuration." +
						"\n\tThe next line should have two fields:" +
						"\n\t\tthe number of dimensions and " +
						"\n\t\tthe number of points." +
						"\n\tFor each point:" +
						"\n\t\tA line containing the label, a semicolon, and the full name for that point." +
						"\n\t\tA line with the coordinate for the point on each dimension separated by commas."
					)
				case "Contest":
					print(
						"\n\tThe Contest command identifies regions defined by the reference points."
					)
				case "Convertibles":
					print(
						"\n\tThe Convertible command identifies regions where" +
						"\n\tindividuals might be converted and switch their preference."
					)
				case "Core supporters":
					print(
						"\n\tThe Core supporters command identifies regions immediately around the reference points." +
						"\n\tIndividuals in these areas prefer these candidates the most."
					)
				case "Correlations":
					print(
						"\n\tThe Correlations command reads in a correlation matrix from a file." +
						"\n\n\tThe file must be in the a format similar to to the OSIRIS format." +
						"\n\tThe correlations may be used as similarities but more likely are used as input to Factor." +
						"\n\n\tThe correlations are stored as similarities and treated as measures of similarity.")
				case "Create":
					print(
						"\n\tThe Create command is used to build the" +
						"\n\tactive configuration by using user supplied information." +
						"\n\tIn addition to creating names and labels, coordinates" +
						"\n\tcan be supplied by the user, assigned randomly, or use" +
						"\n\tthe classic approach of using the order of the points."
					)
				case "Deactivate":
					print(
						"\n\tThe Deactivate command is used to deactivate the active configuration, existing similarities, and" +
						"\n\texisting correlations."
					)
				case "Differences":
					print(
						"\n\tThe Differences command is used to assess how much each " +
						"\n\tpoint is contributing to the lack of fit" +
						"\n\n\tIt displays for each pair of points the difference " +
						"\n\tbetween the inter-point distance " +
						"\n\tand the corresponding similarity measure." +
						"\n\tThis difference can be thought of as the pair's contribution to the overall lack of fit." +
						"\n\tThis can be done as ranks to keep the emphasis on ordinal " +
						"\n\tmeasurement rather than interval." +
						"\n\tCan this be done on their ranks rather than values????????"
					)
				case "Directions":
					print(
						"\n\tThe Directions command is used to display a plot showing the" +
						"\n\tdirection of each point from the origin to the unit circle."
					)
				case "Distances":
					print(
						"\n\tThe Distances command displays a matrix of inter-point distances." +
						"\n\n\tSome alternatives:" +
						"\n\t\tsimilarities could be shown above the diagonal" +
						"\n\t\toptionally ranks could be displayed in place of, or in addition to, values" +
						"\n\t\tinformation could be displayed as a table with a line for each pair of points"
					)
				case "Evaluations":
					print(
						"\n\tThe Evaluations command reads in a file containing " +
						"\n\tindividual evaluations corresponding to the points in the active configuration."
					)
				case "Factor":
					print(
						"\n\tThe Factor command creates a factor analysis of the current correlations." +
						"\n\t\tThe output is a factor matrix with as many points as in the correlation matrix," +
						"\n\t\tThe plot will have vectors from the origin to each point." +
						"\n\t\tIt displays a Scree diagram with the Eigenvalue for each" +
						"\n\t\t\tdimension.  The dimensions on the x-axis (1-n) and the Eigenvalue on the y-axis." +
						"\n\t\tIt asks the user how many dimensions to retain and uses" +
						"\n\t\t\t that to determine the number of dimensions to be retained."
					)
				case "First dimension":
					print(
						"\n\tThe First dimension command identifies regions defined by the first dimension."
					)
				case "Grouped":
					print(
						"\n\tThe Grouped command reads a file with coordinates for a set of groups on all dimensions." +
						"\n\t\tThe number of groups in a file should be small." +
						"\n\t\tThe number of dimensions must be the same as the active configuration" +
						"\n\t\tIf reference points have been established the user can add the points and the bisector."
					)
				case "History":
					print(
						"\n\tThe History command displays a list of commands used in this session."
					)
				case "Individuals":
					print(
						"\n\tThe Individuals command is used to establish variables," +
						"\n\tusually scores, and filters for a set of individuals."
					)
				case "Invert":
					print(
						"\n\tThe Invert command inverts dimension(s)." +
						"\n\tIt asks the user which dimension(s) to invert." +
						"\n\tIt multiples each point's coordinate on a dimension by minus one." +
						"\n\tThe resulting configuration becomes the active configuration."
					)
				case "Joint":
					print(
						"\n\tThe Joint command creates a plot including points" +
						"\n\t for individuals as well as points for referents." +
						"\n\tThe user decides whether to include the full active configuration" +
						"\n\tor just the reference points.  The user also decides whether to" +
						"\n\tinclude the perpendicular bisector between the reference points."
					)
				case "Likely supporters":
					print(
						"\n\tThe Likely supporters command identifies regions immediately around the reference points." +
						"\n\tIndividuals in these areas prefer these candidates."
					)
				case "Line of Sight":
					print(
						"\n\tThe Line of Sight command computes the line of sight measure of association." +
						"\n\tLine of Sight is a measure of dissimilarity.  It was developed by George Rabinowitz"
					)
				case "Battleground":
					print(
						"\n\tThe Battleground command is used to define a region of " +
						"\n\t\tbattleground points within a tolerance from bisector between" +
						"\n\t\treference points."
					)
				case "MDS":
					print(
						"\n\tThe MDS command is used to perform a metric or non-metric multidimensional scaling of the similarities." +
						"\n\tThe user will be presented with a Scree diagram and asked how many dimensions to retain????." +
						"\n\tThe result of MDS will become the active configuration."
					)
				case "Move":
					print(
						"\n\tThe Move command is used to add a constant to the coordinates along dimension(s)." +
						"\n\tThe user will be asked which dimension(s) to move." +
						"\n\tThe user will be asked for a value to be used." +
						"\tThe value, positive or negative, will be added to each of the point's coordinate on that dimension." +
						"\n\tThe resulting configuration becomes the active configuration."
					)
				case "Paired":
					print(
						"\n\tThe Paired command is used to obtain information about two points." +
						"\n\tThe user will be asked which points should be used." +
						"\n\tThe user will be asked for the labels of the points to be used."
					)
				case "Plane":
					print(
						"\n\tThe Plane command is used to define the plane to be displayed." +
						"\n\tThe plane defaults to the first two dimensions."
					)
				case "Plot":
					print(
						"\n\tThe Plot command is used to create a plot of the active configuration."
					)
				case "Principal Components":
					print(
						"\n\tThe Principal Components command is used to obtain the " +
						"\n\t\tdimensions corresponding to the axes having the highest explanatory power to" +
						"\n\t\tdescribe the correlations."
					)
				case "Ranks":
					print(
						"\n\tThe Ranks command is used to display ranks of the similarities and distances ." +
						"\n\tThis assumes the similarities have already been read in and distances have been established."
					)
				case "Reference points":
					print(
						"\n\tThe Reference points command is used to designate two points as reference points." +
						"\n\tThe user will be asked which points should be used as reference points." +
						"\n\tThe user will be asked for the labels of the points to be used as reference points." +
						"\n\tThe reference points will define the bisector shown in the Bisector command." +
						"\n\tThe plot of the active configuration will be shown with a line connecting the reference points." +
						"\n"
					)
				case "Rescale":
					print(
						"\n\tThe Rescale command is used to increase or decrease coordinates." +
						"\n\tThe user will be asked which dimension(s) should be rescaled." +
						"\n\tThe user will be asked for a value." +
						"\n\tThe coordinate of each point will be multiplied by the value." +
						"\n\tThe resulting configuration becomes the active configuration."
					)
				case "Rotate":
					print(
						"\n\tThe Rotate command will be used to rotate the current plane of the active configuration." +
						"\n\tThe user will be asked to enter the degrees to rotated." +
						"\n\tA positive value will indicate clockwise rotation." +
						"\n\tThe resulting configuration becomes the active configuration."
					)
				case "Sample designer":
					print(
						"\n\tThe Sample designer command creates a case selection matrix. " +
						"\n\tThe user will determine the number of respondents in the universe, " +
						"\n\tthe number of repetitions to be created and the probability the case will be included."
						"\n\tThe matrix will contain ones and zeroes to indicate whether whether " +
						"\n\ta case will be included in a replicate." +
						"\n\tThe matrix will contain a row for each respondent and a column for each repetition."
					)
				case "Save configuration":
					print(
						"\n\tThe Save configuration command is used to write the active configuration into a file." +
						"\n\tThe user will be asked for a file to be used."
					)
				case "Save target":
					print(
						"\n\tThe Save target command is used to write the target configuration into a file." +
						"\n\tThe user will be asked for a file to be used."
					)
				case "Scores":
					print(
						"\n\tThe Scores command reads in a file containing individual " +
						"\n\tscores corresponding to the dimensions in the active configuration."
					)
				case "Scree":
					print(
						"\n\tThe Scree command creates a diagram showing stress vs. dimensionality." +
						"\n\tThe Scree diagram is used to assess how many dimensions are needed to fit the similarities." +
						"\n\tThe number of dimensions will be displayed on the x-axis." +
						"\n\tThe measure of fit, stress, will be displayed on the y-axis." +
						"\n\tThe user will be asked how many dimensions should be retained."
					)
				case "Second dimension":
					print(
						"\n\tThe Second dimension command identifies regions defined by the second dimension."
					)
				case "Segment":
					print(
						"\n\tThe Segment command uses the individual" +
						"\n\tscores on the dimensions in the active configuration. " +
						"\n\tIt assigns individuals to a variety of segments" +
						"\n\tbased on the contest defined by the current" +
						"\n\treference points. It provides estimates of the" +
						"\n\tsize of each segment. Segments are not mutually" +
						"\n\texclusive."
					)
				case "Settings":
					print(
						"\n\tThe Settings command provides a mechanism to changes the " +
						"\n\tsettings used throughout the session."
					)
				case "Shepard":
					print(
						"\n\tThe Shepard command is used to create a Shepard Diagram." +
						"\n\tThe user will be asked which axis should be used to display" +
						"\n\tsimilarities. Depending on whether the measures represent" +
						"\n\tdissimilarity or similarity and which axis is used to" +
						"\n\tdisplay similarities, a line will rise or descend from" +
						"\n\tleft to right. Each point on the line represents the" +
						"\n\tdistance between a pair of items and their corresponding" +
						"\n\tsimilarity measure."
					)
				case "Similarities":
					print(
						"\n\tThe Similarities command is used to read in a matrix of similarities." +
						"\n\tThe similarities reflect how similar each item is to each other." +
						"\n\tThe user will be asked for a file name." +
						"\n\tThe file must be formatted correctly." +
						"\n\tThe first line should say 'Lower triangular'" +
						"\n\tThe next line should have one field, the number of items." +
						"\n\tIf there is an active configuration and the number of similarities " +
						"\n\t\twill be checked to ensure it is the same as the number of points" +
						"\n\t\tin the configuration " +
						"\n\tFor each item:" +
						"\n\t\tA line containing the label, a semicolon, and the full name for that item." +
						"\n\t\tA line with the similarity measures for that item and the items below it in the list of items." +
						"\n\tThe user will be asked whether values are measures of similarity or dissimilarity."
					)
				case "Status":
					print(
						"\n\tThe Status command is used to show the status of global variables."
					)
				case "Stress":
					print(
						"\n\tThe Stress command is used to identify points with high " +
						"\n\tcontribution to the lack of fit." +
						"\n\tThe user will be shown all the pairs that include a point in a plot " +
						"\n\tshowing rank of similarity and rank of distance between the points."
					)
				case "Target":
					print(
						"\n\tThe Target command establishes a target configuration." +
						"\n\tThe Compare command will perform a Procrustean rotation on the" +
						"\n\tactive configuration to orient it as closely as possible to the" +
						"\n\ttarget configuration.  The rotated configuration will become the " +
						"\n\tactive configuration."
					)
				case "Undo":
					print(
						"\n\tThe Undo command is used to return to the active configuration " +
						"\n\tas it existed prior to the last command"
					)
				case  "Varimax":
					print(
						"\n\tThe Varimax command is used to perform a Varimax rotation of the current configuration." +
						"\n\tThe resulting configuration becomes the active configuration."
					)
				case "Vectors":
					print(
						"\n\tThe Vectors command is used to display the vectors for the active configuration."
					)
				case "View configuration":
					print(
						"\n\tThe View configuration command is used to display the active configuration."
					)
				case "View correlations":
					print(
						"\n\tThe View correlations command is used to display the correlations."
					)
				case "View grouped data":
					print(
						"\n\tThe View grouped data command is used to display grouped data."
					)
				case "View similarities":
					print(
						"\n\tThe View similarities command is used to display the similarities."
					)
				case "View target":
					print(
						"\n\tThe View target command is used to display the target configuration."
					)

		#
		return None

	# -----------------------------------------------------------------------------

	def error(self, message, feedback):

		QMessageBox.warning(
			None,
			message,
			feedback
		)
		return None

	# -------------------------------------------------------------------------------

	def extrema(self, poison_pill):
		### extrema has been replaced by max_and_min
		###
		conf_max = self.point_coords.max(numeric_only=True).max()
		conf_min = self.point_coords.min(numeric_only=True).min()
		if abs(conf_min) > conf_max:
			conf_max = abs(conf_min)
		#
		if self.have_individual_data():
			ind_max = self.ind_vars.iloc[[1, 2]].max().max()   # hardwired to second and third columns
			ind_min = self.ind_vars.iloc[[1, 2]].min().min()   # hardwired to second and third columns
			if abs(ind_min) > ind_max:
				ind_max = abs(ind_min)
			if ind_max > conf_max:
				conf_max = ind_max

		return conf_max

	# --------------------------------------------------------------------------------------

	def factors_and_scores(self):

		self.fa = FactorAnalyzer(n_factors=self.ndim,
										rotation="varimax", is_corr_matrix=False)
		self.fa.fit(self.evaluations)
		self.range_dims = range(self.ndim)
		for each_dim in self.range_dims:
			self.dim_names.append("Factor " + str(each_dim + 1))
			self.dim_labels.append("FA" + str(each_dim + 1))
		#
		self.range_points = range(self.nreferent)
		for each_point in self.range_points:
			self.point_names.append(self.item_names[each_point])
			self.point_labels.append(self.item_names[each_point][0:4])
		#
		the_loadings = self.fa.loadings_
		#
		# print(f"DEBUG -- {the_loadings = }")
		self.loadings = pd.DataFrame(the_loadings,
			columns=self.dim_names, index=self.item_names)
		# print(f"DEBUG -- {self.loadings = }")
		#
		# self.loadings = pd.DataFrame.from_records(self.loadings,
		#	columns=self.dim_names, index=self.item_names)
		# print(f"DEBUG -- {self.loadings = }")
		#
		# self.point_coords = pd.DataFrame.from_records(self.loadings,
		#	columns=self.dim_names, index=self.item_names)
		self.point_coords = pd.DataFrame(self.loadings, columns=self.dim_names, index=self.item_names)
		# print(f"DEBUG in factors_and_scores -- {self.point_coords = }")
		#
		# Get the eigenvector and the eigenvalues
		ev, v = self.fa.get_eigenvalues()
		#
		self.eigen = pd.DataFrame(data=ev, columns=["Eigenvalue"], index=self.item_names)
		self.eigen_common = pd.DataFrame(data=v, columns=["Eigenvalue"], index=self.item_names)
		commonalities = self.fa.get_communalities()
		self.commonalities = pd.DataFrame(
			data=commonalities, columns=["Commonality"], index=self.item_names)
		self.factor_variance = pd.DataFrame(
			self.fa.get_factor_variance(), columns=self.dim_names,
			index=["Variance", "Proportional", "Cumulative"])
		self.uniquenesses = pd.DataFrame(
			self.fa.get_uniquenesses(), columns=["Uniqueness"], index=self.item_names)
		self.factor_scores = pd.DataFrame(
			self.fa.transform(self.evaluations), columns=self.dim_names)
		return

	# --------------------------------------------------------------------------------

	def have_active_configuration(self):
		#
		# Checks if dataframe is empty
		#
		if self.point_coords.empty:
			return False
		else:
			return True

	# ----------------------------------------------------------------------------------

	def have_bisector_info(self):

		if self.bisector.direction == "Unknown":
			return False
		return True

	# ----------------------------------------------------------------------------------------------

	def have_correlations(self):
		#
		# Checks if correlations data is empty
		#
		if not len(self.correlations) == 0:
			return True
		else:
			return False
	# ------------------------------------------------------------------------------------------------------------

	def have_distances(self):
		#
		# Checks if self.distances is empty
		#
		if len(self.distances) == 0:
			return False
		else:
			return True

	# ----------------------------------------------------------------------------------------------

	def have_evaluations(self):
		#
		# Checks if evaluations data is empty
		#
		# if not len(self.evaluations) == 0:
		if not self.evaluations.empty:
			return True
		else:
			return False
	# -----------------------------------------------------------------------------------------

	def have_grouped_data(self):
		#
		# Checks if grouped dataframe is empty
		#
		# if len(self.point_coords_grpd) == 0:
		if self.point_coords_grpd.empty:
			return False
		else:
			return True

	# ----------------------------------------------------------------------------------------------

	def have_individual_data(self):
		#
		# Checks if individual data is empty
		#
		# if not len(self.ind_vars) == 0:
		if not self.ind_vars.empty:
			return True
		else:
			return False

	# -------------------------------------------------------------------------------------

	def have_mds_results(self):
		#
		# Check if best_stress is reasonable
		#
		if self.best_stress == -1:
			return False
		else:
			return True

	# ---------------------------------------------------------------------------------

	def have_ranks(self):
		#
		# Checks if ranks dataframe is empty
		#
		# if len(self.df) == 0:
		if self.df.empty:
			return False
		else:
			return True

	# ----------------------------------------------------------------------------------

	def have_reference_points(self):
		if (
				self.rival_a == -1
				or self.rival_b == -1
		):
			return False
		else:
			return True

	# -----------------------------------------------------------------------------------

	def have_segments(self):

		try:
			if self.n_individ == 0:
				return False
			# if not self.n_likely_left == 0 \
			#		or not self.n_likely_right == 0:
			if not self.seg.empty:
				return True
			else:
				return False
		except AttributeError:
			return False

	# ---------------------------------------------------------------------------------------------------

	def have_similarities(self):
		#
		# Checks if self.similarities is empty
		#
		if len(self.similarities) == 0:
			return False
		else:
			return True

	# --------------------------------------------------------------------------------

	def have_target_configuration(self):
		# print(f"DEBUG -- At top of have_target_configuration *******************************")
		# print(f"DEBUG -- {self.point_coords = }")
		#
		# Checks if dataframe is empty ?????????????untested
		#
		if self.point_coords.empty:
			return False
		else:
			return True

	# ---------------------------------------------------------------------------

	def include_explanation(self):

		#
		# Searched for last use of Terse or Verbose
		#
		# print(f"\nDEBUG -- At top of include_explanation {len(self.commands_used) = }")

		if len(self.commands_used) > 0:
			if self.commands_used[0] == "Verbose" \
				and len(self.commands_used) == 1:
				return True
		#
		# cmds = len(self.commands_used) - 1  ??????????????????????????
		for i in reversed(range(len(self.commands_used))):
			# print(f"DEBUG -- {i = }")
			# print(f"DEBUG -- {self.commands_used[i] = }")
			if self.commands_used[i] == "Terse":
				# print("DEBUG -- In include_explanation returning False")
				return False
			if self.commands_used[i] == "Verbose":
				# print("DEBUG -- In include_explanation returning True")
				return True
			else:
				continue
		# print("DEBUG -- In include_explanation returning False")
		#
		return False

	# ----------------------------------------------------------------------------------

	def inter_point_distances(self):

		# self.ndims = 0 <<<<<<<<<<<<<<not in class
		# self.npoints = 0  <<<<<<<<<<not in class
		diffs = []
		sqs = []
		sumofsqs = 0
		self.distances.clear()
		self.distances_as_dict = dict()
		self.distances_as_list.clear()
		#
		# Calculate distance between each pair of points
		#
		a_row = []
		from_pts_range = range(1, self.npoint)
		for from_pts in from_pts_range:
			# a_row.clear() -------causes multiple failures
			a_row = []
			to_pts_range = range(from_pts)
			for to_pts in to_pts_range:
				for each_dim in self.range_dims:
					diffs.append(self.point_coords.iloc[from_pts][each_dim] - self.point_coords.iloc[to_pts][each_dim])
					sqs.append(
						(self.point_coords.iloc[from_pts][each_dim] - self.point_coords.iloc[to_pts][each_dim])
						* (self.point_coords.iloc[from_pts][each_dim] - self.point_coords.iloc[to_pts][each_dim]))
					sumofsqs += sqs[each_dim]
				# Takes the sq root of the sum of squares
				distpts = math.sqrt(sumofsqs)
				self.distances_as_list.append(distpts)
				a_row.append(distpts)
				dist_key = str(self.point_labels[to_pts] + "_" + self.point_labels[from_pts])
				self.distances_as_dict[dist_key] = distpts
				#
				# clears the variables needed to calculate next distance corresponding to the
				# next pair of points
				diffs = []
				sqs = []
				sumofsqs = 0
				# on to the next to
			# on to the next from
			self.distances.append(a_row)
		#
		self.sorted_distances = dict(sorted(self.distances_as_dict.items(), key=lambda x: x[1]))
		self.sorted_distances_in_numpy = self.sorted_distances
	#
	# Finished calculating distances

	# ---------------------------------------------------------------------------------------------

	def invert(self, which_dim):
		for each_point in self.range_points:
			self.point_coords.iloc[each_point][which_dim] \
				= self.point_coords.iloc[each_point][which_dim] * -1

# -------------------------------------------------------------------------------------------

	def los(self):
		self.nreferent = 0
		self.zipped = []
		self.value_type = "dissimilarities"
		alt = []

		reflect = "Yes"
		self.item_names.clear()
		self.item_labels.clear()
		self.similarities.clear()
		self.similarities_as_dict = dict()
		self.similarities_as_list.clear()
		self.similarities_as_square.clear()
		self.sorted_similarities = dict()

		self.a_item = []
		self.b_item = []
		unique_vals = []
		best_ranking = []

		sums_s_star = pd.DataFrame()
		diffs_d_star = pd.DataFrame()
		sumsort_s = pd.DataFrame()
		diffsort_d = pd.DataFrame()
		combo_b = pd.DataFrame()
		cum_b_hat = pd.DataFrame()
		ordered = pd.DataFrame()
		u_vals = pd.DataFrame()

		df = pd.read_csv("elections/2004/2004_therms.csv")
		print("\n\tLine of Sight command is using 2004_therms.csv")
		#
		(self.n_individ, self.nreferent) = df.shape
		self.npoints = self.nreferent
		self.range_points = range(self.npoints)
		self.range_items = range(self.nreferent)
		self.item_names = df.columns.tolist()
		self.point_names = self.item_names
		self.range_similarities = range(len(self.item_names))
		#
		for each_item in self.range_similarities:
			self.item_labels.append(self.item_names[each_item][0:4])
		# print(str(self.item_labels[each_item]+" "+self.item_names[each_item]))
		self.point_labels = self.item_labels
		if reflect == "Yes":
			col_max = pd.DataFrame.max(df)
			df = pd.DataFrame.max(col_max) - df

		n_items_less_one = self.nreferent - 1
		for an_item in range(n_items_less_one):
			to_pts = range(an_item + 1, self.nreferent)
			for another_item in to_pts:
				new_name = str(df.columns[an_item] + "_" + df.columns[another_item])
				sums_s_star[new_name] = df[str(df.columns[an_item])] + df[str(df.columns[another_item])]
				diffs_d_star[new_name] = abs(df[str(df.columns[an_item])] - df[str(df.columns[another_item])])
		self.n_pairs = int(self.nreferent * n_items_less_one / 2)
		self.range_similarities = range(self.n_pairs)
		sumsort_s = sums_s_star
		diffsort_d = diffs_d_star

		for each_row in self.range_similarities:
			a_col = list(sums_s_star[str(sums_s_star.columns[each_row])])
			a_col.sort()
			sumsort_s[str(sums_s_star.columns[each_row])] = a_col
			a_col = list(diffs_d_star[str(diffs_d_star.columns[each_row])])
			a_col.sort(reverse=True)
			diffsort_d[str(diffs_d_star.columns[each_row])] = a_col
		combo_b = sumsort_s + diffsort_d
		for each_pair in self.range_similarities:
			cum_b_hat[str(diffs_d_star.columns[each_pair])] = combo_b[str(diffs_d_star.columns[each_pair])].cumsum(axis=0)
		ordered = cum_b_hat.rank(axis=1, method="average")
		if self.n_individ < 750:
			itcon = 4
		else:
			itcon = int(self.n_individ / 150)
		maxadeq = 0
		range_rows = range(1, self.n_individ)
		for each_row in range_rows:
			rho = spearmanr(ordered.iloc[each_row], ordered.iloc[each_row - 1])[0]
			unique_vals = ordered.iloc[each_row].nunique()

			discrim = (unique_vals - 1) / self.n_pairs
			dense = (self.n_individ - each_row) / self.n_individ
			adeq = rho * discrim * dense

			if adeq > maxadeq:
				maxadeq = adeq
				best_ranking = ordered.iloc[each_row]
				loc_best = each_row
			elif maxadeq >= dense:
				print("\nMaxadeq value greater than dense value")
				# self.have_evaluations = "No"
			# return conf, dyad, flags, people			break
			elif (each_row - loc_best) == 4:

				print("\nMaxadeq has stabilized")
				# self.have_evaluations = "No"
				# return conf, dyad, flags, people
				break
		#
		# Create similarities as a list and a dictionary keyed by a dyad

		self.similarities_as_list = list(best_ranking)
		#
		# Create rows and columns from similarities_as_list
		#
		row = []
		for each_row in range(self.nreferent - 1):
			row = []
			row.append(self.similarities_as_list[each_row])
			last_index = each_row
			cols = range(each_row)
			for each_col in cols:
				indexer = last_index + self.nreferent - each_col - 2
				last_index = indexer
				row.append(self.similarities_as_list[indexer])
			#
			self.similarities.append(row)

		self.duplicate_similarities()
	# --------------------------------------------------------------------------------------------

	def max_and_min(self, command):

		conf_max = 0.0
		ind_max = 0.0
		if self.have_active_configuration():
			conf_max = self.point_coords.max(numeric_only=True).max()
			conf_min = self.point_coords.min(numeric_only=True).min()
			if abs(conf_min) > conf_max:
				conf_max = abs(conf_min)
		#
		# if self.have_individual_data():
		if command in ("Individuals", "Joint"):
		#    and self.show_respondent_points:
			ind_max = self.ind_vars.iloc[[1, 2]].max().max()  # hardwired to second and third columns
			ind_min = self.ind_vars.iloc[[1, 2]].min().min()  # hardwired to second and third columns
			if abs(ind_min) > ind_max:
				ind_max = abs(ind_min)

		#
		vals = [conf_max, ind_max]
		the_max = max(vals)
		# print(f"DEBUG -- {the_max = }")
		#
		if the_max < .5:
			plot_max = .5
		else:
			plot_max = math.ceil(the_max)
		#
		self.hor_max = plot_max
		self.hor_min = - plot_max
		self.vert_max = plot_max
		self.vert_min = - plot_max
		self.move_label = .03 * plot_max
		# print(f"DEBUG -- at end of max_and_min - {self.hor_max = }")
		#
		return

	# -------------------------------------------------------------------------------------------

	def mds(self):

		self.dim_names = []
		self.dim_labels = []

		nmds = manifold.MDS(
			n_components=self.n_comp, metric=self.use_metric,
			dissimilarity='precomputed', n_init=10, verbose=1, normalized_stress="auto")
		#
		npos = nmds.fit_transform(X=self.similarities_as_square)
		#
		self.point_coords = pd.DataFrame(npos.tolist())
		self.point_coords.set_index([self.item_labels], inplace=True)
		#
		self.range_dims = range(self.ndim)
		for each_dim in self.range_dims:
			dim_num = str(each_dim + 1)
			self.dim_names.append("Dimension " + dim_num)
			self.dim_labels.append("Dim" + dim_num)

		self.point_coords.columns = self.dim_names
		#
		self.best_stress = nmds.stress_
		print("\n\tBest stress: ", self.best_stress)

		# self.point_labels = self.item_labels
		# self.point_names = self.item_names

		# print("\nEmbedding: \n", nmds.embedding_)

		# print(f"\nDEBUG -- \n{self.mds_scores= }")

# --------------------------------------------------------------------------------------------

	def move(self, which_dim, value):
		for each_point in self.range_points:
			self.point_coords.iloc[each_point][which_dim] = \
				((self.point_coords.iloc[each_point][which_dim]) + value)

	# ---------------------------------------------------------------------------

	def needs_active(self, command):

		if not self.have_active_configuration():

			QMessageBox.critical(
				None,
				f"No Active configuration has been established.",
				f"Open a configuration file or use Models such as MDS, Factor analysis or " +
				f"\nPrincipal components to establish one before using {command}."
			)
			return True
		else:
			return False

	# --------------------------------------------------------------------------

	def needs_evaluations(self, command):
		if not self.have_evaluations():
			QMessageBox.critical(
				None,
				f"No Evaluations have been established.",
				f"Open Evaluations file before using {command}."
			)
			return True
		else:
			return False
	# ---------------------------------------------------------------------------

	def needs_similarities(self, command):
		if not self.have_similarities():
			QMessageBox.critical(
				None,
				f"No similarities have been established.",
				f"Open Similarities before using {command}."
			)
			return True
		else:
			return False

	# ---------------------------------------------------------------------------

	def needs_reference_points(self, command):
		if not self.have_reference_points():
			QMessageBox.critical(
				None,
				f"No reference points have been established.",
				f"Establish Reference points before {command}."
			)
			return True
		else:
			return False

	# ---------------------------------------------------------------------------

	def needs_correlations(self, command):
		if not self.have_correlations():
			QMessageBox.critical(
				None,
				f"No Correlations have been established.",
				f"Open Correlations file before using {command}."
			)
			return True
		else:
			return False
	# --------------------------------------------------------------------------

	def needs_grouped_data(self, command):
		if not self.have_grouped_data():
			QMessageBox.critical(
				None,
				f"No Grouped data has established.",
				f"Open Grouped Data file before using {command}."
			)
			return True
		else:
			return False

	# --------------------------------------------------------------------------

	def needs_individual_data(self, command):
		if not self.have_individual_data():
			QMessageBox.critical(
				None,
				f"No Individual data has established.",
				f"Open Individual file before using {command}."
			)
			return True
		else:
			return False

	# --------------------------------------------------------------------------

	def needs_distances(self, command):
		if not self.have_distances():
			QMessageBox.critical(
				None,
				f"No distances have been established.",
				f"Use Associations Distances before using {command}."
			)
			return True
		else:
			return False

	# --------------------------------------------------------------------------

	def needs_ranks(self, command):
		if not self.have_ranks():
			QMessageBox.critical(
				None,
				f"No ranks have been established.",
				f"Use Associations Ranks before using {command}."
			)
			return True
		else:
			return False

	# --------------------------------------------------------------------------

	def needs_target(self, command):

		# print(f"DEBUG -- At top of needs_target *************************")
		# print(f"{self.point_coords = }")

		if not self.have_target_configuration():

			QMessageBox.critical(
				None,
				f"No Target configuration has been established.",
				f"Open a target file to establish one before using {command}."
			)
			return True
		else:
			return False

	# ----------------------------------------------------------------------------------------------

	def plot_alike(self):
		""" plot alike  -creates a plot with a line joining points with high similarity.
		A plot of the configuration will be created with a line joining pairs of points with
		a similarity above (or if dissimilarities, below) the cutoff.
		"""
		#
		# Initialize variables needed
		#
		x_coords = []
		y_coords = []
		#
		# Determine boundaries for plot and how much to move labels????????????
		#
		# parts = boundaries_function(conf, parts)
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Set aspect ratio
		#
		ax.set_aspect("equal")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# Add point coordinates to coordinate vectors and labels to plot
		#
		for each_point in self.range_points:
			x_coords.append(self.point_coords.iloc[each_point][self.hor_dim])
			y_coords.append(self.point_coords.iloc[each_point][self.vert_dim])
			ax.text(
				self.point_coords.iloc[each_point][self.hor_dim] + self.move_label,
				self.point_coords.iloc[each_point][self.vert_dim], self.point_labels[each_point])
		#
		# Add points to plot by passing in coordinate vectors
		#
		ax.scatter(x_coords, y_coords)
		#
		# Show lines joining each most alike pair based on similarities
		# 		Need to use range because for alike vectors have duplicates
		# 		and .index returns only first instance
		#
		nalike = range(len(self.a_x_alike))
		for each_alike in nalike:
			ax.plot(
				(self.a_x_alike[each_alike], self.b_x_alike[each_alike]),
				(self.a_y_alike[each_alike], self.b_y_alike[each_alike]),
				color="k")
		#
		# Ready to complete plot
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		return fig

	# ---------------------------------------------------------------------------------

	def plot_base(self, reply):
		#
		# Initialize variables needed
		#
		x_coords = []
		y_coords = []
		x_base = []
		y_base = []
		#
		match reply[0:4]:
			case "left":
				x_base = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ if self.seg.loc[ind, "Base"] == 1]
				y_base = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ if self.seg.loc[ind, "Base"] == 1]

			case "righ":
				x_base = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ if self.seg.loc[ind, "Base"] == 3]
				y_base = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ if self.seg.loc[ind, "Base"] == 3]
			case "both":
				x_base = [
					self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Base"] in (1, 3)
					]
				y_base = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Base"] in (1, 3)
					]
			case "neit":
				x_base = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Base"] == 2]
				y_base = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Base"] == 2]
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()

		# Setting aspect ratio
		#
		ax.set_aspect("equal")
		#
		ax.set_title("Base")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# Add reference points labels and coordinate vectors to plot
		#
		ax.text(
			self.point_coords.iloc[self.rival_a][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_a][self.vert_dim],
			self.point_labels[self.rival_a])
		ax.text(
			self.point_coords.iloc[self.rival_b][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_b][self.vert_dim],
			self.point_labels[self.rival_b])
		x_coords.append(self.point_coords.iloc[self.rival_a][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_a][self.vert_dim])
		x_coords.append(self.point_coords.iloc[self.rival_b][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_b][self.vert_dim])
		#
		ax.scatter(x_coords, y_coords)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		# Create base regions around the reference points
		#
		# WEST
		#
		# print(f"DEBUG -- {self.west.case = }")
		# print(f"DEBUG -- {self.east.case = }")
		match self.west.case:
		#
		# west Case 0a west slope is zero from Left side to Right side --- 0a - west & right
		#
			case "0a":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min]
				])
		#
		# west  Case 0b slope is zero - from top to bottom       ----- 0b - west & left
		#
			case "0b":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_min, self.vert_min],
					[self.hor_min, self.vert_max]
				])
		#
		# west Case Ia Positive slope from Left side to Right side and min_y > self.vert_min ----- Ia - west & left
		#
			case "Ia":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_min, self.vert_max]
				])
		#
		# west Case IIa Positive slope from Left side to Top and max_y == self.vert_max ----- IIa - west & left
		#
			case "IIa":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_min, self.vert_max]
				])
		#
		# west Case IIIa Positive slope from Bottom to Right side     ---- IIIa - west & left
		#
			case "IIIa":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_min, self.vert_max],
					[self.hor_min, self.vert_min]
				])
		#
		# west Case IVa Positive slope from Bottom to Top    ----- IVa - west & left
		#
			case "IVa":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_min, self.vert_max],
					[self.hor_min, self.vert_min]
				])
		#
		# west Case Ib Negative slope from Left side to Right side      ---- Ib - west & right
		#
			case "Ib":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min]
				])
		#
		# west Case IIb Negative slope from Left side to Bottom       ----- IIb - west & right
		#
			case "IIb":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_min, self.vert_min]
				])
		#
		# west Case IIIb Negative slope from Top to Right side      ----- IIIb - west & right
		#
			case "IIIb":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min],
					[self.hor_min, self.vert_max]
				])
		#
		# west Case IVb Negative slope from Top to Bottom      ------ IVb - west & right
		#
			case "IVb":
				self.base_left.vertices = np.array([
					[self.west.start_x, self.west.start_y],
					[self.west.end_x, self.west.end_y],
					[self.hor_min, self.vert_min],
					[self.hor_min, self.vert_max]
				])
		#
		# EAST
		#
		match self.east.case:
		#
		# east  Case 0a  slope is zero - from top to bottom    ---- 0a - east & left
		#
			case "0a":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_min, self.vert_max]
				])
		#
		# east Case 0b east slope is zero from Left side to Right side     ---- 0b - east & right
		#
			case "0b":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_max, self.vert_max]
				])
		#
		# east Case Ia Positive slope from Left side to Right side and min_y > self.vert_min     --- Ia - east & right
		#
			case "Ia":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min]
				])
		#
		# east Case IIa Positive slope from Left side to Top and max_y == self.vert_max    ---- IIa - east & right
		#
			case "IIa":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min]
				])
		#
		#
		# east Case IIIa Positive slope from Bottom to Right side    ---- IIIa - east & right
		#
			case "IIIa":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_max, self.vert_min]
				])
		#
		# east Case IVa Positive slope from Bottom to Top      ----  IVa -  east & right
		#
			case "IVa":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_max, self.vert_min]
				])
		#
		# east Case Ib Negative slope from Left side to Right side     ----   east & left
		#
			case "Ib":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_min, self.vert_max]
				])
		#
		# east Case IIb Negative slope from Left side to Bottom      ---- IIb - east & left
		#
			case "IIb":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_min, self.vert_min]
				])
		#
		# east Case IIIb Negative slope from Top to Right side      ---- IIIb - east & left
		#
			case "IIIb":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_max, self.vert_max]
				])
		#
		# east Case IVb Negative slope from Bottom to Top       -----   east & left
		#
			case "IVb":
				self.base_right.vertices = np.array([
					[self.east.start_x, self.east.start_y],
					[self.east.end_x, self.east.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_max, self.vert_max]
				])
		#
		# Shade base regions
		#
		# # Taking transpose
		# x, y = data.T
		self.base_left.x, self.base_left.y = self.base_left.vertices.T
		self.base_left.color = "green"  # eventually replace with blue
		self.base_right.x, self.base_right.y = self.base_right.vertices.T
		self.base_right.color = "pink"  # eventually replace with red

		ax.fill(self.base_right.x, self.base_right.y, self.base_right.color)
		ax.fill(self.base_left.x, self.base_left.y, self.base_left.color)

		#
		# west line goes through a point on line connecting points
		# Solve for that point's coordinates and add it to plot
		#
		ax.text(self.west_connector_cross_x, self.west_connector_cross_y, "W")
		ax.scatter(self.west_connector_cross_x, self.west_connector_cross_y)
		#
		# Label the ends of west
		#
		ax.text(self.west.start_x, self.west.start_y, "W_S")
		ax.text(self.west.end_x, self.west.end_y, "W_E")
		#
		# Add west to plot
		#
		ax.plot(
			[self.west.start_x, self.west.end_x],
			[self.west.start_y, self.west.end_y])
		#
		# east line goes through a point on line connecting points
		# Solve for that point's coordinates and add it to plot
		#
		ax.text(self.east_connector_cross_x, self.east_connector_cross_y, "E")
		ax.scatter(self.east_connector_cross_x, self.east_connector_cross_y)
		#
		# Label the ends of west
		#
		ax.text(self.east.start_x, self.east.start_y, "E_S")
		ax.text(self.east.end_x, self.east.end_y, "E_E")
		#
		# Add east to plot
		#
		ax.plot(
			[self.east.start_x, self.east.end_x],
			[self.east.start_y, self.east.end_y])

		#
		ax.scatter(x_base, y_base, color="black", s=self.point_size)
		#
		# Ready to complete plot
		#
		return fig

	# -------------------------------------------------------------------------------------------

	def plot_compare(self, target):

		# print("DEBUG -- At top of plot_compare")
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Setting aspect ratio causes warning message
		#
		ax.set_aspect("equal")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# print(f"DEBUG -- {self.point_coords = }")
		# print(f"DEBUG -- {target = }")
		#
		for each_point in self.range_points:
			mid_x = (
				(self.point_coords.iloc[each_point][0]
				+ target.iloc[each_point][0])
				/ 2
			)
			mid_y = (
				(self.point_coords.iloc[each_point][1]
				+ target.iloc[each_point][1])
				/ 2
			)
			ax.text(
				mid_x + self.move_label,
				mid_y,
				self.point_labels[each_point])

			ax.plot(
				[self.point_coords.iloc[each_point][0], target.iloc[each_point][0]],
				[self.point_coords.iloc[each_point][1], target.iloc[each_point][1]],
				color="black"
			)
			ax.text(
				self.point_coords.iloc[each_point][0],
				self.point_coords.iloc[each_point][1],
				"A"
			)
			ax.text(
				target.iloc[each_point][0],
				target.iloc[each_point][1],
				"T"
			)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		# Ready to complete plot
		#
		return fig

	# -------------------------------------------------------------------------------------------

	def plot_configuration(self):

		# print("DEBUG -- at top of plot_conf")
		x_coords = []
		y_coords = []
		#
		fig, ax = plt.subplots()
		# ax.scatter(
		#	self.point_coords.iloc[:,0],
		#	self.point_coords.iloc[:,1],

		# )
		# ax.set_xlabel(self.dim_names[0])
		# ax.set_ylabel(self.dim_names[1])

		# plt.text(self.point_coords.iloc[:,0],
		#	self.point_coords.iloc[:,1],
		#		self.point_coords.index)

		# x_coords = self.point_coords.iloc[:,0]
		# y_coords = self.point_coords.iloc[:,1]
		# Begin the building of the plot
		#
		# Setting aspect ratio causes warning message
		#
		ax.set_aspect("equal")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		x_coords.append(self.point_coords.iloc[:, self.hor_dim])
		y_coords.append(self.point_coords.iloc[:, self.vert_dim])

		for each_point in self.range_points:
			ax.text(
				self.point_coords.iloc[each_point][self.hor_dim] + self.move_label,
				self.point_coords.iloc[each_point][self.vert_dim],
				self.point_labels[each_point])
		ax.scatter(x_coords, y_coords, color="black", s=5)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		# Show connector if requested
		#
		if self.show_connector:
			#
			# Draw a line connecting the reference points
			#
			ax.plot(
				[self.point_coords.iloc[self.rival_a][self.hor_dim],
				self.point_coords.iloc[self.rival_b][self.hor_dim]],
				[self.point_coords.iloc[self.rival_a][self.vert_dim],
				self.point_coords.iloc[self.rival_b][self.vert_dim]])
		# print(f"DEBUG -- {self.have_bisector_info() = }")
		if self.have_bisector_info():
			if self.show_bisector:
				#
				# Calculate ends of bisector
				#
				self.ends_of_bisector_function()
				#
				# Add bisector to plot
				# For debugging label Start and End
				#
				ax.text(self.bisector.start_x, self.bisector.start_y, "S")
				ax.text(self.bisector.end_x, self.bisector.end_y, "E")
				#
				# For Debugging label midpoint as M
				#
				ax.text(self.connector_bisector_cross_x, self.connector_bisector_cross_y, "M")
				#
				ax.text(self.bisector.start_x, self.bisector.start_y, "S")
				ax.text(self.bisector.end_x, self.bisector.end_y, "E")
				#
				# Draw bisector
				#
				ax.plot(
					[self.bisector.start_x, self.bisector.end_x],
					[self.bisector.start_y, self.bisector.end_y])
		# Ready to complete plot
		#
		return fig

	# ------------------------------------------------------------------------------------------

	def plot_contest(self):

		#
		# Initialize variables needed
		#
		x_coords = []
		y_coords = []
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Setting aspect ratio
		#
		ax.set_aspect("equal")
		#
		ax.set_title("Contest")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# Add reference points labels and coordinate vectors to plot
		#
		ax.text(
			self.point_coords.iloc[self.rival_a][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_a][self.vert_dim],
			self.point_labels[self.rival_a])
		ax.text(
			self.point_coords.iloc[self.rival_b][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_b][self.vert_dim],
			self.point_labels[self.rival_b])
		x_coords.append(self.point_coords.iloc[self.rival_a][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_a][self.vert_dim])
		x_coords.append(self.point_coords.iloc[self.rival_b][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_b][self.vert_dim])
		#
		ax.scatter(x_coords, y_coords)
		#
		ax.axis([
			self.hor_min, self.hor_max,
			self.vert_min, self.vert_max])
		#
		# Create circle around the reference points
		#
		core_a = plt.Circle((
			self.point_coords.iloc[self.rival_a][self.hor_dim],
			self.point_coords.iloc[self.rival_a][self.vert_dim]),
			radius=self.connector.length * .2, fill=False, hatch="X")
		core_b = plt.Circle((
			self.point_coords.iloc[self.rival_b][self.hor_dim],
			self.point_coords.iloc[self.rival_b][self.vert_dim]),
			radius=self.connector.length * .2, fill=False, hatch="O")
		#
		plt.gca().add_artist(core_a)
		plt.gca().add_artist(core_b)
		#
		self.ends_of_bisector_function()
		#
		ax.text(self.bisector.start_x, self.bisector.start_y, "S")
		ax.text(self.bisector.end_x, self.bisector.end_y, "E")
		#
		# For Debugging label midpoint as M
		#
		ax.text(self.connector_bisector_cross_x, self.connector_bisector_cross_y, "M")
		#
		ax.plot(
			[self.bisector.start_x, self.bisector.end_x],
			[self.bisector.start_y, self.bisector.end_y])
		#
		# Add west and east
		#
		ax.plot(
			[self.west.start_x, self.west.end_x],
			[self.west.start_y, self.west.end_y])
		ax.plot(
			[self.east.start_x, self.east.end_x],
			[self.east.start_y, self.east.end_y])
		#
		# Label the ends of west
		#
		ax.text(self.west.start_x, self.west.start_y, "W_S")
		ax.text(self.west.end_x, self.west.end_y, "W_E")
		#
		# Label the ends of east
		#
		ax.text(self.east.start_x, self.east.start_y, "E_S")
		ax.text(self.east.end_x, self.east.end_y, "E_E")
		#
		# Add Culture War divider
		#
		ax.plot([self.hor_max, self.hor_min], [self.dim2_div, self.dim2_div])
		#
		# Add Partisan divider
		#
		ax.plot([self.dim1_div, self.dim1_div], [self.vert_max, self.vert_min])
		#
		# Ready to complete plot
		#
		return fig

	# ---------------------------------------------------------------------------------------------------------

	def plot_convertible(self, reply):

		#
		# Initialize variables needed
		#
		x_coords = []
		y_coords = []
		x_convertible = []
		y_convertible = []
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Setting aspect ratio
		#
		ax.set_aspect("equal")
		#
		ax.set_title("Convertible")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# Add reference points labels and coordinate vectors to plot
		#
		ax.text(
			self.point_coords.iloc[self.rival_a][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_a][self.vert_dim],
			self.point_labels[self.rival_a])
		ax.text(
			self.point_coords.iloc[self.rival_b][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_b][self.vert_dim],
			self.point_labels[self.rival_b])
		x_coords.append(self.point_coords.iloc[self.rival_a][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_a][self.vert_dim])
		x_coords.append(self.point_coords.iloc[self.rival_b][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_b][self.vert_dim])
		#
		ax.scatter(x_coords, y_coords)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		self.ends_of_bisector_function()
		#
		# Create convertible regions
		#
		# print(f"DEBUG -- {self.bisector.case = }")
		# print(f"DEBUG -- {self.west.case = }")
		# print(f"DEBUG -- {self.east.case = }")
		right_includes_upper_right = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.hor_max, self.vert_max],
			[self.west.end_x, self.west.end_y],
			[self.west.start_x, self.west.start_y]
		])
		right_includes_lower_right = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.hor_max, self.vert_min],
			[self.west.end_x, self.west.end_y],
			[self.west.start_x, self.west.start_y]
		])
		right_includes_upper_left = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.west.end_x, self.west.end_y],
			[self.west.start_x, self.west.start_y],
			[self.hor_min, self.vert_max]
		])
		right_includes_lower_left = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.west.end_x, self.west.end_y],
			[self.west.start_x, self.west.start_y],
			[self.hor_min, self.vert_min]
		])
		left_includes_lower_left = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.hor_min, self.vert_min],
			[self.east.end_x, self.east.end_y],
			[self.east.start_x, self.east.start_y]
		])
		left_includes_upper_left = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.east.end_x, self.east.end_y],
			[self.east.start_x, self.east.start_y],
			[self.hor_min, self.vert_max]
		])
		left_includes_lower_right = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.hor_max, self.vert_min],
			[self.east.end_x, self.east.end_y],
			[self.east.start_x, self.east.start_y]
		])
		left_includes_upper_right = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.hor_max, self.vert_max],
			[self.east.end_x, self.east.end_y],
			[self.east.start_x, self.east.start_y]
		])
		#
		# Establish default areas
		#
		self.convertible_to_left.vertices = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.east.end_x, self.east.end_y],
			[self.east.start_x, self.east.start_y]
		])
		self.convertible_to_right.vertices = np.array([
			[self.bisector.start_x, self.bisector.start_y],
			[self.bisector.end_x, self.bisector.end_y],
			[self.west.end_x, self.west.end_y],
			[self.west.start_x, self.west.start_y]
		])
		#
		# Bisector Case 0a Bisector slope is zero from Left side to Right side
		# Bisector Case 0b Connector slope is zero - from top to bottom
		# Bisector Case Ia Positive slope from Left side to Right side
		# bisector test is redundant, case includes slope, but improves efficiency
		#
		if self.bisector.direction == "Upward slope":
			if self.bisector.case == "Ia":
				#
				# if self.west.case == "Ia":									# test with FP
				# defaults should cover this
				if self.west.case == "IIa":								# test with TG
					self.convertible_to_right.vertices = right_includes_upper_right
				#
				# if self.east.case == "Ia":
				# defaults should cover this							test with BG
				if self.east.case == "IIIa":							# test with TC/FP
					self.convertible_to_left.vertices = left_includes_lower_left
			#
			# Bisector Case IIa Positive slope from Left side to Top and max_y == vert_max
			#
			elif self.bisector.case == "IIa":
				# if self.west.case == "IIa":
				# defaults should cover this							test with TG
				if self.east.case == "Ia":    							# test with  BK
					self.convertible_to_left.vertices = left_includes_upper_right
				# elif self.east.case == "IIa":                     test with BF
				# defaults should cover this
				elif self.east.case == "IIIa":						# test with    TS
					self.convertible_to_left.vertices = np.array([
						[self.bisector.start_x, self.bisector.start_y],
						[self.bisector.end_x, self.bisector.end_y],
						[self.hor_max, self.vert_max],
						[self.east.end_x, self.east.end_y],
						[self.east.start_x, self.east.start_y],
						[self.hor_min, self.vert_min],
					])
				elif self.east.case == "IVa":						# test with BL
					self.convertible_to_left.vertices = left_includes_lower_left
			#
			# Bisector Case IIIa Positive slope from Bottom to Right side
			#
			elif self.bisector.case == "IIIa":
				if self.west.case == "Ia":							# test with GP
					self.convertible_to_right.vertices = right_includes_lower_left
				elif self.west.case == "IIa":						# test with BC
					self.convertible_to_right.vertices = np.array([
						[self.bisector.start_x, self.bisector.start_y],
						[self.bisector.end_x, self.bisector.end_y],
						[self.hor_max, self.vert_max],
						[self.west.end_x, self.west.end_y],
						[self.west.start_x, self.west.start_y],
						[self.hor_min, self.vert_min]
					])
				elif self.west.case == "IVa":						# test with CF
					self.convertible_to_right.vertices = np.array([
						[self.bisector.start_x, self.bisector.start_y],
						[self.bisector.end_x, self.bisector.end_y],
						[self.hor_max, self.vert_max],
						[self.west.end_x, self.west.end_y],
						[self.west.start_x, self.west.start_y]
					])
					# self.convertible_to_left.vertices = np.array([
					# 	[self.bisector.start_x, self.bisector.start_y],
					# 	[self.bisector.end_x, self.bisector.end_y],
					# 	[self.east.end_x, self.east.end_y],
					# 	[self.east.start_x, self.east.start_y],
					# ])
				# elif self.west.case == "IIIa":					test with EH
				# 	defaults should cover this as well as all cases match
				# if self.east.case == "IIIa":						test with EH
				# defaults should cover this as well as all cases match
			#
			# Bisector Case IVa Positive slope from Bottom to Top
			#
			elif self.bisector.case == "IVa":
				if self.west.case == "IIa":							# test with TN
					self.convertible_to_right.vertices = right_includes_lower_left
				#
				# elif self.west.case == "IVa":						test with ????????????????
					# defaults should cover this
				#
				if self.east.case == "IIIa":						# test with EJ
					self.convertible_to_left.vertices = left_includes_upper_right

				# elif self.east.case == "IVa":						test with BD
					# defaults should cover this as well as all cases match
		#
		# Bisector Case Ib Negative slope from Left side to Right side
		#
		elif self.bisector.direction == "Downward slope":
			if self.bisector.case == "Ib":
				# if self.west.case == "Ib":						test with AC
				# defaults should cover this
				#
				if self.west.case == "IIb":							# test with EP
					self.convertible_to_right.vertices = right_includes_lower_right
				#
				# if self.east.case == "Ib":						test with CE
					# defaults should cover this as well as all cases match
				#
				if self.east.case == "IIIb":						# test with AC
					self.convertible_to_left.vertices = left_includes_upper_left
			#
			# Bisector Case IIb Negative slope from Left side to Bottom and min_y == vert_min
			#
			elif self.bisector.case == "IIb":
				if self.west.case == "IIb":							# test with HI
					# defaults should cover this
					if self.east.case == "Ib":						# test with IY
						self.convertible_to_left.vertices = left_includes_lower_right
					# elif self.east.case == "IIb":					test with IK
					# defaults should cover this as well as all cases match
					elif self.east.case == "IIIb":  				# test with PX
						self.convertible_to_left.vertices = np.array([
							[self.bisector.start_x, self.bisector.start_y],
							[self.bisector.end_x, self.bisector.end_y],
							[self.hor_max, self.vert_min],
							[self.east.end_x, self.east.end_y],
							[self.east.start_x, self.east.start_y],
							[self.hor_min, self.vert_max]
						])
					elif self.east.case == "IVb":						# test with IW
						self.convertible_to_left.vertices = left_includes_upper_left
			#
			# Bisector Case IIIb Negative slope from Top to right side
			#
			#
			elif self.bisector.case == "IIIb":
				if self.east.case == "IIIb":
					noop = 13
				#
				if self.west.case == "Ib":								#test with AS
					self.convertible_to_right.vertices = right_includes_upper_left
				#
				elif self.west.case == "IIb":							# test with ?
					self.convertible_to_right.vertices = np.array([
						[self.bisector.start_x, self.bisector.start_y],
						[self.bisector.end_x, self.bisector.end_y],
						[self.hor_max, self.vert_min],
						[self.west.end_x, self.west.end_y],
						[self.west.start_x, self.west.start_y],
						[self.hor_min, self.vert_max]
					])
				#
				# elif self.west.case == "IIIb":					test with AQ
				# defaults should cover this as well as all cases match
				elif self.west.case == "IVb":						# test with AK
					self.convertible_to_right.vertices = right_includes_lower_right
			#
			# Bisector Case IVb Negative slope from Bottom to Top and max_y > vert_max  ????????????????????????
			#
			elif self.bisector.case == "IVb":
				if self.west.case == "IIb":								# test with GH
					self.convertible_to_right.vertices = right_includes_upper_left
					if self.east.case == "IIIb":							#test with GM
						self.convertible_to_left.vertices = left_includes_lower_right
					# if self.east.case == "IVb": 						#test with GY
					# defaults should cover this
				elif self.west.case == "IVb":							#Test with 	DG
					if self.east.case == "IIIb":
						self.convertible_to_left.vertices = left_includes_lower_right
					# elif self.east.case == "IVb":
					# defaults should cover this as well as all cases match
		#
		# Shade convertible regions
		#

		# # Taking transpose
		# x, y = data.T
		self.convertible_to_left.x, self.convertible_to_left.y = self.convertible_to_left.vertices.T
		self.convertible_to_left.color = "green"  # eventually replace with blue
		self.convertible_to_right.x, self.convertible_to_right.y = self.convertible_to_right.vertices.T
		self.convertible_to_right.color = "pink"  # eventually replace with red
		#
		ax.fill(self.convertible_to_right.x, self.convertible_to_right.y, self.convertible_to_right.color)
		ax.fill(self.convertible_to_left.x, self.convertible_to_left.y, self.convertible_to_left.color)
		#
		ax.text(self.west.start_x, self.west.start_y, 'W_S')
		ax.text(self.west.end_x, self.west.end_y, 'W_E')
		ax.text(self.east.start_x, self.east.start_y, 'E_S')
		ax.text(self.east.end_x, self.east.end_y, 'E_E')
		ax.text(self.bisector.start_x, self.bisector.start_y, 'BS')
		ax.text(self.bisector.end_x, self.bisector.end_y, 'BE')
		#
		if reply[0:4] in ("left", "righ", "both", "sett"):
			match reply[0:4]:
				case "left":
					x_convertible = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Convertible"] == 1]
					y_convertible = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Convertible"] == 1]
				case "righ":
					x_convertible = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Convertible"] == 2]
					y_convertible = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Convertible"] == 2]
				case "both":
					x_convertible = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Convertible"] in (1, 2)]
					y_convertible = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Convertible"] in (1, 2)]
				case "sett":
					x_convertible = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Convertible"] == 3]
					y_convertible = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Convertible"] == 3]
			#
			ax.scatter(x_convertible, y_convertible, color="black", s=self.point_size)
		#
		# Ready to complete plot
		#
		return fig

	# ----------------------------------------------------------------------------------------

	def plot_core(self, reply):

		#
		# Initialize variables needed
		#
		x_coords = []
		y_coords = []
		x_core = []
		y_core = []
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Setting aspect ratio
		#
		ax.set_aspect("equal")
		#
		ax.set_title("Core Supporters")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# Add reference points labels and coordinate vectors to plot
		#
		ax.text(
			self.point_coords.iloc[self.rival_a][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_a][self.vert_dim],
			self.point_labels[self.rival_a])
		ax.text(
			self.point_coords.iloc[self.rival_b][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_b][self.vert_dim],
			self.point_labels[self.rival_b])
		x_coords.append(self.point_coords.iloc[self.rival_a][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_a][self.vert_dim])
		x_coords.append(self.point_coords.iloc[self.rival_b][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_b][self.vert_dim])
		#
		ax.scatter(x_coords, y_coords)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		# Determine which reference point is closest to edge 1
		#
		# Create circle around the reference points
		#
		if self.bisector.direction == "Flat":
			if self.point_coords.iloc[self.rival_a][self.vert_dim] > self.point_coords.iloc[self.rival_b][self.vert_dim]:
				core_a = plt.Circle((
					self.point_coords.iloc[self.rival_a][self.hor_dim],
					self.point_coords.iloc[self.rival_a][self.vert_dim]),
					radius=self.core_radius, color='blue')
				core_b = plt.Circle((
					self.point_coords.iloc[self.rival_b][self.hor_dim],
					self.point_coords.iloc[self.rival_b][self.vert_dim]),
					radius=self.core_radius, color='red')
			else:
				core_a = plt.Circle((
					self.point_coords.iloc[self.rival_a][self.hor_dim],
					self.point_coords.iloc[self.rival_a][self.vert_dim]),
					radius=self.core_radius, color='red')
				core_b = plt.Circle((
					self.point_coords.iloc[self.rival_b][self.hor_dim],
					self.point_coords.iloc[self.rival_b][self.vert_dim]),
					radius=self.core_radius, color='blue')
		elif self.point_coords.iloc[self.rival_a][self.hor_dim] > self.point_coords.iloc[self.rival_b][self.hor_dim]:
			core_a = plt.Circle((
				self.point_coords.iloc[self.rival_a][self.hor_dim],
				self.point_coords.iloc[self.rival_a][self.vert_dim]),
				radius=self.core_radius, color='red')
			core_b = plt.Circle((
				self.point_coords.iloc[self.rival_b][self.hor_dim],
				self.point_coords.iloc[self.rival_b][self.vert_dim]),
				radius=self.core_radius, color='blue')
		else:
			core_a = plt.Circle((
				self.point_coords.iloc[self.rival_a][self.hor_dim],
				self.point_coords.iloc[self.rival_a][self.vert_dim]),
				radius=self.core_radius, color='blue')
			core_b = plt.Circle((
				self.point_coords.iloc[self.rival_b][self.hor_dim],
				self.point_coords.iloc[self.rival_b][self.vert_dim]),
				radius=self.core_radius, color='red')
		#
		plt.gca().add_artist(core_a)
		plt.gca().add_artist(core_b)
		#
		if reply != "":
			if reply[0:4] == "left":
				x_core = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Core"] == 1]
				y_core = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Core"] == 1]
			if reply[0:4] == "righ":
				x_core = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Core"] == 3]
				y_core = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Core"] == 3]
			if reply[0:4] == "both":
				x_core = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Core"] == 1
						or self.seg.loc[ind, "Core"] == 3]
				y_core = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Core"] == 1
						or self.seg.loc[ind, "Core"] == 3]
			if reply[0:4] == "neit":
				x_core = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Core"] == 2]
				y_core = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
					if self.seg.loc[ind, "Core"] == 2]
			#
			ax.scatter(x_core, y_core, color="black", s=self.point_size)
		#
		# Ready to complete plot
		#
		return fig

	# ------------------------------------------------------------------------------------------

	def plot_cutoff(self, num_bins):

		fig, ax = plt.subplots()
		ax.hist(self.similarities_as_list, num_bins)
		ax.set_xlabel("Similarity")
		ax.set_ylabel("Frequency")
		#
		return fig

	# -------------------------------------------------------------------------------------------

	def plot_directions(self):

		# print("DEBUG -- at top of plot_directions")
		#
		fig, ax = plt.subplots()
		# Begin the building of the plot
		#
		# Setting aspect ratio causes warning message
		#
		ax.set_aspect("equal")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		for each_point in self.range_points:
			length = np.sqrt(self.point_coords.iloc[each_point][self.hor_dim]**2
							+ self.point_coords.iloc[each_point][self.vert_dim]**2
							)
			if length > 0.0:
				x_dir = self.point_coords.iloc[each_point][self.hor_dim] / length
				y_dir = self.point_coords.iloc[each_point][self.vert_dim] / length
			else:
				x_dir = self.point_coords.iloc[each_point][self.hor_dim]
				y_dir = self.point_coords.iloc[each_point][self.vert_dim]

			ax.arrow(
					0.0, 0.0, x_dir, y_dir, color="black",
					head_width=self.vector_head_width,
					width=self.vector_width
			)
			if x_dir >= 0.0:
				ax.text(
					x_dir + self.move_label,
					y_dir,
					self.point_labels[each_point])
			else:
				ax.text(
					x_dir - (2 * (len(self.point_labels[each_point]) - 1) * self.move_label),
					y_dir,
					self.point_labels[each_point])
			print(f"DEBUG -- {length = }")
		#
		unit_circle = plt.Circle((0.0, 0.0), 1.0, fill=False)
		ax.add_artist(unit_circle)
		#
		# ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		ax.axis([-1.5, 1.5, -1.5, 1.5])
		#
		# Ready to complete plot
		#
		return fig
	# ------------------------------------------------------------------------------------------

	def plot_eval(self):

		fig, ax = plt.subplots()

		x = self.avg_eval.index
		y = self.avg_eval
		# print(f"DEBUG -- {x = }")
		# print(f"DEBUG -- {y = }")

		plt.barh(x, y)
		ax.set_xlabel("Average Evaluation")
		#
		return fig

		# --------------------------------------------------------------------------------------------

	def plot_first(self, reply):
		#
		# Initialize variables needed
		#
		x_coords = []
		y_coords = []
		x_left_right = []
		y_left_right = []
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Setting aspect ratio
		#
		ax.set_aspect("equal")
		#
		ax.set_title("First Dimension")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# Add reference points labels
		#
		if self.have_reference_points():
			ax.text(
				self.point_coords.iloc[self.rival_a][self.hor_dim] + self.move_label,
				self.point_coords.iloc[self.rival_a][self.vert_dim],
				self.point_labels[self.rival_a])
			ax.text(
				self.point_coords.iloc[self.rival_b][self.hor_dim] + self.move_label,
				self.point_coords.iloc[self.rival_b][self.vert_dim],
				self.point_labels[self.rival_b])
			x_coords.append(self.point_coords.iloc[self.rival_a][self.hor_dim])
			y_coords.append(self.point_coords.iloc[self.rival_a][self.vert_dim])
			x_coords.append(self.point_coords.iloc[self.rival_b][self.hor_dim])
			y_coords.append(self.point_coords.iloc[self.rival_b][self.vert_dim])
			#
			ax.scatter(x_coords, y_coords)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		# Add Partisan divider - cancelled
		#
		# ax.plot([0, 0], [self.vert_max, self.vert_min])
		#
		# Shade regions on first dimension
		#
		# print(f"DEBUG -- {self.dim1_div = }")
		self.first_right.vertices = np.array([
			[self.hor_max, self.vert_max],
			[self.dim1_div, self.vert_max],
			[self.dim1_div, self.vert_min],
			[self.hor_max, self.vert_min]
		])
		self.first_left.vertices = np.array([
			[self.hor_min, self.vert_max],
			[self.dim1_div, self.vert_max],
			[self.dim1_div, self.vert_min],
			[self.hor_min, self.vert_min]
		])
		# # Taking transpose
		# x, y = data.T
		self.first_left.x, self.first_left.y = self.first_left.vertices.T
		self.first_left.color = "green"  # eventually replace with blue
		self.first_right.x, self.first_right.y = self.first_right.vertices.T
		self.first_right.color = "pink"  # eventually replace with red
		#
		ax.fill(self.first_left.x, self.first_left.y, self.first_left.color)
		ax.fill(self.first_right.x, self.first_right.y, self.first_right.color)
		#
		if self.have_segments():
			match reply[0:4]:
				#
				case"left":
					x_left_right = [
						self.seg.loc[each_individ, "Dim1_score"] for each_individ in self.range_n_individ if
						self.seg.loc[each_individ, "Only_Dim1"] == 1]
					y_left_right = [
						self.seg.loc[each_individ, "Dim2_score"] for each_individ in self.range_n_individ if
						self.seg.loc[each_individ, "Only_Dim1"] == 1]
				case "righ":
					x_left_right = [
						self.seg.loc[each_individ, "Dim1_score"] for each_individ in self.range_n_individ if
						self.seg.loc[each_individ, "Only_Dim1"] == 2]
					y_left_right = [
						self.seg.loc[each_individ, "Dim2_score"] for each_individ in self.range_n_individ if
						self.seg.loc[each_individ, "Only_Dim1"] == 2]
				# case "both":
				# 	x_left_right = [
				# 		self.seg.loc[each_individ, "Dim1_score"] for each_individ in self.range_n_individ if
				# 		self.seg.loc[each_individ, "Only_Dim1"] in (1, 2)]
				# 	y_left_right = [
				# 		self.seg.loc[each_individ, "Dim2_score"] for each_individ in self.range_n_individ if
				# 		self.seg.loc[each_individ, "Only_Dim1"] in (1, 2)]
			#
			ax.scatter(x_left_right, y_left_right, color="black", s=self.point_size)
		#
		# Ready to complete plot
		#
		return fig
	#
	# ----------------------------------------------------------------------------------------

	def plot_grouped(self):
		x_coords = []
		y_coords = []

		#
		fig, ax = plt.subplots()
		# ax.scatter(
		#	self.point_coords.iloc[:,0],
		#	self.point_coords.iloc[:,1],

		# )
		# ax.set_xlabel(self.dim_names[0])
		# ax.set_ylabel(self.dim_names[1])

		# plt.text(self.point_coords.iloc[:,0],
		#	self.point_coords.iloc[:,1],
		#		self.point_coords.index)

		# x_coords = self.point_coords.iloc[:,0]
		# y_coords = self.point_coords.iloc[:,1]
		# Begin the building of the plot
		#
		# Setting aspect ratio causes warning message
		#
		ax.set_aspect("equal")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names_grpd[self.hor_dim])
		ax.set_ylabel(self.dim_names_grpd[self.vert_dim])
		#
		x_coords.append(self.point_coords_grpd.iloc[:, self.hor_dim])
		y_coords.append(self.point_coords_grpd.iloc[:, self.vert_dim])
		#
		for each_point in self.range_points_grpd:
			ax.text(
				self.point_coords_grpd.iloc[each_point][self.hor_dim] + self.move_label,
				self.point_coords_grpd.iloc[each_point][self.vert_dim],
				self.point_labels_grpd[each_point])
		ax.scatter(x_coords, y_coords, color="black", s=5)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		# Show bisector if requested
		#
		if self.show_connector:
			#
			# Draw a line connecting the reference points
			#
			ax.plot(
				[self.point_coords_grpd.iloc[self.rival_a][self.hor_dim],
				self.point_coords_grpd.iloc[self.rival_b][self.hor_dim]],
				[self.point_coords_grpd.iloc[self.rival_a][self.vert_dim],
				self.point_coords_grpd.iloc[self.rival_b][self.vert_dim]])
		if self.show_bisector:
			#
			# Calculate ends of bisector
			#
			self.ends_of_bisector_function()
			#
			# Add bisector to plot
			# For debugging label Start and End
			#
			ax.text(self.bisector.start_x, self.bisector.start_y, "S")
			ax.text(self.bisector.end_x, self.bisector.end_y, "E")
			#
			# For Debugging label midpoint as M
			#
			ax.text(self.connector_bisector_cross_x, self.connector_bisector_cross_y, "M")
			#
			ax.text(self.bisector.start_x, self.bisector.start_y, "S")
			ax.text(self.bisector.end_x, self.bisector.end_y, "E")
			#
			# Draw bisector
			#
			ax.plot(
				[self.bisector.start_x, self.bisector.end_x],
				[self.bisector.start_y, self.bisector.end_y])
		# Ready to complete plot
		#
		return fig

	# ---------------------------------------------------------------------------------------

	def plot_individuals(self):

		fig, ax = plt.subplots()
		#
		ax.set_aspect("equal")
		ax.set_xlabel(self.hor_axis_name)
		ax.set_ylabel(self.vert_axis_name)
		ax.grid(True)
		print(f"DEBUG -- {self.hor_min = }")
		print(f"DEBUG -- {self.hor_max = }")
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		# add points and labels to plot
		#
		ax.scatter(self.dim1, self.dim2, color="green", s=self.point_size)
		#
		# Ready to complete plot
		#
		return fig

	# --------------------------------------------------------------------------------------

	def plot_joint(self):

		fig, ax = plt.subplots()

		x_coords = []
		y_coords = []
		#
		# Begin the building of the plot
		#
		ax.set_aspect("equal")
		# Add all or just reference point's coordinates to coordinate vectors and labels to plot
		#
		for each_point in self.range_points:
			if self.show_just_reference_points == "No":
				x_coords.append(self.point_coords.iloc[each_point][self.hor_dim])
				y_coords.append(self.point_coords.iloc[each_point][self.vert_dim])
				ax.text(
					self.point_coords.iloc[each_point][self.hor_dim] + self.move_label,
					self.point_coords.iloc[each_point][self.vert_dim],
					self.point_labels[each_point])
			elif self.show_just_reference_points == "Yes":
				if each_point == self.rival_a or each_point == self.rival_b:
					x_coords.append(self.point_coords.iloc[each_point][self.hor_dim])
					y_coords.append(self.point_coords.iloc[each_point][self.vert_dim])
					ax.text(
						self.point_coords.iloc[each_point][self.hor_dim] + self.move_label,
						self.point_coords.iloc[each_point][self.vert_dim],
						self.point_labels[each_point])
		#
		# Add points to plot by passing in coordinate vectors
		# This is done for all points regardless of whether they are basic or reference
		# all of which have been added conditionally above ???????????????????????
		#
		ax.scatter(x_coords, y_coords)
		#
		# Add individual data to plot
		#
		maxdim1 = max(self.dim1)
		mindim1 = min(self.dim1)
		maxdim2 = max(self.dim2)
		mindim2 = min(self.dim2)
		all_max = maxdim1
		if maxdim2 > maxdim1:
			all_max = maxdim2
		all_min = mindim1
		if mindim2 < mindim1:
			all_min = mindim2
		#
		# This adds a little to the axes so points do not sit on edge
		#
		all_max = all_max + (.05 * all_max)
		all_min = all_min - (.05 * all_min)

		if all_max < abs(all_min):
			all_max = abs(all_min)
		#
		# If Reference points exist
		#
		if self.have_reference_points():
			#
			#  Show bisector if requested
			#
			if self.show_bisector:
				#
				# Calculate ends of bisector
				#
				self.ends_of_bisector_function()
				#
				# Draw a line connecting the reference points
				#
				ax.plot(
					[self.point_coords.iloc[self.rival_a][self.hor_dim],
					self.point_coords.iloc[self.rival_b][self.hor_dim]],
					[self.point_coords.iloc[self.rival_a][self.vert_dim],
					self.point_coords.iloc[self.rival_b][self.vert_dim]])
				#
				# need to extend bisector to larger space defined by individual points
				#
				needed_extra = all_max - self.hor_max
				#
				if self.bisector.direction == "Vertical":
					ax.plot(
						[self.bisector.start_x, self.bisector.end_x],
						[self.bisector.start_y + needed_extra, self.bisector.end_y - needed_extra])
					#
					# For debugging label Start and End
					#
					ax.text(self.bisector.start_x, (self.bisector.start_y + needed_extra), "S")
					ax.text(self.bisector.end_x, (self.bisector.end_y - needed_extra), "E")
				#
				elif self.bisector.direction == "Flat":
					ax.plot(
						[(self.bisector.start_x - needed_extra), (self.bisector.end_x + needed_extra)],
						[self.bisector.start_y, self.bisector.end_y])
					#
					# For debugging label Start and End
					#
					ax.text((self.bisector.start_x - needed_extra), self.bisector.start_y, "S")
					ax.text((self.bisector.end_x + needed_extra), self.bisector.end_y, "E")

				elif self.bisector.direction == "Upward slope":
					ax.plot(
						[self.bisector.start_x - needed_extra, self.bisector.end_x + needed_extra],
						[self.bisector.start_y - needed_extra, self.bisector.end_y + needed_extra])
					#
					# For debugging label Start and End
					#
					ax.text((self.bisector.start_x - needed_extra), (self.bisector.start_y - needed_extra), "S")
					ax.text((self.bisector.end_x + needed_extra), (self.bisector.end_y + needed_extra), "E")
				#
				elif self.bisector.direction in ("Downward slope", "Flat"):
					ax.plot(
						[(self.bisector.start_x - needed_extra), (self.bisector.end_x + needed_extra)],
						[(self.bisector.start_y + needed_extra), (self.bisector.end_y - needed_extra)])
					#
					# For debugging label Start and End
					#
					ax.text((self.bisector.start_x - needed_extra), (self.bisector.start_y + needed_extra), "S")
					ax.text((self.bisector.end_x + needed_extra), (self.bisector.end_y - needed_extra), "E")
				#
				# For Debugging label midpoint as M
				#
				ax.text(self.connector_bisector_cross_x, self.connector_bisector_cross_y, "M")
		#
		# Begin the building of the plot - why plot, why hot just read ??????????????????????????
		#
		ax.set_xlabel(self.hor_axis_name)
		ax.set_ylabel(self.vert_axis_name)
		ax.grid(True)
		#
		# add points and labels to plot
		#
		ax.scatter(self.dim1, self.dim2, c="green", s=self.point_size)
		#
		# Ready to complete plot
		#
		ax.axis([(-all_max), all_max, (-all_max), all_max])
		#
		# Plot Configuration
		#
		return fig

	# -------------------------------------------------------------------------------------------

	def plot_likely(self, reply):

		#
		# Initialize variables needed
		#
		x_coords = []
		y_coords = []
		x_likely = []
		y_likely = []
		# self.likely_right_x.clear()
		# self.likely_right_y.clear()
		# self.likely_left_x.clear()
		# self.likely_left_y.clear()
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Setting aspect ratio
		#
		ax.set_aspect("equal")
		#
		ax.set_title("Likely Supporters")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# Add reference points labels and coordinate vectors to plot
		#
		ax.text(
			self.point_coords.iloc[self.rival_a][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_a][self.vert_dim],
			self.point_labels[self.rival_a])
		ax.text(
			self.point_coords.iloc[self.rival_b][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_b][self.vert_dim],
			self.point_labels[self.rival_b])
		x_coords.append(self.point_coords.iloc[self.rival_a][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_a][self.vert_dim])
		x_coords.append(self.point_coords.iloc[self.rival_b][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_b][self.vert_dim])
		#
		ax.scatter(x_coords, y_coords)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])

		# Calculate ends of bisector
		#
		self.ends_of_bisector_function()
		#
		# Create likely regions around the reference points
		#
		# Bisector Case 0a Bisector is flat - from left to right
		#
		# print(f"DEBUG -- {self.bisector.case = }")

		match self.bisector.case:
			case "0a":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_min, self.vert_max]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min]
				])
			#
			# Bisector Case 0b Bisector is vertical - from top to bottom
			#
			case "0b":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_max, self.vert_max]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_min, self.vert_min],
					[self.hor_min, self.vert_max]
				])
			#
			# Bisector Case Ia Positive slope from Left side to Right side and min_y > vert_min
			#
			case "Ia":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_min, self.vert_max]
				])
			#
			# Bisector Case IIa Positive slope from Left side to Top and max_y == vert_max
			#
			case "IIa":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_min, self.vert_max]
				])
			#
			# Bisector Case IIIa Positive slope from Bottom to Right side
			#
			case "IIIa":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_min, self.vert_max],
					[self.hor_min, self.vert_min]
				])
			#
			# Bisector Case IVa Positive slope from Bottom to Top and min_x < hor_min
			#
			case "IVa":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_max, self.vert_min]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_min, self.vert_max],
					[self.hor_min, self.vert_min]
				])
			#
			# Bisector Case Ib Negative slope from Left side to Right side
			#
			case "Ib":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_max],
					[self.hor_min, self.vert_max]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min]
				])
			#
			# Bisector Case IIb Negative slope from Left side to Bottom and min_y == vert_min
			#
			case "IIb":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_max, self.vert_max],
					[self.hor_min, self.vert_max]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_min, self.vert_min]
				])
			#
			# Bisector Case IIIb Negative slope from Top to Right side and min_x < hor_min ?????????
			#
			case "IIIb":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_max]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_min, self.vert_min],
					[self.hor_min, self.vert_max]
				])
			#
			# Bisector Case IVb Negative slope from Bottom to Top and max_y > vert_max  ????????????????????????
			#
			case "IVb":
				self.likely_right.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_max, self.vert_min],
					[self.hor_max, self.vert_max]
				])
				self.likely_left.vertices = np.array([
					[self.bisector.start_x, self.bisector.start_y],
					[self.bisector.end_x, self.bisector.end_y],
					[self.hor_min, self.vert_min],
					[self.hor_min, self.vert_max]
				])

		# # Taking transpose
		# x, y = data.T
		self.likely_left.x, self.likely_left.y = self.likely_left.vertices.T
		self.likely_left.color = "green"  # eventually replace with blue
		self.likely_right.x, self.likely_right.y = self.likely_right.vertices.T
		self.likely_right.color = "pink"  # eventually replace with red
		#
		# Shade likely regions
		#
		if self.bisector.case in ("0a", "0b", "Ia", "IIa", "IIIa", "IVa", "Ib", "IIb", "IIIb", "IVb"):
			ax.fill(self.likely_right.x, self.likely_right.y, self.likely_right.color)
			ax.fill(self.likely_left.x, self.likely_left.y, self.likely_left.color)
		else:
			ax.fill(self.likely_right_x, self.likely_right_y, "red")
			ax.fill(self.likely_left_x, self.likely_left_y, "blue")
		#
		if self.have_segments():
			if reply[0:4] in ["left", "righ", "both"]:
				match reply[0:4]:
					case "left":
						x_likely = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ if self.seg.loc[ind, "Likely"] == 1]
						y_likely = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ if self.seg.loc[ind, "Likely"] == 1]
					case "righ":
						x_likely = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ if self.seg.loc[ind, "Likely"] == 2]
						y_likely = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ if self.seg.loc[ind, "Likely"] == 2]
					case "both":
						x_likely = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Likely"] in (1, 2)]
						y_likely = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Likely"] in (1, 2)]
				#
				ax.scatter(x_likely, y_likely, color="black", s=self.point_size)
		#
		# Ready to complete plot
		#
		return fig

	# --------------------------------------------------------------------------------------------

	def plot_battleground(self, reply):
		""" show battleground function - creates a plot showing the reference points
			and an area where battleground individuals are most likely found.
		"""
		#
		# This function assumes the calling function has determined that reference
		# point have been established and a bisector should be shown
		#
		# Initialize variables needed
		#
		x_coords = []
		y_coords = []
		x = []
		y = []
		x_battleground = []
		y_battleground = []

		if self.have_segments():
			match reply[0:4]:
				case "batt":
					x_battleground = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Battle_ground"] == 1]
					y_battleground = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Battle_ground"] == 1]
				case "sett":
					x_battleground = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Battle_ground"] == 2]
					y_battleground = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
									if self.seg.loc[ind, "Battle_ground"] == 2]

		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Setting aspect ratio
		#
		ax.set_aspect("equal")
		#
		ax.set_title("Battleground")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# Add reference points to coordinate vectors and labels to plot
		#
		ax.text(
			self.point_coords.iloc[self.rival_a][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_a][self.vert_dim],
			self.point_labels[self.rival_a])
		ax.text(
			self.point_coords.iloc[self.rival_b][self.hor_dim] + self.move_label,
			self.point_coords.iloc[self.rival_b][self.vert_dim],
			self.point_labels[self.rival_b])
		x_coords.append(self.point_coords.iloc[self.rival_a][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_a][self.vert_dim])
		x_coords.append(self.point_coords.iloc[self.rival_b][self.hor_dim])
		y_coords.append(self.point_coords.iloc[self.rival_b][self.vert_dim])
		#
		# Add points to plot by passing in coordinate vectors
		#
		ax.scatter(x_coords, y_coords)
		#
		# Draw a line connecting the reference points
		#
		ax.plot(
			[self.point_coords.iloc[self.rival_a][self.hor_dim], self.point_coords.iloc[self.rival_b][self.hor_dim]],
			[self.point_coords.iloc[self.rival_a][self.vert_dim], self.point_coords.iloc[self.rival_b][self.vert_dim]])
		#
		# For Debugging label midpoint as M
		#
		ax.text(self.connector_bisector_cross_x, self.connector_bisector_cross_y, "M")
		#
		# Add bisector to plot
		#
		# For debugging label Start and End
		#
		ax.text(self.bisector.start_x, self.bisector.start_y, "S")
		ax.text(self.bisector.end_x, self.bisector.end_y, "E")
		#
		ax.plot(
			[self.bisector.start_x, self.bisector.end_x],
			[self.bisector.start_y, self.bisector.end_y])
		#
		# Edge 1 line goes through a point on line connecting points
		# Solve for that point's coordinates and add it to plot
		#
		ax.text(self.west_connector_cross_x, self.west_connector_cross_y, "W")
		ax.scatter(self.west_connector_cross_x, self.west_connector_cross_y)
		#
		# Label the ends of edge 1
		#
		ax.text(self.west.start_x, self.west.start_y, "W_S")
		ax.text(self.west.end_x, self.west.end_y, "W_E")
		#
		# Add edge 1 to plot
		#
		ax.plot([self.west.start_x, self.west.end_x], [self.west.start_y, self.west.end_y])
		#
		# Begin edge 2
		#
		# Edge 2 line goes through a point on line connecting points
		#
		ax.text(self.east_connector_cross_x, self.east_connector_cross_y, "E")
		ax.scatter(self.east_connector_cross_x, self.east_connector_cross_y)
		#
		# Label the ends of edge 2
		#
		ax.text(self.east.start_x, self.east.start_y, "E_S")
		ax.text(self.east.end_x, self.east.end_y, "E_E")
		#
		# Add edge 2 to plot
		#
		ax.plot([self.east.start_x, self.east.end_x], [self.east.start_y, self.east.end_y])

		# print(f"DEBUG -- {self.bisector.case = }")
		# print(f"DEBUG -- {self.west.case = }")
		# print(f"DEBUG -- {self.east.case = }")
		#
		# west Case 0a west slope is zero from Left side to Right side --- 0a - west & right
		#
		# Set up shading battleground area
		#
		if (self.west.goes_through_right_side == self.east.goes_through_right_side) \
			and (self.west.goes_through_left_side == self.east.goes_through_left_side) \
			and (self.west.goes_through_top == self.east.goes_through_top) \
			and (self.west.goes_through_bottom == self.east.goes_through_bottom):
			self.battleground.vertices = np.array([
				[self.west.start_x, self.west.start_y],
				[self.west.end_x, self.west.end_y],
				[self.east.end_x, self.east.end_y],
				[self.east.start_x, self.east.start_y]
			])
			if (
				self.bisector.goes_through_top == "Yes"
				and self.bisector.goes_through_bottom == "Yes") \
				or (
				self.bisector.goes_through_left_side == "Yes"
					and self.bisector.goes_through_right_side == "Yes"):

				self.convertible_to_left_x = [
					self.west.start_x, self.west.end_x,
					self.bisector.end_x, self.bisector.start_x]
				self.convertible_to_left_y = [
					self.west.start_y, self.west.end_y,
					self.bisector.end_y, self.bisector.start_y]
				self.convertible_to_right_x = [
					self.east.start_x, self.east.end_x,
					self.bisector.end_x, self.bisector.start_x]
				self.convertible_to_right_y = [
					self.east.start_y, self.east.end_y,
					self.bisector.end_y, self.bisector.start_y]

		#
		# Top Left only
		elif \
			((self.east.goes_through_top == "Yes")
				and (self.east.goes_through_bottom == "Yes")
				and (self.west.goes_through_left_side == "Yes")
				and (self.west.goes_through_bottom == "Yes")) \
			or \
			((self.east.goes_through_top == "Yes")
				and (self.east.goes_through_right_side == "Yes")
				and (self.west.goes_through_left_side == "Yes")
				and (self.west.goes_through_right_side == "Yes")):
			#
			self.battleground.vertices = np.array([
				[self.east.start_x, self.east.start_y],
				[self.east.end_x, self.east.end_y],
				[self.west.end_x, self.west.end_y],
				[self.west.start_x, self.west.start_y],
				[self.hor_min, self.vert_max]
			])
		#
		# Top right only
		elif \
			((self.west.goes_through_top == "Yes")
				and (self.west.goes_through_bottom == "Yes")
				and (self.east.goes_through_right_side == "Yes")
				and (self.east.goes_through_bottom == "Yes")) \
			or \
			((self.west.goes_through_top == "Yes")
				and (self.west.goes_through_left_side == "Yes")
				and (self.east.goes_through_right_side == "Yes")
				and (self.east.goes_through_left_side == "Yes")):
			#
			self.battleground.vertices = np.array([
				[self.west.start_x, self.west.start_y],
				[self.west.end_x, self.west.end_y],
				[self.hor_max, self.vert_max],
				[self.east.end_x, self.east.end_y],
				[self.east.start_x, self.east.start_y]
			])
		#
		# Bottom right only
		elif \
			((self.east.goes_through_top == "Yes")
				and (self.east.goes_through_right_side == "Yes")
				and (self.west.goes_through_top == "Yes")
				and (self.west.goes_through_bottom == "Yes")) \
			or \
			((self.east.goes_through_left_side == "Yes")
				and (self.east.goes_through_right_side == "Yes")
				and (self.west.goes_through_left_side == "Yes")
				and (self.west.goes_through_bottom == "Yes")):
			#
			self.battleground.vertices = np.array([
				[self.west.start_x, self.west.start_y],
				[self.west.end_x, self.west.end_y],
				[self.hor_max, self.vert_min],
				[self.east.end_x, self.east.end_y],
				[self.east.start_x, self.east.start_y]
			])
		#
		# Bottom left only
		elif \
			((self.west.goes_through_top == "Yes")
				and (self.west.goes_through_left_side == "Yes")
				and (self.east.goes_through_top == "Yes")
				and (self.east.goes_through_bottom == "Yes")) \
			or \
			((self.west.goes_through_right_side == "Yes")
				and (self.west.goes_through_left_side == "Yes")
				and (self.east.goes_through_right_side == "Yes")
				and (self.east.goes_through_bottom == "Yes")):
			#
			self.battleground.vertices = np.array([
				[self.west.start_x, self.west.start_y],
				[self.west.end_x, self.west.end_y],
				[self.east.end_x, self.east.end_y],
				[self.east.start_x, self.east.start_y],
				[self.hor_min, self.vert_min]
			])
			#
		# Both top right and bottom left ------ need reverse of 1 and 2 ????
		elif (self.west.goes_through_top == "Yes") and (self.west.goes_through_left_side == "Yes") \
			and (self.east.goes_through_right_side == "Yes") and (self.east.goes_through_bottom == "Yes"):
			#
			self.battleground.vertices = np.array([
				[self.west.start_x, self.west.start_y],
				[self.west.end_x, self.west.end_y],
				[self.hor_max, self.vert_max],
				[self.east.end_x, self.east.end_y],
				[self.east.start_x, self.east.start_y],
				[self.hor_min, self.vert_min]
			])
		#
		# Both top left and bottom right ------ need reverse of 1 and 2 ????
		elif (self.east.goes_through_top == "Yes") and (self.east.goes_through_right_side == "Yes") \
			and (self.west.goes_through_bottom == "Yes") and (self.west.goes_through_left_side == "Yes"):
			#
			self.battleground.vertices = np.array([
				[self.west.start_x, self.west.start_y],
				[self.west.end_x, self.west.end_y],
				[self.hor_max, self.vert_min],
				[self.east.end_x, self.east.end_y],
				[self.east.start_x, self.east.start_y],
				[self.hor_min, self.vert_max]
			])
		#
		# Shade battleground area
		#
		# # Taking transpose
		# x, y = data.T
		self.battleground.x, self.battleground.y = self.battleground.vertices.T
		self.battleground.color = "green"  # eventually replace with purple

		ax.fill(self.battleground.x, self.battleground.y, self.battleground.color)

		#
		if self.have_segments():
			ax.scatter(x_battleground, y_battleground, color="black", s=self.point_size)
		#
		# Ready to complete plot
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		return fig

	# ------------------------------------------------------------------------------------------

	def plot_ranks(self):

		#
		#
		sim = []
		dim = []
		#
		fig, ax = plt.subplots()

		ax.set_aspect("equal")
		#
		ax.set_title("Rank of Inter-Point Distance vs Rank of Similarity ")
		#
		ax.set_xlabel("Rank of Inter-Point Distance")
		ax.set_ylabel("Rank of Similarity")
		#
		sim = self.df["Similarity_Rank"].values.tolist()
		dim = self.df["Distance_Rank"].values.tolist()
		scree_max = len(sim) + 10

		ax.scatter(sim, dim, color="black", s=self.point_size)

		ax.axis([0, scree_max, 0, scree_max])

		# ax1 = self.df.plot.scatter("Similarity_Rank", "Distance_Rank")
		#
		return fig

	# -------------------------------------------------------------------------------------------

	def plot_scree(self):
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Setting aspect ratio
		#
		ax.set_aspect("equal")
		#
		ax.set_title("Scree Diagram")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel("Number of Dimensions")
		ax.set_ylabel("Stress")
		#
		x_coords = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		# x_coords = [1, 2, 3, 4]
		y_coords = self.min_stress
		#
		ax.plot(x_coords, y_coords)
		#
		show_max_stress = int(self.min_stress[0] + 2)
		ax.axis([1, 10, 0, show_max_stress])
		#
		# Ready to complete plot
		#
		return fig

# ----------------------------------------------------------------------------------------------------

	def plot_second(self, reply):

		x_coords = []
		y_coords = []
		x_up_down = []
		y_up_down = []

		if self.have_segments():
			match reply[0:4]:

				case "uppe":
					x_up_down = [
						self.seg.loc[ind, "Dim1_score"]
						for ind in self.range_n_individ
						if self.seg.loc[ind, "Only_Dim2"] == 1
					]
					y_up_down = [
						self.seg.loc[ind, "Dim2_score"]
						for ind in self.range_n_individ
						if self.seg.loc[ind, "Only_Dim2"] == 1
					]

				case "lowe":
					x_up_down = [
						self.seg.loc[ind, "Dim1_score"]
						for ind in self.range_n_individ
						if self.seg.loc[ind, "Only_Dim2"] == 2
					]
					y_up_down = [
						self.seg.loc[ind, "Dim2_score"]
						for ind in self.range_n_individ
						if self.seg.loc[ind, "Only_Dim2"] == 2
					]

				# case "both":
				# 	x_up_down = [self.seg.loc[ind, "Dim1_score"] for ind in self.range_n_individ
				# 				if self.seg.loc[ind, "Only_Dim2"] in (1, 2)]
				# 	y_up_down = [self.seg.loc[ind, "Dim2_score"] for ind in self.range_n_individ
				# 				if self.seg.loc[ind, "Only_Dim2"] in (1, 2)]
		#
		# Begin the building of the plot
		#
		fig, ax = plt.subplots()
		#
		# Setting aspect ratio
		#
		ax.set_aspect("equal")
		#
		ax.set_title("Second Dimension")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		# Add reference points labels
		#
		if self.have_reference_points():
			ax.text(
				self.point_coords.iloc[self.rival_a][self.hor_dim] + self.move_label,
				self.point_coords.iloc[self.rival_a][self.vert_dim],
				self.point_labels[self.rival_a])
			ax.text(
				self.point_coords.iloc[self.rival_b][self.hor_dim] + self.move_label,
				self.point_coords.iloc[self.rival_b][self.vert_dim],
				self.point_labels[self.rival_b])
			x_coords.append(self.point_coords.iloc[self.rival_a][self.hor_dim])
			y_coords.append(self.point_coords.iloc[self.rival_a][self.vert_dim])
			x_coords.append(self.point_coords.iloc[self.rival_b][self.hor_dim])
			y_coords.append(self.point_coords.iloc[self.rival_b][self.vert_dim])
			#
			ax.scatter(x_coords, y_coords)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#
		# Add Second dimension divider - cancelled
		#
		# ax.plot([self.hor_max, self.hor_min], [0.0, 0.0])
		#
		# Shade regions on second dimension
		#
		# print(f"DEBUG -- {self.dim2_div = }")
		self.second_down.vertices = np.array([
			[self.hor_max, self.dim2_div],
			[self.hor_max, self.vert_min],
			[self.hor_min, self.vert_min],
			[self.hor_min, self.dim2_div]
		])
		self.second_up.vertices = np.array([
			[self.hor_min, self.dim2_div],
			[self.hor_min, self.vert_max],
			[self.hor_max, self.vert_max],
			[self.hor_max, self.dim2_div]
		])
		# # Taking transpose
		# x, y = data.T
		self.second_up.x, self.second_up.y = self.second_up.vertices.T
		self.second_up.color = "green"  # eventually replace with blue
		self.second_down.x, self.second_down.y = self.second_down.vertices.T
		self.second_down.color = "pink"  # eventually replace with red

		ax.fill(self.second_up.x, self.second_up.y, self.second_up.color)
		ax.fill(self.second_down.x, self.second_down.y, self.second_down.color)
		#
		if self.have_segments():
			ax.scatter(x_up_down, y_up_down, color="black", s=self.point_size)
		# Ready to complete plot
		#
		return fig

# ------------------------------------------------------------------------------------

	def plot_shep(self, axis):

		fig, ax = plt.subplots()
		#
		ax.set_title("Shepard Diagram")
		#
		if axis == 'y':
			hor_max = max(self.distances_as_list)
			hor_min = min(self.distances_as_list)
			vert_max = min(self.similarities_as_list)
			vert_min = max(self.similarities_as_list)
			#
			ax.set_xlabel("Distance")
			ax.set_ylabel("Similarity")
			#
			for each_dyad in self.range_similarities:
				if each_dyad != self.ndyad - 1:
					ax.plot(
						(self.df.at[each_dyad, "Distance_AB"],
						self.df.at[each_dyad + 1, "Distance_AB"]),
						(self.df.at[each_dyad, "Similarity"],
						self.df.at[each_dyad + 1, "Similarity"]),
						color="k")
		else:
			vert_max = max(self.distances_as_list)
			vert_min = min(self.distances_as_list)
			hor_max = min(self.similarities_as_list)
			hor_min = max(self.similarities_as_list)

			ax.set_ylabel("Distance")
			ax.set_xlabel("Similarity")
			#
			for each_dyad in self.range_similarities:
				if each_dyad != self.ndyad - 1:
					ax.plot(
						(self.df.at[each_dyad, "Similarity"],
						self.df.at[each_dyad + 1, "Similarity"]),
						(self.df.at[each_dyad, "Distance_AB"],
						self.df.at[each_dyad + 1, "Distance_AB"]),
						color="k")
		#
		# Ready to complete plot
		#
		ax.axis([hor_min, hor_max, vert_min, vert_max])
		#
		return fig
	# -----------------------------------------------------------------------------------

	def plot_stress_by_point(self, point, point_index):

		fig, ax = plt.subplots()
		#
		x_others = []
		y_others = []
		label_others = []
		index_others = []
		for each_dyad in self.range_similarities:
			if self.df.iloc[each_dyad][point] != "INAP":
				x_others.append(self.df.at[each_dyad, "Similarity_Rank"])
				y_others.append(self.df.at[each_dyad, "Distance_Rank"])
				label_others.append(self.df.at[each_dyad, point])
				index_others.append(each_dyad)
		#
		# Create plot for selected item
		#
		ax.set_aspect("equal")
		ax.set_title(self.item_names[point_index])
		ax.set_xlabel("Similarity Rank")
		ax.set_ylabel("Distance Rank")
		ax.scatter(x_others, y_others)
		other_range = range(len(x_others))
		for each_item in other_range:
			ax.text(
				x_others[each_item],
				y_others[each_item],
				label_others[each_item])
			ax.plot(
				[index_others[each_item] + 1, x_others[each_item]],
				[index_others[each_item] + 1, y_others[each_item]], 'b')
		#
		# Add line to indicate what a perfect relationship, no stress, would be
		#
		ax.plot((1, self.ndyad + 1), (1, self.ndyad + 1), 'r')
		#
		return fig

	# -------------------------------------------------------------------------------------------

	def plot_vectors(self):

		# print("DEBUG -- at top of plot_vectors")

		#x_coords = []
		#y_coords = []
		#
		fig, ax = plt.subplots()
			# Begin the building of the plot
		#
		# Setting aspect ratio causes warning message
		#
		ax.set_aspect("equal")
		#
		# Label the horizontal and vertical dimensions
		#
		ax.set_xlabel(self.dim_names[self.hor_dim])
		ax.set_ylabel(self.dim_names[self.vert_dim])
		#
		for each_point in self.range_points:
			if self.point_coords.iloc[each_point][self.hor_dim] >= 0.0:
				ax.text(
					self.point_coords.iloc[each_point][self.hor_dim]
						+ (2 * self.move_label),
					self.point_coords.iloc[each_point][self.vert_dim],
					self.point_labels[each_point])
			else:
				ax.text(
					self.point_coords.iloc[each_point][self.hor_dim]
						- (2 * len(self.point_labels[each_point]) * self.move_label),
					self.point_coords.iloc[each_point][self.vert_dim],
					self.point_labels[each_point])
			ax.arrow(
					0, 0,
					self.point_coords.iloc[each_point][self.hor_dim],
					self.point_coords.iloc[each_point][self.vert_dim],
					color="black",
					head_width=self.vector_head_width,
					width=self.vector_width
			)
		#
		ax.axis([self.hor_min, self.hor_max, self.vert_min, self.vert_max])
		#

		# Ready to complete plot
		#
		return fig

	# ---------------------------------------------------------------------------------

	def print_active_function(self):

		""" print active function - is used by many command to print the active configuration.
		"""
		print(self.point_coords)
		return None

	# ----------------------------------------------------------------------------------

	def print_grouped_function(self):
		""" print active grouped - is used by many command to print the active configuration.
		"""
		print(self.point_coords_grpd)

		return None

	# --------------------------------------------------------------------------------------

	def print_lower_triangle(self, decimals, labels, names, nelements, values, width):
		""" print_lower_triangle function - is used by commands to print values in a
			lower triangular format. Within print_lower_triangle the elements of the
			matrix are considered values regardless of whether the calling routine
			called them correlations, similarities or even dissimilarities.
			Analogously within this function it refers to labels and names rather
			than item_labels and item_names. And lastly it uses nelements rather
			than npoints.
		"""
		#
		# 	Define needed variables
		#
		one_less = int(width - 1)
		#
		# Print the labels and names of points in the lower triangle
		#
		print("\n\tItems:")
		for index, each_item in enumerate(labels):
			print("\t\t", each_item, "\t", names[index])
		#
		# Print column headings using the labels of the points
		#
		# labs = labels
		# print("\n       ", *labs, sep="    ")
		formatted_labels = ""
		# for each_point in npoints:
		# curr_label = labels[each_point]
		range_labels = range(len(labels))
		for each_label in range_labels:
			formatted_labels = formatted_labels + '{:>{width}}'.format(labels[each_label], width=width)
		print("\n", (one_less * " "), formatted_labels)
		#
		# Print line with just the label of the first item
		#
		an_item = 0
		print("  ", labels[an_item], "   -----")
		#
		# Assemble a line with the values for all the points below this point
		# Print a line for each of the remaining points in the lower triangle
		#
		npoints = range(1, nelements)
		for each_point in npoints:
			curr_values = values[each_point - 1]
			formatted_values = ""
			for value in curr_values:
				formatted_values = formatted_values + '{:{width}.{decimals}f}'.format(value, width=width,
					decimals=decimals)
			print("  ", labels[each_point], formatted_values)
		#
		return None

	#

	# ------------------------------------------------------------------------------------------

	def print_most_similar(self, cut_point):
		self.a_x_alike.clear()
		self.a_y_alike.clear()
		self.b_x_alike.clear()
		self.b_y_alike.clear()
		cut = "Null"

		#
		# Print pairs above/below cutoff depending on value_type
		#
		print("\t\tMost similar pairs: ")
		#
		each_similarity = 0
		while self.zipped[each_similarity][0] < cut_point:
			#
			print(
				"\t\t", self.zipped[each_similarity][1], ' ',
				self.zipped[each_similarity][2], " ",
				self.zipped[each_similarity][0])
			each_similarity += 1
		#
		# create coordinate lists for plotting
		#
		from_points = range(1, self.nreferent)
		for an_item in from_points:
			to_points = range(an_item)
			for another_item in to_points:
				if self.value_type == "similarities":
					if self.similarities[an_item - 1][another_item] > cut_point:
						# printing here is not in sorted order - replaced by code above
						# print(
						# "\t\t", self.point_labels[an_item], self.point_labels[another_item],
						# self.similarities[an_item - 1][another_item])
						self.a_x_alike.append(self.point_coords.iloc[an_item][self.hor_dim])
						self.b_x_alike.append(self.point_coords.iloc[another_item][self.hor_dim])
						self.a_y_alike.append(self.point_coords.iloc[an_item][self.vert_dim])
						self.b_y_alike.append(self.point_coords.iloc[another_item][self.vert_dim])
				elif self.value_type == "dissimilarities":
					if self.similarities[an_item - 1][another_item] < cut_point:
						# print(
						# "\t\t", self.point_labels[an_item], self.point_labels[another_item],
						# self.similarities[an_item - 1][another_item])
						self.a_x_alike.append(self.point_coords.iloc[an_item][self.hor_dim])
						self.b_x_alike.append(self.point_coords.iloc[another_item][self.hor_dim])
						self.a_y_alike.append(self.point_coords.iloc[an_item][self.vert_dim])
						self.b_y_alike.append(self.point_coords.iloc[another_item][self.vert_dim])

	# --------------------------------------------------------------------------------------------------

	def print_segments(self, width, decimals):

		#
		print("\tSegment and size")
		#
		print(
			f"Percent likely left:  {self.like_pcts[1]:{width}.{decimals}f}" +
			f"\tPercent likely right: {self.like_pcts[2]:{width}.{decimals}f}")
		print(
			f"Percent base left:    {self.base_pcts[1]:{width}.{decimals}f}" +
			f"\tPercent base right:   {self.base_pcts[3]:{width}.{decimals}f}" +
			f"\tPercent base neither:   {self.base_pcts[2]:{width}.{decimals}f}")
		print(
			f"Percent core left:    {self.core_pcts[1]:{width}.{decimals}f}" +
			f"\tPercent core right:   {self.core_pcts[3]:{width}.{decimals}f}" +
			f"\tPercent core neither:   {self.core_pcts[2]:{width}.{decimals}f}")
		print(
			f"Percent left only:    {self.dim1_pcts[1]:{width}.{decimals}f}" +
			f"\tPercent right only:   {self.dim1_pcts[2]:{width}.{decimals}f}")
		print(
			f"Percent up only:      {self.dim2_pcts[2]:{width}.{decimals}f}" +
			f"\tPercent down only:    {self.dim2_pcts[1]:{width}.{decimals}f}")
		print(
			f"Percent battleground:    {self.battleground_pcts[1]:{width}.{decimals}f}" +
			f"\tPercent settled:      {self.battleground_pcts[2]:{width}.{decimals}f}")
		print(
			f"Percent convertible to left:    {self.conv_pcts[1]:{width}.{decimals}f}" +
			f"\tPercent convertible to right:   {self.conv_pcts[2]:{width}.{decimals}f}" +
			f"\tPercent settled:    {self.conv_pcts[3]:{width}.{decimals}f}")
		# #
		# print("\tSegment and size")
		# #
		# print(
		# 	f"\t\tPercent likely left:  {self.pct_likely_left:{width}.{decimals}f}" +
		# 	f"\tPercent likely right: {self.pct_likely_right:{width}.{decimals}f}")
		# print(
		# 	f"\t\tPercent base left:    {self.pct_base_left:{width}.{decimals}f}" +
		# 	f"\tPercent base right:   {self.pct_base_right:{width}.{decimals}f}" +
		# 	f"\tPercent base neither:   {self.pct_base_neither:{width}.{decimals}f}")
		# print(
		# 	f"\t\tPercent core left:    {self.pct_core_left:{width}.{decimals}f}" +
		# 	f"\tPercent core right:   {self.pct_core_right:{width}.{decimals}f}" +
		# 	f"\tPercent core neither:   {self.pct_core_neither:{width}.{decimals}f}")
		# print(
		# 	f"\t\tPercent left only:    {self.pct_left_only:{width}.{decimals}f}" +
		# 	f"\tPercent right only:   {self.pct_right_only:{width}.{decimals}f}")
		# print(
		# 	f"\t\tPercent up only:      {self.pct_up_only:{width}.{decimals}f}" +
		# 	f"\tPercent down only:    {self.pct_down_only:{width}.{decimals}f}")
		# print(
		# 	f"\t\tPercent battleground:    {self.pct_battleground:{width}.{decimals}f}" +
		# 	f"\tPercent settled:      {self.pct_settled:{width}.{decimals}f}")
		# print(
		# 	f"\t\tPercent convertible to left:    {self.pct_convertible_to_left:{width}.{decimals}f}" +
		# 	f"\tPercent convertible to right:   {self.pct_convertible_to_right:{width}.{decimals}f}" +
		# 	f"\tPercent settled:    {self.pct_convertible_to_neither:{width}.{decimals}f}")

		return

	# --------------------------------------------------------------------------------------

	def rank(self):
		#
		# Create dataframe which is used for computing and displaying ranks
		# It will start with self.zipped which has similarities sorted ascending and
		# labels of the items in each pair
		#

		self.df = pd.DataFrame(self.zipped, columns=['Similarity', 'A', 'B'])
		#
		# Rank the similarities
		#
		self.df['Similarity_Rank'] = self.df['Similarity'].rank(method='average')
		#
		# Add and rank the distances
		#
		temp_list = []
		for each_dyad in self.range_similarities:
			temp_list.append(str(self.df.iloc[each_dyad]["A"] + "_" + self.df.iloc[each_dyad]["B"]))
		self.df["Dyad"] = temp_list
		self.df["Distance_AB"] = 0.0
		#
		for each_dyad in self.range_similarities:
			self.df.at[each_dyad, "Distance_AB"] = self.distances_as_dict[temp_list[each_dyad]]
		self.df["Distance_Rank"] = self.df['Distance_AB'].rank(method='average')
		#
		# Compute the difference in ranks between the similarities and the distances
		#
		self.df["AB_Rank_Difference"] = self.df["Similarity_Rank"] - self.df["Distance_Rank"]
		self.selection = self.df[["A", "B", "Similarity_Rank", "Distance_Rank", "AB_Rank_Difference"]]
		print(self.selection)
		#
		# Plot rank of similarity vs rank of distance as scatter plot
		#   ????????????????????????????????????????????????????????????????????
		fig = self.plot_ranks()
		#
		# Create filter for each item  indicating when an item is part of a dyad
		# and the other item they are paired with. Item can be either  the A or B item.
		#
		for each_item in self.range_points:
			self.df[self.point_labels[each_item]] = "INAP"
		for each_pair in self.range_similarities:
			for each_item in self.range_points:
				if self.df.iloc[each_pair]["A"] == self.point_labels[each_item] \
						or self.df.iloc[each_pair]["B"] == self.point_labels[each_item]:
					if self.df.iloc[each_pair]["A"] == self.point_labels[each_item]:
						self.df.at[each_pair, self.point_labels[each_item]] = self.df.iloc[each_pair]["B"]
					else:
						self.df.at[each_pair, self.point_labels[each_item]] = self.df.iloc[each_pair]["A"]
				else:
					self.df.at[each_pair, self.point_labels[each_item]] = "INAP"

		return fig

	# ----------------------------------------------------------------------------------------

	def read_configuration_function(self, file_name):
		""" read_configuration function - is used by commands needing to read a configuration
			from a file.
		"""
		# Initialize variables needed to read in new configuration and discard any
		# existing configuration
		#
		# 	Read first line defining file type, must be "configuration"
		#
		problem_reading_file = False
		#
		try:
			with open(file_name, 'rt') as file_handle:
				header = file_handle.readline()
				if len(header) == 0:
					self.error("Empty Header.",
							file_name)
					problem_reading_file = True
					return problem_reading_file
				#
				# 	Reject non-configuration files
				#
				if not (header.lower().strip() == "configuration"):
					self.error("File is not a configuration file:",
							file_name)
					problem_reading_file = True
					return problem_reading_file
				#
				# 	Line 2: Read line defining number of dimensions, number of points
				#
				dim = file_handle.readline()
				dim_list = dim.strip().split()
				expected_dim = int(dim_list[0])
				expected_points = int(dim_list[1])
				self.ndim = expected_dim
				self.npoint = expected_points
				self.range_dims = range(self.ndim)
				self.range_points = range(self.npoint)
				#
				#   Read in DIMENSION labels / names
				#
				for i in range(expected_dim):
					(dim_label, dim_name) = file_handle.readline().rstrip('\n').split(';')
					self.dim_labels.append(dim_label)
					self.dim_names.append(dim_name)
				#
				#   Read in POINT labels / names
				#
				for i in range(expected_points):
					(point_label, point_name) = file_handle.readline().split(';')
					self.point_labels.append(point_label)
					self.point_names.append(point_name)
					self.point_names[i] = self.point_names[i].strip()
				#
				#   Read in POINTS
				#
				self.point_coords = pd.DataFrame(
					[
						[
							float(p) for p in
							file_handle.readline().split()
						]
						for i in range(expected_points)
					],
					index=self.point_names,
					columns=self.dim_labels
				)
		except FileNotFoundError:
			problem_reading_file = True
			return problem_reading_file

		#
		return problem_reading_file

	# ----------------------------------------------------------------------------------------

	def read_grouped_data(self, file_name):
		""" read_groups - is used by group command needing to read a group configuration
			from a file.
		"""
		# Initialize variables needed to read in new configuration and discard any
		# existing configuration
		#
		# 	Read first line defining file type, must be "configuration"
		#
		problem_reading_file = False
		#
		try:
			with open(file_name, 'rt') as file_handle:
				header = file_handle.readline()
				if len(header) == 0:
					self.error("Empty Header in file ",
							file_name)
					problem_reading_file = True
					return problem_reading_file
				#
				# 	Reject non-configuration files
				#
				if not (header.lower().strip() == "grouped"):
					self.error("File is not a grouped file:",
							file_name)
					problem_reading_file = True
					return problem_reading_file
				#
				# 	Line 2: Read line defining grouping variable
				#
				grouping_var = file_handle.readline()
				if len(grouping_var) == 0:
					self.error("Line for grouping variable name is empty in file ",
							file_name)
					problem_reading_file = True
					return problem_reading_file

				self.grouping_var = grouping_var.strip("\n")
				#
				# Echo to the user
				#
				print(f"\n\tGroups were defined using {self.grouping_var} variable.\n")
				#
				# 	Line 3: Read line defining number of dimensions, number of points
				#
				dim = file_handle.readline()
				dim_list = dim.strip().split()
				expected_dim = int(dim_list[0])
				expected_points = int(dim_list[1])
				self.ndim_grpd = expected_dim
				self.npoint_grpd = expected_points
				if not (self.ndim == self.ndim_grpd):
					print(
						"\n\tNumber of dimensions in Grouped file is" +
						" different than the number of dimensions in active " +
						"configuration.")
					problem_reading_file = True
					file_handle.close()
					return problem_reading_file
				#
				self.range_dims_grpd = range(self.ndim_grpd)
				self.range_points_grpd = range(self.npoint_grpd)
				#
				#   Read in DIMENSION labels / names
				#
				for i in range(expected_dim):
					(dim_label_grpd, dim_name_grpd) = file_handle.readline().split(';')
					self.dim_labels_grpd.append(dim_label_grpd)
					dim_name_grpd.strip("\n")
					self.dim_names_grpd.append(dim_name_grpd)
				#
				# Check to see they are the same as in the active configuration
				#
					if not self.dim_labels_grpd[i] == self.dim_labels[i]:
						print(
							"\n\tDimension label in grouped file is not the " +
							"same as in active configuration")
						problem_reading_file = True
						file_handle.close()
						return problem_reading_file
				#
				#   Read in POINT labels / codes / names
				#
				for i in range(expected_points):
					(point_label_grpd, point_code_grpd, point_name_grpd) = file_handle.readline().split(';')
					self.point_labels_grpd.append(point_label_grpd)
					self.point_names_grpd.append(point_name_grpd)
					self.point_names_grpd[i] = self.point_names_grpd[i].strip()
				#
				#   Read in POINTS
				#
				self.point_coords_grpd = pd.DataFrame(
					[
						[
							float(p) for p in
							file_handle.readline().split()
						]
						for i in range(expected_points)
					],
					index=self.point_names_grpd,
					columns=self.dim_labels_grpd
				)
		except FileNotFoundError:
			problem_reading_file = True
			return problem_reading_file
		#
		problem_reading_file = False
		#
		return problem_reading_file

# ------------------------------------------------------------------------------------

	def read_lower_triangular(self, file_name):
		""" read lower triangular function - this function is used to read information stored
			in lower triangular form such as similarities, distances and correlations.
		"""
		# print(f"DEBUG -- at top of read_lower_triangular")
		#
		# Note:
		# Within read_lower_triangular the elements of the matrix are considered values
		# regardless of whether the calling routine called them correlations,
		# similarities or even dissimilarities. Analogously within this function
		# it refers to labels and names rather than item_labels and item_names.
		# And lastly it uses nelements??????? rather than self.range_points.  self.range_points refers specifically
		# to the active configuration.
		#
		# Define variables needed to read in lower triangle
		#
		self.nreferent = 0
		self.docs = []
		self.item_labels.clear()
		self.item_names.clear()
		self.values = []
		line = ""
		#
		# Read first line defining file type and check file type - must be "lower triangular"
		#
		try:
			with open(file_name, 'rt') as file_handle:
				line = file_handle.readline()
				flower = line.lower()
				file_type = flower.strip()
				if len(file_type) == 0:
					self.error("Empty line.",
							"Review file name and contents")
					problem_reading_file = True
					return problem_reading_file, self.values
				#
				# Handle bad file type
				#
				if not (file_type.lower().strip() == "lower triangular"):
					self.error(
						"Problem reading first line of file.  ",
						"Should be lower triangular")
					problem_reading_file = True
					return problem_reading_file, self.values
				#
				# Read second line defining number of items/referents/stimuli
				#
				nite = file_handle.readline()
				if len(nite) == 0:
					self.error(
						"Problem reading second line of file. ",
						"Should contain the number of stimuli.")
					problem_reading_file = True
					return problem_reading_file, self.values
				self.nreferent = int(nite)
				#
				# Handle situation where there are a different number of stimuli/referents
				# than points in active configuration
				#
				# print(f"DEBUG -- about to check against active configuration")
				if self.have_active_configuration():
					if not (self.npoint == self.nreferent):
						self.error(
							"The number of values differs from the number of" +
							" points in active configuration.",
							"Use the Configuration command to read in " +
							"a configuration with the matching number of items or use the " +
							"Deactivate command to abandon the currently" +
							" active configuration."
						)
						problem_reading_file = True
						return problem_reading_file, self.values
				# print(f"DEBUG -- after check against active configuration")
				#
				# Read as many lines as stimuli.
				# Each contains label and name of stimuli separated by a semicolon
				#
				self.range_items = range(self.nreferent)
				if self.npoint == 0:
					self.range_points = range(self.nreferent)
				#
				# print(f"DEBUG - {self.range_items = } {self.range_points = }")
				for each_item in self.range_items:
					try:
						it = file_handle.readline()
						if len(it) == 0:
							self.error(
								"Problem reading labels and names of stimuli.",
								"Review file name and contents")
							problem_reading_file = True
							return problem_reading_file, self.values
						it = it.rstrip()
						self.docs.append(it)
						its = self.docs[each_item].split(";")
						self.item_labels.append(its[0])
						self.item_names.append(its[1])
					except EOFError:
						self.error("Unexpected End of File.",
								"Review file name and contents")
						problem_reading_file = True
						return problem_reading_file, self.values
					except IOError:
						self.error("Problem reading file.",
								"Review file name and contents")
						problem_reading_file = True
						return problem_reading_file, self.values
					except ValueError:
						self.error("Unexpected input.",
								"Review file name and contents")
						problem_reading_file = True
						return problem_reading_file, self.values
				#
				# Create column headings from labels
				#
				labs = []
				#
				# for each_item in nreferent:
				for each_item in self.range_items:
					labs.append(self.item_labels[each_item])
				#
				# Read npoint-1 lines with values of items, each containing one more value than the previous line
				#
				one_less = self.nreferent - 1
				self.nitems = range(one_less)
				# print(f"DEBUG -- about to read values of points")
				for each_item in self.nitems:
					try:
						aitem = file_handle.readline()
						aitem = aitem.rstrip()
						if len(aitem) == 0:
							self.error("Problem reading values in lower triangular matrix.",
									"Review file name and contents")
							problem_reading_file = True
							return problem_reading_file, self.values
					except EOFError:
						self.error("Unexpected End of File.",
								"Review file name and contents")
						problem_reading_file = True
						return problem_reading_file, self.values
					except IOError:
						self.error("Problem reading file.",
								"Review file name and contents")
						problem_reading_file = True
						return problem_reading_file, self.values
					except ValueError:
						self.error("Unexpected input.",
								"Review file name and contents")
						problem_reading_file = True
						return problem_reading_file, self.values

					item = aitem.split()
					self.values.append(item)
					#
					range_ites = range(each_item + 1)
					# print("fDEBUG -- about to read line of correlations")
					# print(f"DEBUG -- {range_ites = }")
					for ites in range_ites:
						try:
							self.values[each_item][ites] = float(self.values[each_item][ites])
						#
						except EOFError:
							self.error("Unexpected End of File.",
									"Review file name and contents")
							problem_reading_file = True
							return problem_reading_file, self.values
						except IOError:
							self.error("Problem reading file.",
								"Review file name and contents")
							problem_reading_file = True
							return problem_reading_file, self.values
						except ValueError:
							self.error("Unexpected input.",
									"Review file name and contents")
							problem_reading_file = True
							return problem_reading_file, self.values
		except EOFError:
			self.error("Unexpected End of File.",
					"Review file name and contents")
			problem_reading_file = True
			return problem_reading_file, self.values
		except IOError:
			self.error("Problem reading file.",
					"Review file name and contents")
			problem_reading_file = True
			return problem_reading_file, self.values
		except ValueError:
			self.error("Unexpected input.",
					"Review file name and contents")
			problem_reading_file = True
			return problem_reading_file, self.values

		#
		problem_reading_file = False
		#
		# Return characteristics of the file read as well as the contents
		#
		return problem_reading_file, self.values
	# --------------------------------------------------------------------------------------

	def rescale(self, which_dim, value):

		#
		# Multiply all point coordinates by user supplied value
		#
		for each_point in self.range_points:
			self.point_coords.iloc[each_point][which_dim] \
				= self.point_coords.iloc[each_point][which_dim] * value

	# ---------------------------------------------------------------------------

	def rotate(self, radians):
		# for each_point in self.point_coords:
		for each_point in self.range_points:
			new_x = (
				(math.cos(radians) * self.point_coords.iloc[each_point][self.hor_dim])
				- (math.sin(radians) * self.point_coords.iloc[each_point][self.vert_dim])
				)
			new_y = (
				(math.sin(radians) * self.point_coords.iloc[each_point][self.hor_dim])
				+ (math.cos(radians) * self.point_coords.iloc[each_point][self.vert_dim])
				)
			self.point_coords.iloc[each_point][self.hor_dim] = new_x
			self.point_coords.iloc[each_point][self.vert_dim] = new_y

	# ----------------------------------------------------------------------------------------

	def scree(self):
		self.dim_names.append("Dimension 1")
		self.dim_labels.append("Dim1")
		#
		range_ncomps = range(1, 11)
		for each_n_comp in range_ncomps:
			nmds = manifold.MDS(n_components=each_n_comp, metric=self.use_metric,
				dissimilarity='precomputed', n_init=20, verbose=0, normalized_stress="auto")
			npos = nmds.fit_transform(X=self.similarities_as_square)
			self.point_coords = pd.DataFrame(npos.tolist())
			# self.point_coords = npos <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			self.dim_names.append("Dimension "+str(each_n_comp))
			self.dim_labels.append("Di"+str(each_n_comp))

			# print and plot result in each dimensionality
			# print_active_function(conf)
			# show_plot_function(conf, flags, parts, refs)
			self.best_stress = nmds.stress_
			print("\tBest stress in ", each_n_comp, "dimensions: ", self.best_stress)
			# print("\n")
			#
			self.min_stress.append(nmds.stress_)

	# ---------------------------------------------------------------------------------

	def segmenter(self, in_group, out_group, in_count, all_count):

		in_group.append(1)
		out_group.append(0)
		in_count += 1
		all_count += 1
		#

		return in_group, out_group, in_count, all_count

	# ---------------------------------------------------------------------------------

	def set_bisector_case(self):

		#
		# Bisector Case 0a Bisector slope is zero from Left side to Right side
		#
		if self.bisector.direction == "Flat":
			self.bisector.case = "0a"
		#
		# Bisector 1 Case 0b Connector slope is zero - from top to bottom
		#
		if self.connector.direction == "Flat":
			self.bisector.case = "0b"
		#
		# Bisector Case Ia Positive slope from Left side to Right side and min_y > vert_min
		#
		if self.bisector.direction == "Upward slope":
			#
			if self.bisector.goes_through_left_side == "Yes" \
				and self.bisector.goes_through_right_side == "Yes":
				self.bisector.case = "Ia"
			#
			# Bisector Case IIa Positive slope from Left side to Top and bisector_max_y == vert_max
			#
			elif self.bisector.goes_through_left_side == "Yes" \
				and self.bisector.goes_through_top == "Yes":
				self.bisector.case = "IIa"
			#
			# Bisector Case IIIa Positive slope from Bottom to Right side
			#
			elif self.bisector.goes_through_bottom == "Yes" \
				and self.bisector.goes_through_right_side == "Yes":
				self.bisector.case = "IIIa"
			#
			# Bisector Case IVa Positive slope from Bottom to Top and min_x < hor_min
			#
			elif self.bisector.goes_through_bottom == "Yes" \
				and self.bisector.goes_through_top == "Yes":
				self.bisector.case = "IVa"
		#
		# Bisector Case Ib Negative slope from Left side to Right side
		#
		if self.bisector.direction == "Downward slope":
			#
			if self.bisector.goes_through_left_side == "Yes" \
				and self.bisector.goes_through_right_side == "Yes":
				self.bisector.case = "Ib"
			#
			# Bisector Case IIb Negative slope from Left side to Bottom and min_y == vert_min
			#
			elif self.bisector.goes_through_left_side == "Yes" \
				and self.bisector.goes_through_bottom == "Yes":
				self.bisector.case = "IIb"
			#
			# Bisector Case IIIb Negative slope from Top to Right side and min_x < hor_min ?????????
			#
			elif self.bisector.goes_through_top == "Yes" \
				and self.bisector.goes_through_right_side == "Yes":
				self.bisector.case = "IIIb"
			#
			# Bisector Case IVb Negative slope from Bottom to Top and max_y > vert_max  ????????????????????????
			#
			elif self.bisector.goes_through_bottom == "Yes" \
				and self.bisector.goes_through_top == "Yes":
				self.bisector.case = "IVb"

	# ---------------------------------------------------------------------------------

	def set_direction_flags(self):
		""" set direction flags function - determines the direction of the slope of the
			connector and the bisector
		Assumes that the calling function has checked to see the reference points
		have been established
		"""
		#
		# Reset connector indicator
		#
		if self.point_coords.iloc[self.rival_a][self.hor_dim] \
			== self.point_coords.iloc[self.rival_b][self.hor_dim]:
			self.connector.direction = "Vertical"
		if self.point_coords.iloc[self.rival_a][self.vert_dim] \
			== self.point_coords.iloc[self.rival_b][self.vert_dim]:
			self.connector.direction = "Flat"
		#
		if \
			(
				(self.point_coords.iloc[self.rival_a][self.hor_dim]
				< self.point_coords.iloc[self.rival_b][self.hor_dim])
				and
				(self.point_coords.iloc[self.rival_a][self.vert_dim]
				> self.point_coords.iloc[self.rival_b][self.vert_dim])
				or
				(self.point_coords.iloc[self.rival_a][self.hor_dim]
				> self.point_coords.iloc[self.rival_b][self.hor_dim])
				and
				(self.point_coords.iloc[self.rival_a][self.vert_dim]
				< self.point_coords.iloc[self.rival_b][self.vert_dim])
			):
			self.connector.direction = "Downward slope"
		#
		if \
			(
				(self.point_coords.iloc[self.rival_a][self.hor_dim]
				< self.point_coords.iloc[self.rival_b][self.hor_dim])
				and
				(self.point_coords.iloc[self.rival_a][self.vert_dim]
				< self.point_coords.iloc[self.rival_b][self.vert_dim])
				or
				(self.point_coords.iloc[self.rival_a][self.hor_dim]
				> self.point_coords.iloc[self.rival_b][self.hor_dim])
				and
				(self.point_coords.iloc[self.rival_a][self.vert_dim]
				> self.point_coords.iloc[self.rival_b][self.vert_dim])
			):
			self.connector.direction = "Upward slope"
		#
		# Reset bisector direction
		#
		if self.point_coords.iloc[self.rival_a][self.vert_dim] \
			== self.point_coords.iloc[self.rival_b][self.vert_dim]:
			self.bisector.direction = "Vertical"
			self.west.direction = "Vertical"
			self.east.direction = "Vertical"
		if self.point_coords.iloc[self.rival_a][self.hor_dim] \
			== self.point_coords.iloc[self.rival_b][self.hor_dim]:
			self.bisector.direction = "Flat"
			self.west.direction = "Flat"
			self.east.direction = "Flat"
		if self.connector.direction == "Downward slope":
			self.bisector.direction = "Upward slope"
			self.west.direction = "Upward slope"
			self.east.direction = "Upward slope"
		if self.connector.direction == "Upward slope":
			self.bisector.direction = "Downward slope"
			self.west.direction = "Downward slope"
			self.east.direction = "Downward slope"
		#
		return

	# ---------------------------------------------------------------------------------------

	def set_line_case(self):
		""" set line case function - determines the case of the bisector and edges
		Assumes that the calling function has checked to see the reference points
		have been established
		"""
		#
		#   self.bisector.case = "Unknown"   ???????????????????????????????
		#
		# Initialize variables needed
		#
		x_coords = []
		y_coords = []
		x = []
		y = []
		self.west.goes_through_top = "No"
		self.west.goes_through_bottom = "No"
		self.west.goes_through_left_side = "No"
		self.west.goes_through_right_side = "No"
		self.east.goes_through_top = "No"
		self.east.goes_through_bottom = "No"
		self.east.goes_through_left_side = "No"
		self.east.goes_through_right_side = "No"
		self.west.start_x = 0.0
		self.west.start_y = 0.0
		self.west.end_x = 0.0
		self.west.end_y = 0.0
		self.east.start_x = 0.0
		self.east.start_y = 0.0
		self.east.end_x = 0.0
		self.east.end_y = 0.0

		#
		# Determine distance and each dimension's contribution to distance
		#
		differ_x = self.point_coords.iloc[self.rival_a][0] - self.point_coords.iloc[self.rival_b][0]
		differ_y = self.point_coords.iloc[self.rival_a][1] - self.point_coords.iloc[self.rival_b][1]
		sq_diff_x = differ_x * differ_x
		sq_diff_y = differ_y * differ_y
		x_dist = math.sqrt(sq_diff_x)
		y_dist = math.sqrt(sq_diff_y)
		#
		# Determine width of battleground based on portion of distance between reference point
		# tolerance is set in main  - default is.1
		#
		half_battleground_x = (self.tolerance * x_dist)
		half_battleground_y = (self.tolerance * y_dist)
		#
		# Set west.slope, direction, intercept, case
		# Switching  from bisector to self.west
		#
		self.west.slope = self.bisector.slope
		#
		if self.bisector.direction == "Upward slope":
			self.west_connector_cross_x = self.connector_bisector_cross_x - half_battleground_x
			self.west_connector_cross_y = self.connector_bisector_cross_y + half_battleground_y
			self.west.intercept = \
				self.west_connector_cross_y \
				- (self.west.slope * self.west_connector_cross_x)
		elif self.bisector.direction == "Downward slope":
			# self.west_connector_cross_x = self.connector_bisector_cross_x + half_battleground_x
			# self.west_connector_cross_y = self.connector_bisector_cross_y + half_battleground_y
			self.west_connector_cross_x = self.connector_bisector_cross_x - half_battleground_x
			self.west_connector_cross_y = self.connector_bisector_cross_y - half_battleground_y
			self.west.intercept = self.west_connector_cross_y \
				- (self.west.slope * self.west_connector_cross_x)
		elif self.bisector.direction == "Flat":
			self.west_connector_cross_x = self.connector_bisector_cross_x
			self.west_connector_cross_y = self.connector_bisector_cross_y + half_battleground_y
			self.west.intercept = self.west_connector_cross_y
		elif self.bisector.direction == "Vertical":
			self.west_connector_cross_x = self.connector_bisector_cross_x + half_battleground_x
			self.west_connector_cross_y = self.connector_bisector_cross_y + half_battleground_y
			self.west.intercept = self.west_connector_cross_y \
				- (self.west.slope * self.west_connector_cross_x)
		#
		# self.west passes through top or bottom
		#   y at max x
		self.west.y_at_max_x = (self.west.slope * self.hor_max) \
			+ self.west.intercept

		# 	y at min x
		self.west.y_at_min_x = (self.west.slope * self.hor_min) \
			+ self.west.intercept

		self.west.x_at_max_y = ((
			self.vert_max
			- self.west.intercept)
			/ self.west.slope)

		# 	x at min y
		self.west.x_at_min_y = ((
			self.vert_min
			- self.west.intercept)
			/ self.west.slope)
		# -----------------------------------------------------------
		#

		if self.bisector.direction == "Flat":
			#
			if self.west.y_at_max_x > self.vert_max and self.west.y_at_max_x > self.vert_min:
				self.west.y_at_max_x = self.vert_max
			if self.west.y_at_max_x < self.vert_min and self.west.y_at_max_x < self.vert_max:
				self.west.y_at_max_x = self.vert_min
			if self.west.y_at_min_x > self.vert_max and self.west.y_at_min_x > self.vert_min:
				self.west.y_at_min_x = self.vert_max
			if self.west.y_at_min_x < self.vert_min and self.west.y_at_min_x < self.vert_max:
				self.west.y_at_min_x = self.vert_min
			if self.west.x_at_max_y > self.hor_max and self.west.x_at_max_y > self.hor_min:
				self.west.x_at_max_y = self.hor_max
			if self.west.x_at_max_y < self.hor_min and self.west.x_at_max_y < self.hor_max:
				self.west.x_at_max_y = self.hor_min
			if self.west.x_at_min_y > self.hor_max and self.west.x_at_min_y > self.hor_min:
				self.west.x_at_min_y = self.hor_max
			if self.west.x_at_min_y < self.hor_min and self.west.x_at_min_y < self.hor_max:
				self.west.x_at_min_y = self.hor_min
		#
		if self.bisector.direction == "Flat":
			self.west.goes_through_right_side = "Yes"
			self.west.goes_through_left_side = "Yes"
			self.west.goes_through_top = "No"
			self.west.goes_through_bottom = "No"
		#
		if self.connector.direction == "Flat":
			self.west.goes_through_right_side = "No"
			self.west.goes_through_left_side = "No"
			self.west.goes_through_top = "Yes"
			self.west.goes_through_bottom = "Yes"
		#
		if self.bisector.direction != "Flat" and self.connector.direction != "Flat":
			if self.vert_min <= self.west.y_at_max_x <= self.vert_max:
				self.west.goes_through_right_side = "Yes"
			if self.vert_min <= self.west.y_at_min_x <= self.vert_max:
				self.west.goes_through_left_side = "Yes"
			if self.hor_min <= self.west.x_at_max_y <= self.hor_max:
				self.west.goes_through_top = "Yes"
			if self.hor_min <= self.west.x_at_min_y <= self.hor_max:
				self.west.goes_through_bottom = "Yes"
		#
		# Handle lines going through corners
		#
		# upper right
		if self.west.x_at_max_y == self.hor_max \
			and self.west.y_at_max_x == self.vert_max:
			self.west.goes_through_top, \
				self.west.goes_through_right_side = self.choose_a_side_function()
		# upper left
		if self.west.x_at_max_y == self.hor_min \
			and self.west.y_at_min_x == self.vert_max:
			(self.west.goes_through_top,
				self.west.goes_through_left_side) = self.choose_a_side_function()
		# lower right
		if self.west.x_at_min_y == self.hor_max \
			and self.west.y_at_max_x == self.vert_min:
			(self.west.goes_through_bottom,
				self.west.goes_through_right_side) = self.choose_a_side_function()
		# lower left
		if self.west.x_at_min_y == self.hor_min \
			and self.west.y_at_min_x == self.vert_min:
			(self.west.goes_through_bottom,
			self.west.goes_through_left_side) = self.choose_a_side_function()
		#
		# Set west.case
		#
		if self.west.direction == "Vertical":
			self.west.case = "0b"			# had been 0a
		elif self.west.direction == "Flat":
			self.west.case = "0a"			# had been 0b
		elif self.west.direction == "Upward slope" \
				and self.west.goes_through_left_side == "Yes" \
				and self.west.goes_through_right_side == "Yes":
			self.west.case = "Ia"
		elif self.west.direction == "Upward slope" \
				and self.west.goes_through_left_side == "Yes" \
				and self.west.goes_through_top == "Yes":
			self.west.case = "IIa"
		elif self.west.direction == "Upward slope" \
				and self.west.goes_through_right_side == "Yes" \
				and self.west.goes_through_bottom == "Yes":
			self.west.case = "IIIa"
		elif self.west.direction == "Upward slope" \
				and self.west.goes_through_top == "Yes" \
				and self.west.goes_through_bottom == "Yes":
			self.west.case = "IVa"
		elif self.west.direction == "Downward slope" \
				and self.west.goes_through_left_side == "Yes" \
				and self.west.goes_through_right_side == "Yes":
			self.west.case = "Ib"
		elif self.west.direction == "Downward slope" \
				and self.west.goes_through_left_side == "Yes" \
				and self.west.goes_through_bottom == "Yes":
			self.west.case = "IIb"
		elif self.west.direction == "Downward slope" \
				and self.west.goes_through_right_side == "Yes" \
				and self.west.goes_through_top == "Yes":
			self.west.case = "IIIb"
		elif self.west.direction == "Downward slope" \
				and self.west.goes_through_top == "Yes" \
				and self.west.goes_through_bottom == "Yes":
			self.west.case = "IVb"

		match self.west.case:
			#
			# west Case 0a  Bisector slope is zero - from Left side to Right side
			#  had been 0b
			case "0a":
				self.west.start_x = self.hor_min
				self.west.start_y = self.connector_bisector_cross_y - half_battleground_y
				self.west.end_x = self.hor_max
				self.west.end_y = self.connector_bisector_cross_y - half_battleground_y
			#
			# west Case 0b Connector slope is zero - self.west from top to bottom
			#  had been 0a
			case "0b":
				self.west.start_x = self.connector_bisector_cross_x - half_battleground_x
				self.west.start_y = self.vert_max
				self.west.end_x = self.connector_bisector_cross_x - half_battleground_x
				self.west.end_y = self.vert_min
			#
			# west Case Ia Positive slope from Left side to Right side
			#
			case "Ia":
				self.west.start_x = self.hor_min
				self.west.start_y = self.west.y_at_min_x
				self.west.end_x = self.hor_max
				self.west.end_y = self.west.y_at_max_x
			#
			# west Case IIa Positive slope from Left side to Top
			#
			case "IIa":
				self.west.start_x = self.hor_min
				self.west.start_y = self.west.y_at_min_x
				self.west.end_x = self.west.x_at_max_y
				self.west.end_y = self.vert_max
			#
			# self.west Case IIIa Positive slope from Bottom to Right side
			#
			case "IIIa":
				self.west.start_x = self.west.x_at_min_y
				self.west.start_y = self.vert_min
				self.west.end_x = self.hor_max
				self.west.end_y = self.west.y_at_max_x
			#
			# west Case IVa Positive slope from Bottom to Top
			#
			case "IVa":
				self.west.start_x = self.west.x_at_min_y
				self.west.start_y = self.vert_min
				self.west.end_x = self.west.x_at_max_y
				self.west.end_y = self.vert_max
			#
			# self.west Case Ib Negative slope from Left side to Right side
			#
			case "Ib":
				self.west.start_x = self.hor_min
				self.west.start_y = self.west.y_at_min_x
				self.west.end_x = self.hor_max
				self.west.end_y = self.west.y_at_max_x
			#
			# self.west Case IIb Negative slope from Left side to Bottom
			#
			case "IIb":
				self.west.start_x = self.hor_min
				self.west.start_y = self.west.y_at_min_x
				self.west.end_x = self.west.x_at_min_y
				self.west.end_y = self.vert_min
			#
			# self.west Case IIIb Negative slope from Top to Right side
			#
			case "IIIb":
				self.west.start_x = self.west.x_at_max_y
				self.west.start_y = self.vert_max
				self.west.end_x = self.hor_max
				self.west.end_y = self.west.y_at_max_x
			#
			# self.west Case IVb Negative slope from Bottom to TopTop to Bottom
			case "IVb":
				self.west.start_x = self.west.x_at_max_y
				self.west.start_y = self.vert_max
				self.west.end_x = self.west.x_at_min_y
				self.west.end_y = self.vert_min
		#
		# Begin east
		#
		self.east.slope = self.bisector.slope
		#
		if self.bisector.direction == "Upward slope":
			self.east_connector_cross_x = self.connector_bisector_cross_x + half_battleground_x
			self.east_connector_cross_y = self.connector_bisector_cross_y - half_battleground_y
			self.east.intercept = \
				self.east_connector_cross_y \
				- (self.east.slope * self.east_connector_cross_x)
		elif self.bisector.direction == "Downward slope":
			# self.east_connector_cross_x = self.connector_bisector_cross_x - half_battleground_x
			# self.east_connector_cross_y = self.connector_bisector_cross_y - half_battleground_y
			self.east_connector_cross_x = self.connector_bisector_cross_x + half_battleground_x
			self.east_connector_cross_y = self.connector_bisector_cross_y + half_battleground_y
			self.east.intercept = \
				self.east_connector_cross_y \
				- (self.east.slope * self.east_connector_cross_x)
		elif self.bisector.direction == "Flat":
			self.east_connector_cross_x = self.connector_bisector_cross_x
			self.east_connector_cross_y = self.connector_bisector_cross_y - half_battleground_y
			self.east.intercept = self.east_connector_cross_y
		elif self.bisector.direction == "Vertical":
			self.east_connector_cross_x = self.connector_bisector_cross_x - half_battleground_x
			self.east_connector_cross_y = self.connector_bisector_cross_y
			self.east.intercept = \
				self.east_connector_cross_y \
				- (self.east.slope * self.east_connector_cross_x)
		#
		# self.east passes through top or bottom
		#   y at max x
		self.east.y_at_max_x = (
			self.east.slope
			* self.hor_max) \
			+ self.east.intercept
		#   y at min x
		self.east.y_at_min_x = (
			self.east.slope
			* self.hor_min) \
			+ self.east.intercept
		#
		# self.east passes through right or left side
		#    x at max y
		self.east.x_at_max_y = ((
			self.vert_max
			- self.east.intercept)
			/ self.east.slope)
		# + self.east.intercept)

		# 	x at min y
		self.east.x_at_min_y = ((
			self.vert_min
			- self.east.intercept)
			/ self.east.slope)
		# + self.east.intercept)

		#
		if self.bisector.direction == "Flat":
			#
			if self.east.y_at_max_x > self.vert_max and self.east.y_at_max_x > self.vert_min:
				self.east.y_at_max_x = self.vert_max
			if self.east.y_at_max_x < self.vert_min and self.east.y_at_max_x < self.vert_max:
				self.east.y_at_max_x = self.vert_min
			if self.east.y_at_min_x > self.vert_max and self.east.y_at_min_x > self.vert_min:
				self.east.y_at_min_x = self.vert_max
			if self.east.y_at_min_x < self.vert_min and self.east.y_at_min_x < self.vert_max:
				self.east.y_at_min_x = self.vert_min
			if self.east.x_at_max_y > self.hor_max and self.east.x_at_max_y > self.hor_min:
				self.east.x_at_max_y = self.hor_max
			if self.east.x_at_max_y < self.hor_min and self.east.x_at_max_y < self.hor_max:
				self.east.x_at_max_y = self.hor_min
			if self.east.x_at_min_y > self.hor_max and self.east.x_at_min_y > self.hor_min:
				self.east.x_at_min_y = self.hor_max
			if self.east.x_at_min_y < self.hor_min and self.east.x_at_min_y < self.hor_max:
				self.east.x_at_min_y = self.hor_min
		#
		if self.connector.direction == "Vertical":
			self.east.goes_through_right_side = "Yes"
			self.east.goes_through_left_side = "Yes"
			self.east.goes_through_top = "No"
			self.east.goes_through_bottom = "No"
		if self.connector.direction == "Flat":
			self.east.goes_through_right_side = "No"
			self.east.goes_through_left_side = "No"
			self.east.goes_through_top = "Yes"
			self.east.goes_through_bottom = "Yes"

		if self.bisector.direction != "Flat" and self.connector.direction != "Flat":
			if self.vert_min <= self.east.y_at_max_x <= self.vert_max:
				self.east.goes_through_right_side = "Yes"
			if self.vert_min <= self.east.y_at_min_x <= self.vert_max:
				self.east.goes_through_left_side = "Yes"
			if self.hor_min <= self.east.x_at_max_y <= self.hor_max:
				self.east.goes_through_top = "Yes"
			if self.hor_min <= self.east.x_at_min_y <= self.hor_max:
				self.east.goes_through_bottom = "Yes"
		#
		# Handle lines going through corners
		#
		# upper right
		if self.east.x_at_max_y == self.hor_max \
			and self.east.y_at_max_x == self.vert_max:
			(self.east.goes_through_top,
			self.east.goes_through_right_side) = self.choose_a_side_function()
		# upper left
		if self.east.x_at_max_y == self.hor_min \
			and self.east.y_at_min_x == self.vert_max:
			(self.east.goes_through_top,
			self.east.goes_through_left_side) = self.choose_a_side_function()
		# lower right
		if self.east.x_at_min_y == self.hor_max \
			and self.east.y_at_max_x == self.vert_min:
			(self.east.goes_through_bottom,
			self.east.goes_through_right_side) = self.choose_a_side_function()
		# lower left
		if self.east.x_at_min_y == self.hor_min \
			and self.east.y_at_min_x == self.vert_min:
			(self.east.goes_through_bottom,
			self.east.goes_through_left_side) = self.choose_a_side_function()
		#
		# Set east_case
		#
		if self.east.direction == "Vertical":
			self.east.case = "0b"			# had been 0a
		elif self.east.direction == "Flat":
			self.east.case = "0a"			# had been 0b
		elif self.east.direction == "Upward slope" \
			and self.east.goes_through_left_side == "Yes" \
			and self.east.goes_through_right_side == "Yes":
			self.east.case = "Ia"
		elif self.east.direction == "Upward slope" \
			and self.east.goes_through_left_side == "Yes" \
			and self.east.goes_through_top == "Yes":
			self.east.case = "IIa"
		elif self.east.direction == "Upward slope" \
			and self.east.goes_through_right_side == "Yes" \
			and self.east.goes_through_bottom == "Yes":
			self.east.case = "IIIa"
		elif self.east.direction == "Upward slope" \
			and self.east.goes_through_top == "Yes" \
			and self.east.goes_through_bottom == "Yes":
			self.east.case = "IVa"
		elif self.east.direction == "Downward slope" \
			and self.east.goes_through_left_side == "Yes" \
			and self.east.goes_through_right_side == "Yes":
			self.east.case = "Ib"
		elif self.east.direction == "Downward slope" \
			and self.east.goes_through_left_side == "Yes" \
			and self.east.goes_through_bottom == "Yes":
			self.east.case = "IIb"
		elif self.east.direction == "Downward slope" \
			and self.east.goes_through_right_side == "Yes" \
			and self.east.goes_through_top == "Yes":
			self.east.case = "IIIb"
		elif self.east.direction == "Downward slope" \
			and self.east.goes_through_top == "Yes" \
			and self.east.goes_through_bottom == "Yes":
			self.east.case = "IVb"
		#
		match self.east.case:
			#
			# east Case 0a Bisector slope is zero - Left side to Right sid
			#   had been 0b
			case "0a":
				self.east.start_x = self.hor_min
				self.east.start_y = self.connector_bisector_cross_y + half_battleground_y
				self.east.end_x = self.hor_max
				self.east.end_y = self.connector_bisector_cross_y + half_battleground_y
			#
			# east Case 0b eConnector slope is zero - top to bottom
			#   had been 0a
			case "0b":
				self.east.start_x = self.connector_bisector_cross_x + half_battleground_x
				self.east.start_y = self.vert_max
				self.east.end_x = self.connector_bisector_cross_x + half_battleground_x
				self.east.end_y = self.vert_min
			#
			# east Case Ia Positive slope from Left side to Right side
			#
			case "Ia":
				self.east.start_x = self.hor_min
				self.east.start_y = self.east.y_at_min_x
				self.east.end_x = self.hor_max
				self.east.end_y = self.east.y_at_max_x
			#
			# east Case IIa Positive slope from Left side to Top and max_y == vert_max
			#
			case "IIa":
				self.east.start_x = self.hor_min
				self.east.start_y = self.east.y_at_min_x
				self.east.end_x = self.east.x_at_max_y
				self.east.end_y = self.vert_max
			#
			# self.east Case IIIa Positive slope from Bottom to Right side
			#
			case "IIIa":
				self.east.start_x = self.east.x_at_min_y
				self.east.start_y = self.vert_min
				self.east.end_x = self.hor_max
				self.east.end_y = self.east.y_at_max_x
			#
			# east Case IVa Positive slope from Bottom to Top and min_x < hor_min
			#
			case "IVa":
				self.east.start_x = self.east.x_at_min_y
				self.east.start_y = self.vert_min
				self.east.end_x = self.east.x_at_max_y
				self.east.end_y = self.vert_max
			#
			# self.east Case Ib Negative slope from Left side to Right side
			#
			case "Ib":
				self.east.start_x = self.hor_min
				self.east.start_y = self.east.y_at_min_x
				self.east.end_x = self.hor_max
				self.east.end_y = self.east.y_at_max_x
			#
			# self.east Case IIb Negative slope from Left side to Bottom
			#
			case "IIb":
				self.east.start_x = self.hor_min
				self.east.start_y = self.east.y_at_min_x
				self.east.end_x = self.east.x_at_min_y
				self.east.end_y = self.vert_min
			#
			# self.east Case IIIb Negative slope from Top to Right side
			#
			case "IIIb":
				self.east.start_x = self.east.x_at_max_y
				self.east.start_y = self.vert_max
				self.east.end_x = self.hor_max
				self.east.end_y = self.east.y_at_max_x
			#
			# self.east Case IVb Negative slope from Top to Bottom
			#
			case "IVb":
				self.east.start_x = self.east.x_at_max_y
				self.east.start_y = self.vert_max
				self.east.end_x = self.east.x_at_min_y
				self.east.end_y = self.vert_min
		return

	# ----------------------------------------------------------------------------------

	def varimax_function(self, Phi, gamma=1.0, q=20, tol=1e-6):
		p, k = np.shape(Phi)
		r = eye(k)
		d = 0
		xrange = range(q)
		for i in xrange:
			d_old = d
			Lambda = dot(Phi, r)
			u, s, vh = svd(
				dot(Phi.T, asarray(Lambda) ** 3 - (gamma / p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
			R = dot(u, vh)
			d = sum(s)
			if d_old != 0 and d / d_old < 1 + tol: break

		return dot(Phi, r)

	# ----------------------------------------------------------------------------------

	def write(self, file_name):

		problem_writing_file = False
		#
		try:
			with open(file_name, 'x') as file_handle:

				#
				# Write line declaring file type to be "Configuration".	\
				#
				file_handle.write("Configuration\n")
				#
				# Write line showing number of dimensions and number of points
				#
				file_handle.write(" " + str(self.ndim) + " " + str(self.npoint) + "\n")
				#
				# Write a line for each dimension showing the label and name for each dimension,
				# separated by a semicolon.
				#
				# semi_string = ";"
				# semi_list = [";"]
				# semi_list2 = {}
				# semi_list2.append(";")
				for each_dim in self.range_dims:
					file_handle.write(f"{self.dim_labels[each_dim]};{self.dim_names[each_dim].strip()}\n")
					# file_handle.write(self.dim_labels[each_dim] + ";" + self.dim_names[each_dim].strip() + "\n")
					# file_handle.write(self.dim_labels[each_dim] + semi_string + self.dim_names[each_dim].strip())  # + "\n")
					# file_handle.write(self.dim_labels[each_dim] + semi_list + self.dim_names[each_dim].strip()) # + "\n")
					# file_handle.write(self.dim_labels[each_dim] + semi_list2 + self.dim_names[each_dim].strip()) # + "\n")
					# file_handle.write(self.dim_labels[each_dim] + semi_list2[0] + self.dim_names[each_dim].strip() + "\n")
				#
				# Write a line for each point showing the label and name for the point,
				# separated by a semicolon.
				#
				for each_point in self.range_points:
					file_handle.write(f"{self.point_labels[each_point]};{self.point_names[each_point]}\n")
					#file_handle.write(self.point_labels[each_point] + ";" + self.point_names[each_point] + "\n")
				#
				# Write a line for each point with the coordinate of the point on each dimension,
				# separated by a blank.
				#
				for each_point in self.range_points:
					for each_dim in self.range_dims:
						file_handle.write(str(self.point_coords.iloc[each_point][each_dim]) + " ")
						continue
					file_handle.write("\n")
					continue
		except FileExistsError:
			self.error("File already exists: ",
				file_name)
			problem_writing_file = True
			return problem_writing_file

		problem_writing_file = False
		#
		return problem_writing_file

#
# End of configuration class
# --------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------
#  Start of functions
#
# --------------------------------------------------------------------------------------------


# def example_plot():
# 	fig, ax = plt.subplots()
# 	ax.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
# 	ax.set_title('Example Plot')
# 	ax.set_xlabel('X-axis')
# 	ax.set_ylabel('Y-axis')
# 	return fig

def main():

	#
	# Identify which version of Python is being used
	#
	print(sys.version)
	print(sys.version_info)
	#
	# Begin creation of GUI window
	#
	spaces_app = QApplication(sys.argv)
	director = Status()
	#
	director.show()
	#
	sys.stdout = MyTextEditWrapper(director.text_edit)
	#
	# Show welcome screen
	#
	ui_file_name = "spaces_welcome.ui"
	ui_file = QFile(ui_file_name)
	if not ui_file.open(QIODevice.ReadOnly):
		print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
		sys.exit(-1)
	loader = QUiLoader()
	welcome_dialog = loader.load(ui_file)
	ui_file.close()
	if not welcome_dialog:
		print(loader.errorString())
		sys.exit(-1)
	#qss = self.editor.toPlainText()

	# QMainWindow
	# {
	#     Background: orange;
	# }

	welcome_dialog.show()
	welcome_dialog.exec()
	#
	# Start event loop
	#
	sys.exit(spaces_app.exec())
	#
	# Here if the user selected Exit
	#
	print(f"DEBUG -- {director.active.commands_used = }")
	print(f"DEBUG -- {director.active.command_exit_code = }")
	print(f"DEBUG -- {director.replies_log = }")
	print(f"DEBUG -- \n{director.undo_stack_source = }\n")
	#
	quit()
	#


if __name__ == '__main__':
	main()
