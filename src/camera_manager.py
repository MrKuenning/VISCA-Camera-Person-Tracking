import json
import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QLineEdit, QPushButton, QListWidget, QMessageBox, 
                           QCheckBox, QGroupBox, QWidget, QFormLayout)
from PyQt6.QtCore import Qt

class CameraManager:
    def __init__(self, json_path=None):
        """Initialize the camera manager with the path to the JSON file"""
        if json_path is None:
            # Use default path in the same directory as this script
            self.json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cameras.json")
        else:
            self.json_path = json_path
        
        self.cameras = self.load_cameras()
    
    def load_cameras(self):
        """Load camera configurations from JSON file"""
        try:
            # Updated path to config folder
            json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "cameras.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    self.cameras = data.get('cameras', [])
            else:
                self.cameras = [] # Initialize if file doesn't exist
        except Exception as e:
            print(f"Error loading cameras: {str(e)}")
            return []
        return self.cameras
    
    def save_cameras(self):
        """Save camera configurations to JSON file"""
        try:
            # Updated path to config folder
            json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "cameras.json")
            with open(json_path, 'w') as f:
                json.dump({'cameras': self.cameras}, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving cameras: {str(e)}")
            return False
    
    def add_camera(self, name, ip, control_port, stream_url, inverted=False):
        """Add a new camera to the list"""
        # Check if camera with this name already exists
        for camera in self.cameras:
            if camera["name"] == name:
                return False, "A camera with this name already exists"
        
        # Add the new camera
        new_camera = {
            "name": name,
            "ip": ip,
            "control_port": int(control_port),
            "stream_url": stream_url,
            "inverted": inverted
        }
        
        self.cameras.append(new_camera)
        success = self.save_cameras()
        
        if success:
            return True, "Camera added successfully"
        else:
            return False, "Failed to save camera configuration"
    
    def remove_camera(self, name):
        """Remove a camera from the list by name"""
        for i, camera in enumerate(self.cameras):
            if camera["name"] == name:
                del self.cameras[i]
                success = self.save_cameras()
                
                if success:
                    return True, "Camera removed successfully"
                else:
                    return False, "Failed to save camera configuration"
        
        return False, "Camera not found"


class CameraEditorDialog(QDialog):
    def __init__(self, parent, camera_manager):
        super().__init__(parent)
        self.setWindowTitle("Camera Manager")
        self.setFixedSize(500, 500)
        
        self.camera_manager = camera_manager
        
        self.setup_ui()
        self.populate_camera_list()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Camera list section
        list_group = QGroupBox("Camera List")
        list_layout = QVBoxLayout()
        
        self.camera_list_widget = QListWidget()
        self.camera_list_widget.itemSelectionChanged.connect(self.on_camera_select)
        list_layout.addWidget(self.camera_list_widget)
        
        # List buttons
        list_btn_layout = QHBoxLayout()
        self.remove_btn = QPushButton("Remove Selected Camera")
        self.remove_btn.clicked.connect(self.remove_camera)
        list_btn_layout.addWidget(self.remove_btn)
        list_btn_layout.addStretch()
        
        list_layout.addLayout(list_btn_layout)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Camera details section
        details_group = QGroupBox("Camera Details")
        details_layout = QFormLayout()
        
        self.name_input = QLineEdit()
        details_layout.addRow("Camera Name:", self.name_input)
        
        self.ip_input = QLineEdit()
        details_layout.addRow("IP Address:", self.ip_input)
        
        self.port_input = QLineEdit()
        self.port_input.setText("52381")
        details_layout.addRow("Control Port:", self.port_input)
        
        self.stream_input = QLineEdit()
        details_layout.addRow("Stream URL:", self.stream_input)
        
        self.inverted_check = QCheckBox("Camera Mounted Upside Down")
        details_layout.addRow("", self.inverted_check)
        
        self.add_btn = QPushButton("Add New Camera")
        self.add_btn.clicked.connect(self.add_camera)
        # Style the add button to look primary
        self.add_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        details_layout.addRow(self.add_btn)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)
        
    def populate_camera_list(self):
        """Populate the list widget with camera names"""
        self.camera_list_widget.clear()
        for camera in self.camera_manager.cameras:
            self.camera_list_widget.addItem(camera["name"])
            
    def on_camera_select(self):
        """Handle camera selection from the list"""
        selected_items = self.camera_list_widget.selectedItems()
        if not selected_items:
            return
            
        name = selected_items[0].text()
        
        # Find camera data
        camera_data = None
        for cam in self.camera_manager.cameras:
            if cam["name"] == name:
                camera_data = cam
                break
        
        if camera_data:
            # Populate fields
            self.name_input.setText(camera_data["name"])
            self.ip_input.setText(camera_data["ip"])
            self.port_input.setText(str(camera_data["control_port"]))
            self.stream_input.setText(camera_data["stream_url"])
            self.inverted_check.setChecked(camera_data.get("inverted", False))
            
    def add_camera(self):
        """Add a new camera"""
        name = self.name_input.text().strip()
        ip = self.ip_input.text().strip()
        port_str = self.port_input.text().strip()
        stream = self.stream_input.text().strip()
        inverted = self.inverted_check.isChecked()
        
        if not name or not ip or not port_str or not stream:
            QMessageBox.warning(self, "Validation Error", "All fields are required.")
            return
            
        try:
            port = int(port_str)
        except ValueError:
            QMessageBox.warning(self, "Validation Error", "Port must be a number.")
            return
            
        success, message = self.camera_manager.add_camera(name, ip, port, stream, inverted)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.populate_camera_list()
            # Clear inputs for next entry
            self.name_input.clear()
            self.ip_input.clear()
            self.port_input.setText("52381")
            self.stream_input.clear()
            self.inverted_check.setChecked(False)
        else:
            QMessageBox.critical(self, "Error", message)
            
    def remove_camera(self):
        """Remove selected camera"""
        selected_items = self.camera_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a camera to remove.")
            return
            
        name = selected_items[0].text()
        
        reply = QMessageBox.question(self, "Confirm Removal", 
                                   f"Are you sure you want to remove '{name}'?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                                   
        if reply == QMessageBox.StandardButton.Yes:
            success, message = self.camera_manager.remove_camera(name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.populate_camera_list()
                # Clear inputs
                self.name_input.clear()
                self.ip_input.clear()
                self.port_input.setText("52381")
                self.stream_input.clear()
                self.inverted_check.setChecked(False)
            else:
                QMessageBox.critical(self, "Error", message)

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    manager = CameraManager()
    dialog = CameraEditorDialog(None, manager)
    dialog.show()
    sys.exit(app.exec())