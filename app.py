"""
Tkinter desktop application for Parkinson's freezing-of-gait prediction.

Required files in the same folder:
    traditional_quantum_kernel_svm.joblib
    traditional_quantum_ml_scaler.joblib
    quantum_kernel_training_features.npy
    traditional_quantum_ml_metadata.json   # optional

The application uses the same quantum fidelity kernel as the training script
and sends the resulting precomputed kernel row to the saved SVM.
"""

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import numpy as np
import pennylane as qml


# ============================================================
# 1. FILES AND MODEL CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "traditional_quantum_kernel_svm.joblib"
SCALER_PATH = APP_DIR / "traditional_quantum_ml_scaler.joblib"
TRAIN_FEATURES_PATH = APP_DIR / "quantum_kernel_training_features.npy"
METADATA_PATH = APP_DIR / "traditional_quantum_ml_metadata.json"

DEFAULT_FEATURES = [
    "ACC ML [g]",
    "ACC AP [g]",
    "ACC SI [g]",
    "GYR ML [deg/s]",
    "GYR AP [deg/s]",
    "GYR SI [deg/s]",
]

LABEL_NAMES = {
    0: "No Freezing",
    1: "Freezing Detected",
}


# ============================================================
# 2. LOAD TRAINED MODEL FILES
# ============================================================

def load_artifacts():
    required_files = [MODEL_PATH, SCALER_PATH, TRAIN_FEATURES_PATH]
    missing_files = [str(path.name) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "The following required model files are missing:\n\n"
            + "\n".join(missing_files)
            + "\n\nCopy them into the same folder as this application."
        )

    svm_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    quantum_training_features = np.load(TRAIN_FEATURES_PATH)

    feature_names = DEFAULT_FEATURES.copy()
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        feature_names = metadata.get("features", feature_names)

    if len(feature_names) != quantum_training_features.shape[1]:
        raise ValueError(
            "The saved feature metadata and quantum training features do not match."
        )

    return svm_model, scaler, quantum_training_features, feature_names


# Load model assets before starting the GUI, so errors are clear.
try:
    svm_model, scaler, X_train_q, FEATURES = load_artifacts()
except Exception as exc:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Model Loading Error", str(exc))
    root.destroy()
    sys.exit(1)

N_QUBITS = len(FEATURES)

try:
    qdev = qml.device("lightning.qubit", wires=N_QUBITS)
    DEVICE_NAME = "lightning.qubit"
except Exception:
    qdev = qml.device("default.qubit", wires=N_QUBITS)
    DEVICE_NAME = "default.qubit"


# ============================================================
# 3. SAME QUANTUM KERNEL USED DURING TRAINING
# ============================================================

@qml.qnode(qdev)
def quantum_fidelity_kernel(x1, x2):
    qml.AngleEmbedding(
        x1,
        wires=range(N_QUBITS),
        rotation="Y",
    )
    qml.adjoint(qml.AngleEmbedding)(
        x2,
        wires=range(N_QUBITS),
        rotation="Y",
    )
    return qml.expval(
        qml.Projector(
            [0] * N_QUBITS,
            wires=range(N_QUBITS),
        )
    )


def compute_kernel_row(new_sample_q, training_features_q):
    """Compute K(new_sample, training_samples) for the precomputed SVM."""
    kernel_row = np.empty((1, len(training_features_q)), dtype=np.float64)

    for index, training_sample in enumerate(training_features_q):
        kernel_row[0, index] = float(
            quantum_fidelity_kernel(new_sample_q, training_sample)
        )

    return kernel_row


# ============================================================
# 4. PREDICTION FUNCTION
# ============================================================

def predict_freezing(values):
    values = np.asarray(values, dtype=np.float64).reshape(1, -1)

    if values.shape[1] != len(FEATURES):
        raise ValueError(f"Exactly {len(FEATURES)} sensor values are required.")

    if not np.isfinite(values).all():
        raise ValueError("All sensor values must be finite numbers.")

    # Apply exactly the same preprocessing used during model training.
    scaled_values = scaler.transform(values)
    quantum_values = np.clip(scaled_values, -np.pi, np.pi)

    kernel_row = compute_kernel_row(quantum_values[0], X_train_q)
    prediction = int(svm_model.predict(kernel_row)[0])
    decision_score = float(svm_model.decision_function(kernel_row)[0])

    return prediction, decision_score


# ============================================================
# 5. TKINTER GUI
# ============================================================

class GaitPredictionApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Quantum-Kernel Gait Freezing Predictor")
        self.root.geometry("720x600")
        self.root.minsize(650, 520)

        self.entries = []
        self.status_var = tk.StringVar(value="Ready for sensor input.")
        self.result_var = tk.StringVar(value="Prediction will appear here.")
        self.score_var = tk.StringVar(value="")
        self.progress_var = tk.StringVar(value="")

        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="Parkinson's Freezing-of-Gait Prediction",
            font=("Arial", 18, "bold"),
        )
        title.pack(pady=(0, 5))

        subtitle = ttk.Label(
            main,
            text="Traditional quantum ML: quantum fidelity kernel + SVM",
            font=("Arial", 10),
        )
        subtitle.pack(pady=(0, 15))

        model_info = ttk.Label(
            main,
            text=(
                f"Quantum device: {DEVICE_NAME}    |    "
                f"Training kernel samples: {len(X_train_q)}"
            ),
        )
        model_info.pack(pady=(0, 15))

        input_frame = ttk.LabelFrame(main, text="Gait Sensor Values", padding=15)
        input_frame.pack(fill="x", pady=5)

        for row, feature_name in enumerate(FEATURES):
            label = ttk.Label(input_frame, text=feature_name, width=20)
            label.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=5)

            entry = ttk.Entry(input_frame, width=30)
            entry.grid(row=row, column=1, sticky="ew", pady=5)
            self.entries.append(entry)

        input_frame.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(main)
        button_frame.pack(pady=18)

        predict_button = ttk.Button(
            button_frame,
            text="Predict Gait State",
            command=self.predict,
        )
        predict_button.grid(row=0, column=0, padx=8)

        clear_button = ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear,
        )
        clear_button.grid(row=0, column=1, padx=8)

        result_frame = ttk.LabelFrame(main, text="Prediction Result", padding=15)
        result_frame.pack(fill="x", pady=5)

        self.result_label = tk.Label(
            result_frame,
            textvariable=self.result_var,
            font=("Arial", 17, "bold"),
            fg="#1f4e79",
            wraplength=600,
        )
        self.result_label.pack(pady=8)

        ttk.Label(
            result_frame,
            textvariable=self.score_var,
            font=("Arial", 10),
        ).pack(pady=4)

        ttk.Label(
            main,
            textvariable=self.progress_var,
            foreground="#555555",
        ).pack(pady=(12, 2))

        ttk.Label(
            main,
            textvariable=self.status_var,
            foreground="#555555",
            wraplength=650,
        ).pack(pady=5)

        note = ttk.Label(
            main,
            text=(
                "Enter the six values using the same units and feature order as the "
                "training dataset. A prediction may take time because the quantum "
                "kernel is evaluated against every saved training sample."
            ),
            wraplength=650,
            justify="left",
        )
        note.pack(pady=(20, 0))

    def predict(self):
        try:
            values = [float(entry.get().strip()) for entry in self.entries]
        except ValueError:
            messagebox.showwarning(
                "Invalid Input",
                "Please enter a valid numeric value in every sensor field.",
            )
            return

        self.progress_var.set("Computing quantum kernel; please wait...")
        self.status_var.set("The prediction is being calculated.")
        self.root.update_idletasks()

        try:
            prediction, decision_score = predict_freezing(values)
        except Exception as exc:
            self.progress_var.set("")
            self.status_var.set("Prediction failed.")
            messagebox.showerror("Prediction Error", str(exc))
            return

        self.progress_var.set("")
        self.result_var.set(LABEL_NAMES.get(prediction, str(prediction)))
        self.score_var.set(f"SVM decision score: {decision_score:.6f}")
        self.status_var.set(
            "Prediction completed. The decision score is not a calibrated probability."
        )

        if prediction == 1:
            self.result_label.configure(fg="#b22222")
        else:
            self.result_label.configure(fg="#228b22")

    def clear(self):
        for entry in self.entries:
            entry.delete(0, tk.END)
        self.result_var.set("Prediction will appear here.")
        self.score_var.set("")
        self.progress_var.set("")
        self.status_var.set("Ready for sensor input.")
        self.result_label.configure(fg="#1f4e79")


if __name__ == "__main__":
    root = tk.Tk()
    app = GaitPredictionApp(root)
    root.mainloop()
