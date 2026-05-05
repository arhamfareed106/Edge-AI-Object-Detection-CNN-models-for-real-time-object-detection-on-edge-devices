# Edge AI Object Detection - Research Project

## Comparative Evaluation and Optimisation of Lightweight Convolutional Neural Networks for Real-Time Object Detection on Raspberry Pi Edge Devices

---

## 📋 Table of Contents
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)
- [Research Metrics](#research-metrics)

---

## 🎯 Overview

This project provides a comprehensive framework for evaluating and optimizing lightweight CNN models for real-time object detection on edge devices, specifically the Raspberry Pi 4. It implements three models with four optimization variants each, resulting in 12 complete experiments with detailed performance metrics.

### Models Implemented:
1. **MobileNetV2** - Efficient architecture using depthwise separable convolutions
2. **SqueezeNet** - Ultra-small parameter count with fire modules
3. **YOLOv4-tiny** - Lightweight YOLO variant for real-time detection

### Optimization Techniques:
- **Baseline** - No optimization (reference performance)
- **Quantization** - Post-training INT8 quantization
- **Pruning** - Structured pruning with 50% sparsity
- **Combined** - Pruning followed by quantization

---

## 🔧 Hardware Requirements

### Minimum Requirements:
- **Raspberry Pi 4** (4GB RAM recommended, 8GB ideal)
- **MicroSD Card** (32GB minimum, Class 10)
- **Power Supply** (5V 3A USB-C)
- **Camera Module** (Pi Camera v2 or USB webcam)
- **Cooling** (Heatsinks + fan recommended for sustained inference)

### Optional Hardware:
- Google Coral USB Accelerator (for Edge TPU support)
- Official Raspberry Pi Touch Display
- Pi Camera Module v2 (8MP)

---

## 💻 Software Requirements

- **Raspberry Pi OS** (64-bit recommended)
- **Python 3.9+**
- **OpenCV 4.8+**
- **TensorFlow Lite 2.13+**

---

## 📦 Installation

### Step 1: Clone/Download the Project
```bash
cd ~/projects
git clone <your-repo-url>
cd Edge-AI-Object
```

### Step 2: Run Setup Script
```bash
chmod +x setup.sh
bash setup.sh
```

This script will:
- Update system packages
- Install system dependencies
- Create Python virtual environment
- Install all Python packages
- Enable camera interface (if using Pi Camera)
- Create required directories

### Step 3: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 4: Download Pre-trained Models
```bash
python src/dataset_prepare.py --download-models
```

---

## 🚀 Quick Start

### Run a Single Experiment:
```bash
python main.py --model mobilenet --optimization baseline --mode single
```

### Run Full Experiment Suite (all 12 combinations):
```bash
python main.py --mode full_experiment
```

### Or use the automated script:
```bash
chmod +x run_experiments.sh
bash run_experiments.sh
```

### Generate Visualizations Only:
```bash
python src/visualization.py --results output/csv/experiment_results.csv
```

### Run Real-time Inference:
```bash
python main.py --model mobilenet --optimization baseline --mode realtime
```

---

## 📁 Project Structure

```
Edge-AI-Object/
├── config.yaml                    # Configuration parameters
├── requirements.txt               # Python dependencies
├── setup.sh                       # Installation script
├── run_experiments.sh             # Automated experiment runner
├── README.md                      # This file
│
├── src/                           # Source code
│   ├── main.py                    # Entry point & CLI
│   ├── hardware_monitor.py        # Resource monitoring
│   ├── model_loader.py            # Model loading utilities
│   ├── model_optimizer.py         # Optimization functions
│   ├── evaluator.py               # Benchmarking & metrics
│   ├── camera_capture.py          # Camera handling
│   ├── inference_with_boxes.py    # Detection visualization
│   ├── main_experiment.py         # Experiment orchestration
│   ├── visualization.py           # Graph generation
│   └── dataset_prepare.py         # Dataset preparation
│
├── models/                        # Saved models
│   ├── mobilenet_v2.tflite
│   ├── squeezenet.tflite
│   └── yolov4-tiny.*
│
├── datasets/                      # Dataset storage
│   ├── pascal_voc/
│   └── cifar10/
│
└── output/                        # Results
    ├── csv/                       # Experiment results CSV
    ├── graphs/                    # Visualization PNGs
    ├── images/                    # Inference images
    └── logs/                      # Log files
```

---

## 🧪 Running Experiments

### Command Line Options:

```bash
python main.py [OPTIONS]

Options:
  --model MODEL          Model to use: mobilenet, squeezenet, yolo
  --optimization OPT     Optimization: baseline, quantization, pruning, combined
  --mode MODE            Mode: single, full_experiment, realtime, benchmark
  --dataset DATASET      Dataset: pascal_voc, cifar10
  --camera CAMERA        Camera: picamera, usb
  --resolution RES       Resolution: 640x480, 320x240
  --save_results         Save results to CSV
  --verbose              Enable debug logging
```

### Examples:

```bash
# Test MobileNet with quantization
python main.py --model mobilenet --optimization quantization --mode single

# Run YOLO baseline experiment
python main.py --model yolo --optimization baseline --mode single --save_results

# Real-time detection with SqueezeNet
python main.py --model squeezenet --optimization baseline --mode realtime --camera usb

# Custom benchmark with specific dataset
python main.py --model mobilenet --optimization pruning --mode benchmark --dataset cifar10
```

---

## 📊 Expected Outputs

### 1. Terminal Output:
```
========================================
Edge AI Object Detection Experiment
Raspberry Pi 4 - Lightweight CNN Evaluation
========================================
[INFO] Loading MobileNet...
[INFO] Applying quantization...
[INFO] Running benchmark...
[INFO] FPS: 28.4 | Latency: 35.2ms | mAP: 0.72
[INFO] Results saved to output/csv/experiment_results.csv
========================================
```

### 2. CSV File (`output/csv/experiment_results.csv`):
| model | optimization | fps | latency_ms | map_score | model_size_mb | energy_joules | cpu_percent | ram_mb | timestamp |

### 3. Graphs (8 PNG files in `output/graphs/`):
- `graph_1_fps_comparison.png` - FPS across models & optimizations
- `graph_2_latency_comparison.png` - Latency comparison
- `graph_3_accuracy_vs_speed.png` - Accuracy-Speed tradeoff scatter plot
- `graph_4_energy_heatmap.png` - Energy consumption heatmap
- `graph_5_model_size.png` - Model size comparison
- `graph_6_radar_chart.png` - Multi-dimensional performance radar
- `graph_7_fps_over_time.png` - Real-time FPS stability
- `graph_8_confusion_matrix.png` - Detection accuracy matrix

### 4. Inference Images (`output/images/`):
- Input images with bounding boxes and labels
- Timestamp and confidence score overlays

### 5. Log Files (`output/logs/`):
- Detailed experiment logs with timestamps
- Error tracking and debugging information

---

## 🔍 Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution:** Activate virtual environment:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Camera not detected
**Solution:** 
```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Or check USB camera
ls /dev/video*
```

### Issue: Out of memory
**Solution:** 
- Use lower resolution: `--resolution 320x240`
- Close other applications
- Use swap file (if available)

### Issue: Low FPS
**Solution:**
- Use TensorFlow Lite models instead of full TensorFlow
- Apply quantization or pruning optimizations
- Reduce input resolution
- Enable multi-threading in config.yaml

### Issue: Models not downloading
**Solution:**
```bash
# Manual download
python src/dataset_prepare.py --download-models --force
```

### Issue: Raspberry Pi overheating
**Solution:**
- Add heatsinks and fan
- Reduce continuous inference duration
- Monitor temperature: `vcgencmd measure_temp`

---

## 📈 Research Metrics

### Performance Metrics:
1. **FPS (Frames Per Second)** - Throughput measurement
2. **Latency (ms)** - Inference time per frame
3. **mAP (Mean Average Precision)** - Detection accuracy (0-1)
4. **Model Size (MB)** - Storage requirement
5. **Energy Consumption (Joules)** - Power efficiency
6. **CPU Usage (%)** - Processing overhead
7. **RAM Usage (MB)** - Memory footprint

### Evaluation Methodology:
- Each experiment runs 100+ inference iterations
- FPS measured over 5-second window
- Latency averaged over 100 runs
- mAP calculated on held-out test set
- Energy estimated using Pi power models
- Resource usage sampled every 0.1 seconds

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{edge-ai-object-2024,
  title={Comparative Evaluation and Optimisation of Lightweight Convolutional Neural Networks for Real-Time Object Detection on Raspberry Pi Edge Devices},
  year={2024},
  howpublished={\url{https://github.com/your-repo/Edge-AI-Object}}
}
```

---

## 📄 License

This project is for educational and research purposes.

---

## 👥 Contributors

- Add your name and contact here

---

## 🆘 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review log files in `output/logs/`

---

**Happy experimenting! 🚀**
"# Edge-AI-Object-Detection-CNN-models-for-real-time-object-detection-on-edge-devices" 
