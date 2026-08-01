# Omnibot — příkazy pro snadné spouštění z konzole.
# Vyžaduje spuštění z terminálu, kde už byl proveden source na setup_env.sh


export PYTHONPATH := justfile_directory() + ":" + justfile_directory() + "/robot"
venv_python := justfile_directory() + "/venv_hailo_rpi_examples/bin/python3"

# OpenCV's Qt5 GUI backend fails to find its native Wayland plugin on this Pi
# ("could not find wayland") - force it onto the XWayland/X11 compatibility
# layer instead, which is present and reliable under the default desktop.
export QT_QPA_PLATFORM := "xcb"

# Zobrazí seznam dostupných příkazů (výchozí příkaz)
default:
    @just --list

# ── Provoz robota ────────────────────────────────────────────────────────────

# Spustí webový ovládací server → http://<pi-ip>:5000
server:
    {{venv_python}} robot/robot_server.py

# Klávesnicový teleop (WASD = pohyb, G = gripper, mezerník = stop)
drive:
    {{venv_python}} robot/drive.py

# ── Hardwarové diagnostiky ───────────────────────────────────────────────────

# Živé odečty ze sonaru HC-SR04
distance:
    {{venv_python}} robot/distance.py

# Diagnostika servo kanálů / kalibrace _TRIM
force-test:
    {{venv_python}} robot/force_test.py

# ── Inference prototypy ──────────────────────────────────────────────────────

# Detekce na jednom obrázku → robot/assets/output_detection.jpg
detect-image:
    {{venv_python}} robot/custom_yolo26.py

# Živá detekce z kamery
detect-video:
    {{venv_python}} robot/video_yolo26.py

# ── Sběr dat a vyhodnocení (kapitola 2.5) ────────────────────────────────────

# Interaktivní sběr testovací sady (SPACE = uložit, Q = konec)
capture OUT="dataset" *ARGS="":
    {{venv_python}} robot/eval/capture_dataset.py --out {{OUT}} {{ARGS}}

# Ruční kontrola/oprava anotací — NUTNÉ udělat před eval-detection (2.5.1)
review DIR="dataset":
    {{venv_python}} robot/eval/review_dataset.py --dir {{DIR}}

# mAP@0.5 vyhodnocení detektoru na ručně zkontrolované sadě (2.5.1)
# Např.: just eval-detection dataset --exclude cup-disposable
eval-detection IMAGES *ARGS="":
    {{venv_python}} robot/eval/eval_detection.py --images {{IMAGES}} {{ARGS}}

# Vyhodnocení bez ohledu na třídu — jen "je tam nějaký beverage container a kde?" (2.5.1)
eval-detection-agnostic IMAGES *ARGS="":
    {{venv_python}} robot/eval/eval_detection_agnostic.py --images {{IMAGES}} {{ARGS}}

# Interaktivní navigační zkoušky — vyžaduje fyzicky přítomného robota (2.5.2)
eval-navigation TRIALS="5" *ARGS="":
    {{venv_python}} robot/eval/eval_navigation.py --trials {{TRIALS}} {{ARGS}}

# Efektivita mapování prostoru ze záznamů relací — bez hardwaru (2.5.3)
eval-mapping *ARGS="":
    {{venv_python}} robot/eval/eval_mapping.py {{ARGS}}

# ── Testování (2.5.3) ────────────────────────────────────────────────────────

# Spustí unit testy
test:
    cd robot && {{venv_python}} -m pytest tests/ -v

# Spustí unit testy s reportem pokrytí jádra systému
coverage:
    cd robot && {{venv_python}} -m pytest tests/ \
        --cov=omnibot --cov=navigator --cov=polar_scan --cov=sonar_guard \
        --cov=state_machine --cov=inference_pipeline --cov=robot_server \
        --cov-report=term-missing

# ── Instalace Hailo SDK ──────────────────────────────────────────────────────

# Plná instalace (čte hailo/config.yaml)
install:
    bash hailo/install.sh

# Pouze znovu stáhne zdroje/modely
install-resources:
    bash hailo/install.sh --no-installation

# Stáhne všechny varianty modelů
install-all:
    bash hailo/install.sh --all

# ── Úklid ────────────────────────────────────────────────────────────────────

# Smaže cache soubory (__pycache__, .pytest_cache)
clean:
    find . -type d -name "__pycache__" -not -path "*/venv_hailo_rpi_examples/*" -exec rm -rf {} +
    rm -rf robot/.pytest_cache
