#!/bin/bash

# setup_env.sh lives one level up (project root)
SETUP_ENV_PATH="$(dirname "${BASH_SOURCE[0]}")/../setup_env.sh"

# Path to your tests directory
TESTS_DIR="$(dirname "${BASH_SOURCE[0]}")/tests"

# Source the setup_env.sh file (it will handle virtual environment activation)
echo "Sourcing setup_env.sh..."
source "$SETUP_ENV_PATH"

# Install pytest requirements if not already installed
pip install -r tests/test_resources/requirements.txt

# Download all HEFs
hailo-download-resources --group all

# Run pytest for all test files
echo "Running tests..."
# pytest --log-cli-level=INFO \
#        #"$TESTS_DIR/test_sanity_check.py" \
#        "$TESTS_DIR/test_hailo_rpi5_examples.py" \
#        #"$TESTS_DIR/test_edge_cases.py" \
#        #"$TESTS_DIR/test_advanced.py" \
#        #"$TESTS_DIR/test_infra.py"

pytest -s "$TESTS_DIR"/test_hailo_rpi5_examples.py

echo "All tests completed."
