cmake_minimum_required(VERSION 3.5)
project(drl_grasping)

# Default to C11
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 11)
endif()
# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Testing and linting
if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)

# Install python module
ament_python_install_module(
    drl_grasping
)

# Install scripts
set(SCRIPTS_DIR scripts)
install(PROGRAMS
    ${SCRIPTS_DIR}/utils/process_collection.py
    ${SCRIPTS_DIR}/enjoy.py
    ${SCRIPTS_DIR}/train.py
    ${SCRIPTS_DIR}/preload_replay_buffer.py
    DESTINATION lib/${PROJECT_NAME}
)

# Install tests
set(TESTS_DIR tests)
install(PROGRAMS
    ${TESTS_DIR}/test_env.py
    ${TESTS_DIR}/test_octree_conv.py
    DESTINATION lib/${PROJECT_NAME}
)

# Install examples
set(EXAMPLES_DIR examples)
install(PROGRAMS
    ${EXAMPLES_DIR}/ex_enjoy.bash
    ${EXAMPLES_DIR}/ex_optimize.bash
    ${EXAMPLES_DIR}/ex_train.bash
    ${EXAMPLES_DIR}/ex_preload_replay_buffer.bash
    DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

# Setup the project
ament_package()
