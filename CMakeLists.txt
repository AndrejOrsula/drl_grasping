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

# # Executables
# set(EXECUTABLE_0 executable_0)
# # Executable 0
# add_executable(${EXECUTABLE_0} src/${EXECUTABLE_0}.cpp)
# target_link_libraries(${EXECUTABLE_0})
# ament_target_dependencies(${EXECUTABLE_0}
#     rclcpp
# )

# # Install cpp targets
# install(TARGETS
#     ${EXECUTABLE_0}
#     DESTINATION lib/${PROJECT_NAME}
# )

# Install python module
ament_python_install_module(
    drl_grasping
)

# Install python scripts and examples
set(SCRIPTS_DIR scripts)
set(EXAMPLES_DIR examples)
install(PROGRAMS
    ${SCRIPTS_DIR}/train_grasping.py
    ${SCRIPTS_DIR}/test_models.py
    # ${EXAMPLES_DIR}/example_0.py
    DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY config launch drl_grasping/envs/worlds DESTINATION share/${PROJECT_NAME})

# Setup the project
ament_package()