from typing import Tuple, Dict, Optional
import subprocess
import tempfile
import xacro


def xacro2sdf(
    input_file_path: str, mappings: Dict, model_path_remap: Optional[Tuple[str, str]]
) -> str:
    """Convert xacro (URDF variant) with given arguments to SDF and return as a string."""

    # Convert xacro to URDF
    urdf_xml = xacro.process(input_file_name=input_file_path, mappings=mappings)

    # Create temporary file for URDF (`ign sdf -p` accepts only files)
    with tempfile.NamedTemporaryFile() as tmp_urdf:
        with open(tmp_urdf.name, "w") as urdf_file:
            urdf_file.write(urdf_xml)

        # Convert to SDF
        result = subprocess.run(
            ["ign", "sdf", "-p", tmp_urdf.name], stdout=subprocess.PIPE
        )
        sdf_xml = result.stdout.decode("utf-8")

        # Remap package name to model name, such that meshes can be located by Ignition
        if model_path_remap is not None:
            sdf_xml = sdf_xml.replace(model_path_remap[0], model_path_remap[1])

        # Return as string
        return sdf_xml
