# RoboCute

Make Robotics Cute!

A robotics framework with physics simulation and rendering capabilities.

## Usage

### Building the Project

Use the provided `build.py` script to build the project:

```bash
python build.py <work_dir> [options]
```

**Arguments:**
- `work_dir` - Path to the RoboCute work directory (required)

**Options:**
- `-r, --rebuild` - Force rebuild by passing -r to build_and_copy.py

**Example:**

```bash
# Build the project
python build.py D:/RoboCute

# Force rebuild
python build.py D:/RoboCute -r
```

**What the build script does:**
1. Configures xmake for release mode (`xmake f -m release -c`)
2. Runs the build and copy script (`uv run scripts/build_and_copy.py`)
3. Builds the project with uv (`uv build`)
4. Copies distribution files from `dist/` and extracts tar.gz archives
