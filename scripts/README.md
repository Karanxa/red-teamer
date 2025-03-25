# Helper Scripts for Red Teaming Framework

This directory contains helper scripts for the Red Teaming Framework.

## fix_streamlit_torch.sh

A helper script for running Streamlit applications with PyTorch compatibility fixes enabled.

### Purpose

This script solves compatibility issues between PyTorch and Streamlit that can cause errors like:
- `RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!`
- `RuntimeError: no running event loop`

These errors occur because of conflicts between how PyTorch's custom module system works and how Streamlit monitors files for changes.

### Usage

```bash
./scripts/fix_streamlit_torch.sh [command]
```

Examples:
```bash
# Run the conversational red teaming tool
./scripts/fix_streamlit_torch.sh python -m redteamer.conversational_redteam_launcher

# Run the static scan tool
./scripts/fix_streamlit_torch.sh python -m redteamer.static_scan_launcher

# View results
./scripts/fix_streamlit_torch.sh python -m redteamer.results_viewer_launcher
```

### How It Works

The script:
1. Sets the environment variable `REDTEAMER_FIX_STREAMLIT=1` to enable the compatibility fixes
2. Runs the command you provide with this environment variable set

### Manual Alternative

If you prefer, you can also manually set the environment variable:

```bash
export REDTEAMER_FIX_STREAMLIT=1
python -m redteamer.conversational_redteam_launcher
```

### Troubleshooting

If you still encounter issues:
1. Make sure the script has execute permissions: `chmod +x scripts/fix_streamlit_torch.sh`
2. Try running the command with the environment variable set manually
3. Check the logs for any error messages related to the patch application 