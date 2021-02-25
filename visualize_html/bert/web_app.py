import json
import os
from pathlib import Path

def show_viz(board_path, outputs, metrics_outputs):
    
    Path(outputs).mkdir(parents=True, exist_ok=True)
    Path(metrics_outputs).mkdir(parents=True, exist_ok=True)

    print("\n\nWEB PATH")
    print(board_path)
    print("\n\n")

    metadata = {
        "outputs": [
            {
                "type": "web-app",
                "source": board_path
            }
        ]
    }

    print("\n\nTEST LOGS....")
    print(board_path)
    print("\n\n")

    with open("/mlpipeline-ui-metadata.json", "w") as f:
        json.dump(metadata, f)

