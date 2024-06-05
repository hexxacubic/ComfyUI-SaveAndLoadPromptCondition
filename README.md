
# ComfyUI Nodes: SaveConditioning and LoadConditioning

## SaveConditioning Node

### Description
The `SaveConditioning` node is designed to save conditioning data to binary files. This is useful for storing and reusing conditioning information across different sessions or applications.

### Input Types
- **conditionings**: A list of tuples where each tuple contains text data and a dictionary with a "pooled_output" key.


## LoadConditioning Node

### Description
The `LoadConditioning` node is designed to load conditioning data from binary files. This allows for the reuse of previously saved conditioning information.

### Return Types
- **conditioning**: A list of conditioning data.

