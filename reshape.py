import torch
from nvfuser import FusionDefinition, DataType

inputs = [
    torch.randn(16, 128, 3072, device="cuda"),
]


def nvfuser_fusion(fd: FusionDefinition) -> None:
    T0 = fd.from_pytorch(inputs[0])
    T1 = fd.ops.reshape(T0, [16, 128, 3072], [16 * 128, 3072])
    fd.add_output(T1)


with FusionDefinition() as fd:
    nvfuser_fusion(fd)

outputs = fd.execute(inputs)
for out in outputs:
    print(out.shape)
