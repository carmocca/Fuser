import torch
from nvfuser import FusionDefinition, DataType

inputs = [
    torch.randn(16, 128, 3072, device="cuda"),
]


def nvfuser_fusion(fd: FusionDefinition) -> None:
    T0 = fd.from_pytorch(inputs[0])
    T0_slice1 = fd.ops.slice(T0, [0, 0, 0], [16, 128, 1024], [1, 1, 1])
    T0_slice2 = fd.ops.slice(T0, [0, 0, 1024], [16, 128, 2048], [1, 1, 1])
    T0_slice3 = fd.ops.slice(T0, [0, 0, 2048], [16, 128, 3072], [1, 1, 1])
    T1_slice1 = fd.ops.reshape(T0_slice1, [16, 128, 1024], [16, 128, 16, 64])
    T1_slice2 = fd.ops.reshape(T0_slice2, [16, 128, 1024], [16, 128, 16, 64])
    T1_slice3 = fd.ops.reshape(T0_slice3, [16, 128, 1024], [16, 128, 16, 64])
    T2_slice1 = fd.ops.permute(T1_slice1, [0, 2, 1, 3])
    T2_slice2 = fd.ops.permute(T1_slice2, [0, 2, 1, 3])
    T2_slice3 = fd.ops.permute(T1_slice3, [0, 2, 1, 3])
    fd.add_output(T2_slice1)
    fd.add_output(T2_slice2)
    fd.add_output(T2_slice3)


with FusionDefinition() as fd:
    nvfuser_fusion(fd)

outputs = fd.execute(inputs)
for out in outputs:
    print(out.shape)
