import os
from safetensors.torch import load_file

def compare_safetensors_weights(path1, path2):
    file1 = os.path.join(path1, "model.safetensors")
    file2 = os.path.join(path2, "model.safetensors")

    if not os.path.exists(file1):
        print(f"文件不存在: {file1}")
        return
    if not os.path.exists(file2):
        print(f"文件不存在: {file2}")
        return

    print(f"正在加载 {file1} ...")
    weights1 = load_file(file1)
    print(f"正在加载 {file2} ...")
    weights2 = load_file(file2)

    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())
    print("keys1:", keys1)
    print("keys2:", keys2)

    if keys1 != keys2:
        print("两个权重文件的参数名不一致。")
        print("仅在第一个文件中的参数：", keys1 - keys2)
        print("仅在第二个文件中的参数：", keys2 - keys1)
        # return

    all_equal = True
    for key in keys1-(keys1-keys2):
        tensor1 = weights1[key]
        tensor2 = weights2[key]
        if tensor1.shape != tensor2.shape:
            print(f"参数 {key} 的形状不一致: {tensor1.shape} vs {tensor2.shape}")
            all_equal = False
            continue
        if not (tensor1 == tensor2).all():
            print(f"参数 {key} 的值不一致")
            all_equal = False

    if all_equal:
        print("两个 model.safetensors 权重完全一致。")
    else:
        print("两个 model.safetensors 权重存在差异。")

if __name__ == "__main__":
    path1 = "/pfs/yangyuanming/code2/models/logs/pickscore/sd3_5-M-1gpu-online-rm/checkpoints/checkpoint-64"
    # path2 = "/pfs/yangyuanming/code2/models/logs/pickscore/sd3_5-M-1gpu-online-rm/checkpoints/checkpoint-64"
    path2 = "/pfs/yangyuanming/code2/models/PickScore_v1/model/yuvalkirstain__PickScore_v1/main/"
    compare_safetensors_weights(path1, path2)
