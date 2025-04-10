import json

# 读取 JSON 文件
with open('datasets/Omni3D/Objectron_test.json', 'r') as f:
    data = json.load(f)

# 创建一个字典用于存储路径到图像 ID 的映射
image_map = {}

# 遍历所有图像信息
for image in data['images']:
    # 提取每张图片的文件路径
    file_path = image['file_path']
    
    # 根据文件路径格式化成所需的路径
    # 格式类似于 "bike/batch-0/6/frame000216"
    # 提取文件名，例如 "bike_batch_4_38_0000000.jpg"
    file_name = file_path.split('/')[-1]  # 获取文件名
    # 防止cereal_box干扰
    file_name = file_name.replace("cereal_box", "cerealbox")
    
    # 按照 "_", 分割文件名以提取相关信息
    parts = file_name.split('_')  # 按下划线分割
    
    # 构造新的路径格式，假设 "batch-0" 是批次编号， "frame000216" 是帧号
    batch = f"batch-{int(parts[2])}"  # parts[2] 是 batch 数字
    frame = f"frame{int(parts[3]):06d}"  # parts[3] 是帧号，例如 "0000000" -> "000216"
    
    # 创建新的路径格式
    new_path = f"{parts[0]}/{batch}/{parts[3]}/{frame}"
    new_path = new_path.replace("cerealbox", "cereal_box")

    # 将新的路径和图片 ID 存储在字典中
    image_map[new_path] = image['id']

# 输出结果
print(image_map)

with open("./objectron_image_path_to_id.json", 'w') as f:
    json.dump(image_map, f)