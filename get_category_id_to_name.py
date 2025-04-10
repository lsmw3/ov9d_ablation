import json

# 读取 JSON 文件
with open('datasets/Omni3D/Objectron_test.json', 'r') as f:
    data = json.load(f)

category_map = {}

# 遍历所有图像信息
for annotation in data['annotations']:
    
    category_id = annotation['category_id']
    category_name = annotation['category_name']
    
    # 将新的路径和图片 ID 存储在字典中
    if category_name in category_map:
        assert category_map[category_name] == category_id
    else:
        category_map[category_name] = category_id

# 输出结果
print(category_map)

with open("./objectron_category_id_to_name.json", 'w') as f:
    json.dump(category_map, f)