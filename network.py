import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取网络数据文件
grn_data = pd.read_csv('grn_output_parallel_with_stage.tsv', sep='\t')

# 提取目标基因列表
genes_of_interest_up = ['NDUFA11', 'SPINK2B', 'FAM83A']
genes_of_interest_down = ['RASL11A', 'KEH36_t03', 'ZEB2']

# 过滤出相关基因的子图
filtered_grn = grn_data[
    (grn_data['Source'].isin(genes_of_interest_up + genes_of_interest_down)) |
    (grn_data['Target'].isin(genes_of_interest_up + genes_of_interest_down))
]

# 创建图
G = nx.from_pandas_edgelist(filtered_grn, 'Source', 'Target', ['Weight'])

# 设置度数阈值
degree_threshold = 2

# 计算每个节点的度数
node_degrees = dict(G.degree())

# 筛选度数大于 degree_threshold 的节点
high_degree_nodes = [node for node, degree in node_degrees.items() if degree > degree_threshold]

# 过滤出度数大于阈值的节点及其边
filtered_nodes = set(high_degree_nodes)
filtered_edges = [(u, v) for u, v in G.edges() if u in filtered_nodes and v in filtered_nodes]

# 创建新图，只包含度数大于阈值的节点和边
G_filtered = G.edge_subgraph(filtered_edges).copy()

# 使用 spring_layout，减少迭代次数以避免节点过于分散
# 调整k参数来控制节点的密度
pos = nx.spring_layout(G_filtered, k=0.5, iterations=100)  # 调整k，增加迭代次数

# 定义节点颜色：上调基因为浅紫色，下调基因为浅绿色，其他为浅红色
node_colors = []
group = []  # 用于存储节点组

for node in G_filtered.nodes():
    if node in genes_of_interest_up:
        node_colors.append((153/255, 153/255, 255/255))  # 上调基因为浅紫色
        group.append(1)  # 为上调基因指定分组
    elif node in genes_of_interest_down:
        node_colors.append((153/255, 255/255, 153/255))  # 下调基因为浅绿色
        group.append(2)  # 为下调基因指定分组
    else:
        node_colors.append((255/255, 153/255, 153/255))  # 其他基因为浅红色
        group.append(3)  # 为其他基因指定分组

# 绘制节点（不同颜色）
plt.figure(figsize=(10, 8))  # 增加图形大小

# 绘制节点，使用不同颜色的节点，确保相同颜色的节点尽量在一起
nx.draw_networkx_nodes(G_filtered, pos, node_size=1500, node_color=node_colors)

# 绘制边（根据权重调整边的粗细）
nx.draw_networkx_edges(G_filtered, pos, width=1.0, alpha=0.7, edge_color='gray')

# 绘制标签（调整字体大小）
nx.draw_networkx_labels(G_filtered, pos, font_size=10, font_weight='bold')

# 设置标题并去除坐标轴
plt.title("Filtered Gene Regulatory Network (Degree > 2)")
plt.axis('off')  # 去掉坐标轴

# 显示图形
plt.show()
