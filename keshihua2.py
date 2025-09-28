import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def load_similarity_matrix(file_path):
    """加载相似度矩阵"""
    try:
        similarity_matrix = np.load(file_path)
        print(f"成功加载相似度矩阵，形状: {similarity_matrix.shape}")
        return similarity_matrix
    except Exception as e:
        print(f"加载相似度矩阵失败: {e}")
        return None


def estimate_n_clusters(distance_matrix, max_clusters=10):
    """使用轮廓系数估计最佳聚类数量"""
    best_score = -1
    best_n_clusters = 2

    for n in range(2, min(max_clusters + 1, distance_matrix.shape[0])):
        try:
            # 使用Agglomerative聚类
            model = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage='complete')
            labels = model.fit_predict(distance_matrix)

            # 计算轮廓系数
            if len(set(labels)) > 1:  # 确保有多个簇
                score = silhouette_score(distance_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_n_clusters = n
        except Exception as e:
            continue

    print(f"估计最佳聚类数量: {best_n_clusters} (轮廓系数: {best_score:.4f})")
    return best_n_clusters


def perform_clustering(similarity_matrix, method='agglomerative', n_clusters=None):
    """执行聚类分析

    Args:
        similarity_matrix: 相似度矩阵
        method: 聚类方法，可选 'kmeans', 'agglomerative', 'dbscan'
        n_clusters: 聚类数量，若为None则使用estimate_n_clusters的结果
    """
    # 转换相似度为距离 (假设相似度范围为0-1)
    distance_matrix = (1 - similarity_matrix)*4

    if n_clusters is None:
        n_clusters = estimate_n_clusters(distance_matrix)

    try:
        if method == 'kmeans':
            # KMeans需要欧氏距离，使用多维缩放(MDS)转换
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(distance_matrix)

        elif method == 'agglomerative':
            # 层次聚类，直接使用预计算的距离矩阵
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='complete'
            )
            labels = model.fit_predict(distance_matrix)

        elif method == 'dbscan':
            # DBSCAN，基于密度的聚类
            # 自动计算eps参数 (使用距离矩阵的中位数)
            triu_indices = np.triu_indices(distance_matrix.shape[0], k=1)
            distances = distance_matrix[triu_indices]
            eps = np.median(distances)

            model = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
            labels = model.fit_predict(distance_matrix)

            # 统计聚类结果
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # -1是噪声点
            print(f"DBSCAN聚类结果: {n_clusters}个簇，{np.sum(labels == -1)}个噪声点")

        else:
            print(f"未知的聚类方法: {method}")
            return None, None

        print(f"聚类完成，共{n_clusters}个簇")
        return labels, n_clusters

    except Exception as e:
        print(f"聚类过程出错: {e}")
        return None, None


def print_cluster_members(labels):
    """打印每个簇的成员客户端ID"""
    if labels is None:
        print("请先执行聚类")
        return

    cluster_members = {}
    for client_id, label in enumerate(labels):
        if label not in cluster_members:
            cluster_members[label] = []
        cluster_members[label].append(client_id)

    # 按簇大小排序
    sorted_clusters = sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)

    print("\n===== 聚类结果 =====")
    for label, members in sorted_clusters:
        if label == -1:
            print(f"簇 {label} (噪声点): {len(members)}个客户端")
        else:
            print(f"簇 {label}: {len(members)}个客户端")
            print(f"  客户端ID: {members}")


def visualize_clusters(similarity_matrix, labels, n_clusters):
    """可视化聚类结果"""
    if labels is None:
        print("请先执行聚类")
        return

    plt.figure(figsize=(10, 8))

    # 按聚类标签排序客户端
    sorted_indices = np.argsort(labels)
    sorted_matrix = similarity_matrix[sorted_indices][:, sorted_indices]

    # 绘制热图
    plt.imshow(sorted_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='相似度')
    plt.title(f'客户端相似度矩阵 (按聚类排序, k={n_clusters})')
    plt.xlabel('客户端ID')
    plt.ylabel('客户端ID')

    # 添加聚类分隔线
    cluster_boundaries = np.where(np.diff(labels[sorted_indices]) != 0)[0] + 0.5
    for boundary in cluster_boundaries:
        plt.axhline(y=boundary, color='r', linestyle='--', alpha=0.7)
        plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.7)

    plt.show()


if __name__ == "__main__":
    # 配置参数
    DATASET_NAME = "Cifar"
    SIMILARITY_MATRIX_PATH = f"keshihua/{DATASET_NAME}similarity_matrix.npy"
    CLUSTERING_METHOD = "kmeans"  # 可选: 'kmeans', 'agglomerative', 'dbscan'
    N_CLUSTERS = 5  # 手动指定聚类数量，若为None则自动估计

    # 加载相似度矩阵
    similarity_matrix = load_similarity_matrix(SIMILARITY_MATRIX_PATH)

    if similarity_matrix is not None:
        # 执行聚类
        labels, n_clusters = perform_clustering(
            similarity_matrix,
            method=CLUSTERING_METHOD,
            n_clusters=N_CLUSTERS
        )

        # 打印聚类成员
        print_cluster_members(labels)

        # 可视化聚类结果
        visualize_clusters(similarity_matrix, labels, n_clusters)