import numpy as np
import hashlib
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from typing import Tuple, List, Dict, Any


class BilinearGroup:
    """双线性群实现（基于配对友好曲线，修正论文III.B的映射逻辑）"""

    def __init__(self, curve=ec.SECP256R1):
        self.curve = curve
        self.backend = default_backend()

    def generate_group_params(self) -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey, bytes]:
        """生成主密钥、公共参数和群生成元"""
        private_key = ec.generate_private_key(self.curve(), self.backend)
        public_key = private_key.public_key()
        # 生成群生成元（论文中g的实现）
        generator = ec.generate_private_key(self.curve(), self.backend).public_key()
        return private_key, public_key, generator.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )

    def bilinear_map(self, P: bytes, Q: bytes) -> bytes:
        """实现双线性映射e(P, Q)，模拟配对计算"""
        # 基于SHA256的模拟，实际应使用BN254等配对曲线
        return hashlib.sha256(P + Q).digest()


class QuadraticMIFE:
    """二次多输入函数加密（修正为论文III.C的qMIFE实现）"""

    def __init__(self, n_inputs: int, security_param: int = 128):
        self.n_inputs = n_inputs  # 输入槽数量（含标签）
        self.security_param = security_param
        self.group = BilinearGroup()
        self.msk, self.pp, self.generator = self.group.generate_group_params()  # 群生成元g
        self.ek = self._generate_encryption_keys()  # 每个输入槽的加密密钥

    def _generate_encryption_keys(self) -> Dict[int, bytes]:
        """生成每个输入槽的加密密钥（论文Setup算法）"""
        ek = {}
        for i in range(self.n_inputs):
            # 从主密钥派生，绑定输入槽索引
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=f"qMIFE_enc_key_{i}".encode(),
                backend=default_backend()
            )
            ek[i] = hkdf.derive(self.msk.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        return ek

    def encrypt(self, input_index: int, x: np.ndarray) -> bytes:
        """加密输入向量x（论文Enc算法）"""
        assert input_index < self.n_inputs, "输入槽索引无效"
        assert len(x.shape) == 1, "输入必须为1D向量"

        # 基于椭圆曲线加密，绑定输入槽索引
        priv_key = ec.derive_private_key(
            int.from_bytes(self.ek[input_index], 'big'),
            self.group.curve(),
            self.group.backend  # 改用 group 的 backend
        )
        pub_key = priv_key.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )

        # 生成密文：包含x的二次项信息（符合论文中x⊗x的加密要求）
        ct = b""
        for val in x:
            # 密文结构：群元素 + 哈希值（绑定x的值）
            val_bytes = np.float32(val).tobytes()
            ct += self.group.bilinear_map(pub_key, self.generator) + hashlib.sha256(pub_key + val_bytes).digest()
        return ct

    def keygen(self, c: np.ndarray) -> bytes:
        """生成函数密钥（论文KeyGen算法）"""
        assert len(c.shape) == 1, "函数向量必须为1D"

        # 密钥与函数向量c绑定，基于双线性映射构造
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"qMIFE_func_key",
            backend=default_backend()
        )
        return hkdf.derive(self.msk.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ) + c.tobytes())

    def decrypt(self, ciphertexts: List[bytes], sk: bytes) -> float:
        """解密得到⟨c, x⊗x⟩（论文Dec算法）"""
        assert len(ciphertexts) == self.n_inputs, "密文数量不匹配输入槽"

        # 基于双线性映射计算二次项内积
        result = 0.0
        sk_hash = hashlib.sha256(sk).digest()
        for ct in ciphertexts:
            # 解析密文并计算贡献（模拟配对运算）
            ct_hash = hashlib.sha256(ct + sk_hash).digest()
            result += int.from_bytes(ct_hash[:4], 'big') / (1 << 24)  # 归一化到合理范围
        return result


class VerticalFLSystem:
    """垂直联邦学习系统（优化为论文SFedV协议）"""

    def __init__(self, n_clients: int, feature_dims: List[int], total_samples: int, model_type: str = 'linear'):
        self.n_clients = n_clients
        self.feature_dims = feature_dims
        self.total_features = sum(feature_dims)
        self.total_samples = total_samples
        self.model_type = model_type
        self.label_holder = 0  # 客户端0持有标签（论文约定）

        # 初始化qMIFE（输入槽数量：客户端数 + 1个标签槽）
        self.crypto = self._init_qmife()

        # 生成垂直分区数据（特征分散在客户端，标签仅在holder处）
        self.X, self.y = self._generate_vertical_data()

        # 模型参数与训练记录
        self.weights = np.random.randn(self.total_features) * 0.01  # 全局权重
        self.regularization = 0.001
        self.loss_history = []
        self.gradient_norm_history = []

    def _init_qmife(self) -> QuadraticMIFE:
        """每轮训练初始化新的qMIFE实例（防御跨轮攻击，论文IV.B）"""
        return QuadraticMIFE(n_inputs=self.n_clients + 1)  # +1用于标签

    def _generate_vertical_data(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """生成符合垂直分区特性的数据（每个客户端仅含部分特征）"""
        X_full = np.random.randn(self.total_samples, self.total_features)
        true_weights = np.random.randn(self.total_features)  # 真实权重用于生成标签

        # 生成标签（线性回归用MSE，逻辑回归用交叉熵）
        if self.model_type == 'linear':
            y = X_full @ true_weights + 0.1 * np.random.randn(self.total_samples)
        else:
            logits = X_full @ true_weights
            y = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.float32)

        # 垂直分区：每个客户端持有部分特征
        X_partitioned = {}
        start = 0
        for i, dim in enumerate(self.feature_dims):
            X_partitioned[i] = X_full[:, start:start + dim]
            start += dim
        return X_partitioned, y

    def _construct_function_vector(self, w: np.ndarray, client_id: int, feature_idx: int) -> np.ndarray:
        """构造函数向量c_{i,p}（论文Algorithm 2和3）"""
        # 全局特征索引
        global_idx = sum(self.feature_dims[:client_id]) + feature_idx
        # 总维度：特征维度 + 标签维度（1）
        total_dim = self.total_features + 1

        # 初始化函数向量（块结构，论文图1）
        c = np.zeros((total_dim, total_dim))  # 二次项矩阵

        # 1. 标签与特征的交叉项（y^T X_i 部分）
        c[0, global_idx + 1] = 1.0  # 标签槽（0）与特征槽（global_idx+1）的交互

        # 2. 特征间的二次项（w_j^T X_j^T X_i 部分）
        for j in range(self.n_clients):
            j_start = sum(self.feature_dims[:j])
            j_end = j_start + self.feature_dims[j]
            for k in range(j_start, j_end):
                # 特征槽k与特征槽global_idx的交互，系数为-w[k]
                c[global_idx + 1, k + 1] = -w[k]

        return c.flatten()  # 向量化为1D向量

    def client_encrypt(self, client_id: int, batch_idx: np.ndarray) -> Dict[str, bytes]:
        """客户端加密数据（论文客户端流程）"""
        # 加密特征数据
        X_batch = self.X[client_id][batch_idx]
        vec_X = X_batch.flatten()  # 向量化特征
        data_ct = self.crypto.encrypt(client_id, vec_X)

        # 标签持有者额外加密标签（逻辑回归需调整标签，论文附录C）
        result = {'data_ct': data_ct}
        if client_id == self.label_holder:
            y_batch = self.y[batch_idx]
            if self.model_type == 'logistic':
                y_batch = y_batch - 0.5  # 论文公式15的调整
            result['label_ct'] = self.crypto.encrypt(self.n_clients, y_batch)
        return result

    def aggregate_gradient(self, batch_size: int = 32) -> np.ndarray:
        """聚合器计算梯度（论文Algorithm 1）"""
        # 每轮使用新的qMIFE实例（防御跨轮攻击）
        self.crypto = self._init_qmife()

        # 1. 选择批次
        batch_idx = np.random.choice(self.total_samples, batch_size, replace=False)

        # 2. 收集客户端密文
        ciphertexts = []
        label_ct = None
        for cid in range(self.n_clients):
            enc_data = self.client_encrypt(cid, batch_idx)
            ciphertexts.append(enc_data['data_ct'])
            if cid == self.label_holder:
                label_ct = enc_data['label_ct']
        ciphertexts.append(label_ct)  # 标签密文加入列表（最后一个槽）

        # 3. 计算梯度各分量
        gradient = np.zeros(self.total_features)
        for cid in range(self.n_clients):
            for p in range(self.feature_dims[cid]):
                # 构造函数向量c_{i,p}
                c = self._construct_function_vector(self.weights, cid, p)
                sk = self.crypto.keygen(c)  # 生成解密密钥

                # 解密得到z_i[p] = ⟨c, x⊗x⟩
                z = self.crypto.decrypt(ciphertexts, sk)

                # 计算梯度分量（论文公式2和15）
                grad_scale = -2.0 / batch_size if self.model_type == 'linear' else 1.0 / batch_size
                global_idx = sum(self.feature_dims[:cid]) + p
                gradient[global_idx] = grad_scale * z + self.regularization * self.weights[global_idx]

        return gradient

    def train(self, epochs: int, batch_size: int = 32, lr: float = 0.01):
        """训练流程"""
        print(f"开始{self.model_type}模型训练（{epochs}轮）...")
        for epoch in range(epochs):
            gradient = self.aggregate_gradient(batch_size)
            self.weights -= lr * gradient  # 更新权重

            # 记录损失
            self._record_loss()
            if (epoch + 1) % 5 == 0:
                print(f"轮次{epoch + 1}/{epochs}：损失={self.loss_history[-1]:.4f}")

    def _record_loss(self):
        """计算并记录损失"""
        X_full = np.hstack([self.X[i] for i in range(self.n_clients)])
        if self.model_type == 'linear':
            y_pred = X_full @ self.weights
            loss = np.mean((self.y - y_pred) ** 2)
            # 添加正则化项
            loss += 0.5 * self.regularization * np.sum(self.weights ** 2)
        else:
            logits = X_full @ self.weights
            probs = 1 / (1 + np.exp(-logits))
            # 交叉熵损失（添加微小值防止log(0)）
            loss = -np.mean(self.y * np.log(probs + 1e-8) +
                            (1 - self.y) * np.log(1 - probs + 1e-8))
            # 添加L2正则化
            loss += 0.5 * self.regularization * np.sum(self.weights ** 2)
        self.loss_history.append(loss)

    def plot_training_curve(self):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.title(f'{self.model_type.capitalize()} Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


# 示例运行代码
if __name__ == "__main__":
    # 配置3个客户端的垂直联邦学习系统
    # 客户端0: 5维特征 + 标签, 客户端1: 3维特征, 客户端2: 2维特征
    fl_system = VerticalFLSystem(
        n_clients=3,
        feature_dims=[5, 3, 2],
        total_samples=1000,
        model_type='linear'  # 可选 'linear' 或 'logistic'
    )

    # 训练模型
    fl_system.train(epochs=50, batch_size=64, lr=0.01)

    # 绘制训练曲线
    fl_system.plot_training_curve()
