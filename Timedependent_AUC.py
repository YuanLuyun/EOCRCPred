import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import matplotlib.backends.backend_pdf
import warnings
from pycox.models import CoxPH as PycoxCoxPH
from sksurv.metrics import cumulative_dynamic_auc
from xgbse import XGBSEKaplanNeighbors
from sklearn.metrics import roc_auc_score
# 忽略警告信息
warnings.filterwarnings("ignore")

# ==================== 数据预处理部分 ====================
# 加载数据并去掉 'Patient_ID' 列
data = pd.read_csv('data_encoded7408.csv')
if 'Patient_ID' in data.columns:
    data = data.drop(columns=['Patient_ID'])  # 删除 Patient_ID 列

# 构建生存数据
X = data.drop(columns=['OS_month', 'Survival_status'])
y = Surv.from_dataframe('Survival_status', 'OS_month', data)
y_time = data['OS_month'].values
y_event = data['Survival_status'].values

# 自动识别数值型特征和独热编码特征
numerical_features = ['Age', 'Grade', 'TNM_Stage', 'T', 'N', 'CEA', 'No.of_resected_LNs', 'Tumor_Deposits',
                      'Tumor_size', 'Median_household_income']
categorical_features = [col for col in X.columns if col not in numerical_features]

# 定义数值型特征的归一化处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', 'passthrough', categorical_features)
    ])

# 应用预处理
X_processed = preprocessor.fit_transform(X)

# 按照 8:2 划分训练集与测试集
X_train, X_test, y_train_time, y_test_time, y_train_event, y_test_event = train_test_split(
    X_processed, y_time, y_event, test_size=0.2, random_state=42
)

# 构建生存数据对象
y_train = Surv.from_arrays(event=y_train_event == 1, time=y_train_time)
y_test = Surv.from_arrays(event=y_test_event == 1, time=y_test_time)

# ==================== 外部验证集数据处理 ====================
external_data = pd.read_csv('external_validation_set.csv')
if 'Patient_ID' in external_data.columns:
    external_data = external_data.drop(columns=['Patient_ID'])

X_external = external_data.drop(columns=['OS_month', 'Survival_status'])
y_external_time = external_data['OS_month'].values
y_external_event = external_data['Survival_status'].values
y_external = Surv.from_arrays(event=y_external_event == 1, time=y_external_time)
X_external_processed = preprocessor.transform(X_external)

# ==================== DeepSurv 模型定义和训练 ====================
class CoxPHModel(nn.Module):
    def __init__(self, in_features, num_nodes, dropout):
        super(CoxPHModel, self).__init__()
        layers = []
        prev_nodes = in_features
        for nodes in num_nodes:
            layers.append(nn.Linear(prev_nodes, nodes))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_nodes = nodes
        layers.append(nn.Linear(prev_nodes, 1))  # 输出层
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 初始化DeepSurv网络并应用权重初始化
net = CoxPHModel(in_features=X_train.shape[1], num_nodes=[16, 16], dropout=0.1)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net.apply(initialize_weights)

# 设置训练参数
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

# 定义 EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

early_stopping = EarlyStopping(patience=10)

# 使用 Pycox 的 CoxPH 模型包装 DeepSurv
modelcoxPHdeepsurv = PycoxCoxPH(net, optimizer)

# 准备训练数据集
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_time_tensor = torch.tensor(y_train_time, dtype=torch.float32)
y_train_event_tensor = torch.tensor(y_train_event, dtype=torch.float32)
train_ds = TensorDataset(X_train_tensor, y_train_time_tensor, y_train_event_tensor)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

# 训练DeepSurv模型
epochs = 100
for epoch in range(epochs):
    modelcoxPHdeepsurv.net.train()
    total_loss = 0
    for x_batch, y_time_batch, y_event_batch in train_dl:
        optimizer.zero_grad()
        output = modelcoxPHdeepsurv.net(x_batch)
        # Pycox 的 loss 函数
        loss = modelcoxPHdeepsurv.loss(output, y_time_batch, y_event_batch)
        if torch.isnan(loss):
            print(f"Epoch {epoch + 1}/{epochs}, Loss: nan detected")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelcoxPHdeepsurv.net.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_dl)
    print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}')
    scheduler.step(average_loss)
    if early_stopping.step(average_loss):
        print("Early stopping triggered.")
        break

# ==================== 定义模型 ====================
models = {
    'RSF': RandomSurvivalForest(n_estimators=1625, max_depth=6, min_samples_split=2, min_samples_leaf=4, n_jobs=-1, random_state=42),
    'CoxPH': CoxPHSurvivalAnalysis(),
    'S-SVM': FastKernelSurvivalSVM(kernel="rbf", alpha=0.01, gamma=0.01, tol=1e-4, max_iter=1000, random_state=42),
    'GBSA': GradientBoostingSurvivalAnalysis(),
    'DeepSurv': net  # 包含 DeepSurv 模型
}

# ==================== 计算风险评分（不包括 XGBSE） ====================
def compute_risk_scores(model, X, model_name):
    """根据模型类型提取风险得分"""
    if model_name in ['RSF', 'CoxPH', 'S-SVM', 'GBSA']:
        return model.predict(X)  # 一维数组，返回风险评分
    elif model_name == 'DeepSurv':
        with torch.no_grad():
            modelcoxPHdeepsurv.net.eval()
            risk_scores = modelcoxPHdeepsurv.net(torch.tensor(X, dtype=torch.float32)).numpy().squeeze()
        return risk_scores  # 一维数组，返回风险评分
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
# ==================== 训练模型 ====================
model_risk_scores_train = {}
model_risk_scores_test = {}
model_risk_scores_external = {}

for name, model in models.items():
    if name != 'DeepSurv':  # DeepSurv 模型已经训练好，其他模型需要训练
        model.fit(X_train, y_train)

    # 计算风险得分
    risk_train = compute_risk_scores(model, X_train, name)
    risk_test = compute_risk_scores(model, X_test, name)
    risk_external = compute_risk_scores(model, X_external_processed, name)

    model_risk_scores_train[name] = risk_train
    model_risk_scores_test[name] = risk_test
    model_risk_scores_external[name] = risk_external

time_horizons = np.arange(6, 121, 6)
def plot_time_dependent_auc(y_train, y_true, model_risk_scores, time_horizons, ax, model_names, dataset_name, auc_results, row_idx):
    """绘制时间依赖AUC曲线，并将mean_auc添加到图例中"""
    for model_name in model_names:
        y_score = model_risk_scores[model_name]

        try:
            # 调用 cumulative_dynamic_auc 函数计算 AUC
            cumulative_auc, mean_auc = cumulative_dynamic_auc(y_train, y_true, y_score, time_horizons)

            # 保存 AUC 到 auc_results 以导出到 CSV
            for t, auc_val in zip(time_horizons, cumulative_auc):
                auc_results.append([dataset_name, model_name, t, auc_val])

            # 添加 mean_auc 到 auc_results
            auc_results.append([dataset_name, model_name, 'mean AUC',mean_auc])

            # 绘制 AUC 曲线，包含 mean_auc 的图例
            ax.plot(time_horizons, cumulative_auc, label=f'{model_name} (mean AUC: {mean_auc:.2f})')

        except Exception as e:
            print(f"Error computing AUC for model {model_name}: {e}")
            continue
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('AUC')
    ax.set_title(f'{dataset_name}')
    ax.legend(loc='best', prop={'size': 10}, framealpha=0.3)

    ax.text(-0.15, 1.15, chr(65 + row_idx), transform=ax.transAxes, fontsize=16, fontweight='bold')
    ax.set_xlim(0, max(time_horizons) + 12)
    ax.set_xticks(np.arange(0, max(time_horizons) + 1, 12))
    ax.set_ylim(0.5, 0.95)  # 手动设置Y轴范围
    ax.set_yticks(np.arange(0.5, 1, 0.05))  # 每0.1设置一个刻度
    ax.grid(True)

def compute_and_plot_all_models_auc(model_risk_scores_list, y_trues, datasets, time_horizons, y_train, auc_results):
    """计算并绘制所有模型的时间依赖型 AUC 曲线"""
    fig_auc, axes_auc_plot = plt.subplots(1, 3, figsize=(20, 5))

    # 遍历每个数据集
    for row_idx, (y_true, model_risk_scores_current, dataset_name) in enumerate(zip(y_trues, model_risk_scores_list, datasets)):
        ax = axes_auc_plot[row_idx]
        plot_time_dependent_auc(
            y_train,            # 用于定义风险集
            y_true,             # 当前数据集的生存对象
            model_risk_scores_current,  # 当前数据集的风险得分
            time_horizons,     # 时间点
            ax,                # 当前子图
            model_risk_scores_current.keys(),  # 模型名称
            dataset_name,      # 数据集名称
            auc_results,       # 结果保存列表
            row_idx            # 当前子图索引
        )
    return fig_auc, axes_auc_plot  # 返回 figure 和 ax 用于后续添加 XGBSE 曲线

# ==================== 计算 XGBSE AUC 函数 ====================
def compute_xgbse_auc_at_times(model, X, y_true, time_horizons):
    survival_predictions = model.predict(X, time_bins=time_horizons)
    aucs = []

    # 遍历每个时间点计算 AUC
    for idx, time_point in enumerate(time_horizons):
        # 取出对应时间点的生存概率
        risk_scores_at_time = 1 - survival_predictions.iloc[:, idx]  # 将生存概率转为风险评分（生存概率越低，风险越高）
        # 计算 AUC
        auc = roc_auc_score(y_true['event'], risk_scores_at_time)  # y_true['event'] 是二分类标签
        aucs.append(auc)
    return aucs

# ==================== 计算 XGBSE 模型的 AUC ====================
xgbse_model = XGBSEKaplanNeighbors()
xgbse_model.fit(X_train, y_train)
# 计算 XGBSE 在每个时间点的 AUC
xgbse_auc_train = compute_xgbse_auc_at_times(xgbse_model, X_train, y_train, time_horizons)
xgbse_auc_test = compute_xgbse_auc_at_times(xgbse_model, X_test, y_test, time_horizons)
xgbse_auc_external = compute_xgbse_auc_at_times(xgbse_model, X_external, y_external, time_horizons)

print("Training AUCs:", xgbse_auc_train)
print("Test AUCs:", xgbse_auc_test)
print("Validation AUCs:", xgbse_auc_external)
# ==================== 绘制并添加 XGBSE 的 AUC 曲线 ====================
def add_xgbse_auc_to_existing_plot(axes_auc_plot, time_horizons, xgbse_auc_list):
    """将 XGBSE 模型的 AUC 曲线添加到现有图像上"""
    datasets = ['Training cohort', 'Test cohort', 'External validation cohort']
    for idx, (ax, xgbse_auc, dataset_name) in enumerate(zip(axes_auc_plot, xgbse_auc_list, datasets)):
        mean_auc = np.mean(xgbse_auc)
        ax.plot(time_horizons, xgbse_auc, label=f'XGBSE (mean AUC: {mean_auc:.2f})')
        ax.legend(loc='best', prop={'size': 10}, framealpha=0.5, ncol=2)

# ==================== 主函数 ====================
def main():
    auc_results = []

    # 数据集名称和对应的风险得分字典
    datasets = ['Training cohort', 'Test cohort', 'External validation cohort']
    model_risk_scores_list = [model_risk_scores_train, model_risk_scores_test, model_risk_scores_external]
    y_trues = [y_train, y_test, y_external]

    # 计算并绘制所有模型的 AUC 曲线（不包括 XGBSE）
    fig_auc, axes_auc_plot = compute_and_plot_all_models_auc(
        model_risk_scores_list, y_trues, datasets, time_horizons, y_train, auc_results
    )

    # 将 XGBSE 模型的 AUC 曲线添加到现有图像上
    xgbse_auc_list = [xgbse_auc_train, xgbse_auc_test, xgbse_auc_external]
    add_xgbse_auc_to_existing_plot(
        axes_auc_plot, time_horizons, xgbse_auc_list
    )

    # 将 XGBSE 的 AUC 结果添加到 auc_results
    for dataset_name, xgbse_auc in zip(datasets, xgbse_auc_list):
        mean_auc = np.mean(xgbse_auc)
        for t, auc_val in zip(time_horizons, xgbse_auc):
            auc_results.append([dataset_name, 'XGBSE', t, auc_val])
        auc_results.append([dataset_name, 'XGBSE', 'Mean AUC', mean_auc])

    # 保存包含所有模型和 XGBSE 模型的 AUC 曲线到同一个 PDF 文件中
    pdf_filename_auc = "Fig.4.pdf"
    fig_auc.tight_layout(pad=4.0)
    pdf_auc = matplotlib.backends.backend_pdf.PdfPages(pdf_filename_auc)
    pdf_auc.savefig(fig_auc)
    pdf_auc.close()

    print(f"包含所有模型和 XGBSE 的时间依赖AUC曲线已经保存到 {pdf_filename_auc}")

    # 导出 AUC 结果到 CSV 文件
    df_auc_results = pd.DataFrame(auc_results, columns=['Dataset', 'Model', 'Times', 'AUC Scores'])
    df_auc_results.to_csv('Time dependent AUC.csv', index=False)
    print(f"AUC结果已保存到 Time dependent AUC.csv")

# 执行主函数
if __name__ == "__main__":
    main()





