import matplotlib.pyplot as plt
import numpy as np

from figure.utils_fuction import get_model_scores_compare

# 假设我们有两个维度
dimensions = ["Classification Task", "Generate_Task"]

csv_path = "../evaluation/evaluation_results.csv"
classification_tasks = ["legal_focus_identification", "legal_timeliness", "legal_concept_concise"]
generation_tasks = ["legal_open_qa", "legal_summary", "legal_recitation"]

# 获取三个模型的 score 和 abstention_rate
model1_scores, model1_abstentions = get_model_scores_compare(csv_path, "wisdom", classification_tasks, generation_tasks)
model2_scores, model2_abstentions = get_model_scores_compare(csv_path, "baichuan-7b", classification_tasks, generation_tasks)
model3_scores, model3_abstentions = get_model_scores_compare(csv_path, "new_model", classification_tasks, generation_tasks)

# x 轴上的位置
x = np.arange(len(dimensions))  # 例如 [0, 1]
bar_width = 0.2  # 控制柱子的宽度

# 创建 figure 和 axes
fig, ax = plt.subplots(figsize=(10, 6))

# 每个维度会有 6 个 bar（3 个模型 * 2 个指标）
model1_score_bars = ax.bar(x - bar_width, model1_scores, width=bar_width, color='peachpuff', label='Wisdom_Interrogatory_score')
model1_abstention_bars = ax.bar(x - bar_width, model1_abstentions, width=bar_width, color='sandybrown', bottom=model1_scores, label='Wisdom_Interrogatory_abstention_rate')

model2_score_bars = ax.bar(x, model2_scores, width=bar_width, color='lightskyblue', label='Baichuan-7B_score')
model2_abstention_bars = ax.bar(x, model2_abstentions, width=bar_width, color='steelblue', bottom=model2_scores, label='Baichuan-7B_abstention_rate')

model3_score_bars = ax.bar(x + bar_width, model3_scores, width=bar_width, color='lightgreen', label='NewModel_score')
model3_abstention_bars = ax.bar(x + bar_width, model3_abstentions, width=bar_width, color='darkgreen', bottom=model3_scores, label='NewModel_abstention_rate')

# 设置 x 轴刻度和标签
ax.set_xticks(x)
ax.set_xticklabels(dimensions)

# 设置 y 轴标签
ax.set_ylabel('zero-shot Result(%)')

# 设置图标题
ax.set_title('Wisdom vs Baichuan-7B vs NewModel')

# 显示图例
ax.legend()

# 在柱子顶部标记数值
for bars in [model1_score_bars, model1_abstention_bars, model2_score_bars, model2_abstention_bars, model3_score_bars, model3_abstention_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
