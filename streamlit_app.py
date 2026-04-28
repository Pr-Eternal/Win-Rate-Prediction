import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# 页面配置
st.set_page_config(page_title="英雄联盟早期胜率预测系统", layout="wide")

# 标题
st.title("英雄联盟前10分钟胜率预测系统")
st.markdown("""
本系统基于高段位（High Diamond）对局前10分钟数据，利用 XGBoost 模型预测蓝方胜率，
并通过 SHAP 值解释关键影响因素。
""")
# --- 1. 加载模型和资源 ---
@st.cache_resource
def load_model_and_resources():
    model_path = 'best_xgb_model.pkl'
    features_path = 'feature_names.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        st.error("错误：未找到 'best_xgb_model.pkl' 或 'feature_names.pkl'。请先运行训练脚本并保存模型。")
        st.stop()
        
    try:
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        st.stop()

model, feature_names = load_model_and_resources()

# --- 2. 侧边栏：选择输入模式 ---
st.sidebar.header("输入方式")
input_mode = st.sidebar.radio("请选择数据输入方式:", ["手动输入", "上传 CSV 文件"])

# 辅助函数：创建数字输入框
def create_input(label, default=0, key=None):
    return st.number_input(label, value=default, step=1, key=key)

# --- 3. 数据处理函数 ---

def calculate_features_from_dict(data_dict):
    """从原始数据字典计算差值特征"""
    features = {}
    
    # 基础差值
    features['goldDiff'] = data_dict['blueTotalGold'] - data_dict['redTotalGold']
    features['experienceDiff'] = data_dict['blueTotalExperience'] - data_dict['redTotalExperience']
    features['csDiff'] = data_dict['blueCSPerMin'] - data_dict['redCSPerMin']
    features['goldPerMinDiff'] = data_dict['blueGoldPerMin'] - data_dict['redGoldPerMin']
    
    # 视野差 (插眼 + 排眼)
    blue_vision_act = data_dict['blueWardsPlaced'] + data_dict['blueWardsDestroyed']
    red_vision_act = data_dict['redWardsPlaced'] + data_dict['redWardsDestroyed']
    features['visionDiff'] = blue_vision_act - red_vision_act
    
    # 事件
    features['blueFirstBlood'] = data_dict['blueFirstBlood']
    
    # 资源差
    features['dragonDiff'] = data_dict['blueDragons'] - data_dict['redDragons']
    features['heraldDiff'] = data_dict['blueHeralds'] - data_dict['redHeralds']
    features['towerDiff'] = data_dict['blueTowersDestroyed'] - data_dict['redTowersDestroyed']
    
    # 转换为 DataFrame 并调整列顺序以匹配训练集
    df_feat = pd.DataFrame([features])
    
    try:
        # 确保列顺序与模型训练时完全一致
        df_feat = df_feat[feature_names]
    except KeyError as e:
        st.error(f"特征不匹配错误: {e}。请检查 feature_names.pkl 是否正确。")
        st.stop()
        
    return df_feat

def process_uploaded_file(uploaded_file):
    """处理上传的 CSV 文件"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # 检查必要的原始列是否存在
        required_cols = [
            'blueTotalGold', 'redTotalGold', 
            'blueTotalExperience', 'redTotalExperience',
            'blueCSPerMin', 'redCSPerMin',
            'blueGoldPerMin', 'redGoldPerMin',
            'blueWardsPlaced', 'blueWardsDestroyed',
            'redWardsPlaced', 'redWardsDestroyed',
            'blueFirstBlood',
            'blueDragons', 'redDragons',
            'blueHeralds', 'redHeralds',
            'blueTowersDestroyed', 'redTowersDestroyed'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV 文件中缺少以下列: {missing_cols}")
            return None
            
        # 为每一行计算特征
        processed_rows = []
        for index, row in df.iterrows():
            data_dict = row.to_dict()
            # 确保数值类型并处理空值
            for k in data_dict:
                try:
                    data_dict[k] = float(data_dict[k]) if pd.notna(data_dict[k]) else 0
                except:
                    data_dict[k] = 0
                    
            feat_df = calculate_features_from_dict(data_dict)
            processed_rows.append(feat_df.iloc[0])
            
        return pd.DataFrame(processed_rows)
        
    except Exception as e:
        st.error(f"处理文件时出错: {e}")
        return None

# --- 4. 主应用程序逻辑 ---

if input_mode == "手动输入":
    st.header("手动数据录入")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("蓝方数据 (Blue Side)")
        b_gold = create_input("总金币 (Total Gold)", 15000, key="b_gold")
        b_exp = create_input("总经验 (Total Exp)", 15000, key="b_exp")
        b_cs_min = create_input("每分钟补刀 (CS Per Min)", 20, key="b_cs_min")
        b_gpm = create_input("每分钟金币 (Gold Per Min)", 1500, key="b_gpm")
        b_wards_placed = create_input("插眼数 (Wards Placed)", 20, key="b_wp")
        b_wards_destroyed = create_input("排眼数 (Wards Destroyed)", 5, key="b_wd")
        b_first_blood = st.selectbox("是否获得一血 (First Blood)?", [0, 1], index=0, key="b_fb")
        b_dragons = create_input("小龙数 (Dragons)", 0, key="b_d")
        b_heralds = create_input("先锋数 (Heralds)", 0, key="b_h")
        b_towers = create_input("推塔数 (Towers Destroyed)", 0, key="b_t")

    with col2:
        st.subheader("红方数据 (Red Side)")
        r_gold = create_input("总金币 (Total Gold)", 15000, key="r_gold")
        r_exp = create_input("总经验 (Total Exp)", 15000, key="r_exp")
        r_cs_min = create_input("每分钟补刀 (CS Per Min)", 20, key="r_cs_min")
        r_gpm = create_input("每分钟金币 (Gold Per Min)", 1500, key="r_gpm")
        r_wards_placed = create_input("插眼数 (Wards Placed)", 20, key="r_wp")
        r_wards_destroyed = create_input("排眼数 (Wards Destroyed)", 5, key="r_wd")
        r_dragons = create_input("小龙数 (Dragons)", 0, key="r_d")
        r_heralds = create_input("先锋数 (Heralds)", 0, key="r_h")
        r_towers = create_input("推塔数 (Towers Destroyed)", 0, key="r_t")

    # 构建输入字典
    input_data = {
        'blueTotalGold': b_gold,
        'blueTotalExperience': b_exp,
        'blueCSPerMin': b_cs_min,
        'blueGoldPerMin': b_gpm,
        'blueWardsPlaced': b_wards_placed,
        'blueWardsDestroyed': b_wards_destroyed,
        'blueFirstBlood': b_first_blood,
        'blueDragons': b_dragons,
        'blueHeralds': b_heralds,
        'blueTowersDestroyed': b_towers,
        
        'redTotalGold': r_gold,
        'redTotalExperience': r_exp,
        'redCSPerMin': r_cs_min,
        'redGoldPerMin': r_gpm,
        'redWardsPlaced': r_wards_placed,
        'redWardsDestroyed': r_wards_destroyed,
        'redDragons': r_dragons,
        'redHeralds': r_heralds,
        'redTowersDestroyed': r_towers,
    }
    
    processed_features = calculate_features_from_dict(input_data)
    
else: # 上传 CSV 模式
    st.header("上传 CSV 文件")
    st.markdown("请上传包含以下原始列名的 CSV 文件：")
    st.code(", ".join(['blueTotalGold', 'redTotalGold', 'blueTotalExperience', 'redTotalExperience', 
                       'blueCSPerMin', 'redCSPerMin', 'blueGoldPerMin', 'redGoldPerMin',
                       'blueWardsPlaced', 'blueWardsDestroyed', 'redWardsPlaced', 'redWardsDestroyed',
                       'blueFirstBlood', 'blueDragons', 'redDragons', 'blueHeralds', 'redHeralds',
                       'blueTowersDestroyed', 'redTowersDestroyed']))
    
    uploaded_file = st.file_uploader("选择 CSV 文件", type="csv")
    
    if uploaded_file is not None:
        processed_features = process_uploaded_file(uploaded_file)
        if processed_features is not None:
            st.success(f"成功处理 {len(processed_features)} 条记录。")
            # 如果有多条记录，允许用户选择查看哪一条的详细分析
            if len(processed_features) > 1:
                record_idx = st.slider("选择要分析的记录索引", 0, len(processed_features)-1, 0)
                current_record = processed_features.iloc[record_idx:record_idx+1]
            else:
                current_record = processed_features
        else:
            current_record = None
    else:
        current_record = None

# --- 5. 预测结果与可视化 ---

if input_mode == "手动输入" or (input_mode == "上传 CSV 文件" and 'current_record' in locals() and current_record is not None):
    
    if input_mode == "上传 CSV 文件":
        features_to_predict = current_record
    else:
        features_to_predict = processed_features
        
    # 预测
    prob_blue_win = model.predict_proba(features_to_predict)[0][1]
    
    st.divider()
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.subheader("预测结果")
        if prob_blue_win > 0.5:
            st.metric("蓝方胜率", f"{prob_blue_win:.2%}", delta="优势")
        else:
            st.metric("蓝方胜率", f"{prob_blue_win:.2%}", delta="-劣势", delta_color="inverse")
            
    with col_res2:
        st.subheader("关键数据概览")
        gold_diff = features_to_predict['goldDiff'].values[0]
        exp_diff = features_to_predict['experienceDiff'].values[0]
                # 使用 .0f 格式化浮点数为整数显示，并保留正负号
        st.metric("经济差 (蓝-红)", f"{gold_diff:+.0f}")
        st.metric("经验差 (蓝-红)", f"{exp_diff:+.0f}")

    # SHAP 可视化
    st.divider()
    st.subheader("SHAP 可解释性分析")
    st.markdown("红色条形表示推动蓝方胜率上升的因素，蓝色条形表示拉低蓝方胜率的因素。")
    
    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_to_predict)
    
    # 处理二分类输出
    if isinstance(shap_values, list):
        shap_val_single = shap_values[1][0] 
    else:
        shap_val_single = shap_values[0]
        
    # 绘制 Force Plot
    try:
        fig_force = shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
            shap_val_single, 
            features_to_predict.iloc[0], 
            matplotlib=True, 
            show=False
        )
        st.pyplot(fig_force)
    except Exception as e:
        st.warning(f"无法生成 Force Plot: {e}")

    # 绘制 Top 5 特征贡献条形图
    st.markdown("####特征贡献条形图")
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_val_single
    }).sort_values(by='SHAP Value', key=abs, ascending=False)
    
    top_5 = shap_df.head(5)
    
    fig_bar, ax = plt.subplots(figsize=(10, 6))
    # 红色代表正向贡献，蓝色代表负向贡献
    colors = ['#FF4B4B' if x > 0 else '#1E88E5' for x in top_5['SHAP Value']] 
    bars = ax.barh(top_5['Feature'], top_5['SHAP Value'], color=colors)
    ax.set_xlabel('SHAP Value (Impact on Blue Win Probability)')
    ax.set_title('Top 5 Contributing Factors for This Match')
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        label_x_pos = bar.get_x() + width / 2
        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, 
                f'{width:.4f}', va='center', ha='center', color='white' if abs(width) > 0.1 else 'black')
                
    st.pyplot(fig_bar)

else:
    if input_mode == "上传 CSV 文件":
        st.info("请上传有效的 CSV 文件以查看预测结果。")
    else:
        st.info("请在左侧或上方调整参数，然后点击“开始预测”按钮（如果有）或直接查看自动更新的结果。")