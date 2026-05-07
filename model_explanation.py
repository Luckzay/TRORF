"""
模型解释脚本
用于解释和可视化XGBoost浓度预测模型
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
import os
from utils.feature_extractor import extract_color_features
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelExplainer:
    """模型解释器类"""
    
    def __init__(self, model_path="../models/latest.pkl"):
        """
        初始化模型解释器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self):
        """获取特征名称列表"""
        return [
            # RGB特征
            'R_mean', 'R_std', 'R_median', 'R_max', 'R_min', 'R_p25', 'R_p75',
            'G_mean', 'G_std', 'G_median', 'G_max', 'G_min', 'G_p25', 'G_p75',
            'B_mean', 'B_std', 'B_median', 'B_max', 'B_min', 'B_p25', 'B_p75',
            
            # HSV特征
            'H_mean', 'H_std', 'H_median', 'H_max', 'H_min', 'H_p25', 'H_p75',
            'S_mean', 'S_std', 'S_median', 'S_max', 'S_min', 'S_p25', 'S_p75',
            'V_mean', 'V_std', 'V_median', 'V_max', 'V_min', 'V_p25', 'V_p75',
            
            # LAB特征
            'L_mean', 'L_std', 'L_median', 'L_max', 'L_min', 'L_p25', 'L_p75',
            'A_mean', 'A_std', 'A_median', 'A_max', 'A_min', 'A_p25', 'A_p75',
            'B_mean', 'B_std', 'B_median', 'B_max', 'B_min', 'B_p25', 'B_p75',
            
            # LUV特征
            'LUV_L_mean', 'LUV_L_std', 'LUV_L_median', 'LUV_L_max', 'LUV_L_min', 'LUV_L_p25', 'LUV_L_p75',
            'LUV_U_mean', 'LUV_U_std', 'LUV_U_median', 'LUV_U_max', 'LUV_U_min', 'LUV_U_p25', 'LUV_U_p75',
            'LUV_V_mean', 'LUV_V_std', 'LUV_V_median', 'LUV_V_max', 'LUV_V_min', 'LUV_V_p25', 'LUV_V_p75',
            
            # YCrCb特征
            'Y_mean', 'Y_std', 'Y_median', 'Y_max', 'Y_min', 'Y_p25', 'Y_p75',
            'Cr_mean', 'Cr_std', 'Cr_median', 'Cr_max', 'Cr_min', 'Cr_p25', 'Cr_p75',
            'Cb_mean', 'Cb_std', 'Cb_median', 'Cb_max', 'Cb_min', 'Cb_p25', 'Cb_p75',
            
            # 额外特征
            'Saturation', 'Color_Uniformity'
        ]
    
    def load_model(self):
        """加载模型"""
        try:
            self.model = joblib.load(self.model_path)
            print(f"✅ 成功加载模型: {self.model_path}")
            print(f"📊 模型类型: {type(self.model).__name__}")
            return True
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def get_model_info(self):
        """获取模型基本信息"""
        if self.model is None:
            print("请先加载模型!")
            return
            
        print("=" * 50)
        print("📊 模型基本信息")
        print("=" * 50)
        print(f"模型类型: {type(self.model).__name__}")
        print(f"特征数量: {len(self.feature_names)}")
        print(f"模型文件大小: {os.path.getsize(self.model_path) / 1024:.1f} KB")
        
        # 如果是XGBoost模型，显示更多参数
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params()
            print("\n🔧 主要超参数:")
            important_params = ['n_estimators', 'max_depth', 'learning_rate', 
                              'subsample', 'colsample_bytree']
            for param in important_params:
                if param in params:
                    print(f"  {param}: {params[param]}")
    
    def analyze_feature_importance(self, top_n=20):
        """分析特征重要性"""
        if self.model is None:
            print("请先加载模型!")
            return
            
        print("\n" + "=" * 50)
        print("📈 特征重要性分析")
        print("=" * 50)
        
        # 获取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            print("该模型不支持特征重要性分析")
            return
            
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 前 {top_n} 个最重要特征:")
        print("-" * 40)
        for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<20} : {row['importance']:.6f}")
        
        # 可视化特征重要性
        self._plot_feature_importance(feature_importance_df.head(15))
        
        return feature_importance_df
    
    def _plot_feature_importance(self, df):
        """绘制特征重要性图"""
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(df)), df['importance'], color='skyblue')
        plt.yticks(range(len(df)), df['feature'])
        plt.xlabel('特征重要性')
        plt.title('Top 15 特征重要性')
        plt.gca().invert_yaxis()
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("💾 特征重要性图已保存为: feature_importance.png")
        plt.show()
    
    def explain_prediction_example(self, sample_features=None):
        """解释单个预测示例"""
        if self.model is None:
            print("请先加载模型!")
            return
            
        print("\n" + "=" * 50)
        print("🔍 单个预测解释")
        print("=" * 50)
        
        # 如果没有提供样本特征，生成一个示例
        if sample_features is None:
            # 生成合理的特征值范围
            sample_features = np.random.uniform(0, 255, len(self.feature_names))
            # 对于某些特征设置合理范围
            sample_features[0:7] = np.random.uniform(100, 200, 7)  # R通道
            sample_features[7:14] = np.random.uniform(80, 180, 7)   # G通道
            sample_features[14:21] = np.random.uniform(60, 160, 7)  # B通道
        
        # 进行预测
        prediction = self.model.predict([sample_features])[0]
        print(f"📊 输入特征维度: {len(sample_features)}")
        print(f"🎯 预测浓度值: {prediction:.6f}")
        
        # SHAP解释
        self._shap_explanation(sample_features)
    
    def _shap_explanation(self, sample_features):
        """使用SHAP进行解释"""
        try:
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values([sample_features])
            
            print(f"\n🔮 SHAP特征贡献分析:")
            print("-" * 30)
            
            # 创建特征贡献DataFrame
            contribution_df = pd.DataFrame({
                'feature': self.feature_names,
                'feature_value': sample_features,
                'shap_contribution': shap_values[0]
            }).sort_values('shap_contribution', key=abs, ascending=False)
            
            print("前10个最具影响力的特征:")
            for i, (_, row) in enumerate(contribution_df.head(10).iterrows()):
                direction = "↑" if row['shap_contribution'] > 0 else "↓"
                print(f"{i+1:2d}. {row['feature']:<20} "
                      f"[值:{row['feature_value']:.1f}] "
                      f"{direction} 贡献: {row['shap_contribution']:.4f}")
            
            # 可视化SHAP值
            self._plot_shap_waterfall(sample_features, shap_values[0])
            
        except Exception as e:
            print(f"⚠️ SHAP分析失败: {e}")
            print("可能需要安装shap库: pip install shap")
    
    def _plot_shap_waterfall(self, features, shap_values):
        """绘制SHAP瀑布图"""
        try:
            # 创建瀑布图数据
            feature_contributions = list(zip(self.feature_names, features, shap_values))
            feature_contributions.sort(key=lambda x: abs(x[2]), reverse=True)
            
            top_features = feature_contributions[:10]
            
            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(top_features))
            contributions = [fc[2] for fc in top_features]
            
            colors = ['red' if c > 0 else 'blue' for c in contributions]
            bars = plt.barh(y_pos, contributions, color=colors, alpha=0.7)
            
            plt.yticks(y_pos, [fc[0][:15] for fc in top_features])
            plt.xlabel('SHAP贡献值')
            plt.title('Top 10 特征SHAP贡献')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # 添加数值标签
            for i, (bar, contrib) in enumerate(zip(bars, contributions)):
                plt.text(contrib, bar.get_y() + bar.get_height()/2, 
                        f'{contrib:.3f}', ha='left' if contrib > 0 else 'right', 
                        va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig('shap_explanation.png', dpi=300, bbox_inches='tight')
            print("💾 SHAP解释图已保存为: shap_explanation.png")
            plt.show()
            
        except Exception as e:
            print(f"⚠️ 绘制SHAP图失败: {e}")
    
    def model_summary_report(self):
        """生成完整的模型总结报告"""
        if not self.load_model():
            return
            
        print("\n" + "=" * 60)
        print("📋 XGBOOST浓度预测模型完整分析报告")
        print("=" * 60)
        
        # 1. 模型基本信息
        self.get_model_info()
        
        # 2. 特征重要性分析
        feature_importance_df = self.analyze_feature_importance(top_n=15)
        
        # 3. 预测示例解释
        self.explain_prediction_example()
        
        # 4. 模型特点总结
        self._summarize_model_characteristics(feature_importance_df)
        
        print("\n" + "=" * 60)
        print("✅ 模型分析完成!")
        print("=" * 60)
    
    def _summarize_model_characteristics(self, feature_importance_df):
        """总结模型特点"""
        print("\n" + "=" * 50)
        print("🧠 模型特点分析")
        print("=" * 50)
        
        # 分析主要特征类型
        rgb_features = [f for f in feature_importance_df['feature'].head(10) if f.startswith(('R_', 'G_', 'B_'))]
        hsv_features = [f for f in feature_importance_df['feature'].head(10) if f.startswith(('H_', 'S_', 'V_'))]
        lab_features = [f for f in feature_importance_df['feature'].head(10) if f.startswith(('L_', 'A_', 'B_'))]
        
        print(f"\n🎨 颜色空间重要性分析:")
        print(f"  RGB特征占比: {len(rgb_features)/10*100:.1f}%")
        print(f"  HSV特征占比: {len(hsv_features)/10*100:.1f}%")
        print(f"  LAB特征占比: {len(lab_features)/10*100:.1f}%")
        
        # 分析统计量重要性
        mean_features = [f for f in feature_importance_df['feature'].head(15) if '_mean' in f]
        std_features = [f for f in feature_importance_df['feature'].head(15) if '_std' in f]
        other_features = [f for f in feature_importance_df['feature'].head(15) if '_' in f and not any(x in f for x in ['_mean', '_std'])]
        
        print(f"\n📊 统计量重要性分析:")
        print(f"  均值特征: {len(mean_features)} 个")
        print(f"  标准差特征: {len(std_features)} 个")
        print(f"  其他统计量: {len(other_features)} 个")
        
        # 关键发现
        print(f"\n🔍 关键发现:")
        top_feature = feature_importance_df.iloc[0]['feature']
        print(f"  • 最重要特征: {top_feature}")
        print(f"  • 模型依赖 {len([f for f in feature_importance_df.head(20)['importance'] if f > 0.01])} 个核心特征")
        print(f"  • 特征利用率为 {len([f for f in feature_importance_df['importance'] if f > 0.001])/len(self.feature_names)*100:.1f}%")

def main():
    """主函数"""
    # 创建模型解释器
    explainer = ModelExplainer("models/latest.pkl")
    
    # 生成完整分析报告
    explainer.model_summary_report()

if __name__ == "__main__":
    main()