import pickle
import sys
import os

def analyze_pkl_keys(file_path, max_items=5):
    """
    分析PKL文件的内容结构和键值
    """
    try:
        # 检查文件存在性
        if not os.path.exists(file_path):
            print(f"❌ 错误：文件不存在 - {file_path}")
            return
            
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        print(f"📂 文件路径: {file_path}")
        print(f"📏 文件大小: {file_size / 1024:.2f} KB")
        
        # 加载PKL文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 打印数据类型
        print(f"📦 数据类型: {type(data).__name__}")
        
        # 处理不同类型的数据
        if isinstance(data, dict):
            print("\n🔑 顶层键值:")
            for key in data.keys():
                print(f"  - {key}")
                
            # 输出部分值预览
            print("\n🔍 值预览 (显示前 {max_items} 项):")
            for i, (key, value) in enumerate(list(data.items())[:max_items]):
                print(f"  {key}: {preview_value(value)}")
                
        elif isinstance(data, list):
            print(f"\n📊 列表元素数量: {len(data)}")
            if data:
                first_item = data[0]
                print(f"📦 首个元素类型: {type(first_item).__name__}")
                
                if isinstance(first_item, dict):
                    print("🔑 首个元素的键值:")
                    for key in first_item.keys():
                        print(f"  - {key}")
                    
                    # 输出首个元素的详细结构
                    print("\n🔍 首元素内容预览:")
                    for i, (key, value) in enumerate(list(first_item.items())[:max_items]):
                        print(f"  {key}: {preview_value(value)}")
                else:
                    print(f"\nℹ️ 首个元素内容: {preview_value(first_item)}")
                    
            print(f"\n✅ 分析完成! 提示：此文件包含 {len(data)} 条检测结果")
            
        else:
            print("\nℹ️ 数据类型不支持自动解析")
            print(f"内容预览: {preview_value(data)}")
            
    except Exception as e:
        print(f"❌ 分析出错: {str(e)}")

def preview_value(value):
    """生成值的预览字符串"""
    # 基础类型
    if isinstance(value, (int, float, str, bool)) or value is None:
        return repr(value)[:50] + ('...' if len(repr(value)) > 50 else '')
    
    # 集合类型
    elif isinstance(value, (list, tuple)):
        preview = f"{type(value).__name__}(len={len(value)}"
        if value:
            return f"{preview}, 首项: {preview_value(value[0])})"
        return f"{preview})"
    
    # 字典类型
    elif isinstance(value, dict):
        if not value:
            return "{}"
        key = next(iter(value.keys()))
        return f"dict(len={len(value)}, 首键: {key} -> {preview_value(value[key])})"
    
    # Numpy或PyTorch对象
    elif hasattr(value, 'shape'):
        return f"{type(value).__name__}(shape={value.shape})"
    
    # 其他对象
    else:
        return f"{type(value).__name__}实例"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python pkl_keys.py <pkl文件路径> [max_items]")
        print("示例: python pkl_keys.py results.pkl 5")
        sys.exit(1)
    
    # 解析参数
    file_path = sys.argv[1]
    max_items = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print("=" * 60)
    print(f"🧪 开始分析 PKL 文件: {file_path}")
    print("=" * 60)
    analyze_pkl_keys(file_path, max_items)