"""
简化的API测试脚本，用于排查连接问题
"""
import requests
import sys

def simple_test():
    """简单测试API连接"""

    print("🔧 开始诊断API连接...")

    api_url = "http://localhost:8000"

    # 测试1: 基本连接
    print("\n1️⃣ 测试基本连接...")
    try:
        response = requests.get(f"{api_url}/", timeout=10)
        print(f"✅ 连接成功！状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！API服务可能未启动")
        print("请先运行: python api_service.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ 连接超时！")
        return False
    except Exception as e:
        print(f"❌ 连接出错: {e}")
        return False

    # 测试2: 健康检查
    print("\n2️⃣ 测试健康检查...")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ 健康检查通过")
        else:
            print(f"⚠️ 健康检查异常: {response.text}")
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")

    # 测试3: 简单查询
    print("\n3️⃣ 测试简单查询...")
    try:
        response = requests.post(
            f"{api_url}/query_simple",
            params={"question": "测试", "max_results": 1},
            timeout=15
        )
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 查询成功！找到 {data.get('count', 0)} 条记录")
        else:
            print(f"⚠️ 查询失败: {response.text}")
    except Exception as e:
        print(f"❌ 查询出错: {e}")

    return True

def interactive_query():
    """交互式查询测试"""
    api_url = "http://localhost:8000"

    print("\n🎮 交互式查询模式")
    print("输入问题进行查询，输入 'quit' 退出")

    while True:
        question = input("\n❓ 请输入问题: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        try:
            print("🔍 查询中...")
            response = requests.post(
                f"{api_url}/query_simple",
                params={"question": question, "max_results": 3},
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                print(f"\n📊 找到 {count} 条相关记录:")

                for i, record in enumerate(data.get('records', [])[:2]):
                    print(f"\n📝 记录 {i+1}:")
                    print(f"发送者: {record.get('sender', '未知')}")
                    print(f"时间: {record.get('time', '未知')}")
                    print(f"内容: {record.get('content', '')[:150]}...")

            else:
                print(f"❌ 查询失败: {response.text}")

        except Exception as e:
            print(f"❌ 查询出错: {e}")

if __name__ == "__main__":
    print("🤖 API连接诊断工具")
    print("=" * 40)

    # 先进行基本测试
    if simple_test():
        print("\n✅ 基本连接正常！")

        # 询问是否要进行交互式测试
        choice = input("\n是否要进行交互式查询测试？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_query()
    else:
        print("\n❌ 连接失败，请检查:")
        print("1. API服务是否已启动: python api_service.py")
        print("2. 端口8000是否被占用")
        print("3. 向量数据库是否存在")

    print("\n👋 诊断完成！")