"""
API客户端测试脚本
用于测试微信聊天记录向量数据库API服务
"""

import requests
import json
import time

# API服务地址
API_BASE_URL = "http://localhost:8000"

def test_api():
    """测试API各个端点"""

    print("🧪 开始测试API服务...")

    # 1. 测试根路径
    print("\n1️⃣ 测试根路径...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"❌ 根路径测试失败: {e}")

    # 2. 测试健康检查
    print("\n2️⃣ 测试健康检查...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")

    # 3. 测试统计信息
    print("\n3️⃣ 测试统计信息...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            stats = response.json()
            print(f"样本记录数: {stats.get('sample_records', 0)}")
            print(f"参与者: {stats.get('unique_senders', [])}")
            print(f"消息类型: {stats.get('message_types', [])}")
    except Exception as e:
        print(f"❌ 统计信息测试失败: {e}")

    # 4. 测试简单查询接口
    print("\n4️⃣ 测试简单查询...")
    test_questions = [
        "张宏哲说了什么？",
        "志愿服务",
        "雷蕾",
        "毕业晚会"
    ]

    for question in test_questions:
        print(f"\n🔍 查询: {question}")
        try:
            response = requests.post(
                f"{API_BASE_URL}/query_simple",
                params={"question": question, "max_results": 5}  # 增加到5条
            )
            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"找到记录数: {data.get('count', 0)}")

                for i, record in enumerate(data.get('records', [])):  # 显示所有返回的记录
                    print(f"记录 {i+1}:")
                    print(f"  发送者: {record.get('sender', '未知')}")
                    print(f"  时间: {record.get('time', '未知')}")
                    print(f"  相似度: {record.get('similarity', 0):.3f}")
                    print(f"  内容: {record.get('content', '')[:100]}...")
            else:
                print(f"查询失败: {response.text}")

        except Exception as e:
            print(f"❌ 查询失败: {e}")

        time.sleep(0.5)  # 避免请求过快

def interactive_test():
    """交互式测试"""
    print("\n🎮 进入交互式测试模式...")
    print("输入问题查询聊天记录，输入 'quit' 退出")

    while True:
        question = input("\n❓ 请输入问题: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 退出测试")
            break

        if not question:
            continue

        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/query_simple",
                params={"question": question, "max_results": 5}
            )
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                print(f"\n📊 查询结果 (耗时: {end_time - start_time:.2f}秒):")
                print(f"找到 {data.get('count', 0)} 条相关记录\n")

                for i, record in enumerate(data.get('records', [])):  # 显示所有返回的记录
                    print(f"📝 记录 {i+1}:")
                    print(f"   发送者: {record.get('sender', '未知')}")
                    print(f"   时间: {record.get('time', '未知')}")
                    print(f"   相似度: {record.get('similarity', 0):.3f}")
                    print(f"   内容: {record.get('content', '')}")
                    print("-" * 50)

            else:
                print(f"❌ 查询失败: {response.text}")

        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到API服务，请确保服务已启动")
        except Exception as e:
            print(f"❌ 查询出错: {e}")

if __name__ == "__main__":
    print("🤖 微信聊天记录API测试客户端")
    print("=" * 50)

    # 检查服务是否运行
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ API服务运行正常")
        else:
            print(f"⚠️ API服务响应异常: {response.status_code}")
    except:
        print("❌ 无法连接API服务，请确保:")
        print("   1. API服务已启动 (python api_service.py)")
        print("   2. 服务地址正确 (http://localhost:8000)")
        exit(1)

    # 选择测试模式
    mode = input("\n选择测试模式:\n1. 自动测试所有功能\n2. 交互式测试\n请选择 (1/2): ").strip()

    if mode == "1":
        test_api()
    elif mode == "2":
        interactive_test()
    else:
        print("无效选择，执行自动测试...")
        test_api()

    print("\n🎉 测试完成!")