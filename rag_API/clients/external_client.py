"""
外部客户端示例
供其他设备连接RAG API服务使用
"""

import requests
import json
import time

class RAGAPIClient:
    """RAG API客户端类"""

    def __init__(self, server_ip, port=8000):
        """
        初始化客户端
        server_ip: 服务器IP地址
        port: 服务端口，默认8000
        """
        self.base_url = f"http://{server_ip}:{port}"
        self.server_ip = server_ip

    def test_connection(self):
        """测试连接是否正常"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                print("✅ 连接成功！")
                return True
            else:
                print(f"⚠️ 连接异常，状态码: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ 连接失败！请检查:")
            print(f"1. 服务器地址是否正确: {self.base_url}")
            print("2. 服务器是否启动")
            print("3. 网络连接是否正常")
            print("4. 防火墙是否允许访问")
            return False
        except Exception as e:
            print(f"❌ 连接错误: {e}")
            return False

    def get_server_info(self):
        """获取服务器信息"""
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"获取服务器信息失败: {e}")
            return None

    def get_health_status(self):
        """获取服务健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"健康检查失败，状态码: {response.status_code}")
                return None
        except Exception as e:
            print(f"健康检查错误: {e}")
            return None

    def get_stats(self):
        """获取服务器统计信息"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return None

    def query(self, question, max_results=5):
        """
        查询聊天记录
        question: 查询问题
        max_results: 最大返回结果数
        """
        try:
            response = requests.post(
                f"{self.base_url}/query_simple",
                params={"question": question, "max_results": max_results},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"查询失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None

        except Exception as e:
            print(f"查询错误: {e}")
            return None

def interactive_client():
    """交互式客户端"""
    print("🤖 RAG API 外部客户端")
    print("=" * 50)

    # 获取服务器地址
    server_ip = input("请输入服务器IP地址 (例: 192.168.1.100): ").strip()

    if not server_ip:
        print("❌ 服务器地址不能为空")
        return

    # 创建客户端
    client = RAGAPIClient(server_ip)

    # 测试连接
    print(f"\n🔍 正在测试连接到 {server_ip}...")
    if not client.test_connection():
        return

    # 显示服务器信息
    print("\n📊 服务器信息:")
    server_info = client.get_server_info()
    if server_info:
        print(f"服务: {server_info.get('service', 'Unknown')}")
        print(f"版本: {server_info.get('version', 'Unknown')}")
        print(f"状态: {server_info.get('status', 'Unknown')}")

    # 显示统计信息
    stats = client.get_stats()
    if stats:
        print(f"\n📈 数据统计:")

        # 显示总记录数
        total_records = stats.get('total_records', 'Unknown')
        if isinstance(total_records, int):
            print(f"总记录数: {total_records:,} 条")
        else:
            print(f"总记录数: {total_records}")

        # 显示分析样本数
        sample_analyzed = stats.get('sample_analyzed', 0)
        if sample_analyzed > 0:
            print(f"分析样本: {sample_analyzed} 条")

        # 显示参与者
        senders = stats.get('unique_senders', [])
        if senders:
            print(f"聊天参与者: {', '.join(senders[:5])}{'...' if len(senders) > 5 else ''} (共{len(senders)}人)")

        # 显示时间范围
        time_range = stats.get('time_range', {})
        if time_range.get('earliest') and time_range.get('latest'):
            print(f"时间范围: {time_range['earliest']} 至 {time_range['latest']}")

        # 显示提醒信息
        if 'note' in stats:
            print(f"⚠️ {stats['note']}")

    # 交互式查询
    print("\n🎮 开始查询模式 (输入 'quit' 退出)")
    print("-" * 30)

    while True:
        question = input("\n❓ 请输入查询问题: ").strip()

        if question.lower() in ['quit', 'exit', 'q', '退出']:
            print("👋 退出客户端")
            break

        if not question:
            continue

        print("🔍 查询中...")
        start_time = time.time()

        result = client.query(question, max_results=20)  # 返回更多相关记录
        end_time = time.time()

        if result:
            count = result.get('count', 0)
            print(f"\n📊 查询结果 (耗时: {end_time - start_time:.2f}秒):")
            print(f"找到 {count} 条相关记录\n")

            for i, record in enumerate(result.get('records', [])):  # 显示所有返回的记录
                print(f"📝 记录 {i+1}:")
                print(f"   发送者: {record.get('sender', '未知')}")
                print(f"   时间: {record.get('time', '未知时间')}")
                print(f"   相似度: {record.get('similarity', 0):.3f}")
                print(f"   内容: {record.get('content', '')}")
                print("-" * 40)
        else:
            print("❌ 查询失败")

if __name__ == "__main__":
    print("🚀 RAG API 外部客户端工具")
    print("=" * 50)
    interactive_client()