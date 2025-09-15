"""
主程序入口
处理用户交互和多视角研究工作流执行
"""

import datetime
import json
from typing import Optional, Dict, Any
from IPython.display import Image, display
from src.agents import build_research_graph
from src.models import Analyst


class MultiPerspectiveResearchAssistant:
    """
    多视角研究助手类
    管理整个研究流程的执行和用户交互
    """
    
    def __init__(self):
        print("\n⏳ 正在初始化研究系统...")
        self.graph = build_research_graph()
        self.thread_config = {"configurable": {"thread_id": "research_001"}}
        print("✅ 系统初始化完成")
    
    def display_menu(self) -> None:
        """
        显示主菜单
        """
        print("\n" + "="*70)
        print("🔬 多视角研究助手系统")
        print("="*70)
        print("\n功能选项:")
        print("  1. 开始新研究")
        print("  2. 查看系统架构图")
        print("  3. 使用快速模板")
        print("  4. 查看使用说明")
        print("  5. 退出系统")
        print("\n" + "="*70)
    
    def display_analysts(self, analysts: list[Analyst]) -> None:
        """
        以格式化方式显示生成的分析员列表
        """
        print("\n" + "="*70)
        print("🎭 生成的分析员团队:")
        print("="*70)
        
        for i, analyst in enumerate(analysts, 1):
            print(f"\n【分析员 {i}】")
            print(f"  📛 姓名: {analyst.name}")
            print(f"  🏢 机构: {analyst.affiliation}")
            print(f"  👤 角色: {analyst.role}")
            print(f"  📝 描述: {analyst.description}")
            print("  " + "-"*60)
        
        print("\n" + "="*70)
    
    def get_research_parameters(self) -> Dict[str, Any]:
        """
        交互式获取研究参数
        """
        print("\n" + "="*70)
        print("📋 研究参数设置")
        print("="*70)
        
        # 研究主题
        while True:
            topic = input("\n1. 请输入研究主题: ").strip()
            if topic:
                break
            print("   ⚠️ 主题不能为空，请重新输入")
        
        # 分析员数量
        print("\n2. 设置分析员数量")
        print("   建议: 2-3个用于快速研究，4-5个用于深度研究")
        while True:
            try:
                max_analysts = int(input("   分析员数量 (1-8, 默认3): ").strip() or "3")
                if 1 <= max_analysts <= 8:
                    break
                print("   ⚠️ 请输入1-8之间的数字")
            except ValueError:
                print("   ⚠️ 无效输入，请输入数字")
        
        # 采访轮数
        print("\n3. 设置每次采访轮数")
        print("   建议: 2轮用于概览，3-4轮用于深入探讨")
        while True:
            try:
                max_turns = int(input("   采访轮数 (1-5, 默认2): ").strip() or "2")
                if 1 <= max_turns <= 5:
                    break
                print("   ⚠️ 请输入1-5之间的数字")
            except ValueError:
                print("   ⚠️ 无效输入，请输入数字")
        
        # 确认参数
        print("\n" + "="*70)
        print("📊 研究参数确认:")
        print(f"   • 研究主题: {topic}")
        print(f"   • 分析员数量: {max_analysts}")
        print(f"   • 采访轮数: {max_turns}")
        print("="*70)
        
        confirm = input("\n确认这些参数吗? (y/n, 默认y): ").strip().lower()
        if confirm == 'n':
            return self.get_research_parameters()
        
        return {
            "topic": topic,
            "max_analysts": max_analysts,
            "max_num_turns": max_turns
        }
    
    def get_user_feedback(self) -> str:
        """
        获取用户对分析员的反馈
        """
        print("\n" + "="*70)
        print("📝 分析员审核")
        print("="*70)
        print("\n请选择操作:")
        print("  1. ✅ 批准并继续研究")
        print("  2. 🔄 提供修改建议")
        print("  3. ❌ 取消本次研究")
        
        while True:
            choice = input("\n请输入选项 (1-3): ").strip()
            
            if choice == '1':
                return 'approve'
            elif choice == '2':
                feedback = input("请输入具体的修改建议: ").strip()
                if feedback:
                    return feedback
                print("⚠️ 修改建议不能为空")
            elif choice == '3':
                return 'quit'
            else:
                print("⚠️ 请输入有效的选项 (1-3)")
    
    def show_progress(self, stage: str, detail: str = "") -> None:
        """
        显示进度信息
        """
        stages = {
            "analysts": "🎭 生成分析员团队",
            "interview": "💬 进行专家访谈",
            "report": "📝 撰写研究报告",
            "intro": "📖 生成引言",
            "conclusion": "📑 生成结论",
            "finalize": "🎯 整合最终报告"
        }
        
        icon = stages.get(stage, "⏳")
        if detail:
            print(f"   {icon} {detail}")
        else:
            print(f"   {icon}")
    
    def run_research(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        执行完整的研究流程
        """
        print(f"\n🚀 开始研究: {params['topic']}")
        
        # 初始状态
        initial_state = {
            "topic": params["topic"],
            "max_analysts": params["max_analysts"],
            "max_num_turns": params["max_num_turns"],
            "human_analyst_feedback": "",
            "analysts": [],
            "sections": [],
            "introduction": "",
            "content": "",
            "conclusion": "",
            "final_report": ""
        }
        
        try:
            # 第一阶段：生成分析员
            print("\n" + "="*70)
            print("阶段 1: 分析员团队构建")
            print("="*70)
            self.show_progress("analysts", "正在生成分析员团队...")
            
            # 执行直到遇到中断点
            for event in self.graph.stream(initial_state, self.thread_config):
                if "create_analysts" in event:
                    analysts_data = event["create_analysts"]
                    if analysts_data and "analysts" in analysts_data:
                        self.display_analysts(analysts_data["analysts"])
                
                if "human_feedback" in event:
                    break
            
            # 获取用户反馈
            feedback = self.get_user_feedback()
            
            if feedback == 'quit':
                print("\n❌ 研究已取消")
                return None
            
            # 处理反馈循环
            iteration = 1
            while feedback != 'approve':
                print(f"\n🔄 第 {iteration} 次修改...")
                print(f"   反馈: {feedback}")
                
                # 更新反馈
                self.graph.update_state(
                    self.thread_config,
                    {"human_analyst_feedback": feedback},
                    as_node="human_feedback"
                )
                
                # 重新生成
                for event in self.graph.stream(None, self.thread_config):
                    if "create_analysts" in event:
                        analysts_data = event["create_analysts"]
                        if analysts_data and "analysts" in analysts_data:
                            self.display_analysts(analysts_data["analysts"])
                    
                    if "human_feedback" in event:
                        break
                
                feedback = self.get_user_feedback()
                if feedback == 'quit':
                    print("\n❌ 研究已取消")
                    return None
                
                iteration += 1
            
            # 第二阶段：执行研究
            print("\n" + "="*70)
            print("阶段 2: 研究执行")
            print("="*70)
            print("✅ 分析员团队已批准，开始研究...")
            
            # 更新状态为批准
            self.graph.update_state(
                self.thread_config,
                {"human_analyst_feedback": ""},
                as_node="human_feedback"
            )
            
            # 继续执行
            interview_count = 0
            section_count = 0
            final_report = None
            
            for event in self.graph.stream(None, self.thread_config):
                for node_name, node_output in event.items():
                    if node_name == "conduct_interview":
                        if "messages" in node_output:
                            interview_count += 1
                            self.show_progress("interview", 
                                f"进行访谈 {interview_count}/{params['max_analysts']}")
                        if "sections" in node_output:
                            section_count += 1
                            self.show_progress("report", 
                                f"生成章节 {section_count}/{params['max_analysts']}")
                    elif node_name == "write_report":
                        self.show_progress("report", "整合研究内容")
                    elif node_name == "write_introduction":
                        self.show_progress("intro", "撰写报告引言")
                    elif node_name == "write_conclusion":
                        self.show_progress("conclusion", "撰写报告结论")
                    elif node_name == "finalize_report":
                        self.show_progress("finalize", "生成最终报告")
                        if node_output and "final_report" in node_output:
                            final_report = node_output["final_report"]
            
            # 显示最终报告
            if final_report:
                print("\n" + "="*70)
                print("📄 研究报告生成完成")
                print("="*70)
                print(final_report)
                
                # 保存报告
                filename = self.save_report(final_report, params["topic"])
                
                return {
                    "final_report": final_report,
                    "filename": filename
                }
            else:
                print("\n❌ 报告生成失败")
                return None
                
        except Exception as e:
            print(f"\n❌ 执行过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_report(self, report: str, topic: str) -> str:
        """
        保存报告到文件
        """
        # 生成安全的文件名
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')[:50]
        
        # 添加时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{safe_topic}_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✅ 报告已保存到: {filename}")
            return filename
        except Exception as e:
            print(f"\n⚠️ 保存报告失败: {str(e)}")
            return ""
    
    def show_architecture(self) -> None:
        """
        生成并将系统架构图保存到当前目录
        """
        print("\n" + "="*70)
        print("🏗️ 系统架构图")
        print("="*70)

        try:
            print("\n生成研究工作流图...")

            graph_obj = self.graph.get_graph()
            # 优先使用 Mermaid PNG，如不可用则尝试通用 PNG
            if hasattr(graph_obj, "draw_mermaid_png"):
                img_bytes = graph_obj.draw_mermaid_png()
            elif hasattr(graph_obj, "draw_png"):
                img_bytes = graph_obj.draw_png()
            else:
                raise RuntimeError("图对象不支持导出 PNG（缺少 draw_mermaid_png 或 draw_png 方法）")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"architecture_{timestamp}.png"

            with open(filename, "wb") as f:
                f.write(img_bytes)

            print(f"✅ 架构图已保存到当前目录: {filename}")
        except Exception as e:
            print(f"⚠️ 无法生成或保存架构图: {str(e)}")
            print("\n文字版架构说明:")
            print("1. 创建分析员 → 2. 人工审核")
            print("3. 多线程访谈 → 4. 生成章节")
            print("5. 撰写报告 → 6. 整合输出")
    
    def use_template(self) -> Dict[str, Any]:
        """
        使用预设研究模板
        """
        templates = [
            {
                "name": "技术评估模板",
                "topic": "大语言模型在企业应用中的最佳实践与挑战",
                "max_analysts": 3,
                "max_num_turns": 2
            },
            {
                "name": "市场分析模板",
                "topic": "人工智能助手市场的现状与发展趋势",
                "max_analysts": 2,
                "max_num_turns": 2
            },
            {
                "name": "深度研究模板",
                "topic": "LangGraph框架的应用场景与技术优势",
                "max_analysts": 3,
                "max_num_turns": 2
            }
        ]
        
        print("\n" + "="*70)
        print("📚 研究模板")
        print("="*70)
        
        for i, template in enumerate(templates, 1):
            print(f"\n{i}. {template['name']}")
            print(f"   主题: {template['topic']}")
            print(f"   分析员: {template['max_analysts']}人, 访谈轮数: {template['max_num_turns']}轮")
        
        while True:
            choice = input("\n请选择模板 (1-3): ").strip()
            if choice in ['1', '2', '3']:
                selected = templates[int(choice) - 1]
                print(f"\n✅ 已选择: {selected['name']}")
                return {
                    "topic": selected["topic"],
                    "max_analysts": selected["max_analysts"],
                    "max_num_turns": selected["max_num_turns"]
                }
            print("⚠️ 请输入有效的选项")
    
    def show_instructions(self) -> None:
        """
        显示使用说明
        """
        print("\n" + "="*70)
        print("📖 使用说明")
        print("="*70)
        print("""
多视角研究助手系统使用指南:

1. 系统概述
   本系统通过多个AI分析员从不同角度研究指定主题，
   每个分析员会进行独立的专家访谈，最后整合成综合报告。

2. 研究流程
   • 第一步: 根据主题生成分析员团队
   • 第二步: 人工审核并调整分析员
   • 第三步: 各分析员进行独立访谈
   • 第四步: 生成各自的研究章节
   • 第五步: 整合成完整研究报告

3. 参数说明
   • 分析员数量: 决定研究的视角数量(建议3-5个)
   • 访谈轮数: 每个访谈的深度(建议2-3轮)
   • 研究主题: 尽量具体和聚焦

4. 输出格式
   最终报告包含引言、多个研究章节、结论和参考来源，
   自动保存为Markdown格式文件。

5. 注意事项
   • 研究过程需要一定时间，请耐心等待
   • 审核分析员时可多次调整直到满意
   • 报告会自动保存到当前目录
        """)
        print("="*70)
        input("\n按回车键返回主菜单...")


def main():
    """
    主程序入口
    """
    print("\n" + "="*70)
    print("🌟 欢迎使用多视角研究助手系统")
    print("="*70)
    
    # 创建助手实例
    assistant = MultiPerspectiveResearchAssistant()
    
    while True:
        assistant.display_menu()
        choice = input("\n请选择功能 (1-5): ").strip()
        
        if choice == '1':
            # 开始新研究
            params = assistant.get_research_parameters()
            result = assistant.run_research(params)
            
            if result:
                print("\n✅ 研究完成！")
                print(f"   报告已保存: {result.get('filename', 'report.md')}")
            
            input("\n按回车键返回主菜单...")
            
        elif choice == '2':
            # 查看系统架构
            assistant.show_architecture()
            input("\n按回车键返回主菜单...")
            
        elif choice == '3':
            # 使用模板
            params = assistant.use_template()
            confirm = input("\n是否立即开始研究? (y/n): ").strip().lower()
            if confirm == 'y':
                result = assistant.run_research(params)
                if result:
                    print("\n✅ 研究完成！")
            input("\n按回车键返回主菜单...")
            
        elif choice == '4':
            # 查看说明
            assistant.show_instructions()
            
        elif choice == '5':
            # 退出
            print("\n👋 感谢使用多视角研究助手系统！")
            print("   祝您研究顺利！")
            print("="*70)
            break
            
        else:
            print("\n⚠️ 无效选项，请输入1-5之间的数字")


if __name__ == "__main__":
    main()