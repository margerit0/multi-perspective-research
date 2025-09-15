import re
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from src.config import llm, llm_think, tavily_retriever
from src.models import (
    GenerateAnalystsState, Perspectives, InterviewState, 
    ResearchGraphState, Analyst
)
from src.prompts import (
    analyst_instructions, question_instructions, search_instructions,
    answer_instructions, section_writer_instructions, 
    report_writer_instructions, intro_conclusion_instructions
)
from src.tools import (
    _invoke_stream_collect_text, _extract_query_text_strict,
    _heuristic_query_from_last_question, _sanitize_query, _format_docs,
    _google_to_docs, _numbered_context, _last_analyst_question
)

# ===== Analyst Generation Nodes =====

def create_analysts(state: GenerateAnalystsState):
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    parser = PydanticOutputParser(pydantic_object=Perspectives)

    format_instructions = parser.get_format_instructions()
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts
    ) + (
        "\n\n严格的输出要求：\n"
        f"{format_instructions}\n"
        "重要：只输出合法 JSON（不要 Markdown、不要多余文字、不要使用 ``` 代码块）。"
    )

    resp = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="请生成这组分析员")
    ])
    text = resp.content.strip()
    perspectives = parser.parse(text)

    return {"analysts": perspectives.analysts}

def human_feedback(state: GenerateAnalystsState):
    pass

def should_continue(state: GenerateAnalystsState):
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    return END

# ===== Interview Nodes =====

def generate_question(state: Dict[str, Any]):
    analyst = state.get("analyst", {})
    messages = state.get("messages", [])
    max_turns = int(state.get("max_num_turns", 2))

    num_questions = sum(
        1 for m in messages
        if isinstance(m, HumanMessage) and getattr(m, "name", "") == "analyst"
    )
    current_turn = num_questions + 1

    if current_turn == 1:
        intro_or_followup = "请先以符合你人物设定的名字进行自我介绍，然后提出你的第一个问题。"
    elif current_turn < max_turns:
        intro_or_followup = f"基于专家的回答，请提出你的第{current_turn}个深入问题。继续挖掘更多细节。"
    else:
        intro_or_followup = f"这是你的最后一个问题（第{current_turn}轮）。请提出一个总结性或最关键的问题"

    system_message = question_instructions.format(
        goals=analyst.persona,
        max_turns=max_turns,
        current_turn=current_turn,
        intro_or_followup=intro_or_followup
    )

    llm_input_messages = [SystemMessage(content=system_message)] + messages

    text = _invoke_stream_collect_text(llm_think, llm_input_messages)
    question_msg = HumanMessage(content=text, name="analyst")
    return {"messages": [question_msg]}

def search_both(state: InterviewState):
    last_q = _last_analyst_question(state.get("messages", [])) or ""
    msgs = [search_instructions, HumanMessage(content=last_q)]

    resp = llm.invoke(msgs)
    raw = getattr(resp, "content", resp)

    search_query = _extract_query_text_strict(raw) or ""
    if not search_query:
        last_q = _last_analyst_question(state.get("messages", [])) or ""
        search_query = _heuristic_query_from_last_question(last_q)

    search_query = _sanitize_query(search_query, limit_words=15)

    web_docs = tavily_retriever.invoke(search_query) if search_query else []
    google_docs = _google_to_docs(search_query) if search_query else []
    blocks = _format_docs(web_docs, "tavily") + _format_docs(google_docs, "google")

    seen, uniq = set(), []
    for b in blocks:
        start = b.find('href="') + 6
        end = b.find('"', start)
        url = b[start:end] if start > 5 and end > start else b
        if url and url not in seen:
            uniq.append(b)
            seen.add(url)

    return {"context": uniq}

def generate_answer(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    context = state.get("context", [])
    numbered = _numbered_context(context)
    question = _last_analyst_question(messages)

    system_message = answer_instructions.format(goals=analyst.persona, context=numbered)

    prompt = HumanMessage(content=f"问题：{question}\n\n请严格依据以上编号来源作答，并在正文中使用 [1]、[2]… 引用；文末列出来源清单。")
    resp = llm.invoke([SystemMessage(content=system_message), prompt])

    answer_msg = AIMessage(content=resp.content, name="expert")
    return {"messages": [answer_msg]}

def save_interview(state: InterviewState):
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    num_questions = sum(
        1 for m in messages if isinstance(m, HumanMessage) and getattr(m, "name", "") == "analyst"
    )
    if num_questions >= max_num_turns:
        return 'save_interview'

    return "ask_question"

def write_section(state: InterviewState):
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    system_message = section_writer_instructions.format(focus=analyst.description)
    msgs = [SystemMessage(content=system_message),
            HumanMessage(content=f"请使用以下来源撰写你的章节：{context}")]

    text = _invoke_stream_collect_text(llm_think, msgs).strip()

    return {"sections": [text]}

# ===== Research Graph Nodes =====

def initiate_all_interviews(state: ResearchGraphState):
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"
    else:
        max_num_turns = state.get("max_num_turns", 2)
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "messages": [],
                    "max_num_turns": max_num_turns,
                },
            )
            for analyst in state["analysts"]
        ]

def write_report(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)

    report = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="请基于上述备忘录撰写一份报告。")
    ])
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = llm.invoke([instructions, HumanMessage(content="请撰写报告的引言")])
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)

    conclusion = llm.invoke([instructions, HumanMessage(content="请撰写报告的结论")])
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    content = state["content"]

    if content.startswith("## Insights"):
        content = content[len("## Insights"):].lstrip()

    parts = re.split(r"\n## Sources\s*\n", content, maxsplit=1)
    if len(parts) == 2:
        content_body, sources_raw = parts
        # 规范化 Sources：转为 bullet list
        lines = [ln.strip() for ln in sources_raw.strip().splitlines() if ln.strip()]
        sources_block = "\n".join(f"- {ln}" for ln in lines)
    else:
        content_body = content
        sources_block = None

    final_report = (
        state["introduction"].rstrip()
        + "\n\n---\n\n"
        + content_body.strip()
        + "\n\n---\n\n"
        + state["conclusion"].strip()
    )
    if sources_block:
        final_report += "\n\n## Sources\n\n" + sources_block + "\n"

    return {"final_report": final_report}

# ===== Graph Builders =====

def build_analyst_graph():
    """构建分析师生成图"""
    builder = StateGraph(GenerateAnalystsState)
    
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    
    builder.add_conditional_edges(
        "human_feedback",
        should_continue,
        ["create_analysts", END]
    )
    
    memory = MemorySaver()
    graph = builder.compile(
        interrupt_before=['human_feedback'],
        checkpointer=memory
    )
    
    return graph

def build_interview_graph():
    """构建访谈图"""
    interview_builder = StateGraph(InterviewState)
    
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_both", search_both)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)
    
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_both")
    interview_builder.add_edge("search_both", "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question','save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)
    
    memory = MemorySaver()
    interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")
    
    return interview_graph

def build_research_graph():
    """构建研究图"""
    interview_graph = build_interview_graph()
    
    builder = StateGraph(ResearchGraphState)
    
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)
    
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)
    
    memory = MemorySaver()
    graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
    
    return graph