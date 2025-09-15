import json
import operator
import os
import re
from textwrap import dedent
from typing import TypedDict, Any, Dict, List, Annotated

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.types import Send

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.documents import Document

from langchain_community.retrievers import TavilySearchAPIRetriever

key = os.environ["MODELSCOPE_TOKEN"]
llm = ChatOpenAI(
    base_url="https://api-inference.modelscope.cn/v1",
    api_key=key,
    model="deepseek-ai/DeepSeek-V3.1",
    temperature=0,
)
llm_think = llm.bind(extra_body={"enable_thinking": True})


# ====================== Schema ======================
class Analyst(BaseModel):
    affiliation: str = Field(description="分析员的主要所属机构。")
    name: str = Field(description="分析员姓名。")
    role: str = Field(description="在该主题语境下的角色与职责。")
    description: str = Field(description="分析员的关注点、担忧与动机描述。")

    @property
    def persona(self) -> str:
        return (
            f"姓名: {self.name}\n"
            f"角色: {self.role}\n"
            f"所属机构: {self.affiliation}\n"
            f"描述: {self.description}\n"
        )


class Perspectives(BaseModel):
    analysts: list[Analyst] = Field(description="包含分析员及其角色与所属机构的完整列表。")


class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: list[Analyst]


# ====================== 生成分析员 ======================
from langchain_core.output_parsers import PydanticOutputParser

analyst_instructions = """你将负责创建一组 AI 分析员人物设定（personas）。请严格遵循以下指示：

1. 首先，审阅研究主题：
{topic}

2. 查看（可选的）用于指导分析员创建的编辑反馈：
{human_analyst_feedback}

3. 基于上述文档和/或反馈，确定最有趣的主题。

4. 选出排名靠前的 {max_analysts} 个主题。

5. 为每个主题分配一名分析员。

6. 输出语言使用中文
"""


def create_analysts(state: GenerateAnalystsState):
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    parser = PydanticOutputParser(pydantic_object=Perspectives)
    format_instructions = parser.get_format_instructions()

    system_message = analyst_instructions.format(
        topic=topic, human_analyst_feedback=human_analyst_feedback, max_analysts=max_analysts
    ) + (
        "\n\n严格的输出要求：\n"
        f"{format_instructions}\n"
        "重要：只输出合法 JSON（不要 Markdown、不要多余文字、不要使用 ``` 代码块）。"
    )

    resp = llm.invoke([SystemMessage(content=system_message), HumanMessage(content="请生成这组分析员")])
    text = resp.content.strip()
    perspectives = parser.parse(text)

    return {"analysts": perspectives.analysts}


def human_feedback(state: GenerateAnalystsState):
    """人工反馈节点（no-op），配合 interrupt_before 在此处中断等待人工输入"""
    return None


def should_continue(state: GenerateAnalystsState):
    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback:
        return "create_analysts"
    return END


# 为“生成分析员”部分单独构建一张图（不会作为 Studio 主图导出）
analyst_builder = StateGraph(GenerateAnalystsState)
analyst_builder.add_node("create_analysts", create_analysts)
analyst_builder.add_node("human_feedback", human_feedback)
analyst_builder.add_edge(START, "create_analysts")
analyst_builder.add_edge("create_analysts", "human_feedback")
analyst_builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])
analyst_graph = analyst_builder.compile(interrupt_before=["human_feedback"])


# ====================== 采访子图（Analyst ↔ Expert） ======================
class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="用于检索的搜索查询")


question_instructions = """
你是一名采访分析师，任务是与某领域专家访谈，层层追问，深入某个特定主题。

回合规则
- 必须进行恰好 {max_turns} 轮；当前为第 {current_turn}/{max_turns} 轮。
- 每轮基于上轮回答追问，单点深入，不并列多个话题。
- 单轮输出：一个聚焦问题（必要时可附最多3个要点/指标要求），避免寒暄与总结。
- 多轮对话由 analyst 和 expert 构成，analyst是上一个提出的问题，expert 是上一个问题的答案

目标（提炼“有趣且具体”的洞见）
- 有趣：出人意料或非直觉的发现。
- 具体：包含专家给出的案例、细节或数据（数值/阈值/配置/时间/成本等）。
- 多轮对话中你将扮演 analyst 提出问题

主题与目标：{goals}

对话策略
- {intro_or_followup}
- 始终保持与提供的人设 persona 与目标一致的语气与角色。
"""


def _invoke_stream_collect_text(chat_model, msgs) -> str:
    pieces = []
    for chunk in chat_model.stream(msgs):
        if getattr(chunk, "content", ""):
            pieces.append(chunk.content)
    return "".join(pieces)


def generate_question(state: Dict[str, Any]):
    analyst = state.get("analyst", {})
    messages = state.get("messages", [])
    max_turns = int(state.get("max_num_turns", 2))

    num_questions = sum(
        1 for m in messages if isinstance(m, HumanMessage) and getattr(m, "name", "") == "analyst"
    )
    current_turn = num_questions + 1

    if current_turn == 1:
        intro_or_followup = "请先以符合你人物设定的名字进行自我介绍，然后提出你的第一个问题。"
    elif current_turn < max_turns:
        intro_or_followup = f"基于专家的回答，请提出你的第{current_turn}个深入问题。继续挖掘更多细节。"
    else:
        intro_or_followup = f"这是你的最后一个问题（第{current_turn}轮）。请提出一个总结性或最关键的问题"

    system_message = question_instructions.format(
        goals=analyst.persona, max_turns=max_turns, current_turn=current_turn, intro_or_followup=intro_or_followup
    )

    llm_input_messages = [SystemMessage(content=system_message)] + messages
    text = _invoke_stream_collect_text(llm_think, llm_input_messages)
    question_msg = HumanMessage(content=text, name="analyst")
    return {"messages": [question_msg]}


tavily_retriever = TavilySearchAPIRetriever(k=3)


def _balanced_json_slice(text: str) -> str | None:
    i = text.find("{")
    if i == -1:
        return None
    depth, j, in_str, esc = 0, i, False, False
    while j < len(text):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
        j += 1
    return None


def _extract_query_text_strict(text: str) -> str:
    t = text.replace("“", '"').replace("”", '"')
    cand = _balanced_json_slice(t)
    if cand:
        try:
            obj = json.loads(cand)
            q = obj.get("search_query", "") if isinstance(obj, dict) else ""
            if isinstance(q, str) and q.strip():
                return q.strip()
        except Exception:
            pass
    m = re.search(r'"search_query"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', t)
    if m:
        q = m.group(1)
        q = bytes(q, "utf-8").decode("unicode_escape")
        return q.strip()
    m2 = re.search(r'"search_query"\s*:\s*"([^"}\n]*)', t)
    if m2:
        return m2.group(1).strip()
    return ""


def _heuristic_query_from_last_question(last_q: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9./:+#_-]+|[\u4e00-\u9fff]+", last_q)
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "with",
        "how",
        "what",
        "why",
        "is",
        "are",
        "can",
        "could",
        "should",
        "would",
    }
    kept = [t for t in tokens if t.lower() not in stop]
    return " ".join(kept[:15]).strip()


def _sanitize_query(q: str, limit_words: int = 15) -> str:
    q = q.replace("\n", " ").replace("\t", " ").strip()
    q = re.sub(r"\s+", " ", q)
    words = q.split(" ")
    return " ".join(words[:limit_words]).strip()


from langchain_google_community import GoogleSearchAPIWrapper

google_search = GoogleSearchAPIWrapper(
    google_api_key=os.environ["GOOGLE_SEARCH_KEY"],
    google_cse_id=os.environ["GOOGLE_CSE_ID"],
    k=3,
)


def _google_to_docs(query: str, k: int = 3) -> List[Document]:
    results = google_search.results(query, num_results=k) or []
    docs: List[Document] = []
    for r in results:
        url = r.get("link") or r.get("url") or ""
        title = r.get("title") or ""
        snippet = r.get("snippet") or ""
        page_content = (f"{title}\n{snippet}".strip()) or url
        docs.append(Document(page_content=page_content, metadata={"source": url, "title": title}))
    return docs


def _format_docs(docs, origin: str):
    blocks = []
    for doc in docs:
        src = doc.metadata.get("source", doc.metadata.get("url", ""))
        title = doc.metadata.get("title", "")
        blocks.append(f'<Document origin="{origin}" href="{src}" title="{title}"/>\n{doc.page_content}\n</Document>')
    return blocks


def _to_messages(msgs):
    out = []
    for m in msgs:
        if isinstance(m, BaseMessage):
            out.append(m)
        elif isinstance(m, str) and m.strip():
            out.append(HumanMessage(content=m.strip()))
    return out

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
    google_docs = _google_to_docs(search_query, k=6) if search_query else []
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

search_instructions = SystemMessage(
    content=dedent(
        """
# 角色
你是一个专家系统，专长是将对话中的最后一个问题，转化为一个为技术专家量身打造的、高度精确的英文网页检索查询。

# 任务
分析提供的“分析师”与“专家”之间的对话，定位最后一个问题，并严格按照下面的格式和规则生成一个检索查询。

# 输出格式 (绝对严格)
- **唯一输出**: 你的全部输出必须是且只能是一个单行的 JSON 对象。
- **JSON 结构**: `{"search_query":"<query>"}`
- **无额外内容**: 禁止包含任何解释、说明、代码块标记(```)、前缀或后缀。
- **标准引号**: 键和值都必须使用标准的 ASCII 双引号 (")。
- **空查询**: 如果无法根据最后一个问题生成有意义的查询，则值必须为空字符串，如：`{"search_query":""}`。

# 查询生成规则
1.  **聚焦**: 查询必须且仅能基于对话的**最后一个问题**。
2.  **关键词化**: 提取问题的核心技术概念，而非完整的问句。
3.  **专业术语**: 使用具体、公认的技术术语。如果问题中提到了产品名、库名或特定的功能，必须在查询中体现。
4.  **精炼**: 查询应简明扼要，目标是高效检索，长度严格控制在15个词以内。

---
## 示例

### 正例 (必须遵循的格式)
{"search_query":"LangGraph agent workflow orchestration"}

### 反例 (必须避免的错误)
- **错误 (包含前缀)**: Here is the JSON: {"search_query":"LangGraph agent workflow orchestration"}
- **错误 (使用代码块)**: json\n{"search_query":"LangGraph agent workflow orchestration"}\n
- **错误 (自然语言问句)**: {"search_query":"how to orchestrate agent workflow in LangGraph"}
"""
    )
)

answer_instructions = """你是一位专家，正在接受一名分析师的采访。

分析师关注的领域如下：{goals}。

你的目标是回答采访者提出的问题。

为回答问题，请使用以下上下文：

{context}

回答时请遵循以下准则：

1. 仅使用上下文中提供的信息。

2. 不要引入上下文之外的信息，也不要做出超出上下文明确表述的假设。

3. 上下文中的每份文档都在开头处标注了来源信息。

4. 对与某条陈述相关的内容，请在陈述旁标注其来源编号。例如，来源 1 标注为 [1]。

5. 在答案底部按顺序列出你的来源。[1] 来源 1，[2] 来源 2，等等。

6. 如果来源为：<Document source="assistant/docs/llama3_1.pdf" page="7"/>，则在参考文献处只需列出：

[1] assistant/docs/llama3_1.pdf, page 7

并且不要额外添加尖括号或 “Document source” 这类前缀描述。"""


def _numbered_context(context_list: list[str]) -> str:
    lines = []
    for i, block in enumerate(context_list, 1):
        lines.append(f"[{i}] {block}")
    return "\n\n".join(lines)


def _last_analyst_question(messages):
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and getattr(m, "name", "") == "analyst":
            return m.content
    return ""


def generate_answer(state: InterviewState):
    analyst = state["analyst"]
    messages = state["messages"]
    context = state.get("context", [])
    numbered = _numbered_context(context)
    question = _last_analyst_question(messages)

    system_message = answer_instructions.format(goals=analyst.persona, context=numbered)
    prompt = HumanMessage(content=f"问题：{question}\n\n请严格依据以上编号来源作答，并在正文中使用 [1]、[2]… 引用；文末列出“来源”清单。")
    resp = llm.invoke([SystemMessage(content=system_message), prompt])

    answer_msg = AIMessage(content=resp.content, name="expert")
    return {"messages": [answer_msg]}


def save_interview(state: InterviewState):
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return {"interview": interview}


def route_messages(state: InterviewState, name: str = "expert"):
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    num_questions = sum(1 for m in messages if isinstance(m, HumanMessage) and getattr(m, "name", "") == "analyst")
    if num_questions >= max_num_turns:
        return "save_interview"

    last_q = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and getattr(m, "name", "") == "analyst":
            last_q = m
            break

    return "ask_question"


section_writer_instructions = """你是一名资深技术写作者。

你的任务是基于一组来源文档，撰写一段简短且易读的报告章节。

1. 分析来源文档的内容：
- 每份来源文档的名称位于文档开头，使用 <Document 标签标注。

2. 用 Markdown 创建报告结构：
- 使用 ## 作为章节标题
- 使用 ### 作为小节标题

3. 按以下结构撰写报告：
a. 标题（##）
b. 摘要（###）
c. 来源（###）

4. 让标题与分析师的关注领域相呼应并具有吸引力：
{focus}

5. 关于“摘要”：
- 先给出与分析师关注领域相关的背景/上下文
- 强调从采访中得到的新颖、有趣或令人意外的洞见
- 不要提及采访者或专家的姓名
- 目标长度约为 1000 字
- 在正文中按使用顺序用 [1]、[2] 等与 Sources 中相对应的编号引用来源

### Sources
- 编号不要基于来源文档，从 [1] 开始重新连续编号,不要跳过数字编号
- 使用编号在 Sources 中列出你用到的来源文档
- 格式:
[1] 链接或文档名
[2] 链接或文档名

7. 合并重复来源。例如下面是错误的：
[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

正确做法是仅保留一条：

[3] https://ai.meta.com/blog/meta-llama-3-1/

8. 最终检查：
- 确保报告符合上述结构
- 标题前不要有任何前置说明
- 确保已经遵循所有指南"""


def write_section(state: InterviewState):
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    system_message = section_writer_instructions.format(focus=analyst.description)
    msgs = [SystemMessage(content=system_message), HumanMessage(content=f"请使用以下来源撰写你的章节：{context}")]
    text = _invoke_stream_collect_text(llm_think, msgs).strip()
    return {"sections": [text]}


# Interview 子图
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_both", search_both)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_both")
interview_builder.add_edge("search_both", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ["ask_question", "save_interview"])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

interview_graph = interview_builder.compile().with_config(run_name="Conduct Interviews")


# ====================== 汇总报告（主图） ======================
class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


def initiate_all_interviews(state: ResearchGraphState):
    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback','approve')
    if human_analyst_feedback.lower() != 'approve':
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


report_writer_instructions = """你是一名技术写作者，正在围绕以下总主题创作一份报告：

{topic}

你有一个分析师团队。每位分析师都完成了两件事：

1. 他们就一个具体的子主题采访了一位专家。
2. 他们将自己的发现写成了一份备忘录(memo)。

你的任务：

1. 你将获得一组来自分析师的备忘录(memo)。
2. 仔细思考每份备忘录中的洞见。
3. 将这些洞见整合为一份清晰凝练的整体总结，把所有备忘录的核心思想串联起来。
4. 将每份备忘录的要点总结为一个连贯的单一叙述。

报告的格式要求：

1.  使用 markdown 格式。
2.  报告前不添加任何前言。
3.  不要使用子标题。
4.  让你的报告以一个单一的标题头开始：`## Insights`
5.  在报告中不要提及任何分析师的名字。
6.  保留备忘录中的引用标注，这些引用会以方括号呈现，例如 [1] 或 [2]。
7.  **关于“Sources”章节的创建规则（关键指令，请严格遵守）：**
    a. 在完成 `## Insights` 报告正文的撰写之后，另起一行，创建标题为 `## Sources` 的新章节。
    b. **核心要求：** 你 **必须只列出** 那些其引用编号（如 [1], [5] 等） **在你的 `## Insights` 正文中实际出现过** 的来源。
    c. **反向约束：** 如果一个来源的编号在 `## Insights` 正文中 **没有被引用**，那么该来源 **绝对不能** 出现在最终的 `## Sources` 列表中。最终的来源列表必须是你实际引用来源的一个子集，而不是全部来源的简单合并。
    d. 确保所有列出的来源都按照编号升序排列，并且没有重复项。
8.  输出请使用中文输出。

[1] Source 1
[2] Source 2

以下是你需要据此撰写报告的分析师备忘录(memo)：

{context}"""


def write_report(state: ResearchGraphState):
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)

    report = llm.invoke([SystemMessage(content=system_message), HumanMessage(content="请基于上述备忘录撰写一份报告。")])
    return {"content": report.content}


intro_conclusion_instructions = """你是一名技术写作者，正在为 {topic} 完成一份报告。

你将获得该报告的所有章节。

你的工作是撰写一段简洁有力的引言或结论。

用户会指示你撰写引言还是结论。

两部分均不要添加任何前置性前言。

目标约 100 个词：对于引言，简洁预览所有章节；对于结论，简洁回顾所有章节。

使用 markdown 格式，输出使用中文。

对于引言：请创建一个有吸引力的标题，并对该标题使用 # 作为标题头。

对于引言：请使用 ## Introduction 作为该节的节标题。

对于结论：请使用 ## Conclusion 作为该节的节标题。

以下是供你反思并据此撰写的章节内容：{formatted_str_sections}"""


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
        content = content[len("## Insights") :].lstrip()

    parts = re.split(r"\n## Sources\s*\n", content, maxsplit=1)
    if len(parts) == 2:
        content_body, sources_raw = parts
        lines = [ln.strip() for ln in sources_raw.strip().splitlines() if ln.strip()]
        sources_block = "\n".join(f"- {ln}" for ln in lines)
    else:
        content_body = content
        sources_block = None

    final_report = state["introduction"].rstrip() + "\n\n---\n\n" + content_body.strip() + "\n\n---\n\n" + state[
        "conclusion"
    ].strip()
    if sources_block:
        final_report += "\n\n## Sources\n\n" + sources_block + "\n"

    return {"final_report": final_report}


# ====================== 主图构建 ======================
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
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
builder.add_edge("write_report", "finalize_report")
builder.add_edge("write_introduction", "finalize_report")
builder.add_edge("write_conclusion", "finalize_report")

graph = builder.compile(interrupt_before=["human_feedback"])