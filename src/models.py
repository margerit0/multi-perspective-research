from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
import operator
from langgraph.graph import MessagesState

class Analyst(BaseModel):
    affiliation: str = Field(
        description="分析员的主要所属机构。",
    )
    name: str = Field(
        description="分析员姓名。"
    )
    role: str = Field(
        description="在该主题语境下的角色与职责。",
    )
    description: str = Field(
        description="分析员的关注点、担忧与动机描述。",
    )

    @property
    def persona(self) -> str:
        return (
            f"姓名: {self.name}\n"
            f"角色: {self.role}\n"
            f"所属机构: {self.affiliation}\n"
            f"描述: {self.description}\n"
        )

class Perspectives(BaseModel):
    analysts: list[Analyst] = Field(
        description="包含分析员及其角色与所属机构的完整列表。",
    )

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: list[Analyst]

class InterviewState(MessagesState):
    max_num_turns: int  # 对话轮数
    context: Annotated[list, operator.add]  # 源文档
    analyst: Analyst  # 提出问题的分析师
    interview: str  # 对话记录
    sections: list  # 从访谈中衍生的报告章节

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="用于检索的搜索查询")

class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analyst asking questions
    sections: Annotated[list, operator.add]  # Send() API key
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report