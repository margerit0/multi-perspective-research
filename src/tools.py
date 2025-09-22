import json, re, asyncio, ast, html
from typing import Any, Dict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_mcp_tools import convert_mcp_to_langchain_tools
from src.config import MCP_WEB_URL
import asyncio, atexit, inspect

def _invoke_stream_collect_text(chat_model, msgs) -> str:
    """收集流式响应的文本"""
    pieces = []
    for chunk in chat_model.stream(msgs):
        if getattr(chunk, "content", ""):
            pieces.append(chunk.content)
    return "".join(pieces)

def _balanced_json_slice(text: str) -> str | None:
    """提取平衡的 JSON 对象"""
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
                    return text[i:j+1]
        j += 1
    return None

def _extract_query_text_strict(text: str) -> str:
    """严格提取搜索查询文本"""
    t = text.replace(""", '"').replace(""", '"')
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
    """从最后一个问题启发式生成查询"""
    tokens = re.findall(r"[A-Za-z0-9./:+#_-]+|[\u4e00-\u9fff]+", last_q)
    stop = {"the","a","an","and","or","to","of","for","in","on","with",
            "how","what","why","is","are","can","could","should","would"}
    kept = [t for t in tokens if t.lower() not in stop]
    return " ".join(kept[:15]).strip()

def _sanitize_query(q: str, limit_words: int = 15) -> str:
    """清理和限制查询长度"""
    q = q.replace("\n"," ").replace("\t"," ").strip()
    q = re.sub(r"\s+", " ", q)
    words = q.split(" ")
    return " ".join(words[:limit_words]).strip()

def _format_docs(docs, origin: str):
    """格式化文档"""
    blocks = []
    for doc in docs:
        src = doc.metadata.get("source", doc.metadata.get("url", ""))
        title = doc.metadata.get("title", "")
        blocks.append(
            f'<Document origin="{origin}" href="{src}" title="{title}"/>\n{doc.page_content}\n</Document>'
        )
    return blocks

def _google_to_docs(query: str, k: int = 3) -> list[Document]:
    """Google 搜索结果转换为文档"""
    from src.config import google_search
    results = google_search.results(query, num_results=k) or []
    docs: list[Document] = []
    for r in results:
        url = r.get("link") or r.get("url") or ""
        title = r.get("title") or ""
        snippet = r.get("snippet") or ""
        page_content = (f"{title}\n{snippet}".strip()) or url
        docs.append(Document(page_content=page_content,
                           metadata={"source": url, "title": title}))
    return docs

def _to_messages(msgs):
    """转换为消息格式"""
    out = []
    for m in msgs:
        if isinstance(m, BaseMessage):
            out.append(m)
        elif isinstance(m, str) and m.strip():
            out.append(HumanMessage(content=m.strip()))
    return out

def _numbered_context(context_list: list[str]) -> str:
    """为上下文添加编号"""
    lines = []
    for i, block in enumerate(context_list, 1):
        lines.append(f"[{i}] {block}")
    return "\n\n".join(lines)

def _last_analyst_question(messages):
    """获取最后一个分析师问题"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and getattr(m, "name", "") == "analyst":
            return m.content
    return ""

# === MCP 工具连接与缓存 ===
#
_mcp_tools_cache: Dict[str, Any] = {"tools": None, "cleanup": None}
async def _ensure_mcp_tools():
    if _mcp_tools_cache["tools"] is None:
        tools, cleanup = await convert_mcp_to_langchain_tools({
            "web-search": {"type": "http", "url": MCP_WEB_URL}
        })
        _mcp_tools_cache["tools"], _mcp_tools_cache["cleanup"] = tools, cleanup
    return _mcp_tools_cache["tools"]

def _cleanup_mcp_tools_at_exit():
    cleanup = _mcp_tools_cache.get("cleanup")
    if not cleanup:
        return
    try:
        res = cleanup()
        # 兼容 cleanup 可能是异步/同步的两种实现
        if inspect.isawaitable(res):
            loop = _get_persistent_loop()
            loop.run_until_complete(res)
    except Exception:
        # 退出时尽量静默
        pass

atexit.register(_cleanup_mcp_tools_at_exit)

# === MCP 响应格式化与事件循环兜底 ===
#
def _as_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    content = getattr(obj, "content", None)
    if isinstance(content, (str, bytes)):
        try:
            return json.loads(content if isinstance(content, str) else content.decode())
        except Exception:
            try:
                return ast.literal_eval(content if isinstance(content, str) else content.decode())
            except Exception:
                return {}
    if isinstance(obj, (str, bytes)):
        s = obj if isinstance(obj, str) else obj.decode()
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return {}
    try:
        return dict(obj)
    except Exception:
        return {}

def _format_blocks_from_mcp(resp) -> list[str]:
    data = _as_dict(resp)
    results = data.get("results") or []
    if isinstance(results, (str, bytes)):
        results = _as_dict(results).get("results", [])
    blocks: list[str] = []
    for r in results:
        rd = _as_dict(r)
        url = (rd.get("url") or "").strip()
        title = (rd.get("title") or "").strip()
        snippet = (rd.get("snippet") or "").strip()
        url_e = html.escape(url, quote=True)
        title_e = html.escape(title, quote=True)
        blocks.append(f'<Document origin="mcp:web" href="{url_e}" title="{title_e}"/>{snippet}</Document>')
    # 按 href 去重
    seen, uniq = set(), []
    for b in blocks:
        start = b.find('href="') + 6
        end = b.find('"', start)
        url = b[start:end] if start > 5 and end > start else b
        if url and url not in seen:
            uniq.append(b)
            seen.add(url)
    return uniq


# 持久事件循环（避免每次调用都新建/销毁 loop）
_persistent_loop: asyncio.AbstractEventLoop | None = None

def _get_persistent_loop() -> asyncio.AbstractEventLoop:
    global _persistent_loop
    # 如果当前线程已有 loop 且在运行（如 Jupyter），就复用它
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        try:
            import nest_asyncio  # 可选：pip install nest_asyncio
            nest_asyncio.apply()
        except Exception:
            pass
        return loop
    # 否则创建一个持久 loop 并复用
    if _persistent_loop is None or _persistent_loop.is_closed():
        _persistent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_persistent_loop)
    return _persistent_loop

def _run_async(coro):
    loop = _get_persistent_loop()
    return loop.run_until_complete(coro)