"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
import base64
import requests
import psycopg
from src.config.config import Config
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, ToolMessage
from src.node.supervisor_agent import SupervisorAgent
from src.observability.live_scoring import score_live_trace

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy-init agent

    def retrieve_docs(self, state: RAGState, config: RunnableConfig) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question, config=config)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description=(
                "Search the indexed corpus of documents the user has uploaded to this system "
                "(e.g. resumes, research papers, PDFs placed in the data/ directory). Use this "
                "for ANY question that could plausibly be about a specific person, project, "
                "paper, or topic covered in those documents - it is usually the right tool to "
                "try first, before general web/knowledge sources."
            ),
            func=retriever_tool_fn,
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description=(
                "Search Wikipedia for general public knowledge - well-known historical figures, "
                "events, science, places, etc. Not useful for private individuals or content "
                "specific to the user's own uploaded documents (use 'retriever' for that)."
            ),
            func=wiki.run,
        )

        github_tool = Tool(
            name="github",
            description=(
                "Look up GitHub info, but ONLY when the user explicitly asks about a GitHub "
                "profile, repository, or source code - not just because a question mentions a "
                "person's name. Pass 'user:<username>' for a user's profile and recent repos; "
                "'list:<owner>/<repo>[/path]' to browse a repo's files/folders (path optional, "
                "defaults to root); 'file:<owner>/<repo>/<path>' to fetch a specific file's "
                "actual source code; or a plain search string to find repositories by name/topic. "
                "To show real code, first use a plain search or 'list:' to find a repo/file, "
                "then use 'file:' to fetch its content."
            ),
            func=self._github_lookup,
        )

        scientist_db_tool = Tool(
            name="scientist_db",
            description=(
                "Search a structured PostgreSQL database of notable computer science / AI "
                "pioneers (name, field, what they're known for, birth year) by name or field "
                "- e.g. 'Geoffrey Hinton', 'Ashish Vaswani', 'machine learning'. Use this for "
                "questions asking about a specific scientist's field, contributions, or birth "
                "year. NOT for the user's own uploaded documents (use 'retriever' for those) "
                "or public figures outside this dataset (use 'wikipedia' for those)."
            ),
            func=self._scientist_db_lookup,
        )

        return [retriever_tool, wikipedia_tool, github_tool, scientist_db_tool]

    @staticmethod
    def _scientist_db_lookup(query: str) -> str:
        """Look up scientists by name or field in the Postgres 'scientists' table."""
        try:
            with psycopg.connect(Config.DATABASE_URL, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT name, field, known_for, birth_year FROM scientists "
                        "WHERE name ILIKE %s OR field ILIKE %s ORDER BY name LIMIT 5",
                        (f"%{query}%", f"%{query}%"),
                    )
                    rows = cur.fetchall()
        except psycopg.OperationalError as e:
            return f"Could not reach the scientists database: {e}"

        if not rows:
            return f"No scientist found matching '{query}' in the database."

        lines = []
        for name, field, known_for, birth_year in rows:
            year = birth_year if birth_year is not None else "unknown birth year"
            lines.append(f"- {name} ({field}, b. {year}): {known_for}")
        return "\n".join(lines)

    @staticmethod
    def _github_lookup(query: str) -> str:
        """Fetch a GitHub user profile or search repositories via the public REST API."""
        headers = {"Accept": "application/vnd.github+json"}
        if Config.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {Config.GITHUB_TOKEN}"

        try:
            if query.strip().lower().startswith("user:"):
                username = query.split(":", 1)[1].strip()

                user_resp = requests.get(
                    f"https://api.github.com/users/{username}",
                    headers=headers,
                    timeout=10,
                )
                if user_resp.status_code == 404:
                    return f"No GitHub user found for '{username}'."
                user_resp.raise_for_status()
                user = user_resp.json()

                repos_resp = requests.get(
                    f"https://api.github.com/users/{username}/repos",
                    params={"sort": "updated", "per_page": 5},
                    headers=headers,
                    timeout=10,
                )
                repos_resp.raise_for_status()
                repo_lines = "\n".join(
                    f"- {r['name']} ({r['stargazers_count']}⭐): {r.get('description') or 'No description'}"
                    for r in repos_resp.json()
                ) or "None found."

                return (
                    f"GitHub user: {user.get('login')}\n"
                    f"Name: {user.get('name') or 'N/A'}\n"
                    f"Bio: {user.get('bio') or 'N/A'}\n"
                    f"Public repos: {user.get('public_repos')}\n"
                    f"Followers: {user.get('followers')}\n"
                    f"Profile: {user.get('html_url')}\n\n"
                    f"Recent repos:\n{repo_lines}"
                )

            if query.strip().lower().startswith("list:"):
                target = query.split(":", 1)[1].strip().strip("/")
                parts = target.split("/", 2)
                if len(parts) < 2:
                    return "Usage: 'list:<owner>/<repo>[/path]'."
                owner, repo = parts[0], parts[1]
                path = parts[2] if len(parts) > 2 else ""

                resp = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code == 404:
                    return f"No such repo/path: '{owner}/{repo}/{path}'."
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    return f"'{path}' is a file, not a directory. Use 'file:{owner}/{repo}/{path}' instead."
                lines = "\n".join(f"- [{item['type']}] {item['path']}" for item in data)
                return f"Contents of {owner}/{repo}/{path or '(root)'}:\n{lines}"

            if query.strip().lower().startswith("file:"):
                target = query.split(":", 1)[1].strip().strip("/")
                parts = target.split("/", 2)
                if len(parts) < 3:
                    return "Usage: 'file:<owner>/<repo>/<path>'."
                owner, repo, path = parts

                resp = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code == 404:
                    return f"No such file: '{owner}/{repo}/{path}'."
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    return f"'{path}' is a directory. Use 'list:{owner}/{repo}/{path}' instead."
                if data.get("encoding") != "base64":
                    return f"Cannot preview '{path}' (not a text file or too large)."
                content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                truncated = content[:3000]
                suffix = "\n... (truncated)" if len(content) > 3000 else ""
                return f"--- {owner}/{repo}/{path} ---\n{truncated}{suffix}"

            search_resp = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc", "per_page": 5},
                headers=headers,
                timeout=10,
            )
            search_resp.raise_for_status()
            items = search_resp.json().get("items", [])
            if not items:
                return f"No GitHub repositories found for '{query}'."

            lines = "\n".join(
                f"- {r['full_name']} ({r['stargazers_count']}⭐): {r.get('description') or 'No description'}\n  {r['html_url']}"
                for r in items
            )
            return f"Top GitHub repositories for '{query}':\n{lines}"

        except requests.RequestException as e:
            return f"GitHub API error: {e}"

    def _build_agent(self):
        """Supervisor agent: knows about every available tool (name + description) and
        routes each step to the right one, instead of one flat ReAct loop guessing."""
        tools = self._build_tools()
        supervisor = SupervisorAgent(self.llm, tools)
        self._agent = supervisor.build()

    def generate_answer(self, state: RAGState, config: RunnableConfig) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]}, config=config)

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)
        answer = answer or "Could not generate answer."

        tool_context = "\n\n".join(m.content for m in messages if isinstance(m, ToolMessage))
        score_live_trace(self.llm, state.question, answer, tool_context)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer
        )
