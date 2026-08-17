"""Supervisor agent: routes each step to the right tool instead of relying on a single
flat ReAct loop to guess it. One single-tool worker agent is built per available tool;
an LLM supervisor is told each tool's name and description and picks which worker acts
next, looping until it has enough information to answer."""

from typing import Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import Field, create_model


class SupervisorState(MessagesState):
    next: str
    steps: int


class SupervisorAgent:
    """Builds a supervisor node + one single-tool ReAct worker per tool, coordinated by an LLM router."""

    def __init__(self, llm, tools: List[BaseTool], max_steps: int = 4):
        self.llm = llm
        self.tools_by_name: Dict[str, BaseTool] = {t.name: t for t in tools}
        self.max_steps = max_steps
        self._workers: Optional[Dict[str, object]] = None
        self._router = None
        self._system_prompt: Optional[str] = None

    def _build(self):
        self._workers = {
            name: create_react_agent(self.llm, tools=[tool])
            for name, tool in self.tools_by_name.items()
        }

        tool_list = "\n".join(
            f"- '{name}': {tool.description}" for name, tool in self.tools_by_name.items()
        )
        self._system_prompt = (
            "You are a supervisor coordinating specialized worker agents, one per tool "
            "available in this system:\n"
            f"{tool_list}\n\n"
            "Given the conversation so far, decide which single tool's worker should act "
            "next to help answer the user's question. Prefer a document-retrieval tool "
            "first when the question could plausibly be answered from the user's own "
            "uploaded documents. Only route to other tools for information the documents "
            "clearly don't cover. Once enough information has been gathered to answer the "
            "question, respond FINISH."
        )

        route_options = tuple(self.tools_by_name.keys()) + ("FINISH",)
        route_model = create_model(
            "Route",
            next=(
                Literal[route_options],
                Field(description="Which tool's worker should act next, or FINISH if ready to answer."),
            ),
        )
        self._router = self.llm.with_structured_output(route_model)

    def _supervisor(self, state: SupervisorState, config: RunnableConfig) -> dict:
        if self._router is None:
            self._build()

        steps = state.get("steps", 0) + 1
        if steps > self.max_steps:
            return {"next": "FINISH", "steps": steps}

        messages = [SystemMessage(content=self._system_prompt)] + state["messages"]
        route = self._router.invoke(messages, config=config)
        return {"next": route.next, "steps": steps}

    def _call_worker(self, name: str):
        def _node(state: SupervisorState, config: RunnableConfig) -> dict:
            worker = self._workers[name]
            result = worker.invoke({"messages": state["messages"]}, config=config)
            return {"messages": result["messages"][len(state["messages"]):]}

        return _node

    def _synthesize(self, state: SupervisorState, config: RunnableConfig) -> dict:
        question = state["messages"][0].content if state["messages"] else ""
        wrap_up = HumanMessage(
            content=(
                "Using the conversation and tool results above, give a final, clear answer "
                f"to the original question: {question}\n\n"
                "If the tools found no relevant information, say so plainly rather than guessing."
            )
        )
        response = self.llm.invoke(state["messages"] + [wrap_up], config=config)
        return {"messages": [response]}

    def _route(self, state: SupervisorState) -> str:
        next_ = state.get("next", "FINISH")
        return "synthesize" if next_ == "FINISH" else next_

    def build(self):
        builder = StateGraph(SupervisorState)
        builder.add_node("supervisor", self._supervisor)
        builder.add_node("synthesize", self._synthesize)
        for name in self.tools_by_name:
            builder.add_node(name, self._call_worker(name))
            builder.add_edge(name, "supervisor")

        builder.set_entry_point("supervisor")
        builder.add_conditional_edges("supervisor", self._route)
        builder.add_edge("synthesize", END)

        return builder.compile()
