from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from agent_protocol import AgentProtocol, AgentMessage, AgentRole
from specialized_agents import ResearchAgent, AnalysisAgent, FormattingAgent
from typing import AsyncGenerator

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the agent protocol and register agents
protocol = AgentProtocol()
research_agent = ResearchAgent("researcher")
analysis_agent = AnalysisAgent("analyzer")
formatting_agent = FormattingAgent("formatter")

protocol.register_agent("researcher", research_agent)
protocol.register_agent("analyzer", analysis_agent)
protocol.register_agent("formatter", formatting_agent)

async def generate_search_events(query: str) -> AsyncGenerator[str, None]:
    """Generate SSE events for the search process using agent protocol"""
    try:
        # Create initial user message
        user_message = AgentMessage(
            role=AgentRole.USER,
            content=query,
            metadata={"query_type": "search"}
        )
        
        # Send starting event
        yield f'data: {json.dumps({"type": "status", "content": "I am analyzing your question to understand what you need..."})}\n\n'
        await asyncio.sleep(0.1)

        # Research phase
        yield f'data: {json.dumps({"type": "status", "content": "I am searching for relevant information..."})}\n\n'
        research_result = await protocol.send_message(
            "user",
            "researcher",
            query,
            {"research_type": "web_search"}
        )
        yield f'data: {json.dumps({"type": "research", "content": research_result.content})}\n\n'
        await asyncio.sleep(0.2)

        # Analysis phase
        yield f'data: {json.dumps({"type": "status", "content": "I am analyzing the information..."})}\n\n'
        analysis_result = await protocol.send_message(
            "researcher",
            "analyzer",
            research_result.content,
            {"analysis_type": "comprehensive"}
        )
        yield f'data: {json.dumps({"type": "analysis", "content": analysis_result.content})}\n\n'
        await asyncio.sleep(0.2)

        # Formatting phase
        yield f'data: {json.dumps({"type": "status", "content": "I am formatting the results..."})}\n\n'
        formatted_result = await protocol.send_message(
            "analyzer",
            "formatter",
            analysis_result.content,
            {"format_type": "markdown"}
        )
        
        # Send final results
        yield f'data: {json.dumps({"type": "results", "content": formatted_result.content})}\n\n'
        await asyncio.sleep(0.2)

        # Send completion event
        yield f'data: {json.dumps({"type": "complete", "content": "I have completed my search and prepared a response for you!"})}\n\n'

    except Exception as e:
        yield f'data: {json.dumps({"type": "error", "content": f"Sorry, I encountered an error while processing your request: {str(e)}"})}\n\n'

@app.get("/search")
async def search(request: Request, query: str):
    """SSE endpoint for search functionality"""
    return StreamingResponse(
        generate_search_events(query),
        media_type="text/event-stream"
    )

@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 