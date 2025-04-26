from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import time
import uuid
from agent_protocol import AgentProtocol, AgentMessage, AgentRole
from specialized_agents import ResearchAgent, AnalysisAgent, FormattingAgent
from typing import AsyncGenerator
import logging
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
logger.debug("Initializing agents")
protocol = AgentProtocol()
research_agent = ResearchAgent("research-1")
analysis_agent = AnalysisAgent("analysis-1")
formatting_agent = FormattingAgent("formatting-1")

protocol.register_agent("researcher", research_agent)
protocol.register_agent("analyzer", analysis_agent)
protocol.register_agent("formatter", formatting_agent)
logger.debug("Agents initialized and registered")

async def generate_search_events(query: str):
    """Generate SSE events for the search process"""
    try:
        # Research phase
        logger.debug("Starting research phase")
        research_message = AgentMessage(
            role=AgentRole.USER,
            content=query,
            metadata={"research_type": "semantic_search"}
        )
        research_result = await research_agent.process_message(research_message)
        yield f"data: {json.dumps({'phase': 'research', 'content': research_result.content, 'reasoning': research_result.metadata.get('reasoning', 'Research completed')})}\n\n"
        logger.debug("Research phase completed")

        # Analysis phase
        logger.debug("Starting analysis phase")
        analysis_message = AgentMessage(
            role=AgentRole.RESEARCHER,
            content=research_result.content,
            metadata={"analysis_type": "comprehensive"}
        )
        analysis_result = await analysis_agent.process_message(analysis_message)
        yield f"data: {json.dumps({'phase': 'analysis', 'content': analysis_result.content, 'reasoning': analysis_result.metadata.get('reasoning', 'Analysis completed')})}\n\n"
        logger.debug("Analysis phase completed")

        # Formatting phase
        logger.debug("Starting formatting phase")
        formatting_message = AgentMessage(
            role=AgentRole.ANALYZER,
            content=analysis_result.content,
            metadata={"format_type": "markdown"}
        )
        formatting_result = await formatting_agent.process_message(formatting_message)
        yield f"data: {json.dumps({'phase': 'formatting', 'content': formatting_result.content, 'reasoning': formatting_result.metadata.get('reasoning', 'Formatting completed')})}\n\n"
        logger.debug("Formatting phase completed")

    except Exception as e:
        logger.error(f"Error in search process: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/")
async def read_root():
    """Serve the main page"""
    logger.debug("Serving main page")
    return FileResponse("static/index.html")

@app.get("/search")
async def search(query: str):
    """Handle search requests"""
    logger.debug(f"Received search query: {query}")
    return StreamingResponse(
        generate_search_events(query),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 