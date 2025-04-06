from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from agents import build_multi_node_graph, AgentState, ChatOpenAI, ReasoningModel
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

# Initialize the models
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
reasoner = ReasoningModel(llm)

async def process_node(state: AgentState, node_type: str, context: dict = None) -> str:
    """Process a single node and return its result"""
    try:
        response = await asyncio.to_thread(
            reasoner.reason,
            node_type,
            state.messages[-1]['content'] if state.messages else '',
            context or {"task": f"{node_type}_task"}
        )
        return response
    except Exception as e:
        return f"Error in {node_type}: {str(e)}"

async def generate_search_events(query: str) -> AsyncGenerator[str, None]:
    """Generate SSE events for the search process"""
    try:
        # Create initial state
        state = AgentState(
            messages=[
                {"role": "user", "content": query}
            ]
        )

        # Send starting event
        yield f"data: {json.dumps({'type': 'status', 'content': 'Starting analysis...'})}\n\n"
        await asyncio.sleep(0.1)  # Small delay for UI update

        # Query Analysis
        yield f"data: {json.dumps({'type': 'status', 'content': '🤔 Analyzing query...'})}\n\n"
        analysis_result = await process_node(state, "analysis")
        state.context["query_analysis"] = analysis_result
        yield f"data: {json.dumps({'type': 'analysis', 'content': analysis_result})}\n\n"
        await asyncio.sleep(0.2)  # Small delay between steps

        # Query Expansion
        yield f"data: {json.dumps({'type': 'status', 'content': '🔍 Expanding search terms...'})}\n\n"
        expansion_result = await process_node(
            state, 
            "expansion",
            {"task": "query_expansion", "analysis": analysis_result}
        )
        state.context["expanded_queries"] = expansion_result
        yield f"data: {json.dumps({'type': 'expansion', 'content': expansion_result})}\n\n"
        await asyncio.sleep(0.2)

        # Semantic Analysis
        yield f"data: {json.dumps({'type': 'status', 'content': '🧠 Performing semantic analysis...'})}\n\n"
        semantic_result = await process_node(
            state,
            "semantics",
            {
                "task": "semantic_analysis",
                "analysis": analysis_result,
                "expansion": expansion_result
            }
        )
        state.context["semantic_analysis"] = semantic_result
        yield f"data: {json.dumps({'type': 'semantics', 'content': semantic_result})}\n\n"
        await asyncio.sleep(0.2)

        # Web Search
        yield f"data: {json.dumps({'type': 'status', 'content': '🌐 Searching...'})}\n\n"
        search_result = await process_node(
            state,
            "search",
            {
                "task": "web_search",
                "analysis": analysis_result,
                "expansion": expansion_result,
                "semantics": semantic_result
            }
        )
        
        # Add search results to state
        state.messages.append({
            "role": "assistant",
            "content": f"Search Results:\n\n{search_result}"
        })
        state.context["search_results"] = search_result
        
        # Send search results
        yield f"data: {json.dumps({'type': 'results', 'content': search_result})}\n\n"
        await asyncio.sleep(0.2)

        # Send completion event
        yield f"data: {json.dumps({'type': 'complete', 'content': 'Search completed'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

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