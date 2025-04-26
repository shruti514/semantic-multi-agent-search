from typing import Dict, Any, List
from agent_protocol import BaseAgent, AgentMessage, AgentRole
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time
import uuid
import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import logging
from langchain_core.messages import AIMessage

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """Agent specialized in conducting research and finding information"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentRole.RESEARCHER)
        logger.debug(f"Initializing ResearchAgent with ID: {agent_id}")
        self.llm = ChatOpenAI()
        self.query_expansion_prompt = PromptTemplate(
            template="""Given the following search query, generate 3 different variations that capture different aspects or perspectives of the query.
            Original query: {query}
            
            Think step by step:
            1. Identify the main topic and key concepts
            2. Consider different angles or perspectives
            3. Think about related concepts or aspects
            4. Generate variations that cover these different aspects
            
            Return the variations as a JSON array of strings, and include your reasoning.""",
            input_variables=["query"]
        )
        self.search_prompt = PromptTemplate(
            template="""Search for information about: {query}
            Focus on: {aspect}
            
            Think step by step:
            1. What specific information is needed for this aspect?
            2. What are the key points to look for?
            3. How does this aspect relate to the main query?
            
            Return relevant information in a structured format, and include your reasoning.""",
            input_variables=["query", "aspect"]
        )
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process research requests"""
        logger.debug(f"ResearchAgent processing message: {message.content}")
        # Extract query from message
        query = message.content
        metadata = message.metadata
        
        # Perform query expansion
        logger.debug("Starting query expansion")
        expanded_queries = await self._expand_query(query)
        logger.debug(f"Expanded queries: {expanded_queries}")
        
        # Perform parallel searches
        logger.debug("Starting parallel searches")
        search_tasks = [
            self._conduct_search(query, aspect)
            for aspect in expanded_queries
        ]
        search_results = await asyncio.gather(*search_tasks)
        logger.debug(f"Search results received: {len(search_results)}")
        
        # Combine and rank results
        logger.debug("Starting result ranking")
        combined_results = await self._rank_results(search_results, query)
        logger.debug("Results ranked and combined")
        
        # Create response message with reasoning
        response = AgentMessage(
            role=AgentRole.RESEARCHER,
            content=combined_results,
            metadata={
                "query": query,
                "expanded_queries": expanded_queries,
                "research_type": metadata.get("research_type", "semantic_search"),
                "reasoning": "Research completed: " + str(len(search_results)) + " results found and analyzed"
            }
        )
        logger.debug("ResearchAgent response created")
        return response
    
    async def _expand_query(self, query: str) -> List[str]:
        """Expand the query into multiple semantic variations"""
        logger.debug(f"Expanding query: {query}")
        chain = self.query_expansion_prompt | self.llm | JsonOutputParser()
        result = await chain.ainvoke({"query": query})
        logger.debug(f"Query expansion result: {result}")
        
        # Extract variations from the result
        if isinstance(result, dict) and "variations" in result:
            variations = result["variations"]
        elif isinstance(result, list):
            variations = result
        else:
            # Generic fallback variations
            variations = [
                f"{query} overview and background",
                f"{query} key aspects and features",
                f"{query} impact and significance",
                f"{query} current status and developments"
            ]
        
        return variations
    
    async def _conduct_search(self, query: str, aspect: str) -> str:
        """Conduct a search for a specific aspect of the query"""
        logger.debug(f"Conducting search for aspect: {aspect}")
        chain = self.search_prompt | self.llm
        result = await chain.ainvoke({"query": query, "aspect": aspect})
        # Extract content from AIMessage if needed
        if isinstance(result, AIMessage):
            result = result.content
        logger.debug(f"Search result length: {len(result)}")
        return result
    
    async def _rank_results(self, results: List[str], original_query: str) -> str:
        """Rank and combine search results"""
        logger.debug("Starting result ranking")
        ranking_prompt = PromptTemplate(
            template="""Given the following search results and original query, combine and rank them by relevance.
            Original query: {query}
            Results to rank:
            {results}
            
            Think step by step:
            1. Evaluate the relevance of each result to the original query
            2. Identify overlapping or complementary information
            3. Determine the most important points from each result
            4. Organize the information in a logical flow
            
            Return a comprehensive response that combines the most relevant information, and include your reasoning.""",
            input_variables=["query", "results"]
        )
        chain = ranking_prompt | self.llm
        combined_results = "\n\n".join([f"Result {i+1}:\n{result}" for i, result in enumerate(results)])
        ranked_result = await chain.ainvoke({
            "query": original_query,
            "results": combined_results
        })
        # Extract content from AIMessage if needed
        if isinstance(ranked_result, AIMessage):
            ranked_result = ranked_result.content
        logger.debug("Results ranked successfully")
        return ranked_result

class AnalysisAgent(BaseAgent):
    """Agent specialized in analyzing information and providing direct answers"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentRole.ANALYZER)
        logger.debug(f"Initializing AnalysisAgent with ID: {agent_id}")
        self.llm = ChatOpenAI()
        self.analysis_prompt = PromptTemplate(
            template="""You are an AI assistant tasked with providing a direct answer to the user's question based on the following search results.

            User's Question: {query}
            
            Search Results:
            {content}

            Your task is to:
            1. Understand the user's question and what they're looking for
            2. Extract relevant information from the search results
            3. Synthesize the information into a clear, comprehensive answer
            4. Provide specific details and examples where relevant
            5. Address different aspects of the question if it's multi-faceted
            6. Be objective and factual, citing information from the search results
            7. If there are conflicting viewpoints, present them fairly
            8. If information is incomplete, acknowledge the limitations

            Format your answer as a direct response to the user's question, using:
            - Clear, concise language
            - Logical organization
            - Specific examples and details
            - Proper context and background where needed

            Remember: Your goal is to provide a helpful, accurate answer that directly addresses the user's question.""",
            input_variables=["content", "query"]
        )
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process analysis requests"""
        logger.debug("AnalysisAgent processing message")
        content = message.content
        metadata = message.metadata
        
        # Extract the original query from metadata
        original_query = metadata.get("query", "")
        
        analysis_result = await self._analyze_content(content, original_query)
        logger.debug("Analysis completed")
        
        response = AgentMessage(
            role=AgentRole.ANALYZER,
            content=analysis_result,
            metadata={
                "analysis_type": metadata.get("analysis_type", "comprehensive"),
                "reasoning": "Analysis completed: Generated a direct answer based on the search results"
            }
        )
        return response
    
    async def _analyze_content(self, content: str, query: str) -> str:
        """Analyze the content and provide a direct answer"""
        logger.debug("Starting content analysis")
        chain = self.analysis_prompt | self.llm
        result = await chain.ainvoke({"content": content, "query": query})
        # Extract content from AIMessage if needed
        if isinstance(result, AIMessage):
            result = result.content
        logger.debug("Analysis completed")
        return result

class FormattingAgent(BaseAgent):
    """Agent specialized in formatting content"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentRole.FORMATTER)
        logger.debug(f"Initializing FormattingAgent with ID: {agent_id}")
        self.llm = ChatOpenAI()
        self.formatting_prompt = PromptTemplate(
            template="""Format the following content in a clear, organized, and visually appealing way using Markdown formatting:

            {content}

            Follow these formatting guidelines:
            1. Use proper heading hierarchy (H1, H2, H3)
            2. Create bullet points for lists
            3. Use bold and italic text for emphasis
            4. Include code blocks where appropriate
            5. Add horizontal rules to separate major sections
            6. Use blockquotes for important quotes or highlights
            7. Create tables for structured data
            8. Add links to relevant resources
            9. Use proper spacing and indentation

            Return the formatted content in Markdown format. Make sure to:
            - Start with a clear title
            - Organize information in logical sections
            - Use appropriate formatting for different types of content
            - Make the content easy to scan and read
            - Highlight key points and important information""",
            input_variables=["content"]
        )
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process formatting requests"""
        logger.debug("FormattingAgent processing message")
        content = message.content
        metadata = message.metadata
        
        formatted_result = await self._format_content(content, metadata)
        logger.debug("Formatting completed")
        
        response = AgentMessage(
            role=AgentRole.FORMATTER,
            content=formatted_result,
            metadata={
                "format_type": metadata.get("format_type", "markdown"),
                "reasoning": "Formatting completed: Content organized with proper Markdown formatting for better readability"
            }
        )
        return response
    
    async def _format_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format the content according to specified style"""
        logger.debug("Starting content formatting")
        chain = self.formatting_prompt | self.llm
        result = await chain.ainvoke({"content": content})
        # Extract content from AIMessage if needed
        if isinstance(result, AIMessage):
            result = result.content
        logger.debug("Formatting completed")
        return result 