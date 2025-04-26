from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import json
import time
import uuid

class AgentRole(str, Enum):
    """Standardized roles for agents in the protocol"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    FORMATTER = "formatter"

class AgentMessage(BaseModel):
    """Standardized message format for agent communication"""
    role: AgentRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=lambda: time.time())
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class AgentState(BaseModel):
    """State management for agent conversations"""
    messages: List[AgentMessage] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="idle")

class AgentProtocol:
    """Implementation of the agent-to-agent communication protocol"""
    
    def __init__(self):
        self.state = AgentState()
        self.agents: Dict[str, Any] = {}
    
    def register_agent(self, agent_id: str, agent: Any):
        """Register a new agent with the protocol"""
        self.agents[agent_id] = agent
    
    async def send_message(self, 
                          from_agent: str, 
                          to_agent: str, 
                          content: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """Send a message from one agent to another"""
        if to_agent not in self.agents:
            raise ValueError(f"Agent {to_agent} not found")
            
        message = AgentMessage(
            role=AgentRole.ASSISTANT,
            content=content,
            metadata=metadata or {}
        )
        
        # Add message to state
        self.state.messages.append(message)
        
        # Process message in target agent
        response = await self.agents[to_agent].process_message(message)
        
        return response
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[AgentMessage]:
        """Get the conversation history"""
        if limit:
            return self.state.messages[-limit:]
        return self.state.messages
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get the current state of a specific agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        return self.agents[agent_id].get_state()
    
    def update_context(self, context: Dict[str, Any]):
        """Update the shared context"""
        self.state.context.update(context)
    
    def clear_context(self):
        """Clear the shared context"""
        self.state.context.clear()

class BaseAgent:
    """Base class for all agents implementing the protocol"""
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.state = AgentState()
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process an incoming message and return a response"""
        raise NotImplementedError("Subclasses must implement process_message")
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent"""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "state": self.state.dict()
        }
    
    def update_state(self, state: Dict[str, Any]):
        """Update the agent's state"""
        self.state = AgentState(**state) 