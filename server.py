from fastapi import FastAPI, Request, HTTPException, Header, Depends
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import sys
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# Enable LiteLLM debug mode for performance analysis
litellm.set_verbose = True
# litellm._turn_on_debug()  # Uncomment for even more verbose output

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
print("OPENAI_API_BASE:",os.environ.get("OPENAI_API_BASE","https://api.openai.com/v1"), flush=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to show detailed timing
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Set Base URL
BASE_URL = os.environ.get("BASE_URL")
if BASE_URL:
    parsed = urlparse(BASE_URL)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        logger.warning(f"Invalid BASE_URL ignored: {BASE_URL}")
        BASE_URL = None
    else:
        if BASE_URL.endswith('/'):
            BASE_URL = BASE_URL[:-1]
        logger.warning(f"Using BASE_URL (external prefix): {BASE_URL}")

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI(
    title="Claude2OpenAI Proxy",
    description="Proxy server that accepts Anthropic-style requests and forwards to OpenAI via LiteLLM. Provides OpenAPI schema and Swagger UI at /docs and /redoc.",
    version="0.1.0",
    contact={"name": "Maintainers", "url": "https://github.com/ziozzang/claude2openai-proxy"},
)

_original_openapi = app.openapi

def custom_openapi():
    schema = _original_openapi()
    if BASE_URL:
        schema["servers"] = [{"url": BASE_URL}]
    return schema

app.openapi = custom_openapi

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "65535"))

# Optional prefix-based mapping from env
BIG_PREFIXES = [p.strip().lower() for p in os.environ.get("BIG_PREFIXES", "opus,sonnet").split(",") if p.strip()]
SMALL_PREFIXES = [p.strip().lower() for p in os.environ.get("SMALL_PREFIXES", "haiku").split(",") if p.strip()]

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini" # Added default small model
]

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str = Field(..., description="Text content block")

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any] = Field(..., description="Anthropic image block source with type (e.g., base64, url) and media_type")

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str = Field(..., description="Unique tool invocation id")
    name: str = Field(..., description="Tool name")
    input: Dict[str, Any] = Field(default_factory=dict, description="Tool input JSON payload")

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str = Field(..., description="ID of corresponding tool_use event")
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any] = Field(
        ..., description="Tool result content; may be text, JSON object, or list of content blocks"
    )

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str = Field(..., description="System prompt text block")

class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]] = Field(
        ..., description="String content or list of content blocks per Anthropic spec"
    )

class Tool(BaseModel):
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for tool input")

class ThinkingConfig(BaseModel):
    enabled: Optional[bool] = Field(None, description="Enable thinking mode (Anthropic experimental)")
    type: Optional[str] = Field(None, description="Thinking type (e.g., 'enabled')")
    budget_tokens: Optional[int] = Field(None, description="Budget tokens for thinking")
    
    class Config:
        extra = "allow"  # Allow additional fields from client

class MessagesRequest(BaseModel):
    model: str = Field(..., description="Model id; accepts provider-prefixed (openai/..., anthropic/...) or will be normalized.", examples=["anthropic/claude-3-5-sonnet-20240620", "openai/gpt-4o-mini"])
    max_tokens: int = Field(..., description="Maximum output tokens", examples=[256])
    messages: List[Message] = Field(..., description="Conversation messages per Anthropic messages API")
    system: Optional[Union[str, List[SystemContent]]] = Field(None, description="System prompt as string or list of text blocks")
    stop_sequences: Optional[List[str]] = Field(None, description="Custom stop sequences")
    stream: Optional[bool] = Field(False, description="If true, returns SSE stream of Anthropic events")
    temperature: Optional[float] = Field(1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, description="Nucleus sampling top_p")
    top_k: Optional[int] = Field(None, description="Top-k sampling")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Opaque metadata passthrough")
    tools: Optional[List[Tool]] = Field(None, description="List of tools available to the model")
    tool_choice: Optional[Dict[str, Any]] = Field(None, description="Tool choice control (auto|any|tool with name)")
    thinking: Optional[ThinkingConfig] = Field(None, description="Anthropic thinking config")
    original_model: Optional[str] = Field(None, description="Original model before internal normalization")

    @field_validator('model')
    def validate_model_field(cls, v, info):
        original_model = v
        def map_model(name: str) -> str:
            clean = name
            if clean.startswith('anthropic/'):
                clean = clean[10:]
            elif clean.startswith('openai/'):
                clean = clean[7:]
            lower_clean = clean.lower()
            if any(lower_clean.startswith(p) for p in SMALL_PREFIXES):
                return f"openai/{SMALL_MODEL}"
            if any(lower_clean.startswith(p) for p in BIG_PREFIXES):
                return f"openai/{BIG_MODEL}"
            if clean in OPENAI_MODELS and not name.startswith('openai/'):
                return f"openai/{clean}"
            if not name.startswith(('openai/', 'anthropic/')):
                logger.debug(f"Model passthru: '{original_model}'.")
            return name
        new_model = map_model(v)
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model
        return new_model

class TokenCountRequest(BaseModel):
    model: str = Field(..., description="Model id used for tokenization context")
    messages: List[Message] = Field(..., description="Messages to count tokens for")
    system: Optional[Union[str, List[SystemContent]]] = Field(None, description="Optional system prompt")
    tools: Optional[List[Tool]] = Field(None, description="Tools (may affect tokenization)")
    thinking: Optional[ThinkingConfig] = Field(None, description="Thinking config")
    tool_choice: Optional[Dict[str, Any]] = Field(None, description="Tool choice control")
    original_model: Optional[str] = Field(None, description="Original model before normalization")

    @field_validator('model')
    def validate_model_token_count(cls, v, info):
        original_model = v
        def map_model(name: str) -> str:
            clean = name
            if clean.startswith('anthropic/'):
                clean = clean[10:]
            elif clean.startswith('openai/'):
                clean = clean[7:]
            lower_clean = clean.lower()
            if any(lower_clean.startswith(p) for p in SMALL_PREFIXES):
                return f"openai/{SMALL_MODEL}"
            if any(lower_clean.startswith(p) for p in BIG_PREFIXES):
                return f"openai/{BIG_MODEL}"
            if clean in OPENAI_MODELS and not name.startswith('openai/'):
                return f"openai/{clean}"
            if not name.startswith(('openai/', 'anthropic/')):
                logger.debug(f"Model passthru: '{original_model}'.")
            return name
        new_model = map_model(v)
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model
        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int = Field(..., description="Estimated prompt token count")

class Usage(BaseModel):
    input_tokens: int = Field(..., description="Prompt tokens")
    output_tokens: int = Field(..., description="Completion tokens")
    cache_creation_input_tokens: int = Field(0, description="Anthropic cache creation tokens")
    cache_read_input_tokens: int = Field(0, description="Anthropic cache read tokens")

class MessagesResponse(BaseModel):
    id: str = Field(..., description="Message id")
    model: str = Field(..., description="Model id used")
    role: Literal["assistant"] = Field("assistant", description="Always 'assistant' for responses")
    content: List[Union[ContentBlockText, ContentBlockToolUse]] = Field(..., description="Assistant content blocks")
    type: Literal["message"] = Field("message", description="Anthropic message type")
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = Field(None, description="Why generation stopped")
    stop_sequence: Optional[str] = Field(None, description="Matched stop sequence if any")
    usage: Usage = Field(..., description="Token usage")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item, ensure_ascii=False) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)
            
    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # For user messages with tool_result, split into separate messages
                text_content = ""
                
                # Extract all text parts and concatenate them
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            # Add tool result as a message by itself - simulate the normal flow
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            
                            # Handle different formats of tool result content
                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    # If content is a list of blocks, extract text from each
                                    for content_block in block.content:
                                        if hasattr(content_block, "type") and content_block.type == "text":
                                            result_content += content_block.text + "\n"
                                        elif isinstance(content_block, dict) and content_block.get("type") == "text":
                                            result_content += content_block.get("text", "") + "\n"
                                        elif isinstance(content_block, dict):
                                            # Handle any dict by trying to extract text or convert to JSON
                                            if "text" in content_block:
                                                result_content += content_block.get("text", "") + "\n"
                                            else:
                                                try:
                                                    result_content += json.dumps(content_block, ensure_ascii=False) + "\n"
                                                except:
                                                    result_content += str(content_block) + "\n"
                                elif isinstance(block.content, dict):
                                    # Handle dictionary content
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content, ensure_ascii=False)
                                        except:
                                            result_content = str(block.content)
                                else:
                                    # Handle any other type by converting to string
                                    try:
                                        result_content = str(block.content)
                                    except:
                                        result_content = "Unparseable content"
                            
                            # In OpenAI format, tool results come from the user (rather than being content blocks)
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                
                # Add as a single user message with all the content
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                # For assistant messages with tool_use/tool_result, convert to OpenAI tool_calls format
                # For user messages with non-text content, extract text only for LM Studio compatibility
                processed_content = []
                has_tool_use = False
                has_tool_result = False
                
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            # OpenAI format for text content
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            # Convert Anthropic image format to OpenAI image_url format
                            # Anthropic: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
                            # OpenAI: {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
                            if hasattr(block, "source") and isinstance(block.source, dict):
                                source = block.source
                                if source.get("type") == "base64":
                                    media_type = source.get("media_type", "image/png")
                                    data = source.get("data", "")
                                    image_url = f"data:{media_type};base64,{data}"
                                    processed_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": image_url}
                                    })
                                elif source.get("type") == "url":
                                    # Handle URL-based images
                                    url = source.get("url", "")
                                    processed_content.append({
                                        "type": "image_url",
                                        "image_url": {"url": url}
                                    })
                                else:
                                    logger.warning(f"Unsupported image source type: {source.get('type')}")
                            else:
                                logger.warning(f"Image block missing source: {block}")
                        elif block.type == "tool_use":
                            # LM Studio doesn't support tool_use in content blocks
                            # Convert to plain text representation
                            has_tool_use = True
                            tool_text = f"[Tool Use: {block.name}]\nInput: {json.dumps(block.input, ensure_ascii=False)}"
                            processed_content.append({"type": "text", "text": tool_text})
                        elif block.type == "tool_result":
                            # LM Studio doesn't support tool_result in content blocks
                            # Convert to plain text representation
                            has_tool_result = True
                            result_text = parse_tool_result_content(block.content if hasattr(block, "content") else "")
                            tool_result_text = f"[Tool Result]\n{result_text}"
                            processed_content.append({"type": "text", "text": tool_result_text})
                
                # If all content blocks are text-only, simplify to string format
                if len(processed_content) == 1 and processed_content[0].get("type") == "text":
                    messages.append({"role": msg.role, "content": processed_content[0]["text"]})
                elif all(block.get("type") == "text" for block in processed_content):
                    # Multiple text blocks - concatenate them
                    combined_text = "\n\n".join(block["text"] for block in processed_content)
                    messages.append({"role": msg.role, "content": combined_text})
                else:
                    # Mixed content types (text + image)
                    messages.append({"role": msg.role, "content": processed_content})
    
    # Cap max_tokens using env-defined MAX_TOKENS for OpenAI models
    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/"):
        max_tokens = min(max_tokens, MAX_TOKENS)
        logger.debug(f"Capping max_tokens to {MAX_TOKENS} for OpenAI model (original value: {anthropic_request.max_tokens})")
    
    # Prepare model name for LiteLLM
    # When using custom api_base (like LM Studio), preserve the exact model name
    model_for_litellm = anthropic_request.model
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    
    # If using custom api_base (LM Studio), strip provider prefixes
    # because LiteLLM will strip them anyway and we want consistent behavior
    if openai_api_base:
        if model_for_litellm.startswith("openai/"):
            model_for_litellm = model_for_litellm[7:]  # Remove "openai/" prefix
            logger.debug(f"Stripped openai/ prefix for custom endpoint: {model_for_litellm}")
        elif model_for_litellm.startswith("anthropic/"):
            model_for_litellm = model_for_litellm[10:]  # Remove "anthropic/" prefix
            logger.debug(f"Stripped anthropic/ prefix for custom endpoint: {model_for_litellm}")
        logger.debug(f"Using custom api_base with model: {model_for_litellm}")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": model_for_litellm,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }
    
    # Add api_base from environment if set (for LM Studio, etc.)
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    if openai_api_base:
        litellm_request["api_base"] = openai_api_base
        # Use custom_llm_provider to prevent LiteLLM from modifying the model name
        # This ensures the exact model name is sent to LM Studio
        litellm_request["custom_llm_provider"] = "openai"
        logger.debug(f"Using custom endpoint: {openai_api_base} with model: {model_for_litellm}")
    
    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}")
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format) - only for Claude models
        if tool_calls and is_claude_model:
            logger.debug(f"Processing tool calls: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}
                
                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
                
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        elif tool_calls and not is_claude_model:
            # For non-Claude models, convert tool calls to text format
            logger.debug(f"Converting tool calls to text for non-Claude model: {clean_model}")
            
            # We'll append tool info to the text content
            tool_text = "\n\nTool usage:\n"
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        args_dict = json.loads(arguments)
                        arguments_str = json.dumps(args_dict, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, indent=2, ensure_ascii=False)
                
                tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"
            
            # Add or append tool text to content
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data, ensure_ascii=False)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}, ensure_ascii=False)}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'}, ensure_ascii=False)}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # Process each chunk
        async for chunk in response_generator:
            try:

                
                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}}, ensure_ascii=False)}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0}, ensure_ascii=False)}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}}, ensure_ascii=False)}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0}, ensure_ascii=False)}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0}, ensure_ascii=False)}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}}, ensure_ascii=False)}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments, ensure_ascii=False)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments
                                
                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""
                                
                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}}, ensure_ascii=False)}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i}, ensure_ascii=False)}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}}, ensure_ascii=False)}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0}, ensure_ascii=False)}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage}, ensure_ascii=False)}\n\n"
                        
                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'}, ensure_ascii=False)}\n\n"
                        
                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i}, ensure_ascii=False)}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0}, ensure_ascii=False)}\n\n"
            
            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage}, ensure_ascii=False)}\n\n"
            
            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'}, ensure_ascii=False)}\n\n"
            
            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}}, ensure_ascii=False)}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'}, ensure_ascii=False)}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post(
    "/v1/messages",
    summary="Create message (Anthropic-compatible)",
    description=(
        "Accepts Anthropic Messages API-compatible body and optionally streams Anthropic SSE events when stream=true.\n"
        "SSE event order: message_start → content_block_start (text) → content_block_delta(s) → content_block_stop;\n"
        "optional tool_use content_block_start/delta/stop; then message_delta(stop_reason, usage) → message_stop → [DONE].\n"
        "Send X-API-Key header with upstream OpenAI key."
    ),
)
async def create_message(
    request: MessagesRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", description="OpenAI API key used upstream"),
):
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8') or '{}')

        # Determine model: pass through if provided; else BIG/SMALL defaults
        incoming_model = body_json.get("model")
        if incoming_model:
            resolved_model = incoming_model
        else:
            size_hint = body_json.get("size")
            if size_hint == "small" and SMALL_MODEL:
                resolved_model = f"openai/{SMALL_MODEL}"
            else:
                resolved_model = f"openai/{BIG_MODEL}"

        # If no explicit provider prefix, assume openai
        if not resolved_model.startswith(("openai/", "anthropic/")):
            resolved_model = f"openai/{resolved_model}"

        # Inject resolved model into pydantic request for downstream conversion
        request.model = resolved_model

        # Logging names
        original_model = incoming_model or resolved_model
        display_model = original_model.split("/")[-1] if "/" in original_model else original_model

        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]

        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)

        # Build backend auth headers based on provider; prefer passthrough
        incoming_x_api_key = x_api_key or raw_request.headers.get("x-api-key") or raw_request.headers.get("X-API-Key")
        
        # Also check Authorization header (Bearer token)
        if not incoming_x_api_key:
            auth_header = raw_request.headers.get("authorization") or raw_request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                incoming_x_api_key = auth_header[7:]  # Remove "Bearer " prefix
        
        # Fallback to environment variable if no key provided (for LM Studio, etc.)
        if not incoming_x_api_key and OPENAI_API_KEY:
            incoming_x_api_key = OPENAI_API_KEY
            logger.debug("Using OPENAI_API_KEY from environment")
        
        backend_headers = {}
        
        if incoming_x_api_key:
            os.environ['OPENAI_API_KEY'] = incoming_x_api_key.strip()
            backend_headers["Authorization"] = f"Bearer {incoming_x_api_key.strip()}"
        else:
            raise HTTPException(status_code=401, detail="Missing Authorization for OpenAI request")
        ua = raw_request.headers.get("user-agent") or raw_request.headers.get("User-Agent")
        if ua:
            backend_headers["User-Agent"] = ua
        litellm_request["headers"] = backend_headers
        
        # For OpenAI models - modify request format to work with limitations
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            
            # For OpenAI models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # Special case - handle message content directly when it's a list of tool_result
                # This is a specific case we're seeing in the error
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])
                            
                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get("text", json.dumps(item, ensure_ascii=False))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content, ensure_ascii=False) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"
                        
                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                        continue  # Skip normal processing for this message
                
                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                
                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"
                                    
                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item, ensure_ascii=False) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content, ensure_ascii=False) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content, ensure_ascii=False) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                
                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}), ensure_ascii=False)
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                                
                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"
                        
                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."
                        
                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # Empty content not allowed
                
                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]
            
            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
                
                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'), ensure_ascii=False)}")
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'), ensure_ascii=False)}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..." # Fallback placeholder
        
        # Log request details before sending to LiteLLM
        total_content_length = sum(len(str(msg.get('content', ''))) for msg in litellm_request.get('messages', []))
        logger.info(f"🚀 LITELLM REQUEST: model={litellm_request.get('model')}, "
                   f"api_base={litellm_request.get('api_base', 'default')}, "
                   f"messages={len(litellm_request.get('messages', []))}, "
                   f"content_length={total_content_length}, "
                   f"max_tokens={litellm_request.get('max_tokens')}, "
                   f"stream={litellm_request.get('stream', False)}")
        
        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Ensure we use the async version for streaming
            logger.debug(f"Calling litellm.acompletion with model: {litellm_request.get('model')}")
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            start_time = time.time()
            logger.debug(f"Calling litellm.completion with model: {litellm_request.get('model')}")
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        # Log all error details (use default=str to handle non-serializable objects)
        try:
            logger.error(f"Error processing request: {json.dumps(error_details, indent=2, ensure_ascii=False, default=str)}")
        except Exception:
            # Fallback: convert all values to strings
            safe_error_details = {k: str(v) for k, v in error_details.items()}
            logger.error(f"Error processing request: {json.dumps(safe_error_details, indent=2, ensure_ascii=False)}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        # Return detailed error
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post(
    "/v1/messages/count_tokens",
    summary="Count tokens (Anthropic-compatible)",
    description="Estimates input token count for the provided messages/system using upstream model tokenizer.",
)
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", description="OpenAI API key used upstream"),
):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Resolve model similar to /v1/messages: pass-through or BIG/SMALL with openai default
        original_incoming_model = request.original_model or request.model
        resolved_model = original_incoming_model
        if not resolved_model:
            size_hint = None
            resolved_model = f"openai/{SMALL_MODEL}" if size_hint == "small" and SMALL_MODEL else f"openai/{BIG_MODEL}"
        if not resolved_model.startswith(("openai/", "anthropic/")):
            resolved_model = f"openai/{resolved_model}"

        request.model = resolved_model

        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Count tokens
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )
            
            # Return Anthropic-style response
            os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    docs = f"{BASE_URL}/docs" if BASE_URL else "/docs"
    redoc = f"{BASE_URL}/redoc" if BASE_URL else "/redoc"
    openapi = f"{BASE_URL}/openapi.json" if BASE_URL else "/openapi.json"
    return {"message": "Anthropic Proxy for LiteLLM", "docs": docs, "redoc": redoc, "openapi": openapi}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
