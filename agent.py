import json
import copy
import inspect

from openai import OpenAI
from pydantic import BaseModel
from typing_extensions import Literal
from typing import Union, Callable, List, Optional
from typing_extensions import NotRequired,TypedDict

def get_msg_title_repr(title: str) -> str:
    """Get a title representation for a message.

    Args:
        title: The title.

    Returns:
        The title representation.
    """
    padded = " " + title + " "
    sep_len = (80 - len(padded)) // 2
    sep = "=" * sep_len
    second_sep = sep + "=" if len(padded) % 2 else sep
    return (f"{sep}{padded}{second_sep}")

def prettyPrint(role,
                content="",
                id=None,
                name=None,
                tool_calls=None,
                invalid_tool_calls=None,
                refusal=None,
                audio=None,
                function_call=None,
                sender=None,
                tool_call_id=None,
                ) -> str:
        """Get a pretty representation of the message.
           It handles ChatCompletionMessage for all the roles i.e assistant, user, Tool.
           we are passing argumnents as **(dict(ChatCompletionMessage)) so many parameters like audio , refusal have nothing to do
           in prettyPrint have to set them to None

        Args:
            role: {'user','assistant','tool'}
            id : tool_call_id of tool_call
            name: name of the tool_call
            tool_calls: tools_calls by the assistant(model) mentioned in ChatCompletionMessage
            invalid_tool_calls: if any tool_calls have error then the attribute will be in ChatcompletionMessage of model(assistant)
            audio : only applicable for audio tokens


        Returns:
            A pretty representation of the message.
        """
        role = {"user": "Human", "assistant": "Ai"}.get(role, "Tool")
        title = get_msg_title_repr(role + " Message")
        if name is not None:
            title += f"\nName: {name}"
        print(f"{title}\n\n")
        if content:
            print(f"content:{content}")
        lines = []

        def _format_tool_args(tc: Union[ToolCall, InvalidToolCall]) -> list[str]:
            lines = [
                f"  {tc.get('name', 'Tool')} ({tc.get('id')})",
                f" Call ID: {tc.get('id')}",
            ]
            if tc.get("error"):
                lines.append(f"  Error: {tc.get('error')}")
            lines.append("  Args:")
            if tc.get("function"):
                args = tc.get("function").get("arguments")
            else:
                args = None
            if isinstance(args, str):
                lines.append(f"    {args}")
            elif isinstance(args, dict):
                for arg, value in args.items():
                    lines.append(f"    {arg}: {value}")
            return lines

        if tool_calls:
            lines.append("Tool Calls:")
            for tc in tool_calls:
                lines.extend(_format_tool_args(tc))
        if invalid_tool_calls:
            lines.append("Invalid Tool Calls:")
            for itc in self.invalid_tool_calls:
                lines.extend(_format_tool_args(itc))
        if lines:  # Check if lines is not None and not empty
            print("\n" + "\n".join(lines).strip())
        else:
            pass





def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")


def function_to_json(func) -> dict:
    """
    Sample Input:
    def add_two_numbers(a: int, b: int) -> int:
        # Adds two numbers together
        return a + b
    
    Sample Output:
    {
        'type': 'function',
        'function': {
            'name': 'add_two_numbers',
            'description': 'Adds two numbers together',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {'type': 'integer'},
                    'b': {'type': 'integer'}
                },
                'required': ['a', 'b']
            }
        }
    }
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

AgentFunction = Callable[[], Union[str, "Agent", dict]]

class Agent(BaseModel):
    # Just a simple class. Doesn't contain any methods out of the box
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True

class Response(BaseModel):
    # Response is used to encapsulate the entire conversation output
    messages: List = []
    agent: Optional[Agent] = None
    
class Function(BaseModel):
    arguments: Union[str,dict]
    name: str

class ChatCompletionMessageToolCall(BaseModel):
    id: str # The ID of the tool call
    function: Function # The function that the model called
    type: Literal["function"] # The type of the tool. Currently, only `function` is supported

class Result(BaseModel):
    # Result is used to encapsulate the return value of a single function/tool call
    value: str = "" # The result value as a string.
    agent: Optional[Agent] = None # The agent instance, if applicable.

class ToolCall(TypedDict):
    """Represents a request to call a tool.

    Example:

        .. code-block:: python

            {
                "name": "foo",
                "function": {'arguments': {'a'=2,'b'=3}},
                "id": "123"
            }

        This represents a request to call the tool named "foo" with arguments {"a": 1}
        and an identifier of "123".
    """

    name: str
    """The name of the tool to be called."""
    function: Function
    """The arguments to the tool call."""
    id: Optional[str]
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    """
    type: NotRequired[Literal["tool_call"]]

class InvalidToolCall(TypedDict):
    """Allowance for errors made by LLM.

    Here we add an `error` key to surface errors made during generation
    (e.g., invalid JSON arguments.)
    """

    name: Optional[str]
    """The name of the tool to be called."""
    args: Optional[str]
    """The arguments to the tool call."""
    id: Optional[str]
    """An identifier associated with the tool call."""
    error: Optional[str]
    """An error message associated with the tool call."""
    type: NotRequired[Literal["invalid_tool_call"]]


class Swarm:
    # Implements the core logic of orchestrating a single/multi-agent system
    def __init__(
        self,
        client=None,
    ):
        if not client:
            client = OpenAI()
        self.client = client
        self.history_stream=None
        self.bool_stream = False
    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        model_override: str
    ):
        messages = [{"role": "assistant", "content": agent.instructions}] + history
        tools = [function_to_json(f) for f in agent.functions]
        
        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
        }
        
        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls
        
        return self.client.chat.completions.create(**create_params)

    def handle_function_result(self, result) -> Result:
        match result:
            case Result() as result:
                return result
            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    raise TypeError(e)

    def handle_tool_calls(
        self,
        bool_stream,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction]
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None)
        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "id": tool_call.id,
                        "tool_call_id":tool_call.id,
                        "name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            raw_result = function_map[name](**args)
            if not bool_stream:
                print(f'Called function {name} with args: {args} and obtained result: {raw_result}')
                print('#############################################')
            result: Result = self.handle_function_result(raw_result)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "id": tool_call.id,
                    "tool_call_id":tool_call.id,
                    "name": name,
                    "content": result.value,
                }
            )
            if result.agent:
                partial_response.agent = result.agent

        return partial_response
    
    def run(
        self,
        agent: Agent,
        messages: List,
        model_override: str = None,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        active_agent = agent
        history = copy.deepcopy(messages)
        init_len = len(messages)

        print('#############################################')
        print(f'history: {history}')
        print('#############################################')
        while len(history) - init_len < max_turns and active_agent:
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                model_override=model_override
            )
            print("--------------------------------Result of the AI ----------------------------------")
            print(f"Type of completion{type(completion)}")
            print(f"Completion->{completion}")
            message = completion.choices[0].message
            print(f"type of message :{type(message)}")
            print(message)
            message.sender = active_agent.name
            print(f'Active agent: {active_agent.name}')
            print(f"message: {message}")
            print('#############################################')
            
            
            history.append(json.loads(message.model_dump_json()))

            if not message.tool_calls or not execute_tools:
                print('No tool calls hence breaking')
                print('#############################################')
                break
            
            partial_response = self.handle_tool_calls(self.bool_stream,message.tool_calls, active_agent.functions)
            history.extend(partial_response.messages)
            
            if partial_response.agent:
                active_agent = partial_response.agent
                message.sender = active_agent.name
        return Response(
            messages=history[init_len:],
            agent=active_agent,
        )
    def stream(self,
               agent:Agent,
               messages:List,
               max_turns: int = float("inf"),
               model_override: str = None,
               execute_tools: bool = True):
                self.bool_stream=True
                if self.history_stream==None:
                    self.history_stream = copy.deepcopy(messages) 
                else:
                    self.history_stream.extend(messages)
                active_agent = agent
                init_length  = len(self.history_stream)
                while(len(self.history_stream) - init_length <max_turns and active_agent):
                    prettyPrint(**self.history_stream[-1])
                    
                    response =  self.get_chat_completion(
                        agent=active_agent,
                        history=self.history_stream,
                        model_override=model_override)
                    
                    message = response.choices[0].message
                    message.sender = active_agent.name
                
                    self.history_stream.append(json.loads(message.model_dump_json()))
                    prettyPrint(**self.history_stream[-1])
                    

                    if not message.tool_calls or not execute_tools:
                        break

                    tool_response = self.handle_tool_calls(self.bool_stream,message.tool_calls, active_agent.functions)
                    self.history_stream.extend(tool_response.messages)

                    if tool_response.agent:
                        active_agent = tool_response.agent
                        message.sender = active_agent.name
                    