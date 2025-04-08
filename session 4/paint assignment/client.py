import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
from google import genai
from concurrent.futures import TimeoutError
from functools import partial

# Load environment variables from .env file
load_dotenv()

# Access your API key and initialize Gemini client correctly
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

max_iterations = 9
last_response = None
iteration = 0
iteration_response = []

async def generate_with_timeout(client, prompt, timeout=10):
    """Generate content with a timeout"""
    print("Starting LLM generation...")
    try:
        # Convert the synchronous generate_content call to run in a thread
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        print("LLM generation completed")
        return response
    except TimeoutError:
        print("LLM generation timed out!")
        raise
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        raise

def reset_state():
    """Reset all global variables to their initial state"""
    global last_response, iteration, iteration_response
    last_response = None
    iteration = 0
    iteration_response = []

async def main():
    reset_state()  # Reset at the start of main
    print("Starting main execution...")
    try:
        # Create a single MCP server connection
        print("Establishing connection to MCP server...")
        server_params = StdioServerParameters(
            command="python",
            args=["server.py"]
        )

        async with stdio_client(server_params) as (read, write):
            print("Connection established, creating session...")
            async with ClientSession(read, write) as session:
                print("Session created, initializing...")
                await session.initialize()
                
                # Get available tools
                print("Requesting tool list...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                print(f"Successfully retrieved {len(tools)} tools")

                tools_description = []
                for i, tool in enumerate(tools):
                    try:
                        params = tool.inputSchema
                        desc = getattr(tool, 'description', 'No description available')
                        name = getattr(tool, 'name', f'tool_{i}')
                        
                        if 'properties' in params:
                            param_details = []
                            for param_name, param_info in params['properties'].items():
                                param_type = param_info.get('type', 'unknown')
                                param_details.append(f"{param_name}: {param_type}")
                            params_str = ', '.join(param_details)
                        else:
                            params_str = 'no parameters'

                        tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                        tools_description.append(tool_desc)
                    except Exception as e:
                        tools_description.append(f"{i+1}. Error processing tool")
                
                tools_description = "\n".join(tools_description)
                print("Created system prompt...")
                
                system_prompt = f"""You are a math agent solving problems in iterations. You have access to various mathematical tools.

Available tools:
{tools_description}

You have a unique ability to draw rectangles and add text to the canvas. You must:
1. Calculate the result first
2. Then draw a rectangle in Paint
3. Add the result as text inside the rectangle
4. Give the final answer

You must respond with EXACTLY ONE line in one of these formats (no additional text):
1. For function calls:
   FUNCTION_CALL: function_name|param1|param2|...

2. For calculations:
   CALCULATION_ANSWER: [number]
   
3. For final answer:
   FINAL_ANSWER: [number]

Important:
- Each operation must be a separate function call
- Wait for each operation to complete before the next one
- When adding text, specify the x and y coordinates

Example sequence:
FUNCTION_CALL: strings_to_chars_to_int|INDIA
CALCULATION_ANSWER: [42]
FUNCTION_CALL: open_paint
FUNCTION_CALL: select_rectangle_tool
FUNCTION_CALL: draw_rectangle|500|500|800|800
FUNCTION_CALL: add_text_in_paint|[42]|550|550
FINAL_ANSWER: [42]

DO NOT include any explanations or additional text.
Your response should be a single line starting with FUNCTION_CALL:, CALCULATION_ANSWER:, or FINAL_ANSWER:"""

                query = """Find the ASCII values of characters in INDIA and then return sum of exponentials of those values. Draw the result in Paint."""
                print("Starting iteration loop...")
                
                # Use global iteration variables
                global iteration, last_response
                
                while iteration < max_iterations:
                    print(f"\n--- Iteration {iteration + 1} ---")
                    if last_response is None:
                        current_query = query
                    else:
                        current_query = current_query + "\n\n" + " ".join(iteration_response)
                        current_query = current_query + "  What should I do next?"

                    # Get model's response with timeout
                    print("Preparing to generate LLM response...")
                    prompt = f"{system_prompt}\n\nQuery: {current_query}"
                    try:
                        response = await generate_with_timeout(client, prompt)
                        response_text = response.text.strip()
                        print(f"LLM Response: {response_text}")
                        
                        # Process all lines in the response
                        for line in response_text.split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                                
                            print(f"Processing line: {line}")
                            
                            if line.startswith("FUNCTION_CALL:"):
                                _, function_info = line.split(":", 1)
                                parts = [p.strip() for p in function_info.split("|")]
                                func_name, params = parts[0], parts[1:]
                                
                                try:
                                    # Find the matching tool to get its input schema
                                    tool = next((t for t in tools if t.name == func_name), None)
                                    if not tool:
                                        raise ValueError(f"Unknown tool: {func_name}")

                                    arguments = {}
                                    schema_properties = tool.inputSchema.get('properties', {})

                                    for param_name, param_info in schema_properties.items():
                                        if not params:
                                            raise ValueError(f"Not enough parameters provided for {func_name}")
                                            
                                        value = params.pop(0)
                                        param_type = param_info.get('type', 'string')
                                        
                                        if param_type == 'integer':
                                            arguments[param_name] = int(value)
                                        elif param_type == 'number':
                                            arguments[param_name] = float(value)
                                        elif param_type == 'array':
                                            # Handle array input
                                            if isinstance(value, str):
                                                value = value.strip('[]').split(',')
                                            arguments[param_name] = [int(x.strip()) for x in value]
                                        else:
                                            arguments[param_name] = str(value)
                                    
                                    result = await session.call_tool(func_name, arguments=arguments)
                                    
                                    # Get the full result content
                                    if hasattr(result, 'content'):
                                        if isinstance(result.content, list):
                                            iteration_result = [
                                                item.text if hasattr(item, 'text') else str(item)
                                                for item in result.content
                                            ]
                                        else:
                                            iteration_result = str(result.content)
                                    else:
                                        iteration_result = str(result)
                                    
                                    # Format the response based on result type
                                    if isinstance(iteration_result, list):
                                        result_str = f"[{', '.join(iteration_result)}]"
                                    else:
                                        result_str = str(iteration_result)
                                    
                                    iteration_response.append(
                                        f"In the {iteration + 1} iteration you called {func_name} with {arguments} parameters, "
                                        f"and the function returned {result_str}."
                                    )
                                    last_response = iteration_result

                                    # Add delay after each Paint operation
                                    if func_name in ['open_paint', 'select_rectangle_tool', 'draw_rectangle', 'add_text_in_paint']:
                                        await asyncio.sleep(1)

                                except Exception as e:
                                    iteration_response.append(f"Error in iteration {iteration + 1}: {str(e)}")
                                    break
                                    
                            elif line.startswith("CALCULATION_ANSWER:"):
                                print(f"Got calculation result: {line}")
                                last_response = line
                                iteration_response.append(line)
                                
                            elif line.startswith("FINAL_ANSWER:"):
                                print("\n=== Agent Execution Complete ===")
                                break
                                
                    except Exception as e:
                        print(f"Failed to get LLM response: {e}")
                        break

                    iteration += 1

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reset_state()  # Reset at the end of main

if __name__ == "__main__":
    asyncio.run(main())
    
    
