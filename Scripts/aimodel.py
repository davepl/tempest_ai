import time

# Dummy AI model function (future magic)

def ai_model(params):
    return params.upper()

# Main loop to read from Lua and write back the reponses on the named pipes

while True:
    # Read parameters from Lua
    with open("/tmp/lua_to_py", "r") as pipe_in:
        params = pipe_in.readline().strip()
        if not params:
            continue

    # Process with AI model
    action = ai_model(params)

    # Write action back to Lua
    with open("/tmp/py_to_lua", "w") as pipe_out:
        pipe_out.write(action + "\n")
        pipe_out.flush()
        

