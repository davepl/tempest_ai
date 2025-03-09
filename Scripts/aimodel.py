import time
import struct

# Dummy AI model function (future magic)

def ai_model(params):
    return params.upper()

# Main loop to read from Lua and write back the reponses on the named pipes

def process_frame_data(data):
    # Unpack the binary data into 16-bit integers
    num_values = len(data) // 2  # Each value is 2 bytes
    unpacked_data = struct.unpack(f">{num_values}H", data)

    # Normalize the data to 0-1 floats
    normalized_data = [value / 65535.0 for value in unpacked_data]

    # Process the normalized data as needed
    # print(normalized_data)  # For debugging, print the normalized data

    return normalized_data

def main():
    while True:
        # Read parameters from Lua
        with open("/tmp/lua_to_py", "rb") as pipe_in:  # Open in binary mode
            data = pipe_in.read()  # Read all available data
            if not data:
                continue

        # Process the frame data
        normalized_data = process_frame_data(data)

        # Process with AI model
        action = ai_model("Hello World!")
        # print(action)
        # Write action back to Lua
        with open("/tmp/py_to_lua", "w") as pipe_out:
            pipe_out.write(action + "\n")
            pipe_out.flush()

if __name__ == "__main__":
    main()
        

